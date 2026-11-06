// Local thickness of labeled images, based on Hildebrand and Ruegsegger, J. of Microscopy, 1997
// https://doi.org/10.1046/j.1365-2818.1997.1340694.x
//
// The medial axis follows skimage.morphology.medial_axis, which has the following license:
//
//   Copyright (C) 2019, the scikit-image team
//   All rights reserved.
//
//   Redistribution and use in source and binary forms, with or without
//   modification, are permitted provided that the following conditions are
//   met:
//
//    1. Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//    2. Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in
//       the documentation and/or other materials provided with the
//       distribution.
//    3. Neither the name of skimage nor the names of its contributors may be
//       used to endorse or promote products derived from this software without
//       specific prior written permission.
//
//   THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
//   IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//   DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
//   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
//   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
//   STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
//   IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//   POSSIBILITY OF SUCH DAMAGE.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

namespace nb = nanobind;

namespace {

// Whether to keep the center of each 3x3 configuration, with bit 4 the center: if it's foreground and removing it changes
// the number of 8-connected components, or there are at most 2 pixels in the neighborhood.
std::array<std::uint8_t, 512> BuildMedialAxisTable() {
    const auto countComponents = [](int pattern) {
        int count = 0;
        for (int start = 0; start < 9; start++) {
            if (!(pattern >> start & 1)) {
                continue;
            }
            count++;
            int stack[9];
            int size = 0;
            stack[size++] = start;
            pattern &= ~(1 << start);
            while (size > 0) {
                const int pixel = stack[--size];
                for (int other = 0; other < 9; other++) {
                    if ((pattern >> other & 1) && std::abs(other / 3 - pixel / 3) <= 1 && std::abs(other % 3 - pixel % 3) <= 1) {
                        stack[size++] = other;
                        pattern &= ~(1 << other);
                    }
                }
            }
        }
        return count;
    };

    std::array<std::uint8_t, 512> table{};
    for (int index = 0; index < 512; index++) {
        const bool isForeground = index >> 4 & 1;
        const bool changesConnectivity = countComponents(index) != countComponents(index & ~(1 << 4));
        const bool isSmall = __builtin_popcount(index) < 3;
        table[index] = isForeground && (changesConnectivity || isSmall);
    }
    return table;
}

const std::array<std::uint8_t, 512> medialAxisTable = BuildMedialAxisTable();

// Ties in the order of processing are broken at random, or the skeleton would lean in one direction. Multiplying by an odd
// number is a permutation modulo a power of 2 that scatters neighboring pixels, and which the second number undoes.
constexpr int tieBits = 32;
constexpr std::uint64_t tieMask = (std::uint64_t(1) << tieBits) - 1;
constexpr std::uint64_t scatter = 0x9E3779B1;
constexpr std::uint64_t unscatter = 0x0E8B2F51;
static_assert(((scatter * unscatter) & tieMask) == 1);

// Squared distances to the closest zero, by the lower envelope of parabolas. Rows of the result are the rows of the image.
void SquaredDistances(const std::vector<std::uint8_t> &image, long rows, long columns, std::vector<std::uint32_t> &distances) {
    constexpr std::int64_t infinity = std::numeric_limits<std::int64_t>::max() / 4;
    std::vector<std::int64_t> intermediate(rows * columns);
    for (long column = 0; column < columns; column++) {
        std::int64_t distance = infinity;
        for (long row = 0; row < rows; row++) {
            distance = image[row * columns + column] ? (distance >= infinity ? infinity : distance + 1) : 0;
            intermediate[row * columns + column] = distance;
        }
        distance = infinity;
        for (long row = rows - 1; row >= 0; row--) {
            distance = image[row * columns + column] ? (distance >= infinity ? infinity : distance + 1) : 0;
            std::int64_t &value = intermediate[row * columns + column];
            value = std::min(value, distance);
            value = value >= infinity ? infinity : value * value;
        }
    }

    distances.resize(rows * columns);
    std::vector<long> vertices(columns);
    std::vector<double> boundaries(columns + 1);
    for (long row = 0; row < rows; row++) {
        const std::int64_t *values = intermediate.data() + row * columns;
        long last = -1;
        for (long column = 0; column < columns; column++) {
            if (values[column] >= infinity) {
                continue;
            }
            double crossing = -std::numeric_limits<double>::infinity();
            while (last >= 0) {
                const long vertex = vertices[last];
                crossing = (static_cast<double>(values[column] + column * column) - static_cast<double>(values[vertex] + vertex * vertex))
                           / (2.0 * (column - vertex));
                if (crossing > boundaries[last]) {
                    break;
                }
                last--;
            }
            if (last < 0) {
                crossing = -std::numeric_limits<double>::infinity();
            }
            last++;
            vertices[last] = column;
            boundaries[last] = crossing;
        }
        std::uint32_t *output = distances.data() + row * columns;
        long parabola = 0;
        for (long column = 0; column < columns; column++) {
            while (parabola < last && boundaries[parabola + 1] < column) {
                parabola++;
            }
            const long vertex = vertices[parabola];
            output[column] = static_cast<std::uint32_t>(values[vertex] + (column - vertex) * (column - vertex));
        }
    }
}

struct SliceBuffers {
    std::vector<std::uint8_t> image;
    std::vector<std::uint8_t> skeleton;
    std::vector<std::uint32_t> distances;
    std::vector<std::uint64_t> keys;
    std::vector<std::uint32_t> pixels;
    std::vector<float> thickness;
};

// Thickness squared of the foreground of a rows x columns image, in buffers.thickness
void SliceThickness(long rows, long columns, bool hasBackground, SliceBuffers &buffers) {
    const std::vector<std::uint8_t> &image = buffers.image;
    if (hasBackground) {
        SquaredDistances(image, rows, columns, buffers.distances);
    } else {
        // What scipy.ndimage.distance_transform_edt gives without any zero in the image
        buffers.distances.resize(rows * columns);
        for (long row = 0; row < rows; row++) {
            for (long column = 0; column < columns; column++) {
                buffers.distances[row * columns + column] = static_cast<std::uint32_t>((row + 1) * (row + 1) + column * column);
            }
        }
    }

    buffers.pixels.clear();
    for (long pixel = 0; pixel < rows * columns; pixel++) {
        if (image[pixel]) {
            buffers.pixels.push_back(static_cast<std::uint32_t>(pixel));
        }
    }
    const std::uint32_t numberOfPixels = static_cast<std::uint32_t>(buffers.pixels.size());

    // Ordered by distance, then by how cornery a pixel is as those are to be processed last, and then at random
    buffers.keys.resize(numberOfPixels);
    constexpr int cornerBits = 4;
    for (std::uint32_t i = 0; i < numberOfPixels; i++) {
        const long pixel = buffers.pixels[i];
        const long row = pixel / columns;
        const long column = pixel % columns;
        int neighbors = 0;
        for (long r = std::max(row - 1, 0L); r <= std::min(row + 1, rows - 1); r++) {
            for (long c = std::max(column - 1, 0L); c <= std::min(column + 1, columns - 1); c++) {
                neighbors += image[r * columns + c] != 0;
            }
        }
        const std::uint64_t corner = 9 - neighbors;
        buffers.keys[i] = (static_cast<std::uint64_t>(buffers.distances[pixel]) << (tieBits + cornerBits)) | (corner << tieBits) | ((pixel * scatter) & tieMask);
    }

    std::sort(buffers.keys.begin(), buffers.keys.end());

    // Remove the pixels not on the medial axis in one pass, whose order makes that possible
    buffers.skeleton = image;
    std::uint8_t *skeleton = buffers.skeleton.data();
    for (const std::uint64_t key : buffers.keys) {
        const long pixel = static_cast<long>(((key & tieMask) * unscatter) & tieMask);
        const long row = pixel / columns;
        const long column = pixel % columns;
        int configuration = 16;
        if (row > 0) {
            if (column > 0 && skeleton[pixel - columns - 1]) configuration += 1;
            if (skeleton[pixel - columns]) configuration += 2;
            if (column < columns - 1 && skeleton[pixel - columns + 1]) configuration += 4;
        }
        if (column > 0 && skeleton[pixel - 1]) configuration += 8;
        if (column < columns - 1 && skeleton[pixel + 1]) configuration += 32;
        if (row < rows - 1) {
            if (column > 0 && skeleton[pixel + columns - 1]) configuration += 64;
            if (skeleton[pixel + columns]) configuration += 128;
            if (column < columns - 1 && skeleton[pixel + columns + 1]) configuration += 256;
        }
        skeleton[pixel] = medialAxisTable[configuration];
    }

    // The largest ball on the medial axis that a pixel is in
    constexpr double eps = 1e-3;
    buffers.thickness.assign(rows * columns, 0.0f);
    float *thickness = buffers.thickness.data();
    for (const std::uint32_t pixel : buffers.pixels) {
        if (!skeleton[pixel]) {
            continue;
        }
        const double distance = std::sqrt(static_cast<double>(buffers.distances[pixel]));
        const float radius = static_cast<float>(distance);
        const float radiusSquared = radius * radius;
        const double threshold = static_cast<double>(radiusSquared) + eps;
        const long reach = static_cast<long>(distance) + 1;
        const long row = pixel / columns;
        const long column = pixel % columns;
        const long lastRow = std::min(row + reach, rows - 1);
        const long firstColumn = std::max(column - reach, 0L);
        const long lastColumn = std::min(column + reach, columns - 1);
        for (long r = std::max(row - reach, 0L); r <= lastRow; r++) {
            const long dr2 = (r - row) * (r - row);
            const std::uint8_t *imageRow = image.data() + r * columns;
            float *thicknessRow = thickness + r * columns;
            for (long c = firstColumn; c <= lastColumn; c++) {
                const long dc = c - column;
                if (imageRow[c] && static_cast<double>(dr2 + dc * dc) < threshold) {
                    thicknessRow[c] = std::max(thicknessRow[c], radiusSquared);
                }
            }
        }
    }
}

template <typename T>
nb::ndarray<nb::numpy, T> OwnedArray(T *data, std::vector<size_t> shape, char order) {
    nb::capsule owner(data, [](void *d) noexcept { std::free(d); });
    return nb::ndarray<nb::numpy, T>(data, shape.size(), shape.data(), owner, nullptr, nb::dtype<T>(), nb::device::cpu::value, 0, order);
}

template <typename T>
T *Allocate(size_t count) {
    auto *data = static_cast<T *>(std::calloc(count, sizeof(T)));
    if (!data) {
        throw std::bad_alloc();
    }
    return data;
}

// Thickness of the voxels with each label in all slices along one axis, as the lowest over the axes done so far. Slices along
// one axis have no voxels in common, so they can be written to from different threads.
void ThicknessAlongAxis(const std::uint16_t *labels, const std::array<long, 3> &shape, int axis, int rowAxis, int columnAxis, float *output) {
    const long strides[3] = {1, shape[0], shape[0] * shape[1]};
    const long rows = shape[rowAxis];
    const long columns = shape[columnAxis];
    const long rowStride = strides[rowAxis];
    const long columnStride = strides[columnAxis];
    tbb::parallel_for(0L, shape[axis], [&](long slice) {
        const long origin = slice * strides[axis];
        std::vector<std::uint16_t> sliceLabels(rows * columns);
        for (long r = 0; r < rows; r++) {
            for (long c = 0; c < columns; c++) {
                sliceLabels[r * columns + c] = labels[origin + r * rowStride + c * columnStride];
            }
        }
        std::vector<std::uint16_t> present(sliceLabels);
        std::sort(present.begin(), present.end());
        present.erase(std::unique(present.begin(), present.end()), present.end());

        SliceBuffers buffers;
        for (const std::uint16_t label : present) {
            if (label == 0) {
                continue;
            }
            long firstRow = rows;
            long lastRow = -1;
            long firstColumn = columns;
            long lastColumn = -1;
            for (long r = 0; r < rows; r++) {
                for (long c = 0; c < columns; c++) {
                    if (sliceLabels[r * columns + c] == label) {
                        firstRow = std::min(firstRow, r);
                        lastRow = std::max(lastRow, r);
                        firstColumn = std::min(firstColumn, c);
                        lastColumn = std::max(lastColumn, c);
                    }
                }
            }

            // Going one pixel beyond the label keeps the closest background in the image, and leaves the order of its pixels alone
            firstRow = std::max(firstRow - 1, 0L);
            lastRow = std::min(lastRow + 1, rows - 1);
            firstColumn = std::max(firstColumn - 1, 0L);
            lastColumn = std::min(lastColumn + 1, columns - 1);
            const long cropRows = lastRow - firstRow + 1;
            const long cropColumns = lastColumn - firstColumn + 1;
            buffers.image.resize(cropRows * cropColumns);
            bool hasBackground = false;
            for (long r = 0; r < cropRows; r++) {
                for (long c = 0; c < cropColumns; c++) {
                    const bool isForeground = sliceLabels[(r + firstRow) * columns + c + firstColumn] == label;
                    buffers.image[r * cropColumns + c] = isForeground;
                    hasBackground |= !isForeground;
                }
            }
            SliceThickness(cropRows, cropColumns, hasBackground, buffers);
            for (const std::uint32_t pixel : buffers.pixels) {
                const float value = std::sqrt(buffers.thickness[pixel]);
                if (value >= 1e-3f) {
                    const long r = pixel / cropColumns + firstRow;
                    const long c = pixel % cropColumns + firstColumn;
                    float &voxel = output[origin + r * rowStride + c * columnStride];
                    voxel = std::min(voxel, value);
                }
            }
        }
    });
}

using Labels = nb::ndarray<const std::uint16_t, nb::ndim<3>, nb::f_contig>;

nb::ndarray<nb::numpy, float> CalcThickness(const Labels &labelImage) {
    const std::array<long, 3> shape{static_cast<long>(labelImage.shape(0)), static_cast<long>(labelImage.shape(1)), static_cast<long>(labelImage.shape(2))};
    const size_t numberOfVoxels = labelImage.size();
    const std::uint16_t *labels = labelImage.data();
    float *output = Allocate<float>(numberOfVoxels);
    {
        nb::gil_scoped_release release;
        constexpr float infinity = std::numeric_limits<float>::infinity();
        tbb::parallel_for(size_t(0), numberOfVoxels, [&](size_t voxel) { output[voxel] = labels[voxel] ? infinity : 0.0f; });

        // Rows and columns of the slices along each axis are as they were for numpy's swapaxes
        ThicknessAlongAxis(labels, shape, 0, 1, 2, output);
        ThicknessAlongAxis(labels, shape, 1, 0, 2, output);
        ThicknessAlongAxis(labels, shape, 2, 1, 0, output);

        // Voxels that never got a thickness fall back to 1.
        tbb::parallel_for(size_t(0), numberOfVoxels, [&](size_t voxel) {
            if (std::isinf(output[voxel])) {
                output[voxel] = 1.0f;
            }
        });
    }
    return OwnedArray(output, {size_t(shape[0]), size_t(shape[1]), size_t(shape[2])}, 'F');
}

nb::ndarray<nb::numpy, float> ThicknessSlice(const nb::ndarray<const std::uint8_t, nb::ndim<2>, nb::c_contig> &slice) {
    const long rows = slice.shape(0);
    const long columns = slice.shape(1);
    SliceBuffers buffers;
    buffers.image.assign(slice.data(), slice.data() + rows * columns);
    bool hasBackground = false;
    bool hasForeground = false;
    for (const std::uint8_t pixel : buffers.image) {
        hasBackground |= pixel == 0;
        hasForeground |= pixel != 0;
    }
    float *output = Allocate<float>(rows * columns);
    if (hasForeground) {
        SliceThickness(rows, columns, hasBackground, buffers);
        for (long pixel = 0; pixel < rows * columns; pixel++) {
            output[pixel] = std::sqrt(buffers.thickness[pixel]);
        }
    }
    return OwnedArray(output, {size_t(rows), size_t(columns)}, 'C');
}

std::unique_ptr<tbb::global_control> threadLimit;

void SetNumberOfThreads(int numberOfThreads) {
    threadLimit = std::make_unique<tbb::global_control>(tbb::global_control::max_allowed_parallelism, numberOfThreads);
}

} // namespace

NB_MODULE(_thickness, m) {
    m.def("calc_thickness", &CalcThickness, nb::arg("label_img"), "Local thickness of each nonzero label, per voxel");
    m.def("thickness_slice", &ThicknessSlice, nb::arg("slice"));
    m.def("set_number_of_threads", &SetNumberOfThreads, nb::arg("number_of_threads"));
}
