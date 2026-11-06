#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace {

template <typename T>
using Matrix = nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
template <typename T>
using Vector = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <typename T>
T *Allocate(size_t count) {
    auto *data = static_cast<T *>(std::calloc(count, sizeof(T)));
    if (!data) {
        throw std::bad_alloc();
    }
    return data;
}

template <typename T>
nb::ndarray<nb::numpy, T> OwnedArray(T *data, std::vector<size_t> shape) {
    nb::capsule owner(data, [](void *d) noexcept { std::free(d); });
    return nb::ndarray<nb::numpy, T>(data, shape.size(), shape.data(), owner);
}

// The field at the voxels inside each tetrahedron, linearly interpolated from its nodes if the field is per node and
// constant if it's per tetrahedron. Voxels in more than one tetrahedron take the value of the last.
nb::ndarray<nb::numpy, double> InterpGrid(const Vector<std::int64_t> &numberOfVoxels, const Matrix<double> &field,
                                          const Matrix<double> &nodes, const Matrix<std::int64_t> &tetrahedra) {
    if (numberOfVoxels.shape(0) != 3 || nodes.shape(1) != 3 || tetrahedra.shape(1) != 4) {
        throw nb::value_error("expected 3 voxel counts, 3D nodes and 4 nodes per tetrahedron");
    }
    const std::array<long, 3> size{numberOfVoxels(0), numberOfVoxels(1), numberOfVoxels(2)};
    const long components = field.shape(1);
    const bool isNodeData = field.shape(0) == nodes.shape(0);
    const long numberOfTetrahedra = tetrahedra.shape(0);
    double *image = Allocate<double>(size[0] * size[1] * size[2] * components);
    {
        nb::gil_scoped_release release;
        constexpr double eps = 1e-5;
        for (long t = 0; t < numberOfTetrahedra; t++) {
            std::array<const double *, 4> p;
            for (int i = 0; i < 4; i++) {
                p[i] = nodes.data() + 3 * tetrahedra(t, i);
            }
            std::array<long, 3> low, high;
            bool inside = true;
            for (int a = 0; a < 3; a++) {
                const double lowest = std::min({p[0][a], p[1][a], p[2][a], p[3][a]});
                const double highest = std::max({p[0][a], p[1][a], p[2][a], p[3][a]});
                // Rounds half to even, as numpy's rint
                low[a] = std::lrint(lowest);
                high[a] = std::lrint(highest);
                inside &= low[a] <= size[a] && high[a] >= 0;
                low[a] = std::max(low[a], 0L);
                high[a] = std::min(high[a], size[a] - 1);
            }
            if (!inside) {
                continue;
            }

            // Barycentric coordinates of the first three nodes are the inverse of the matrix of edges from the fourth
            // node applied to the offset from it
            double m[3][3];
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 3; c++) {
                    m[r][c] = p[c][r] - p[3][r];
                }
            }
            const double determinant = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                                       m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                                       m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
            if (determinant == 0.0) {
                continue;
            }
            const double inverse[3][3] = {
                {(m[1][1] * m[2][2] - m[1][2] * m[2][1]) / determinant, (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / determinant,
                 (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / determinant},
                {(m[1][2] * m[2][0] - m[1][0] * m[2][2]) / determinant, (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / determinant,
                 (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / determinant},
                {(m[1][0] * m[2][1] - m[1][1] * m[2][0]) / determinant, (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / determinant,
                 (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / determinant},
            };

            for (long x = low[0]; x <= high[0]; x++) {
                const double xc = x - p[3][0];
                for (long y = low[1]; y <= high[1]; y++) {
                    const double yc = y - p[3][1];
                    for (long z = low[2]; z <= high[2]; z++) {
                        const double zc = z - p[3][2];
                        double b[4];
                        b[0] = inverse[0][0] * xc + inverse[0][1] * yc + inverse[0][2] * zc;
                        if (!(b[0] > -eps && b[0] < 1.0 + eps)) {
                            continue;
                        }
                        b[1] = inverse[1][0] * xc + inverse[1][1] * yc + inverse[1][2] * zc;
                        if (!(b[1] > -eps && b[0] + b[1] < 1.0 + eps)) {
                            continue;
                        }
                        b[2] = inverse[2][0] * xc + inverse[2][1] * yc + inverse[2][2] * zc;
                        b[3] = 1.0 - b[0] - b[1] - b[2];
                        if (!(b[2] > -eps && b[3] > -eps)) {
                            continue;
                        }
                        double *voxel = image + ((x * size[1] + y) * size[2] + z) * components;
                        for (long k = 0; k < components; k++) {
                            if (isNodeData) {
                                voxel[k] = 0.0;
                                for (int i = 0; i < 4; i++) {
                                    voxel[k] += b[i] * field(tetrahedra(t, i), k);
                                }
                            } else {
                                voxel[k] = field(t, k);
                            }
                        }
                    }
                }
            }
        }
    }
    return OwnedArray(image, {size_t(size[0]), size_t(size[1]), size_t(size[2]), size_t(components)});
}

// One step of moving each surface node towards the mean of its neighbors, in place. Every node moves from where all of
// them were before the step.
void GaussSmoothSimple(const Vector<std::uint64_t> &surfaceNodes,
                       nb::ndarray<double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu> nodesPosition,
                       const Vector<std::uint64_t> &adjacencyIndices, const Vector<std::uint64_t> &adjacencyPointers,
                       double factor) {
    double *position = nodesPosition.data();
    const std::uint64_t *indices = adjacencyIndices.data();
    const std::uint64_t *pointers = adjacencyPointers.data();
    nb::gil_scoped_release release;
    const std::vector<double> before(position, position + nodesPosition.size());
    for (size_t i = 0; i < surfaceNodes.shape(0); i++) {
        const std::uint64_t n = surfaceNodes(i);
        double mean[3] = {0.0, 0.0, 0.0};
        for (std::uint64_t j = pointers[n]; j < pointers[n + 1]; j++) {
            for (int k = 0; k < 3; k++) {
                mean[k] += before[3 * indices[j] + k];
            }
        }
        const double count = double(pointers[n + 1] - pointers[n]);
        for (int k = 0; k < 3; k++) {
            position[3 * n + k] += factor * (mean[k] / count - before[3 * n + k]);
        }
    }
}

nb::ndarray<nb::numpy, double> TetrahedraGamma(const nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu> &nodes,
                                               const nb::ndarray<const std::int32_t, nb::shape<-1, 4>, nb::c_contig, nb::device::cpu> &tetrahedra) {
    const size_t n = tetrahedra.shape(0);
    double *gamma = Allocate<double>(n);
    {
        nb::gil_scoped_release release;
        const double *coordinates = nodes.data();
        for (size_t i = 0; i < n; i++) {
            const double *p0 = coordinates + 3 * (tetrahedra(i, 0) - 1);
            const double *p1 = coordinates + 3 * (tetrahedra(i, 1) - 1);
            const double *p2 = coordinates + 3 * (tetrahedra(i, 2) - 1);
            const double *p3 = coordinates + 3 * (tetrahedra(i, 3) - 1);
            const double ax = p1[0] - p0[0], ay = p1[1] - p0[1], az = p1[2] - p0[2];
            const double bx = p2[0] - p0[0], by = p2[1] - p0[1], bz = p2[2] - p0[2];
            const double cx = p3[0] - p0[0], cy = p3[1] - p0[1], cz = p3[2] - p0[2];
            const double volume = (ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)) / 6.0;
            if (volume < 0) {
                gamma[i] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            const double squared = ax * ax + ay * ay + az * az + bx * bx + by * by + bz * bz + cx * cx + cy * cy + cz * cz +
                                   (bx - ax) * (bx - ax) + (by - ay) * (by - ay) + (bz - az) * (bz - az) + (cx - ax) * (cx - ax) +
                                   (cy - ay) * (cy - ay) + (cz - az) * (cz - az) + (cx - bx) * (cx - bx) + (cy - by) * (cy - by) +
                                   (cz - bz) * (cz - bz);
            const double rms = std::sqrt(squared / 6.0);
            gamma[i] = rms * rms * rms / volume;
        }
    }
    return OwnedArray(gamma, {n});
}

} // namespace

NB_MODULE(_kernels, m) {
    m.def("interp_grid", &InterpGrid, nb::arg("n_voxels"), nb::arg("field"), nb::arg("nodes"), nb::arg("tetrahedra"),
          "A field on the nodes or tetrahedra of a mesh, sampled at the voxels of a grid of the given size");
    m.def("gauss_smooth_simple", &GaussSmoothSimple, nb::arg("surf_nodes"), nb::arg("nodes_pos").noconvert(),
          nb::arg("adj_indices"), nb::arg("adj_indptr"), nb::arg("factor"), "One step of Gaussian smoothing of the given nodes, in place");
    m.def("tetrahedra_gamma", &TetrahedraGamma, nb::arg("nodes"), nb::arg("tetrahedra"),
          "The root mean square edge length cubed over the volume of each tetrahedron given by its one-based nodes, "
          "nan if the volume is negative");
}
