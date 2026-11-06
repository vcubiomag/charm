#include "pyKvlMesh.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include "itkMacro.h"
#include "itkTetrahedronCell.h"
#include "kvlAtlasMeshDrawer.h"
#include "vnl/vnl_det.h"

namespace {

kvl::ImageSize ToImageSize(const std::vector<size_t> &size) {
    return {static_cast<long>(size[0]), static_cast<long>(size[1]), static_cast<long>(size[2])};
}

// Writes a row of probabilities as 16 bit priors
void WriteRow(uint16_t *voxels, long count, double value, double step) {
    for (long i = 0; i < count; i++) {
        const double probability = std::clamp(value + i * step, 0.0, 1.0);
        voxels[i] = static_cast<uint16_t>(probability * 65535 + .5);
    }
}

template <typename T>
T *AsMutable(const T *pointer) {
    return const_cast<T *>(pointer);
}

} // namespace

NumpyArray<double> KvlMesh::GetPointSet() const {
    const auto &points = mesh->GetPoints()->CastToSTLConstContainer();
    auto *data = allocateZeroedArray<double>(points.size() * 3);
    std::memcpy(data, points.data()->GetDataPointer(), points.size() * 3 * sizeof(double));
    return createNumpyArrayCStyle({points.size(), size_t(3)}, data);
}

void KvlMesh::SetPointSet(const Matrix &source) {
    auto &points = AsMutable(mesh->GetPoints())->CastToSTLContainer();
    if (source.shape(1) != 3) {
        itkGenericExceptionMacro("source points must have 3 coordinates not " << source.shape(1));
    }
    if (source.shape(0) != points.size()) {
        itkGenericExceptionMacro("source point count of " << source.shape(0) << " not equal to mesh point count of " << points.size());
    }
    for (size_t node = 0; node < points.size(); node++) {
        for (int axis = 0; axis < 3; axis++) {
            points[node][axis] = source(node, axis);
        }
    }
}

NumpyArray<double> KvlMesh::GetAlphas() const {
    const auto &parameters = mesh->GetPointData()->CastToSTLConstContainer();
    const size_t numberOfLabels = parameters.front().m_Alphas.Size();
    auto *data = allocateZeroedArray<double>(parameters.size() * numberOfLabels);
    double *row = data;
    for (const auto &parameter : parameters) {
        row = std::copy_n(parameter.m_Alphas.data_block(), numberOfLabels, row);
    }
    return createNumpyArrayCStyle({parameters.size(), numberOfLabels}, data);
}

void KvlMesh::SetAlphas(const nb::object &source) {
    // Callers assign nested lists as well as arrays, and ndarray casters only accept array-like objects.
    const nb::object array = nb::module_::import_("numpy").attr("asarray")(source, nb::arg("dtype") = "float64");
    const Matrix alphas = nb::cast<Matrix>(array);
    auto *container = AsMutable(mesh->GetPointData());
    auto &parameters = container->CastToSTLContainer();
    if (alphas.shape(0) != parameters.size()) {
        itkGenericExceptionMacro("source data point count of " << alphas.shape(0) << " not equal to mesh point count of " << parameters.size());
    }
    const size_t numberOfLabels = alphas.shape(1);
    if (numberOfLabels == 0) {
        itkGenericExceptionMacro("source data must have a positive number of labels");
    }
    for (size_t node = 0; node < parameters.size(); node++) {
        parameters[node].m_Alphas.SetSize(numberOfLabels);
        for (size_t label = 0; label < numberOfLabels; label++) {
            parameters[node].m_Alphas[label] = static_cast<float>(alphas(node, label));
        }
    }
    container->Modified();
}

void KvlMesh::Scale(const std::vector<double> &scaling) {
    double factors[3];
    if (scaling.size() == 1) {
        std::fill_n(factors, 3, scaling[0]);
    } else if (scaling.size() == 3) {
        std::copy_n(scaling.begin(), 3, factors);
    } else {
        itkGenericExceptionMacro("mesh scaling dimensions must be either 1 or 3 not " << scaling.size());
    }

    for (auto &point : AsMutable(mesh->GetPoints())->CastToSTLContainer()) {
        for (int axis = 0; axis < 3; axis++) {
            point[axis] *= factors[axis];
        }
    }

    // The reference position scales too, but the cells only keep the first three columns of
    // Z = inv( X ) = inv( [ p0 p1 p2 p3; 1 1 1 1 ] ). With S = diag( factors 1 ), inv( S * X ) = Z * inv( S ),
    // so column i of Z gets divided by factors[i].
    auto *cellData = AsMutable(mesh->GetCellData());
    for (auto &info : cellData->CastToSTLContainer()) {
        info.m_ReferenceVolumeTimesK *= factors[0] * factors[1] * factors[2];

        info.m_Z11 /= factors[0];
        info.m_Z21 /= factors[0];
        info.m_Z31 /= factors[0];
        info.m_Z41 /= factors[0];

        info.m_Z12 /= factors[1];
        info.m_Z22 /= factors[1];
        info.m_Z32 /= factors[1];
        info.m_Z42 /= factors[1];

        info.m_Z13 /= factors[2];
        info.m_Z23 /= factors[2];
        info.m_Z33 /= factors[2];
        info.m_Z43 /= factors[2];
    }
    cellData->Modified();
}

void KvlMeshCollection::Assign(const nb::ndarray<const double, nb::ndim<2>, nb::c_contig> &positions,
                               const nb::ndarray<const std::uint32_t, nb::ndim<2>, nb::c_contig> &tetrahedra,
                               const nb::ndarray<const float, nb::ndim<2>, nb::c_contig> &alphas,
                               const nb::ndarray<const std::uint8_t, nb::ndim<2>, nb::c_contig> &canMove, double k) {
    const size_t numberOfPoints = positions.shape(0);
    if (positions.shape(1) != 3 || tetrahedra.shape(1) != 4 || canMove.shape(1) != 3 || alphas.shape(0) != numberOfPoints || canMove.shape(0) != numberOfPoints) {
        itkGenericExceptionMacro("expected positions and can_move with 3 columns, tetrahedra with 4, and one row per node for positions, alphas and can_move");
    }
    meshCollection->Assign(positions.data(), numberOfPoints, tetrahedra.data(), tetrahedra.shape(0), alphas.data(), alphas.shape(1), canMove.data(), k);
}

nb::tuple KvlMeshCollection::Contents() const {
    const auto &points = meshCollection->GetReferencePosition()->CastToSTLConstContainer();
    const auto &cells = meshCollection->GetCells()->CastToSTLConstContainer();
    const auto &parameters = meshCollection->GetPointParameters()->CastToSTLConstContainer();
    const size_t numberOfLabels = parameters.front().m_Alphas.Size();

    auto *positions = allocateZeroedArray<double>(points.size() * 3);
    auto *tetrahedra = allocateZeroedArray<std::uint32_t>(cells.size() * 4);
    auto *alphas = allocateZeroedArray<float>(points.size() * numberOfLabels);
    auto *canMove = allocateZeroedArray<std::uint8_t>(points.size() * 3);
    for (size_t node = 0; node < points.size(); node++) {
        std::copy_n(points[node].GetDataPointer(), 3, positions + node * 3);
        std::copy_n(parameters[node].m_Alphas.data_block(), numberOfLabels, alphas + node * numberOfLabels);
        canMove[node * 3] = parameters[node].m_CanMoveX;
        canMove[node * 3 + 1] = parameters[node].m_CanMoveY;
        canMove[node * 3 + 2] = parameters[node].m_CanMoveZ;
    }
    for (size_t cell = 0; cell < cells.size(); cell++) {
        const auto *ids = cells[cell]->PointIdsBegin();
        for (int vertex = 0; vertex < 4; vertex++) {
            tetrahedra[cell * 4 + vertex] = static_cast<std::uint32_t>(ids[vertex]);
        }
    }
    return nb::make_tuple(createNumpyArrayCStyle<double>({points.size(), size_t(3)}, positions),
                          createNumpyArrayCStyle<std::uint32_t>({cells.size(), size_t(4)}, tetrahedra),
                          createNumpyArrayCStyle<float>({points.size(), numberOfLabels}, alphas),
                          createNumpyArrayCStyle<std::uint8_t>({points.size(), size_t(3)}, canMove),
                          meshCollection->GetK());
}

void KvlMeshCollection::Transform(const KvlTransform &kvlTransform) {
    const auto transform = kvlTransform.GetTransform();
    meshCollection->Transform(transform);

    // A transform with a negative determinant turns positive tetrahedra into negative ones.
    // Swapping the points assigned to the first two vertices turns them into positive ones again.
    if (vnl_det(transform->GetMatrix().GetVnlMatrix()) < 0) {
        auto *cells = meshCollection->GetCells();
        for (auto *cell : cells->CastToSTLContainer()) {
            auto *pointIds = cell->PointIdsBegin();
            std::swap(pointIds[0], pointIds[1]);
        }
        cells->Modified();
    }
}

void KvlMeshCollection::SetK(double k) {
    meshCollection->SetK(k);
}

double KvlMeshCollection::GetK() const {
    return meshCollection->GetK();
}

KvlMesh *KvlMeshCollection::GetReferenceMesh() {
    return new KvlMesh(meshCollection->GetReferenceMesh());
}

NumpyArray<uint16_t> KvlMesh::RasterizeMesh(const std::vector<size_t> &size, int classNumber) {
    const kvl::ImageSize imageSize = ToImageSize(size);
    const size_t numberOfVoxels = size[0] * size[1] * size[2];

    if (classNumber >= 0) {
        auto *buffer = allocateZeroedArray<uint16_t>(numberOfVoxels);

        // Parts not overlaid by the mesh are considered to be background, which is class 0
        const bool isBackground = classNumber == 0;
        if (isBackground) {
            std::fill_n(buffer, numberOfVoxels, uint16_t(65535));
        }

        const auto table = kvl::TetrahedronTable::Get(mesh);
        const auto &parameters = mesh->GetPointData()->CastToSTLConstContainer();
        kvl::DrawMesh<float>(mesh, imageSize,
            [&](size_t t, auto &channels) {
                const auto &ids = table->m_Vertices[t];
                const std::array<float, 4> values{parameters[ids[0]].m_Alphas[classNumber], parameters[ids[1]].m_Alphas[classNumber],
                                                  parameters[ids[2]].m_Alphas[classNumber], parameters[ids[3]].m_Alphas[classNumber]};
                if (isBackground || values[0] != 0 || values[1] != 0 || values[2] != 0 || values[3] != 0) {
                    channels.push_back({0, values});
                }
            },
            [&](size_t, long voxel, long count, double value, double step) { WriteRow(buffer + voxel, count, value, step); });
        return createNumpyArrayFStyle({size[0], size[1], size[2]}, buffer);
    }

    const size_t numberOfClasses = mesh->GetPointData()->Begin().Value().m_Alphas.Size();
    auto *buffer = allocateZeroedArray<uint16_t>(numberOfVoxels * numberOfClasses);
    kvl::DrawAlphas(mesh, imageSize, [&](size_t channel, long voxel, long count, double value, double step) {
        WriteRow(buffer + channel * numberOfVoxels + voxel, count, value, step);
    });
    return createNumpyArrayFStyle({size[0], size[1], size[2], numberOfClasses}, buffer);
}

KvlSparsePriors *KvlMesh::RasterizePriors(const kvl::VoxelSelection &voxels) {
    const size_t numberOfVoxels = voxels.GetNumberOfVoxels();
    const size_t numberOfClasses = mesh->GetPointData()->Begin().Value().m_Alphas.Size();
    if (numberOfClasses > 256) {
        itkGenericExceptionMacro("sparse priors support at most 256 classes, not " << numberOfClasses);
    }

    const auto table = kvl::TetrahedronTable::Get(mesh);
    const auto alphas = kvl::TetrahedronAlphas::Get(mesh);
    const auto &points = mesh->GetPoints()->CastToSTLConstContainer();
    const auto forEachTetrahedronVoxels = [&](auto &&body) {
        kvl::ParallelForBlocks(table->Size(), 256, [&](size_t, size_t first, size_t last) {
            thread_local kvl::VoxelList coordinates;
            thread_local std::vector<std::uint32_t> rows;
            for (size_t t = first; t < last; t++) {
                const auto &ids = table->m_Vertices[t];
                const kvl::TetrahedronRasterizer rasterizer(points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]], voxels.GetSize());
                const size_t candidates = rasterizer.CollectVoxels(voxels.GetSize(), coordinates);
                rows.resize(std::max(rows.size(), candidates));
                body(t, rasterizer, coordinates, rows.data(), voxels.Select(coordinates, candidates, rows.data()));
            }
        });
    };

    // Each voxel gets drawn from exactly one tetrahedron, which has the same classes for all of its voxels
    kvl::SparsePriors priors;
    priors.m_Offsets.assign(numberOfVoxels + 1, 0);
    forEachTetrahedronVoxels([&](size_t t, const auto &, const auto &, const std::uint32_t *rows, size_t count) {
        const std::uint32_t numberOfEntries = alphas->m_Offsets[t + 1] - alphas->m_Offsets[t];
        for (size_t i = 0; i < count; i++) {
            priors.m_Offsets[rows[i] + 1] = numberOfEntries;
        }
    });
    std::uint64_t total = 0;
    for (size_t row = 1; row <= numberOfVoxels; row++) {
        total += priors.m_Offsets[row];
        if (total > std::numeric_limits<std::uint32_t>::max()) {
            itkGenericExceptionMacro("too many sparse prior entries");
        }
        priors.m_Offsets[row] = static_cast<std::uint32_t>(total);
    }
    priors.m_Classes.resize(total);
    priors.m_Values.resize(total);

    forEachTetrahedronVoxels([&](size_t t, const auto &rasterizer, const kvl::VoxelList &coordinates, const std::uint32_t *rows, size_t count) {
        const std::uint32_t firstAlpha = alphas->m_Offsets[t];
        const std::uint32_t numberOfEntries = alphas->m_Offsets[t + 1] - firstAlpha;
        for (size_t i = 0; i < count; i++) {
            double pi[4];
            rasterizer.GetBarycentricCoordinates(coordinates.m_X[i], coordinates.m_Y[i], coordinates.m_Z[i], pi);
            const std::uint32_t offset = priors.m_Offsets[rows[i]];
            for (std::uint32_t a = 0; a < numberOfEntries; a++) {
                const auto &vertexAlphas = alphas->m_Alphas[firstAlpha + a];
                const double value = vertexAlphas[0] * pi[0] + vertexAlphas[1] * pi[1] + vertexAlphas[2] * pi[2] + vertexAlphas[3] * pi[3];
                priors.m_Classes[offset + a] = static_cast<std::uint8_t>(alphas->m_Classes[firstAlpha + a]);
                priors.m_Values[offset + a] = static_cast<float>(std::clamp(value, 0.0, 1.0));
            }
        }
    });
    return new KvlSparsePriors(std::move(priors));
}

// The structure of highest posterior of every voxel of the image, which is only labeled from its prior if it isn't selected.
// Each voxel is drawn from exactly one tetrahedron, in which only a few of the structures are possible, so only their
// classes need their Gaussians evaluated and nothing needs to be kept per structure.
NumpyArray<uint16_t> KvlMesh::Segment(const kvl::VoxelSelection &voxels, const RowMatrix &data, const RowMatrix &means,
                                      const RowTensor &inverseCholeskys, const RowVector &logScales,
                                      const std::vector<int> &numberOfGaussiansPerClass, const RowMatrix &classFractions,
                                      const std::vector<double> &structureFactors, const std::vector<uint16_t> &labels) {
    constexpr double minimumFraction = 1e-10;
    const auto model = MakeMixtureModel(means, inverseCholeskys, logScales, numberOfGaussiansPerClass);
    const size_t numberOfContrasts = model.m_NumberOfContrasts;
    const size_t numberOfClasses = classFractions.shape(0);
    const size_t numberOfStructures = classFractions.shape(1);
    const auto table = kvl::TetrahedronTable::Get(mesh);
    const auto alphas = kvl::TetrahedronAlphas::Get(mesh);
    if (alphas->m_NumberOfClasses != numberOfStructures || labels.size() != numberOfStructures || structureFactors.size() != numberOfStructures) {
        itkGenericExceptionMacro("the fractions, factors and labels must have one column or entry for each of the " << alphas->m_NumberOfClasses << " structures in the mesh");
    }

    std::vector<std::uint32_t> structureOffsets(numberOfStructures + 1, 0);
    std::vector<std::uint16_t> structureClasses;
    std::vector<double> structureFractions;
    for (size_t structure = 0; structure < numberOfStructures; structure++) {
        for (size_t classNumber = 0; classNumber < numberOfClasses; classNumber++) {
            const double fraction = classFractions(classNumber, structure);
            if (fraction > minimumFraction) {
                structureClasses.push_back(static_cast<std::uint16_t>(classNumber));
                structureFractions.push_back(fraction);
            }
        }
        structureOffsets[structure + 1] = static_cast<std::uint32_t>(structureClasses.size());
    }

    const auto &size = voxels.GetSize();
    const size_t numberOfVoxels = static_cast<size_t>(size[0]) * size[1] * size[2];
    auto *buffer = allocateZeroedArray<uint16_t>(numberOfVoxels);
    std::fill_n(buffer, numberOfVoxels, labels[0]);

    const auto &points = mesh->GetPoints()->CastToSTLConstContainer();
    kvl::detail::DispatchOnContrasts(numberOfContrasts, [&](auto contrastsConstant) {
        constexpr int TContrasts = decltype(contrastsConstant)::value;
        kvl::ParallelForBlocks(table->Size(), 256, [&](size_t, size_t first, size_t last) {
            thread_local kvl::VoxelList coordinates;
            std::vector<std::int64_t> rows;
            std::vector<std::uint32_t> selected;
            std::vector<double> densities;
            std::vector<double> likelihoods;
            std::vector<std::int32_t> classPositions(numberOfClasses, -1);
            std::vector<std::uint16_t> tetrahedronClasses;
            for (size_t t = first; t < last; t++) {
                const auto &ids = table->m_Vertices[t];
                const kvl::TetrahedronRasterizer rasterizer(points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]], size);
                const size_t count = rasterizer.CollectVoxels(size, coordinates);
                if (count == 0) {
                    continue;
                }

                const std::uint32_t firstAlpha = alphas->m_Offsets[t];
                const std::uint32_t lastAlpha = alphas->m_Offsets[t + 1];
                tetrahedronClasses.clear();
                size_t gaussiansPerVoxel = 0;
                for (std::uint32_t a = firstAlpha; a < lastAlpha; a++) {
                    const size_t structure = alphas->m_Classes[a];
                    for (std::uint32_t entry = structureOffsets[structure]; entry < structureOffsets[structure + 1]; entry++) {
                        const size_t classNumber = structureClasses[entry];
                        if (classPositions[classNumber] < 0) {
                            classPositions[classNumber] = static_cast<std::int32_t>(tetrahedronClasses.size());
                            tetrahedronClasses.push_back(static_cast<std::uint16_t>(classNumber));
                            gaussiansPerVoxel += model.m_ClassOffsets[classNumber + 1] - model.m_ClassOffsets[classNumber];
                        }
                    }
                }

                rows.resize(count);
                selected.clear();
                for (size_t i = 0; i < count; i++) {
                    const size_t offset = coordinates.m_X[i] + size[0] * (coordinates.m_Y[i] + size[1] * static_cast<size_t>(coordinates.m_Z[i]));
                    rows[i] = voxels.GetRow(offset);
                    if (rows[i] >= 0) {
                        selected.push_back(static_cast<std::uint32_t>(i));
                    }
                }

                // Exponentials go through one loop for them to be vectorized
                densities.resize(selected.size() * gaussiansPerVoxel);
                size_t position = 0;
                for (const std::uint32_t i : selected) {
                    const double *intensities = data.data() + rows[i] * numberOfContrasts;
                    for (const std::uint16_t classNumber : tetrahedronClasses) {
                        for (size_t gaussian = model.m_ClassOffsets[classNumber]; gaussian < model.m_ClassOffsets[classNumber + 1]; gaussian++) {
                            densities[position++] = kvl::detail::LogDensity<TContrasts>(intensities, model, gaussian, numberOfContrasts);
                        }
                    }
                }
                for (size_t i = 0; i < densities.size(); i++) {
                    densities[i] = kvl::FastExp(densities[i]);
                }

                const size_t numberOfTetrahedronClasses = tetrahedronClasses.size();
                likelihoods.resize(selected.size() * numberOfTetrahedronClasses);
                position = 0;
                size_t likelihood = 0;
                for (size_t voxel = 0; voxel < selected.size(); voxel++) {
                    for (const std::uint16_t classNumber : tetrahedronClasses) {
                        double sum = 0.0;
                        for (size_t gaussian = model.m_ClassOffsets[classNumber]; gaussian < model.m_ClassOffsets[classNumber + 1]; gaussian++) {
                            sum += densities[position++];
                        }
                        likelihoods[likelihood++] = sum;
                    }
                }

                size_t selectedVoxel = 0;
                for (size_t i = 0; i < count; i++) {
                    double pi[4];
                    rasterizer.GetBarycentricCoordinates(coordinates.m_X[i], coordinates.m_Y[i], coordinates.m_Z[i], pi);
                    const bool isSelected = rows[i] >= 0;
                    const double *voxelLikelihoods = isSelected ? likelihoods.data() + numberOfTetrahedronClasses * selectedVoxel++ : nullptr;
                    double highest = 0.0;
                    std::uint16_t label = labels[0];
                    for (std::uint32_t a = firstAlpha; a < lastAlpha; a++) {
                        const size_t structure = alphas->m_Classes[a];
                        const auto &vertexAlphas = alphas->m_Alphas[a];
                        double value = std::clamp(vertexAlphas[0] * pi[0] + vertexAlphas[1] * pi[1] + vertexAlphas[2] * pi[2] + vertexAlphas[3] * pi[3], 0.0, 1.0);
                        if (isSelected) {
                            double structureLikelihood = 0.0;
                            for (std::uint32_t entry = structureOffsets[structure]; entry < structureOffsets[structure + 1]; entry++) {
                                structureLikelihood += structureFractions[entry] * voxelLikelihoods[classPositions[structureClasses[entry]]];
                            }
                            value *= structureLikelihood * structureFactors[structure];
                        }
                        if (value > highest) {
                            highest = value;
                            label = labels[structure];
                        }
                    }
                    buffer[coordinates.m_X[i] + size[0] * (coordinates.m_Y[i] + size[1] * static_cast<size_t>(coordinates.m_Z[i]))] = label;
                }

                for (const std::uint16_t classNumber : tetrahedronClasses) {
                    classPositions[classNumber] = -1;
                }
            }
        });
    });
    return createNumpyArrayFStyle({size_t(size[0]), size_t(size[1]), size_t(size[2])}, buffer);
}

template <typename TValue>
NumpyArray<TValue> KvlMesh::RasterizeValues(std::vector<size_t> size, const nb::ndarray<const TValue, nb::c_contig> &values) {
    const size_t numberOfFrames = (values.ndim() > 1) ? values.shape(1) : 1;
    const size_t numberOfVoxels = size[0] * size[1] * size[2];
    auto *buffer = allocateZeroedArray<TValue>(numberOfVoxels * numberOfFrames);
    const auto table = kvl::TetrahedronTable::Get(mesh);
    const TValue *nodeValues = values.data();

    kvl::DrawMesh<TValue>(mesh, ToImageSize(size),
        [&](size_t t, auto &channels) {
            const auto &ids = table->m_Vertices[t];
            for (size_t frame = 0; frame < numberOfFrames; frame++) {
                const std::array<TValue, 4> frameValues{nodeValues[ids[0] * numberOfFrames + frame], nodeValues[ids[1] * numberOfFrames + frame],
                                                        nodeValues[ids[2] * numberOfFrames + frame], nodeValues[ids[3] * numberOfFrames + frame]};
                channels.push_back({frame, frameValues});
            }
        },
        [&](size_t frame, long voxel, long count, double value, double step) {
            TValue *row = buffer + frame * numberOfVoxels + voxel;
            for (long i = 0; i < count; i++) {
                row[i] = static_cast<TValue>(value + i * step);
            }
        });

    if (numberOfFrames > 1) {
        size.push_back(numberOfFrames);
    }
    return createNumpyArrayFStyle(size, buffer);
}

template NumpyArray<float> KvlMesh::RasterizeValues<float>(std::vector<size_t>, const nb::ndarray<const float, nb::c_contig> &);
template NumpyArray<double> KvlMesh::RasterizeValues<double>(std::vector<size_t>, const nb::ndarray<const double, nb::c_contig> &);

kvl::AtlasMesh::PointsContainer::Pointer PointsFromNumpy(const Matrix &source) {
    if (source.shape(1) != 3) {
        itkGenericExceptionMacro("source points must have 3 coordinates not " << source.shape(1));
    }
    auto points = kvl::AtlasMesh::PointsContainer::New();
    auto &values = points->CastToSTLContainer();
    values.resize(source.shape(0));
    for (size_t node = 0; node < values.size(); node++) {
        for (int axis = 0; axis < 3; axis++) {
            values[node][axis] = source(node, axis);
        }
    }
    return points;
}

KvlMesh* KvlMesh::GetSubmesh( const nb::ndarray<bool, nb::ndim<1>> &mask ) {
    const auto &points = mesh->GetPoints()->CastToSTLConstContainer();
    const auto &parameters = mesh->GetPointData()->CastToSTLConstContainer();
    const auto &cells = mesh->GetCells()->CastToSTLConstContainer();
    const auto &infos = mesh->GetCellData()->CastToSTLConstContainer();
    if (points.size() != mask.shape(0)) {
        itkGenericExceptionMacro("Number of nodes in mask not compatible with number of nodes in mesh ("
                                 << mask.shape(0) << " vs. " << points.size() << ")");
    }

    // All the tetrahedra attached to a node in the mask are part of the submesh. Their other nodes have
    // to be part of it as well, but those are immobile because not all their tetrahedra are there.
    std::vector<char> isIncluded(points.size());
    for (size_t node = 0; node < points.size(); node++) {
        isIncluded[node] = mask(node);
    }
    std::vector<size_t> tetrahedra;
    std::vector<char> isMobile(isIncluded);
    for (size_t cellNumber = 0; cellNumber < cells.size(); cellNumber++) {
        const auto *cell = cells[cellNumber];
        if (cell->GetType() != itk::CellGeometryEnum::TETRAHEDRON_CELL) {
            continue;
        }
        const auto *ids = cell->PointIdsBegin();
        if (isMobile[ids[0]] || isMobile[ids[1]] || isMobile[ids[2]] || isMobile[ids[3]]) {
            tetrahedra.push_back(cellNumber);
            for (int vertex = 0; vertex < 4; vertex++) {
                isIncluded[ids[vertex]] = true;
            }
        }
    }

    // Numbering in order of the original ids keeps the ascending vertex order within tetrahedra consistent
    constexpr std::uint32_t notIncluded = ~std::uint32_t(0);
    std::vector<std::uint32_t> newIds(points.size(), notIncluded);
    auto subPoints = kvl::AtlasMesh::PointsContainer::New();
    auto subPointData = kvl::AtlasMesh::PointDataContainer::New();
    auto &subPointValues = subPoints->CastToSTLContainer();
    auto &subParameters = subPointData->CastToSTLContainer();
    for (size_t node = 0; node < points.size(); node++) {
        if (isIncluded[node]) {
            newIds[node] = static_cast<std::uint32_t>(subPointValues.size());
            subPointValues.push_back(points[node]);
            subParameters.push_back(parameters[node]);
            if (!isMobile[node]) {
                subParameters.back().m_CanMoveX = false;
                subParameters.back().m_CanMoveY = false;
                subParameters.back().m_CanMoveZ = false;
            }
        }

        // Report the added boundary nodes back to the caller through the mask
        mask(node) = isIncluded[node];
    }

    using TetrahedronCell = itk::TetrahedronCell<kvl::AtlasMesh::CellType>;
    auto subCells = kvl::AtlasMesh::CellsContainer::New();
    auto subCellData = kvl::AtlasMesh::CellDataContainer::New();
    auto &subCellValues = subCells->CastToSTLContainer();
    auto &subInfos = subCellData->CastToSTLContainer();
    subCellValues.reserve(tetrahedra.size());
    subInfos.reserve(tetrahedra.size());
    for (const size_t cellNumber : tetrahedra) {
        const auto *ids = cells[cellNumber]->PointIdsBegin();
        auto *newCell = new TetrahedronCell;
        for (int vertex = 0; vertex < 4; vertex++) {
            newCell->SetPointId(vertex, newIds[ids[vertex]]);
        }
        subCellValues.push_back(newCell);
        subInfos.push_back(infos[cellNumber]);
    }

    kvl::AtlasMesh::Pointer subMesh = kvl::AtlasMesh::New();
    subMesh->SetPoints(subPoints);
    subMesh->SetPointData(subPointData);
    subMesh->SetCells(subCells);
    subMesh->SetCellData(subCellData);
    return new KvlMesh(subMesh.GetPointer());
}
