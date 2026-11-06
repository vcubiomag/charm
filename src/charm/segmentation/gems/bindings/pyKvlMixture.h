#ifndef GEMS_PYKVLMIXTURE_H
#define GEMS_PYKVLMIXTURE_H

#include <algorithm>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>
#include "kvlGaussianMixture.h"
#include "pyKvlNumpy.h"

using RowMatrix = nb::ndarray<const double, nb::ndim<2>, nb::c_contig>;
using RowTensor = nb::ndarray<const double, nb::ndim<3>, nb::c_contig>;
using RowVector = nb::ndarray<const double, nb::ndim<1>, nb::c_contig>;

inline kvl::GaussianMixtureModel MakeMixtureModel(const RowMatrix &means, const RowTensor &inverseCholeskys, const RowVector &logScales,
                                                  const std::vector<int> &numberOfGaussiansPerClass) {
    std::vector<size_t> classOffsets(numberOfGaussiansPerClass.size() + 1, 0);
    std::partial_sum(numberOfGaussiansPerClass.begin(), numberOfGaussiansPerClass.end(), classOffsets.begin() + 1);
    return {means.shape(1), std::move(classOffsets), means.data(), inverseCholeskys.data(), logScales.data()};
}

template <typename T>
NumpyArray<T> VectorToNumpy(const std::vector<T> &values, size_t rows, size_t columns) {
    auto *data = allocateZeroedArray<T>(values.size());
    std::copy(values.begin(), values.end(), data);
    return createNumpyArrayCStyle<T>({rows, columns}, data);
}

// The class priors of the modeled voxels, along with the posteriors of the E-step last run on them
class KvlSparsePriors {
    kvl::SparsePriors priors;
    std::vector<std::uint32_t> posteriorOffsets;
    std::vector<float> posteriors;
    std::vector<size_t> posteriorClassOffsets;

public:
    explicit KvlSparsePriors(kvl::SparsePriors &&priors) : priors(std::move(priors)) {}

    std::pair<double, NumpyArray<double>> ComputePosteriors(const RowMatrix &data, const RowMatrix &means, const RowTensor &inverseCholeskys,
                                                            const RowVector &logScales, const std::vector<int> &numberOfGaussiansPerClass,
                                                            bool keepPosteriors) {
        const auto model = MakeMixtureModel(means, inverseCholeskys, logScales, numberOfGaussiansPerClass);
        posteriorClassOffsets = model.m_ClassOffsets;
        if (keepPosteriors && posteriorOffsets.empty()) {
            const size_t numberOfVoxels = priors.GetNumberOfVoxels();
            posteriorOffsets.resize(numberOfVoxels + 1, 0);
            for (size_t voxel = 0; voxel < numberOfVoxels; voxel++) {
                std::uint32_t count = 0;
                for (std::uint32_t entry = priors.m_Offsets[voxel]; entry < priors.m_Offsets[voxel + 1]; entry++) {
                    count += numberOfGaussiansPerClass[priors.m_Classes[entry]];
                }
                posteriorOffsets[voxel + 1] = posteriorOffsets[voxel] + count;
            }
            posteriors.resize(posteriorOffsets.back());
        }
        auto [minLogLikelihood, statistics] = kvl::ComputePosteriors(data.data(), priors, model, posteriorOffsets.data(), keepPosteriors ? posteriors.data() : nullptr);
        const size_t numberOfGaussians = means.shape(0);
        return {minLogLikelihood, VectorToNumpy(statistics, numberOfGaussians, statistics.size() / numberOfGaussians)};
    }

    NumpyArray<double> ComputeClassStatistics(const RowMatrix &data, size_t numberOfClasses) const {
        const auto statistics = kvl::ComputeClassStatistics(data.data(), priors, data.shape(1), numberOfClasses);
        return VectorToNumpy(statistics, numberOfClasses, statistics.size() / numberOfClasses);
    }

    // For the bias field fit: the ( voxels x contrast pairs ) sums over Gaussians of posterior * pairWeights, and the
    // ( voxels x contrasts ) right-hand sides, both from the posteriors of the last E-step
    NumpyArray<float> ComputeBiasFieldTerms(const RowMatrix &maskedImages, const RowMatrix &pairWeights, const RowMatrix &weightedMeans,
                                            const std::vector<int> &firstContrasts, const std::vector<int> &secondContrasts) const {
        const size_t numberOfVoxels = priors.GetNumberOfVoxels();
        const size_t numberOfPairs = pairWeights.shape(1);
        const size_t numberOfContrasts = maskedImages.shape(1);
        const size_t numberOfColumns = numberOfPairs + numberOfContrasts;
        auto *terms = allocateZeroedArray<float>(numberOfVoxels * numberOfColumns);
        const double *pairs = pairWeights.data();
        const double *means = weightedMeans.data();
        const double *images = maskedImages.data();
        kvl::detail::DispatchOnContrasts(numberOfContrasts, [&](auto contrastsConstant) {
            constexpr int TContrasts = decltype(contrastsConstant)::value;
            constexpr int TPairs = TContrasts * (TContrasts + 1) / 2;
            const int contrasts = TContrasts ? TContrasts : static_cast<int>(numberOfContrasts);
            const int pairsPerVoxel = TContrasts ? TPairs : static_cast<int>(numberOfPairs);
            kvl::ParallelForBlocks(numberOfVoxels, kvl::detail::blockSize, [&](size_t, size_t first, size_t last) {
                std::vector<double> buffer(pairsPerVoxel + contrasts);
                double *weights = buffer.data();
                double *rightHandSides = weights + pairsPerVoxel;
                for (size_t voxel = first; voxel < last; voxel++) {
                    std::fill_n(weights, pairsPerVoxel + contrasts, 0.0);
                    std::uint32_t position = posteriorOffsets[voxel];
                    for (std::uint32_t entry = priors.m_Offsets[voxel]; entry < priors.m_Offsets[voxel + 1]; entry++) {
                        const size_t classNumber = priors.m_Classes[entry];
                        for (size_t gaussian = posteriorClassOffsets[classNumber]; gaussian < posteriorClassOffsets[classNumber + 1]; gaussian++) {
                            const double posterior = posteriors[position++];
                            const double *gaussianPairs = pairs + gaussian * pairsPerVoxel;
                            const double *gaussianMeans = means + gaussian * contrasts;
                            for (int pair = 0; pair < pairsPerVoxel; pair++) {
                                weights[pair] += posterior * gaussianPairs[pair];
                            }
                            for (int contrast = 0; contrast < contrasts; contrast++) {
                                rightHandSides[contrast] -= posterior * gaussianMeans[contrast];
                            }
                        }
                    }
                    const double *intensities = images + voxel * contrasts;
                    float *row = terms + voxel * numberOfColumns;
                    for (int pair = 0; pair < pairsPerVoxel; pair++) {
                        const int contrast1 = firstContrasts[pair];
                        const int contrast2 = secondContrasts[pair];
                        rightHandSides[contrast1] += weights[pair] * intensities[contrast2];
                        if (contrast1 != contrast2) {
                            rightHandSides[contrast2] += weights[pair] * intensities[contrast1];
                        }
                        row[pair] = static_cast<float>(weights[pair]);
                    }
                    for (int contrast = 0; contrast < contrasts; contrast++) {
                        row[pairsPerVoxel + contrast] = static_cast<float>(rightHandSides[contrast]);
                    }
                }
            });
        });
        return createNumpyArrayCStyle({numberOfVoxels, numberOfColumns}, terms);
    }
};

using VoxelOffsets = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig>;

// Writes a column of a ( voxels x columns ) matrix to the voxels of an F-ordered volume
inline void ScatterToVolume(const nb::ndarray<const float, nb::ndim<2>, nb::c_contig> &values, size_t column, const VoxelOffsets &voxels,
                            const nb::ndarray<double, nb::ndim<3>, nb::f_contig> &volume) {
    const size_t numberOfColumns = values.shape(1);
    const float *source = values.data() + column;
    const std::int64_t *offsets = voxels.data();
    double *destination = volume.data();
    kvl::ParallelForBlocks(voxels.shape(0), kvl::detail::blockSize, [&](size_t, size_t first, size_t last) {
        for (size_t i = first; i < last; i++) {
            destination[offsets[i]] = source[i * numberOfColumns];
        }
    });
}

// Computes a column of data as the same column of images minus a bias field at the voxels. The field is given by its basis
// functions along the first image axis ( n0 x M ) and its coefficients for those for each row of the image ( n2 x n1 x M ).
// All of the voxels are off the same few numbers, so this doesn't need to compute the whole volume of the bias field.
inline void SubtractBiasField(const RowMatrix &images, size_t column, const RowTensor &rowCoefficients, const RowMatrix &basisFunctions,
                              const VoxelOffsets &voxels, const nb::ndarray<double, nb::ndim<2>, nb::c_contig> &data) {
    const size_t numberOfColumns = images.shape(1);
    const double *source = images.data() + column;
    double *destination = data.data() + column;
    const std::int64_t *offsets = voxels.data();
    const double *coefficients = rowCoefficients.data();
    const double *basis = basisFunctions.data();
    const size_t n0 = basisFunctions.shape(0);
    const size_t M = basisFunctions.shape(1);
    kvl::ParallelForBlocks(voxels.shape(0), kvl::detail::blockSize, [&](size_t, size_t first, size_t last) {
        for (size_t i = first; i < last; i++) {
            const size_t offset = offsets[i];
            const double *rowValues = coefficients + (offset / n0) * M;
            const double *basisValues = basis + (offset % n0) * M;
            double field = 0.0;
            for (size_t m = 0; m < M; m++) {
                field += rowValues[m] * basisValues[m];
            }
            destination[i * numberOfColumns] = source[i * numberOfColumns] - field;
        }
    });
}

inline void ComputeClassLikelihoods(const RowMatrix &data, const RowMatrix &means, const RowTensor &inverseCholeskys, const RowVector &logScales,
                                    const std::vector<int> &numberOfGaussiansPerClass, const nb::ndarray<float, nb::ndim<2>, nb::c_contig> &likelihoods) {
    kvl::ComputeClassLikelihoods(data.data(), data.shape(0), MakeMixtureModel(means, inverseCholeskys, logScales, numberOfGaussiansPerClass), likelihoods.data());
}

#endif //GEMS_PYKVLMIXTURE_H
