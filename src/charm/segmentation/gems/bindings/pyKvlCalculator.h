#ifndef GEMS_PYKVLCALCULATOR_H
#define GEMS_PYKVLCALCULATOR_H

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "kvlAtlasMeshPositionCostAndGradientCalculator.h"
#include "kvlAtlasMeshToIntensityImageCostAndGradientCalculator.h"
#include "kvlMutualInformationCostAndGradientCalculator.h"
#include "kvlAtlasMeshToPointSetCostAndGradientCalculator.h"
#include "pyKvlImage.h"
#include "pyKvlMesh.h"

using LikelihoodMatrix = nb::ndarray<const float, nb::ndim<2>, nb::c_contig>;

class KvlCostAndGradientCalculator {
    // The calculator reads these without owning them
    LikelihoodMatrix likelihoods;
    nb::object voxelSelection;

public:
    kvl::AtlasMeshPositionCostAndGradientCalculator::Pointer calculator;

    KvlCostAndGradientCalculator(const std::string &typeName,
                                 const std::vector<KvlImage *> &images,
                                 const std::string &boundaryCondition,
                                 const KvlTransform *transform,
                                 const LikelihoodMatrix &likelihoods,
                                 const nb::object &voxelSelection,
                                 const Matrix &targetPoints
    ) : likelihoods(likelihoods), voxelSelection(voxelSelection) {
        if (typeName == "AtlasMeshToIntensityImage") {
            if (!likelihoods.is_valid() || voxelSelection.is_none()) {
                throw std::invalid_argument("AtlasMeshToIntensityImage requires likelihoods and voxelSelection.");
            }
            auto myCalculator = kvl::AtlasMeshToIntensityImageCostAndGradientCalculator::New();
            myCalculator->SetLikelihoods(likelihoods.data(), likelihoods.shape(1), nb::cast<const kvl::VoxelSelection *>(voxelSelection));
            calculator = myCalculator;
        } else if (typeName == "MutualInformation") {
            auto myCalculator = kvl::MutualInformationCostAndGradientCalculator::New();
            myCalculator->SetImage(images[0]->m_image);
            calculator = myCalculator;
        } else if (typeName == "PointSet") {
            if (!targetPoints.is_valid()) {
                throw std::invalid_argument("PointSet requires targetPoints.");
            }
            auto myCalculator = kvl::AtlasMeshToPointSetCostAndGradientCalculator::New();
            myCalculator->SetTargetPoints(PointsFromNumpy(targetPoints));
            calculator = myCalculator;
        } else {
            throw std::invalid_argument("Calculator type not supported.");
        }

        if (boundaryCondition.rfind('S', 0) == 0) {
            calculator->SetBoundaryCondition(kvl::AtlasMeshPositionCostAndGradientCalculator::SLIDING);
            if (transform) {
                calculator->SetMeshToImageTransform(transform->m_transform);
            }
        } else if (boundaryCondition.rfind('A', 0) == 0) {
            calculator->SetBoundaryCondition(kvl::AtlasMeshPositionCostAndGradientCalculator::AFFINE);
        } else {
            throw std::invalid_argument("Boundary condition type not supported.");
        }
    }

    std::pair<double, NumpyArray<double>> EvaluateMeshPosition(const KvlMesh &mesh) {
        calculator->Rasterize(mesh.mesh);
        const auto &gradient = calculator->GetPositionGradient()->CastToSTLConstContainer();
        auto *data = allocateZeroedArray<double>(gradient.size() * 3);
        std::memcpy(data, gradient.data()->GetDataPointer(), gradient.size() * 3 * sizeof(double));
        return {calculator->GetMinLogLikelihoodTimesPrior(), createNumpyArrayCStyle<double>({gradient.size(), size_t(3)}, data)};
    }

    double EvaluateMeshCost(const KvlMesh &mesh) {
        calculator->RasterizeCost(mesh.mesh);
        return calculator->GetMinLogLikelihoodTimesPrior();
    }
};

#endif //GEMS_PYKVLCALCULATOR_H
