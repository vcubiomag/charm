#ifndef GEMS_PYKVLOPTIMIZER_H
#define GEMS_PYKVLOPTIMIZER_H

#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include "kvlAtlasMeshDeformationLBFGSOptimizer.h"
#include "pyKvlMesh.h"
#include "pyKvlCalculator.h"

class KvlOptimizer {
    kvl::AtlasMeshDeformationLBFGSOptimizer::Pointer optimizer = kvl::AtlasMeshDeformationLBFGSOptimizer::New();

    static kvl::AtlasMesh *AsMutable(const KvlMesh &mesh) {
        return const_cast<kvl::AtlasMesh *>(mesh.mesh.GetPointer());
    }

public:
    KvlOptimizer(const std::string &typeName,
                 const KvlMesh &mesh,
                 const KvlCostAndGradientCalculator &calculator,
                 const std::map<std::string, double> &arguments){
        if (typeName.rfind('L', 0) != 0) {
            throw std::invalid_argument("Optimizer type not supported.");
        }

        for (const auto &[name, value] : arguments) {
            if (name == "MaximalDeformationStopCriterion") {
                optimizer->SetMaximalDeformationStopCriterion(value);
            } else if (name == "MaximumNumberOfIterations") {
                optimizer->SetMaximumNumberOfIterations(static_cast<int>(value));
            } else if (name == "LineSearchMaximalDeformationIntervalStopCriterion") {
                optimizer->SetLineSearchMaximalDeformationIntervalStopCriterion(value);
            } else if (name == "BFGS-MaximumMemoryLength") {
                optimizer->SetMaximumMemoryLength(static_cast<int>(value));
            } else if (name != "Verbose") {
                throw std::invalid_argument("optionName: " + name + " not understood");
            }
        }
        optimizer->SetMesh(AsMutable(mesh));
        optimizer->SetCostAndGradientCalculator(calculator.calculator);
    }

    std::pair<double, double> StepOptimizer(){
        const double maximalDeformation = optimizer->Step();
        return {optimizer->GetMinLogLikelihoodTimesPrior(), maximalDeformation};
    }

    void UpdateCalculator(const KvlCostAndGradientCalculator &calculator){
        optimizer->SetCostAndGradientCalculator(calculator.calculator, true, false, true);
    }

    void UpdateMesh(const KvlMesh &mesh){
        optimizer->SetMesh(AsMutable(mesh), false, false, true);
    }
};
#endif //GEMS_PYKVLOPTIMIZER_H
