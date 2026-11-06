#ifndef GEMS_PYKVLREGISTRATION_H
#define GEMS_PYKVLREGISTRATION_H

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "itkBSplineInterpolateImageFunction.h"
#include "itkCenteredTransformInitializer.h"
#include "itkConjugateGradientLineSearchOptimizerv4.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegistrationMethodv4.h"
#include "itkMattesMutualInformationImageToImageMetricv4.h"
#include "itkNormalizeImageFilter.h"
#include "itkRegistrationParameterScalesFromPhysicalShift.h"
#include "itkResampleImageFilter.h"
#include "pyKvlNumpy.h"

template <typename TTransform>
class KvlRegistration {
    using RegistrationImageType = itk::Image<float, 3>;
    using MetricType = itk::MattesMutualInformationImageToImageMetricv4<RegistrationImageType, RegistrationImageType>;
    using OptimizerType = itk::ConjugateGradientLineSearchOptimizerv4Template<double>;

    RegistrationImageType::Pointer fixedImage;
    RegistrationImageType::Pointer movingImage;
    typename TTransform::Pointer initialTransform = TTransform::New();
    typename TTransform::Pointer finalTransform;
    double translationScale = -100;
    int numberOfIterations = 100;
    int numberOfHistogramBins = 64;
    std::vector<double> shrinkScales = {2.0, 1.0, 0.0};
    double backgroundGrayLevel = 0;
    std::vector<double> smoothingSigmas = {4.0, 2.0, 0.0};
    bool useCenterOfMass = false;
    double samplingPercentage = 0.5;

public:
    KvlRegistration() = default;

    KvlRegistration(double translationScale, int numberOfIterations, int numberOfHistogramBins,
                    const std::vector<double> &shrinkScales, double backgroundGrayLevel,
                    const std::vector<double> &smoothingSigmas, bool useCenterOfMass, double samplingPercentage,
                    const std::string &interpolator)
        : translationScale(translationScale), numberOfIterations(numberOfIterations),
          numberOfHistogramBins(numberOfHistogramBins), shrinkScales(shrinkScales),
          backgroundGrayLevel(backgroundGrayLevel), smoothingSigmas(smoothingSigmas),
          useCenterOfMass(useCenterOfMass), samplingPercentage(samplingPercentage) {
        if (interpolator != "b") {
            throw std::invalid_argument("Only B-spline interpolation is supported.");
        }
    }

    void ReadImages(const std::string &fixedImageFileName, const std::string &movingImageFileName) {
        fixedImage = itk::ReadImage<RegistrationImageType>(fixedImageFileName);
        movingImage = itk::ReadImage<RegistrationImageType>(movingImageFileName);
    }

    void InitializeTransform() {
        auto initializer = itk::CenteredTransformInitializer<TTransform, RegistrationImageType, RegistrationImageType>::New();
        initializer->SetTransform(initialTransform);
        initializer->SetFixedImage(fixedImage);
        initializer->SetMovingImage(movingImage);
        if (useCenterOfMass) {
            initializer->MomentsOn();
        } else {
            initializer->GeometryOn();
        }
        initializer->InitializeTransform();
    }

    void Register() {
        using RegistrationType = itk::ImageRegistrationMethodv4<RegistrationImageType, RegistrationImageType, TTransform>;

        auto metric = MetricType::New();
        metric->SetNumberOfHistogramBins(numberOfHistogramBins);

        auto fixedNormalizer = itk::NormalizeImageFilter<RegistrationImageType, RegistrationImageType>::New();
        auto movingNormalizer = itk::NormalizeImageFilter<RegistrationImageType, RegistrationImageType>::New();
        fixedNormalizer->SetInput(fixedImage);
        movingNormalizer->SetInput(movingImage);
        fixedNormalizer->Update();
        movingNormalizer->Update();

        auto optimizer = OptimizerType::New();
        optimizer->SetDoEstimateLearningRateOnce(false);
        optimizer->SetDoEstimateLearningRateAtEachIteration(true);
        optimizer->SetLowerLimit(0);
        optimizer->SetUpperLimit(2);
        optimizer->SetEpsilon(0.2);
        optimizer->SetNumberOfIterations(numberOfIterations);
        optimizer->SetMinimumConvergenceValue(1e-4);
        optimizer->SetConvergenceWindowSize(5);

        auto scalesEstimator = itk::RegistrationParameterScalesFromPhysicalShift<MetricType>::New();
        if (translationScale > 0) {
            OptimizerType::ScalesType scales(initialTransform->GetNumberOfParameters());
            scales.fill(1.0);
            scales[scales.Size() - 1] = translationScale;
            scales[scales.Size() - 2] = translationScale;
            scales[scales.Size() - 3] = translationScale;
            optimizer->SetScales(scales);
        } else {
            scalesEstimator->SetMetric(metric);
            scalesEstimator->SetTransformForward(true);
            optimizer->SetScalesEstimator(scalesEstimator);
        }

        auto registration = RegistrationType::New();
        registration->SetOptimizer(optimizer);
        registration->SetMetric(metric);
        registration->SetFixedImage(fixedNormalizer->GetOutput());
        registration->SetMovingImage(movingNormalizer->GetOutput());
        registration->SetInitialTransform(initialTransform);
        registration->MetricSamplingReinitializeSeed(121213);
        registration->SetMetricSamplingStrategy(RegistrationType::MetricSamplingStrategyEnum::REGULAR);
        const unsigned int numberOfLevels = shrinkScales.size();
        typename RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
        typename RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
        shrinkFactorsPerLevel.SetSize(numberOfLevels);
        smoothingSigmasPerLevel.SetSize(numberOfLevels);
        for (unsigned int level = 0; level < numberOfLevels; ++level) {
            shrinkFactorsPerLevel[level] = static_cast<unsigned long>(std::pow(2, shrinkScales.at(level)));
            smoothingSigmasPerLevel[level] = smoothingSigmas.at(level);
        }
        // A transform has few parameters, and the metric takes nearly all of the time, in proportion to the number of samples.
        // A regular grid of a fixed number of samples scores as well on independent mutual information as half of all
        // voxels of a 0.7mm image do at random, in a seventh of the time.
        constexpr double maximumNumberOfSamples = 1e6;
        const double numberOfVoxels = static_cast<double>(fixedImage->GetLargestPossibleRegion().GetNumberOfPixels());
        typename RegistrationType::MetricSamplingPercentageArrayType samplingPercentagePerLevel;
        samplingPercentagePerLevel.SetSize(numberOfLevels);
        for (unsigned int level = 0; level < numberOfLevels; ++level) {
            const double shrink = static_cast<double>(shrinkFactorsPerLevel[level]);
            const double voxelsAtLevel = numberOfVoxels / (shrink * shrink * shrink);
            samplingPercentagePerLevel[level] = std::min(samplingPercentage, maximumNumberOfSamples / voxelsAtLevel);
        }
        // Changing the number of levels resets the sampling of all of them
        registration->SetNumberOfLevels(numberOfLevels);
        registration->SetMetricSamplingPercentagePerLevel(samplingPercentagePerLevel);
        registration->SetShrinkFactorsPerLevel(shrinkFactorsPerLevel);
        registration->SetSmoothingSigmasAreSpecifiedInPhysicalUnits(true);
        registration->SetSmoothingSigmasPerLevel(smoothingSigmasPerLevel);

        registration->Update();

        // ITK's metric sums over its threads in whichever order they finish, which leaves differences of around 1e-12
        // between runs. Everything downstream is sensitive enough to rounding for those to add up to a visibly different
        // result, whereas this rounding is far from anything that matters to the registration itself.
        constexpr double precision = 1e6;
        auto parameters = registration->GetOutput()->Get()->GetParameters();
        for (unsigned int i = 0; i < parameters.Size(); i++) {
            parameters[i] = std::round(parameters[i] * precision) / precision;
        }
        finalTransform = TTransform::New();
        finalTransform->SetFixedParameters(registration->GetOutput()->Get()->GetFixedParameters());
        finalTransform->SetParameters(parameters);
    }

    void WriteOutResults(const std::string &outputFileName) {
        auto interpolator = itk::BSplineInterpolateImageFunction<RegistrationImageType, double>::New();
        interpolator->SetSplineOrder(4);

        auto resampler = itk::ResampleImageFilter<RegistrationImageType, RegistrationImageType>::New();
        resampler->SetTransform(finalTransform);
        resampler->SetInput(movingImage);
        resampler->SetReferenceImage(fixedImage);
        resampler->UseReferenceImageOn();
        resampler->SetDefaultPixelValue(backgroundGrayLevel);
        resampler->SetInterpolator(interpolator);

        itk::WriteImage(resampler->GetOutput(), outputFileName);
    }

    NumpyArray<double> GetFinalTransformation() const {
        return transformToNumpy(finalTransform->GetMatrix(), finalTransform->GetOffset());
    }
};

#endif //GEMS_PYKVLREGISTRATION_H
