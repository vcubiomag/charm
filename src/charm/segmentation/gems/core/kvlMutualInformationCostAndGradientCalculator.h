#ifndef __kvlMutualInformationCostAndGradientCalculator_h
#define __kvlMutualInformationCostAndGradientCalculator_h

#include <vector>

#include "itkImage.h"
#include "kvlAtlasMeshPositionCostAndGradientCalculator.h"

namespace kvl
{

/**
 * Probabilistic-atlas-based mutual information in the flavor of D'Agostino (MICCAI 2004), with two tricks taken from
 * John Ashburner's spm_maff8.m in SPM12:
 *
 *   1. The class-conditional distributions are estimated iteratively with EM. Standard partial volume interpolation
 *      (PVI) implementations of mutual information (cf. Maes, TMI 1997) do only a single iteration of this.
 *
 *   2. The gradients are approximate. They ignore the effect of the registration position on the class-conditional
 *      distributions, on which voxels fall in the region the histogram is computed over, and on the non-smooth
 *      entering and leaving of such voxels, which is what PVI was originally meant to address.
 *
 * The priors are rasterized once per evaluation, and the EM iterations run on the resulting flat array.
 */
class MutualInformationCostAndGradientCalculator: public AtlasMeshPositionCostAndGradientCalculator
{
public :
  using Self = MutualInformationCostAndGradientCalculator;
  using Superclass = AtlasMeshPositionCostAndGradientCalculator;
  using Pointer = itk::SmartPointer< Self >;
  using ConstPointer = itk::SmartPointer< const Self >;

  itkNewMacro( Self );
  itkOverrideGetNameOfClassMacro(MutualInformationCostAndGradientCalculator);

  using ImageType = itk::Image< float, 3 >;

  void SetImage( const ImageType* image );

  void Rasterize( const AtlasMesh* mesh ) override;

protected:
  MutualInformationCostAndGradientCalculator() = default;
  ~MutualInformationCostAndGradientCalculator() override = default;

  void AddDataContribution( std::size_t tetrahedron,
                            const AtlasMesh::PointType& p0,
                            const AtlasMesh::PointType& p1,
                            const AtlasMesh::PointType& p2,
                            const AtlasMesh::PointType& p3,
                            TetrahedronContribution& contribution ) const override;

private:
  ITK_DISALLOW_COPY_AND_MOVE(MutualInformationCostAndGradientCalculator);

  static constexpr int  m_NumberOfBins = 64;

  // The voxels with intensities inside of the robust range, which the flat arrays below are indexed by
  VoxelSelection  m_Voxels;
  std::vector< std::uint8_t >  m_Bins;

  std::vector< float >  m_Priors;
  std::vector< double >  m_ConditionalIntensityDistributions;
  std::size_t  m_NumberOfClasses = 0;
  double  m_NumberOfVoxels = 0.0;
};

} // end namespace kvl

#endif
