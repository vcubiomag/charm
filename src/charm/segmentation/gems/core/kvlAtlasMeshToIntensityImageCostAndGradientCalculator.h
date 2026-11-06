#ifndef __kvlAtlasMeshToIntensityImageCostAndGradientCalculator_h
#define __kvlAtlasMeshToIntensityImageCostAndGradientCalculator_h

#include "kvlAtlasMeshPositionCostAndGradientCalculator.h"

namespace kvl
{

class AtlasMeshToIntensityImageCostAndGradientCalculator:
        public AtlasMeshPositionCostAndGradientCalculator
{
public :
  using Self = AtlasMeshToIntensityImageCostAndGradientCalculator;
  using Superclass = AtlasMeshPositionCostAndGradientCalculator;
  using Pointer = itk::SmartPointer< Self >;
  using ConstPointer = itk::SmartPointer< const Self >;

  itkNewMacro( Self );
  itkOverrideGetNameOfClassMacro(AtlasMeshToIntensityImageCostAndGradientCalculator);

  // Likelihoods holds P( intensity | class ) as ( voxels of the selection x classes ). The caller keeps both alive.
  void SetLikelihoods( const float* likelihoods, std::size_t numberOfClasses, const VoxelSelection* voxels )
    {
    m_Likelihoods = likelihoods;
    m_NumberOfClasses = numberOfClasses;
    m_Voxels = voxels;
    }

protected:
  AtlasMeshToIntensityImageCostAndGradientCalculator() = default;
  ~AtlasMeshToIntensityImageCostAndGradientCalculator() override = default;

  void AddDataContribution( std::size_t tetrahedron,
                            const AtlasMesh::PointType& p0,
                            const AtlasMesh::PointType& p1,
                            const AtlasMesh::PointType& p2,
                            const AtlasMesh::PointType& p3,
                            TetrahedronContribution& contribution ) const override;

private:
  ITK_DISALLOW_COPY_AND_MOVE(AtlasMeshToIntensityImageCostAndGradientCalculator);

  const float*  m_Likelihoods = nullptr;
  std::size_t  m_NumberOfClasses = 0;
  const VoxelSelection*  m_Voxels = nullptr;
};

} // end namespace kvl

#endif
