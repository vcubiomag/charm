#ifndef __kvlAtlasMeshPositionCostAndGradientCalculator_h
#define __kvlAtlasMeshPositionCostAndGradientCalculator_h

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "itkAffineTransform.h"
#include "kvlAtlasMesh.h"
#include "kvlContainerPool.h"
#include "kvlMixtureContribution.h"
#include "kvlTetrahedronTable.h"

namespace kvl
{

/**
 * Evaluates a cost that sums over the tetrahedra of a mesh, and its gradient with respect to the node positions.
 *
 * Each tetrahedron's contribution goes to its own slot, and the slots are summed in table order afterwards. This makes
 * the rounding of the results, and so the outcome of the optimizers driven by them, independent of the number of threads
 * and of how the tetrahedra were scheduled over them.
 */
class AtlasMeshPositionCostAndGradientCalculator: public itk::Object
{
public :
  using Self = AtlasMeshPositionCostAndGradientCalculator;
  using Superclass = itk::Object;
  using Pointer = itk::SmartPointer< Self >;
  using ConstPointer = itk::SmartPointer< const Self >;

  itkNewMacro( Self );
  itkOverrideGetNameOfClassMacro(AtlasMeshPositionCostAndGradientCalculator);

  using TransformType = itk::AffineTransform< double, 3 >;

  void SetMeshToImageTransform( const TransformType* meshToImageTransform );

  AtlasPositionGradientContainerType* GetPositionGradient() const
    {
    return m_PositionGradient;
    }

  double GetMinLogLikelihoodTimesPrior() const
    {
    return m_MinLogLikelihoodTimesPrior;
    }

  virtual void Rasterize( const AtlasMesh* mesh );

  // Evaluates only the cost, which is enough for searches that compare positions
  void RasterizeCost( const AtlasMesh* mesh );

  enum BoundaryConditionType { NONE, SLIDING, AFFINE };
  void SetBoundaryCondition( BoundaryConditionType boundaryCondition )
    {
    m_BoundaryCondition = boundaryCondition;
    }

protected:
  AtlasMeshPositionCostAndGradientCalculator();
  ~AtlasMeshPositionCostAndGradientCalculator() override = default;

  // The contribution of one tetrahedron, with the gradient indexed by vertex and axis
  struct TetrahedronContribution
  {
    double  m_Cost;
    double  m_Gradient[ 4 ][ 3 ];
  };

  virtual void AddDataContribution( std::size_t tetrahedron,
                                    const AtlasMesh::PointType& p0,
                                    const AtlasMesh::PointType& p1,
                                    const AtlasMesh::PointType& p2,
                                    const AtlasMesh::PointType& p3,
                                    TetrahedronContribution& contribution ) const
    {
    }

  // For cost and gradient contributions that don't come from tetrahedra
  virtual void PostProcessCostAndGradient( const AtlasMesh* mesh )
    {
    }

  template< bool TAddCost, typename TGetValues >
  void AddMixtureContribution( std::size_t tetrahedron,
                               const AtlasMesh::PointType& p0,
                               const AtlasMesh::PointType& p1,
                               const AtlasMesh::PointType& p2,
                               const AtlasMesh::PointType& p3,
                               const VoxelSelection& voxels,
                               TGetValues&& getValues,
                               std::size_t classStride,
                               double gradientScale,
                               TetrahedronContribution& contribution ) const
    {
    const std::uint32_t  firstAlpha = m_Alphas->m_Offsets[ tetrahedron ];
    kvl::AddMixtureContribution< TAddCost >( TetrahedronRasterizer( p0, p1, p2, p3, voxels.GetSize() ), voxels,
                                             m_Alphas->m_Classes.data() + firstAlpha, m_Alphas->m_Alphas.data() + firstAlpha,
                                             m_Alphas->m_Offsets[ tetrahedron + 1 ] - firstAlpha,
                                             std::forward< TGetValues >( getValues ), classStride, gradientScale,
                                             contribution.m_Cost, contribution.m_Gradient );
    }

  double  m_MinLogLikelihoodTimesPrior = 0;
  bool  m_ComputeGradient = true;
  std::shared_ptr< const TetrahedronAlphas >  m_Alphas;

private:
  ITK_DISALLOW_COPY_AND_MOVE(AtlasMeshPositionCostAndGradientCalculator);

  void ImposeSlidingBoundaryConditions( const AtlasMesh* mesh );
  void ImposeAffineBoundaryConditions( const AtlasMesh* mesh );

  AtlasPositionGradientContainerType::Pointer  m_PositionGradient;
  std::atomic< bool >  m_Abort{ false };
  BoundaryConditionType  m_BoundaryCondition = SLIDING;
  vnl_matrix< double >  m_AffineProjectionMatrix;
  itk::Matrix< double >  m_SlidingBoundaryCorrectionMatrices[ 8 ];
  std::vector< TetrahedronContribution >  m_Contributions;
  ContainerPool< AtlasPositionGradientContainerType >  m_GradientPool;
};

} // end namespace kvl

#endif
