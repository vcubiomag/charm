#include "kvlAtlasMeshToIntensityImageCostAndGradientCalculator.h"

namespace kvl
{

void
AtlasMeshToIntensityImageCostAndGradientCalculator
::AddDataContribution( std::size_t tetrahedron,
                       const AtlasMesh::PointType& p0,
                       const AtlasMesh::PointType& p1,
                       const AtlasMesh::PointType& p2,
                       const AtlasMesh::PointType& p3,
                       TetrahedronContribution& contribution ) const
{
  this->AddMixtureContribution< true >( tetrahedron, p0, p1, p2, p3, *m_Voxels,
                                        [ this ]( std::uint32_t row ) { return m_Likelihoods + row * m_NumberOfClasses; },
                                        1, 1.0, contribution );
}

} // end namespace kvl
