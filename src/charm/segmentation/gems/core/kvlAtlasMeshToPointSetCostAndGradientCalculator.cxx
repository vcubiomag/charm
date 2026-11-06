#include "kvlAtlasMeshToPointSetCostAndGradientCalculator.h"

namespace kvl
{

void
AtlasMeshToPointSetCostAndGradientCalculator
::PostProcessCostAndGradient( const AtlasMesh* mesh )
{
  if ( !m_TargetPoints )
    {
    itkExceptionMacro( << "No target point set!" );
    }

  const auto&  targets = m_TargetPoints->CastToSTLConstContainer();
  const auto&  points = mesh->GetPoints()->CastToSTLConstContainer();
  auto&  gradient = this->GetPositionGradient()->CastToSTLContainer();
  for ( std::size_t node = 0; node < targets.size(); node++ )
    {
    for ( int axis = 0; axis < 3; axis++ )
      {
      // Zero marks a missing target
      if ( targets[ node ][ axis ] == 0 )
        {
        continue;
        }

      const double  distance = points[ node ][ axis ] - targets[ node ][ axis ];
      m_MinLogLikelihoodTimesPrior += 0.5 * distance * distance;
      gradient[ node ][ axis ] += distance;
      }
    }
}

} // end namespace kvl
