#include "kvlAtlasMeshDeformationOptimizer.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace kvl
{

namespace
{

using Deformation = AtlasPositionGradientContainerType;

const double* Begin( const Deformation* deformation )
{
  return deformation->CastToSTLConstContainer().data()->GetDataPointer();
}

double* Begin( Deformation* deformation )
{
  return deformation->CastToSTLContainer().data()->GetDataPointer();
}

std::size_t NumberOfElements( const Deformation* deformation )
{
  return 3 * deformation->Size();
}

// One evaluated point along the search direction
struct LineSearchPoint
{
  double  m_Alpha;
  double  m_Cost;
  double  m_DirectionalDerivative;
  Deformation::ConstPointer  m_Gradient;
};

} // end anonymous namespace

void
AtlasMeshDeformationOptimizer
::GetCostAndGradient( const AtlasMesh::PointsContainer* position, double& cost, Deformation::Pointer& gradient )
{
  if ( !m_EvaluationMesh )
    {
    m_EvaluationMesh = AtlasMesh::New();
    m_EvaluationMesh->SetCells( m_Mesh->GetCells() );
    m_EvaluationMesh->SetPointData( m_Mesh->GetPointData() );
    m_EvaluationMesh->SetCellData( m_Mesh->GetCellData() );
    }
  m_EvaluationMesh->SetPoints( const_cast< AtlasMesh::PointsContainer* >( position ) );

  m_Calculator->Rasterize( m_EvaluationMesh );

  cost = m_Calculator->GetMinLogLikelihoodTimesPrior();
  gradient = m_Calculator->GetPositionGradient();
}

double
AtlasMeshDeformationOptimizer
::Step()
{
  if ( m_IterationNumber >= m_MaximumNumberOfIterations )
    {
    return 0.0;
    }

  if ( !m_Initialized )
    {
    if ( !m_Calculator )
      {
      itkExceptionMacro( << "Cost and gradient calculator missing!" );
      }
    m_Position = m_Mesh->GetPoints();
    this->GetCostAndGradient( m_Position, m_Cost, m_Gradient );
    m_Initialized = true;
    }

  double  maximalDeformation = 0.0;
  try
    {
    maximalDeformation = this->FindAndOptimizeNewSearchDirection();
    }
  catch( itk::ExceptionObject& err )
    {
    std::cout << "Something wrong with finding and/or optimizing the search direction -- giving up" << std::endl;
    std::cout << "(" << err << ")" << std::endl;
    return 0.0;
    }

  m_Mesh->SetPoints( m_Position );

  if ( maximalDeformation <= m_MaximalDeformationStopCriterion && ( m_IterationNumber > 0 || maximalDeformation == 0.0 ) )
    {
    return 0.0;
    }

  m_IterationNumber++;
  return maximalDeformation;
}

double
AtlasMeshDeformationOptimizer
::ComputeMaximalDeformation( const Deformation* deformation )
{
  double  maximalSquaredMagnitude = 0.0;
  for ( const auto& vector : deformation->CastToSTLConstContainer() )
    {
    maximalSquaredMagnitude = std::max( maximalSquaredMagnitude, vector.GetSquaredNorm() );
    }
  return std::sqrt( maximalSquaredMagnitude );
}

double
AtlasMeshDeformationOptimizer
::ComputeInnerProduct( const Deformation* deformation1, const Deformation* deformation2 )
{
  const double*  begin = Begin( deformation1 );
  return std::inner_product( begin, begin + NumberOfElements( deformation1 ), Begin( deformation2 ), 0.0 );
}

void
AtlasMeshDeformationOptimizer
::AddScaled( Deformation* deformation, double beta, const Deformation* other )
{
  double*  values = Begin( deformation );
  const double*  otherValues = Begin( other );
  const std::size_t  count = NumberOfElements( deformation );
  for ( std::size_t i = 0; i < count; i++ )
    {
    values[ i ] += beta * otherValues[ i ];
    }
}

void
AtlasMeshDeformationOptimizer
::Scale( Deformation* deformation, double beta )
{
  double*  values = Begin( deformation );
  const std::size_t  count = NumberOfElements( deformation );
  for ( std::size_t i = 0; i < count; i++ )
    {
    values[ i ] *= beta;
    }
}

AtlasMeshDeformationOptimizer::Deformation::Pointer
AtlasMeshDeformationOptimizer
::Copy( const Deformation* deformation )
{
  auto  copy = Deformation::New();
  copy->CastToSTLContainer() = deformation->CastToSTLConstContainer();
  return copy;
}

void
AtlasMeshDeformationOptimizer
::DoLineSearch( const AtlasMesh::PointsContainer* startPosition,
                double startCost,
                const Deformation* startGradient,
                const Deformation* searchDirection,
                double startAlpha,
                double c1,
                double c2,
                AtlasMesh::PointsContainer::Pointer& newPosition,
                double& newCost,
                Deformation::Pointer& newGradient,
                double& alphaUsed )
{
  // startPosition and newPosition may be the same object. Hold a reference so that assigning newPosition can't destroy
  // startPosition while it's still in use.
  const AtlasMesh::PointsContainer::ConstPointer  startPositionHolder = startPosition;

  const auto  getPosition =
    [ & ]( double alpha )
      {
      auto  position = m_PositionPool.Get();
      const auto&  start = startPosition->CastToSTLConstContainer();
      const auto&  direction = searchDirection->CastToSTLConstContainer();
      auto&  values = position->CastToSTLContainer();
      values.resize( start.size() );
      for ( std::size_t node = 0; node < start.size(); node++ )
        {
        values[ node ] = start[ node ] + alpha * direction[ node ];
        }
      return position;
      };

  const auto  finish =
    [ & ]( const LineSearchPoint& point, AtlasMesh::PointsContainer* position )
      {
      newPosition = position ? AtlasMesh::PointsContainer::Pointer( position ) : getPosition( point.m_Alpha );
      newCost = point.m_Cost;
      newGradient = const_cast< Deformation* >( point.m_Gradient.GetPointer() );
      alphaUsed = point.m_Alpha;
      };

  const double  initialDirectionalDerivative = ComputeInnerProduct( startGradient, searchDirection );
  if ( initialDirectionalDerivative >= 0 )
    {
    itkExceptionMacro( << "search direction is not a descent direction!" )
    }

  LineSearchPoint  previous{ 0.0, startCost, initialDirectionalDerivative, startGradient };

  const double  maximalDeformationOfSearchDirection = ComputeMaximalDeformation( searchDirection );
  const double  alphaMax = m_LineSearchMaximalDeformationLimit / maximalDeformationOfSearchDirection;
  double  alpha = std::min( startAlpha, alphaMax );

  bool  bracketingFailed = true;
  constexpr int  maximumOfBracketingIterations = 10;
  LineSearchPoint  low{};
  LineSearchPoint  high{};

  const auto  evaluate =
    [ & ]( double evaluatedAlpha, AtlasMesh::PointsContainer::Pointer& position )
      {
      position = getPosition( evaluatedAlpha );
      LineSearchPoint  point{ evaluatedAlpha, 0.0, 0.0, nullptr };
      Deformation::Pointer  gradient;
      this->GetCostAndGradient( position, point.m_Cost, gradient );
      point.m_DirectionalDerivative = ComputeInnerProduct( gradient, searchDirection );
      point.m_Gradient = gradient.GetPointer();
      return point;
      };

  const auto  isSufficientlyFlat =
    [ & ]( const LineSearchPoint& point )
      {
      return std::abs( point.m_DirectionalDerivative ) <= -c2 * initialDirectionalDerivative;
      };

  for ( int bracketingIterationNumber = 0; bracketingIterationNumber < maximumOfBracketingIterations; bracketingIterationNumber++ )
    {
    AtlasMesh::PointsContainer::Pointer  position;
    const LineSearchPoint  current = evaluate( alpha, position );

    // A point that is too costly or has a positive slope brackets a minimum, so zooming can start. A point that meets
    // both Wolfe conditions is accepted right away. Any other point means the minimum may lie farther out.
    if ( current.m_Cost > startCost + c1 * alpha * initialDirectionalDerivative ||
         ( current.m_Cost >= previous.m_Cost && bracketingIterationNumber > 0 ) )
      {
      low = previous;
      high = current;
      bracketingFailed = false;
      break;
      }

    if ( isSufficientlyFlat( current ) )
      {
      finish( current, position );
      return;
      }

    if ( current.m_DirectionalDerivative > 0 )
      {
      low = current;
      high = previous;
      bracketingFailed = false;
      break;
      }

    if ( alpha >= alphaMax )
      {
      previous = current;
      break;
      }

    // Expand to the minimum of a fitted cubic if it has one, but by enough to reach alphaMax within
    // maximumOfBracketingIterations, and never beyond alphaMax
    const double  allowableRangeMin = std::exp( std::log( alpha ) + ( std::log( alphaMax ) - std::log( alpha ) ) /
                                                ( maximumOfBracketingIterations - bracketingIterationNumber ) );
    const double  d1 = previous.m_DirectionalDerivative + current.m_DirectionalDerivative -
                       3 * ( previous.m_Cost - current.m_Cost ) / ( previous.m_Alpha - alpha );
    const double  d2Square = d1 * d1 - previous.m_DirectionalDerivative * current.m_DirectionalDerivative;
    double  newAlpha = allowableRangeMin;
    if ( d2Square > 0 )
      {
      const double  d2 = std::sqrt( d2Square );
      newAlpha = alpha - ( alpha - previous.m_Alpha ) * ( current.m_DirectionalDerivative + d2 - d1 ) /
                         ( current.m_DirectionalDerivative - previous.m_DirectionalDerivative + 2 * d2 );
      newAlpha = std::min( std::max( newAlpha, allowableRangeMin ), alphaMax );
      }

    previous = current;
    alpha = newAlpha;
    }

  if ( bracketingFailed )
    {
    // No point was bad enough to bracket a minimum or good enough to accept, so take the last one
    finish( previous, nullptr );
    return;
    }

  while ( true )
    {
    // Use the minimum of the fitted cubic if it has one, otherwise bisect
    const LineSearchPoint&  left = ( high.m_Alpha > low.m_Alpha ) ? low : high;
    const LineSearchPoint&  right = ( high.m_Alpha > low.m_Alpha ) ? high : low;

    const double  d1 = left.m_DirectionalDerivative + right.m_DirectionalDerivative -
                       3 * ( left.m_Cost - right.m_Cost ) / ( left.m_Alpha - right.m_Alpha );
    const double  d2Square = d1 * d1 - left.m_DirectionalDerivative * right.m_DirectionalDerivative;
    if ( d2Square < 0 || high.m_Cost == itk::NumericTraits< double >::max() )
      {
      alpha = ( left.m_Alpha + right.m_Alpha ) / 2;
      }
    else
      {
      const double  d2 = std::sqrt( d2Square );
      alpha = right.m_Alpha - ( right.m_Alpha - left.m_Alpha ) * ( right.m_DirectionalDerivative + d2 - d1 ) /
                              ( right.m_DirectionalDerivative - left.m_DirectionalDerivative + 2 * d2 );
      }

    // Staying away from the ends shrinks the interval by at least 10% per iteration
    const double  margin = 0.1 * ( right.m_Alpha - left.m_Alpha );
    alpha = std::min( std::max( alpha, left.m_Alpha + margin ), right.m_Alpha - margin );

    AtlasMesh::PointsContainer::Pointer  position;
    const LineSearchPoint  current = evaluate( alpha, position );

    if ( current.m_Cost > startCost + c1 * alpha * initialDirectionalDerivative || current.m_Cost >= low.m_Cost )
      {
      high = current;
      }
    else
      {
      if ( isSufficientlyFlat( current ) )
        {
        finish( current, position );
        return;
        }

      if ( current.m_DirectionalDerivative * ( high.m_Alpha - low.m_Alpha ) >= 0 )
        {
        high = low;
        }
      low = current;
      }

    if ( std::abs( high.m_Alpha - low.m_Alpha ) * maximalDeformationOfSearchDirection < m_LineSearchMaximalDeformationIntervalStopCriterion )
      {
      finish( low, nullptr );
      return;
      }
    }
}

} // end namespace kvl
