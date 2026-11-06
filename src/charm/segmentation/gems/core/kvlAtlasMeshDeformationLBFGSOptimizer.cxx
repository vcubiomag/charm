#include "kvlAtlasMeshDeformationLBFGSOptimizer.h"

#include <algorithm>
#include <numeric>

#include "kvlParallel.h"

namespace kvl
{

namespace
{

constexpr std::size_t  blockSize = 1 << 12;

const double* Begin( const AtlasPositionGradientContainerType* deformation )
{
  return deformation->CastToSTLConstContainer().data()->GetDataPointer();
}

double* Begin( AtlasPositionGradientContainerType* deformation )
{
  return deformation->CastToSTLContainer().data()->GetDataPointer();
}

// The inner products of vector with each of others, in one pass over the memory. The per-block results are summed in block
// order to not depend on the number of threads.
std::vector< double > ComputeInnerProducts( const AtlasPositionGradientContainerType* vector,
                                            const std::vector< const AtlasPositionGradientContainerType* >& others )
{
  const std::size_t  numberOfElements = 3 * vector->Size();
  const std::size_t  numberOfBlocks = ( numberOfElements + blockSize - 1 ) / blockSize;
  std::vector< double >  blockResults( numberOfBlocks * others.size() );
  const double*  values = Begin( vector );
  ParallelForBlocks( numberOfElements, blockSize,
    [ & ]( std::size_t block, std::size_t first, std::size_t last )
      {
      for ( std::size_t i = 0; i < others.size(); i++ )
        {
        blockResults[ block * others.size() + i ] = std::inner_product( values + first, values + last, Begin( others[ i ] ) + first, 0.0 );
        }
      } );

  std::vector< double >  results( others.size(), 0.0 );
  for ( std::size_t block = 0; block < numberOfBlocks; block++ )
    {
    for ( std::size_t i = 0; i < others.size(); i++ )
      {
      results[ i ] += blockResults[ block * others.size() + i ];
      }
    }
  return results;
}

} // end anonymous namespace

void
AtlasMeshDeformationLBFGSOptimizer
::WipeMemory()
{
  m_OldGradient = nullptr;
  m_OldSearchDirection = nullptr;
  m_Ss.clear();
  m_Ys.clear();
  m_SYs.clear();
  m_YYs.clear();
}

AtlasMeshDeformationOptimizer::Deformation::Pointer
AtlasMeshDeformationLBFGSOptimizer
::ComputeSearchDirection( double gamma ) const
{
  // The classic two-loop recursion does
  //
  //    q = gradient
  //    for i = 0 ... m-1:      alpha_i = ( s_i' * q ) / ( s_i' * y_i );   q = q - alpha_i * y_i
  //    r = gamma * q
  //    for i = m-1 ... 0:      beta_i = ( y_i' * r ) / ( s_i' * y_i );    r = r + s_i * ( alpha_i - beta_i )
  //
  // That is 4m operations over the whole deformation, each too small to run in parallel on its own, whereas this is by
  // far the largest serial part of the optimization. But every inner product in there expands into inner products of the
  // gradient with s_i and y_i, which are one pass over the memory, and inner products between the s_i and y_i, which don't
  // change from one iteration to the next. All the rest is arithmetic on these numbers, with r in the end a linear
  // combination of the gradient and the s_i and y_i: another single pass.
  const std::size_t  memoryLength = m_Ss.size();
  std::vector< const Deformation* >  history;
  for ( const auto& s : m_Ss )
    {
    history.push_back( s );
    }
  for ( const auto& y : m_Ys )
    {
    history.push_back( y );
    }
  const std::vector< double >  gradientProducts = ComputeInnerProducts( m_Gradient, history );

  std::vector< double >  alphas( memoryLength );
  for ( std::size_t i = 0; i < memoryLength; i++ )
    {
    double  sTimesQ = gradientProducts[ i ];
    for ( std::size_t j = 0; j < i; j++ )
      {
      sTimesQ -= alphas[ j ] * m_SYs[ i ][ j ];
      }
    alphas[ i ] = sTimesQ / m_SYs[ i ][ i ];
    }

  // The final coefficients of y_i in r, and of s_i
  std::vector< double >  yCoefficients( memoryLength );
  std::vector< double >  sCoefficients( memoryLength );
  for ( std::size_t i = 0; i < memoryLength; i++ )
    {
    yCoefficients[ i ] = -gamma * alphas[ i ];
    }
  for ( std::size_t i = memoryLength; i-- > 0; )
    {
    double  yTimesR = gamma * gradientProducts[ memoryLength + i ];
    for ( std::size_t j = 0; j < memoryLength; j++ )
      {
      yTimesR += yCoefficients[ j ] * m_YYs[ i ][ j ];
      }
    for ( std::size_t j = i + 1; j < memoryLength; j++ )
      {
      yTimesR += sCoefficients[ j ] * m_SYs[ j ][ i ];
      }
    sCoefficients[ i ] = alphas[ i ] - yTimesR / m_SYs[ i ][ i ];
    }

  // Direction is -r
  auto  searchDirection = Deformation::New();
  searchDirection->CastToSTLContainer().resize( m_Gradient->Size() );
  double*  direction = Begin( searchDirection.GetPointer() );
  const double*  gradient = Begin( m_Gradient.GetPointer() );
  const std::size_t  numberOfElements = 3 * m_Gradient->Size();
  ParallelForBlocks( numberOfElements, blockSize,
    [ & ]( std::size_t, std::size_t first, std::size_t last )
      {
      for ( std::size_t k = first; k < last; k++ )
        {
        direction[ k ] = -gamma * gradient[ k ];
        }
      for ( std::size_t i = 0; i < memoryLength; i++ )
        {
        const double*  s = Begin( m_Ss[ i ].GetPointer() );
        const double*  y = Begin( m_Ys[ i ].GetPointer() );
        const double  sCoefficient = -sCoefficients[ i ];
        const double  yCoefficient = -yCoefficients[ i ];
        for ( std::size_t k = first; k < last; k++ )
          {
          direction[ k ] += sCoefficient * s[ k ] + yCoefficient * y[ k ];
          }
        }
      } );
  return searchDirection;
}

double
AtlasMeshDeformationLBFGSOptimizer
::FindAndOptimizeNewSearchDirection()
{
  // The search direction is -H * gradient, with H the L-BFGS approximation of the inverse Hessian built from
  // H_0 = gamma * I and the stored differences in position and gradient
  double  gamma = 0.0;
  if ( !m_OldSearchDirection )
    {
    // With no history the direction is -gamma * gradient, and the line search starts at alpha = 1, so the first try
    // moves the farthest node by m_StartDistance
    gamma = m_StartDistance / ComputeMaximalDeformation( m_Gradient );
    }
  else
    {
    // s = x - xOld = alphaUsed * pOld, scaled in place since the old search direction isn't needed anymore
    Deformation::Pointer  s = m_OldSearchDirection;
    Scale( s, m_AlphaUsedLastTime );

    Deformation::Pointer  y = Copy( m_Gradient );
    AddScaled( y, -1.0, m_OldGradient );

    const double  inverseRho = ComputeInnerProduct( s, y );
    if ( inverseRho > 1e-10 )
      {
      m_Ss.insert( m_Ss.begin(), s );
      m_Ys.insert( m_Ys.begin(), y );
      if ( m_Ss.size() > m_MaximumMemoryLength )
        {
        m_Ss.pop_back();
        m_Ys.pop_back();
        }

      // Only the inner products with the new entries need computing
      const std::size_t  memoryLength = m_Ss.size();
      std::vector< const Deformation* >  allYs( m_Ys.begin(), m_Ys.end() );
      std::vector< const Deformation* >  allSs( m_Ss.begin(), m_Ss.end() );
      const std::vector< double >  newYProducts = ComputeInnerProducts( y, allYs );
      const std::vector< double >  newSProducts = ComputeInnerProducts( s, allYs );
      const std::vector< double >  newYWithSs = ComputeInnerProducts( y, allSs );

      const auto  grow =
        [ & ]( std::vector< std::vector< double > >& matrix, const std::vector< double >& newRow, const std::vector< double >& newColumn )
          {
          std::vector< std::vector< double > >  grown( memoryLength, std::vector< double >( memoryLength ) );
          for ( std::size_t i = 0; i < memoryLength; i++ )
            {
            grown[ 0 ][ i ] = newRow[ i ];
            grown[ i ][ 0 ] = newColumn[ i ];
            }
          for ( std::size_t i = 1; i < memoryLength; i++ )
            {
            for ( std::size_t j = 1; j < memoryLength; j++ )
              {
              grown[ i ][ j ] = matrix[ i - 1 ][ j - 1 ];
              }
            }
          matrix = std::move( grown );
          };
      grow( m_SYs, newSProducts, newYWithSs );
      grow( m_YYs, newYProducts, newYProducts );

      // gamma = ( s' * y ) / ( y' * y );
      gamma = inverseRho / newYProducts[ 0 ];
      }
    }

  Deformation::Pointer  searchDirection = this->ComputeSearchDirection( gamma );

  // BFGS requires a start of 1.0 for its proof of convergence properties
  constexpr double  startAlpha = 1.0;
  constexpr double  c1 = 1e-4;
  constexpr double  c2 = 0.9;
  m_OldGradient = m_Gradient;
  m_OldSearchDirection = searchDirection;
  double  alphaUsed = 0.0;
  this->DoLineSearch( m_Position, m_Cost, m_Gradient, searchDirection, startAlpha, c1, c2,
                      m_Position, m_Cost, m_Gradient, alphaUsed );

  m_AlphaUsedLastTime = alphaUsed;
  return alphaUsed * ComputeMaximalDeformation( searchDirection );
}

} // end namespace kvl
