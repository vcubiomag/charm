#ifndef __kvlMixtureContribution_h
#define __kvlMixtureContribution_h

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "kvlFastMath.h"
#include "kvlTetrahedronRasterizer.h"
#include "kvlVoxelSelection.h"

namespace kvl
{

/**
 * Adds the gradient, and optionally the cost, of a data term that for each voxel is a mixture of per-class values, with the
 * mixing weights the interpolated alphas. For the voxel with row r in the selection, the value of class c is
 * getValues( r )[ c * classStride ]. The alphas are those of the classes in classes, for each vertex.
 *
 * The likelihood of a voxel is linear in the barycentric coordinates pi: likelihood = sum_u weights[ u ] * pi[ u ]. Its
 * contribution to the gradient of vertex v along an axis is pi[ v ] / likelihood * sum_u weights[ u ] * dpi[ u ], with dpi
 * constant over the tetrahedron, so that only the voxel-dependent part is accumulated voxel by voxel. Since the pi sum to
 * one and the dpi to zero, the last vertex can be eliminated from both sums, which takes a quarter of the terms away and
 * avoids the cancellation of adding up nearly equal weights in a homogeneous region.
 *
 * Everything per voxel is in single precision over the four vertices: likelihoods and alphas are stored in single
 * precision to begin with, a likelihood has a floor far above where that matters, and tetrahedra have too few voxels on
 * average for vectorizing across voxels to pay for the extra loops ending at an unpredictable moment.
 */
template< bool TAddCost, typename TGetValues >
void AddMixtureContribution( const TetrahedronRasterizer& rasterizer,
                             const VoxelSelection& voxels,
                             const std::uint16_t* classes,
                             const std::array< float, 4 >* classAlphas,
                             std::size_t numberOfAlphas,
                             TGetValues&& getValues,
                             std::size_t classStride,
                             double gradientScale,
                             double& cost,
                             double ( &gradient )[ 4 ][ 3 ] )
{
  thread_local VoxelList  voxelList;
  thread_local std::vector< std::uint32_t >  rows;
  thread_local std::vector< float >  likelihoods;
  thread_local std::vector< std::array< float, 4 > >  alphas;
  thread_local std::vector< std::size_t >  valueOffsets;

  const std::size_t  numberOfCandidates = rasterizer.CollectVoxels( voxels.GetSize(), voxelList );
  if ( rows.size() < numberOfCandidates )
    {
    rows.resize( 2 * numberOfCandidates );
    }
  const std::size_t  numberOfVoxels = voxels.Select( voxelList, numberOfCandidates, rows.data() );
  if ( numberOfVoxels == 0 )
    {
    return;
    }

  alphas.resize( numberOfAlphas );
  valueOffsets.resize( numberOfAlphas );
  for ( std::size_t a = 0; a < numberOfAlphas; a++ )
    {
    const auto&  vertexAlphas = classAlphas[ a ];
    alphas[ a ] = { vertexAlphas[ 0 ] - vertexAlphas[ 3 ], vertexAlphas[ 1 ] - vertexAlphas[ 3 ], vertexAlphas[ 2 ] - vertexAlphas[ 3 ], vertexAlphas[ 3 ] };
    valueOffsets[ a ] = classes[ a ] * classStride;
    }

  const std::int32_t* const  xs = voxelList.m_X.data();
  const std::int32_t* const  ys = voxelList.m_Y.data();
  const std::int32_t* const  zs = voxelList.m_Z.data();

  // Coordinates relative to a voxel of the tetrahedron are small integers, which single precision represents exactly
  double  piAtOrigin[ 4 ];
  rasterizer.GetBarycentricCoordinates( xs[ 0 ], ys[ 0 ], zs[ 0 ], piAtOrigin );
  float  piStart[ 4 ];
  float  piStepX[ 4 ];
  float  piStepY[ 4 ];
  float  piStepZ[ 4 ];
  for ( int vertex = 0; vertex < 4; vertex++ )
    {
    piStart[ vertex ] = static_cast< float >( piAtOrigin[ vertex ] );
    piStepX[ vertex ] = static_cast< float >( rasterizer.GetBarycentricGradient( vertex, 0 ) );
    piStepY[ vertex ] = static_cast< float >( rasterizer.GetBarycentricGradient( vertex, 1 ) );
    piStepZ[ vertex ] = static_cast< float >( rasterizer.GetBarycentricGradient( vertex, 2 ) );
    }

  for ( std::size_t voxel = 0; voxel < numberOfVoxels; voxel++ )
    {
#if defined( __GNUC__ ) || defined( __clang__ )
    __builtin_prefetch( getValues( rows[ voxel ] ) );
#endif
    }

  if ( likelihoods.size() < numberOfVoxels )
    {
    likelihoods.resize( 2 * numberOfVoxels );
    }

  // Single precision sums are only good for so many terms
  constexpr std::size_t  batchSize = 32;
  double  accumulated[ 4 ][ 3 ] = {};
  for ( std::size_t batchStart = 0; batchStart < numberOfVoxels; batchStart += batchSize )
    {
    const std::size_t  batchEnd = std::min( numberOfVoxels, batchStart + batchSize );
    float  sum0[ 4 ] = {};
    float  sum1[ 4 ] = {};
    float  sum2[ 4 ] = {};
    for ( std::size_t voxel = batchStart; voxel < batchEnd; voxel++ )
      {
      const auto*  values = getValues( rows[ voxel ] );
      const float  offsetX = static_cast< float >( xs[ voxel ] - xs[ 0 ] );
      const float  offsetY = static_cast< float >( ys[ voxel ] - ys[ 0 ] );
      const float  offsetZ = static_cast< float >( zs[ voxel ] - zs[ 0 ] );
      float  pi[ 4 ];
      for ( int vertex = 0; vertex < 4; vertex++ )
        {
        pi[ vertex ] = piStart[ vertex ] + piStepX[ vertex ] * offsetX + piStepY[ vertex ] * offsetY + piStepZ[ vertex ] * offsetZ;
        }

      // weights[ 0..2 ] are relative to the weight of the last vertex, which is in weights[ 3 ]
      float  weights[ 4 ] = {};
      for ( std::size_t a = 0; a < numberOfAlphas; a++ )
        {
        const float  value = static_cast< float >( values[ valueOffsets[ a ] ] );
        const float*  alpha = alphas[ a ].data();
        for ( int vertex = 0; vertex < 4; vertex++ )
          {
          weights[ vertex ] += value * alpha[ vertex ];
          }
        }

      const float  likelihood = 1e-15f + weights[ 3 ] + weights[ 0 ] * pi[ 0 ] + weights[ 1 ] * pi[ 1 ] + weights[ 2 ] * pi[ 2 ]; // don't want to divide by zero
      likelihoods[ voxel ] = likelihood;
      const float  inverseLikelihood = 1.0f / likelihood;
      for ( int vertex = 0; vertex < 4; vertex++ )
        {
        const float  scale = pi[ vertex ] * inverseLikelihood;
        sum0[ vertex ] += scale * weights[ 0 ];
        sum1[ vertex ] += scale * weights[ 1 ];
        sum2[ vertex ] += scale * weights[ 2 ];
        }
      }
    for ( int vertex = 0; vertex < 4; vertex++ )
      {
      accumulated[ vertex ][ 0 ] += sum0[ vertex ];
      accumulated[ vertex ][ 1 ] += sum1[ vertex ];
      accumulated[ vertex ][ 2 ] += sum2[ vertex ];
      }
    }

  if constexpr ( TAddCost )
    {
    cost -= SumOfLogarithms( likelihoods.data(), numberOfVoxels );
    }
  for ( int axis = 0; axis < 3; axis++ )
    {
    const double  step0 = rasterizer.GetBarycentricGradient( 0, axis );
    const double  step1 = rasterizer.GetBarycentricGradient( 1, axis );
    const double  step2 = rasterizer.GetBarycentricGradient( 2, axis );
    for ( int vertex = 0; vertex < 4; vertex++ )
      {
      gradient[ vertex ][ axis ] += ( accumulated[ vertex ][ 0 ] * step0 + accumulated[ vertex ][ 1 ] * step1 + accumulated[ vertex ][ 2 ] * step2 ) * gradientScale;
      }
    }
}

} // end namespace kvl

#endif
