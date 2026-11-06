#ifndef __kvlAtlasMeshDrawer_h
#define __kvlAtlasMeshDrawer_h

#include <algorithm>
#include <array>
#include <vector>

#include "kvlParallel.h"
#include "kvlTetrahedronRasterizer.h"
#include "kvlTetrahedronTable.h"

namespace kvl
{

// A channel of a tetrahedron: the values to interpolate between its vertices
template< typename TValue >
struct DrawnChannel
{
  std::size_t  m_Channel;
  std::array< TValue, 4 >  m_Values;
};

// Interpolates values given at the mesh nodes over all voxels the mesh covers. For each tetrahedron,
// getChannels( t, channels ) fills in the channels to draw, and write( channel, voxel, count, value, step ) gets one
// row of one channel, with voxel the F-order offset of the first voxel and value + i * step the value of voxel + i.
template< typename TValue, typename TGetChannels, typename TWrite >
void DrawMesh( const AtlasMesh* mesh, const ImageSize& size, TGetChannels&& getChannels, TWrite&& write )
{
  const auto  table = TetrahedronTable::Get( mesh );
  const auto&  points = mesh->GetPoints()->CastToSTLConstContainer();
  const long  sliceSize = size[ 0 ] * size[ 1 ];
  ParallelForBlocks( table->Size(), 1024,
    [ & ]( std::size_t, std::size_t first, std::size_t last )
      {
      std::vector< DrawnChannel< TValue > >  channels;
      for ( std::size_t t = first; t < last; t++ )
        {
        channels.clear();
        getChannels( t, channels );
        if ( channels.empty() )
          {
          continue;
          }

        const auto&  vertices = table->m_Vertices[ t ];
        const TetrahedronRasterizer  rasterizer( points[ vertices[ 0 ] ], points[ vertices[ 1 ] ],
                                                 points[ vertices[ 2 ] ], points[ vertices[ 3 ] ], size );
        rasterizer.ForEachRow( size,
          [ & ]( long x, long y, long z, long count, const double* pi, const double* piStep )
            {
            const long  voxel = x + y * size[ 0 ] + z * sliceSize;
            for ( const auto& [ channel, values ] : channels )
              {
              double  value = 0.0;
              double  step = 0.0;
              for ( int vertex = 0; vertex < 4; vertex++ )
                {
                value += values[ vertex ] * pi[ vertex ];
                step += values[ vertex ] * piStep[ vertex ];
                }
              write( channel, voxel, count, value, step );
              }
            } );
        }
      } );
}

template< typename TWrite >
void DrawAlphas( const AtlasMesh* mesh, const ImageSize& size, TWrite&& write )
{
  const auto  alphas = TetrahedronAlphas::Get( mesh );
  DrawMesh< float >( mesh, size,
    [ & ]( std::size_t t, auto& channels )
      {
      for ( std::uint32_t a = alphas->m_Offsets[ t ]; a < alphas->m_Offsets[ t + 1 ]; a++ )
        {
        channels.push_back( { alphas->m_Classes[ a ], alphas->m_Alphas[ a ] } );
        }
      },
    std::forward< TWrite >( write ) );
}

} // end namespace kvl

#endif
