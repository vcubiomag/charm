#ifndef __kvlVoxelSelection_h
#define __kvlVoxelSelection_h

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "kvlTetrahedronRasterizer.h"

namespace kvl
{

/**
 * A subset of the voxels of an image, numbering the selected ones in F order. This is what the flat per-voxel arrays are
 * indexed by, and looking up a voxel in rasterization loops is one of their main costs. It's one bit per voxel with a
 * running count for each 64 of them, which unlike an image of indices fits in the caches: 32 times less memory.
 */
class VoxelSelection
{
public:
  VoxelSelection() = default;

  // voxels are the F-order offsets of the selected voxels, in ascending order
  template< typename TOffset >
  VoxelSelection( const ImageSize& size, const TOffset* voxels, std::size_t numberOfVoxels )
    : m_Size( size ), m_NumberOfVoxels( numberOfVoxels ),
      m_Bits( ( static_cast< std::size_t >( size[ 0 ] ) * size[ 1 ] * size[ 2 ] + 63 ) / 64, 0 )
  {
    for ( std::size_t i = 0; i < numberOfVoxels; i++ )
      {
      const std::size_t  voxel = static_cast< std::size_t >( voxels[ i ] );
      m_Bits[ voxel / 64 ] |= std::uint64_t( 1 ) << ( voxel % 64 );
      }
    m_Ranks.resize( m_Bits.size() );
    std::uint32_t  rank = 0;
    for ( std::size_t word = 0; word < m_Bits.size(); word++ )
      {
      m_Ranks[ word ] = rank;
      rank += PopulationCount( m_Bits[ word ] );
      }
  }

  const ImageSize& GetSize() const
  {
    return m_Size;
  }

  std::size_t GetNumberOfVoxels() const
  {
    return m_NumberOfVoxels;
  }

  // Keeps only the selected ones of the given voxels, in place, and writes their row numbers. Returns how many there are.
  std::size_t Select( VoxelList& voxels, std::size_t count, std::uint32_t* rows ) const
  {
    std::int32_t* const  xs = voxels.m_X.data();
    std::int32_t* const  ys = voxels.m_Y.data();
    std::int32_t* const  zs = voxels.m_Z.data();
    std::size_t  kept = 0;
    for ( std::size_t i = 0; i < count; i++ )
      {
      const std::int32_t  x = xs[ i ];
      const std::int32_t  y = ys[ i ];
      const std::int32_t  z = zs[ i ];
      const std::size_t  offset = x + m_Size[ 0 ] * ( y + m_Size[ 1 ] * static_cast< std::size_t >( z ) );
      const std::size_t  word = offset / 64;
      const int  bit = static_cast< int >( offset % 64 );
      xs[ kept ] = x;
      ys[ kept ] = y;
      zs[ kept ] = z;
      rows[ kept ] = m_Ranks[ word ] + PopulationCount( m_Bits[ word ] & ( ( std::uint64_t( 1 ) << bit ) - 1 ) );
      kept += ( m_Bits[ word ] >> bit ) & 1;
      }
    return kept;
  }

  // The row number of the voxel at an F-order offset if it's selected, or -1 otherwise
  std::int64_t GetRow( std::size_t offset ) const
  {
    const std::size_t  word = offset / 64;
    const int  bit = static_cast< int >( offset % 64 );
    if ( !( ( m_Bits[ word ] >> bit ) & 1 ) )
      {
      return -1;
      }
    return m_Ranks[ word ] + PopulationCount( m_Bits[ word ] & ( ( std::uint64_t( 1 ) << bit ) - 1 ) );
  }

  // Calls visit( i, row ) for the selected voxels among the count voxels from the F-order offset voxel on, with i the
  // position in that range and row the number of the voxel in the selection
  template< typename TVisit >
  void ForEachInRow( long voxel, long count, TVisit&& visit ) const
  {
    std::size_t  word = static_cast< std::size_t >( voxel ) / 64;
    const int  bit = static_cast< int >( voxel % 64 );
    std::uint64_t  bits = m_Bits[ word ] >> bit;
    std::uint32_t  row = m_Ranks[ word ] + PopulationCount( m_Bits[ word ] & ( ( std::uint64_t( 1 ) << bit ) - 1 ) );
    int  remainingInWord = 64 - bit;
    for ( long i = 0; i < count; i++ )
      {
      if ( bits & 1 )
        {
        visit( i, row++ );
        }
      bits >>= 1;
      if ( --remainingInWord == 0 && i + 1 < count )
        {
        bits = m_Bits[ ++word ];
        remainingInWord = 64;
        }
      }
  }

private:
  static std::uint32_t PopulationCount( std::uint64_t bits )
  {
#if defined( __GNUC__ ) || defined( __clang__ )
    return static_cast< std::uint32_t >( __builtin_popcountll( bits ) );
#else
    std::uint32_t  count = 0;
    for ( ; bits; bits &= bits - 1 )
      {
      count++;
      }
    return count;
#endif
  }

  ImageSize  m_Size{};
  std::size_t  m_NumberOfVoxels = 0;
  std::vector< std::uint64_t >  m_Bits;
  std::vector< std::uint32_t >  m_Ranks;
};

} // end namespace kvl

#endif
