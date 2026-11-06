#ifndef __kvlTetrahedronRasterizer_h
#define __kvlTetrahedronRasterizer_h

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace kvl
{

using ImageSize = std::array< long, 3 >;

// Voxel coordinates, separately per axis since loops over them vectorize that way. The arrays have room for writes beyond
// the last voxel.
struct VoxelList
{
  std::vector< std::int32_t >  m_X;
  std::vector< std::int32_t >  m_Y;
  std::vector< std::int32_t >  m_Z;

  void Reserve( std::size_t capacity )
  {
    if ( capacity > m_X.size() )
      {
      m_X.resize( 2 * capacity );
      m_Y.resize( 2 * capacity );
      m_Z.resize( 2 * capacity );
      }
  }
};

/**
 * Visits the voxels inside a tetrahedron one row at a time, given its vertices in ascending node id order.
 *
 * With pi0..pi3 the barycentric coordinates of a voxel, pi_i is the signed distance to the face opposite vertex i,
 * scaled so that it is 1 at vertex i. Each coordinate is linear along a row, so the voxels inside the tetrahedron
 * form a span whose ends follow directly from where the four faces cross the row.
 *
 * The ascending vertex order makes this watertight. The two tetrahedra sharing a face compute that face's plane from
 * the same three vertices in the same order, and so get the exact same floating point crossing for every row, on
 * opposite sides. One takes the voxels from the crossing up and the other those below it, with no voxel visited twice
 * or missed regardless of rounding. Besides keeping cost functions consistent, it makes writing the voxels of different
 * tetrahedra to the same image from different threads free of races and so reproducible.
 */
class TetrahedronRasterizer
{
public:
  template< typename TPoint >
  TetrahedronRasterizer( const TPoint& p0, const TPoint& p1, const TPoint& p2, const TPoint& p3, const ImageSize& size )
  {
    // Images are cropped around the head while meshes are not, so many tetrahedra don't overlap with the image at all
    for ( int axis = 0; axis < 3; axis++ )
      {
      const double  lowerOf01 = std::min( p0[ axis ], p1[ axis ] );
      const double  lowerOf23 = std::min( p2[ axis ], p3[ axis ] );
      const double  higherOf01 = std::max( p0[ axis ], p1[ axis ] );
      const double  higherOf23 = std::max( p2[ axis ], p3[ axis ] );
      const double  lowest = std::min( lowerOf01, lowerOf23 );
      const double  highest = std::max( higherOf01, higherOf23 );
      m_Lowest[ axis ] = lowest;
      m_Highest[ axis ] = highest;
      if ( highest < 0 || lowest > static_cast< double >( size[ axis ] - 1 ) )
        {
        m_HasNoVoxels = true;
        return;
        }
      }

    constexpr double  infinity = std::numeric_limits< double >::infinity();
    const std::array< const TPoint*, 4 >  vertices{ &p0, &p1, &p2, &p3 };
    for ( int i = 0; i < 4; i++ )
      {
      m_VerticesYZ[ i ] = { ( *vertices[ i ] )[ 1 ], ( *vertices[ i ] )[ 2 ] };
      }
    constexpr int  faces[ 4 ][ 3 ] = { { 1, 2, 3 }, { 0, 2, 3 }, { 0, 1, 3 }, { 0, 1, 2 } };
    for ( int i = 0; i < 4; i++ )
      {
      const TPoint&  a = *vertices[ faces[ i ][ 0 ] ];
      const TPoint&  b = *vertices[ faces[ i ][ 1 ] ];
      const TPoint&  c = *vertices[ faces[ i ][ 2 ] ];
      const double  u[ 3 ] = { b[ 0 ] - a[ 0 ], b[ 1 ] - a[ 1 ], b[ 2 ] - a[ 2 ] };
      const double  v[ 3 ] = { c[ 0 ] - a[ 0 ], c[ 1 ] - a[ 1 ], c[ 2 ] - a[ 2 ] };
      m_NormalX[ i ] = u[ 1 ] * v[ 2 ] - u[ 2 ] * v[ 1 ];
      m_NormalY[ i ] = u[ 2 ] * v[ 0 ] - u[ 0 ] * v[ 2 ];
      m_NormalZ[ i ] = u[ 0 ] * v[ 1 ] - u[ 1 ] * v[ 0 ];
      m_Offsets[ i ] = m_NormalX[ i ] * a[ 0 ] + m_NormalY[ i ] * a[ 1 ] + m_NormalZ[ i ] * a[ 2 ];
      }

    // The distance of a vertex to the plane of its opposite face is the same determinant for all four vertices, with
    // alternating signs because of the order in which each face takes its vertices
    const double  determinant = m_NormalX[ 1 ] * p1[ 0 ] + m_NormalY[ 1 ] * p1[ 1 ] + m_NormalZ[ 1 ] * p1[ 2 ] - m_Offsets[ 1 ];
    m_HasNoVoxels = determinant == 0;
    const double  inverseDeterminant = 1.0 / determinant;
    for ( int i = 0; i < 4; i++ )
      {
      const bool  flipped = ( i % 2 ) == 0;
      m_InverseHeights[ i ] = flipped ? -inverseDeterminant : inverseDeterminant;
      m_Positive[ i ] = flipped != ( determinant > 0 );

      // Infinite penalties take a face out of the bound it doesn't contribute to, without any branching per row. A
      // face parallel to the row doesn't bound its span at all.
      if ( m_NormalX[ i ] == 0 )
        {
        m_HasFaceParallelToRows = true;
        m_InverseNormalX[ i ] = 0.0;
        m_LowerPenalties[ i ] = -infinity;
        m_UpperPenalties[ i ] = infinity;
        }
      else
        {
        const bool  takesUpperSide = m_Positive[ i ] == ( m_NormalX[ i ] > 0 );
        m_InverseNormalX[ i ] = 1.0 / m_NormalX[ i ];
        m_LowerPenalties[ i ] = takesUpperSide ? 0.0 : -infinity;
        m_UpperPenalties[ i ] = takesUpperSide ? infinity : 0.0;
        }
      }

    // A face parallel to the rows has a crossing of zero, which has no effect as a lower bound
    int  firstUpperBound = 0;
    int  lastLowerBound = 3;
    for ( int i = 0; i < 4; i++ )
      {
      const bool  boundsFromBelow = m_UpperPenalties[ i ] == infinity;
      const int  position = boundsFromBelow ? firstUpperBound++ : lastLowerBound--;
      m_RowFaceNormalY[ position ] = m_NormalY[ i ];
      m_RowFaceNormalZ[ position ] = m_NormalZ[ i ];
      m_RowFaceOffsets[ position ] = m_Offsets[ i ];
      m_RowFaceInverseNormalX[ position ] = m_InverseNormalX[ i ];
      }
    m_NumberOfLowerBounds = firstUpperBound;
  }

  // The change of pi_vertex for a step of one voxel along axis
  double GetBarycentricGradient( int vertex, int axis ) const
  {
    const double  normal = ( axis == 0 ) ? m_NormalX[ vertex ] : ( axis == 1 ) ? m_NormalY[ vertex ] : m_NormalZ[ vertex ];
    return normal * m_InverseHeights[ vertex ];
  }

  // Calls visit( x, y, z, count, pi, piStep ) for every non-empty row, with pi the barycentric coordinates at x
  template< typename TVisit >
  void ForEachRow( const ImageSize& size, TVisit&& visit ) const
  {
    if ( m_HasNoVoxels )
      {
      return;
      }

    const long  firstZ = std::max( 0L, FirstIndex( m_Lowest[ 2 ], size[ 2 ] ) );
    const long  lastZ = std::min( size[ 2 ] - 1, LastIndex( m_Highest[ 2 ], size[ 2 ] ) );
    const double  sizeX = static_cast< double >( size[ 0 ] );

    double  edgeLowerZ[ 6 ];
    double  edgeUpperZ[ 6 ];
    double  edgeSlopes[ 6 ];
    double  edgeOffsets[ 6 ];
    this->ComputeEdges( edgeLowerZ, edgeUpperZ, edgeSlopes, edgeOffsets );

    alignas( 32 ) double  zTerms[ 4 ];
    alignas( 32 ) double  distances[ 4 ];
    alignas( 32 ) double  crossings[ 4 ];
    for ( long z = firstZ; z <= lastZ; z++ )
      {
      for ( int i = 0; i < 4; i++ )
        {
        zTerms[ i ] = m_NormalZ[ i ] * z - m_Offsets[ i ];
        }

      long  firstY;
      long  lastY;
      GetRowsOfSlice( z, edgeLowerZ, edgeUpperZ, edgeSlopes, edgeOffsets, size[ 1 ], firstY, lastY );

      for ( long y = firstY; y <= lastY; y++ )
        {
        for ( int i = 0; i < 4; i++ )
          {
          distances[ i ] = m_NormalY[ i ] * y + zTerms[ i ];
          crossings[ i ] = -distances[ i ] * m_InverseNormalX[ i ];
          }

        double  lower = 0.0;
        double  upper = sizeX;
        for ( int i = 0; i < 4; i++ )
          {
          lower = std::max( lower, crossings[ i ] + m_LowerPenalties[ i ] );
          upper = std::min( upper, crossings[ i ] + m_UpperPenalties[ i ] );
          }
        const double  first = std::ceil( lower );
        const double  end = std::ceil( upper );
        if ( first >= end || ( m_HasFaceParallelToRows && this->IsRowOutsideParallelFaces( static_cast< std::int32_t >( y ), static_cast< std::int32_t >( z ) ) ) )
          {
          continue;
          }

        alignas( 32 ) double  pi[ 4 ];
        alignas( 32 ) double  piStep[ 4 ];
        for ( int i = 0; i < 4; i++ )
          {
          piStep[ i ] = m_NormalX[ i ] * m_InverseHeights[ i ];
          pi[ i ] = ( m_NormalX[ i ] * first + distances[ i ] ) * m_InverseHeights[ i ];
          }
        visit( static_cast< long >( first ), y, z, static_cast< long >( end - first ), pi, piStep );
        }
      }
  }

  // Finds the same voxels as ForEachRow and returns how many there are, for the cost function loops that spend much of
  // their time on this. Tetrahedra are a few voxels across, so that a loop over the rows of a slice or the voxels of a row
  // ends after an unpredictable few steps: a mispredicted branch each time, which is what the time goes to otherwise. So
  // all candidate rows go into a single list, whose spans are computed in one loop that vectorizes across rows. Short runs
  // of rows or voxels are always written in full, with the count advancing by the actual number, which takes away the
  // branches on their lengths and on rows being empty.
  std::size_t CollectVoxels( const ImageSize& size, VoxelList& voxels ) const
  {
    if ( m_HasNoVoxels )
      {
      return 0;
      }

    const long  firstZ = std::max( 0L, FirstIndex( m_Lowest[ 2 ], size[ 2 ] ) );
    const long  lastZ = std::min( size[ 2 ] - 1, LastIndex( m_Highest[ 2 ], size[ 2 ] ) );
    const long  ySize = std::max( 0L, std::min( size[ 1 ] - 1, LastIndex( m_Highest[ 1 ], size[ 1 ] ) ) - std::max( 0L, FirstIndex( m_Lowest[ 1 ], size[ 1 ] ) ) + 1 );

    constexpr int  rowsWrittenAtOnce = 8;
    thread_local RowList  rows;
    rows.Reserve( static_cast< std::size_t >( std::max( 0L, lastZ - firstZ + 1 ) ) * ( ySize + rowsWrittenAtOnce ) );
    std::int32_t* const  rowsY = rows.m_Y.data();
    std::int32_t* const  rowsZ = rows.m_Z.data();

    double  edgeLowerZ[ 6 ];
    double  edgeUpperZ[ 6 ];
    double  edgeSlopes[ 6 ];
    double  edgeOffsets[ 6 ];
    this->ComputeEdges( edgeLowerZ, edgeUpperZ, edgeSlopes, edgeOffsets );

    std::size_t  numberOfRows = 0;
    for ( long z = firstZ; z <= lastZ; z++ )
      {
      long  firstY;
      long  lastY;
      GetRowsOfSlice( z, edgeLowerZ, edgeUpperZ, edgeSlopes, edgeOffsets, size[ 1 ], firstY, lastY );
      const long  length = std::max( 0L, lastY - firstY + 1 );
      std::int32_t*  outputY = rowsY + numberOfRows;
      std::int32_t*  outputZ = rowsZ + numberOfRows;
      const auto  y = static_cast< std::int32_t >( firstY );
      const auto  sliceZ = static_cast< std::int32_t >( z );
      for ( std::int32_t i = 0; i < rowsWrittenAtOnce; i++ )
        {
        outputY[ i ] = y + i;
        outputZ[ i ] = sliceZ;
        }
      for ( std::int32_t i = rowsWrittenAtOnce; i < length; i++ )
        {
        outputY[ i ] = y + i;
        outputZ[ i ] = sliceZ;
        }
      numberOfRows += length;
      }

    const double  sizeX = static_cast< double >( size[ 0 ] );
    std::int32_t* const  starts = rows.m_Starts.data();
    std::int32_t* const  lengths = rows.m_Lengths.data();
    std::size_t  numberOfVoxels = 0;
    switch ( m_NumberOfLowerBounds )
      {
      case 1:
        numberOfVoxels = this->ComputeSpans< 1 >( rowsY, rowsZ, numberOfRows, sizeX, starts, lengths );
        break;
      case 2:
        numberOfVoxels = this->ComputeSpans< 2 >( rowsY, rowsZ, numberOfRows, sizeX, starts, lengths );
        break;
      default:
        numberOfVoxels = this->ComputeSpans< 3 >( rowsY, rowsZ, numberOfRows, sizeX, starts, lengths );
        break;
      }
    if ( m_HasFaceParallelToRows )
      {
      for ( std::size_t r = 0; r < numberOfRows; r++ )
        {
        if ( this->IsRowOutsideParallelFaces( rowsY[ r ], rowsZ[ r ] ) )
          {
          lengths[ r ] = 0;
          }
        }
      }

    constexpr int  voxelsWrittenAtOnce = 4;
    voxels.Reserve( numberOfVoxels + voxelsWrittenAtOnce );
    std::int32_t*  outputX = voxels.m_X.data();
    std::int32_t*  outputY = voxels.m_Y.data();
    std::int32_t*  outputZ = voxels.m_Z.data();
    std::size_t  count = 0;
    for ( std::size_t r = 0; r < numberOfRows; r++ )
      {
      const std::int32_t  x = starts[ r ];
      const std::int32_t  y = rowsY[ r ];
      const std::int32_t  z = rowsZ[ r ];
      const std::int32_t  length = lengths[ r ];
      for ( std::int32_t i = 0; i < voxelsWrittenAtOnce; i++ )
        {
        outputX[ count + i ] = x + i;
        outputY[ count + i ] = y;
        outputZ[ count + i ] = z;
        }
      for ( std::int32_t i = voxelsWrittenAtOnce; i < length; i++ )
        {
        outputX[ count + i ] = x + i;
        outputY[ count + i ] = y;
        outputZ[ count + i ] = z;
        }
      count += length;
      }
    return count;
  }

  void GetBarycentricCoordinates( std::int32_t x, std::int32_t y, std::int32_t z, double* pi ) const
  {
    for ( int i = 0; i < 4; i++ )
      {
      pi[ i ] = ( m_NormalX[ i ] * x + m_NormalY[ i ] * y + m_NormalZ[ i ] * z - m_Offsets[ i ] ) * m_InverseHeights[ i ];
      }
  }

private:
  struct RowList
  {
    std::vector< std::int32_t >  m_Y;
    std::vector< std::int32_t >  m_Z;
    std::vector< std::int32_t >  m_Starts;
    std::vector< std::int32_t >  m_Lengths;

    void Reserve( std::size_t capacity )
    {
      if ( capacity > m_Y.size() )
        {
        m_Y.resize( 2 * capacity );
        m_Z.resize( 2 * capacity );
        m_Starts.resize( 2 * capacity );
        m_Lengths.resize( 2 * capacity );
        }
    }
  };

  // The voxels of each row inside, returning how many there are in total. Spans never start beyond the image, as a sliver
  // can have crossings beyond what an integer can represent.
  template< int TNumberOfLowerBounds >
  std::size_t ComputeSpans( const std::int32_t* rowsY, const std::int32_t* rowsZ, std::size_t numberOfRows, double sizeX,
                            std::int32_t* starts, std::int32_t* lengths ) const
  {
    const double  normalY0 = m_RowFaceNormalY[ 0 ], normalY1 = m_RowFaceNormalY[ 1 ], normalY2 = m_RowFaceNormalY[ 2 ], normalY3 = m_RowFaceNormalY[ 3 ];
    const double  normalZ0 = m_RowFaceNormalZ[ 0 ], normalZ1 = m_RowFaceNormalZ[ 1 ], normalZ2 = m_RowFaceNormalZ[ 2 ], normalZ3 = m_RowFaceNormalZ[ 3 ];
    const double  offset0 = m_RowFaceOffsets[ 0 ], offset1 = m_RowFaceOffsets[ 1 ], offset2 = m_RowFaceOffsets[ 2 ], offset3 = m_RowFaceOffsets[ 3 ];
    const double  inverse0 = m_RowFaceInverseNormalX[ 0 ], inverse1 = m_RowFaceInverseNormalX[ 1 ], inverse2 = m_RowFaceInverseNormalX[ 2 ], inverse3 = m_RowFaceInverseNormalX[ 3 ];
    std::uint32_t  numberOfVoxels = 0;
#pragma omp simd reduction( + : numberOfVoxels )
    for ( std::size_t r = 0; r < numberOfRows; r++ )
      {
      const double  y = static_cast< double >( rowsY[ r ] );
      const double  z = static_cast< double >( rowsZ[ r ] );
      const double  crossing0 = -( normalY0 * y + ( normalZ0 * z - offset0 ) ) * inverse0;
      const double  crossing1 = -( normalY1 * y + ( normalZ1 * z - offset1 ) ) * inverse1;
      const double  crossing2 = -( normalY2 * y + ( normalZ2 * z - offset2 ) ) * inverse2;
      const double  crossing3 = -( normalY3 * y + ( normalZ3 * z - offset3 ) ) * inverse3;
      double  lower = crossing0 > 0.0 ? crossing0 : 0.0;
      double  upper = crossing3 < sizeX ? crossing3 : sizeX;
      if constexpr ( TNumberOfLowerBounds >= 2 )
        {
        lower = crossing1 > lower ? crossing1 : lower;
        }
      else
        {
        upper = crossing1 < upper ? crossing1 : upper;
        }
      if constexpr ( TNumberOfLowerBounds >= 3 )
        {
        lower = crossing2 > lower ? crossing2 : lower;
        }
      else
        {
        upper = crossing2 < upper ? crossing2 : upper;
        }
      // Rounding up by hand since baseline x86-64 has no vector instruction for it. Both are within [ 0, sizeX ] here.
      const double  clampedLower = lower < sizeX ? lower : sizeX;
      const double  clampedUpper = upper > 0.0 ? upper : 0.0;
      const std::int32_t  truncatedLower = static_cast< std::int32_t >( clampedLower );
      const std::int32_t  truncatedUpper = static_cast< std::int32_t >( clampedUpper );
      const std::int32_t  start = truncatedLower + ( static_cast< double >( truncatedLower ) < clampedLower ? 1 : 0 );
      const std::int32_t  end = truncatedUpper + ( static_cast< double >( truncatedUpper ) < clampedUpper ? 1 : 0 );
      const std::int32_t  voxelsInRow = end > start ? end - start : 0;
      starts[ r ] = start;
      lengths[ r ] = voxelsInRow;
      numberOfVoxels += static_cast< std::uint32_t >( voxelsInRow );
      }
    return static_cast< std::size_t >( numberOfVoxels );
  }

  void ComputeEdges( double* lowerZ, double* upperZ, double* slopes, double* offsets ) const
  {
    constexpr int  edges[ 6 ][ 2 ] = { { 0, 1 }, { 0, 2 }, { 0, 3 }, { 1, 2 }, { 1, 3 }, { 2, 3 } };
    for ( int edge = 0; edge < 6; edge++ )
      {
      const auto&  a = m_VerticesYZ[ edges[ edge ][ 0 ] ];
      const auto&  b = m_VerticesYZ[ edges[ edge ][ 1 ] ];
      lowerZ[ edge ] = std::min( a[ 1 ], b[ 1 ] );
      upperZ[ edge ] = std::max( a[ 1 ], b[ 1 ] );
      slopes[ edge ] = ( a[ 1 ] != b[ 1 ] ) ? ( b[ 0 ] - a[ 0 ] ) / ( b[ 1 ] - a[ 1 ] ) : 0.0;
      offsets[ edge ] = a[ 0 ] - slopes[ edge ] * a[ 1 ];
      }
  }

  // The y range of a slice, from where it crosses the edges, leaves out most of the rows of the bounding box that the
  // tetrahedron doesn't reach. It's made slightly larger to leave the decision on voxels near its ends to the watertight
  // test on rows, the only one that the tetrahedra on either side of a face agree on exactly.
  static void GetRowsOfSlice( long z, const double* lowerZ, const double* upperZ, const double* slopes, const double* offsets,
                              long ySize, long& firstY, long& lastY )
  {
    constexpr double  infinity = std::numeric_limits< double >::infinity();
    constexpr double  slack = 1e-6;
    const double  position = static_cast< double >( z );
    double  lowestY = infinity;
    double  highestY = -infinity;
    for ( int edge = 0; edge < 6; edge++ )
      {
      const bool  crossesSlice = ( position >= lowerZ[ edge ] ) & ( position <= upperZ[ edge ] );
      const double  crossing = slopes[ edge ] * position + offsets[ edge ];
      lowestY = std::min( lowestY, crossesSlice ? crossing : infinity );
      highestY = std::max( highestY, crossesSlice ? crossing : -infinity );
      }
    firstY = std::max( 0L, FirstIndex( lowestY - slack, ySize ) );
    lastY = std::min( ySize - 1, LastIndex( highestY + slack, ySize ) );
  }

  // Bounds can be far outside of any image for slivers, beyond what a long can represent
  static long FirstIndex( double position, long size )
  {
    return static_cast< long >( std::ceil( std::clamp( position, -1.0, static_cast< double >( size ) ) ) );
  }

  static long LastIndex( double position, long size )
  {
    return static_cast< long >( std::floor( std::clamp( position, -1.0, static_cast< double >( size ) ) ) );
  }

  bool IsRowOutsideParallelFaces( std::int32_t y, std::int32_t z ) const
  {
    for ( int i = 0; i < 4; i++ )
      {
      if ( m_NormalX[ i ] == 0 )
        {
        // A voxel exactly on such a face goes to the tetrahedron on the side the normal points towards, which is a
        // choice the two tetrahedra sharing this face agree on.
        double  side = m_NormalY[ i ] * y + ( m_NormalZ[ i ] * z - m_Offsets[ i ] );
        if ( side == 0 )
          {
          side = ( m_NormalY[ i ] != 0 ) ? m_NormalY[ i ] : m_NormalZ[ i ];
          }
        if ( ( side > 0 ) != m_Positive[ i ] )
          {
          return true;
          }
        }
      }
    return false;
  }

  alignas( 32 ) double  m_NormalX[ 4 ];
  alignas( 32 ) double  m_NormalY[ 4 ];
  alignas( 32 ) double  m_NormalZ[ 4 ];
  alignas( 32 ) double  m_Offsets[ 4 ];
  alignas( 32 ) double  m_InverseHeights[ 4 ];
  alignas( 32 ) double  m_InverseNormalX[ 4 ];
  alignas( 32 ) double  m_LowerPenalties[ 4 ];
  alignas( 32 ) double  m_UpperPenalties[ 4 ];
  double  m_RowFaceNormalY[ 4 ];
  double  m_RowFaceNormalZ[ 4 ];
  double  m_RowFaceOffsets[ 4 ];
  double  m_RowFaceInverseNormalX[ 4 ];
  double  m_Lowest[ 3 ];
  double  m_Highest[ 3 ];
  std::array< std::array< double, 2 >, 4 >  m_VerticesYZ;
  bool  m_Positive[ 4 ];
  bool  m_HasFaceParallelToRows = false;
  bool  m_HasNoVoxels = false;
  int  m_NumberOfLowerBounds = 1;
};

} // end namespace kvl

#endif
