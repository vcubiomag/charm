#ifndef __kvlTetrahedronTable_h
#define __kvlTetrahedronTable_h

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "kvlAtlasMesh.h"

namespace kvl
{

// Z is inv( [ p0 p1 p2 p3; 1 1 1 1 ] ) of the tetrahedron in reference position, of which the last column is never used
struct ReferenceTetrahedron
{
  double  m_VolumeTimesK;
  double  m_Z[ 4 ][ 3 ];
};

/**
 * The tetrahedra of a mesh as flat arrays, which is what the rasterization loops work on: itk::Mesh stores each cell as
 * a separate heap object behind virtual calls and mixes tetrahedra with the other cell types. Vertices are in ascending
 * node id order, as TetrahedronRasterizer requires, and tetrahedra are in Morton order so that consecutive ones touch
 * nearby memory in images and gradients.
 *
 * Tables are cached for as long as the mesh containers they derive from live and are unmodified, because a mesh is
 * rasterized many times. Code that changes those containers in place must call Modified() on them.
 */
class TetrahedronTable
{
public:
  std::vector< std::array< std::uint32_t, 4 > >  m_Vertices;
  std::vector< ReferenceTetrahedron >  m_References;

  // For each node, the vertices of tetrahedra it is, as tetrahedron * 4 + vertex
  std::vector< std::uint32_t >  m_NodeOffsets;
  std::vector< std::uint32_t >  m_NodeVertices;

  std::size_t Size() const
  {
    return m_Vertices.size();
  }

  static std::shared_ptr< const TetrahedronTable > Get( const AtlasMesh* mesh );
};

/**
 * For each tetrahedron of a table, the classes with a non-zero alpha in any of its vertices, with those alphas per
 * vertex. Atlases are sparse: only a few of the classes are non-zero around any tetrahedron.
 */
class TetrahedronAlphas
{
public:
  std::size_t  m_NumberOfClasses = 0;
  std::shared_ptr< const TetrahedronTable >  m_Table;
  std::vector< std::uint32_t >  m_Offsets;
  std::vector< std::uint16_t >  m_Classes;
  std::vector< std::array< float, 4 > >  m_Alphas;

  static std::shared_ptr< const TetrahedronAlphas > Get( const AtlasMesh* mesh );
};

} // end namespace kvl

#endif
