#include "kvlTetrahedronTable.h"

#include <algorithm>
#include <mutex>
#include <numeric>
#include <unordered_map>

#include "itkCommand.h"

namespace kvl
{

namespace
{

// Entries are dropped when their key object is deleted, through ITK's delete event, so the cache never extends the
// lifetime of a mesh. A stale entry for a reused address can't be returned either since modification times only grow.
template< typename TValue >
class ObjectCache
{
public:
  template< typename TBuild, typename TIsValid >
  std::shared_ptr< const TValue > Get( const itk::Object* key, itk::ModifiedTimeType stamp, TBuild&& build, TIsValid&& isValid )
  {
    const std::lock_guard< std::mutex >  lock( m_Mutex );
    const auto  [ entry, isNew ] = m_Entries.try_emplace( key );
    if ( isNew )
      {
      key->AddObserver( itk::DeleteEvent(), [ this, key ]( const itk::EventObject& ) { this->Forget( key ); } );
      }
    if ( !entry->second.m_Value || entry->second.m_Stamp != stamp || !isValid( *entry->second.m_Value ) )
      {
      entry->second.m_Value = build();
      entry->second.m_Stamp = stamp;
      }
    return entry->second.m_Value;
  }

private:
  void Forget( const itk::Object* key )
  {
    const std::lock_guard< std::mutex >  lock( m_Mutex );
    m_Entries.erase( key );
  }

  struct Entry
  {
    std::shared_ptr< const TValue >  m_Value;
    itk::ModifiedTimeType  m_Stamp = 0;
  };

  std::mutex  m_Mutex;
  std::unordered_map< const itk::Object*, Entry >  m_Entries;
};

// Leaked on purpose: ITK objects may be released after static destructors have run, and would notify a dead cache.
template< typename TValue >
ObjectCache< TValue >& GetCache()
{
  static auto* const  cache = new ObjectCache< TValue >;
  return *cache;
}

std::uint32_t SpreadBits( std::uint32_t value )
{
  value = ( value | ( value << 16 ) ) & 0x030000ff;
  value = ( value | ( value << 8 ) ) & 0x0300f00f;
  value = ( value | ( value << 4 ) ) & 0x030c30c3;
  value = ( value | ( value << 2 ) ) & 0x09249249;
  return value;
}

void BuildMortonOrder( TetrahedronTable& table, std::vector< ReferenceTetrahedron >& references, const AtlasMesh* mesh )
{
  const auto&  points = mesh->GetPoints()->CastToSTLConstContainer();
  std::array< double, 3 >  lowest{};
  std::array< double, 3 >  scale{};
  for ( int axis = 0; axis < 3; axis++ )
    {
    const auto  [ low, high ] = std::minmax_element( points.begin(), points.end(),
                                                     [ axis ]( const auto& a, const auto& b ) { return a[ axis ] < b[ axis ]; } );
    lowest[ axis ] = ( *low )[ axis ];
    scale[ axis ] = 1023.0 / std::max( ( *high )[ axis ] - lowest[ axis ], 1e-9 );
    }

  std::vector< std::pair< std::uint32_t, std::uint32_t > >  keys( table.Size() );
  for ( std::uint32_t t = 0; t < keys.size(); t++ )
    {
    std::uint32_t  key = 0;
    for ( int axis = 0; axis < 3; axis++ )
      {
      double  center = 0.0;
      for ( const std::uint32_t id : table.m_Vertices[ t ] )
        {
        center += points[ id ][ axis ];
        }
      key |= SpreadBits( static_cast< std::uint32_t >( ( center / 4 - lowest[ axis ] ) * scale[ axis ] ) ) << axis;
      }
    keys[ t ] = { key, t };
    }
  std::sort( keys.begin(), keys.end() );

  std::vector< std::array< std::uint32_t, 4 > >  vertices( keys.size() );
  table.m_References.resize( references.empty() ? 0 : keys.size() );
  for ( std::size_t t = 0; t < keys.size(); t++ )
    {
    vertices[ t ] = table.m_Vertices[ keys[ t ].second ];
    if ( !references.empty() )
      {
      table.m_References[ t ] = references[ keys[ t ].second ];
      }
    }
  table.m_Vertices = std::move( vertices );
}

std::shared_ptr< const TetrahedronTable > BuildTable( const AtlasMesh* mesh )
{
  auto  table = std::make_shared< TetrahedronTable >();
  const auto*  cellData = mesh->GetCellData();
  std::vector< ReferenceTetrahedron >  references;
  for ( auto cellIt = mesh->GetCells()->Begin(); cellIt != mesh->GetCells()->End(); ++cellIt )
    {
    const AtlasMesh::CellType*  cell = cellIt.Value();
    if ( cell->GetType() != itk::CellGeometryEnum::TETRAHEDRON_CELL )
      {
      continue;
      }

    // The vertices go in ascending order, and the rows of Z follow since each belongs to one vertex
    std::array< std::uint32_t, 4 >  ids;
    std::copy_n( cell->PointIdsBegin(), 4, ids.begin() );
    std::array< int, 4 >  order{ 0, 1, 2, 3 };
    std::sort( order.begin(), order.end(), [ & ]( int a, int b ) { return ids[ a ] < ids[ b ]; } );
    std::array< std::uint32_t, 4 >  sorted;
    std::transform( order.begin(), order.end(), sorted.begin(), [ & ]( int vertex ) { return ids[ vertex ]; } );
    table->m_Vertices.push_back( sorted );

    if ( cellData && cellData->Size() > cellIt.Index() )
      {
      const ReferenceTetrahedronInfo&  info = cellData->ElementAt( cellIt.Index() );
      const double  z[ 4 ][ 3 ] = { { info.m_Z11, info.m_Z12, info.m_Z13 },
                                    { info.m_Z21, info.m_Z22, info.m_Z23 },
                                    { info.m_Z31, info.m_Z32, info.m_Z33 },
                                    { info.m_Z41, info.m_Z42, info.m_Z43 } };
      ReferenceTetrahedron  reference;
      reference.m_VolumeTimesK = info.m_ReferenceVolumeTimesK;
      for ( int vertex = 0; vertex < 4; vertex++ )
        {
        std::copy_n( z[ order[ vertex ] ], 3, reference.m_Z[ vertex ] );
        }
      references.push_back( reference );
      }
    }

  BuildMortonOrder( *table, references, mesh );

  const std::size_t  numberOfNodes = mesh->GetPoints()->Size();
  table->m_NodeOffsets.assign( numberOfNodes + 1, 0 );
  for ( const auto& vertices : table->m_Vertices )
    {
    for ( const std::uint32_t id : vertices )
      {
      table->m_NodeOffsets[ id + 1 ]++;
      }
    }
  std::partial_sum( table->m_NodeOffsets.begin(), table->m_NodeOffsets.end(), table->m_NodeOffsets.begin() );
  table->m_NodeVertices.resize( table->m_NodeOffsets.back() );
  std::vector< std::uint32_t >  next( table->m_NodeOffsets.begin(), table->m_NodeOffsets.end() - 1 );
  for ( std::uint32_t t = 0; t < table->Size(); t++ )
    {
    for ( std::uint32_t vertex = 0; vertex < 4; vertex++ )
      {
      table->m_NodeVertices[ next[ table->m_Vertices[ t ][ vertex ] ]++ ] = t * 4 + vertex;
      }
    }
  return table;
}

std::shared_ptr< const TetrahedronAlphas > BuildAlphas( const AtlasMesh* mesh, std::shared_ptr< const TetrahedronTable > table )
{
  auto  result = std::make_shared< TetrahedronAlphas >();
  const auto&  parameters = mesh->GetPointData()->CastToSTLConstContainer();
  const std::size_t  numberOfClasses = parameters.front().m_Alphas.Size();
  result->m_NumberOfClasses = numberOfClasses;
  result->m_Table = table;

  // Merging the sparse classes of the vertices is much cheaper than testing every class of every tetrahedron
  std::vector< std::uint32_t >  nodeOffsets( parameters.size() + 1, 0 );
  std::vector< std::uint16_t >  nodeClasses;
  for ( std::size_t node = 0; node < parameters.size(); node++ )
    {
    const auto&  alphas = parameters[ node ].m_Alphas;
    for ( std::size_t classNumber = 0; classNumber < numberOfClasses; classNumber++ )
      {
      if ( alphas[ classNumber ] != 0 )
        {
        nodeClasses.push_back( static_cast< std::uint16_t >( classNumber ) );
        }
      }
    nodeOffsets[ node + 1 ] = static_cast< std::uint32_t >( nodeClasses.size() );
    }

  result->m_Offsets.reserve( table->Size() + 1 );
  result->m_Offsets.push_back( 0 );
  std::vector< std::uint16_t >  merged;
  for ( const auto& vertices : table->m_Vertices )
    {
    merged.clear();
    for ( const std::uint32_t id : vertices )
      {
      merged.insert( merged.end(), nodeClasses.begin() + nodeOffsets[ id ], nodeClasses.begin() + nodeOffsets[ id + 1 ] );
      }
    std::sort( merged.begin(), merged.end() );
    merged.erase( std::unique( merged.begin(), merged.end() ), merged.end() );
    for ( const std::uint16_t classNumber : merged )
      {
      result->m_Classes.push_back( classNumber );
      result->m_Alphas.push_back( { parameters[ vertices[ 0 ] ].m_Alphas[ classNumber ],
                                    parameters[ vertices[ 1 ] ].m_Alphas[ classNumber ],
                                    parameters[ vertices[ 2 ] ].m_Alphas[ classNumber ],
                                    parameters[ vertices[ 3 ] ].m_Alphas[ classNumber ] } );
      }
    result->m_Offsets.push_back( static_cast< std::uint32_t >( result->m_Classes.size() ) );
    }
  return result;
}

} // end anonymous namespace

std::shared_ptr< const TetrahedronTable >
TetrahedronTable
::Get( const AtlasMesh* mesh )
{
  const auto*  cells = mesh->GetCells();
  const auto*  cellData = mesh->GetCellData();
  const itk::Object*  key = cellData ? static_cast< const itk::Object* >( cellData ) : cells;
  const itk::ModifiedTimeType  stamp = std::max( cells->GetMTime(), key->GetMTime() );
  return GetCache< TetrahedronTable >().Get( key, stamp,
                                             [ & ]() { return BuildTable( mesh ); },
                                             []( const TetrahedronTable& ) { return true; } );
}

std::shared_ptr< const TetrahedronAlphas >
TetrahedronAlphas
::Get( const AtlasMesh* mesh )
{
  auto  table = TetrahedronTable::Get( mesh );
  const auto*  parameters = mesh->GetPointData();
  return GetCache< TetrahedronAlphas >().Get( parameters, parameters->GetMTime(),
                                              [ & ]() { return BuildAlphas( mesh, table ); },
                                              [ & ]( const TetrahedronAlphas& alphas ) { return alphas.m_Table == table; } );
}

} // end namespace kvl
