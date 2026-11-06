#include "kvlAtlasMeshCollection.h"

#include <algorithm>

#include "itkTetrahedronCell.h"

namespace kvl
{

AtlasMeshCollection
::~AtlasMeshCollection()
{
  this->ReleaseCells();
}

void
AtlasMeshCollection
::ReleaseCells()
{
  // A mesh still holding on to the cells deletes them itself when it goes
  if ( m_Cells && m_Cells->GetReferenceCount() == 1 )
    {
    for ( const auto* cell : m_Cells->CastToSTLConstContainer() )
      {
      delete cell;
      }
    }
  m_Cells = nullptr;
}

AtlasMesh::ConstPointer
AtlasMeshCollection
::GetReferenceMesh() const
{
  AtlasMesh::Pointer  mesh = AtlasMesh::New();
  mesh->SetPoints( m_ReferencePosition );
  mesh->SetCells( m_Cells );
  mesh->SetPointData( m_PointParameters );
  mesh->SetCellData( const_cast< CellDataContainerType* >( this->GetReferenceTetrahedronInfos() ) );
  return mesh.GetPointer();
}

const AtlasMeshCollection::CellDataContainerType*
AtlasMeshCollection
::GetReferenceTetrahedronInfos() const
{
  if ( m_ReferenceTetrahedronInfos )
    {
    return m_ReferenceTetrahedronInfos;
    }

  m_ReferenceTetrahedronInfos = CellDataContainerType::New();
  const auto&  cells = m_Cells->CastToSTLConstContainer();
  auto&  infos = m_ReferenceTetrahedronInfos->CastToSTLContainer();
  infos.resize( cells.size() );
  const auto&  positions = m_ReferencePosition->CastToSTLConstContainer();
  for ( std::size_t cellNumber = 0; cellNumber < cells.size(); cellNumber++ )
    {
    const auto*  pointIds = cells[ cellNumber ]->PointIdsBegin();
    const auto&  p0 = positions[ pointIds[ 0 ] ];
    const auto&  p1 = positions[ pointIds[ 1 ] ];
    const auto&  p2 = positions[ pointIds[ 2 ] ];
    const auto&  p3 = positions[ pointIds[ 3 ] ];

    // Lambda is the Jacobian of the transform from a standardized tetrahedron, which has volume 1/6, to the
    // tetrahedron in reference position. Its inverse has what's needed of inv( [ p0 p1 p2 p3; 1 1 1 1 ] ): the
    // last three rows, with the first row making the columns sum to zero.
    const double  a = p1[ 0 ] - p0[ 0 ], b = p2[ 0 ] - p0[ 0 ], c = p3[ 0 ] - p0[ 0 ];
    const double  d = p1[ 1 ] - p0[ 1 ], e = p2[ 1 ] - p0[ 1 ], f = p3[ 1 ] - p0[ 1 ];
    const double  g = p1[ 2 ] - p0[ 2 ], h = p2[ 2 ] - p0[ 2 ], i = p3[ 2 ] - p0[ 2 ];
    const double  A = e * i - f * h, B = f * g - d * i, C = d * h - e * g;
    const double  determinant = a * A + b * B + c * C;
    const double  inverse = 1.0 / determinant;

    ReferenceTetrahedronInfo&  info = infos[ cellNumber ];
    info.m_ReferenceVolumeTimesK = m_K * determinant / 6;
    info.m_Z21 = A * inverse;
    info.m_Z31 = B * inverse;
    info.m_Z41 = C * inverse;
    info.m_Z22 = ( c * h - b * i ) * inverse;
    info.m_Z32 = ( a * i - c * g ) * inverse;
    info.m_Z42 = ( b * g - a * h ) * inverse;
    info.m_Z23 = ( b * f - c * e ) * inverse;
    info.m_Z33 = ( c * d - a * f ) * inverse;
    info.m_Z43 = ( a * e - b * d ) * inverse;
    info.m_Z11 = -( info.m_Z21 + info.m_Z31 + info.m_Z41 );
    info.m_Z12 = -( info.m_Z22 + info.m_Z32 + info.m_Z42 );
    info.m_Z13 = -( info.m_Z23 + info.m_Z33 + info.m_Z43 );
    }

  return m_ReferenceTetrahedronInfos;
}

void
AtlasMeshCollection
::Assign( const double* positions, std::size_t numberOfPoints, const std::uint32_t* tetrahedra, std::size_t numberOfTetrahedra,
          const float* alphas, std::size_t numberOfLabels, const std::uint8_t* canMove, double K )
{
  m_ReferenceTetrahedronInfos = nullptr;
  m_K = K;

  m_ReferencePosition = PointsContainerType::New();
  auto&  points = m_ReferencePosition->CastToSTLContainer();
  points.resize( numberOfPoints );
  for ( std::size_t node = 0; node < numberOfPoints; node++ )
    {
    for ( int axis = 0; axis < 3; axis++ )
      {
      points[ node ][ axis ] = positions[ node * 3 + axis ];
      }
    }

  using TetrahedronCell = itk::TetrahedronCell< AtlasMesh::CellType >;
  this->ReleaseCells();
  m_Cells = CellsContainerType::New();
  auto&  cells = m_Cells->CastToSTLContainer();
  cells.reserve( numberOfTetrahedra );
  for ( std::size_t tetrahedron = 0; tetrahedron < numberOfTetrahedra; tetrahedron++ )
    {
    auto*  cell = new TetrahedronCell;
    std::copy_n( tetrahedra + tetrahedron * 4, 4, cell->PointIdsBegin() );
    cells.push_back( cell );
    }

  m_PointParameters = PointDataContainerType::New();
  auto&  parameters = m_PointParameters->CastToSTLContainer();
  parameters.resize( numberOfPoints );
  for ( std::size_t node = 0; node < numberOfPoints; node++ )
    {
    PointParameters&  parameter = parameters[ node ];
    parameter.m_Alphas.SetSize( numberOfLabels );
    std::copy_n( alphas + node * numberOfLabels, numberOfLabels, parameter.m_Alphas.data_block() );
    parameter.m_CanChangeAlphas = true;
    parameter.m_CanMoveX = canMove[ node * 3 ];
    parameter.m_CanMoveY = canMove[ node * 3 + 1 ];
    parameter.m_CanMoveZ = canMove[ node * 3 + 2 ];
    }
}

void
AtlasMeshCollection
::Transform( const TransformType* transform )
{
  for ( auto& point : m_ReferencePosition->CastToSTLContainer() )
    {
    point = transform->TransformPoint( point );
    }

  // Cached data members are no longer valid
  m_ReferenceTetrahedronInfos = nullptr;
}

} // end namespace kvl
