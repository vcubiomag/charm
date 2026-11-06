#include "kvlAtlasMeshPositionCostAndGradientCalculator.h"

#include <cmath>
#include <limits>

#include "kvlParallel.h"
#include "vnl/vnl_inverse.h"

namespace kvl
{

namespace
{

// Adds Ashburner's prior cost for the tetrahedron deformation from its reference position, and its derivatives in each
// of the vertices. Returns false if the tetrahedron has folded.
bool AddPriorContribution( const AtlasMesh::PointType* const vertices[ 4 ], const ReferenceTetrahedron& reference,
                           double& cost, double gradient[ 4 ][ 3 ] )
{
  // J = Y * Z, with Y = [ p0 p1 p2 p3; 1 1 1 1 ]
  double  m[ 3 ][ 3 ];
  for ( int row = 0; row < 3; row++ )
    {
    for ( int column = 0; column < 3; column++ )
      {
      m[ row ][ column ] = 0.0;
      for ( int vertex = 0; vertex < 4; vertex++ )
        {
        m[ row ][ column ] += reference.m_Z[ vertex ][ column ] * ( *vertices[ vertex ] )[ row ];
        }
      }
    }

  // K = inv( J ) * det( J ), whose elements are also the derivatives of det( J ) with respect to J
  const double  k11 = m[1][1]*m[2][2] - m[1][2]*m[2][1];
  const double  k12 = m[0][2]*m[2][1] - m[0][1]*m[2][2];
  const double  k13 = m[0][1]*m[1][2] - m[1][1]*m[0][2];
  const double  k21 = m[1][2]*m[2][0] - m[1][0]*m[2][2];
  const double  k22 = m[0][0]*m[2][2] - m[0][2]*m[2][0];
  const double  k23 = m[0][2]*m[1][0] - m[0][0]*m[1][2];
  const double  k31 = m[1][0]*m[2][1] - m[2][0]*m[1][1];
  const double  k32 = m[0][1]*m[2][0] - m[0][0]*m[2][1];
  const double  k33 = m[0][0]*m[1][1] - m[0][1]*m[1][0];

  const double  detJ = m[0][0] * k11 + m[0][1] * k21 + m[0][2] * k31;
  if ( detJ <= 0 )
    {
    return false;
    }

  // The affine registration runs with zero stiffness, where all it needs from the prior is the check above
  const double  referenceVolumeTimesK = reference.m_VolumeTimesK;
  if ( referenceVolumeTimesK == 0 )
    {
    return true;
    }

  // Trace of J' * J, which is the sum of the squared singular values of J
  double  sumOfSquaresOfSingularValuesOfJ = 0.0;
  for ( int row = 0; row < 3; row++ )
    {
    for ( int column = 0; column < 3; column++ )
      {
      sumOfSquaresOfSingularValuesOfJ += m[ row ][ column ] * m[ row ][ column ];
      }
    }

  // Trace of inv( J )' * inv( J ), which is the sum of the squared reciprocals of the singular values of J
  const double  traceOfKTransposeTimesK = k11*k11 + k12*k12 + k13*k13 + k21*k21 + k22*k22 + k23*k23 + k31*k31 + k32*k32 + k33*k33;
  const double  sumOfSquaresOfReciprocalsOfSingularValuesOfJ = traceOfKTransposeTimesK / ( detJ * detJ );

  const double  shape = sumOfSquaresOfSingularValuesOfJ + sumOfSquaresOfReciprocalsOfSingularValuesOfJ - 6;
  cost += referenceVolumeTimesK * ( 1 + detJ ) * shape;

  const double  tmp1 = referenceVolumeTimesK * shape;
  const double  tmp2 = 2 * referenceVolumeTimesK * ( 1 + detJ );
  const double  tmp3 = detJ * detJ * detJ;

  const double  ddetJ[ 3 ][ 3 ] = { { k11, k21, k31 }, { k12, k22, k32 }, { k13, k23, k33 } };
  const double  dtrace[ 3 ][ 3 ] =
    { {  k22*m[2][2] - k23*m[1][2] - k32*m[2][1] + k33*m[1][1],
        -k12*m[2][2] + k13*m[1][2] + k32*m[2][0] - k33*m[1][0],
         k12*m[2][1] - k13*m[1][1] - k22*m[2][0] + k23*m[1][0] },
      { -k21*m[2][2] + k23*m[0][2] + k31*m[2][1] - k33*m[0][1],
         k11*m[2][2] - k13*m[0][2] - k31*m[2][0] + k33*m[0][0],
        -k11*m[2][1] + k13*m[0][1] + k21*m[2][0] - k23*m[0][0] },
      {  k21*m[1][2] - k22*m[0][2] - k31*m[1][1] + k32*m[0][1],
        -k11*m[1][2] + k12*m[0][2] + k31*m[1][0] - k32*m[0][0],
         k11*m[1][1] - k12*m[0][1] - k21*m[1][0] + k22*m[0][0] } };

  for ( int row = 0; row < 3; row++ )
    {
    double  dcost[ 3 ];
    for ( int column = 0; column < 3; column++ )
      {
      dcost[ column ] = tmp1 * ddetJ[ row ][ column ]
                        + tmp2 * ( m[ row ][ column ] + ( dtrace[ row ][ column ] * detJ - traceOfKTransposeTimesK * ddetJ[ row ][ column ] ) / tmp3 );
      }
    for ( int vertex = 0; vertex < 4; vertex++ )
      {
      for ( int column = 0; column < 3; column++ )
        {
        gradient[ vertex ][ row ] += dcost[ column ] * reference.m_Z[ vertex ][ column ];
        }
      }
    }

  return true;
}

} // end anonymous namespace

AtlasMeshPositionCostAndGradientCalculator
::AtlasMeshPositionCostAndGradientCalculator()
{
  this->SetMeshToImageTransform( TransformType::New() );
}

void
AtlasMeshPositionCostAndGradientCalculator
::RasterizeCost( const AtlasMesh* mesh )
{
  m_ComputeGradient = false;
  this->Rasterize( mesh );
  m_ComputeGradient = true;
}

void
AtlasMeshPositionCostAndGradientCalculator
::Rasterize( const AtlasMesh* mesh )
{
  const auto  table = TetrahedronTable::Get( mesh );
  m_Alphas = TetrahedronAlphas::Get( mesh );
  const auto&  points = mesh->GetPoints()->CastToSTLConstContainer();

  m_Abort = false;
  m_Contributions.resize( table->Size() );
  ParallelForBlocks( table->Size(), 64,
    [ & ]( std::size_t, std::size_t first, std::size_t last )
      {
      for ( std::size_t t = first; t < last && !m_Abort.load( std::memory_order_relaxed ); t++ )
        {
        const auto&  ids = table->m_Vertices[ t ];
        const AtlasMesh::PointType* const  vertices[ 4 ] = { &points[ ids[ 0 ] ], &points[ ids[ 1 ] ], &points[ ids[ 2 ] ], &points[ ids[ 3 ] ] };
        TetrahedronContribution&  contribution = m_Contributions[ t ];
        contribution = TetrahedronContribution{};
        if ( !AddPriorContribution( vertices, table->m_References[ t ], contribution.m_Cost, contribution.m_Gradient ) )
          {
          m_Abort.store( true, std::memory_order_relaxed );
          return;
          }
        if ( m_ComputeGradient )
          {
          this->AddDataContribution( t, *vertices[ 0 ], *vertices[ 1 ], *vertices[ 2 ], *vertices[ 3 ], contribution );
          }
        }
      } );

  m_PositionGradient = m_GradientPool.Get();
  auto&  gradient = m_PositionGradient->CastToSTLContainer();
  gradient.assign( points.size(), AtlasPositionGradientType( 0.0 ) );

  if ( m_Abort )
    {
    m_MinLogLikelihoodTimesPrior = std::numeric_limits< double >::max();
    return;
    }

  // The sum per node has a fixed order, whichever thread computes it
  constexpr std::size_t  blockSize = 1 << 10;
  ParallelForBlocks( points.size(), blockSize,
    [ & ]( std::size_t, std::size_t first, std::size_t last )
      {
      for ( std::size_t node = first; node < last; node++ )
        {
        double  nodeGradient[ 3 ] = {};
        for ( std::uint32_t entry = table->m_NodeOffsets[ node ]; entry < table->m_NodeOffsets[ node + 1 ]; entry++ )
          {
          const std::uint32_t  tetrahedronAndVertex = table->m_NodeVertices[ entry ];
          const double*  contribution = m_Contributions[ tetrahedronAndVertex / 4 ].m_Gradient[ tetrahedronAndVertex % 4 ];
          for ( int axis = 0; axis < 3; axis++ )
            {
            nodeGradient[ axis ] += contribution[ axis ];
            }
          }
        for ( int axis = 0; axis < 3; axis++ )
          {
          gradient[ node ][ axis ] = nodeGradient[ axis ];
          }
        }
      } );

  double  cost = 0.0;
  for ( const TetrahedronContribution& contribution : m_Contributions )
    {
    cost += contribution.m_Cost;
    }
  if ( !std::isfinite( cost ) )
    {
    m_MinLogLikelihoodTimesPrior = std::numeric_limits< double >::max();
    return;
    }
  m_MinLogLikelihoodTimesPrior = cost;

  this->PostProcessCostAndGradient( mesh );

  if ( !m_ComputeGradient )
    {
    return;
    }
  if ( m_BoundaryCondition == SLIDING )
    {
    this->ImposeSlidingBoundaryConditions( mesh );
    }
  else if ( m_BoundaryCondition == AFFINE )
    {
    this->ImposeAffineBoundaryConditions( mesh );
    }
}

void
AtlasMeshPositionCostAndGradientCalculator
::SetMeshToImageTransform( const TransformType* meshToImageTransform )
{
  // Sliding boundary conditions project the gradient of a node onto the image axes it can move along. The 8 combinations
  // of axes are indexed by canMoveX * 4 + canMoveY * 2 + canMoveZ. The projection is C * inv( C' * C ) * C', with C the
  // columns of the transform's 3x3 matrix for the axes that can move.
  const TransformType::Pointer  identity = TransformType::New();
  const auto&  matrix = ( meshToImageTransform ? meshToImageTransform : identity.GetPointer() )->GetMatrix();

  m_SlidingBoundaryCorrectionMatrices[ 0 ].Fill( 0.0 );
  m_SlidingBoundaryCorrectionMatrices[ 7 ].SetIdentity();
  for ( int index = 1; index < 7; index++ )
    {
    vnl_matrix< double >  C( 3, ( index & 1 ) + ( ( index >> 1 ) & 1 ) + ( ( index >> 2 ) & 1 ) );
    int  column = 0;
    for ( int axis = 0; axis < 3; axis++ )
      {
      if ( index & ( 4 >> axis ) )
        {
        for ( int row = 0; row < 3; row++ )
          {
          C( row, column ) = matrix( row, axis );
          }
        column++;
        }
      }
    const vnl_matrix< double >  correction = C * vnl_inverse( C.transpose() * C ) * C.transpose();
    for ( int row = 0; row < 3; row++ )
      {
      for ( int col = 0; col < 3; col++ )
        {
        m_SlidingBoundaryCorrectionMatrices[ index ]( row, col ) = correction( row, col );
        }
      }
    }
}

void
AtlasMeshPositionCostAndGradientCalculator
::ImposeSlidingBoundaryConditions( const AtlasMesh* mesh )
{
  auto&  gradient = m_PositionGradient->CastToSTLContainer();
  const auto&  parameters = mesh->GetPointData()->CastToSTLConstContainer();
  for ( std::size_t node = 0; node < gradient.size(); node++ )
    {
    const int  index = ( parameters[ node ].m_CanMoveX ? 4 : 0 ) + ( parameters[ node ].m_CanMoveY ? 2 : 0 ) + ( parameters[ node ].m_CanMoveZ ? 1 : 0 );

    // Index 7 is the identity matrix, and also the vast majority of vertices
    if ( index < 7 )
      {
      gradient[ node ] = m_SlidingBoundaryCorrectionMatrices[ index ] * gradient[ node ];
      }
    }
}

void
AtlasMeshPositionCostAndGradientCalculator
::ImposeAffineBoundaryConditions( const AtlasMesh* mesh )
{
  // Project the gradient onto the affine deformations of the mesh. With X the ( points x 4 ) matrix with rows
  // ( 1 x1 x2 x3 ) and G the ( points x 3 ) gradient, that is X * inv( X' * X ) * X' * G. The positions are those of
  // the first call, which assumes that the mesh is still undeformed then. Gram-Schmidt turns X into an orthonormal
  // basis Z of the same subspace, so the projection is Z * Z' * G with no matrix to invert.
  auto&  gradient = m_PositionGradient->CastToSTLContainer();
  const auto&  points = mesh->GetPoints()->CastToSTLConstContainer();
  const std::size_t  numberOfRows = points.size();

  if ( m_AffineProjectionMatrix.empty() )
    {
    m_AffineProjectionMatrix = vnl_matrix< double >( numberOfRows, 4 );
    for ( std::size_t row = 0; row < numberOfRows; row++ )
      {
      m_AffineProjectionMatrix( row, 0 ) = 1.0;
      m_AffineProjectionMatrix( row, 1 ) = points[ row ][ 0 ];
      m_AffineProjectionMatrix( row, 2 ) = points[ row ][ 1 ];
      m_AffineProjectionMatrix( row, 3 ) = points[ row ][ 2 ];
      }

    m_AffineProjectionMatrix.set_column( 0, m_AffineProjectionMatrix.get_column( 0 ).normalize() );
    for ( int columnNumber = 1; columnNumber < 4; columnNumber++ )
      {
      vnl_vector< double >  column = m_AffineProjectionMatrix.get_column( columnNumber );
      for ( int previousColumnNumber = 0; previousColumnNumber < columnNumber; previousColumnNumber++ )
        {
        column -= inner_product( column, m_AffineProjectionMatrix.get_column( previousColumnNumber ) )
                  * m_AffineProjectionMatrix.get_column( previousColumnNumber );
        }
      m_AffineProjectionMatrix.set_column( columnNumber, column.normalize() );
      }
    }

  double  projection[ 4 ][ 3 ] = {};
  for ( std::size_t row = 0; row < numberOfRows; row++ )
    {
    const double*  basis = m_AffineProjectionMatrix[ row ];
    for ( int column = 0; column < 4; column++ )
      {
      for ( int axis = 0; axis < 3; axis++ )
        {
        projection[ column ][ axis ] += basis[ column ] * gradient[ row ][ axis ];
        }
      }
    }
  for ( std::size_t row = 0; row < numberOfRows; row++ )
    {
    const double*  basis = m_AffineProjectionMatrix[ row ];
    for ( int axis = 0; axis < 3; axis++ )
      {
      gradient[ row ][ axis ] = basis[ 0 ] * projection[ 0 ][ axis ] + basis[ 1 ] * projection[ 1 ][ axis ]
                                + basis[ 2 ] * projection[ 2 ][ axis ] + basis[ 3 ] * projection[ 3 ][ axis ];
      }
    }
}

} // end namespace kvl
