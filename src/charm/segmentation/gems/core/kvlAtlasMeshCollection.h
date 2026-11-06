#ifndef __kvlAtlasMeshCollection_h
#define __kvlAtlasMeshCollection_h

#include <cstddef>
#include <cstdint>

#include "itkAffineTransform.h"
#include "kvlAtlasMesh.h"

namespace kvl
{

class AtlasMeshCollection: public itk::Object
{
public :
  using Self = AtlasMeshCollection;
  using Superclass = itk::Object;
  using Pointer = itk::SmartPointer< Self >;
  using ConstPointer = itk::SmartPointer< const Self >;

  itkNewMacro( Self );
  itkOverrideGetNameOfClassMacro(AtlasMeshCollection);

  using PointsContainerType = AtlasMesh::PointsContainer;
  using PointDataContainerType = AtlasMesh::PointDataContainer;
  using CellsContainerType = AtlasMesh::CellsContainer;
  using CellDataContainerType = AtlasMesh::CellDataContainer;
  using TransformType = itk::AffineTransform< double, 3 >;

  CellsContainerType* GetCells()
    { return m_Cells; }

  void SetK( double K )
    { m_K = K; m_ReferenceTetrahedronInfos = nullptr; }
  double GetK() const
    { return m_K; }

  const CellDataContainerType* GetReferenceTetrahedronInfos() const;

  const PointsContainerType* GetReferencePosition() const
    { return m_ReferencePosition; }

  const PointDataContainerType* GetPointParameters() const
    { return m_PointParameters; }

  AtlasMesh::ConstPointer GetReferenceMesh() const;

  // All arrays are row-major and have a row per node, or per tetrahedron for tetrahedra. canMove is nodes x 3.
  void Assign( const double* positions, std::size_t numberOfPoints, const std::uint32_t* tetrahedra, std::size_t numberOfTetrahedra,
               const float* alphas, std::size_t numberOfLabels, const std::uint8_t* canMove, double K );

  void Transform( const TransformType* transform );

protected :
  AtlasMeshCollection() = default;
  ~AtlasMeshCollection() override;

private :
  void ReleaseCells();

  ITK_DISALLOW_COPY_AND_MOVE(AtlasMeshCollection);

  PointDataContainerType::Pointer  m_PointParameters;
  CellsContainerType::Pointer  m_Cells;
  PointsContainerType::Pointer  m_ReferencePosition;
  double  m_K = 10;

  mutable CellDataContainerType::Pointer  m_ReferenceTetrahedronInfos;
};

} // end namespace kvl

#endif
