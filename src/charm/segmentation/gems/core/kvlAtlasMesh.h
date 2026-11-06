#ifndef __kvlAtlasMesh_h
#define __kvlAtlasMesh_h

#include "itkMesh.h"

#include "itkDefaultStaticMeshTraits.h"
#include "itkVector.h"

namespace kvl
{
struct ReferenceTetrahedronInfo
{
  double  m_ReferenceVolumeTimesK;

  // Elements of the inverse of the matrix [ p0 p1 p2 p3; 1 1 1 1 ]
  double  m_Z11;
  double  m_Z21;
  double  m_Z31;
  double  m_Z41;

  double  m_Z12;
  double  m_Z22;
  double  m_Z32;
  double  m_Z42;

  double  m_Z13;
  double  m_Z23;
  double  m_Z33;
  double  m_Z43;
};

using AtlasAlphasType = itk::Array< float >;

struct PointParameters
{
  AtlasAlphasType  m_Alphas;
  bool  m_CanChangeAlphas;
  bool  m_CanMoveX;
  bool  m_CanMoveY;
  bool  m_CanMoveZ;
};

using AtlasMeshTraits = itk::DefaultStaticMeshTraits< PointParameters, 3, 3, double, double, ReferenceTetrahedronInfo >;

using AtlasMesh = itk::Mesh< PointParameters, 3, AtlasMeshTraits >;

using AtlasPositionGradientType = itk::Vector< double, 3 >;
  using AtlasPositionGradientContainerType = itk::VectorContainer< AtlasMesh::PointIdentifier, AtlasPositionGradientType >;
}

#endif
