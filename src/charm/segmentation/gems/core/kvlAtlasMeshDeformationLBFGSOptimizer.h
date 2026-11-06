#ifndef __kvlAtlasMeshDeformationLBFGSOptimizer_h
#define __kvlAtlasMeshDeformationLBFGSOptimizer_h

#include <vector>

#include "kvlAtlasMeshDeformationOptimizer.h"

namespace kvl
{

class AtlasMeshDeformationLBFGSOptimizer: public AtlasMeshDeformationOptimizer
{
public :
  using Self = AtlasMeshDeformationLBFGSOptimizer;
  using Superclass = AtlasMeshDeformationOptimizer;
  using Pointer = itk::SmartPointer< Self >;
  using ConstPointer = itk::SmartPointer< const Self >;

  itkNewMacro( Self );
  itkOverrideGetNameOfClassMacro(AtlasMeshDeformationLBFGSOptimizer);

  void SetMaximumMemoryLength( int maximumMemoryLength )
    {
    m_MaximumMemoryLength = maximumMemoryLength;
    }

protected:
  AtlasMeshDeformationLBFGSOptimizer() = default;
  ~AtlasMeshDeformationLBFGSOptimizer() override = default;

  void WipeMemory() override;

  double FindAndOptimizeNewSearchDirection() override;

private:
  ITK_DISALLOW_COPY_AND_MOVE(AtlasMeshDeformationLBFGSOptimizer);

  // Computes the search direction -H * gradient, with the first entries of m_Ss and m_Ys holding the most recent history
  Deformation::Pointer ComputeSearchDirection( double gamma ) const;

  Deformation::Pointer  m_OldGradient;
  Deformation::Pointer  m_OldSearchDirection;
  double  m_AlphaUsedLastTime = 0.0;

  // Most recent first, with the inner products between all of them: elements ( i, j ) are s_i' * y_j and y_i' * y_j
  std::vector< Deformation::Pointer >  m_Ss;
  std::vector< Deformation::Pointer >  m_Ys;
  std::vector< std::vector< double > >  m_SYs;
  std::vector< std::vector< double > >  m_YYs;

  double  m_StartDistance = 1.0; // Measured in voxels
  std::size_t  m_MaximumMemoryLength = 12; // Less than 20 recommended
};

} // end namespace kvl

#endif
