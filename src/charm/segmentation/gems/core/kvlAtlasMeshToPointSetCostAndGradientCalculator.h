#ifndef __kvlAtlasMeshToPointSetCostAndGradientCalculator_h
#define __kvlAtlasMeshToPointSetCostAndGradientCalculator_h

#include "kvlAtlasMeshPositionCostAndGradientCalculator.h"

namespace kvl
{

class AtlasMeshToPointSetCostAndGradientCalculator: public AtlasMeshPositionCostAndGradientCalculator
{
public :
  using Self = AtlasMeshToPointSetCostAndGradientCalculator;
  using Superclass = AtlasMeshPositionCostAndGradientCalculator;
  using Pointer = itk::SmartPointer< Self >;
  using ConstPointer = itk::SmartPointer< const Self >;

  itkNewMacro( Self );
  itkOverrideGetNameOfClassMacro(AtlasMeshToPointSetCostAndGradientCalculator);

  void SetTargetPoints( const AtlasMesh::PointsContainer* targetPoints )
    {
    m_TargetPoints = targetPoints;
    }

protected:
  AtlasMeshToPointSetCostAndGradientCalculator() = default;
  ~AtlasMeshToPointSetCostAndGradientCalculator() override = default;

  void PostProcessCostAndGradient( const AtlasMesh* mesh ) override;

private:
  ITK_DISALLOW_COPY_AND_MOVE(AtlasMeshToPointSetCostAndGradientCalculator);

  AtlasMesh::PointsContainer::ConstPointer  m_TargetPoints;
};

} // end namespace kvl

#endif
