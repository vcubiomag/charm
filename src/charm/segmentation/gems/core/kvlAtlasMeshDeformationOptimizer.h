#ifndef __kvlAtlasMeshDeformationOptimizer_h
#define __kvlAtlasMeshDeformationOptimizer_h

#include "kvlAtlasMesh.h"
#include "kvlAtlasMeshPositionCostAndGradientCalculator.h"
#include "kvlContainerPool.h"

namespace kvl
{

/**
 * Base class for gradient-based optimization of mesh deformation. Based on "Numerical Optimization" by Nocedal and
 * Wright (Springer, 1999): Chapter 3 for the line search implemented here, and Chapter 9 for the L-BFGS subclass.
 */
class AtlasMeshDeformationOptimizer: public itk::Object
{
public :
  using Self = AtlasMeshDeformationOptimizer;
  using Superclass = itk::Object;
  using Pointer = itk::SmartPointer< Self >;
  using ConstPointer = itk::SmartPointer< const Self >;

  itkOverrideGetNameOfClassMacro(AtlasMeshDeformationOptimizer);

  void SetMesh( AtlasMesh* mesh, bool resetIterations=true, bool resetMemory=true, bool reinitialize=true )
    {
    m_Mesh = mesh;
    m_EvaluationMesh = nullptr;
    this->Reset( resetIterations, resetMemory, reinitialize );
    }

  void SetCostAndGradientCalculator( AtlasMeshPositionCostAndGradientCalculator* calculator,
                                     bool resetIterations=false, bool resetMemory=false, bool reinitialize=false )
    {
    m_Calculator = calculator;
    this->Reset( resetIterations, resetMemory, reinitialize );
    }

  void SetMaximumNumberOfIterations( int maximumNumberOfIterations )
    { m_MaximumNumberOfIterations = maximumNumberOfIterations; }

  void SetMaximalDeformationStopCriterion( double maximalDeformationStopCriterion )
    { m_MaximalDeformationStopCriterion = maximalDeformationStopCriterion; }

  void SetLineSearchMaximalDeformationIntervalStopCriterion( double lineSearchMaximalDeformationIntervalStopCriterion )
    { m_LineSearchMaximalDeformationIntervalStopCriterion = lineSearchMaximalDeformationIntervalStopCriterion; }

  double GetMinLogLikelihoodTimesPrior() const
    { return m_Cost; }

  double Step();

protected:
  using Deformation = AtlasPositionGradientContainerType;

  AtlasMeshDeformationOptimizer() = default;
  ~AtlasMeshDeformationOptimizer() override = default;

  virtual double FindAndOptimizeNewSearchDirection() = 0;

  virtual void WipeMemory()
    {
    }

  void GetCostAndGradient( const AtlasMesh::PointsContainer* position, double& cost, Deformation::Pointer& gradient );

  static double ComputeMaximalDeformation( const Deformation* deformation );

  static double ComputeInnerProduct( const Deformation* deformation1, const Deformation* deformation2 );

  static void AddScaled( Deformation* deformation, double beta, const Deformation* other );

  static void Scale( Deformation* deformation, double beta );

  static Deformation::Pointer Copy( const Deformation* deformation );

  void DoLineSearch( const AtlasMesh::PointsContainer* startPosition,
                     double startCost,
                     const Deformation* startGradient,
                     const Deformation* searchDirection,
                     double startAlpha,
                     double c1,
                     double c2,
                     AtlasMesh::PointsContainer::Pointer& newPosition,
                     double& newCost,
                     Deformation::Pointer& newGradient,
                     double& alphaUsed );

  double  m_Cost = 0;
  AtlasMesh::PointsContainer::Pointer  m_Position;
  Deformation::Pointer  m_Gradient;

private:
  ITK_DISALLOW_COPY_AND_MOVE(AtlasMeshDeformationOptimizer);

  void Reset( bool resetIterations, bool resetMemory, bool reinitialize )
    {
    if ( resetIterations )
      {
      m_IterationNumber = 0;
      }
    if ( resetMemory )
      {
      this->WipeMemory();
      }
    if ( reinitialize )
      {
      m_Initialized = false;
      }
    }

  int  m_IterationNumber = 0;
  int  m_MaximumNumberOfIterations = 1000;
  bool  m_Initialized = false;

  ContainerPool< AtlasMesh::PointsContainer >  m_PositionPool;
  AtlasMesh::Pointer  m_Mesh;
  AtlasMesh::Pointer  m_EvaluationMesh;
  AtlasMeshPositionCostAndGradientCalculator::Pointer  m_Calculator;
  double  m_MaximalDeformationStopCriterion = 0.05;

  // Measured in voxels
  double  m_LineSearchMaximalDeformationLimit = 50.0;
  double  m_LineSearchMaximalDeformationIntervalStopCriterion = 0.05;
};

} // end namespace kvl

#endif
