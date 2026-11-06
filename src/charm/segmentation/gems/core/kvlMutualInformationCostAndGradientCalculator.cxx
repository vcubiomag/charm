#include "kvlMutualInformationCostAndGradientCalculator.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>

#include "itkMath.h"
#include "kvlAtlasMeshDrawer.h"

namespace kvl
{

namespace
{

// The intensities between which 99.9% of the voxels lie
std::pair< double, double > ComputeRobustRange( const float* intensities, std::size_t numberOfVoxels )
{
  const auto  [ lowest, highest ] = std::minmax_element( intensities, intensities + numberOfVoxels );
  constexpr int  numberOfBins = 4000;
  const double  slope = static_cast< double >( numberOfBins - 1 ) / ( static_cast< double >( *highest ) - *lowest );
  const double  offset = *lowest;

  std::vector< std::size_t >  histogram( numberOfBins, 0 );
  std::for_each( intensities, intensities + numberOfVoxels,
                 [ & ]( float intensity ) { histogram[ itk::Math::Round< int >( slope * ( intensity - offset ) ) ]++; } );

  int  robustMinBin = numberOfBins - 1;
  int  robustMaxBin = numberOfBins - 1;
  std::size_t  cumulative = 0;
  bool  foundMin = false;
  bool  foundMax = false;
  for ( int bin = 0; bin < numberOfBins && !foundMax; bin++ )
    {
    cumulative += histogram[ bin ];
    const double  fraction = static_cast< double >( cumulative ) / numberOfVoxels;
    if ( !foundMin && fraction > 0.0005 )
      {
      robustMinBin = bin;
      foundMin = true;
      }
    if ( fraction > 0.9995 )
      {
      robustMaxBin = bin;
      foundMax = true;
      }
    }
  return { robustMinBin / slope + offset, robustMaxBin / slope + offset };
}

} // end anonymous namespace

void
MutualInformationCostAndGradientCalculator
::SetImage( const ImageType* image )
{
  const auto  size = image->GetBufferedRegion().GetSize();
  const std::size_t  numberOfVoxels = image->GetBufferedRegion().GetNumberOfPixels();
  const float*  intensities = image->GetBufferPointer();

  const auto  [ robustMin, robustMax ] = ComputeRobustRange( intensities, numberOfVoxels );
  const double  slope = static_cast< double >( m_NumberOfBins - 1 ) / ( robustMax - robustMin );

  std::vector< std::uint32_t >  selected;
  m_Bins.clear();
  for ( std::size_t voxel = 0; voxel < numberOfVoxels; voxel++ )
    {
    if ( intensities[ voxel ] >= robustMin && intensities[ voxel ] <= robustMax )
      {
      selected.push_back( static_cast< std::uint32_t >( voxel ) );
      m_Bins.push_back( static_cast< std::uint8_t >( itk::Math::Round< int >( slope * ( intensities[ voxel ] - robustMin ) ) ) );
      }
    }
  m_Voxels = VoxelSelection( { static_cast< long >( size[ 0 ] ), static_cast< long >( size[ 1 ] ), static_cast< long >( size[ 2 ] ) },
                             selected.data(), selected.size() );
}

void
MutualInformationCostAndGradientCalculator
::Rasterize( const AtlasMesh* mesh )
{
  const std::size_t  numberOfClasses = mesh->GetPointData()->Begin().Value().m_Alphas.Size();
  const std::size_t  numberOfRows = m_Bins.size();
  m_NumberOfClasses = numberOfClasses;

  m_Priors.assign( numberOfRows * numberOfClasses, 0.0f );
  DrawAlphas( mesh, m_Voxels.GetSize(),
    [ & ]( std::size_t classNumber, long voxel, long count, double value, double step )
      {
      m_Voxels.ForEachInRow( voxel, count,
        [ & ]( long i, std::uint32_t row )
          {
          m_Priors[ row * numberOfClasses + classNumber ] = static_cast< float >( value + i * step );
          } );
      } );

  // EM estimate of the class-conditional intensity histograms
  const std::size_t  histogramSize = numberOfClasses * m_NumberOfBins;
  std::vector< double >  histogram( histogramSize );
  m_ConditionalIntensityDistributions.assign( histogramSize, 1.0 / m_NumberOfBins );
  double  minLogLikelihood = std::numeric_limits< double >::max();
  constexpr std::size_t  blockSize = 1 << 14;
  const std::size_t  numberOfBlocks = ( numberOfRows + blockSize - 1 ) / blockSize;
  std::vector< double >  blockHistograms( numberOfBlocks * histogramSize );
  std::vector< double >  blockMinLogLikelihoods( numberOfBlocks );
  for ( int iterationNumber = 0; iterationNumber < 10; iterationNumber++ )
    {
    ParallelForBlocks( numberOfRows, blockSize,
      [ & ]( std::size_t block, std::size_t first, std::size_t last )
        {
        double*  blockHistogram = blockHistograms.data() + block * histogramSize;
        std::fill_n( blockHistogram, histogramSize, 0.0 );
        std::vector< double >  posterior( numberOfClasses );
        double  blockMinLogLikelihood = 0.0;
        for ( std::size_t row = first; row < last; row++ )
          {
          const int  bin = m_Bins[ row ];
          const float*  priors = m_Priors.data() + row * numberOfClasses;
          double  denominator = 1e-15;
          double  weightToDistribute = 1e-15;
          for ( std::size_t classNumber = 0; classNumber < numberOfClasses; classNumber++ )
            {
            posterior[ classNumber ] = m_ConditionalIntensityDistributions[ classNumber * m_NumberOfBins + bin ] * priors[ classNumber ];
            denominator += posterior[ classNumber ];
            weightToDistribute += priors[ classNumber ];
            }
          blockMinLogLikelihood -= weightToDistribute * std::log( denominator );
          const double  scale = weightToDistribute / denominator;
          for ( std::size_t classNumber = 0; classNumber < numberOfClasses; classNumber++ )
            {
            blockHistogram[ classNumber * m_NumberOfBins + bin ] += scale * posterior[ classNumber ];
            }
          }
        blockMinLogLikelihoods[ block ] = blockMinLogLikelihood;
        } );

    std::fill( histogram.begin(), histogram.end(), 0.0 );
    double  currentMinLogLikelihood = 0.0;
    for ( std::size_t block = 0; block < numberOfBlocks; block++ )
      {
      currentMinLogLikelihood += blockMinLogLikelihoods[ block ];
      const double*  blockHistogram = blockHistograms.data() + block * histogramSize;
      std::transform( histogram.begin(), histogram.end(), blockHistogram, histogram.begin(), std::plus<>() );
      }

    m_NumberOfVoxels = 0.0;
    for ( std::size_t classNumber = 0; classNumber < numberOfClasses; classNumber++ )
      {
      const double*  classHistogram = histogram.data() + classNumber * m_NumberOfBins;
      const double  numberOfVoxelsInThisClass = std::accumulate( classHistogram, classHistogram + m_NumberOfBins, 1e-15 );
      std::transform( classHistogram, classHistogram + m_NumberOfBins,
                      m_ConditionalIntensityDistributions.begin() + classNumber * m_NumberOfBins,
                      [ & ]( double count ) { return count / numberOfVoxelsInThisClass; } );
      m_NumberOfVoxels += numberOfVoxelsInThisClass;
      }

    const double  changeInCostPerVoxel = ( minLogLikelihood - currentMinLogLikelihood ) / m_NumberOfVoxels;
    minLogLikelihood = currentMinLogLikelihood;
    if ( changeInCostPerVoxel < 1e-3 )
      {
      break;
      }
    }

  // The rasterization only gives the approximate gradients. The data cost is the mutual information computed below.
  Superclass::Rasterize( mesh );
  if ( m_MinLogLikelihoodTimesPrior == std::numeric_limits< double >::max() )
    {
    return;
    }

  double  negativeMutualInformation = 0.0;
  std::vector< double >  marginalIntensityDistribution( m_NumberOfBins, 0.0 );
  for ( std::size_t classNumber = 0; classNumber < numberOfClasses; classNumber++ )
    {
    double  marginalProbabilityOfClass = 0.0;
    for ( int binNumber = 0; binNumber < m_NumberOfBins; binNumber++ )
      {
      const double  jointProbability = histogram[ classNumber * m_NumberOfBins + binNumber ] / m_NumberOfVoxels + 1e-15;
      negativeMutualInformation -= jointProbability * std::log( jointProbability );
      marginalProbabilityOfClass += jointProbability;
      marginalIntensityDistribution[ binNumber ] += jointProbability;
      }
    negativeMutualInformation += marginalProbabilityOfClass * std::log( marginalProbabilityOfClass );
    }
  for ( const double marginalProbabilityOfIntensity : marginalIntensityDistribution )
    {
    negativeMutualInformation += marginalProbabilityOfIntensity * std::log( marginalProbabilityOfIntensity );
    }

  m_MinLogLikelihoodTimesPrior += negativeMutualInformation;
}

void
MutualInformationCostAndGradientCalculator
::AddDataContribution( std::size_t tetrahedron,
                       const AtlasMesh::PointType& p0,
                       const AtlasMesh::PointType& p1,
                       const AtlasMesh::PointType& p2,
                       const AtlasMesh::PointType& p3,
                       TetrahedronContribution& contribution ) const
{
  this->AddMixtureContribution< false >( tetrahedron, p0, p1, p2, p3, m_Voxels,
                                         [ this ]( std::uint32_t row ) { return m_ConditionalIntensityDistributions.data() + m_Bins[ row ]; },
                                         m_NumberOfBins, 1.0 / m_NumberOfVoxels, contribution );
}

} // end namespace kvl
