#ifndef __kvlGaussianMixture_h
#define __kvlGaussianMixture_h

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "kvlFastMath.h"
#include "kvlParallel.h"

namespace kvl
{

// The class priors of a set of voxels. Atlases are sparse, with only a few classes possible in any voxel.
struct SparsePriors
{
  std::vector< std::uint32_t >  m_Offsets; // Per voxel, with the number of entries appended
  std::vector< std::uint8_t >  m_Classes;
  std::vector< float >  m_Values;

  std::size_t GetNumberOfVoxels() const
  {
    return m_Offsets.size() - 1;
  }
};

// The Gaussians of a mixture model, prepared for evaluation at many voxels. All arrays are row-major.
struct GaussianMixtureModel
{
  std::size_t  m_NumberOfContrasts;
  std::vector< std::size_t >  m_ClassOffsets; // First Gaussian of each class, with the number of Gaussians appended
  const double*  m_Means; // Gaussians x contrasts
  const double*  m_InverseCholeskys; // Gaussians x contrasts x contrasts, lower triangular
  const double*  m_LogScales; // Gaussians
};

// The running sum of a weight, of the weight times the intensities, and of the weight times the products of
// intensities, in the order of the upper triangle of contrast pairs.
inline std::size_t GetNumberOfStatistics( std::size_t numberOfContrasts )
{
  return 1 + numberOfContrasts + numberOfContrasts * ( numberOfContrasts + 1 ) / 2;
}

namespace detail
{

constexpr std::size_t  blockSize = 1 << 12;

// Calls body with the number of contrasts as a compile time constant if it's small enough to have been specialized for,
// which lets the compiler unroll all loops over contrasts. Otherwise that constant is 0.
template< int... TContrasts, typename TBody >
void DispatchOnContrasts( std::size_t numberOfContrasts, std::integer_sequence< int, TContrasts... >, TBody&& body )
{
  const bool  dispatched = ( ( numberOfContrasts == TContrasts ? ( body( std::integral_constant< int, TContrasts >{} ), true ) : false ) || ... );
  if ( !dispatched )
    {
    body( std::integral_constant< int, 0 >{} );
    }
}

template< typename TBody >
void DispatchOnContrasts( std::size_t numberOfContrasts, TBody&& body )
{
  DispatchOnContrasts( numberOfContrasts, std::integer_sequence< int, 1, 2, 3 >{}, std::forward< TBody >( body ) );
}

template< int TContrasts >
inline double LogDensity( const double* intensities, const GaussianMixtureModel& model, std::size_t gaussian, std::size_t numberOfContrasts )
{
  const int  contrasts = TContrasts ? TContrasts : static_cast< int >( numberOfContrasts );
  const double*  mean = model.m_Means + gaussian * contrasts;
  const double*  inverseCholesky = model.m_InverseCholeskys + gaussian * contrasts * contrasts;
  double  squaredDistance = 0.0;
  for ( int row = 0; row < contrasts; row++ )
    {
    double  whitened = 0.0;
    for ( int column = 0; column <= row; column++ )
      {
      whitened += inverseCholesky[ row * contrasts + column ] * ( intensities[ column ] - mean[ column ] );
      }
    squaredDistance += whitened * whitened;
    }
  return model.m_LogScales[ gaussian ] - 0.5 * squaredDistance;
}

// Adds a voxel to the running sum, with statistics pointing at the group ( Gaussian or class ) it's added to
template< int TContrasts >
inline void AccumulateStatistics( double* statistics, double weight, const double* intensities, std::size_t numberOfContrasts )
{
  const int  contrasts = TContrasts ? TContrasts : static_cast< int >( numberOfContrasts );
  int  position = 0;
  statistics[ position++ ] += weight;
  for ( int contrast = 0; contrast < contrasts; contrast++ )
    {
    statistics[ position++ ] += weight * intensities[ contrast ];
    }
  for ( int first = 0; first < contrasts; first++ )
    {
    for ( int second = first; second < contrasts; second++ )
      {
      statistics[ position++ ] += weight * intensities[ first ] * intensities[ second ];
      }
    }
}

// Runs body( block, first, last, blockStatistics ) in parallel with a statistics array per block, summed in block order
// so that the rounding doesn't depend on the number of threads
template< typename TBody >
std::vector< double > SumStatisticsOverBlocks( std::size_t numberOfVoxels, std::size_t numberOfValues, TBody&& body )
{
  const std::size_t  numberOfBlocks = ( numberOfVoxels + blockSize - 1 ) / blockSize;
  std::vector< double >  blockStatistics( numberOfBlocks * numberOfValues, 0.0 );
  ParallelForBlocks( numberOfVoxels, blockSize,
    [ & ]( std::size_t block, std::size_t first, std::size_t last )
      {
      body( block, first, last, blockStatistics.data() + block * numberOfValues );
      } );

  std::vector< double >  statistics( numberOfValues, 0.0 );
  for ( std::size_t block = 0; block < numberOfBlocks; block++ )
    {
    for ( std::size_t value = 0; value < numberOfValues; value++ )
      {
      statistics[ value ] += blockStatistics[ block * numberOfValues + value ];
      }
    }
  return statistics;
}

} // end namespace detail

// E-step. Computes the posterior of each Gaussian at each voxel, and returns the minus log-likelihood along with the
// sufficient statistics of each Gaussian ( Gaussians x statistics ). If posteriors isn't null it's filled with the
// posteriors, one for each Gaussian of each entry of the priors. A class with a zero prior in a voxel has a zero
// posterior there, so none of its Gaussians need to be evaluated: that's most of them.
inline std::pair< double, std::vector< double > >
ComputePosteriors( const double* data, const SparsePriors& priors, const GaussianMixtureModel& model, const std::uint32_t* posteriorOffsets, float* posteriors )
{
  const std::size_t  numberOfVoxels = priors.GetNumberOfVoxels();
  const std::size_t  numberOfContrasts = model.m_NumberOfContrasts;
  const std::size_t  numberOfGaussians = model.m_ClassOffsets.back();
  const std::size_t  numberOfStatistics = GetNumberOfStatistics( numberOfContrasts );
  const std::size_t  numberOfValues = numberOfGaussians * numberOfStatistics + 1;

  std::vector< double >  statistics;
  detail::DispatchOnContrasts( numberOfContrasts,
    [ & ]( auto contrastsConstant )
      {
      constexpr int  TContrasts = decltype( contrastsConstant )::value;
      statistics = detail::SumStatisticsOverBlocks( numberOfVoxels, numberOfValues,
        [ & ]( std::size_t, std::size_t first, std::size_t last, double* blockStatistics )
          {
          // Exponentials go through one loop for the whole block for them to be vectorized, which needs the Gaussians of
          // a voxel's classes laid out first
          const std::size_t  numberOfVoxelsInBlock = last - first;
          std::size_t  numberOfDensities = 0;
          for ( std::uint32_t entry = priors.m_Offsets[ first ]; entry < priors.m_Offsets[ last ]; entry++ )
            {
            const std::size_t  classNumber = priors.m_Classes[ entry ];
            numberOfDensities += model.m_ClassOffsets[ classNumber + 1 ] - model.m_ClassOffsets[ classNumber ];
            }
          std::vector< double >  densities( numberOfDensities );
          std::vector< std::uint16_t >  gaussians( numberOfDensities );
          std::vector< std::uint32_t >  voxelEnds( numberOfVoxelsInBlock );

          std::size_t  position = 0;
          for ( std::size_t voxel = first; voxel < last; voxel++ )
            {
            const double*  intensities = data + voxel * numberOfContrasts;
            for ( std::uint32_t entry = priors.m_Offsets[ voxel ]; entry < priors.m_Offsets[ voxel + 1 ]; entry++ )
              {
              const std::size_t  classNumber = priors.m_Classes[ entry ];
              for ( std::size_t gaussian = model.m_ClassOffsets[ classNumber ]; gaussian < model.m_ClassOffsets[ classNumber + 1 ]; gaussian++ )
                {
                densities[ position ] = detail::LogDensity< TContrasts >( intensities, model, gaussian, numberOfContrasts );
                gaussians[ position++ ] = static_cast< std::uint16_t >( gaussian );
                }
              }
            voxelEnds[ voxel - first ] = static_cast< std::uint32_t >( position );
            }
          for ( std::size_t i = 0; i < numberOfDensities; i++ )
            {
            densities[ i ] = FastExp( densities[ i ] );
            }

          position = 0;
          double  blockLogLikelihood = 0.0;
          for ( std::size_t voxel = first; voxel < last; voxel++ )
            {
            const double*  intensities = data + voxel * numberOfContrasts;
            const std::size_t  firstOfVoxel = position;
            double  normalizer = std::numeric_limits< double >::epsilon();
            for ( std::uint32_t entry = priors.m_Offsets[ voxel ]; entry < priors.m_Offsets[ voxel + 1 ]; entry++ )
              {
              const std::size_t  classNumber = priors.m_Classes[ entry ];
              const double  prior = priors.m_Values[ entry ];
              const std::size_t  numberOfComponents = model.m_ClassOffsets[ classNumber + 1 ] - model.m_ClassOffsets[ classNumber ];
              for ( std::size_t component = 0; component < numberOfComponents; component++ )
                {
                densities[ position ] *= prior;
                normalizer += densities[ position++ ];
                }
              }
            blockLogLikelihood += FastLog( normalizer );

            const double  inverseNormalizer = 1.0 / normalizer;
            for ( std::size_t i = firstOfVoxel; i < position; i++ )
              {
              const double  posterior = densities[ i ] * inverseNormalizer;
              detail::AccumulateStatistics< TContrasts >( blockStatistics + gaussians[ i ] * numberOfStatistics, posterior, intensities, numberOfContrasts );
              if ( posteriors )
                {
                posteriors[ posteriorOffsets[ voxel ] + i - firstOfVoxel ] = static_cast< float >( posterior );
                }
              }
            }
          blockStatistics[ numberOfValues - 1 ] = blockLogLikelihood;
          } );
      } );

  const double  minLogLikelihood = -statistics.back();
  statistics.pop_back();
  return { minLogLikelihood, std::move( statistics ) };
}

// The sufficient statistics of each class ( classes x statistics ), with the priors as the weights
inline std::vector< double > ComputeClassStatistics( const double* data, const SparsePriors& priors, std::size_t numberOfContrasts, std::size_t numberOfClasses )
{
  const std::size_t  numberOfStatistics = GetNumberOfStatistics( numberOfContrasts );
  std::vector< double >  statistics;
  detail::DispatchOnContrasts( numberOfContrasts,
    [ & ]( auto contrastsConstant )
      {
      constexpr int  TContrasts = decltype( contrastsConstant )::value;
      statistics = detail::SumStatisticsOverBlocks( priors.GetNumberOfVoxels(), numberOfClasses * numberOfStatistics,
        [ & ]( std::size_t, std::size_t first, std::size_t last, double* blockStatistics )
          {
          for ( std::size_t voxel = first; voxel < last; voxel++ )
            {
            for ( std::uint32_t entry = priors.m_Offsets[ voxel ]; entry < priors.m_Offsets[ voxel + 1 ]; entry++ )
              {
              detail::AccumulateStatistics< TContrasts >( blockStatistics + priors.m_Classes[ entry ] * numberOfStatistics,
                                                          priors.m_Values[ entry ], data + voxel * numberOfContrasts, numberOfContrasts );
              }
            }
          } );
      } );
  return statistics;
}

// Evaluates P( data | class ) for all classes at all voxels ( voxels x classes )
inline void ComputeClassLikelihoods( const double* data, std::size_t numberOfVoxels, const GaussianMixtureModel& model, float* likelihoods )
{
  const std::size_t  numberOfContrasts = model.m_NumberOfContrasts;
  const std::size_t  numberOfClasses = model.m_ClassOffsets.size() - 1;
  detail::DispatchOnContrasts( numberOfContrasts,
    [ & ]( auto contrastsConstant )
      {
      constexpr int  TContrasts = decltype( contrastsConstant )::value;
      ParallelForBlocks( numberOfVoxels, detail::blockSize,
        [ & ]( std::size_t, std::size_t first, std::size_t last )
          {
          const std::size_t  numberOfGaussians = model.m_ClassOffsets.back();
          std::vector< double >  densities( ( last - first ) * numberOfGaussians );
          std::size_t  position = 0;
          for ( std::size_t voxel = first; voxel < last; voxel++ )
            {
            const double*  intensities = data + voxel * numberOfContrasts;
            for ( std::size_t gaussian = 0; gaussian < numberOfGaussians; gaussian++ )
              {
              densities[ position++ ] = detail::LogDensity< TContrasts >( intensities, model, gaussian, numberOfContrasts );
              }
            }
          for ( std::size_t i = 0; i < densities.size(); i++ )
            {
            densities[ i ] = FastExp( densities[ i ] );
            }

          position = 0;
          for ( std::size_t voxel = first; voxel < last; voxel++ )
            {
            for ( std::size_t classNumber = 0; classNumber < numberOfClasses; classNumber++ )
              {
              double  likelihood = 0.0;
              for ( std::size_t gaussian = model.m_ClassOffsets[ classNumber ]; gaussian < model.m_ClassOffsets[ classNumber + 1 ]; gaussian++ )
                {
                likelihood += densities[ position++ ];
                }
              likelihoods[ voxel * numberOfClasses + classNumber ] = static_cast< float >( likelihood );
              }
            }
          } );
      } );
}

} // end namespace kvl

#endif
