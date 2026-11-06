#ifndef __kvlParallel_h
#define __kvlParallel_h

#include <algorithm>
#include <atomic>
#include <cstddef>

#include "itkMultiThreaderBase.h"

namespace kvl
{

// Calls body( index ) for every index in [ 0, count ), with the workers pulling indices from a shared counter, on as many
// threads as ITK is configured to use. ITK itself splits a range statically over its work units, which leaves every core
// waiting for the slowest one on hybrid CPUs. Its default of four work units per thread would balance that too, but
// queues them through a mutex for every dispatch, and the optimizers dispatch thousands of times per second.
template< typename TBody >
void ParallelFor( std::size_t count, TBody&& body )
{
  const std::size_t  numberOfWorkers = std::min< std::size_t >( itk::MultiThreaderBase::GetGlobalDefaultNumberOfThreads(), count );
  if ( numberOfWorkers <= 1 )
    {
    for ( std::size_t index = 0; index < count; ++index )
      {
      body( index );
      }
    return;
    }

  const auto  threader = itk::MultiThreaderBase::New();
  std::atomic< std::size_t >  next{ 0 };
  threader->SetNumberOfWorkUnits( static_cast< itk::ThreadIdType >( numberOfWorkers ) );
  threader->ParallelizeArray(
    0, numberOfWorkers,
    [ & ]( itk::SizeValueType )
      {
      for ( std::size_t index = next.fetch_add( 1, std::memory_order_relaxed ); index < count;
            index = next.fetch_add( 1, std::memory_order_relaxed ) )
        {
        body( index );
        }
      },
    nullptr );
}

// Splits [ 0, count ) into blocks of at most blockSize and calls body( blockNumber, first, last ). The block grid
// depends only on count, so reducing per-block results in block order rounds identically for any number of threads.
template< typename TBody >
std::size_t ParallelForBlocks( std::size_t count, std::size_t blockSize, TBody&& body )
{
  const std::size_t  numberOfBlocks = ( count + blockSize - 1 ) / blockSize;
  ParallelFor( numberOfBlocks,
               [ & ]( std::size_t block )
                 {
                 body( block, block * blockSize, std::min( count, ( block + 1 ) * blockSize ) );
                 } );
  return numberOfBlocks;
}

} // end namespace kvl

#endif
