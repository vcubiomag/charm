#ifndef __kvlFastMath_h
#define __kvlFastMath_h

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace kvl
{

// log( x ) for a positive and normal x, to within about one unit in the last place. This is the algorithm of fdlibm,
// without its branches so that the compiler can vectorize a loop calling it: a libm call in a loop can't be.
inline double FastLog( double x )
{
  std::uint64_t  bits;
  std::memcpy( &bits, &x, sizeof( bits ) );
  std::int64_t  exponent = static_cast< std::int64_t >( bits >> 52 ) - 1023;
  bits = ( bits & 0x000fffffffffffffULL ) | 0x3ff0000000000000ULL;
  double  mantissa;
  std::memcpy( &mantissa, &bits, sizeof( mantissa ) );

  // Bring the mantissa to [ sqrt(1/2), sqrt(2) ], where the series converges fastest
  const std::int64_t  isLarge = mantissa > 1.4142135623730951;
  mantissa *= isLarge ? 0.5 : 1.0;
  exponent += isLarge;

  constexpr double  ln2Hi = 6.93147180369123816490e-01;
  constexpr double  ln2Lo = 1.90821492927058770002e-10;
  constexpr double  lg1 = 6.666666666666735130e-01;
  constexpr double  lg2 = 3.999999999940941908e-01;
  constexpr double  lg3 = 2.857142874366239149e-01;
  constexpr double  lg4 = 2.222219843214978396e-01;
  constexpr double  lg5 = 1.818357216161805012e-01;
  constexpr double  lg6 = 1.531383769920937332e-01;
  constexpr double  lg7 = 1.479819860511658591e-01;

  const double  f = mantissa - 1.0;
  const double  s = f / ( 2.0 + f );
  const double  z = s * s;
  const double  w = z * z;
  const double  r = z * ( lg1 + w * ( lg3 + w * ( lg5 + w * lg7 ) ) ) + w * ( lg2 + w * ( lg4 + w * lg6 ) );
  const double  halfSquare = 0.5 * f * f;
  const double  k = static_cast< double >( exponent );
  return k * ln2Hi - ( ( halfSquare - ( s * ( halfSquare + r ) + k * ln2Lo ) ) - f );
}

// exp( x ) to within about one unit in the last place, for any x with results underflowing to zero and without branches
// for the same reason as FastLog. The polynomial is that of fdlibm's exp, reduced by a multiple of log( 2 ).
inline double FastExp( double x )
{
  constexpr double  inverseLn2 = 1.44269504088896338700e+00;
  constexpr double  ln2Hi = 6.93147180369123816490e-01;
  constexpr double  ln2Lo = 1.90821492927058770002e-10;
  constexpr double  p1 = 1.66666666666666019037e-01;
  constexpr double  p2 = -2.77777777770155933842e-03;
  constexpr double  p3 = 6.61375632143793436117e-05;
  constexpr double  p4 = -1.65339022054652515390e-06;
  constexpr double  p5 = 4.13813679705723846039e-08;

  // Clamping keeps the exponent arithmetic in range. Anything below the range of normal results is set to zero at the end.
  const double  clamped = x < -708.0 ? -708.0 : x;
  const double  rounded = clamped * inverseLn2 + ( clamped < 0 ? -0.5 : 0.5 );
  const std::int64_t  k = static_cast< std::int64_t >( rounded );
  const double  kAsDouble = static_cast< double >( k );
  const double  high = clamped - kAsDouble * ln2Hi;
  const double  low = kAsDouble * ln2Lo;
  const double  reduced = high - low;
  const double  t = reduced * reduced;
  const double  c = reduced - t * ( p1 + t * ( p2 + t * ( p3 + t * ( p4 + t * p5 ) ) ) );
  const double  y = 1.0 - ( ( low - ( reduced * c ) / ( 2.0 - c ) ) - high );

  std::uint64_t  bits;
  std::memcpy( &bits, &y, sizeof( bits ) );
  bits += static_cast< std::uint64_t >( k ) << 52;
  double  result;
  std::memcpy( &result, &bits, sizeof( result ) );
  return x < -708.0 ? 0.0 : result;
}

// log( x ) for a positive and normal x, to within a few units in the last place, again without branches. The polynomial
// is that of Cephes' logf, which unlike fdlibm's doesn't need a division.
inline float FastLog( float x )
{
  std::uint32_t  bits;
  std::memcpy( &bits, &x, sizeof( bits ) );
  std::int32_t  exponent = static_cast< std::int32_t >( bits >> 23 ) - 127;
  bits = ( bits & 0x007fffffU ) | 0x3f800000U;
  float  mantissa;
  std::memcpy( &mantissa, &bits, sizeof( mantissa ) );

  const std::int32_t  isLarge = mantissa > 1.41421356f;
  mantissa *= isLarge ? 0.5f : 1.0f;
  exponent += isLarge;

  const float  f = mantissa - 1.0f;
  const float  z = f * f;
  const float  series = 3.3333331174e-1f + f * ( -2.4999993993e-1f + f * ( 2.0000714765e-1f + f * ( -1.6668057665e-1f + f * ( 1.4249322787e-1f
                        + f * ( -1.2420140846e-1f + f * ( 1.1676998740e-1f + f * ( -1.1514610310e-1f + f * 7.0376836292e-2f ) ) ) ) ) ) );
  const float  k = static_cast< float >( exponent );
  return f + ( f * z * series - 2.12194440e-4f * k - 0.5f * z ) + 0.693359375f * k;
}

// The sum of the logarithms of the values, summing in blocks to not depend on the order of evaluation
inline double SumOfLogarithms( const double* values, std::size_t count )
{
  constexpr std::size_t  blockSize = 32;
  double  sum = 0.0;
  double  logarithms[ blockSize ];
  for ( std::size_t first = 0; first < count; first += blockSize )
    {
    const std::size_t  size = ( count - first < blockSize ) ? count - first : blockSize;
    for ( std::size_t i = 0; i < size; i++ )
      {
      logarithms[ i ] = FastLog( values[ first + i ] );
      }
    for ( std::size_t i = 0; i < size; i++ )
      {
      sum += logarithms[ i ];
      }
    }
  return sum;
}

inline double SumOfLogarithms( const float* values, std::size_t count )
{
  double  sum = 0.0;
#pragma omp simd reduction( + : sum )
  for ( std::size_t i = 0; i < count; i++ )
    {
    sum += FastLog( values[ i ] );
    }
  return sum;
}

} // end namespace kvl

#endif
