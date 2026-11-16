/* Sine, cosine, arcsine and arctangent for the small angles in bend drifts, rounded almost correctly.
 *
 * CUDA's asin and atan are biased for small arguments: asin differs from glibc on 36% of
 * arguments from 0.01 to 0.1, by +0.34 ulp on average, and atan on 12%, by +0.12 ulp. CUDA's sin
 * and cos have no such bias. In a sector drift a difference is multiplied by the bend radius, and
 * a biased one builds up over turns instead of averaging out. These stay within about 0.51 ulp by
 * carrying the leading term in extra precision, so they differ from glibc only where a result is
 * very near a rounding boundary. Outside their range they use CUDA's functions.
 */
#ifndef GPU_ACCURATE_TRIG_H
#define GPU_ACCURATE_TRIG_H

#define GPU_ACCURATE_TRIG_LIMIT 0.4
#define GPU_ACCURATE_ASIN_LIMIT 0.125

/* 1/6 as a double and the remainder */
#define GPU_ONE_SIXTH 0.16666666666666666
#define GPU_ONE_SIXTH_REMAINDER 9.2518585385429707e-18
/* 1/3 as a double and the remainder */
#define GPU_ONE_THIRD 0.33333333333333331
#define GPU_ONE_THIRD_REMAINDER 1.8503717077085941e-17

/* The sine as a double and a small remainder, so that the sum carries extra precision */
__device__ __forceinline__ void gpuAccurateSinParts(double x, double *high, double *low) {
  double x2, x2Low, x3, x3Low, leading, leadingLow, tail;

  x2 = x * x;
  x2Low = fma(x, x, -x2);
  x3 = x * x2;
  x3Low = fma(x, x2, -x3) + x * x2Low;
  leading = -x3 * GPU_ONE_SIXTH;
  leadingLow = fma(-x3, GPU_ONE_SIXTH, -leading) - x3 * GPU_ONE_SIXTH_REMAINDER -
               x3Low * GPU_ONE_SIXTH;
  tail = x3 * x2 * (1.0 / 120 + x2 * (-1.0 / 5040 + x2 * (1.0 / 362880 + x2 * (-1.0 / 39916800 + x2 * (1.0 / 6227020800.0 + x2 * (-1.0 / 1307674368000.0))))));
  *high = x + leading;
  *low = ((x - *high) + leading) + leadingLow + tail;
}

__device__ __forceinline__ double gpuAccurateSin(double x) {
  double high, low;

  if (fabs(x) > GPU_ACCURATE_TRIG_LIMIT)
    return sin(x);
  gpuAccurateSinParts(x, &high, &low);
  return high + low;
}

__device__ __forceinline__ double gpuAccurateCos(double x) {
  double x2, x2Low, half, high, tail;

  if (fabs(x) > GPU_ACCURATE_TRIG_LIMIT)
    return cos(x);
  x2 = x * x;
  x2Low = fma(x, x, -x2);
  half = 0.5 * x2;
  high = 1.0 - half;
  tail = x2 * x2 * (1.0 / 24 + x2 * (-1.0 / 720 + x2 * (1.0 / 40320 + x2 * (-1.0 / 3628800 + x2 * (1.0 / 479001600 + x2 * (-1.0 / 87178291200.0))))));
  return high + (((1.0 - high) - half) - 0.5 * x2Low + tail);
}

/* The series is enough for small slopes. Larger ones take one Newton step from CUDA's asin. */
__device__ __forceinline__ double gpuAccurateAsin(double x) {
  double x2, x2Low, x3, x3Low, leading, leadingLow, high, tail;

  if (fabs(x) > GPU_ACCURATE_ASIN_LIMIT) {
    double phi = asin(x), low;

    if (fabs(phi) > GPU_ACCURATE_TRIG_LIMIT)
      return phi;
    gpuAccurateSinParts(phi, &high, &low);
    return phi + ((x - high) - low) / cos(phi);
  }
  x2 = x * x;
  x2Low = fma(x, x, -x2);
  x3 = x * x2;
  x3Low = fma(x, x2, -x3) + x * x2Low;
  leading = x3 * GPU_ONE_SIXTH;
  leadingLow = fma(x3, GPU_ONE_SIXTH, -leading) + x3 * GPU_ONE_SIXTH_REMAINDER +
               x3Low * GPU_ONE_SIXTH;
  tail = x3 * x2 * (3.0 / 40 + x2 * (15.0 / 336 + x2 * (105.0 / 3456 + x2 * (945.0 / 42240 + x2 * (10395.0 / 599040 + x2 * (135135.0 / 9676800 + x2 * (2027025.0 / 175472640 + x2 * (34459425.0 / 3530096640.0))))))));
  high = x + leading;
  return high + (((x - high) + leading) + leadingLow + tail);
}

__device__ __forceinline__ double gpuAccurateAtan(double x) {
  double x2, x2Low, x3, x3Low, leading, leadingLow, high, tail;

  if (fabs(x) > GPU_ACCURATE_ASIN_LIMIT)
    return atan(x);
  x2 = x * x;
  x2Low = fma(x, x, -x2);
  x3 = x * x2;
  x3Low = fma(x, x2, -x3) + x * x2Low;
  leading = -x3 * GPU_ONE_THIRD;
  leadingLow = fma(-x3, GPU_ONE_THIRD, -leading) - x3 * GPU_ONE_THIRD_REMAINDER -
               x3Low * GPU_ONE_THIRD;
  tail = x3 * x2 * (1.0 / 5 + x2 * (-1.0 / 7 + x2 * (1.0 / 9 + x2 * (-1.0 / 11 + x2 * (1.0 / 13 + x2 * (-1.0 / 15 + x2 * (1.0 / 17 + x2 * (-1.0 / 19 + x2 * (1.0 / 21)))))))));
  high = x + leading;
  return high + (((x - high) + leading) + leadingLow + tail);
}

#endif /* GPU_ACCURATE_TRIG_H */
