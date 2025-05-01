#ifndef DL_PATCH_MANUAL_H
#define DL_PATCH_MANUAL_H


// note that even at level 0, some code has been moved around to headers
// 0 - least possible changes
// 1 - csbend field caching
// 2 - layout changes/static fields
// 3 - static declarations
// 4 - more statics
// 5 - csbendfield
// 6 - matrix memory layout overhaul (why output changes is unclear)
// 7 - static momenta conversions
// 8 - extract non-distribution radiation kick
// 9 - try to fix alignment of rootname copy, still need to reduce amount of copying or buffer size

// 1-10 are integrated into the code already

// REFERENCE_CONFIG
//#define TURBO_TRACKING 0
//#define TURBO_STRINGS 0
//#define TURBO_CCBEND_REFACTOR 0
//#define TURBO_CCBEND_STATICS 0
//#define TURBO_RECIPROCALS 0
//#define TURBO_COEF_CACHE 0
//#define TURBO_APPLY_KICKS_FAST 0
//#define TURBO_MATRICES 0
//#define TURBO_QUADFRINGE 0


// 1 - simplify tracking loop
#ifndef TURBO_TRACKING
#define TURBO_TRACKING 1
#endif

// 1 - use pointers instead of copying name into tracking context
#ifndef TURBO_STRINGS
#define TURBO_STRINGS 1
#endif

// 1 - assume drift!=0 in csbend, removing some conditionals
// 2 - change some computations in integrator to conditionals
#ifndef TURBO_CCBEND_REFACTOR
#define TURBO_CCBEND_REFACTOR 2
#endif

// Further changes in ccbend tracking to use static arrays [THIS LIMITS MAX ORDER to a fixed but configurable value]
#ifndef TURBO_CCBEND_STATICS
#define TURBO_CCBEND_STATICS 1
#endif

// Reciprocals
// [0|1]
#ifndef TURBO_RECIPROCALS
#define TURBO_RECIPROCALS 0
#endif

// Define as compile-time constants
// 1 - only orders 0,1,2
// 3 - also cache order 3 - this changes output because 1/6 != 1/2 * 1/3 in float64
// 4 - order 4
#ifndef TURBO_COEF_CACHE
#define TURBO_COEF_CACHE 1
#endif

// specialize multipole kick methods
// 1 - no reverse
// 2 - integrate division by nparts
// 3 - try pre-filtering active orders
// 4 - remove branches
// 5 - do not use
// 6 - reverse x powers, do not use
#ifndef TURBO_APPLY_KICKS_FAST
#define TURBO_APPLY_KICKS_FAST 4
#endif

//[0|1]
#ifndef TURBO_MATRICES
#define TURBO_MATRICES 1
#endif

//[0|1]
#ifndef TURBO_QUADFRINGE
#define TURBO_QUADFRINGE 1
#endif

//[0|1] Do string comparisons by length first before strcmp
#ifndef TURBO_STRLEN
#define TURBO_STRLEN 1
#endif

//[0|1|2|3]
#ifndef TURBO_MATLOCAL
#define TURBO_MATLOCAL 1
#endif

//[0|1] Replace isinf+isnan with isfinite, which is faster for IEEE floats (only 1 bitmask check)
#ifndef TURBO_FASTFINITE
#define TURBO_FASTFINITE 0
#endif

//[0|1] Replace isabs with inline bit twiddling
#ifndef TURBO_FASTABS
#define TURBO_FASTABS 0
#endif

//[0|1|2|3|4] Unroll and simplify matrix multiplication
// 1-3 no output changes
// 4 is best, changes output
#ifndef TURBO_FASTMATTRACK
#define TURBO_FASTMATTRACK 3
#endif

//[0|1|2|3|4] Improve poisson solver
// 0 - stock
// 1 - change memory management to reuse buffers
// 2 - tune various expensive array operations, reduce divisions
// 3 - use inplace FFT (not working)
// 4 - use threaded FFTW
#ifndef TURBO_FASTPOISSON
#define TURBO_FASTPOISSON 2
#endif

//[0|1] Faster wofz implementation
//http://ab-initio.mit.edu/wiki/index.php/Faddeeva_Package
#ifndef TURBO_FADDEEVA
#define TURBO_FADDEEVA 1
#endif

//[0|1] Disable flush in tracking loop
#ifndef TURBO_NOFLUSH
#define TURBO_NOFLUSH 1
#endif

//#ifndef TURBO_STRUCTLAYOUT
//#define TURBO_STRUCTLAYOUT 1
//#endif

// These commands aim to enable useful parts of fast math without destroying NaN/Inf handling
// attribute version is used for functions, and start/end can be used for blocks/regions (depends on compiler)
// Note that this will result in numeric differences between compilers - use with caution
#if defined(__GNUC__) && !defined(__clang__)
// #define FAST_MATH [[gnu::optimize("-funsafe-math-optimizations")]]
#define FAST_MATH __attribute__((optimize("no-trapping-math,associative-math,reciprocal-math,no-signed-zeros,no-signaling-nans,fp-contract=fast")))
#define FAST_MATH_BEGIN _Pragma("GCC push_options") \
                        _Pragma("GCC optimize (\"no-trapping-math,associative-math,reciprocal-math,no-signed-zeros,no-signaling-nans,fp-contract=fast\")")
#define FAST_MATH_END   _Pragma("GCC pop_options")
#elif defined(__clang__)
#define FAST_MATH __attribute__((target("fp-model=fast"))) // roughly equivalent to -funsafe-math-optimizations in GCC
// To use outside compound statements, add push
// float_control is a recent addition to clang at v21
// #define FAST_MATH_BEGIN _Pragma(float_control(precise, off)) _Pragma(fp_contract(on))                     
// reciprocal(on) not supported on clang 16
#define FAST_MATH_BEGIN _Pragma("clang fp reassociate(on) contract(fast)")
// #define FAST_MATH_END _Pragma(float_control(precise, on)) _Pragma(fp_contract(off))               
// reciprocal(off)
#define FAST_MATH_END _Pragma("clang fp reassociate(off) contract(on)")
#elif defined(_MSC_VER)
// MSVC: Use pragmas around the function definition.
#define FAST_MATH
#define FAST_MATH_BEGIN __pragma(float_control(push)) \
                        __pragma(float_control(precise, off)) \
                        __pragma(float_control(except, off)) \
                        __pragma(fp_contract(on))
#define FAST_MATH_END   __pragma(float_control(pop))
#else
#define FAST_MATH
#define FAST_MATH_BEGIN
#define FAST_MATH_END
#endif

#ifdef _MSC_VER
#define UNUSED /*empty*/
#else
#define UNUSED __attribute__((unused))
#endif

// Bit 0 is sign bit, need to set it to 0, so bitwise AND by 0111111....
UNUSED
static inline double fast_fabs(double a) {
  union {
    uint64_t i;
    double f;
  } cast_helper;

  cast_helper.f = a;
  cast_helper.i &= 0x7fffffffffffffff;
  return cast_helper.f;
}

// All exponent bits 1 indicates NaN or inf
UNUSED
static inline int fast_isfinite(double a) {
  union {
    uint64_t i;
    double f;
  } u;
  u.f = a;
  return (u.i & 0x7fffffffffffffff) < 0x7ff0000000000000 ? 1 : 0;
}

//int __fpclassify(double x)
//{
//  union {double f; uint64_t i;} u = {x};
//  int e = u.i>>52 & 0x7ff;
//  if (!e) return u.i<<1 ? FP_SUBNORMAL : FP_ZERO;
//  if (e==0x7ff) return u.i<<12 ? FP_NAN : FP_INFINITE;
//  return FP_NORMAL;
//}
// (*(((long*) &x)) & 0x7fffffffffffffffL) < 0x7ff0000000000000L
//bool myIsnan(double v) {
//  std::uint64_t i;
//  memcpy(&i, &v, 8);
//  return ((i&0x7ff0000000000000)==0x7ff0000000000000)&&(i&0xfffffffffffff);
//}

#endif //DL_PATCH_MANUAL_H
