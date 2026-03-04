#ifndef DL_PATCH_MANUAL_H
#define DL_PATCH_MANUAL_H

// Reciprocals
// [0|1]
#ifndef TURBO_RECIPROCALS
#define TURBO_RECIPROCALS 0
#endif

//[0|1] Replace isinf+isnan with isfinite, which is faster for IEEE floats (only 1 bitmask check)
#ifndef TURBO_FASTFINITE
#define TURBO_FASTFINITE 0
#endif

//[0|1] Replace isabs with inline bit twiddling
#ifndef TURBO_FASTABS
#define TURBO_FASTABS 0
#endif

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
