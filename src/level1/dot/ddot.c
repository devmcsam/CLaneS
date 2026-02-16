//
// Created by sammc on 2/16/26.
//
#include "clanes_internal.h"
#include "level1/level1.h"

CLANES_TARGET_AVX512_FMA
static double ddot_dispatch_avx512_fma(const int n,
                                       const double* restrict x,
                                       const int incx,
                                       const double* restrict y,
                                       const int incy) {
    // Implementation goes here
    return 0.0;
}

CLANES_TARGET_AVX512
static double ddot_dispatch_avx512(const int n,
                                   const double* restrict x,
                                   const int incx,
                                   const double* restrict y,
                                   const int incy) {
    // Implementation goes here
    return 0.0;
}

CLANES_TARGET_AVX2_FMA
static double ddot_dispatch_avx2_fma(const int n,
                                     const double* restrict x,
                                     const int incx,
                                     const double* restrict y,
                                     const int incy) {
    // Implementation goes here
    return 0.0;
}

CLANES_TARGET_AVX2
static double ddot_dispatch_avx2(const int n,
                                 const double* restrict x,
                                 const int incx,
                                 const double* restrict y,
                                 const int incy) {
    // Implementation goes here
    return 0.0;
}

CLANES_TARGET_AVX
static double ddot_dispatch_avx(const int n,
                                const double* restrict x,
                                const int incx,
                                const double* restrict y,
                                const int incy) {
    // Implementation goes here
    return 0.0;
}

CLANES_TARGET_SSE41
static double ddot_dispatch_sse41(const int n,
                                  const double* restrict x,
                                  const int incx,
                                  const double* restrict y,
                                  const int incy) {
    // Implementation goes here
    return 0.0;
}

CLANES_TARGET_SSE2
static double ddot_dispatch_sse2(const int n,
                                 const double* restrict x,
                                 const int incx,
                                 const double* restrict y,
                                 const int incy) {
    // Implementation goes here
    return 0.0;
}

// SSE1 has no double precision simd so we just have to do it with scalars unfortunately
CLANES_TARGET_DEFAULT
static double ddot_dispatch_sse(const int n,
                                const double* restrict x,
                                const int incx,
                                const double* restrict y,
                                const int incy) {
    // Implementation goes here
    return 0.0;
}

CLANES_TARGET_DEFAULT
static double ddot_dispatch_scalar(const int n,
                                   const double* restrict x,
                                   const int incx,
                                   const double* restrict y,
                                   const int incy) {
    // Implementation goes here
    return 0.0;
}


CLANES_RESOLVER_BEGIN(double, clanes_ddot, int, const double*, int, const double*, int)
    CLANES_SELECT_DISPATCHER(ddot_dispatch)
CLANES_RESOLVER_END

CLANES_DECLARE_IFUNC(double,
                     clanes_ddot,
                     int n,
                     const double* restrict x,
                     int incx,
                     const double* restrict y,
                     int incy);

/*
 * What we branch on for the kernel:
 * 1. incx and incy = 1, this is the golden case since we can fully vectorize
 * 2. incx == incy > 1, we can vectorize still but is not as perfect as them both equaling 1
 * 3. incx != incy, we cannot easily vectorize reliably, must fall to a scalar fallback (we still manually unwind though)
 */

