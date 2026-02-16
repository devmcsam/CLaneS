//
// Created by sammc on 2/16/26.
//
#include "clanes_internal.h"
#include "kernels/level1/dot/sdot/sdot_avx512_fma_ker.h"
#include "level1/level1.h"

/*
 * What we branch on for the kernels:
 * 1. incx and incy = 1, this is the golden case since we can fully vectorize
 * 2. incx == incy > 1, we can vectorize still but is not as perfect as them both equaling 1
 * 3. incx != incy, we cannot easily vectorize reliably, must fall to a scalar fallback (we still manually unwind though)
 */

CLANES_TARGET_AVX512_FMA
static float sdot_dispatch_avx512_fma(const int n,
                                      const float* restrict x,
                                      const int incx,
                                      const float* restrict y,
                                      const int incy) {
    // n <= 0 or zero increments: BLAS returns 0
    if(n <= 0 || incx == 0 || incy == 0) {
        return 0.0F;
    }

    // Golden path: both strides are 1, fully contiguous
    if(incx == 1 && incy == 1) {
        return sdot_kernel_avx512_fma_b1(n, x, y);
    }

    // strides equal and positive
    if(incx == incy && incx > 1) {
        return sdot_kernel_avx512_fma_bep(n, x, y, incx);
    }

    // mismatched strides, negative strides
    return sdot_kernel_avx512_fma_mismatched(n, x, incx, y, incy);
}

CLANES_TARGET_AVX512
static float sdot_dispatch_avx512(const int n,
                                  const float* restrict x,
                                  const int incx,
                                  const float* restrict y,
                                  const int incy) {
    // Implementation goes here
    return 0.0f;
}

CLANES_TARGET_AVX2_FMA
static float sdot_dispatch_avx2_fma(const int n,
                                    const float* restrict x,
                                    const int incx,
                                    const float* restrict y,
                                    const int incy) {
    // Implementation goes here
    return 0.0f;
}

CLANES_TARGET_AVX2
static float sdot_dispatch_avx2(const int n,
                                const float* restrict x,
                                const int incx,
                                const float* restrict y,
                                const int incy) {
    // Implementation goes here
    return 0.0f;
}

CLANES_TARGET_AVX
static float sdot_dispatch_avx(const int n,
                               const float* restrict x,
                               const int incx,
                               const float* restrict y,
                               const int incy) {
    // Implementation goes here
    return 0.0f;
}

CLANES_TARGET_SSE41
static float sdot_dispatch_sse41(const int n,
                                 const float* restrict x,
                                 const int incx,
                                 const float* restrict y,
                                 const int incy) {
    // Implementation goes here
    return 0.0f;
}

CLANES_TARGET_SSE2
static float sdot_dispatch_sse2(const int n,
                                const float* restrict x,
                                const int incx,
                                const float* restrict y,
                                const int incy) {
    // Implementation goes here
    return 0.0f;
}

CLANES_TARGET_SSE
static float sdot_dispatch_sse(const int n,
                               const float* restrict x,
                               const int incx,
                               const float* restrict y,
                               const int incy) {
    // Implementation goes here
    return 0.0f;
}

CLANES_TARGET_DEFAULT
static float sdot_dispatch_scalar(const int n,
                                  const float* restrict x,
                                  const int incx,
                                  const float* restrict y,
                                  const int incy) {
    // Implementation goes here
    return 0.0f;
}


CLANES_RESOLVER_BEGIN(float, clanes_sdot, int, const float*, int, const float*, int)
    CLANES_SELECT_DISPATCHER(sdot_dispatch)
CLANES_RESOLVER_END

CLANES_DECLARE_IFUNC(float, clanes_sdot, int n, const float* restrict x, int incx, const float* restrict y, int incy);
