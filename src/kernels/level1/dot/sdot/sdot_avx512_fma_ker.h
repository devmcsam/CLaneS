//
// Created by sammc on 2/16/26.
//
#pragma once
#ifndef CLANES_SDOT_AVX512_KER_H
#define CLANES_SDOT_AVX512_KER_H
#include <immintrin.h>
#include <stddef.h>

// literal best case scenario, avx512_fma and incx == incy == 1, fully vectorizable
static float sdot_kernel_avx512_fma_b1(const size_t n, const float* restrict x, const float* restrict y) {
    size_t i = 0;
    float final_res = 0.0f;

    // accumulators, help hide latency of fpu ops
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();

    // 64 floats at once because of out accumulators
    for(; i + 63 < n; i += 64) {
        const __m512 x0 = _mm512_loadu_ps(&x[i]);
        const __m512 y0 = _mm512_loadu_ps(&y[i]);
        acc0 = _mm512_fmadd_ps(x0, y0, acc0);

        const __m512 x1 = _mm512_loadu_ps(&x[i + 16]);
        const __m512 y1 = _mm512_loadu_ps(&y[i + 16]);
        acc1 = _mm512_fmadd_ps(x1, y1, acc1);

        const __m512 x2 = _mm512_loadu_ps(&x[i + 32]);
        const __m512 y2 = _mm512_loadu_ps(&y[i + 32]);
        acc2 = _mm512_fmadd_ps(x2, y2, acc2);

        const __m512 x3 = _mm512_loadu_ps(&x[i + 48]);
        const __m512 y3 = _mm512_loadu_ps(&y[i + 48]);
        acc3 = _mm512_fmadd_ps(x3, y3, acc3);
    }

    // combine
    __m512 sum = _mm512_add_ps(_mm512_add_ps(acc0, acc1),
                               _mm512_add_ps(acc2, acc3));

    // remaining handled by masking
    for(; i < n; i += 16) {
        const int remaining = n - i;
        // if tail < 16, create mask
        const __mmask16 mask = (remaining >= 16) ? 0xFFFF : (1U << remaining) - 1;

        const __m512 xt = _mm512_maskz_loadu_ps(mask, &x[i]);
        const __m512 yt = _mm512_maskz_loadu_ps(mask, &y[i]);
        sum = _mm512_mask3_fmadd_ps(xt, yt, sum, mask);
    }

    // 512 reg -> 32 reg
    final_res = _mm512_reduce_add_ps(sum);

    return final_res;
}


// have avx512_fma and both incx and incy are the same and > 1
static float sdot_kernel_avx512_fma_bep(const size_t n,
                                        const float* restrict x,
                                        const float* restrict y,
                                        const int64_t inc) {
    size_t i = 0;
    __m512 acc = _mm512_setzero_ps();

    // Convert stride to byte offset (float is 4 bytes), keep in signed arithmetic
    const int32_t byte_stride = (int32_t)(inc * (int64_t)sizeof(float));

    // Pre-calculate the index vector: {0, 1, 2, ..., 15}
    const __m512i indices = _mm512_set_epi32(15,
                                             14,
                                             13,
                                             12,
                                             11,
                                             10,
                                             9,
                                             8,
                                             7,
                                             6,
                                             5,
                                             4,
                                             3,
                                             2,
                                             1,
                                             0);

    // Scale by stride to get rel offset
    const __m512i v_offsets = _mm512_mullo_epi32(indices, _mm512_set1_epi32(byte_stride));

    // 16 floats at a time, these are strided though
    for(; i + 15 < n; i += 16) {
        __m512 vx = _mm512_i32gather_ps(v_offsets, &x[i * inc], 1);
        __m512 vy = _mm512_i32gather_ps(v_offsets, &y[i * inc], 1);
        acc = _mm512_fmadd_ps(vx, vy, acc);
    }

    // tail handling if doesnt fit in nice range
    if(i < n) {
        const int remaining = (int)(n - i);
        const __mmask16 mask = (1U << remaining) - 1;

        __m512 vx = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, v_offsets, &x[i * inc], 1);
        __m512 vy = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, v_offsets, &y[i * inc], 1);

        acc = _mm512_fmadd_ps(vx, vy, acc);
    }

    return _mm512_reduce_add_ps(acc);
}

// General mismatched/fallback kernel: handles any incx, incy combination (including negatives)
static float sdot_kernel_avx512_fma_mismatched(const size_t n,
                                               const float* restrict x,
                                               const int64_t incx,
                                               const float* restrict y,
                                               const int64_t incy) {
    // Adjust base pointers for negative strides (BLAS convention)
    const float* ptr_x = (incx < 0) ? x + ((int64_t)(n - 1) * (-incx)) : x;
    const float* ptr_y = (incy < 0) ? y + ((int64_t)(n - 1) * (-incy)) : y;

    // Use absolute values of strides for forward traversal from adjusted base
    const int32_t stride_x = (int32_t)(incx < 0 ? -incx : incx);
    const int32_t stride_y = (int32_t)(incy < 0 ? -incy : incy);

    __m512 acc = _mm512_setzero_ps();

    // Constant index vector {0, 1, ..., 15}
    const __m512i v_idx = _mm512_set_epi32(15,
                                           14,
                                           13,
                                           12,
                                           11,
                                           10,
                                           9,
                                           8,
                                           7,
                                           6,
                                           5,
                                           4,
                                           3,
                                           2,
                                           1,
                                           0);

    // Pre-calculate byte-offset vectors for both (signed arithmetic)
    const __m512i v_off_x = _mm512_mullo_epi32(v_idx, _mm512_set1_epi32((int32_t)(stride_x * (int32_t)sizeof(float))));
    const __m512i v_off_y = _mm512_mullo_epi32(v_idx, _mm512_set1_epi32((int32_t)(stride_y * (int32_t)sizeof(float))));

    size_t i = 0;
    for(; i + 15 < n; i += 16) {
        __m512 vx;
        __m512 vy;

        // Optimization: Use contiguous load if stride is 1 (forward)
        if(stride_x == 1) {
            vx = _mm512_loadu_ps(ptr_x + i);
        }
        else {
            vx = _mm512_i32gather_ps(v_off_x, ptr_x + (i * stride_x), 1);
        }

        if(stride_y == 1) {
            vy = _mm512_loadu_ps(ptr_y + i);
        }
        else {
            vy = _mm512_i32gather_ps(v_off_y, ptr_y + (i * stride_y), 1);
        }

        acc = _mm512_fmadd_ps(vx, vy, acc);
    }

    // Tail cleanup with masking
    if(i < n) {
        const int rem = (int)(n - i);
        const __mmask16 mask = (1U << rem) - 1;

        __m512 vx;
        __m512 vy;
        if(stride_x == 1) {
            vx = _mm512_maskz_loadu_ps(mask, ptr_x + i);
        }
        else {
            vx = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, v_off_x, ptr_x + (i * stride_x), 1);
        }

        if(stride_y == 1) {
            vy = _mm512_maskz_loadu_ps(mask, ptr_y + i);
        }
        else {
            vy = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, v_off_y, ptr_y + (i * stride_y), 1);
        }

        acc = _mm512_fmadd_ps(vx, vy, acc);
    }

    return _mm512_reduce_add_ps(acc);
}

#endif //CLANES_SDOT_AVX512_KER_H
