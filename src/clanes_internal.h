//
// Created by sammc on 2/15/26.
//
#pragma once
#ifndef CLANES_CLANES_INTERNAL_H
#define CLANES_CLANES_INTERNAL_H

#include <stdint.h>

// cpu id register bit positions

// Leaf 1: EDX bits
#define CLANES_CPUID_1_EDX_SSE    25u
#define CLANES_CPUID_1_EDX_SSE2   26u

// Leaf 1: ECX bits
#define CLANES_CPUID_1_ECX_SSE3   0u
#define CLANES_CPUID_1_ECX_SSSE3  9u
#define CLANES_CPUID_1_ECX_SSE41  19u
#define CLANES_CPUID_1_ECX_SSE42  20u
#define CLANES_CPUID_1_ECX_FMA    12u
#define CLANES_CPUID_1_ECX_AVX    28u

// Leaf 7, Sub-leaf 0: EBX bits
#define CLANES_CPUID_7_0_EBX_AVX2      5u
#define CLANES_CPUID_7_0_EBX_AVX512_F   16u
#define CLANES_CPUID_7_0_EBX_AVX_512DQ  17u
#define CLANES_CPUID_7_0_EBX_AVX512_BW  30u
#define CLANES_CPUID_7_0_EBX_AVX512_VL  31u

// cpu feature bits
#define CLANES_CPU_SSE       (1u <<  0)
#define CLANES_CPU_SSE2      (1u <<  1)
#define CLANES_CPU_SSE3      (1u <<  2)
#define CLANES_CPU_SSSE3     (1u <<  3)
#define CLANES_CPU_SSE4_1    (1u <<  4)
#define CLANES_CPU_SSE4_2    (1u <<  5)
#define CLANES_CPU_AVX       (1u <<  6)
#define CLANES_CPU_AVX2      (1u <<  7)
#define CLANES_CPU_FMA       (1u <<  8)
#define CLANES_CPU_AVX512_F   (1u <<  9)
#define CLANES_CPU_AVX512_BW  (1u << 10)
#define CLANES_CPU_AVX512_VL  (1u << 11)
#define CLANES_CPU_AVX_512DQ  (1u << 12)

// Useful compound masks
#define CLANES_CPU_AVX2_FMA   (CLANES_CPU_AVX2   | CLANES_CPU_FMA)
#define CLANES_CPU_AVX512_FMA (CLANES_CPU_AVX512_F | CLANES_CPU_FMA)

// Widest usable SIMD tier
typedef enum {
    CLANES_SIMD_SCALAR = 0,
    CLANES_SIMD_SSE    = 128,
    CLANES_SIMD_SSE2   = 129, // same width, distinct level
    CLANES_SIMD_SSE41  = 130,
    CLANES_SIMD_AVX    = 256,
    CLANES_SIMD_AVX2   = 257,
    CLANES_SIMD_AVX512 = 512,
} clanes_simd_level;

typedef struct {
    uint32_t features; // bitfield of CLANES_CPU_* flags
    clanes_simd_level max_level; // resolved widest tier
} clanes_cpu_info;

// Immutable after clanes_init().
extern clanes_cpu_info g_clanes_cpu;

// Quick feature test — compiles to: load + test + jz
#define CLANES_HAS(mask) ((g_clanes_cpu.features & (mask)) == (mask))

// cpu target attribute helpers
#define CLANES_TARGET_DEFAULT    __attribute__((target("default")))
#define CLANES_TARGET_SSE        __attribute__((target("sse")))
#define CLANES_TARGET_SSE2       __attribute__((target("sse2")))
#define CLANES_TARGET_SSE41      __attribute__((target("sse4.1")))
#define CLANES_TARGET_AVX        __attribute__((target("avx")))
#define CLANES_TARGET_AVX2       __attribute__((target("avx2")))
#define CLANES_TARGET_AVX2_FMA   __attribute__((target("avx2,fma")))
#define CLANES_TARGET_AVX512     __attribute__((target("avx512f")))
#define CLANES_TARGET_AVX512_FMA __attribute__((target("avx512f,fma")))


//  ifunc dispatch macros

#define CLANES_STR_(x) #x
#define CLANES_STR(x)  CLANES_STR_(x)
#define CLANES_CAT_(a, b) a##b
#define CLANES_CAT(a, b)  CLANES_CAT_(a, b)

/**
 * CLANES_DECLARE_IFUNC(ret, name, ...)
 *   Declares the public symbol `name` as an ifunc resolved by `resolve_##name`.
 *
 *   Usage:
 *     CLANES_DECLARE_IFUNC(void, clanes_saxpy, int n, float a, const float *x, float *y);
 */
#define CLANES_DECLARE_IFUNC(ret, name, ...)                                \
    ret name(__VA_ARGS__)                                                   \
        __attribute__((ifunc(CLANES_STR(CLANES_CAT(resolve_, name)))))

/**
 * CLANES_RESOLVER_BEGIN / END
 *   Opens/closes the resolver. Use CLANES_RESOLVE_* inside.
 *
 *   Usage:
 *     CLANES_RESOLVER_BEGIN(void, clanes_saxpy, int, float, const float*, float*)
 *         CLANES_RESOLVE(CLANES_CPU_AVX512_FMA, saxpy_avx512)
 *         CLANES_RESOLVE(CLANES_CPU_AVX2_FMA,   saxpy_avx2)
 *         CLANES_RESOLVE_FALLBACK(saxpy_scalar)
 *     CLANES_RESOLVER_END
 */
#define CLANES_RESOLVER_BEGIN(ret, name, ...)                               \
    static ret (*CLANES_CAT(resolve_, name)(void))(__VA_ARGS__) {

#define CLANES_RESOLVER_END  }

// Generic: pass any CLANES_CPU_* mask (or compound)
#define CLANES_RESOLVE(mask, fn)                                            \
    if (CLANES_HAS(mask)) return (fn);

#define CLANES_RESOLVE_FALLBACK(fn)                                         \
    return (fn);

/**
 * CLANES_SELECT_DISPATCHER(name_prefix)
 *   Standard resolver body for selecting a dispatcher.
 *   Prioritizes FMA versions for AVX2 and AVX512.
 */
#define CLANES_SELECT_DISPATCHER(name_prefix) \
switch (g_clanes_cpu.max_level) { \
case CLANES_SIMD_AVX512: \
if (CLANES_HAS(CLANES_CPU_FMA)) return CLANES_CAT(name_prefix, _avx512_fma); \
return CLANES_CAT(name_prefix, _avx512); \
case CLANES_SIMD_AVX2: \
if (CLANES_HAS(CLANES_CPU_FMA)) return CLANES_CAT(name_prefix, _avx2_fma); \
return CLANES_CAT(name_prefix, _avx2); \
case CLANES_SIMD_AVX:    return CLANES_CAT(name_prefix, _avx);    \
case CLANES_SIMD_SSE41:  return CLANES_CAT(name_prefix, _sse41);  \
case CLANES_SIMD_SSE2:   return CLANES_CAT(name_prefix, _sse2);   \
case CLANES_SIMD_SSE:    return CLANES_CAT(name_prefix, _sse);    \
default:                 return CLANES_CAT(name_prefix, _scalar); \
}

#endif //CLANES_CLANES_INTERNAL_H
