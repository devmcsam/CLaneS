#include "clanes.h"
#include <cpuid.h>
#include "clanes_internal.h"

clanes_cpu_info g_clanes_cpu = {0};

__attribute__((constructor))
void clanes_init(void) {
    unsigned int eax;
    unsigned int ebx;
    unsigned int ecx;
    unsigned int edx;
    uint32_t f = 0;

    // ---- Leaf 1: basic feature flags ----
    if(!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        g_clanes_cpu.features = 0;
        g_clanes_cpu.max_level = CLANES_SIMD_SCALAR;
        return;
    }

    if(edx & (1U << CLANES_CPUID_1_EDX_SSE)) {
        f |= CLANES_CPU_SSE;
    }
    if(edx & (1U << CLANES_CPUID_1_EDX_SSE2)) {
        f |= CLANES_CPU_SSE2;
    }
    if(ecx & (1U << CLANES_CPUID_1_ECX_SSE3)) {
        f |= CLANES_CPU_SSE3;
    }
    if(ecx & (1U << CLANES_CPUID_1_ECX_SSSE3)) {
        f |= CLANES_CPU_SSSE3;
    }
    if(ecx & (1U << CLANES_CPUID_1_ECX_SSE41)) {
        f |= CLANES_CPU_SSE4_1;
    }
    if(ecx & (1U << CLANES_CPUID_1_ECX_SSE42)) {
        f |= CLANES_CPU_SSE4_2;
    }
    if(ecx & (1U << CLANES_CPUID_1_ECX_AVX)) {
        f |= CLANES_CPU_AVX;
    }
    if(ecx & (1U << CLANES_CPUID_1_ECX_FMA)) {
        f |= CLANES_CPU_FMA;
    }

    // ---- Leaf 7, sub-leaf 0: extended features ----
    if(__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        if(ebx & (1U << CLANES_CPUID_7_0_EBX_AVX2)) {
            f |= CLANES_CPU_AVX2;
        }
        if(ebx & (1U << CLANES_CPUID_7_0_EBX_AVX512_F)) {
            f |= CLANES_CPU_AVX512_F;
        }
        if(ebx & (1U << CLANES_CPUID_7_0_EBX_AVX_512DQ)) {
            f |= CLANES_CPU_AVX_512DQ;
        }
        if(ebx & (1U << CLANES_CPUID_7_0_EBX_AVX512_BW)) {
            f |= CLANES_CPU_AVX512_BW;
        }
        if(ebx & (1U << CLANES_CPUID_7_0_EBX_AVX512_VL)) {
            f |= CLANES_CPU_AVX512_VL;
        }
    }

    g_clanes_cpu.features = f;

    // ---- Resolve widest tier (widest first) ----
    if(f & CLANES_CPU_AVX512_F) {
        g_clanes_cpu.max_level = CLANES_SIMD_AVX512;
    }
    else if(f & CLANES_CPU_AVX2) {
        g_clanes_cpu.max_level = CLANES_SIMD_AVX2;
    }
    else if(f & CLANES_CPU_AVX) {
        g_clanes_cpu.max_level = CLANES_SIMD_AVX;
    }
    else if(f & CLANES_CPU_SSE4_1) {
        g_clanes_cpu.max_level = CLANES_SIMD_SSE41;
    }
    else if(f & CLANES_CPU_SSE2) {
        g_clanes_cpu.max_level = CLANES_SIMD_SSE2;
    }
    else if(f & CLANES_CPU_SSE) {
        g_clanes_cpu.max_level = CLANES_SIMD_SSE;
    }
    else {
        g_clanes_cpu.max_level = CLANES_SIMD_SCALAR;
    }
}
