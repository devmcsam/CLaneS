cmake_minimum_required(VERSION 4.1)
project(CLaneS VERSION 0.1.0 LANGUAGES C)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(RELEASE_CFLAGS
        -O3 -funroll-loops -fno-strict-aliasing -fno-unsafe-math-optimizations
)

set(DEBUG_CFLAGS
        -Og -g
        -Wall -Wextra -Wpedantic
        -Wshadow -Wcast-align -Wconversion -Wsign-conversion -Wfloat-equal
        -Wpointer-arith -Wundef -Wlogical-op -Wformat=2
)

set(RELWITHDEBINFO_CFLAGS
        -g ${RELEASE_CFLAGS}
)

option(CLANES_ENABLE_MARCH_NATIVE
        "Enable -march=native for this library, NOTE: With this enabled it is not a portable library"
        OFF)

add_library(CLaneS STATIC src/clanes.c
        include/level1/level1.h
        include/level2/level2.h
        include/level3/level3.h
        src/clanes_internal.h
        include/clanes_tg.h
        include/level1/level1_tg.h
        include/level2/level2_tg.h
        include/level3/level3_tg.h
        src/level1/dot/sdot.c
        src/level1/dot/ddot.c
        src/kernels/level1/dot/sdot/sdot_avx512_fma_ker.h)

target_include_directories(CLaneS PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(CLaneS PRIVATE ${PROJECT_SOURCE_DIR}/src)

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)
if (ipo_supported)
    set_target_properties(CLaneS PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif ()

if (CLANES_ENABLE_MARCH_NATIVE)
    target_compile_options(CLaneS
            PRIVATE
            $<$<COMPILE_LANGUAGE:C>:-march=native>
    )
endif ()

# Apply flags by configuration — library
target_compile_options(CLaneS PRIVATE
        $<$<CONFIG:Release>:${RELEASE_CFLAGS}>
        $<$<CONFIG:Debug>:${DEBUG_CFLAGS}>
        $<$<CONFIG:RelWithDebInfo>:${RELWITHDEBINFO_CFLAGS}>
)

# Apply macros by configuration
target_compile_definitions(CLaneS PRIVATE
        $<$<CONFIG:Release>:NDEBUG>
        $<$<CONFIG:Debug>:DEBUG>
        $<$<CONFIG:RelWithDebInfo>:NDEBUG>
)
