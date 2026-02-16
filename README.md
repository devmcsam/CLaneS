# CLaneS

## Overview

CLaneS (C Lane Scalar) is a blas implementation done in C. The goal of this project is to provide an easily optimizable
blas implementation in C23. Instead of using manual assembly, CLaneS uses compiler intrinsics with a scalar fallback.
CLaneS does one cpuid call at program startup and then saves it to an internal variable. It then secretly passes this
struct to each function (so you don't deal with it every time). Then each function looks at the struct and uses the most
optimal implementation it can given the runtime architecture.

## Usage

### Init

You must call `clanes_init()` if you would like CLaneS to use the cpuid information. This function initializes the
struct that holds the information that the other blas functions use. It is recommended to call this at the beginning of
your main function so that everything is initialized at program startup. If you do not call this function, all functions
will use the scalar fallback which is not nearly as fast as the optimized versions.

### Aliasing

CLaneS does not support pointer aliasing. This means that you cannot pass the same pointer as the input and output to a
function. The functions use the `restrict` keyword; this means if you pass the same pointer to both input and output, it
is UB.

### BLAS Functions

The BLAS functions are all defined in [clanes.h](include/clanes.h). They are the same api as a standard blas
implementation.

## Why Intrinsics?

There are a few reasons that I used compiler intrinsics instead of assembly.

1. Skill issue, I am not good enough at assembly to make it faster than a compiler's intrinsic instruction, especially
   for simd.
2. Compiler intrinsics allow for more optimization within the compiler. For example, when writing an inline assembly,
   the compiler has to treat it kind of as a black box, where as it understands intrinsics semantically and can optimize
   memory loads and other stuff around it.
3. As long as the Cpu supports the width and operation, it can use the intrinsic; this makes the code much more portable
   than if I were to write a different assembly instruction per architecture.

## License

CLaneS is licensed under the MIT License. See [LICENSE](LICENSE) for details.