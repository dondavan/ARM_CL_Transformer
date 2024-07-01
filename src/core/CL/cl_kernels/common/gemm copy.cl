/*
 * Copyright (c) 2017-2022 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "gemm_helpers.h"
#include "repeat.h"

#if defined(M0) && defined(N0) && defined(K0) && defined(H0) && defined(DATA_TYPE)

#define CONCAT(a, b) a##b

#define ARM_DOT1(a, b, c) \
    ({                    \
        c = fma(a, b, c); \
    })
#define ARM_DOT2(a, b, c)       \
    ({                          \
        c = fma(a.s0, b.s0, c); \
        c = fma(a.s1, b.s1, c); \
    })
#define ARM_DOT3(a, b, c)           \
    ({                              \
        ARM_DOT2(a, b, c);          \
        c = fma((a.s2), (b.s2), c); \
    })
#define ARM_DOT4(a, b, c)           \
    ({                              \
        ARM_DOT3(a, b, c);          \
        c = fma((a.s3), (b.s3), c); \
    })
#define ARM_DOT8(a, b, c)            \
    ({                               \
        ARM_DOT4((a.lo), (b.lo), c); \
        ARM_DOT4((a.hi), (b.hi), c); \
    })
#define ARM_DOT16(a, b, c)           \
    ({                               \
        ARM_DOT8((a.lo), (b.lo), c); \
        ARM_DOT8((a.hi), (b.hi), c); \
    })

#if N0 == 2
#define ARM_DOT_K0XN0(k0, a, b, c) \
    ({                             \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##0), (c.s0));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##1), (c.s1));     \
    })
#elif N0 == 3 // N0 == 3
#define ARM_DOT_K0XN0(k0, a, b, c) \
    ({                             \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##0), (c.s0));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##1), (c.s1));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##2), (c.s2));     \
    })
#elif N0 == 4 // N0 == 4
#define ARM_DOT_K0XN0(k0, a, b, c) \
    ({                             \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##0), (c.s0));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##1), (c.s1));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##2), (c.s2));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##3), (c.s3));     \
    })
#elif N0 == 8 // N0 == 8
#define ARM_DOT_K0XN0(k0, a, b, c) \
    ({                             \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##0), (c.s0));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##1), (c.s1));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##2), (c.s2));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##3), (c.s3));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##4), (c.s4));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##5), (c.s5));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##6), (c.s6));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##7), (c.s7));     \
    })
#elif N0 == 16 // N0 == 16
#define ARM_DOT_K0XN0(k0, a, b, c) \
    ({                             \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##0), (c.s0));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##1), (c.s1));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##2), (c.s2));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##3), (c.s3));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##4), (c.s4));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##5), (c.s5));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##6), (c.s6));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##7), (c.s7));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##8), (c.s8));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##9), (c.s9));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##A), (c.sA));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##B), (c.sB));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##C), (c.sC));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##D), (c.sD));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##E), (c.sE));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##F), (c.sF));     \
    })
#else // N0 not supported
#error "N0 value not supported"
#endif // N0 conditions



#define VFMA(a, b, c)     \
    ({                    \
        c = fma(a, b, c); \
    })

#if M0 == 1
#define VFMA_M0xN0(i, a, b, c)                                        \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
    })
#elif M0 == 2 // M0 == 2
#define VFMA_M0xN0(i, a, b, c)                                        \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
    })
#elif M0 == 3 // M0 == 3
#define VFMA_M0xN0(i, a, b, c)                                        \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
    })
#elif M0 == 4 // M0 == 4
#define VFMA_M0xN0(i, a, b, c)                                        \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
    })
#elif M0 == 5 // M0 == 5
#define VFMA_M0xN0(i, a, b, c)                                        \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##4).s##i), b, (c##4)); \
    })
#elif M0 == 6 // M0 == 6
#define VFMA_M0xN0(i, a, b, c)                                        \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##4).s##i), b, (c##4)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##5).s##i), b, (c##5)); \
    })
#elif M0 == 7 // M0 == 7
#define VFMA_M0xN0(i, a, b, c)                                        \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##4).s##i), b, (c##4)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##5).s##i), b, (c##5)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##6).s##i), b, (c##6)); \
    })
#elif M0 == 8 // M0 == 8
#define VFMA_M0xN0(i, a, b, c)                                        \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##4).s##i), b, (c##4)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##5).s##i), b, (c##5)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##6).s##i), b, (c##6)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##7).s##i), b, (c##7)); \
    })
#else // M0 not supported
#error "M0 not supported"
#endif // M0 not supported

#endif // defined(M0) && defined(N0) && defined(K0) && defined(H0) && defined(DATA_TYPE)

#if defined(M0) && defined(N0) && defined(K0) && defined(V0) && defined(H0) && defined(DATA_TYPE) && defined(DATA_TYPE_ACCUMULATOR)

#if defined(MIXED_PRECISION)
#if K0 == 2
#define ARM_DOT_K0(a, b, c) \
    ({                      \
        c += a.s0 * b.s0;   \
        c += a.s1 * b.s1;   \
    })
#elif K0 == 3 // K0 == 3
#define ARM_DOT_K0(a, b, c) \
    ({                      \
        c += a.s0 * b.s0;   \
        c += a.s1 * b.s1;   \
        c += a.s2 * b.s2;   \
    })
#elif K0 == 4 // K0 == 4
#define ARM_DOT_K0(a, b, c) \
    ({                      \
        c += a.s0 * b.s0;   \
        c += a.s1 * b.s1;   \
        c += a.s2 * b.s2;   \
        c += a.s3 * b.s3;   \
    })
#elif K0 == 8 // K0 == 8
#define ARM_DOT_K0(a, b, c) \
    ({                      \
        c += a.s0 * b.s0;   \
        c += a.s1 * b.s1;   \
        c += a.s2 * b.s2;   \
        c += a.s3 * b.s3;   \
        c += a.s4 * b.s4;   \
        c += a.s5 * b.s5;   \
        c += a.s6 * b.s6;   \
        c += a.s7 * b.s7;   \
    })
#elif K0 == 16 // K0 == 16
#define ARM_DOT_K0(a, b, c) \
    ({                      \
        c += a.s0 * b.s0;   \
        c += a.s1 * b.s1;   \
        c += a.s2 * b.s2;   \
        c += a.s3 * b.s3;   \
        c += a.s4 * b.s4;   \
        c += a.s5 * b.s5;   \
        c += a.s6 * b.s6;   \
        c += a.s7 * b.s7;   \
        c += a.s8 * b.s8;   \
        c += a.s9 * b.s9;   \
        c += a.sA * b.sA;   \
        c += a.sB * b.sB;   \
        c += a.sC * b.sC;   \
        c += a.sD * b.sD;   \
        c += a.sE * b.sE;   \
        c += a.sF * b.sF;   \
    })
#else // K0 not supported
#error "K0 value not supported"
#endif // K0 conditions
#else  // defined(MIXED_PRECISION)
#if K0 == 2
#define ARM_DOT_K0(a, b, c)     \
    ({                          \
        c = fma(a.s0, b.s0, c); \
        c = fma(a.s1, b.s1, c); \
    })
#elif K0 == 3 // K0 == 3
#define ARM_DOT_K0(a, b, c)     \
    ({                          \
        c = fma(a.s0, b.s0, c); \
        c = fma(a.s1, b.s1, c); \
        c = fma(a.s2, b.s2, c); \
    })
#elif K0 == 4 // K0 == 4
#define ARM_DOT_K0(a, b, c)     \
    ({                          \
        c = fma(a.s0, b.s0, c); \
        c = fma(a.s1, b.s1, c); \
        c = fma(a.s2, b.s2, c); \
        c = fma(a.s3, b.s3, c); \
    })
#elif K0 == 8 // K0 == 8
#define ARM_DOT_K0(a, b, c)     \
    ({                          \
        c = fma(a.s0, b.s0, c); \
        c = fma(a.s1, b.s1, c); \
        c = fma(a.s2, b.s2, c); \
        c = fma(a.s3, b.s3, c); \
        c = fma(a.s4, b.s4, c); \
        c = fma(a.s5, b.s5, c); \
        c = fma(a.s6, b.s6, c); \
        c = fma(a.s7, b.s7, c); \
    })
#elif K0 == 16 // K0 == 16
#define ARM_DOT_K0(a, b, c)     \
    ({                          \
        c = fma(a.s0, b.s0, c); \
        c = fma(a.s1, b.s1, c); \
        c = fma(a.s2, b.s2, c); \
        c = fma(a.s3, b.s3, c); \
        c = fma(a.s4, b.s4, c); \
        c = fma(a.s5, b.s5, c); \
        c = fma(a.s6, b.s6, c); \
        c = fma(a.s7, b.s7, c); \
        c = fma(a.s8, b.s8, c); \
        c = fma(a.s9, b.s9, c); \
        c = fma(a.sA, b.sA, c); \
        c = fma(a.sB, b.sB, c); \
        c = fma(a.sC, b.sC, c); \
        c = fma(a.sD, b.sD, c); \
        c = fma(a.sE, b.sE, c); \
        c = fma(a.sF, b.sF, c); \
    })
#else // K0 not supported
#error "K0 value not supported"
#endif // K0 conditions
#endif // defined(MIXED_PRECISION)

#if defined(ARM_DOT_K0XN0)
#undef ARM_DOT_K0XN0
#endif // defined(ARM_DOT_K0XN0)

#if N0 == 2
#define ARM_DOT_K0XN0(a, b, c)           \
    ({                                   \
        ARM_DOT_K0((a), (b##0), (c.s0)); \
        ARM_DOT_K0((a), (b##1), (c.s1)); \
    })
#elif N0 == 3 // N0 == 3
#define ARM_DOT_K0XN0(a, b, c)           \
    ({                                   \
        ARM_DOT_K0((a), (b##0), (c.s0)); \
        ARM_DOT_K0((a), (b##1), (c.s1)); \
        ARM_DOT_K0((a), (b##2), (c.s2)); \
    })
#elif N0 == 4 // N0 == 4
#define ARM_DOT_K0XN0(a, b, c)           \
    ({                                   \
        ARM_DOT_K0((a), (b##0), (c.s0)); \
        ARM_DOT_K0((a), (b##1), (c.s1)); \
        ARM_DOT_K0((a), (b##2), (c.s2)); \
        ARM_DOT_K0((a), (b##3), (c.s3)); \
    })
#elif N0 == 8 // N0 == 8
#define ARM_DOT_K0XN0(a, b, c)           \
    ({                                   \
        ARM_DOT_K0((a), (b##0), (c.s0)); \
        ARM_DOT_K0((a), (b##1), (c.s1)); \
        ARM_DOT_K0((a), (b##2), (c.s2)); \
        ARM_DOT_K0((a), (b##3), (c.s3)); \
        ARM_DOT_K0((a), (b##4), (c.s4)); \
        ARM_DOT_K0((a), (b##5), (c.s5)); \
        ARM_DOT_K0((a), (b##6), (c.s6)); \
        ARM_DOT_K0((a), (b##7), (c.s7)); \
    })
#elif N0 == 16 // N0 == 16
#define ARM_DOT_K0XN0(a, b, c)           \
    ({                                   \
        ARM_DOT_K0((a), (b##0), (c.s0)); \
        ARM_DOT_K0((a), (b##1), (c.s1)); \
        ARM_DOT_K0((a), (b##2), (c.s2)); \
        ARM_DOT_K0((a), (b##3), (c.s3)); \
        ARM_DOT_K0((a), (b##4), (c.s4)); \
        ARM_DOT_K0((a), (b##5), (c.s5)); \
        ARM_DOT_K0((a), (b##6), (c.s6)); \
        ARM_DOT_K0((a), (b##7), (c.s7)); \
        ARM_DOT_K0((a), (b##8), (c.s8)); \
        ARM_DOT_K0((a), (b##9), (c.s9)); \
        ARM_DOT_K0((a), (b##A), (c.sA)); \
        ARM_DOT_K0((a), (b##B), (c.sB)); \
        ARM_DOT_K0((a), (b##C), (c.sC)); \
        ARM_DOT_K0((a), (b##D), (c.sD)); \
        ARM_DOT_K0((a), (b##E), (c.sE)); \
        ARM_DOT_K0((a), (b##F), (c.sF)); \
    })
#else // N0 not supported
#error "N0 value not supported"
#endif // N0 conditions


#if defined(LHS_TRANSPOSE)

#define VTYPE(TYPE, SIZE) VEC_DATA_TYPE(TYPE, SIZE)

#if defined(MIXED_PRECISION)

#if(GPU_ARCH == GPU_ARCH_MIDGARD)
#define ARM_VFMA(N0, a, b, c) c += (CONVERT(a, VEC_DATA_TYPE(DATA_TYPE_ACCUMULATOR, N0))) * (CONVERT(b, VEC_DATA_TYPE(DATA_TYPE_ACCUMULATOR, N0)));
#else // GPU_ARCH == GPU_ARCH_MIDGARD
#define ARM_VFMA(N0, a, b, c) c = fma((CONVERT(a, VEC_DATA_TYPE(DATA_TYPE_ACCUMULATOR, N0))), (CONVERT(b, VEC_DATA_TYPE(DATA_TYPE_ACCUMULATOR, N0))), (c));
#endif // GPU_ARCH == GPU_ARCH_MIDGARD

#else // defined(MIXED_PRECISION

#if(GPU_ARCH == GPU_ARCH_MIDGARD)
#define ARM_VFMA(N0, a, b, c) c += (a) * (b);
#else // GPU_ARCH == GPU_ARCH_MIDGARD
#define ARM_VFMA(N0, a, b, c) c = fma((a), (b), (c));
#endif // GPU_ARCH == GPU_ARCH_MIDGARD

#endif // defined(MIXED_PRECISION)

#define ARM_VVM_T_NT_1xN0x1(N0, TYPE, a, b, C)         \
    ({                                                 \
        ARM_VFMA(N0, (VTYPE(TYPE, N0))(a), b, (C##0)); \
    })
#define ARM_VVM_T_NT_2xN0x1(N0, TYPE, a, b, C)            \
    ({                                                    \
        ARM_VFMA(N0, (VTYPE(TYPE, N0))(a.s0), b, (C##0)); \
        ARM_VFMA(N0, (VTYPE(TYPE, N0))(a.s1), b, (C##1)); \
    })
#define ARM_VVM_T_NT_3xN0x1(N0, TYPE, a, b, C)            \
    ({                                                    \
        ARM_VVM_T_NT_2xN0x1(N0, TYPE, a, b, C);           \
        ARM_VFMA(N0, (VTYPE(TYPE, N0))(a.s2), b, (C##2)); \
    })
#define ARM_VVM_T_NT_4xN0x1(N0, TYPE, a, b, C)            \
    ({                                                    \
        ARM_VVM_T_NT_3xN0x1(N0, TYPE, a, b, C);           \
        ARM_VFMA(N0, (VTYPE(TYPE, N0))(a.s3), b, (C##3)); \
    })
#define ARM_VVM_T_NT_8xN0x1(N0, TYPE, a, b, C)            \
    ({                                                    \
        ARM_VVM_T_NT_4xN0x1(N0, TYPE, a, b, C);           \
        ARM_VFMA(N0, (VTYPE(TYPE, N0))(a.s4), b, (C##4)); \
        ARM_VFMA(N0, (VTYPE(TYPE, N0))(a.s5), b, (C##5)); \
        ARM_VFMA(N0, (VTYPE(TYPE, N0))(a.s6), b, (C##6)); \
        ARM_VFMA(N0, (VTYPE(TYPE, N0))(a.s7), b, (C##7)); \
    })

// Factory macro for the column-vector (transposed) by row-vector (not transposed) multiplication. K0 = 1
// a is the column-vector (transposed)
// b is the row-vector (not transposed)
// C is the output matrix
// Lower case is a vector (a, b)
// Upper case is a matrix (C)
#define ARM_VVM_T_NT_M0xN0x1(M0, N0, TYPE, a, b, C) ARM_VVM_T_NT_##M0##xN0x1(N0, TYPE, a, b, C)

#define ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, A, B, C)             \
    ({                                                         \
        ARM_VVM_T_NT_M0xN0x1(M0, N0, TYPE, (A##0), (B##0), C); \
    })
#define ARM_MM_T_NT_M0xN0x2(M0, N0, TYPE, A, B, C)             \
    ({                                                         \
        ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, A, B, C);            \
        ARM_VVM_T_NT_M0xN0x1(M0, N0, TYPE, (A##1), (B##1), C); \
    })
#define ARM_MM_T_NT_M0xN0x3(M0, N0, TYPE, A, B, C)             \
    ({                                                         \
        ARM_MM_T_NT_M0xN0x2(M0, N0, TYPE, A, B, C);            \
        ARM_VVM_T_NT_M0xN0x1(M0, N0, TYPE, (A##2), (B##2), C); \
    })
#define ARM_MM_T_NT_M0xN0x4(M0, N0, TYPE, A, B, C)             \
    ({                                                         \
        ARM_MM_T_NT_M0xN0x3(M0, N0, TYPE, A, B, C);            \
        ARM_VVM_T_NT_M0xN0x1(M0, N0, TYPE, (A##3), (B##3), C); \
    })
#define ARM_MM_T_NT_M0xN0x8(M0, N0, TYPE, A, B, C)             \
    ({                                                         \
        ARM_MM_T_NT_M0xN0x4(M0, N0, TYPE, A, B, C);            \
        ARM_VVM_T_NT_M0xN0x1(M0, N0, TYPE, (A##4), (B##4), C); \
        ARM_VVM_T_NT_M0xN0x1(M0, N0, TYPE, (A##5), (B##5), C); \
        ARM_VVM_T_NT_M0xN0x1(M0, N0, TYPE, (A##6), (B##6), C); \
        ARM_VVM_T_NT_M0xN0x1(M0, N0, TYPE, (A##7), (B##7), C); \
    })
#define ARM_MM_T_NT_M0xN0x16(M0, N0, TYPE, A, B, C)           \
    ({                                                        \
        ARM_MM_T_NT_M0xN0x8(M0, N0, TYPE, A, B, C);           \
        ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, (A##8), (B##8), C); \
        ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, (A##9), (B##9), C); \
        ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, (A##A), (B##A), C); \
        ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, (A##B), (B##B), C); \
        ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, (A##C), (B##C), C); \
        ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, (A##D), (B##D), C); \
        ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, (A##E), (B##E), C); \
        ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, (A##F), (B##F), C); \
    })

// Factory macro for the matrix (transposed) by matrix (not transposed) multiplication.
// The dimensions for this matrix multiplications are defined through M0, N0 and K0
// The dimensions supported are:
// M0: 1, 2, 3, 4, 8
// N0: 1, 2, 3, 4, 8, 16
// K0: 1, 2, 3, 4, 8, 16
// This macro calls the vector-by-matrix macro K0 times
// A, B and C are matrices
#define ARM_MM_T_NT(M0, N0, K0, TYPE, A, B, C) \
    CONCAT(ARM_MM_T_NT_M0xN0x, K0)             \
    (M0, N0, TYPE, A, B, C)

#endif // defined(LHS_TRANSPOSE)

#endif // defined(M0) && defined(N0) && defined(K0) && defined(V0) && defined(H0) && defined(DATA_TYPE) && defined(DATA_TYPE_ACCUMULATOR)

#if defined(M0) && defined(N0) && defined(K0) && defined(DATA_TYPE)

#define VFMA(a, b, c)     \
    ({                    \
        c = fma(a, b, c); \
    })

#if M0 == 1
#define RHS_VFMA_M0xN0(i, a, b, c)                                    \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
    })
#elif M0 == 2 // M0 == 2
#define RHS_VFMA_M0xN0(i, a, b, c)                                    \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
    })
#elif M0 == 3 // M0 == 3
#define RHS_VFMA_M0xN0(i, a, b, c)                                    \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
    })
#elif M0 == 4 // M0 == 4
#define RHS_VFMA_M0xN0(i, a, b, c)                                    \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
    })
#elif M0 == 5 // M0 == 5
#define RHS_VFMA_M0xN0(i, a, b, c)                                    \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##4).s##i), b, (c##4)); \
    })
#elif M0 == 6 // M0 == 6
#define RHS_VFMA_M0xN0(i, a, b, c)                                    \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##4).s##i), b, (c##4)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##5).s##i), b, (c##5)); \
    })
#elif M0 == 7 // M0 == 7
#define RHS_VFMA_M0xN0(i, a, b, c)                                    \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##4).s##i), b, (c##4)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##5).s##i), b, (c##5)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##6).s##i), b, (c##6)); \
    })
#elif M0 == 8 // M0 == 8
#define RHS_VFMA_M0xN0(i, a, b, c)                                    \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##4).s##i), b, (c##4)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##5).s##i), b, (c##5)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##6).s##i), b, (c##6)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##7).s##i), b, (c##7)); \
    })
#else // M0 not supported
#error "M0 not supported"
#endif // M0 not supported

#if defined(GEMM_MM_NATIVE)
/** This OpenCL kernel computes the matrix multiplication between 2 matrices.
 *  The LHS matrix is NOT reshaped
 *  The RHS matrix is NOT reshaped
 *
 * @note If the first two dimensions of NDRange have been dispatched with "dummy_work_items" support, the option -DDUMMY_WORK_ITEMS must be passed at compile time.
 * @note The GEMM's dimensions (M,N and K) must be passed at runtime as kernel parameters.
 * @note The number of M0 rows to process must be passed at compile time using -DM0 (e.g. -DM0=2)
 * @note The number of K0 partial accumulations must be passed at compile time using -DK0 (e.g., -DK0=2)
 * @note The number of N0 columns to process must be passed at compile time using -DN0 (e.g. -DN0=2)
 * @note The size of the partial store block in y must be passed at compile time using -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_M0=1)
 * @note The size of the partial store block in x must be passed at compile time using -DPARTIAL_STORE_N0 (e.g. -DPARTIAL_STORE_N0=1)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 1, 2, 3, 4, 5, 6, 7, 8
 *  - N0 = 2, 3, 4, 8, 16
 *  - K0 = 2, 3, 4, 8, 16
 *
 * @note If the activation type were passed at compile time through -DACTIVATION_TYPE (e.g. -DACTIVATION_TYPE=RELU), A, B variables, required by some activation functions, should be passed at compile time as well using -DA_VAL= and -DB_VAL= respectively.
 *       The activation function is performed after the bias addition
 * @note In case the input or output have to be reinterpreted as a 3D tensor, the following information must be passed at compile time:
 *       -# REINTERPRET_INPUT_AS_3D: To reinterpret the input as 3D
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns LHS matrix
 *
 * @param[in]  lhs_ptr                            Pointer to the LHS matrix. Supported data type: F16/F32
 * @param[in]  lhs_stride_x                       Stride of the LHS matrix in X dimension (in bytes)
 * @param[in]  lhs_step_x                         lhs_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  lhs_stride_y                       Stride of the LHS matrix in Y dimension (in bytes)
 * @param[in]  lhs_step_y                         lhs_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  lhs_offset_first_element_in_bytes  The offset of the first element in the LHS matrix
 * @param[in]  rhs_ptr                            Pointer to the RHS matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  rhs_stride_x                       Stride of the RHS matrix in X dimension (in bytes)
 * @param[in]  rhs_step_x                         rhs_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  rhs_stride_y                       Stride of the RHS matrix in Y dimension (in bytes)
 * @param[in]  rhs_step_y                         rhs_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  rhs_offset_first_element_in_bytes  The offset of the first element in the RHS matrix
 * @param[in]  bias_ptr                           (Optional) Pointer to the bias matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  bias_stride_x                      (Optional) Stride of the bias matrix in X dimension (in bytes)
 * @param[in]  bias_step_x                        (Optional) bias_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  bias_stride_y                      (Optional) Stride of the bias matrix in Y dimension (in bytes)
 * @param[in]  bias_step_y                        (Optional) bias_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  bias_offset_first_element_in_bytes (Optional) The offset of the first element in the bias matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data type: same as @p lhs_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  lhs_stride_z                       Stride of the LHS matrix in Z dimension (in bytes)
 * @param[in]  rhs_stride_z                       Stride of the RHS matrix in Z dimension (in bytes)
 * @param[in]  bias_stride_z                      (Optional) Stride of the bias matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  M                                  Number of rows in LHS matrix not reshaped.
 * @param[in]  N                                  Number of columns in RHS matrix not reshaped.
 * @param[in]  K                                  Number of columns in LHS matrix and rows in RHS matrix not reshaped.
 * @param[in]  lhs_cross_plane_pad                (Optional) Bottom paddings for LHS matrix in unit of elements (only if defined REINTERPRET_INPUT_AS_3D)
 * @param[in]  dst_cross_plane_pad                (Optional) Bottom paddings for the output matrix in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_native(IMAGE_DECLARATION(lhs),
                             IMAGE_DECLARATION(rhs),
#if defined(BETA)
                             IMAGE_DECLARATION(bias),
#endif // defined(BETA)
                             IMAGE_DECLARATION(dst),
                             uint lhs_stride_z,
                             uint rhs_stride_z,
#if defined(BETA)
                             uint bias_stride_z,
#endif //defined(BETA)
                             uint      dst_stride_z,
                             const int M,
                             const int N,
                             const int K
#if defined(REINTERPRET_INPUT_AS_3D)
                             ,
                             uint lhs_cross_plane_pad
#endif // REINTERPRET_INPUT_AS_3D
#if defined(REINTERPRET_OUTPUT_AS_3D)
                             ,
                             uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                            )
{
    // Block size
#define RHS_BLOCK_SIZE ((K0) * (N0))

    // RHS offset and step X
#define RHS_OFFSET_X (RHS_BLOCK_SIZE)

    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint z = get_global_id(2);

#if defined(DUMMY_WORK_ITEMS)
    if((x * N0 >= N) || (y * M0 >= M))
    {
        return;
    }
#endif // defined(DUMMY_WORK_ITEMS)

    // Compute LHS matrix address
    uint lhs_offset = lhs_offset_first_element_in_bytes + COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * (uint)lhs_stride_y;

    // Compute RHS matrix address
    uint rhs_offset = rhs_offset_first_element_in_bytes + x * N0 * sizeof(DATA_TYPE);

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    rhs_offset += (z % MATRIX_B_DEPTH) * rhs_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    rhs_offset += z * rhs_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    REPEAT_VAR_INIT_TO_CONST(M0, uint, zlhs, 0);
    REPEAT_VAR_INIT_TO_CONST(16, uint, zero, 0);

#if defined(REINTERPRET_INPUT_AS_3D)
    // The plane (zlhs) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zlhs, COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0), HEIGHT_GEMM3D, DEPTH_GEMM3D, lhs_cross_plane_pad, lhs_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply lhs_stride_z by DEPTH_GEMM3D
    lhs_offset += z * lhs_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_INPUT_AS_3D)

    // Add offset for batched GEMM
    lhs_offset += z * lhs_stride_z;

#endif // defined(REINTERPRET_INPUT_AS_3D)

    // Initialize the accumulators
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(DATA_TYPE, N0), c, 0); //VEC_DATA_TYPE(DATA_TYPE, N0)    c0=0,c1=0,c2=0,... c(M0-1)=0;

    int i = 0;
#if K0 > 1
    for(; i <= (K - K0); i += K0)
    {
        // Supported cases (M0, K0):
        // 1,2 - 1,3 - 1,4 - 1,8 - 1,16
        // 2,2 - 2,3 - 2,4 - 2,8 - 2,16
        // 3,2 - 3,3 - 3,4 - 3,8 - 3,16
        // 4,2 - 4,3 - 4,4 - 4,8 - 4,16
        // 5,2 - 5,3 - 5,4 - 5,8 - 5,16
        // 6,2 - 6,3 - 6,4 - 6,8 - 6,16
        // 7,2 - 7,3 - 7,4 - 7,8 - 7,16
        // 8,2 - 8,3 - 8,4 - 8,8 - 8,16
        // Load values from LHS matrix
        LOAD_BLOCK(M0, K0, DATA_TYPE, a, lhs_ptr, lhs_offset, lhs_stride_y, zlhs);

        // Load values from RHS matrix
        LOAD_BLOCK(K0, N0, DATA_TYPE, b, rhs_ptr, rhs_offset, rhs_stride_y, zero);

        RHS_VFMA_M0xN0(0, a, b0, c);
        RHS_VFMA_M0xN0(1, a, b1, c);
#if K0 > 2
        RHS_VFMA_M0xN0(2, a, b2, c);
#endif // K0 > 2
#if K0 > 3
        RHS_VFMA_M0xN0(3, a, b3, c);
#endif // K0 > 3
#if K0 > 4
        RHS_VFMA_M0xN0(4, a, b4, c);
        RHS_VFMA_M0xN0(5, a, b5, c);
        RHS_VFMA_M0xN0(6, a, b6, c);
        RHS_VFMA_M0xN0(7, a, b7, c);
#endif // K0 > 4
#if K0 > 8
        RHS_VFMA_M0xN0(8, a, b8, c);
        RHS_VFMA_M0xN0(9, a, b9, c);
        RHS_VFMA_M0xN0(A, a, bA, c);
        RHS_VFMA_M0xN0(B, a, bB, c);
        RHS_VFMA_M0xN0(C, a, bC, c);
        RHS_VFMA_M0xN0(D, a, bD, c);
        RHS_VFMA_M0xN0(E, a, bE, c);
        RHS_VFMA_M0xN0(F, a, bF, c);
#endif // K0 > 8

        lhs_offset += K0 * sizeof(DATA_TYPE);
        rhs_offset += K0 * rhs_stride_y;
    }
#endif // K0 > 1
    // Left-over accumulations
    for(; i < K; ++i)
    {
        // Load values from LHS matrix
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a0 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 0 * lhs_stride_y + zlhs0));
#if M0 > 1
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a1 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 1 * lhs_stride_y + zlhs1));
#endif // M0 > 1
#if M0 > 2
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a2 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 2 * lhs_stride_y + zlhs2));
#endif // M0 > 2
#if M0 > 3
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a3 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 3 * lhs_stride_y + zlhs3));
#endif // M0 > 3
#if M0 > 4
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a4 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 4 * lhs_stride_y + zlhs4));
#endif // M0 > 4
#if M0 > 5
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a5 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 5 * lhs_stride_y + zlhs5));
#endif // M0 > 5
#if M0 > 6
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a6 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 6 * lhs_stride_y + zlhs6));
#endif // M0 > 6
#if M0 > 7
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a7 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 7 * lhs_stride_y + zlhs7));
#endif // M0 > 7

        VEC_DATA_TYPE(DATA_TYPE, N0)
        b = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 0 * rhs_stride_y));
        RHS_VFMA_M0xN0(0, a, b, c);

        lhs_offset += sizeof(DATA_TYPE);
        rhs_offset += rhs_stride_y;
    }

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE)) + (COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * dst_stride_y);

    REPEAT_VAR_INIT_TO_CONST(M0, uint, zout, 0);

#if defined(REINTERPRET_OUTPUT_AS_3D)
    // The plane (zout) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zout, COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0), HEIGHT_GEMM3D, DEPTH_GEMM3D, dst_cross_plane_pad, dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_OUTPUT_AS_3D)

    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;

#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    SCALE_BLOCK(M0, DATA_TYPE, c, ALPHA);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
#if defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE));

    LOAD_BLOCK(1, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(1, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
    ADD_BLOCK_BROADCAST(M0, c, bias0);

#else // defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE)) + (COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * bias_stride_y) + z * bias_stride_z;

    LOAD_BLOCK(M0, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(M0, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias
    ADD_BLOCK(M0, c, bias);

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

#if defined(ACTIVATION_TYPE)
    ACTIVATION_BLOCK(M0, ACTIVATION_TYPE, DATA_TYPE, N0, c, A_VAL, B_VAL);
#endif // defined(ACTIVATION_TYPE)

    const bool cond_y = y == 0;
    const bool cond_x = ((x + 1) * N0 >= N);

    // Store output block
    STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, c, dst_addr, dst_stride_y, zout, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);
}
#endif // defined(GEMM_MM_NATIVE)
#endif // defined(M0) && defined(N0) && defined(K0) && defined(DATA_TYPE)

#if defined(BETA)
/** This OpenCL kernel performs the in-place matrix addition between 2 matrices taking into account that the second matrix might be weighted by a scalar value beta:
 *
 * @note The beta's value need to be passed at compile time using -DBETA
 *
 * @param[in]  src_ptr                           Pointer to the source matrix. Supported data types: F32
 * @param[in]  src_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination matrix
 */
__kernel void gemm_ma_f32(TENSOR3D_DECLARATION(src),
                          TENSOR3D_DECLARATION(dst))
{
    // Compute source and destination addresses
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT(dst);

    // Load values from A x B
    float4 alpha_ab = vload4(0, (__global float *)dst.ptr);

    // Load values from Matrix C
    float4 c = vload4(0, (__global float *)src.ptr);

    // Computes alpha * axb + beta * c
    float4 out = alpha_ab + (float4)BETA * c;

    // Store final result in axb matrix
    vstore4(out, 0, (__global float *)dst.ptr);
}

#endif // defined(BETA)
