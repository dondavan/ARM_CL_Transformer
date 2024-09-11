/*
 * Copyright (c) 2023 Arm Limited.
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
#include "helpers.h"
#include "tile_helpers.h"

#ifdef BIAS
// This function performs in-place bias addition for float and half datatypes when bias is enabled.
// Note The tile's dimensions used for the LHS and RHS matrices (M0, N0) must be passed at compile time using -DN0, -DM0 (e.g. -DN0=8, -DM0=4).
inline void perform_bias_addition(uchar *bias_ptr, uint bias_offset_first_element_in_bytes,  uint x)
{
    TILE(DATA_TYPE, 1, N0, bias_tile);

    // below expands to use bias_ptr and bias_offset_first_element_in_bytes
    T_LOAD(DATA_TYPE, 1, N0, BUFFER, bias, x, 0, 1, 0, bias_tile);

    // c = c + bias[broadcasted]
    //T_ELTWISE_BROADCAST_ADD_X(DATA_TYPE, M0, N0, ret, bias_tile, ret);
}
#endif // defined(BIAS)


#define HUGH_2D(DATA_TYPE, H, W, BASENAME) HUGH_2D_STR(DATA_TYPE, H, W, BASENAME)
#define HUGH_2D_STR(DATA_TYPE, H, W, BASENAME) DATA_TYPE BASENAME[W * H]

#define T_LOAD_HUGH(DATA_TYPE, HEIGHT, WIDTH, TENSOR_TYPE, TENSOR, X, Y, YI_MULTIPLIER, STRIDE_Y, dst)                      \
    {                                                                                                                       \
        LOOP_UNROLLING(int, _y, 0, 1, HEIGHT,                                                                               \
        {                                                                                                                   \
            LOOP_UNROLLING(int, _x, 0, 1, WIDTH,                                                                               \
            {                                                                                                                   \
                dst[_y*WIDTH + _x] = *(__global DATA_TYPE *)(TENSOR##_ptr + TENSOR##_offset_first_element_in_bytes + (X+_x) * sizeof(DATA_TYPE) + (Y+_y) * (STRIDE_Y));      \
            })                                                                                                                  \
        })                                                                                                                  \
    }

#define V_LOAD_HUGH(DATA_TYPE, WIDTH, TENSOR, X, Y, STRIDE_Y) V_LOAD_HUGH_STR(DATA_TYPE, WIDTH, TENSOR, X, Y, STRIDE_Y)
#define V_LOAD_HUGH_STR(DATA_TYPE, WIDTH, TENSOR, X, Y, STRIDE_Y) \
VLOAD(WIDTH)(0, (DATA_TYPE *)(TENSOR + (X) + (Y) * (STRIDE_Y)))

    
#define HUGH_2D_ACCESS(BASENAME,Y,X,WIDTH) BASENAME[Y*WIDTH+X]

#if defined(MAT_MUL_MMUL_HUGH_NT_T)
/** This OpenCL kernel performs the batch matrix multiplication (BatchMatMul) using MMUL: LHS non-transposed, RHS non-transposed - buffer only
 *
 * @note the "batch" here expresses the number of matrix multiplications to run in parallel. However, it
 *       should NOT be confused with the batch size of the model. For NHWC the "batch" is the "H" dimension
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=float)
 * @note The tile's dimensions used for the LHS and RHS matrices (M0, N0 and K0) must be passed at compile time using -DN0, -DM0 and -DK0 (e.g. -DN0=8, -DM0=4, -DK0=1).
 * @note The number of leftover outputs rows/columns must be passed using -DN0_LEFTOVER and -DM0_LEFTOVER (e.g. -DN0_LEFTOVER=2, -DM0_LEFTOVER=3)
 * @note The MMUL block dimension (MMUL_M0, MMUL_N0, MMUL_K0) must be passed at compile time using -DMMUL_M0, -DMMUL_N0 and -DMMUL_K0 (e.g. -DMMUL_M0=4, -DMMUL_N0=4, -DMMUL_K0=4).
 * @note The kernel name in uppercase must be passed at compile time (e.g. -DMAT_MUL_NATIVE_MMUL_NT_NT)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 > 0
 *  - N0 = 1, 2, 3, 4, 8, 16
 *  - K0 = 1
 * @note Values > 8 for M0 are not expected to be efficient
 *
 * @param[in]  lhs_ptr                            Pointer to the lhs matrix. Supported data types: F32/F16
 * @param[in]  lhs_stride_y                       Stride of the lhs matrix in Y (2nd) dimension (in bytes)
 * @param[in]  lhs_stride_z                       Stride of the lhs tensor in Z (3rd) dimension (in bytes)
 * @param[in]  lhs_w                              The width of the lhs tensor
 * @param[in]  lhs_h                              The height of the lhs tensor
 * @param[in]  lhs_n                              Number of the matrices (buffers) in the batch
 * @param[in]  lhs_offset_first_element_in_bytes  The offset of the first element in the lhs matrix
 * @param[in]  rhs_ptr                            Pointer to the rhs matrix. Supported data types: same as @p lhs_ptr
 * @param[in]  rhs_stride_y                       Stride of the rhs matrix in Y (2nd) dimension (in bytes)
 * @param[in]  rhs_stride_z                       Stride of the rhs tensor in Z (3rd) dimension (in bytes)
 * @param[in]  rhs_w                              The width of the rhs tensor
 * @param[in]  rhs_h                              The height of the rhs tensor
 * @param[in]  rhs_n                              Number of the matrices (buffers) in the batch
 * @param[in]  rhs_offset_first_element_in_bytes  The offset of the first element in the rhs matrix
 * @param[in]  bias_ptr                           (Optional) Pointer to the bias tensor. Supported data type: same as @p lhs_ptr
 * @param[in]  bias_stride_y                      (Optional) Stride of the bias tensor in Y dimension (in bytes)
 * @param[in]  bias_stride_z                      (Optional) Stride of the bias tensor in Z dimension (in bytes)
 * @param[in]  bias_w                             (Optional) The size of the width dimension of the bias tensor
 * @param[in]  bias_h                             (Optional) The size of the height dimension of the bias tensor
 * @param[in]  bias_n                             (Optional) The size of the depth dimension of the bias tensor
 * @param[in]  bias_offset_first_element_in_bytes (Optional) The offset of the first element in the bias tensor
 * @param[out] dst_ptr                            Pointer to the dst matrix. Supported data types: same as @p lhs_ptr
 * @param[in]  dst_stride_y                       Stride of the dst matrix in Y (2nd) dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the dst tensor in Z (3rd) dimension (in bytes)
 * @param[in]  dst_w                              The width of the dst tensor
 * @param[in]  dst_h                              The height of the dst tensor
 * @param[in]  dst_n                              Number of the matrices (buffers) in the batch
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the dst matrix
 * @param[in]  M                                  Number of rows in LHS matrix
 * @param[in]  N                                  Number of columns in RHS matrix
 * @param[in]  K                                  Number of columns in LHS matrix and rows in RHS matrix, which is multiple of MMUL_K0.
 */
 //mat_mul_native_mmul_nt_nt
__kernel void mat_mul_mmul_hugh_nt_t(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, RHS_TENSOR_TYPE),
#ifdef BIAS
    TENSOR3D_T(bias, BUFFER),
#endif // defined(BIAS)
    TENSOR3D_T(dst, BUFFER))
{
   
}
#endif // defined(MAT_MUL_MMUL_HUGH_NT_T)


#if defined(MAT_MUL_MMUL_HUGH_NT_NT)
__kernel void mat_mul_mmul_hugh_nt_nt(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, RHS_TENSOR_TYPE),
#ifdef BIAS
    TENSOR3D_T(bias, BUFFER),
#endif // defined(BIAS)
    TENSOR3D_T(dst, BUFFER))
{
}
#endif // defined(MAT_MUL_MMUL_HUGH_NT_NT)