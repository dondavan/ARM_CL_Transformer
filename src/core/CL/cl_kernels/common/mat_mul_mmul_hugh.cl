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
                dst[_y*HEIGHT + _x] = *(__global DATA_TYPE *)(TENSOR##_ptr + TENSOR##_offset_first_element_in_bytes + (X+_x) * sizeof(DATA_TYPE) + (Y+_y) * (STRIDE_Y));      \
            })                                                                                                                  \
        })                                                                                                                  \
    }
    
#define HUGH_2D_ACCESS(BASENAME,X,Y,WIDTH) BASENAME[Y*WIDTH+X]

#if defined(MAT_MUL_MMUL_HUGH)
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
__kernel void mat_mul_mmul_hugh(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, RHS_TENSOR_TYPE),
#ifdef BIAS
    TENSOR3D_T(bias, BUFFER),
#endif // defined(BIAS)
    TENSOR3D_T(dst, BUFFER))
{
    uint x = GET_SPATIAL_IDX(0, N0, PARTIAL_STORE_N0);
    uint y = GET_SPATIAL_IDX(1, M0, PARTIAL_STORE_M0);
    uint z = GET_SPATIAL_IDX(2, 1, 0);
    
    // Compute LHS/RHS/DST matrix address
    lhs_offset_first_element_in_bytes += y * lhs_stride_y + z * lhs_stride_z;
    dst_offset_first_element_in_bytes += x * sizeof(DATA_TYPE) + y * dst_stride_y + z * dst_stride_z;

    // Initialize the accumulators
    TILE(DATA_TYPE, M0, N0, ret);
    TILE(DATA_TYPE, M0, K0, acc);


    HUGH_2D(DATA_TYPE, M0, N0, shabi);
    T_LOAD_HUGH(DATA_TYPE, M0, N0, BUFFER, lhs, 0, 0, 1, lhs_stride_y, shabi);
    //TILE(DATA_TYPE, M0, N0, shabi);
    //T_LOAD(DATA_TYPE, M0, N0, BUFFER, lhs, 0, 0, 1, lhs_stride_y, shabi);

    for(int _m = 0; _m < M0; _m++)
    {
        ret[_m].s[0] = 0.f;
        ret[_m].s[1] = 0.f;
        acc[_m].v = 0.f;
    }

    const int rhs_z = z * rhs_h;
    int       k;
    for(k = 0; k <= K - K0; k += K0)
    {
        
        TILE(DATA_TYPE, M0, K0, a);
        TILE(DATA_TYPE, N0, K0, b);

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, M0, K0, BUFFER, lhs, k, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, N0, K0, RHS_TENSOR_TYPE, rhs, k, x + rhs_z, 1, rhs_stride_y, b);

        for(int _m = 0; _m < M0; _m++)
        {
            a[_m].s[0] = a[_m].v.s0;
            a[_m].s[1] = a[_m].v.s1;
            a[_m].s[2] = a[_m].v.s2;
            a[_m].s[3] = a[_m].v.s3;

            a[_m].s[4] = a[_m].v.s4;
            a[_m].s[5] = a[_m].v.s5;
            a[_m].s[6] = a[_m].v.s6;
            a[_m].s[7] = a[_m].v.s7;
        }

        for(int _n = 0; _n < N0; _n++)
        {
            b[_n].s[0] = b[_n].v.s0;
            b[_n].s[1] = b[_n].v.s1;
            b[_n].s[2] = b[_n].v.s2;
            b[_n].s[3] = b[_n].v.s3;

            b[_n].s[4] = b[_n].v.s4;
            b[_n].s[5] = b[_n].v.s5;
            b[_n].s[6] = b[_n].v.s6;
            b[_n].s[7] = b[_n].v.s7;
        }

        LOOP_UNROLLING(int, _m, 0, 1, M0,
        {

            acc[_m].v = fma((DATA_TYPE)(a[_m].v), (DATA_TYPE)(b[0].v), acc[_m].v);
            ret[_m].s[0] = SUM_REDUCE(acc[_m].v,K0);

            acc[_m].v = fma((DATA_TYPE)(a[_m].v), (DATA_TYPE)(b[0].v), acc[_m].v);
            ret[_m].s[0] = SUM_REDUCE(acc[_m].v,K0);
            
            ret[_m].s[1] = fma((DATA_TYPE)(a[_m].s[0]), (DATA_TYPE)(b[1].s[0]), ret[_m].s[1]);
            ret[_m].s[1] = fma((DATA_TYPE)(a[_m].s[1]), (DATA_TYPE)(b[1].s[1]), ret[_m].s[1]);
            ret[_m].s[1] = fma((DATA_TYPE)(a[_m].s[2]), (DATA_TYPE)(b[1].s[2]), ret[_m].s[1]);
            ret[_m].s[1] = fma((DATA_TYPE)(a[_m].s[3]), (DATA_TYPE)(b[1].s[3]), ret[_m].s[1]);
            ret[_m].s[1] = fma((DATA_TYPE)(a[_m].s[4]), (DATA_TYPE)(b[1].s[4]), ret[_m].s[1]);
            ret[_m].s[1] = fma((DATA_TYPE)(a[_m].s[5]), (DATA_TYPE)(b[1].s[5]), ret[_m].s[1]);
            ret[_m].s[1] = fma((DATA_TYPE)(a[_m].s[6]), (DATA_TYPE)(b[1].s[6]), ret[_m].s[1]);
            ret[_m].s[1] = fma((DATA_TYPE)(a[_m].s[7]), (DATA_TYPE)(b[1].s[7]), ret[_m].s[1]);

        }) 
        
        
        //lhs_offset_first_element_in_bytes += K0 * sizeof(DATA_TYPE);
    }

    const bool x_cond = PARTIAL_STORE_N0 != 0 && get_global_id(0) == 0;
    const bool y_cond = PARTIAL_STORE_M0 != 0 && get_global_id(1) == 0;

    TILE(int, M0, 1, indirect_buffer);
    LOOP_UNROLLING(int, _i, 0, 1, M0,
    {
        indirect_buffer[_i].v = min(_i, select(M0 - 1, PARTIAL_STORE_M0 - 1, y_cond));
    });

#ifdef BIAS
    TILE(DATA_TYPE, 1, N0, bias_tile);

    // below expands to use bias_ptr and bias_offset_first_element_in_bytes
    T_LOAD(DATA_TYPE, 1, N0, BUFFER, bias, x, 0, 1, 0, bias_tile);

    bias_tile[0].s[0] = bias_tile[0].v.s0;
    bias_tile[0].s[1] = bias_tile[0].v.s1;

    LOOP_UNROLLING(int, _m, 0, 1, M0,
    {
        ret[_m].s[0] += bias_tile[0].s[0];
        ret[_m].s[1] += bias_tile[0].s[1];
    }) 
#endif // defined(BIAS)

     LOOP_UNROLLING(int, _m, 0, 1, M0,
    {
        ret[_m].v.s0 = HUGH_2D_ACCESS(shabi, 0, _m, N0); //ret[_m].s[0]; HUGH_2D_ACCESS(shabi, 0,_m, N0);
        ret[_m].v.s1 = HUGH_2D_ACCESS(shabi, 1, _m, N0);
    }) 
    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, M0, N0, PARTIAL_STORE_N0, BUFFER, dst, 0, dst_stride_y, x_cond, ret, indirect_buffer);
   
}
#endif // defined(MAT_MUL_MMUL_HUGH)