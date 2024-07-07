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
inline void perform_bias_addition(uchar *bias_ptr, uint bias_offset_first_element_in_bytes, TILE(DATA_TYPE, M0, N0, acc), uint x)
{
    TILE(DATA_TYPE, 1, N0, bias_tile);

    // below expands to use bias_ptr and bias_offset_first_element_in_bytes
    T_LOAD(DATA_TYPE, 1, N0, BUFFER, bias, x, 0, 1, 0, bias_tile);

    // c = c + bias[broadcasted]
    T_ELTWISE_BROADCAST_ADD_X(DATA_TYPE, M0, N0, acc, bias_tile, acc);
}
#endif // defined(BIAS)

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
    //lhs_offset_first_element_in_bytes += y * lhs_stride_y + z * lhs_stride_z;
    dst_offset_first_element_in_bytes += x * sizeof(DATA_TYPE) + y * dst_stride_y + z * dst_stride_z;

    // Initialize the accumulators
    //TILE(DATA_TYPE, M0, N0, acc);
    /*
    union acc_private{
        DATA_TYPE s[2];
        float2 v;
    };
    union acc_private acc[M0];

    */
    DATA_TYPE acc_s[M0][N0];
    float2 acc_v[M0];

    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        acc_v[i] = (float)x;
    })

    uint rhs_z = z * rhs_h;
    uint       k;
    for(k = 0; k <= K - K0; k += K0)
    {

        TILE(DATA_TYPE, M0, K0, a);
        TILE(DATA_TYPE, N0, K0, b);

        LOOP_UNROLLING(int, i, 0, 1, M0,
        {
            a[i].v = 0.f;
        })

        LOOP_UNROLLING(int, i, 0, 1, N0,
        {
            b[i].v = 0.f;
        })

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, M0, K0, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, N0, K0, RHS_TENSOR_TYPE, rhs, k, x + rhs_z, 1, rhs_stride_y, b);

        //(DATA_TYPE, DATA_TYPE, DATA_TYPE, M0, N0, K0, NT, NT, a, b, acc);
        
        LOOP_UNROLLING(int, _m, 0, 1, M0,
        {
            LOOP_UNROLLING(int, _k, 0, 1, K0,
            {
                acc_v[_m] = fma((DATA_TYPE)(a[_m].s[_k]), (b[_k].v), acc_v[_m]);
            })
        })
        

        lhs_offset_first_element_in_bytes += K0 * sizeof(DATA_TYPE);
    }
    

    const bool x_cond = PARTIAL_STORE_N0 != 0 && get_global_id(0) == 0;
    const bool y_cond = PARTIAL_STORE_M0 != 0 && get_global_id(1) == 0;

    TILE(int, M0, 1, indirect_buffer);
    LOOP_UNROLLING(int, _i, 0, 1, M0,
    {
        indirect_buffer[_i].v = min(_i, select(M0 - 1, PARTIAL_STORE_M0 - 1, y_cond));
    });


    //T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, M0, N0, PARTIAL_STORE_N0, BUFFER, dst, 0, dst_stride_y, x_cond, acc, indirect_buffer);
    //T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, HEIGHT, WIDTH0, WIDTH1, TENSOR_TYPE, TENSOR, X, STRIDE_Y, WIDTH1_CONDITION, src, indirect_y)
    
    if(x_cond)
    {
        LOOP_UNROLLING(int, _i, 0, 1, M0,
        {
            VSTORE_PARTIAL(N0, PARTIAL_STORE_N0)(CONVERT(acc_v[M0 - 1 - _i], VEC_DATA_TYPE(DATA_TYPE, N0)), 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + 0 * sizeof(DATA_TYPE) + (indirect_buffer[M0 - 1 - _i].v) * dst_stride_y));
        })
    }
    else
    {
        LOOP_UNROLLING(int, _i, 0, 1, M0,
        {
            VSTORE(N0)(CONVERT(acc_v[M0 - 1 - _i], VEC_DATA_TYPE(DATA_TYPE, N0)), 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + (indirect_buffer[M0 - 1 - _i].v) * dst_stride_y));
        })
    } 
    

}
#endif // defined(MAT_MUL_MMUL_HUGH)