#include "activation_float_helpers.h"
#include "helpers.h"
#include "tile_helpers.h"

#if defined(LINEAR)

/** This OpenCL kernel performs the batch matrix multiplication (BatchMatMul): LHS non-transposed, RHS non-transposed - buffer only
 *
 * @note the "batch" here expresses the number of matrix multiplications to run in parallel. However, it
 *       should NOT be confused with the batch size of the model. For NHWC the "batch" is the "H" dimension
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=float)
 * @note The block's dimensions used for the LHS and RHS matrices (M0, N0 and K0) must be passed at compile time using -DN0, -DM0 and -DK0 (e.g. -DN0=8, -DM0=4, -DK0=4).
 * @note The fused activation function used should be passed with -DACTIVATION_TYPE, -DA_VAL and -DB_VAL are used for min and max output bounded activation functions.
 * @note The number of leftover outputs rows/columns must be passed using -DPARTIAL_STORE_N0 and -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_N0=2, -DPARTIAL_STORE_M0=3)
 * @note The dimension K must be passed at compile time using -DK (e.g. -DK=6)
 * @note The tensor type ("BUFFER" or "IMAGE") of the rhs tensor must be passed at compile time using -DRHS_TENSOR_TYPE (e.g. -DRHS_TENSOR_TYPE=BUFFER)
 * @note The kernel name in uppercase must be passed at compile time (e.g. -DMAT_MUL_NATIVE_NT_NT)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 > 0
 *  - N0 = 1, 2, 3, 4, 8, 16 (only 4, 8, 16 if RHS_TENSOR_TYPE=IMAGE)
 *  - K0 = 1, 2, 3, 4, 8, 16
 * @note Values > 8 for M0 are not expected to be efficient
 *
 * @param[in]  lhs_ptr                            Pointer to the lhs matrix. Supported data types: F32/F16
 * @param[in]  lhs_stride_y                       Stride of the lhs matrix in Y (2nd) dimension (in bytes)
 * @param[in]  lhs_stride_z                       Stride of the lhs tensor in Z (3rd) dimension (in bytes)
 * @param[in]  lhs_w                              The width of the lhs tensor
 * @param[in]  lhs_h                              The height of the lhs tensor
 * @param[in]  lhs_n                              Number of the matrices (buffers) in the batch
 * @param[in]  lhs_offset_first_element_in_bytes  The offset of the first element in the lhs matrix
 * @param[in]  rhs_img                            (Optional) Read only cl_image object for the rhs tensor. Included when RHS_TENSOR_TYPE=IMAGE
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
 */
__kernel void linear(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, RHS_TENSOR_TYPE),
#ifdef BIAS
    TENSOR3D_T(bias, BUFFER),
#endif // defined(BIAS)
    TENSOR3D_T(dst, BUFFER))
{
    const uint x = GET_SPATIAL_IDX(0, N0, PARTIAL_STORE_N0);
    const uint y = GET_SPATIAL_IDX(1, M0, PARTIAL_STORE_M0);
    const uint z = GET_SPATIAL_IDX(2, 1, 0);

    // Compute LHS/RHS/DST matrix address
    lhs_offset_first_element_in_bytes += y * lhs_stride_y + z * lhs_stride_z;
    dst_offset_first_element_in_bytes += x * sizeof(DATA_TYPE) + y * dst_stride_y + z * dst_stride_z;

    // Initialize the accumulators
    TILE(DATA_TYPE, M0, N0, acc);

#pragma unroll
    for(int i = 0; i < M0; ++i)
    {
        acc[i].v = 0.f;
    }

    const int rhs_z = z * rhs_h;
    int       k;
    for(k = 0; k <= K - K0; k += K0)
    {
        TILE(DATA_TYPE, M0, K0, a);
        TILE(DATA_TYPE, K0, N0, b);

#pragma unroll
        for(int i = 0; i < M0; ++i)
        {
            a[i].v = 0.f;
        }

#pragma unroll
        for(int i = 0; i < K0; ++i)
        {
            b[i].v = 0.f;
        }

#pragma unroll
        for(int i = 0; i < M0; ++i)
        {
            a[i].v = V_LOAD(DATA_TYPE, K0, BUFFER, lhs, 0, (i * (int)(1)), lhs_stride_y);
        }

        
        #pragma unroll
        for(int i = 0; i < K0; ++i)
        {
            b[i].v = V_LOAD(DATA_TYPE, N0,BUFFER, rhs, x, ((k + rhs_z) + i * (int)(1)), rhs_stride_y);
        }
        

#pragma unroll
        for(int _m = 0; _m < M0; ++_m)
        {
#pragma unroll
            for(int _k = 0; _k < K0; ++_k)
            {
                acc[_m].v = fma((DATA_TYPE)(a[_m].s[_k]), (b[_k].v), acc[_m].v);
            }
        }

        lhs_offset_first_element_in_bytes += K0 * sizeof(DATA_TYPE);
    }

#if K % K0 != 0
    /* Leftover Loop */
    for(; k < K; ++k)
    {
        TILE(DATA_TYPE, M0, 1, a);
        TILE(DATA_TYPE, 1, N0, b);

#pragma unroll
        for(int i = 0; i < M0; ++i)
        {
            a[i].v = 0.f;
        }

#pragma unroll
        for(int i = 0; i < 1; ++i)
        {
            b[i].v = 0.f;
        }

// Load tile from the lhs/rhs tensors
#pragma unroll
        for(int i = 0; i < M0; ++i)
        {
            a[i].v = V_LOAD(DATA_TYPE, 1, BUFFER, lhs, 0, (i * (int)(1)), lhs_stride_y);
        }

        
        #pragma unroll
        for(int i = 0; i < K0; ++i)
        {
            b[i].v = V_LOAD(DATA_TYPE, N0,BUFFER, rhs, x, ((k + rhs_z) + i * (int)(1)), rhs_stride_y);
        }
        

#pragma unroll
        for(int _m = 0; _m < M0; ++_m)
        {
#pragma unroll
            for(int _k = 0; _k < 1; ++_k)
            {
                acc[_m].v = fma((DATA_TYPE)(a[_m].s[_k]), (b[_k].v), acc[_m].v);
            }
        }

        lhs_offset_first_element_in_bytes += 1 * sizeof(DATA_TYPE);
    }
#endif // K % K0 != 0

    const bool x_cond = PARTIAL_STORE_N0 != 0 && get_global_id(0) == 0;
    const bool y_cond = PARTIAL_STORE_M0 != 0 && get_global_id(1) == 0;

    TILE(int, M0, 1, indirect_buffer);

#pragma unroll
    for(int _i = 0; _i < M0; ++_i)
    {
        indirect_buffer[_i].v = min(_i, select(M0 - 1, PARTIAL_STORE_M0 - 1, y_cond));
    }

    //T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, M0, N0, PARTIAL_STORE_N0, BUFFER, dst, 0, dst_stride_y, x_cond, acc, indirect_buffer);

    
    if(x_cond)
    {
#pragma unroll
        for(int _i = 0; _i < M0; ++_i)
        {
        //  vstore##N0 
            VSTORE(N0)
            (
                CONVERT(acc[M0 - 1 - _i].v,VEC_DATA_TYPE(DATA_TYPE, N0)),
                //                          float2 
                //acc[M0 - 1 - _i].v,      
                0,
                (__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + 0 * sizeof(DATA_TYPE) + (indirect_buffer[M0 - 1 - _i].v) * dst_stride_y)
            );
        }
    }
    else
    {
#pragma unroll
        for(int _i = 0; _i < M0; ++_i)
        {
        //  vstore##N0 
            VSTORE(N0)
            (
                CONVERT(acc[M0 - 1 - _i].v,VEC_DATA_TYPE(DATA_TYPE, N0)),
                //                          float2 
                //acc[M0 - 1 - _i].v,      
                0,
                (__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + 0 * sizeof(DATA_TYPE) + (indirect_buffer[M0 - 1 - _i].v) * dst_stride_y)
            );
        }
    }
    


}
#endif // defined(LINEAR)