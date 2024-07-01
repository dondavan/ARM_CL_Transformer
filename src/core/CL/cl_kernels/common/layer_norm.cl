#include "helpers.h"

/** This kernel performs l2 normalization on x-axis
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE_X=size. e.g. -DVEC_SIZE_X=16
 * @note The leftover size in the X dimension shoud be given as preprocessor argument using -DVEC_SIZE_LEFTOVER_X is; x_dimension % VEC_SIZE_X. e.g. -DVEC_SIZE_LEFTOVER_X=1
 *
 * @param[in]  input_ptr                            Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[in]  sum_ptr                              Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  sum_stride_x                         Stride of the source tensor in X dimension (in bytes)
 * @param[in]  sum_step_x                           sum_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_stride_y                         Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  sum_step_y                           sum_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  sum_offset_first_element_in_bytes    The offset of the first element in the source tensor
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  epsilon                              Epsilon value
 */
__kernel void layer_norm(
    IMAGE_DECLARATION(input),
    IMAGE_DECLARATION(sum),
    IMAGE_DECLARATION(output),
    DATA_TYPE epsilon)
{
    // Offset computation
    const uint x_offs = max((int)(get_global_id(0) * VEC_SIZE_X - (VEC_SIZE_X - VEC_SIZE_LEFTOVER_X) % VEC_SIZE_X), 0);

    // Address computation
    __global uchar *input_addr  = input_ptr + input_offset_first_element_in_bytes + x_offs * sizeof(DATA_TYPE) + get_global_id(1) * input_stride_y;
    __global uchar *sum_addr    = sum_ptr + sum_offset_first_element_in_bytes + get_global_id(1) * sum_stride_y;
    __global uchar *output_addr = output_ptr + output_offset_first_element_in_bytes + x_offs * sizeof(DATA_TYPE) + get_global_id(1) * output_stride_y;

    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    in = VLOAD(VEC_SIZE_X)(0, (__global DATA_TYPE *)input_addr);

    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    normalize_value = (VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X))rsqrt(fmax(*((__global DATA_TYPE *)sum_addr), epsilon));

    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    data0 = in * normalize_value;

    STORE_VECTOR_SELECT(data, DATA_TYPE, output_addr, VEC_SIZE_X, VEC_SIZE_LEFTOVER_X, VEC_SIZE_LEFTOVER_X != 0 && get_global_id(0) == 0);
}
