#include "helpers.h"

/** Perform token vectorization
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 *
 * @param[in]  input_ptr                            Pointer to the first source tensor. Supported data types: All
 * @param[in]  input_stride_x                       Stride of the first source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the first source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the first source tensor
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  input_shape                          Input spatial shape
 * @param[in]  output_shape                         Output spatial shape
 */
__kernel void get_global_id(TENSOR3D_DECLARATION(input),
                            TENSOR3D_DECLARATION(output),
                            int2 input_shape,
                            int2 output_shape)
{
    int out_x = get_global_id(0);
    int out_y = get_global_id(1);
    int out_z = get_global_id(2);

    // Compute the output linearized index
    int out_linear_idx = out_x + out_y * output_shape.x + out_z * output_shape.x * output_shape.y;

    // Translate to intput
    int in_x = out_linear_idx % input_shape.x;
    int in_y = (out_linear_idx / input_shape.x) % input_shape.y;
    int in_z = out_linear_idx / (input_shape.x * input_shape.y);

    // Store result
    input_ptr += input_offset_first_element_in_bytes + in_x * input_stride_x + in_y * input_stride_y + in_z * input_stride_z;
    output_ptr += output_offset_first_element_in_bytes + out_x * output_stride_x + out_y * output_stride_y + out_z * output_stride_z;
    *((__global DATA_TYPE *)output_ptr) = *((__global DATA_TYPE *)input_ptr);
}
