#include "src/gpu/cl/kernels/ClLayerNormKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/Cast.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{

namespace
{

void layer_norm_fp32(const ITensor *src, ITensor *dst, const Window &window, float epsilon,
                     float gamma,
                     float beta,
                     int   layer_axis)
{
    const int  window_step_axis  = 1;
    const auto window_start_axis = static_cast<int>(window[layer_axis].start());
    const auto window_end_axis   = static_cast<int>(window[layer_axis].end());

    Window win = window; //.collapse_if_possible(window, Window::DimZ)
    win.set(layer_axis, Window::Dimension(0, 1, 1));

    Iterator input(src, win);
    Iterator output(dst, win);

    execute_window_loop(
        win,
        [&](const Coordinates &)
        {
            const auto input_ptr  = reinterpret_cast<const float *>(input.ptr());
            const auto output_ptr = reinterpret_cast<float *>(output.ptr());
            float      mean       = 0;
            float      var        = 0;
            float      res;

            const int axis_len = window_end_axis - window_step_axis;
            /* Calculate mean */
            int axis = window_start_axis;
            for(; axis <= axis_len; axis += window_step_axis)
            {
                mean += *(input_ptr + axis);
            }
            mean = mean / (axis_len + 1);

            /* Calculate variance */
            axis = window_start_axis;
            for(; axis <= axis_len; axis += window_step_axis)
            {
                var += (*(input_ptr + axis) - mean) * (*(input_ptr + axis) - mean);
            }
            var = var / (axis_len + 1);

            /* Calculate layer normalization */
            axis = window_start_axis;
            for(; axis <= axis_len; axis += window_step_axis)
            {
                res                                           = ((*(input_ptr + axis) - mean) / sqrt(var + epsilon)) * gamma + beta;
                *reinterpret_cast<float *>(output_ptr + axis) = res;
            }

            ARM_COMPUTE_UNUSED(epsilon);
            ARM_COMPUTE_UNUSED(input_ptr);
            ARM_COMPUTE_UNUSED(output_ptr);
        },
        input, output);
}

} // namespace

void ClLayerNormKernel::configure(const ClCompileContext &compile_context,
                                  const ITensorInfo      *input,
                                  ITensorInfo            *output,
                                  LayerNormLayerInfo      info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate(input, output, info));
    ARM_COMPUTE_UNUSED(compile_context);

    _info = info;

    TensorShape out_shape = input->tensor_shape();
    // Auto initialize if empty
    set_shape_if_empty(*output, out_shape);
    set_data_type_if_unknown(*output, input->data_type());

    Window win = calculate_max_window(*input, Steps());
    ICLKernel::configure_internal(win);
}

Status ClLayerNormKernel::validate(const ITensorInfo *input,
                                   const ITensorInfo *output,
                                   LayerNormLayerInfo info)
{
    ARM_COMPUTE_UNUSED(input);
    ARM_COMPUTE_UNUSED(output);
    ARM_COMPUTE_UNUSED(info);
    return Status{};
}

void ClLayerNormKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_UNUSED(queue);
    const ITensor *src = tensors.get_const_tensor(TensorType::ACL_SRC);
    ITensor       *dst = tensors.get_tensor(TensorType::ACL_DST);
    layer_norm_fp32(src, dst, window, _info.epsilon(), _info.gamma(), _info.beta(), _info.axis());
}

} // namespace kernels
} // namespace opencl
} // namespace arm_compute
