#include "src/gpu/cl/kernels/ClVectorizeKernel.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"

#include "src/core/CPP/Validate.h"
#include "src/core/common/Registrars.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/vectorize/list.h"

#include "support/Cast.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{

namespace
{
/*
static const std::vector<ClVectorizeKernel::VectorizeKernel> available_kernels = {
    { "neon_vectorize_int_2_float32", [](const VectorizeKernelDataTypeISASelectorData &data)
      { return data.dt == DataType::F32; },
      REGISTER_FP32_NEON(arm_compute::cpu::neon_vectorize_int_2_float32) },

};*/
} // namespace

void ClVectorizeKernel::configure(const CLCompileContext &compile_context, const ITensorInfo *src, const ITensorInfo *vector, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src);
    ARM_COMPUTE_ERROR_ON_NULLPTR(vector);

    //const auto uk = ClVectorizeKernel::get_implementation( VectorizeKernelDataTypeISASelectorData{ dst->data_type(), CPUInfo::get().get_isa() });

    // Configure output tensor info.
    const TensorShape dst_shape(vector->tensor_shape().x(), src->tensor_shape().x());
    if(dst->tensor_shape().total_size() == 0)
    {
        auto_init_if_empty(*dst, TensorInfo(*vector->clone()).set_tensor_shape(dst_shape));
    }
    else
    {
        dst->set_tensor_shape(dst_shape);
    }

    ARM_COMPUTE_ERROR_ON_NULLPTR(uk);

    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE_IN_BYTES=" + support::cpp11::to_string(src->element_size()));

    _kernel = create_kernel(compile_context, "transpose", build_opts.options());

    Window win = calculate_max_window(*src, Steps());
    ICLKernel::configure_internal(win);
}

Status ClVectorizeKernel::validate(const ITensorInfo *src, const ITensorInfo *vector, ITensorInfo *dst)
{
    ARM_COMPUTE_UNUSED(src);
    ARM_COMPUTE_UNUSED(vector);
    ARM_COMPUTE_UNUSED(dst);
    return Status{};
}

void ClVectorizeKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    ARM_COMPUTE_ERROR_ON(tensors.empty());
    ARM_COMPUTE_ERROR_ON(_run_method == nullptr);

    const auto src =
        utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_0));
    //const auto vector = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_1));
    auto dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    Window slice = window.first_slice_window_3D();

    unsigned int idx = 0;
    add_3D_tensor_argument(idx, src, window);
    add_3D_tensor_argument(idx, dst, window);
    enqueue(queue, *this, slice, lws_hint());
}

} // namespace kernels
} // namespace opencl
} // namespace arm_compute