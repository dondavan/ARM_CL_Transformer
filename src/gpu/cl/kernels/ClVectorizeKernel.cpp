#include "src/gpu/cl/kernels/ClVectorizeKernel.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"

#include "src/core/CPP/Validate.h"
#include "src/core/common/Registrars.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/vectorize/list.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{

namespace
{
static const std::vector<ClVectorizeKernel::VectorizeKernel> available_kernels = {
    /*
#ifdef ARM_COMPUTE_ENABLE_SVE
    // TBA
#endif // ARM_COMPUTE_ENABLE_SVE

#ifdef __aarch64__
    // TBA
#endif // __aarch64__
*/
    { "neon_vectorize_int_2_float32", [](const VectorizeKernelDataTypeISASelectorData &data)
      { return data.dt == DataType::F32; },
      REGISTER_FP32_NEON(arm_compute::cpu::neon_vectorize_int_2_float32) },

};
} // namespace

void ClVectorizeKernel::configure(const ITensorInfo *src, const ITensorInfo *vector, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src);
    ARM_COMPUTE_ERROR_ON_NULLPTR(vector);

    const auto uk = ClVectorizeKernel::get_implementation(
        VectorizeKernelDataTypeISASelectorData{ dst->data_type(), CPUInfo::get().get_isa() });

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

    _run_method = uk->ukernel;
    _name       = std::string("ClVectorizeKernel").append("/").append(uk->name);

    Window win;

    win = calculate_max_window(*src, Steps());
    ICPPKernel::configure(win);
}

Status ClVectorizeKernel::validate(const ITensorInfo *src, const ITensorInfo *vector, ITensorInfo *dst)
{
    ARM_COMPUTE_UNUSED(src);
    ARM_COMPUTE_UNUSED(vector);
    ARM_COMPUTE_UNUSED(dst);
    return Status{};
}


void ClVectorizeKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    ARM_COMPUTE_ERROR_ON(tensors.empty());
    ARM_COMPUTE_ERROR_ON(_run_method == nullptr);

    const ITensor *src    = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const ITensor *vector = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    ITensor       *dst    = tensors.get_tensor(TensorType::ACL_DST);

    _run_method(src, vector, dst, window);
}


const std::vector<ClVectorizeKernel::VectorizeKernel> &ClVectorizeKernel::get_available_kernels()
{
    return available_kernels;
}

} // namespace kernels
} // namespace opencl
} // namespace arm_compute