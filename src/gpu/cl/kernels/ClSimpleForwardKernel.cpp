#include "src/gpu/cl/kernels/ClSimpleForwardKernel.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"

#include "src/core/CPP/Validate.h"
#include "src/core/common/Registrars.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include "support/Cast.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{

void ClSimpleForwardKernel::configure(const CLCompileContext &compile_context, const ITensorInfo *src1,
                                  const ITensorInfo *src2,
                                  const ITensorInfo *src3,
                                  ITensorInfo       *dst1,
                                  ITensorInfo       *dst2,
                                  ITensorInfo       *dst3)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src);
    ARM_COMPUTE_ERROR_ON_NULLPTR(vector);
    ARM_COMPUTE_UNUSED(compile_context);

    std::cout << "src/gpu/cl/kernels/ClSimpleForwardKernel.cpp configure start" << std::endl;

    auto_init_if_empty(*dst1, src1->clone()->set_tensor_shape(src1->tensor_shape()));
    auto_init_if_empty(*dst2, src2->clone()->set_tensor_shape(src2->tensor_shape()));
    auto_init_if_empty(*dst3, src3->clone()->set_tensor_shape(src3->tensor_shape()));

    Window win;
    win = calculate_max_window(*dst1, Steps());
    // Configure kernel window
    ICLKernel::configure_internal(win);

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(src1->data_type()));

    std::string kernel_name("simple_forward");
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());


    std::cout << "src/gpu/cl/kernels/ClSimpleForwardKernel.cpp configure end" << std::endl;
}

Status ClSimpleForwardKernel::validate(const ITensorInfo *src, const ITensorInfo *vector, ITensorInfo *dst)
{
    ARM_COMPUTE_UNUSED(src);
    ARM_COMPUTE_UNUSED(vector);
    ARM_COMPUTE_UNUSED(dst);
    return Status{};
}

void ClSimpleForwardKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    std::cout << "src/gpu/cl/kernels/ClSimpleForwardKernel.cpp run start" << std::endl;

    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(tensors.empty());
    const auto src1 = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    auto       dst1 = tensors.get_tensor(TensorType::ACL_DST_0);
    const auto src2 = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    auto       dst2 = tensors.get_tensor(TensorType::ACL_DST_1);
    const auto src3 = tensors.get_const_tensor(TensorType::ACL_SRC_2);
    auto       dst3 = tensors.get_tensor(TensorType::ACL_DST_2);

    dst1->copy_from(*src1);
    dst2->copy_from(*src2);
    dst3->copy_from(*src3);

    std::cout << "src/gpu/cl/kernels/ClSimpleForwardKernel.cpp run end" << std::endl;
}

} // namespace kernels
} // namespace opencl
} // namespace arm_compute