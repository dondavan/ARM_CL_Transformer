#include "src/gpu/cl/kernels/ClSimpleForward1Kernel.h"

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

void ClSimpleForward1Kernel::configure(const CLCompileContext &compile_context, const ITensorInfo *src1, ITensorInfo       *dst1)
{
    ARM_COMPUTE_UNUSED(compile_context);

    std::cout << "src/gpu/cl/kernels/ClSimpleForward1Kernel.cpp configure start" << std::endl;

    auto_init_if_empty(*dst1, src1->clone()->set_tensor_shape(src1->tensor_shape()));

    dst1->set_tensor_shape(src1->tensor_shape());
    std::cout << "ff dst x " << dst1->tensor_shape().x() << std::endl;
    std::cout << "ff dst y " << dst1->tensor_shape().y() << std::endl;
    std::cout << "ff dst z " << dst1->tensor_shape().z() << std::endl;


    Window win;
    win = calculate_max_window(*dst1, Steps());
    // Configure kernel window
    ICLKernel::configure_internal(win);

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(src1->data_type()));

    std::string kernel_name("simple_forward_1");
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    std::cout << "src/gpu/cl/kernels/ClSimpleForward1Kernel.cpp configure end" << std::endl;
}

Status ClSimpleForward1Kernel::validate(const ITensorInfo *src, const ITensorInfo *vector, ITensorInfo *dst)
{
    ARM_COMPUTE_UNUSED(src);
    ARM_COMPUTE_UNUSED(vector);
    ARM_COMPUTE_UNUSED(dst);
    return Status{};
}

void ClSimpleForward1Kernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    std::cout << "src/gpu/cl/kernels/ClSimpleForward1Kernel.cpp run start" << std::endl;

    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(tensors.empty());

    Window     slice = window.first_slice_window_3D();
    const auto src1 = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_0));
    auto       dst1 = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST_0));

    // Set srcs
    unsigned int idx = 0;
    add_3D_tensor_argument(idx, src1, window);
    add_3D_tensor_argument(idx, dst1, window);

    enqueue(queue, *this, slice, lws_hint());

    std::cout << "src/gpu/cl/kernels/ClSimpleForward1Kernel.cpp run end" << std::endl;
}

} // namespace kernels
} // namespace opencl
} // namespace arm_compute