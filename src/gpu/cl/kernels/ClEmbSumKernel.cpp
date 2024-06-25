#include "src/gpu/cl/kernels/ClEmbSumKernel.h"

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


void ClEmbSumKernel::configure(const CLCompileContext &compile_context,
                               const ITensorInfo      *token,
                               const ITensorInfo      *segemnt,
                               const ITensorInfo      *position,
                               ITensorInfo            *dst)
{
    ARM_COMPUTE_UNUSED(segemnt,position);
    ARM_COMPUTE_UNUSED(compile_context);

    std::cout << "src/gpu/cl/kernels/ClEmbSumKernel.cpp configure start" << std::endl;

    // Configure output tensor info.
    auto_init_if_empty(*dst, TensorInfo(*token->clone()));

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(dst->data_type()));

    _kernel = create_kernel(compile_context, "embsum", build_opts.options());

    // Configure kernel window
    Window win = calculate_max_window(*dst, Steps());
    ICLKernel::configure_internal(win);


    std::cout << "src/gpu/cl/kernels/ClEmbSumKernel.cpp configure end" << std::endl;
}

Status ClEmbSumKernel::validate(const ITensorInfo *src, const ITensorInfo *vector, ITensorInfo *dst)
{
    ARM_COMPUTE_UNUSED(src);
    ARM_COMPUTE_UNUSED(vector);
    ARM_COMPUTE_UNUSED(dst);
    return Status{};
}

void ClEmbSumKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    std::cout << "src/gpu/cl/kernels/ClEmbSumKernel.cpp run start" << std::endl;

    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(tensors.empty());

    Window slice = window.first_slice_window_3D();

    const auto token    = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_0));
    const auto segemnt  = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_1));
    const auto position = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_2));
    auto       dst      = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    //run_vectorize<float>(window, src, vector, dst);
    //slice.set_broadcasted(Window::DimZ);
    
    std::cout << "slice x " << slice.x().end() << std::endl;
    std::cout << "slice y " << slice.y().end() << std::endl;
    std::cout << "slice z " << slice.z().end() << std::endl;

    std::cout << "token->info()->strides_in_bytes().x() " << token->info()->strides_in_bytes().x() << std::endl;
    std::cout << "token->info()->strides_in_bytes().y() " << token->info()->strides_in_bytes().y() << std::endl;
    std::cout << "token->info()->strides_in_bytes().z() " << token->info()->strides_in_bytes().z() << std::endl;

    std::cout << "segemnt->info()->strides_in_bytes().x() " << segemnt->info()->strides_in_bytes().x() << std::endl;
    std::cout << "segemnt->info()->strides_in_bytes().y() " << segemnt->info()->strides_in_bytes().y() << std::endl;
    std::cout << "segemnt->info()->strides_in_bytes().z() " << segemnt->info()->strides_in_bytes().z() << std::endl;

    std::cout << "position->info()->strides_in_bytes().x() " << position->info()->strides_in_bytes().x() << std::endl;
    std::cout << "position->info()->strides_in_bytes().y() " << position->info()->strides_in_bytes().y() << std::endl;
    std::cout << "position->info()->strides_in_bytes().z() " << position->info()->strides_in_bytes().z() << std::endl;

    std::cout << "dst->info()->strides_in_bytes().x() " << dst->info()->strides_in_bytes().x() << std::endl;
    std::cout << "dst->info()->strides_in_bytes().y() " << dst->info()->strides_in_bytes().y() << std::endl;
    std::cout << "dst->info()->strides_in_bytes().z() " << dst->info()->strides_in_bytes().z() << std::endl;

    // Set srcs
    unsigned int idx = 0;
    add_3D_tensor_argument(idx, token, window);
    add_3D_tensor_argument(idx, segemnt, window);
    add_3D_tensor_argument(idx, position, window);
    add_3D_tensor_argument(idx, dst, window);

    enqueue(queue, *this, slice, lws_hint());

    std::cout << "src/gpu/cl/kernels/ClEmbSumKernel.cpp run end" << std::endl;
}

} // namespace kernels
} // namespace opencl
} // namespace arm_compute