#include "src/gpu/cl/kernels/ClPositionEmbeddingKernel.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/Cast.h"

#include <cmath>

namespace arm_compute
{
namespace opencl
{
namespace kernels
{

namespace
{
/**  Vectorize pretrained position embedding*/
template <typename T>
void run_position_embedding(const Window &window, const ITensor *src, const ITensor *vector, ITensor *dst)
{
    std::cout << "run_position_embedding start" << std::endl; 
    
    std::cout << "run_position_embedding end" << std::endl;
}

} // namespace

void ClPositionEmbeddingKernel::configure(const CLCompileContext &compile_context,
                                          const ITensorInfo      *src,
                                          const ITensorInfo      *pos,
                                          ITensorInfo            *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_UNUSED(pos);
    ARM_COMPUTE_UNUSED(compile_context);

    std::cout << "src/gpu/cl/kernels/ClPositionEmbeddingKernel.cpp configure start" << std::endl;

    // Configure output tensor info.
    auto_init_if_empty(*dst, TensorInfo(*src->clone()));

    // Configure kernel window
    Window win = calculate_max_window(*src, Steps());
    ICLKernel::configure_internal(win);

    std::cout << "src/gpu/cl/kernels/ClPositionEmbeddingKernel.cpp configure end" << std::endl;
}

Status ClPositionEmbeddingKernel::validate(const ITensorInfo *src, const ITensorInfo *pos, const ITensorInfo *dst)
{
    ARM_COMPUTE_UNUSED(pos);
    ARM_COMPUTE_UNUSED(src);
    ARM_COMPUTE_UNUSED(dst);

    return Status{};
}

void ClPositionEmbeddingKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_UNUSED(queue);

    std::cout << "src/gpu/cl/kernels/ClPositionEmbeddingKernel.cpp run start" << std::endl;

    auto *src = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_0));
    auto *pos = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_1));
    auto  dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    std::cout << "Input " << src->info()->tensor_shape().x() << std::endl;
    std::cout << "Input " << src->info()->tensor_shape().y() << std::endl;
    std::cout << "Input " << src->info()->tensor_shape().z() << std::endl;

    run_position_embedding<float>(window, src, pos, dst);

    std::cout << "src/gpu/cl/kernels/ClPositionEmbeddingKernel.cpp run end" << std::endl;
}

} // namespace kernels
} // namespace opencl
} // namespace arm_compute
