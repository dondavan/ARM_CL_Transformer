#include "src/gpu/cl/operators/ClPositionEmbed.h"

#include "arm_compute/runtime/CL/CLScheduler.h"

#include "src/common/IOperator.h"
#include "src/common/utils/LegacySupport.h"
#include "src/common/utils/Log.h"
#include "src/cpu/CpuContext.h"
#include "src/gpu/cl/kernels/ClPositionEmbeddingKernel.h"

namespace arm_compute
{
namespace opencl
{
void ClPositionEmbed::configure(const ClCompileContext &compile_context,
                                const ITensorInfo      *input,
                                const ITensorInfo      *position,
                                ITensorInfo            *output)
{
    ARM_COMPUTE_LOG_PARAMS(input, output);

    std::cout << "src/gpu/cl/operators/ClPositionEmbed.cpp configure start" << std::endl;

    auto k = std::make_unique<kernels::ClPositionEmbeddingKernel>();
    k->configure(compile_context, input, position, output);
    _kernel = std::move(k);
    
    std::cout << "src/gpu/cl/operators/ClPositionEmbed.cpp configure end" << std::endl;
}

Status
ClPositionEmbed::validate(const ITensorInfo *input, const ITensorInfo *position, const ITensorInfo *output)
{
    ARM_COMPUTE_UNUSED(input);
    ARM_COMPUTE_UNUSED(position);
    ARM_COMPUTE_UNUSED(output);
    return Status{};
}

void ClPositionEmbed::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");

    std::cout << "src/gpu/cl/operators/ClPositionEmbed.cpp run start" << std::endl;

    CLScheduler::get().enqueue_op(*_kernel.get(), tensors);

    std::cout << "src/gpu/cl/operators/ClPositionEmbed.cpp run end" << std::endl;
}

} // namespace opencl
} // namespace arm_compute
