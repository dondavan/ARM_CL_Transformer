#include "src/gpu/cl/operators/ClSegmentEmbed.h"

#include "arm_compute/runtime/CL/CLScheduler.h"

#include "src/common/IOperator.h"
#include "src/common/utils/LegacySupport.h"
#include "src/common/utils/Log.h"
#include "src/gpu/cl/kernels/ClVectorizeKernel.h"


namespace arm_compute
{
namespace opencl
{
void ClSegmentEmbed::configure(const ITensorInfo *input, const ITensorInfo *segment,  ITensorInfo *output)
{
    ARM_COMPUTE_LOG_PARAMS(input, output);

    auto k = std::make_unique<kernels::ClVectorizeKernel>();
    k->configure(input, segment, output);
    _kernel = std::move(k);

}

Status
ClSegmentEmbed::validate(const ITensorInfo *input, const ITensorInfo *segment, const ITensorInfo *output)
{
    ARM_COMPUTE_UNUSED(input);
    ARM_COMPUTE_UNUSED(segment);
    ARM_COMPUTE_UNUSED(output);
    return Status{};
}

void ClSegmentEmbed::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
    auto split_dimension = static_cast<kernels::ClVectorizeKernel *>(_kernel.get())->get_split_dimension_hint();

    ARM_COMPUTE_UNUSED(split_dimension);
    ARM_COMPUTE_UNUSED(tensors);


    CLScheduler::get().enqueue_op(*_kernel.get(), tensors);
    //NEScheduler::get().schedule_op(_kernel.get(), split_dimension, _kernel->window(), tensors);
}


} // namespace cpu
} // namespace arm_compute
