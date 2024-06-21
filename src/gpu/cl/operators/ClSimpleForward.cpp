#include "src/gpu/cl/operators/ClSimpleForward.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include "src/common/utils/Log.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/gpu/cl/ClCompileContext.h"

#include "src/gpu/cl/kernels/ClVectorizeKernel.h"

namespace arm_compute
{
namespace opencl
{
void ClSimpleForward::configure(const ClCompileContext &compile_context,
                                const ITensorInfo      *src1,
                                const ITensorInfo      *src2,
                                const ITensorInfo      *src3,
                                ITensorInfo            *dst1,
                                ITensorInfo            *dst2,
                                ITensorInfo            *dst3)
{
    ARM_COMPUTE_UNUSED(src1, src2, src3, dst1, dst2, dst3);
    auto k = std::make_unique<kernels::ClVectorizeKernel>();
    k->configure(compile_context, src1, src2, dst1);
    _kernel = std::move(k);
}

void ClSimpleForward::run(ITensorPack &tensors)
{
    ARM_COMPUTE_UNUSED(tensors);
}

} // namespace opencl
} // namespace arm_compute
