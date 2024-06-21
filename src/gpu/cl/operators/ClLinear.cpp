#include "src/gpu/cl/operators/ClLinear.h"

#include "src/common/utils/Log.h"
#include "src/gpu/cl/ClCompileContext.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "src/gpu/cl/kernels/ClVectorizeKernel.h"
#include "src/gpu/cl/utils/ClAuxTensorHandler.h"

namespace arm_compute
{
namespace opencl
{
void ClLinear::configure(const ClCompileContext &compile_context,
                         const ITensorInfo      *a,
                         const ITensorInfo      *b,
                         const ITensorInfo      *c,
                         ITensorInfo            *d,
                         float                   alpha,
                         float beta, const LinearLayerInfo &linear_info)
{
    ARM_COMPUTE_LOG_PARAMS(a, b, c, d, alpha, beta, linear_info);
    ARM_COMPUTE_UNUSED(a, b, c, d, alpha);
    ARM_COMPUTE_UNUSED(linear_info);
    ARM_COMPUTE_UNUSED(beta);
}

Status
ClLinear::validate(const ITensorInfo *a,
                   const ITensorInfo *b,
                   const ITensorInfo *c,
                   ITensorInfo       *d,
                   float              alpha,
                   float beta, const LinearLayerInfo &linear_info)
{
    ARM_COMPUTE_UNUSED(a);
    ARM_COMPUTE_UNUSED(b);
    ARM_COMPUTE_UNUSED(c);
    ARM_COMPUTE_UNUSED(d);
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_UNUSED(beta);
    ARM_COMPUTE_UNUSED(linear_info);
    return Status{};
}

void ClLinear::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
    auto a = tensors.get_const_tensor(ACL_SRC_0);
    auto b = tensors.get_const_tensor(ACL_SRC_1);
    auto c = tensors.get_const_tensor(ACL_SRC_2);
    auto d = tensors.get_tensor(ACL_DST);
    ARM_COMPUTE_UNUSED(a, b, c, d);
}

} // namespace opencl
} // namespace arm_compute
