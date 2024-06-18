#include "src/gpu/cl/operators/ClEmbedSum.h"

#include "src/common/utils/Log.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/kernels/ClElementwiseKernel.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "src/gpu/cl/utils/ClAuxTensorHandler.h"

namespace arm_compute
{
namespace opencl
{
void ClEmbedSum::configure(const ClCompileContext   &compile_context,
                           ITensorInfo              *token,
                           ITensorInfo              *segemnt,
                           ITensorInfo              *position,
                           ITensorInfo              *output,
                           const EmbeddingLayerInfo &emb_info)
{
    ARM_COMPUTE_UNUSED(position);
    _add_kernel_1 = std::make_unique<kernels::ClSaturatedArithmeticKernel>();

    _add_kernel_1->configure(compile_context, ArithmeticOperation::ADD, token, segemnt, output,emb_info.c_policy());

}

Status
ClEmbedSum::validate(const ITensorInfo        *token,
                     const ITensorInfo        *segemnt,
                     const ITensorInfo        *position,
                     ITensorInfo              *output,
                     const EmbeddingLayerInfo &emb_info)
{
    ARM_COMPUTE_UNUSED(token);
    ARM_COMPUTE_UNUSED(segemnt);
    ARM_COMPUTE_UNUSED(position);
    ARM_COMPUTE_UNUSED(output);
    ARM_COMPUTE_UNUSED(emb_info);
    return Status{};
}

void ClEmbedSum::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
    auto token    = tensors.get_const_tensor(ACL_SRC_0);
    auto segment  = tensors.get_const_tensor(ACL_SRC_1);
    auto position = tensors.get_const_tensor(ACL_SRC_2);
    auto output   = tensors.get_tensor(ACL_DST);

    CLAuxTensorHandler aux_token_segemnt(offset_int_vec(TokenSegmentOutput), _tmp_token_segment, tensors, true);

    ITensorPack run_pack{ { ACL_SRC_0, token }, { ACL_SRC_1, segment }, { ACL_DST, aux_token_segemnt.get() } };

    CLScheduler::get().enqueue_op(*_add_kernel_1, run_pack, true);
    //NEScheduler::get().schedule_op(_add_kernel_1.get(), Window::DimY, _add_kernel_1->window(), run_pack);

    // Add position
    run_pack.add_const_tensor(ACL_SRC_0, aux_token_segemnt.get());
    run_pack.add_const_tensor(ACL_SRC_1, position);
    run_pack.add_tensor(ACL_DST, output);
    CLScheduler::get().enqueue_op(*_add_kernel_2, run_pack, true);
    //NEScheduler::get().schedule_op(_add_kernel_2.get(), Window::DimY, _add_kernel_2->window(), run_pack);

    
}

} // namespace opencl
} // namespace arm_compute