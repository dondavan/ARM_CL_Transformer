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
    std::cout << "src/gpu/cl/operators/ClEmbedSum.cpp configure start" << std::endl;

    auto k_1 = std::make_unique<kernels::ClSaturatedArithmeticKernel>();
    auto k_2 = std::make_unique<kernels::ClSaturatedArithmeticKernel>();

    k_1->configure(compile_context, ArithmeticOperation::ADD, token, segemnt, &_tmp_token_segment, emb_info.c_policy());

    _aux_mem[TokenSegmentOutput] =
        experimental::MemoryInfo(offset_int_vec(TokenSegmentOutput),
                                 experimental::MemoryLifetime::Persistent,
                                 _tmp_token_segment.total_size());

    k_2->configure(compile_context, ArithmeticOperation::ADD, &_tmp_token_segment, position, output, emb_info.c_policy());

    _add_kernel_1 = std::move(k_1);
    _add_kernel_2 = std::move(k_2);

    std::cout << "src/gpu/cl/operators/ClEmbedSum.cpp configure end" << std::endl;
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
    ARM_COMPUTE_UNUSED(tensors);
    std::cout << "src/gpu/cl/operators/ClEmbedSum.cpp run start" << std::endl;
    
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
    auto token    = tensors.get_const_tensor(ACL_SRC_0);
    auto segment  = tensors.get_const_tensor(ACL_SRC_1);
    auto position = tensors.get_const_tensor(ACL_SRC_2);
    auto output   = tensors.get_tensor(ACL_DST);

    CLAuxTensorHandler aux_token_segemnt(offset_int_vec(TokenSegmentOutput), _tmp_token_segment, tensors, true);

    ITensorPack run_pack{ { ACL_SRC_0, token }, { ACL_SRC_1, segment }, { ACL_DST, aux_token_segemnt.get() } };

    CLScheduler::get().enqueue_op(*_add_kernel_1.get(), run_pack, true);

    // Add position
    run_pack.add_const_tensor(ACL_SRC_0, aux_token_segemnt.get());
    run_pack.add_const_tensor(ACL_SRC_1, position);
    run_pack.add_tensor(ACL_DST, output);
    CLScheduler::get().enqueue_op(*_add_kernel_2.get(), run_pack, true);

    std::cout << "src/gpu/cl/operators/ClEmbedSum.cpp run end" << std::endl;
}

} // namespace opencl
} // namespace arm_compute
