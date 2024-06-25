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
    ARM_COMPUTE_UNUSED(position);
    
    auto k_1 = std::make_unique<kernels::ClSaturatedArithmeticKernel>();

    k_1->configure(compile_context, ArithmeticOperation::ADD, token, segemnt, output, emb_info.c_policy());
    //k_2->configure(compile_context, ArithmeticOperation::ADD, &_tmp_token_segment, position, output, emb_info.c_policy());

    _kernel = std::move(k_1);

    //_aux_mem[TokenSegmentOutput] = experimental::MemoryInfo(offset_int_vec(TokenSegmentOutput), experimental::MemoryLifetime::Persistent, _tmp_token_segment.total_size());
    
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



} // namespace opencl
} // namespace arm_compute
