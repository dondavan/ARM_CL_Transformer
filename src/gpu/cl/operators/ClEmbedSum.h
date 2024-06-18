#ifndef ARM_COMPUTE_CL_EMBED_SUM_H
#define ARM_COMPUTE_CL_EMBED_SUM_H

#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClOperator.h"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"

#include "src/gpu/cl/kernels/ClElementwiseKernel.h"

namespace arm_compute
{
namespace opencl
{
/** A function use @ref kernels::CpuAddKernel to sum 3 embedding output*/
class ClEmbedSum : public IClOperator
{
    public:
    /** Configure operator for a given list of arguments
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  token        Token embedding input, Data type supported: F32
     * @param[in]  segemnt      Token embedding input, Data type supported: F32
     * @param[in]  position     Token embedding input, Data type supported: F32
     * @param[out] output       Destination tensor info. Data type supported: F32
     * @param[in]  emb_info     Embedding layer parameters.
     */
    void configure(const ClCompileContext   &compile_context,
                    ITensorInfo        *token,
                    ITensorInfo        *segemnt,
                    ITensorInfo        *position,
                   ITensorInfo              *output,
                   const EmbeddingLayerInfo &emb_info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuEmbedSum::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo        *token,
                           const ITensorInfo        *segemnt,
                           const ITensorInfo        *position,
                           ITensorInfo              *output,
                           const EmbeddingLayerInfo &emb_info);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;

    private:
    enum AuxTensorIdx
    {
        TokenSegmentOutput = 0,
        Count
    };

    std::unique_ptr<kernels::ClSaturatedArithmeticKernel> _add_kernel_1{ nullptr };
    std::unique_ptr<kernels::ClSaturatedArithmeticKernel> _add_kernel_2{ nullptr };

    TensorInfo _tmp_token_segment{};

    experimental::MemoryRequirements _aux_mem{ Count };

    size_t _split_dimension{ Window::DimY };
};
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_EMBED_SUM_H */
