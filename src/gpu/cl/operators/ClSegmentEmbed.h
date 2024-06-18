#ifndef ARM_COMPUTE_CL_SEGMENT_EMBED_H
#define ARM_COMPUTE_CL_SEGMENT_EMBED_H


#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"

#include "src/gpu/cl/IClOperator.h"


namespace arm_compute
{
namespace opencl
{
/** Basic function to run @ref kernels::CpuEmbedKernel */
class ClSegmentEmbed : public IClOperator
{
public:
    /** Configure operator for a given list of arguments
     *
     * @param[in]  input           Source tensor info. Data types supported: U8.
     * @param[in]  segment         Const segment vector, Data type supported: F32
     * @param[out] output          Destination tensor info. Data type supported: F32
     */
    void configure(const ITensorInfo *input, const ITensorInfo *segment, ITensorInfo *output);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuSegmentEmbed::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *segment, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
};
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_SEGMENT_EMBED_H */
