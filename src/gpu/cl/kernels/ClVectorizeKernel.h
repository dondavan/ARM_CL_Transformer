#ifndef SRC_CPU_KERNELS_CL_VECTORIZE_KERNEL_H
#define SRC_CPU_KERNELS_CL_VECTORIZE_KERNEL_H


#include "src/core/common/Macros.h"
#include "src/gpu/cl/IClKernel.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
/** Interface for the vectorization kernel */
class ClVectorizeKernel : public IClKernel
{
private:
    using VectorizeKernelPtr =
        std::add_pointer<void(const ITensor *, const ITensor *, ITensor *, const Window &)>::type;
public:
    /* Default Constructor */
    ClVectorizeKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClVectorizeKernel);
    /** Configure kernel for a given list of arguments
     *
     * @param[in]   src             Source tensor info. Data types supported: U8.
     * @param[in]   vector          Const target vector tensor info, Data type supported: F32
     * @param[out]  dst             Destination tensor info. Data type supported: F32
     * @param[in]   tkemb_info      Token embedding layer information.
     */
    void configure(const ITensorInfo *src, const ITensorInfo *vector,  ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to ClVectorizeKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *vector, ITensorInfo *dst);

    // Inherited methods overridden:
    void        run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;

    /** Get the preferred dimension in which the scheduler splits the work into multiple jobs.
     *
     * @return The split dimension hint.
     */
    size_t get_split_dimension_hint() const
    {
        return _split_dimension;
    }

    struct VectorizeKernel
    {
        const char                                         *name;
        const VectorizeKernelDataTypeISASelectorDataPtr    is_selected;
        VectorizeKernelPtr                                 ukernel;
    };

    static const std::vector<VectorizeKernel> &get_available_kernels();

    
private:
    VectorizeKernelPtr      _run_method{nullptr};
    size_t                  _split_dimension{Window::DimY};
    std::string             _name{};
};

} // kernels
} // namespace cpu
} // namespace arm_compute

#endif /* SRC_CPU_KERNELS_CL_VECTORIZE_KERNEL_H */