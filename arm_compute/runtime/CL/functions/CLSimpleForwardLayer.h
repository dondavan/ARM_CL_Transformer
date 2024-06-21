#ifndef ARM_COMPUTE_CLSIMPLEFORWARDLAYER_H
#define ARM_COMPUTE_CLSIMPLEFORWARDLAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{

// Forward declarations
class CLCompileContext;
class ICLTensor;
class ICLTensorInfo;

/** Forward input to output */
class CLSimpleForwardLayer : public IFunction
{
public:
    /** Constructor */
    CLSimpleForwardLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLSimpleForwardLayer(const CLSimpleForwardLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    CLSimpleForwardLayer(CLSimpleForwardLayer &&) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLSimpleForwardLayer &operator=(const CLSimpleForwardLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    CLSimpleForwardLayer &operator=(CLSimpleForwardLayer &&) = delete;
    /** Destructor */
    ~CLSimpleForwardLayer();

    /** Initialise the kernel's inputs and output
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * - All
     *
     * @param[in]  tensors        Tensors. Data type supported: All.
     */
    void configure(const ICLTensor *src1,
                   const ICLTensor *src2,
                   const ICLTensor *src3,
                   ICLTensor *dst1,
                   ICLTensor *dst2,
                   ICLTensor *dst3);
    /** Initialise the kernel's inputs and output
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * - All
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  tensors        Tensors. Data type supported: All.
     */
    void configure(const CLCompileContext &compile_context,
                   const ICLTensor *src1,
                   const ICLTensor *src2,
                   const ICLTensor *src3,
                   ICLTensor *dst1,
                   ICLTensor *dst2,
                   ICLTensor *dst3);

    // Inherited methods overridden
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace arm_compute

#endif /* ARM_COMPUTE_CLSIMPLEFORWARDLAYER_H */