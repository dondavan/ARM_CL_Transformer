#ifndef ARM_COMPUTE_NESIMPLEFORWARDLAYER_H
#define ARM_COMPUTE_NESIMPLEFORWARDLAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class ITensor;
class ITensorInfo;

/** Forward input to output */
class NESimpleForwardLayer : public IFunction
{
public:
    /** Constructor */
    NESimpleForwardLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NESimpleForwardLayer(const NESimpleForwardLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NESimpleForwardLayer(NESimpleForwardLayer &&) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NESimpleForwardLayer &operator=(const NESimpleForwardLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NESimpleForwardLayer &operator=(NESimpleForwardLayer &&) = delete;
    /** Destructor */
    ~NESimpleForwardLayer();

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
    void configure(const ITensor *src1,
                   const ITensor *src2,
                   const ITensor *src3,
                   ITensor *dst1,
                   ITensor *dst2,
                   ITensor *dst3);

    // Inherited methods overridden
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace arm_compute

#endif /* ARM_COMPUTE_NESIMPLEFORWARDLAYER_H */