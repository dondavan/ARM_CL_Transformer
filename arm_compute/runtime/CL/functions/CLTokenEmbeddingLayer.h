#ifndef ARM_COMPUTE_CLTOKENEMBEDDINGLAYER_H
#define ARM_COMPUTE_CLTOKENEMBEDDINGLAYER_H


#include "arm_compute/core/Types.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

class CLTokenEmbeddingLayer : public IFunction
{
public:
    /** Default Constructor */
    CLTokenEmbeddingLayer();
    /** Default Destructor */
    ~CLTokenEmbeddingLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLTokenEmbeddingLayer(const CLTokenEmbeddingLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLTokenEmbeddingLayer &operator=(const CLTokenEmbeddingLayer &) = delete;

    /** Set the input and output tensor.
     * 
     * @param[in]  input        Input tensor of char text, Data type supported: U8
     * @param[in]  vocab        Const tenser of char 2 vec, Data type supported: F32
     * @param[in]  emb_info     Token Embedding Layer Info.
     * @param[out] output       Output tensor, shape (seq_len,d_model). Data type supported: F32
     */
    void configure(const CLCompileContext &compile_context,
                  ICLTensor *input, 
                  ICLTensor *vocab, 
                  ICLTensor *output, 
                  const EmbeddingLayerInfo& emb_info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLTokenEmbeddingLayer
     *
     * @param[in] output Destination tensor info. Data type supported: same as @p input
     * @param[in] tkemb_info Token Embedding Layer Info.
     *
     * @return a status
     */
    static Status validate(ICLTensor *output, const EmbeddingLayerInfo& tkemb_info);

    void prepare() override;
    // Inherited methods overridden:
    void run() override;
private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute

#endif /* ARM_COMPUTE_CLTOKENEMBEDDINGLAYER_H */