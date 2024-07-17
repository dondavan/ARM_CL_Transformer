#include "arm_compute/graph/nodes/AttentionLinearNode.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

namespace arm_compute
{
namespace graph
{
AttentionLinearNode::AttentionLinearNode(LinearLayerInfo info): _linear_info(std::move(info))
{
    _input_edges.resize(9, EmptyEdgeID); // Input, weight, bias * QKV
    _outputs.resize(3, NullTensorID);    // QKV
}

const LinearLayerInfo& AttentionLinearNode::linear_info() const
{
    return _linear_info;
}

bool AttentionLinearNode::forward_descriptors()
{
    for(size_t idx=0; idx <num_inputs(); idx++)
    {   
        if ((input_id(idx) == NullTensorID) || (output_id(idx) == NullTensorID))
        {
            return false;
        }
        Tensor *dst = output(idx);
        ARM_COMPUTE_ERROR_ON(dst == nullptr);
        dst->desc() = configure_output(idx);
    }
    return true;
}


TensorDescriptor AttentionLinearNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());

    const Tensor *src = input(idx);
    ARM_COMPUTE_ERROR_ON(src == nullptr);

    TensorDescriptor output_desc = src->desc();
    return src->desc();
}


NodeType AttentionLinearNode::type() const
{
    return NodeType::AttentionLinearLayer;
}

void AttentionLinearNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute
