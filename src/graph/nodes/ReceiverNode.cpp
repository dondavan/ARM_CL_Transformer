/*
 * Copyright (c) 2018 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/graph/nodes/ReceiverNode.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

namespace arm_compute
{
namespace graph
{
ReceiverNode::ReceiverNode(TensorDescriptor desc)
		: _desc(desc)
    //: _desc(std::move(desc))
{
    _outputs.resize(1, NullTensorID);
    receiver_tensor=new TensorPipelineReceiver();
    receiver_tensor->set_is_npu(desc.target==arm_compute::graph::Target::NPU);
}

bool ReceiverNode::forward_descriptors()
{
    if(output_id(0) != NullTensorID)
    {
        Tensor *t = output(0);
        ARM_COMPUTE_ERROR_ON(t == nullptr);
        //std::cerr<<"Tensor receiver: "<<t<<std::endl;
        t->desc() = configure_output(0);
        //Add the tensor pointer into the TensorPipelineReceiver
        Edge *e = _graph->edge(*_output_edges.begin());
        auto tt= (e != nullptr) ? e->tensor() : nullptr;



        receiver_tensor->set_tensor(t);
        receiver_tensor->set_graph_id(_graph->id());
        return true;
    }
    return false;
}

TensorDescriptor ReceiverNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    return _desc;
}

NodeType ReceiverNode::type() const
{
    return NodeType::Receiver;
}

void ReceiverNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute
