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
#include "arm_compute/graph/nodes/SenderNode.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"
#include "arm_compute/graph/Tensor.h"

namespace arm_compute
{
namespace graph
{
SenderNode::SenderNode(NodeParams params)
{
    _input_edges.resize(1, EmptyEdgeID);
    sender_tensor=new TensorPipelineSender();
    sender_tensor->set_is_npu(params.target==arm_compute::graph::Target::NPU);
}

bool SenderNode::forward_descriptors()
{

	/*********
	 This function is called twice in GraphBuilder::add_sender_node:
	1-When creating node g.add_node<SenderNode>()
	2-When add connection between this node and previous node, at the end of this function: g.add_connection(input.node_id, input.index, nid, 0);
	At the first time, the previous node is not connected to this node (there is no edge between them yet)
	Therefore the output tensor of previous node observable from this node
	In add connection an edge is created and is added to the input_edges of this sink_node and also the output tensor of source_node is bound
	to this edge So at that time this sink node also have input_edge and a tensor which is bounded to it
	So the second time it could set this tensor (which binded to input edge) to the TensorPipeline
	Notice that when a node is created the output tensors is created so the next node will not create tensor for the input
	just add an edge and bound that tensor to that edge
	Interesting method is that g.add<NodeType>() first create an instance of NodeType and each node type resize the _input_edges or _output_edges
	With EmptyEdgeId then g.add<NodeType>() function for each output_edge creates a tensor
	********/
	//if(input_edges.size()>0)
	//std::cerr<<"Tensor Sender: "<<input(0)<<std::endl;
	auto tt=input(0);

	sender_tensor->set_tensor(input(0));
	sender_tensor->set_graph_id(_graph->id());
    return true;
}

TensorDescriptor SenderNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    return TensorDescriptor();
}

NodeType SenderNode::type() const
{
    return NodeType::Sender;
}

void SenderNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
void SenderNode::set_tensor(Tensor *t){
	//Add tensor to TensorPipelineSender
	sender_tensor->set_tensor(t);
}
} // namespace graph
} // namespace arm_compute
