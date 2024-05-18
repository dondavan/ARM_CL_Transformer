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
#include "arm_compute/graph/nodes/NPUNode.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"
#include "arm_compute/graph/Tensor.h"

namespace arm_compute
{
namespace graph
{
NPUNode::NPUNode(std::vector<NodeIdxPair> inputs, std::vector<NodeIdxPair> outputs)
    //: _info(info), _out_quant_info(std::move(out_quant_info))
{
    //_input_edges.resize(inputs.size(), EmptyEdgeID);
    //We do not resize outputs to real number of outputs, because the output tensors are already created(unlike normal routine which
    //output tensors always create by source node but because the nodes and tensors alreay created and we just want to replace
    //this node we use the already created tensors so now we set the number of outputs to 0, and in addnode function it does not
    //create tensor for each output but after that we set number of outputs to correct value
    //_outputs.resize(outputs.size(), NullTensorID);
    //_outputs.resize(0,NullTensorID);
	//std::cerr<<"init NPU node\n";
    Inputs=inputs;
    Outputs=outputs;
}



bool NPUNode::forward_descriptors()
{
	//The restructure graph code could not be here because:
	//This function is called inside the Graph::addnode() function which(lock(_mtx)) is already got the lock
	//And in that part in add connection need to get the lock wich cause forever waiting
    return true;
}

bool NPUNode::restructure_graph(){
	_input_edges.resize(Inputs.size(), EmptyEdgeID);
	//_outputs.resize(Outputs.size(), NullTensorID);
	int i=0;
	for(auto output:Outputs){

		auto output_node=_graph->node(output.node_id);
		std::cerr<<"re-struct output node "<<output_node->name()<<" to the npu node\n";
		auto indx=output.index;
		int j=0;
		int n_inputs=output_node->num_inputs();
		for (int j=0;j<n_inputs;j++){
			auto edge=output_node->input_edge(j);
			if(edge->producer()->type()!=arm_compute::graph::NodeType::NPU){
				std::cerr<<"producer is "<<edge->producer()->name()<<std::endl;
				_outputs.push_back(edge->tensor_id());
				//It remove the the edge_id from producer.output_edges,
				//and set consumer.input_edges[edge.consumer_indx] to Emptyedgeid
				std::cerr<<"removing edge between producer and the output\n";
				_graph->remove_connection(edge->id());
				std::cerr<<"adding edge between NPU node and the output\n";
				_graph->add_connection(_id, i, output.node_id, j);
				i++;
			}
		}
	}

	i=0;
	for(auto input:Inputs){
		auto input_node=_graph->node(input.node_id);
		auto index=input.index;
		int j=0;
		int n_output_edges=input_node->output_edges().size();
		std::set<EdgeID> output_edges_copy = input_node->output_edges();
		std::cerr<<"re-struct input node "<<input_node->name()<<" with "<<n_output_edges<<"out edges to the npu node\n";
		for(auto edge_id:output_edges_copy){
			auto edge=_graph->edge(edge_id);
			if(edge->consumer()->type()!=arm_compute::graph::NodeType::NPU){
				std::cerr<<"consumer is "<<edge->consumer()->name()<<std::endl;
				//edge->producer()=nullptr;
				std::cerr<<"removing edge between input and the consumer\n";
				_graph->remove_connection(edge_id);
				std::cerr<<"adding edge between input and NPU node\n";
				_graph->add_connection(input.node_id, index, _id, i);
				i++;
			}
		}
	}
	return true;
}

TensorDescriptor NPUNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());

    const Tensor *src = input(0);
    const Tensor *t=_graph->tensor(_outputs[idx]);
    ARM_COMPUTE_ERROR_ON(src == nullptr);

    TensorDescriptor output_info = t->desc();
    /*if(!_out_quant_info.empty())
    {
        output_info.quant_info = _out_quant_info;
    }*/

    return output_info;
}

NodeType NPUNode::type() const
{
    return NodeType::NPU;
}


void NPUNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute

