/*
 * Copyright (c) 2018-2019 Arm Limited.
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
//Ehsan
#include "arm_compute/graph/frontend/StreamPipeline.h"
#include <regex>
//#include "utils/GraphUtils.h"
//#include "utils/GraphUtils.cpp"

#include "arm_compute/graph/nodes/ReceiverNode.h"
#include "arm_compute/graph/nodes/SenderNode.h"
#include "arm_compute/graph/nodes/NPUNode.h"

#include "arm_compute/graph/Utils.h"
#include "arm_compute/graph/frontend/ILayer.h"

namespace arm_compute
{
namespace graph
{
namespace frontend
{
void StreamPipeline::set_common_params(arm_compute::utils::CommonGraphParams  _common_params){
	common_params=_common_params;
	//std::cerr<<"set common params; num threads: "<<common_params.threads<<std::endl;
}
StreamPipeline::StreamPipeline(size_t id, std::string _name)
    : _manager(),  num_graphs(0), name(std::move(_name))//, current_layer(0)
{
	//std::cerr<<"hey\n";
	tail_graph_id=0;
}


cpu_set_t* StreamPipeline::set_cores(cpu_set_t *set,int _core, bool _one_master_core){
	CPU_ZERO(set);
	if(_one_master_core){
		CPU_SET(_core,set);
	}
	else{
		if(_core < common_params.little_cores){
			for(int i=0;i<common_params.little_cores;i++){
				CPU_SET(i,set);
			}
		}
		else{
			for(int i=common_params.little_cores;i<common_params.total_cores;i++){
				CPU_SET(i,set);
			}
		}
	}
	return set;
}
cpu_set_t* StreamPipeline::set_cores(cpu_set_t *set,char cluster){
	//std::cerr<<"set to cluster "<<cluster<<std::endl;
	CPU_ZERO(set);
	if(cluster=='L'){
		for(int i=0;i<common_params.little_cores;i++){
			//std::cerr<<"set core "<<i<<std::endl;
			CPU_SET(i,set);
		}
	}
	if(cluster=='B'){
		for(int i=common_params.little_cores;i<common_params.total_cores;i++){
			//std::cerr<<"set core "<<i<<std::endl;
			CPU_SET(i,set);
		}
	}
	return set;
}


class JustAccessor final : public graph::ITensorAccessor
{
public:
    /** Constructor
     *
     * @param[in] maximum Maximum elements to write
     */
    //JustAccessor();
    /** Allows instances to move constructed */
    //JustAccessor(JustAccessor &&) = default;

    // Inherited methods overriden:
    bool access_tensor(ITensor &tensor) override{ return true;};


};


void StreamPipeline::finalize(Target target, const GraphConfig &_config, std::set<int> *b, int blocking)
{

	std::vector<int> indicesToRemove;
	for(auto k=0;k<_gs.size();k++){
		if(_gs[k]->nodes().size()==0){
			indicesToRemove.push_back(k);
		}
	}
	num_graphs=num_graphs-indicesToRemove.size();
	_manager.set_num_graphs(num_graphs);
	//std::cerr<<"\n\n\nWorking on NPU Graphs\n";
	std::set<NodeType> PreservedTypes = {NodeType::NPU, NodeType::Input, NodeType::Receiver, NodeType::Output, NodeType::Sender};
	for(auto k=0;k<_gs.size();k++){
		std::vector<NodeIdxPair> inputs;
		std::vector<NodeIdxPair> outputs;

		if(all_hints[k].target_hint==arm_compute::graph::Target::NPU){
			std::cerr<<"graph "<<k<<" target is NPU, merging nodes into one NPU Node ...\n";
			Graph& g=*(_gs[k].get());
			for(auto &node : g.nodes())
			{
				if(node != nullptr && node->type() == NodeType::Input)
				{
					//std::cerr<<"adding input node "<<node->name()<<" to inputs\n";
					inputs.push_back({node->id(),0});

				}
				else if(node != nullptr && node->type() == NodeType::Receiver)
				{
					//std::cerr<<"adding rec node "<<node->name()<<" to inputs\n";
					inputs.push_back({node->id(),0});
				}
				else if(node != nullptr && node->type() == NodeType::Sender)
				{
					//std::cerr<<"adding sender node "<<node->name()<<" to outputs\n";
					outputs.push_back({node->id(),0});
				}

				else if(node != nullptr && node->type() == NodeType::Output)
				{
					//std::cerr<<"adding output node "<<node->name()<<" to outputs\n";
					outputs.push_back({node->id(),0});
				}
				/*else{
					g->remove_node(node->id());
				}*/

			}

			/*std::cerr<<g.nodes().size()<<std::endl;
			for(auto &node : g.nodes()){
				if(node)
				std::cerr<<"checking node "<<node->id()<<"\n";

			}*/


			//add npu node:
			std::string name=g.name() + "_" + std::to_string(start_layer[k]) + "_" + std::to_string(end_layer[k]);
			std::string substr = "Net";
			// Find the position of the substring
			size_t pos = name.find(substr);

			// If the substring is found, erase it from the string
			if (pos != std::string::npos) {
				name.erase(pos, substr.length());
			}
			name="NPU_"+name;
			NodeParams  common_params_node = { name, all_hints[k].target_hint };
			NodeID nid=GraphBuilder::add_npu_node(g, common_params_node, inputs, outputs);
			NPUNode* n=dynamic_cast<NPUNode*>(g.node(nid));
			//Remove all nodes except preserved node types
			for(auto &node : g.nodes()){
				//std::cerr<<"checking node "<<node->name()<<"\n";
				if (PreservedTypes.find(node->type()) == PreservedTypes.end()){
					//std::cerr<<"removing "<<node->name()<<std::endl;
					g.remove_node(node->id());
				}
			}
			/*std::cerr<<"NPU graph node counts original graph: "<<g.nodes().size()<<std::endl;
			for(auto &node : g.nodes()){
				if(node)
				std::cerr<<"node "<<node->name()<<"\n";
			}*/
		}

	}

	std::cerr<<"\n\n*********************\nStart finalizing Graphs\n*******************\n\n";
	_manager.set_num_graphs(num_graphs);
	std::vector<std::thread> threads;
	bool p=false;
	if(p){
		for(auto i=0;i<_gs.size();i++){
				_ctxs[i].set_config(configs[i]);
				threads.push_back(std::thread(&StreamPipeline::finalize_parallel,this,i,b,blocking));
		}
		for(auto i=0;i<_gs.size();i++){
				threads[i].join();
		}
	}
	else{
		for(auto i=0;i<_gs.size();i++){
				_ctxs[i].set_config(configs[i]);
				finalize_parallel(i,b,blocking);
		}
	}
	/*for (int c=0;c<_ctxs.size();c++){
		//std::cerr<<"context:\n";
		for(const auto& elem : _ctxs[c].memory_managers())
		{
		   std::cout << std::to_string((int)elem.first) << " " << std::to_string((int)elem.second.target)  << "\n";
		}
	}*/
}

void StreamPipeline::finalize_parallel(int i,std::set<int> *b, int blocking)
{
	if(_gs[i]->nodes().size()==0){
		std::cerr<<"Ignoring empty graph "<<i<<std::endl;
		return;
	}
	PassManager pm = create_default_pass_manager(all_hints[i].target_hint, configs[i]);
	cpu_set_t set;

	char cluster='B';
	if(PE[i]=='L')
		cluster='L';
	std::stringstream stream;
	//stream<<"Graph "<<i<<" setting affinity to "<<cluster<<std::endl;
	std::cerr<<stream.str();
	stream.str(std::string());
	set_cores(&set,cluster);
	ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");
	//std::cerr<<"Thread "<<i<<" set: "<<set.__bits<<std::endl;
	//std::this_thread::sleep_for(std::chrono::milliseconds(20000));
	//stream<<"Starting finalizing graph "<<i<<" target "<<std::to_string((int)(all_hints[i].target_hint))<<std::endl;
	std::cerr<<stream.str();
	stream.str(std::string());

	_manager.finalize_graph(*(_gs[i]), _ctxs[i], pm, all_hints[i].target_hint, b, blocking);
	stream<<"Finish finalizing graph "<<i<<"\n\n\n"<<std::endl;
	std::cerr<<stream.str();
	stream.str(std::string());
	return;
}

void StreamPipeline::run(int n)
{

	int method=1;
	if (method==0){
		auto t1=std::chrono::high_resolution_clock::now();
		warmup(n);
		auto t2=std::chrono::high_resolution_clock::now();
		reset_timings();
		double x1=std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
		std::cerr<<"Warm up took "<<x1*1000<<"ms\n";
		std::cerr<<"Start running graphs\n";
		std::vector<std::thread> threads;
		t1=std::chrono::high_resolution_clock::now();
		for(auto i=0;i<_gs.size();i++){
			threads.push_back(std::thread(&StreamPipeline::run_parallel,this,i,n_runs));
		}
		t2=std::chrono::high_resolution_clock::now();
		//double x1=std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();//157ms for 3 threads
		//std::cerr<<x1<<" cost of creating threads for running\n";
		for(auto i=0;i<_gs.size();i++){
			threads[i].join();
		}
	}
	if(method==1){
		std::cerr<<"\n\n\n*********************************************\nstart running graphs\n*************************************************\n\n\n";
		std::vector<std::thread> threads;
		for(auto i=0;i<_gs.size();i++){
			threads.push_back(std::thread(&StreamPipeline::run_w_parallel,this,i,n_runs));
		}
		for(auto i=0;i<_gs.size();i++){
			threads[i].join();
		}
		_manager.print_times(n_runs);
	}

}
void StreamPipeline::warmup(int nn)
{
	std::cerr<<"start  warming up...\n";
	std::vector<std::thread> threads;
	auto t1=std::chrono::high_resolution_clock::now();
	for(auto i=0;i<_gs.size();i++){
		threads.push_back(std::thread(&StreamPipeline::run_parallel,this,i,nn));
	}
	auto t2=std::chrono::high_resolution_clock::now();
	//double x1=std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();//157ms for 3 threads
	//std::cerr<<x1<<" cost of creating threads for running\n";
	for(auto i=0;i<_gs.size();i++){
		threads[i].join();
	}
}

void StreamPipeline::run_parallel(int i, int n)
{
	if(_gs[i]->nodes().size()==0){
		std::cerr<<"Ignoring empty graph "<<i<<std::endl;
		return;
	}
	cpu_set_t set;
	char cluster='B';
	if(PE[i]=='L')
		cluster='L';
	std::stringstream stream;
	//stream<<"Graph "<<i<<" setting affinity to "<<cluster<<std::endl;
	std::cerr<<stream.str();
	stream.str(std::string());
	set_cores(&set,cluster);
	ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");
	stream.str(std::string());
	//stream<<"runing graph "<<i<<std::endl;
	std::cerr<<stream.str();
    _manager.execute_graph(*_gs[i],n);
    //_manager.execute_graph(_g,n);
}

void StreamPipeline::run_w_parallel(int i, int n)
{
	if(_gs[i]->nodes().size()==0){
		std::cerr<<"Ignoring empty graph "<<i<<std::endl;
		return;
	}
	cpu_set_t set;
	char cluster='B';
	if(PE[i]=='L')
		cluster='L';
	std::stringstream stream;
	//stream<<"Graph "<<i<<" setting affinity to "<<cluster<<std::endl;
	//std::cerr<<stream.str();
	//stream.str(std::string());
	set_cores(&set,cluster);
	ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");
	//stream.str(std::string());
	//stream<<"runing graph "<<i<<std::endl;
	//std::cerr<<stream.str();
	std::this_thread::sleep_for(std::chrono::milliseconds(5));
	bool pipeline=false;
	if(pipeline){
		_manager.warmup_and_execute_graph_pipeline(*_gs[i],n_runs);
	}
	else{
		_manager.warmup_and_execute_graph_serial(*_gs[i],n_runs);
	}
    //_manager.execute_graph(_g,n);
}


void StreamPipeline::reset_timings(){
	for(int i=0;i<num_graphs;i++){
		_manager.reset_timing(i);
	}
}
/*
StreamPipeline::StreamPipeline(size_t id, std::string name)
    : _ctx(), _manager(), _g(id, std::move(name))
{
}
void StreamPipeline::finalize(Target target, const GraphConfig &config)
{
	_manager.set_num_graphs(1);
    PassManager pm = create_default_pass_manager(target, config);
    _ctx.set_config(config);
    _manager.finalize_graph(_g, _ctx, pm, target);
}

void StreamPipeline::run(int n)
{
    _manager.execute_graph(_g,n);
}
*/


void StreamPipeline::measure(int n)
{
	_manager.print_times(n);
	//_manager.print_times(*_gs[tail_graph_id], n);
	//_manager.print_times(_g, n);
}

void StreamPipeline::reset()
{
	_manager.reset(*_gs[tail_graph_id]);
	//_manager.reset(_g);
}




/*void Stream::run(double &in,double &task, double &out)
{
    _manager.execute_graph(_g,in,task,out);
}*/





void StreamPipeline::add_layer(ILayer &layer)
{

	//std::cerr<<"StreamPipeline add_layer, on graph: "<<tail_graph_id<<"("<<IStreamPipeline::_target_graph<<") tail_node: "<<tail_node()<<" with "<< graph().nodes().size()<<" nodes\n";
    auto nid   = layer.create_layer(*this);
    //std::cerr<<"Graph:"<<IStreamPipeline::_target_graph<<"  "<<_tail_node<<"->"<<nid<<std::endl;
    _tail_node=nid;
    tail_graph_id=IStreamPipeline::_target_graph;
}

const Graph &StreamPipeline::graph() const
{
	//return _g;
    return *(_gs[IStreamPipeline::_target_graph]);
}

Graph &StreamPipeline::graph()
{
	//return _g;
    return *(_gs[IStreamPipeline::_target_graph]);
}

StreamPipeline & StreamPipeline::operator<<(ILayer &layer)
{
	////IStreamPipeline::_target_graph=target_graph(current_layer);
	//std::cerr<<"(streampipeline) "<<" << operator, layer: "<<current_layer<<" : "<<layer.name()<<" On graph: "<<tail_graph_id<<"("<<IStreamPipeline::_target_graph<<") tail_node: "<<tail_node()<<" with "<< graph().nodes().size()<<" nodes\n";
	//Added by me to add input nodes of a layer into the input_nodes attribute of the layer for later that we want to check if the input nodes are inside this graph
	layer.add_input_node(_tail_node,tail_graph_id);
	/*Be careful that you shoud directly add _tail_node
	  Because calling tail_node() function checks if there is mapped node in _target_graph for _tail_node and return that
	  So if you want to use _tail_node() you should also use _target_graph
	  So you need to check if it gives the mapped node then add that with _target graph, but if not (there is no mapped node in _target_graph) then use tail_graph_id
	auto _n=node_map.find(std::make_pair(_tail_node, tail_graph_id), target_graph);
	if (_n.second==tail_graph_id){
		layer.add_input_node(_tail_node,tail_graph_id);
		//or
		//layer.add_input_node(tail_node(),get_tail_graph_id());
	}
	if (_n.second==_target_graph){
		layer.add_input_node(_n.first,_n.second);
		//or
		//layer.add_input_node(tail_node(),_target_graph);
	}
	*/
	//Check input nodes if they are not in this graph add transfer and/or receiver nodes
	/*std::string formatPattern = ".*_g\\d*|.*relu.*";
	std::regex pattern(formatPattern, std::regex_constants::icase);
	if (regex_search(layer.name(), pattern)) {
		std::cerr << "Skipping layer: "<<layer.name() << std::endl;
	} else {
		next_layer(layer.get_input_nodes(), _tail_node, tail_graph_id);
	}*/
	next_layer(layer.get_input_nodes(), _tail_node, tail_graph_id, layer.name());
    add_layer(layer);
    //std::cerr<<"*******************************\n";
    return *this;
}
StreamPipeline & StreamPipeline::operator<<(ILayer &&layer)
{
	////IStreamPipeline::_target_graph=target_graph(current_layer);
	//std::cerr<<"(streampipeline) "<<" << operator, layer: "<<current_layer<<" : "<<layer.name()<<" On graph: "<<tail_graph_id<<"("<<IStreamPipeline::_target_graph<<") tail_node: "<<tail_node()<<" with "<< graph().nodes().size()<<" nodes\n";
	layer.add_input_node(_tail_node,tail_graph_id);


	next_layer(layer.get_input_nodes(), _tail_node, tail_graph_id, layer.name());


	add_layer(layer);
    //std::cerr<<"*******************************\n";
    return *this;
}
/** Overloaded stream operator to provide a target hint to the graph
 *
 * @param[in, out] s           Stream to provide the hint to
 * @param[in]      target_hint Target hint to be considered
 *
 * @return Updated stream
 */
StreamPipeline & StreamPipeline::operator<<(Target target_hint)
{
    hints().target_hint = target_hint;
    return *this;
}
/** Overloaded stream operator to provide a convolution method hint to the graph
 *
 * @param[in, out] s                       Stream to provide the hint to
 * @param[in]      convolution_method_hint Convolution method hint to be considered
 *
 * @return Updated stream
 */
StreamPipeline & StreamPipeline::operator<<(ConvolutionMethod convolution_method_hint)
{
    hints().convolution_method_hint = convolution_method_hint;
    return *this;
}
/** Overloaded stream operator to provide a depthwise convolution method hint to the graph
 *
 * @param[in, out] s                                 Stream to provide the hint to
 * @param[in]      depthwise_convolution_method_hint Depthwise Convolution method hint to be considered
 *
 * @return Updated stream
 */
StreamPipeline & StreamPipeline::operator<<(DepthwiseConvolutionMethod depthwise_convolution_method_hint)
{
    hints().depthwise_convolution_method_hint = depthwise_convolution_method_hint;
    return *this;
}
/** Overloaded stream operator to provide a fast math hint to the graph
 *
 * @param[in, out] s              Stream to provide the hint to
 * @param[in]      fast_math_hint Convolution method hint to be considered
 *
 * @return Updated stream
 */
StreamPipeline & StreamPipeline::operator<<(FastMathHint fast_math_hint)
{
    hints().fast_math_hint = fast_math_hint;
    return *this;
}

/*NodeID StreamPipeline::maped_node(NodeID tail, int graph_id, int target_graph)
{

	auto _n=node_map.find(std::make_pair(tail, graph_id), target_graph);
	return _n.first;
	//return Tail_node[target];
}*/

/*NodeID StreamPipeline::tail_node()
{
	//std::cerr<<"(streampipeline) tail_node()- Tail_node: "<<_tail_node<<std::endl;
	return _tail_node;

}*/
void StreamPipeline::add_graph(int start, int end, char _PE, char _Host_PE){
    	int id=num_graphs;
    	num_graphs++;
    	//_gs.push_back(std::make_unique<GraphPipeline>(id, name, _PE, _Host_PE, start, end));
    	_gs.emplace_back(new GraphPipeline(id, name, _PE, _Host_PE, start, end));
    	input_time.push_back(0);
    	task_time.push_back(0);
    	output_time.push_back(0);
    	cost.push_back(0);
    	PE.push_back(_PE);
    	Host_PE.push_back(_PE);
    	start_layer.push_back(start);
    	end_layer.push_back(end);
    	/*arm_compute::graph::Target       target_GPU{ arm_compute::graph::Target::CL };
		arm_compute::graph::Target       target_CPU{ arm_compute::graph::Target::NEON };
		arm_compute::graph::Target       target=(_PE=='G')?target_GPU:target_CPU;*/
    	arm_compute::graph::Target       target;
    	switch(_PE){
    		case 'L':
    		case 'B':
    			target=arm_compute::graph::Target::NEON;
    			break;
    		case 'G':
    			target=arm_compute::graph::Target::CL;
    			break;
    		case 'N':
    			target=arm_compute::graph::Target::NPU;

    	}

		//*(_gs[graph_id])<< target;
		StreamHints hint;
		hint.target_hint = target;
		all_hints.emplace_back(hint);
		GraphConfig config;
		int num_threads=0;
		int cluster=0;
		if(_PE=='B'){
			num_threads=common_params.threads;
			cluster=1;
		}
		if(_PE=='L'){
			num_threads=common_params.threads2;
			cluster=0;
		}
		config.use_tuner   = common_params.enable_tuner;
		config.tuner_mode  = common_params.tuner_mode;
		config.tuner_file  = common_params.tuner_file;
		config.mlgo_file   = common_params.mlgo_file;
		config.num_threads = num_threads;
		config.cluster=cluster;
		configs.emplace_back(config);
		/*GraphContext ctx;
		_ctxs.emplace_back(std::move(ctx));*/
		_ctxs.emplace_back(GraphContext());
    	std::cout<<"Adding Graph"<<id<<" target "<<std::to_string((int)(target))<<" PE: "<<_PE<<
    			" Host PE: "<<_Host_PE<<" num threads: "<<num_threads<<" Layers: "<<start<<"-"<<end<<std::endl;
}
NodeID StreamPipeline::next_layer(std::vector<std::pair<NodeID,int>> input_nodes, NodeID& last_tail_node, int& last_tail_graph, std::string layer_name ){
	//std::cerr<<"size of graphs:"<<_gs.size()<<std::endl;
	///std::string ssss;
	///std::cin>>ssss;
	//std::cerr<<"\n\n\n------layer_name: "<<layer_name<<"\n";
	//std::cerr<<"prev layer number: "<<current_layer<<std::endl;

	//If you want to skip first layer (if instead of ending layer go with starting layer)
	static bool starting_layer=true;
	if (is_next_layer(layer_name)){
		if(starting_layer){
			starting_layer=false;
		}
		else{
			current_layer++;
		}
	}
	//std::cerr<<"current layer number: "<<current_layer<<" layer name: "<<layer_name<<std::endl;
	IStreamPipeline::_target_graph=target_graph(current_layer);
	/*For creating the layer after the last node, in case that the layer has multiple layers which are from other graphs, then we need to change the tail node and tail graph of the stream
	to the last node
	std::pair<NodeID,int> last_node=std::make_pair(this->tail_node(),this->get_tail_graph_id());*/
	//If this layer is starting layer of a subgraph
	if (current_layer==start_layer[IStreamPipeline::_target_graph]){
		_hints=all_hints[IStreamPipeline::_target_graph];
		//std::cerr<<"Starting Graph "<<IStreamPipeline::_target_graph<<" containing layers "<<start_layer[IStreamPipeline::_target_graph]<<"-"<<end_layer[IStreamPipeline::_target_graph]<<std::endl;
	}
	//If this layer is ending layer of a subgraph
	if (current_layer==end_layer[IStreamPipeline::_target_graph]){
		//std::cerr<<"Ending layer of Graph "<<IStreamPipeline::_target_graph<<"\n";
	}

	//check input nodes of this layer to see in which subgraph they are
	for(auto &input_node:input_nodes){
		//If input node is in another graph
		//std::cerr<<"input node id: "<<input_node.first<<" graph: "<<input_node.second<<std::endl;
		//if((_gs[IStreamPipeline::_target_graph])->node(input_node.first)==nullptr && input_node.first!=EmptyNodeID){
		if(input_node.second!=IStreamPipeline::_target_graph){
			//Find the node that is mapped to the input node
			//std::cerr<<"The input node "<<input_node.first<<"-"<<input_node.second<<" is not in target graph: "<<IStreamPipeline::_target_graph<<std::endl;
			auto mapped_node=node_map.find(input_node, IStreamPipeline::_target_graph);
			//No node mapped to input node; need to create a T node in origin graph and R node in _target_graph
			//and mapped them to the input_node (to track node mapping)
			if ((mapped_node.second)==-1){
				//create a T node and append to the node key.first in graph key.second (and add it to mapping: node_map.insert(key,std::make_pair(node_id,key.second))))
				//create a R node in new graph (and add it to mapping: node_map.insert(key,std::make_pair(node_id,new__target_graph)) )
				// Add Transmitter to the previous graph containing input node for this node
				std::cerr<<"Input node for the layer "<<current_layer<<" is in graph: "<<input_node.second<<
						" Adding Transmitter to that graph\n";


				//ITensorAccessorUPtr _accessor=get_Sender_accessor(common_params);
				//GraphBuilder::add_sender_node(*(_gs[i]), common_params_node, input, std::move(_accessor));
				int g_id=input_node.second;
				Graph& g=*(_gs[g_id].get());
				//NodeParams  common_params_node = { "Transmitter", hints().target_hint };

				std::string node_name=g.node(input_node.first)->name();
				NodeIdxPair input         = { input_node.first, 0 };
				NodeParams  common_params_node = { "Sender_"+node_name, all_hints[g_id].target_hint };
				common_params.labels="Sender";
				auto just_accessor_sender=std::make_unique<JustAccessor>();
				NodeID tail_sender=GraphBuilder::add_sender_node(g, common_params_node, input,std::move(just_accessor_sender));//arm_compute::graph_utils::get_output_accessor(common_params, 5)
				//NodeID tail_sender=GraphBuilder::add_sender_node(g, common_params_node, input);
				SenderNode* s=dynamic_cast<SenderNode*>(g.node(tail_sender));
				node_map.insert(std::make_pair(input_node.first,input_node.second),std::make_pair(tail_sender,g_id));
				//Add Receiver Node to the next graph
				common_params_node = { "Receiver_"+node_name, all_hints[IStreamPipeline::_target_graph].target_hint };
				auto just_accessor_rec=std::make_unique<JustAccessor>();
				auto _desc = s->get_sender_tensor()->get_tensor()->desc();
				_desc.target=_hints.target_hint;
				NodeID tail_receiver = GraphBuilder::add_receiver_node(*(_gs[IStreamPipeline::_target_graph]), common_params_node, _desc, std::move(just_accessor_rec));
				//NodeID tail_receiver = GraphBuilder::add_receiver_node(*(_gs[IStreamPipeline::_target_graph]), common_params_node, s->get_sender_tensor()->get_tensor()->desc());
				ReceiverNode* r=dynamic_cast<ReceiverNode*>(_gs[IStreamPipeline::_target_graph]->node(tail_receiver));
				node_map.insert(std::make_pair(input_node.first,input_node.second),std::make_pair(tail_receiver,IStreamPipeline::_target_graph));
				//Moving the input_node from prev graph into equivalent node in this graph
				input_node.first=tail_receiver;
				input_node.second=IStreamPipeline::_target_graph;

				s->get_sender_tensor()->add_receiver(r->get_receiver_tensor());
				//std::cerr<<"Add sender: "<<tail_sender<<" receiver node: "<<tail_receiver<<std::endl;

			}
			//There is a mapped node in the target graph for the input node which is originally from other graphs
			else if((mapped_node.second)==IStreamPipeline::_target_graph){
				//change the tail node from original graph to mapped node in target graph
				input_node.first=mapped_node.first;
				input_node.second=mapped_node.second;
				//std::cerr<<"There is a mapped node in the target graph for the input node which is originally from other graphs\n";
			}
			//When the R node is not in target graph(but there is T node in source graph) the mapped node is T node in source graph
			else if((mapped_node.second)==input_node.second){
				//create a R node in new graph (and add it to mapping: node_map.insert(key,std::make_pair(node_id,new_graph_id)) )
				//add R node into the T node(v.first) of origin graph graph[v.second].node(v.first).add_receiver(R);
				std::cerr<<"Input node for the layer "<<current_layer<<" is in graph: "<<input_node.second<<
						"and there is a sender node (mapped) for that node\n";
				int g_id=input_node.second;
				Graph& g=*(_gs[g_id].get());
				SenderNode* s=dynamic_cast<SenderNode*>(g.node(mapped_node.first));
				//NodeParams  common_params_node = { "Receiver", hints().target_hint };
				std::string node_name=g.node(input_node.first)->name();
				NodeParams  common_params_node = { "Receiver_"+node_name, all_hints[IStreamPipeline::_target_graph].target_hint };
				common_params.labels="Receiver";
				//ITensorAccessorUPtr _accessor=get_Sender_accessor(common_params);
				//GraphBuilder::add_sender_node(*(_gs[i]), common_params_node, input, std::move(_accessor));
				//Add Receiver Node to the next graph
				auto just_accessor_rec=std::make_unique<JustAccessor>();
				NodeID tail_receiver = GraphBuilder::add_receiver_node(*(_gs[IStreamPipeline::_target_graph]), common_params_node, s->get_sender_tensor()->get_tensor()->desc(),std::move(just_accessor_rec));
				//NodeID tail_receiver = GraphBuilder::add_receiver_node(*(_gs[IStreamPipeline::_target_graph]), common_params_node, s->get_sender_tensor()->get_tensor()->desc());
				node_map.insert(std::make_pair(input_node.first,input_node.second),std::make_pair(tail_receiver,IStreamPipeline::_target_graph));
				input_node.first=tail_receiver;
				input_node.second=IStreamPipeline::_target_graph;
				ReceiverNode* r=dynamic_cast<ReceiverNode*>(_gs[IStreamPipeline::_target_graph]->node(tail_receiver));
				s->get_sender_tensor()->add_receiver(r->get_receiver_tensor());
			}
			else{
				std::cerr<<"Error: Mapped Node\n";
			}

		}
	}

	//std::cerr<<"Current layer: "<<current_layer<<std::endl;

	//If the last node was in another graph then we should change the tail_node and tail_graph of the stream(streampipeline or substream)
	//be aware that when adding sender and receiver nodes (GraphBuilder::add_receiver_node) you directly add the nodes into the graph
	//rather than adding with add_layer of the stream, so the tail node and tail graph will not change.
	//std::cerr<<"next function tail node: "<<this->tail_node()<<" and tail graph: "<<this->tail_graph_id<<std::endl;
	std::pair<NodeID,int> mapped_node=std::make_pair(last_tail_node,last_tail_graph);
	if (last_tail_graph!=IStreamPipeline::_target_graph){
		//std::cerr<<"If the last node was in another graph then we should change the tail_node and tail_graph of the stream(streampipeline or substream)\n";
		mapped_node=node_map.find(mapped_node, IStreamPipeline::_target_graph);
		last_tail_node=mapped_node.first;
		last_tail_graph=mapped_node.second;
	}


	/*if (is_end_layer(layer_name)){
		current_layer++;
	}*/


	return mapped_node.first;
}


void StreamPipeline::prnt(){
	//std::cerr<<"hi\n";
	//std::cerr<<"num_graphs: "<<num_graphs<<" size: "<<_gs.size()<<std::endl;
	//std::cerr<<"Tail node: "<<Tail_node[_target_graph]<<std::endl;
}
void StreamPipeline::forward_tail(NodeID nid)
{
	//Tail_node[_target_graph] = (nid != NullTensorID) ? nid : Tail_node[_target_graph];
	//To update _tail_node(in IStream) also which is used by SubStream
	IStream::forward_tail(nid);
}


} // namespace frontend
} // namespace graph
} // namespace arm_compute
