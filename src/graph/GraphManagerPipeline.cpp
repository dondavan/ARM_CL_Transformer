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
#include "utils/Power/Power.h"
//#include "arm_compute/NPU.h"


#include "arm_compute/graph/GraphManagerPipeline.h"
#include "arm_compute/graph/nodes/SenderNode.h"
#include "arm_compute/graph/nodes/ReceiverNode.h"




namespace arm_compute
{
namespace graph
{
GraphManagerPipeline::GraphManagerPipeline()
    : GraphManager()
{
	set_num_graphs(1);
	pipeline_ready.store(false);
	measure_when_full=true;
	parallel=false;
}


void GraphManagerPipeline::finalize_graph(Graph &graph, GraphContext &ctx, PassManager &pm, Target target, std::set<int> *blocking_set, int blocking)
{
    // Check if graph has been registered
	//std::cerr<<"graph id: "<<graph.id()<<std::endl;
	//std::this_thread::sleep_for(std::chrono::milliseconds(1000*graph.id()));
    if(_workloads.find(graph.id()) != std::end(_workloads))
    {
        ARM_COMPUTE_ERROR("Graph is already registered!");
    }
    //std::cout<<"graph id:"<<graph.id()<<std::endl;
    // Apply IR mutating passes
    //std::cerr<<"befor pass 1 graph "<<graph.id()<<std::endl;
    //print_times(graph,1);


    pm.run_type(graph, IGraphMutator::MutationType::IR);
    //std::cerr<<"0\n";
    // Force target to all graph construct
    // TODO (COMPMID-2014) : Support heterogeneous execution
    Target forced_target = target;
    if(!is_target_supported(target))
    {
	//Ehsan
	//std::cout<<"target is not supported."<<std::endl;

        forced_target = get_default_target();
        ARM_COMPUTE_LOG_GRAPH_INFO("Switching target from " << target << " to " << forced_target << std::endl);
    }
#if My_print > -1
    //Ehsan
    /*if(target==arm_compute::graph::Target::NPU){
    	std::cerr<<"graph "<<graph.id()<<" target is npu\n";
    	//target=arm_compute::graph::Target::NEON;
    }*/
    std::cerr<<"Graph id: "<<graph.id()<<" Target is: "<<forced_target<<std::endl;
#endif
    /*if (graph.id()==2){
    	std::cerr<<graph<<std::endl;
    }*/
    force_target_to_graph(graph, forced_target);

    //std::cerr<<"1\n";
    // Setup backend context
    // TODO (COMPMID-2014) : Setup all backends needed by the graph



    {
    	//Placed in critical region for parallel setup
    	std::lock_guard<std::mutex> lock(_mtx);
		setup_requested_backend_context(ctx, forced_target);
    }
	// Configure all tensors
	/*Ehsan:
	 * set TensforHandle for all tensors which TensorInfo of TensorAllocator for each TensorHandle is set based on information of each tensor such as shape,datatype,
	 * quantinfo and ...
	 * strides in bytes for all dimensions also is set in tensorInfo
	 */
	//std::cerr<<"graph "<<graph.id()<<" 2\n";


	detail::configure_all_tensors(graph);
    /*if(graph.id()==2){
    	std::cerr<<"node name  -- > "<<graph.node(0)->name()<<std::endl;
    	std::cerr<<"node name  -- > "<<graph.node(1)->name()<<std::endl;
    	std::cerr<<"node name  -- > "<<graph.node(2)->name()<<std::endl;
    	std::cerr<<"node name  -- > "<<graph.node(11)->name()<<std::endl;
    	std::cerr<<"relu input id: "<<graph.node(1)->input_id(0)<<std::endl;
    	std::cerr<<"rec output id: "<<graph.node(0)->output_id(0)<<std::endl;
    	std::cerr<<"sender input id: "<<graph.node(11)->input_id(0)<<std::endl;
    }*/
    //std::cerr<<"graph "<<graph.id()<<" 3\n";
    // Apply backend mutating passes

    //std::cerr<<"befor pass 2 graph "<<graph.id()<<std::endl;
    //print_times(graph,1);
    /*int gid=4;
    int nodid=8;
    if(graph.id()==gid){
    	//int gid=4;
		//int nodid=0;
		//gid=0;
		//nodid=42;
		Graph& gg=graph;
		auto nn=gg.node(nodid);
		std::cerr<<"name:"<<nn->name()<<std::endl;
		int in_nn=nn->num_inputs();
		std::cerr<<"num inputs:"<<in_nn<<std::endl;
		int inedges=nn->input_edges().size();
		for(int i=0;i<inedges;i++){
			std::cerr<<i<<" "<<nn->input_edge(i)->producer()->name()<<std::endl;
			std::cerr<<nn->input(i)->desc().shape<<", id: "<<nn->input(i)->id()<<std::endl;
		}
		int out_nn=nn->num_outputs();
		std::cerr<<"num outputs:"<<out_nn<<std::endl;
		//int outedges=nn->output_edges().size();
		std::cerr<<nn->output(0)->desc().shape<<", id: "<<nn->output(0)->id()<<std::endl;
		std::string test;
		//std::cin>>test;
    }*/
    /* if you enable this(pm.run_type) it leads to segmentation fault at workload.cpp execute_task function in line: task.task->run();*/
    pm.run_type(graph, IGraphMutator::MutationType::Backend);
    /*if(graph.id()==gid){

		//gid=0;
		//nodid=42;
		Graph& gg=graph;
		auto nn=gg.node(nodid);
		std::cerr<<"name:"<<nn->name()<<std::endl;
		int in_nn=nn->num_inputs();
		std::cerr<<"num inputs:"<<in_nn<<std::endl;
		int inedges=nn->input_edges().size();
		for(int i=0;i<inedges;i++){
			std::cerr<<nn->input(i)->desc().shape<<", id: "<<nn->input(i)->id()<<std::endl;
		}
		int out_nn=nn->num_outputs();
		std::cerr<<"num outputs:"<<out_nn<<std::endl;
		//int outedges=nn->output_edges().size();
		std::cerr<<nn->output(0)->desc().shape<<", id: "<<nn->output(0)->id()<<std::endl;
		std::string test;
		std::cin>>test;
	}*/
    if(graph.id()==2){
    	DotGraphPrinter p;
    	p.print(graph,std::cout);
    }

    // Perform topological sort
    std::vector<NodeID> topological_sorted_nodes = dfs(graph);

    //It add npu node twice when there are two edges from input to npu node for example
    //GLLLLNNNNNNNNNNLLLLLLLLLNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNLNLLLLLLLLL (graph2)
    //so remove the repatative nodes in this vector when target in npu
    if(forced_target==arm_compute::graph::Target::NPU){
    	std::set<int> seen;
    	std::vector<NodeID> uniqueVec;
		for (int elem : topological_sorted_nodes) {
			if (seen.insert(elem).second) {
				uniqueVec.push_back(elem);
			}
		}
		topological_sorted_nodes.clear();
		for (int elem : uniqueVec) {
			topological_sorted_nodes.push_back(elem);
		}

		std::cerr<<"topological node counts original graph: "<<topological_sorted_nodes.size()<<std::endl;
		for(auto &node : topological_sorted_nodes){
			if(node)
			std::cerr<<"after topo node "<<graph.node(node)->name()<<"\n";
		}
    }
    /*
     * // Use std::remove_if with a lambda function to remove duplicates in-place
    topological_sorted_nodes.erase(
        std::remove_if(
            topological_sorted_nodes.begin(),
            topological_sorted_nodes.end(),
            [&](int elem) {
                return !seen.insert(elem).second;
            }
        ),
        topological_sorted_nodes.end()
    );
     */



    //std::cerr<<"size of topological sorted nodes: "<<topological_sorted_nodes.size()<<std::endl;
    // Validate all nodes
    detail::validate_all_nodes(graph);
    //std::cerr<<"graph "<<graph.id()<<" 4\n";
    // Configure all nodes

    auto workload = detail::configure_all_nodes_pipeline(graph, ctx, topological_sorted_nodes);
    ARM_COMPUTE_ERROR_ON_MSG(workload.tasks.empty(), "Could not configure all nodes!");
#if My_print > 0
    //Ehsan
    std::cout<<"\nGraphManager, outputs size:"<<workload.outputs.size()<<std::endl;
#endif
    // Allocate const tensors and call accessors
    //std::cerr<<"graph "<<graph.id()<<" 5\n";
    //std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    detail::allocate_const_tensors_pipeline(graph);
    //std::cerr<<"graph "<<graph.id()<<" 5_1\n";
    detail::call_all_const_node_accessors(graph);
    //std::cerr<<"prepare:\n";
    // Prepare graph
    /*if(graph.id()==2){
    	std::string sss;
    	std::cerr<<"hey0 value dare?:"<<graph.tensor(0)->handle()->tensor().buffer()[5]<<std::endl;
    	std::cin>>sss;
    }
    detail::prepare_all_tasks(workload);
    if(graph.id()==2){
        	std::string sss;
        	std::cerr<<"hey1 value dare?:"<<graph.tensor(1)->handle()->tensor().buffer()[5]<<std::endl;
        	std::cin>>sss;
        }
    if(graph.id()==2){
            	std::string sss;
            	std::cerr<<"hey2 value dare?:"<<graph.tensor(2)->handle()->tensor().buffer()[5]<<std::endl;
            	std::cin>>sss;
            }*/
    //std::cerr<<workload.inputs.size()<<std::endl;
    //Ehsan
    //std::cerr<<"3"<<std::endl;
    int ii=0;
    //std::set<int> blocking_set1 {1, 2, 3, 4};
    //std::set<int> *blocking_set=&blocking_set1;
    //std::cerr<<"graph "<<graph.id()<<" 6\n";

    if(blocking_set!=NULL){
		for(auto &task : workload.tasks)
		{
			if(!task.task)
				continue;
			bool b=false;
			if(blocking_set->find(ii) != blocking_set->end()){
				  b=true;
				  task.ending=true;
			}
			if(blocking==1){
				if(blocking_set!=NULL and b && target==arm_compute::graph::Target ::CL)
					task.block=1;
			}
			if(blocking==2){
				if(blocking_set!=NULL && target==arm_compute::graph::Target ::CL){
					task.block=1;
				}
			}

			ii++;
		}
    }



    if(target==arm_compute::graph::Target ::CL){
    	workload.tasks[workload.tasks.size()-1].block=1;
    }
    //std::cerr<<"graph "<<graph.id()<<" 7\n";
    //std::cerr<<"4"<<std::endl;
#if My_print > 0
    //Ehsan
        DotGraphPrinter p;
        p.print(graph,std::cout);
#endif
    //std::cerr<<"5\n";
    // Setup tensor memory (Allocate all tensors or setup transition manager)
    //std::cerr<<"Big cores: "<<ctx.config().big_cores<<" Small cores: "<<ctx.config().little_cores<<std::endl;
    //std::cerr<<ctx.config().use_transition_memory_manager<<std::endl;
    if(ctx.config().use_transition_memory_manager)
    {
#if My_print > 0
    	//Ehsan
    	std::cerr<<"transition memory mangaer is used\n";
#endif

        detail::configure_transition_manager(graph, ctx, workload);
    }
    else
    {
    	//std::cerr<<"else\n";
        detail::allocate_all_tensors(graph);
    }
    //std::cerr<<"graph "<<graph.id()<<" 8\n";
    // Finalize Graph context
    ctx.finalize();
    //std::cerr<<"graph "<<graph.id()<<" 9\n";
    // Register graph
    std::lock_guard<std::mutex> lock(_mtx);
    detail::NPU_set_preallocated_outputs(workload);
    detail::set_ending_tasks(workload);
    blocking=1;
	for(auto &task : workload.tasks)
	{
		if(!task.task)
			continue;

		if(blocking==1){
			if( task.ending && target==arm_compute::graph::Target ::CL)
				task.block=1;
		}
		if(blocking==2){
			if( target==arm_compute::graph::Target ::CL){
				task.block=1;
			}
		}

		ii++;
	}

	char processor='B';
	if (ctx.config().cluster==0){
		processor='L';
	}
	if(target==arm_compute::graph::Target ::CL){
		processor='G';
	}
	if(target==arm_compute::graph::Target ::NPU){
		processor='N';
	}
	for(auto &task:workload.tasks){
			task.processor=processor;
	}

    _workloads.insert(std::make_pair(graph.id(), std::move(workload)));
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Created workload for graph with ID : " << graph.id() << std::endl);
    //std::cerr<<"after pass graph "<<graph.id()<<std::endl;
    //print_times(graph,1);
    //std::cerr<<"graph "<<graph.id()<<" 10\n";
}
void GraphManagerPipeline::reset_timing(int graph_id){
	//std::cerr<<"Reseting timings for graph "<<graph_id<<"\n\n";
	auto it = _workloads.find(graph_id);
	ExecutionWorkload *workload = &it->second;
	for(auto &task:workload->tasks){
		task.reset();
	}
	input_time[graph_id]=0;
	receive_time[graph_id] =0;
	task_time[graph_id]=0;
	send_time[graph_id] = 0;
	output_time[graph_id]=0;
	transmition_time[graph_id]=0;
	latency_time=0;
	detail::reset_transmit_timings(it->second);
	detail::reset_NPU_timings(it->second);
}

void GraphManagerPipeline::execute_graph(Graph &graph, int nn)
{
    // Check if graph is finalized
	/*if(graph.id()==1){
		std::cerr<<"test:\n";
		print_times(graph,1);
	}*/
	std::stringstream stream;
	stream<<"start of execute graph "<<graph.id()<<" in graph manager\n";
	//stream<<"number of workloads: "<<_workloads.size()<<std::endl;
	std::cerr<<stream.str();
	stream.str(std::string());
    auto it = _workloads.find(graph.id());
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");
    //Ehsan measure input, task and output timings:

    int n=4;
    for(int k=0; k<n;k++)
    {
    	if(measure_when_full){
    		if(k==num_graphs){
    			auto t1=std::chrono::high_resolution_clock::now();
    			reset_timing(graph.id());
    			auto t2=std::chrono::high_resolution_clock::now();
    			double x1=std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    			std::string s="timing reset (for reseting measurement when pipeline is full) took "+std::to_string(x1*1000)+"ms\n";
    			std::cerr<<s;
    		}
    	}
    	// Call input accessors
		auto tstart=std::chrono::high_resolution_clock::now();
		//stream<<"graph_id:"<<graph.id()<<std::endl;
		//std::cerr<<stream.str();
		stream<<"graph_id:"<<graph.id()<<" calling inputs"<<std::endl;
		std::cerr<<stream.str();
		stream.str(std::string());
        detail::call_all_input_node_accessors(it->second);

        std::cerr<<"input called\n";
        auto tfinish=std::chrono::high_resolution_clock::now();
        double x1=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        std::cerr<<"size of input_time: "<<input_time.size()<<std::endl;

        input_time[graph.id()] +=x1;



        //Call All receivers
        tstart=std::chrono::high_resolution_clock::now();
        stream<<"graph_id:"<<graph.id()<<" calling receivers"<<std::endl;
        std::cerr<<stream.str();
        stream.str(std::string());
		detail::call_all_receivers(it->second);

		std::cerr<<"receivers called\n";
		tfinish=std::chrono::high_resolution_clock::now();
		x1=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
		input_time[graph.id()] +=x1;





        // Run graph
		//std::cerr<<"\ntask:"<<task<<std::endl;
		stream<<"graph_id:"<<graph.id()<<" Calling all tasks"<<std::endl;
		std::cerr<<stream.str();
		stream.str(std::string());
		detail::call_all_tasks_pipeline(it->second,nn);
		tstart=std::chrono::high_resolution_clock::now();
		double x2=std::chrono::duration_cast<std::chrono::duration<double>>(tstart-tfinish).count();
		task_time[graph.id()] += x2;



		//Call All Senders
		tstart=std::chrono::high_resolution_clock::now();
		stream<<"graph_id:"<<graph.id()<<" Calling all senders"<<std::endl;
		std::cerr<<stream.str();
		stream.str(std::string());
		transmition_time[graph.id()]+=detail::call_all_senders(it->second);

		std::cerr<<"senders called\n";
		tfinish=std::chrono::high_resolution_clock::now();
		x1=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
		output_time[graph.id()] +=x1;




        // Call output accessors
		double x3=0;
		stream<<"graph_id:"<<graph.id()<<" calling outputs"<<std::endl;
		std::cerr<<stream.str();
		stream.str(std::string());
        detail::call_all_output_node_accessors(it->second);
        tfinish=std::chrono::high_resolution_clock::now();
        x3=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();

        stream<<"Graph"<<graph.id()<<"   Input: "<<x1*1000<<"   Task: "<<x2*1000<<"   Out: "<<x3*1000<<"   Proc: "<<(x2+x3)*1000<<std::endl;
        std::cerr<<stream.str();
        stream.str(std::string());
        output_time[graph.id()] +=x3;
    }
}


/*
 * If there is no buffer in between stages
 */
void GraphManagerPipeline::warmup_and_execute_graph_no_buffer(Graph &graph, int nn)
{
    // Check if graph is finalized
	/*if(graph.id()==1){
		std::cerr<<"test:\n";
		print_times(graph,1);
	}*/
	std::stringstream stream;
	stream<<"start of execute graph "<<graph.id()<<" in graph manager\n";
	//stream<<"number of workloads: "<<_workloads.size()<<std::endl;
	std::cerr<<stream.str();
	stream.str(std::string());
    auto it = _workloads.find(graph.id());
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");
    //Ehsan measure input, task and output timings:

    /*int cc=warmup_n+(num_graphs-1)-graph.id();
    for(int k=0; k<cc;k++)
        {
		// Call input accessors
		auto tstart=std::chrono::high_resolution_clock::now();
		//stream<<"graph_id:"<<graph.id()<<std::endl;
		//std::cerr<<stream.str();
		stream<<"graph_id:"<<graph.id()<<" calling inputs"<<std::endl;
		std::cerr<<stream.str();
		stream.str(std::string());
		detail::call_all_input_node_accessors(it->second);

		std::cerr<<"call all input called\n";
		auto tfinish=std::chrono::high_resolution_clock::now();
		double x1=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
		std::cerr<<"size of input_time: "<<input_time.size()<<std::endl;

		input_time[graph.id()] +=x1;



		//Call All receivers
		tstart=std::chrono::high_resolution_clock::now();
		stream<<"graph_id:"<<graph.id()<<" calling receivers"<<std::endl;
		std::cerr<<stream.str();
		stream.str(std::string());
		detail::call_all_receivers(it->second);

		std::cerr<<"receivers called\n";
		tfinish=std::chrono::high_resolution_clock::now();
		x1=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
		input_time[graph.id()] +=x1;





		// Run graph
		//std::cerr<<"\ntask:"<<task<<std::endl;
		stream<<"graph_id:"<<graph.id()<<" Calling all tasks"<<std::endl;
		std::cerr<<stream.str();
		stream.str(std::string());
		detail::call_all_tasks_pipeline(it->second,nn);
		tstart=std::chrono::high_resolution_clock::now();
		double x2=std::chrono::duration_cast<std::chrono::duration<double>>(tstart-tfinish).count();
		task_time[graph.id()] += x2;



		//Call All Senders
		tstart=std::chrono::high_resolution_clock::now();
		stream<<"graph_id:"<<graph.id()<<" Calling all senders"<<std::endl;
		std::cerr<<stream.str();
		stream.str(std::string());
		transmition_time[graph.id()]+=detail::call_all_senders(it->second);

		std::cerr<<"senders called\n";
		tfinish=std::chrono::high_resolution_clock::now();
		x1=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
		output_time[graph.id()] +=x1;




		// Call output accessors
		double x3=0;
		stream<<"graph_id:"<<graph.id()<<" calling outputs"<<std::endl;
		std::cerr<<stream.str();
		stream.str(std::string());
		detail::call_all_output_node_accessors(it->second);
		tfinish=std::chrono::high_resolution_clock::now();
		x3=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();

		stream<<"Graph"<<graph.id()<<"   Input: "<<x1*1000<<"   Task: "<<x2*1000<<"   Out: "<<x3*1000<<"   Proc: "<<(x2+x3)*1000<<std::endl;
		std::cerr<<stream.str();
		stream.str(std::string());
		output_time[graph.id()] +=x3;
	}*/






    int cc=warmup_n+(num_graphs-1)-graph.id();
    int n=10;
    for(int k=0; k<n;k++)
    {

    	if(k==cc){
    		reset_timing(graph.id());
    		//Because there is no buffering so till last stage did not get frame [warmup_n] from previous stage, the previous stage cannot start processing next frame
    		if(graph.id()==num_graphs-1){
    			std::cerr<<"stage "<<graph.id()<<" processed "<<cc<<" frames and will set ready to true after 2 seconds\n";
    			std::this_thread::sleep_for(std::chrono::milliseconds(20000));
    		}
    		if(!parallel){
    			//start power measurement
    			std::cerr<<"non parallel or start with empty pipeline so just first stage synchronized\n";
    		}
    	}
    	// Call input accessors
		auto tstart=std::chrono::high_resolution_clock::now();
		//stream<<"graph_id:"<<graph.id()<<std::endl;
		//std::cerr<<stream.str();
		stream<<"graph_id:"<<graph.id()<<" calling inputs"<<std::endl;
		std::cerr<<stream.str();
		stream.str(std::string());
        detail::call_all_input_node_accessors(it->second);

        std::cerr<<"input called\n";
        auto tfinish=std::chrono::high_resolution_clock::now();
        double x1=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        //std::cerr<<"size of input_time: "<<input_time.size()<<std::endl;

        input_time[graph.id()] +=x1;



        //Call All receivers
        tstart=std::chrono::high_resolution_clock::now();
        stream<<"graph_id:"<<graph.id()<<" calling receivers"<<std::endl;
        std::cerr<<stream.str();
        stream.str(std::string());
		detail::call_all_receivers(it->second);

		std::cerr<<"receivers called\n";
		tfinish=std::chrono::high_resolution_clock::now();
		x1=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
		input_time[graph.id()] +=x1;



        // Run graph
		//std::cerr<<"\ntask:"<<task<<std::endl;
		stream<<"graph_id:"<<graph.id()<<" Calling all tasks"<<std::endl;
		std::cerr<<stream.str();
		stream.str(std::string());
		detail::call_all_tasks_pipeline(it->second,nn);
		tstart=std::chrono::high_resolution_clock::now();
		double x2=std::chrono::duration_cast<std::chrono::duration<double>>(tstart-tfinish).count();
		task_time[graph.id()] += x2;



		//Call All Senders
		tstart=std::chrono::high_resolution_clock::now();
		stream<<"graph_id:"<<graph.id()<<" Calling all senders"<<std::endl;
		std::cerr<<stream.str();
		stream.str(std::string());
		transmition_time[graph.id()]+=detail::call_all_senders(it->second);

		std::cerr<<"senders called\n";
		tfinish=std::chrono::high_resolution_clock::now();
		x1=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
		output_time[graph.id()] +=x1;




        // Call output accessors
		double x3=0;
		stream<<"graph_id:"<<graph.id()<<" calling outputs"<<std::endl;
		std::cerr<<stream.str();
		stream.str(std::string());
        detail::call_all_output_node_accessors(it->second);
        tfinish=std::chrono::high_resolution_clock::now();
        x3=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();

        stream<<"Graph"<<graph.id()<<"   Input: "<<x1*1000<<"   Task: "<<x2*1000<<"   Out: "<<x3*1000<<"   Proc: "<<(x2+x3)*1000<<std::endl;
        std::cerr<<stream.str();
        stream.str(std::string());
        output_time[graph.id()] +=x3;
    }
}



/*If you want to buffer output of a stage if next stage is busy:
 * It also works with no buffer scenario
 */

void GraphManagerPipeline::warmup_and_execute_graph_pipeline(Graph &graph, int nn)
{
    // Check if graph is finalized
	/*if(graph.id()==1){
		std::cerr<<"test:\n";
		print_times(graph,1);
	}*/
	std::stringstream stream;
	std::chrono::time_point<std::chrono::high_resolution_clock> t_arriving;
	std::chrono::time_point<std::chrono::high_resolution_clock> t_completion;
	//stream<<"start of warmup and execute graph "<<graph.id()<<" in graph manager\n";
	//stream<<"number of workloads: "<<_workloads.size()<<std::endl;
	//std::cerr<<stream.str();
	//stream.str(std::string());
    auto it = _workloads.find(graph.id());
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");
    //Ehsan measure input, task and output timings:

    /*int cc=warmup_n+(num_graphs-1)-graph.id();
    for(int k=0; k<cc;k++)
        {
		// Call input accessors
		auto tstart=std::chrono::high_resolution_clock::now();
		//stream<<"graph_id:"<<graph.id()<<std::endl;
		//std::cerr<<stream.str();
		stream<<"graph_id:"<<graph.id()<<" calling inputs"<<std::endl;
		std::cerr<<stream.str();
		stream.str(std::string());
		detail::call_all_input_node_accessors(it->second);

		std::cerr<<"call all input called\n";
		auto tfinish=std::chrono::high_resolution_clock::now();
		double x1=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
		std::cerr<<"size of input_time: "<<input_time.size()<<std::endl;

		input_time[graph.id()] +=x1;



		//Call All receivers
		tstart=std::chrono::high_resolution_clock::now();
		stream<<"graph_id:"<<graph.id()<<" calling receivers"<<std::endl;
		std::cerr<<stream.str();
		stream.str(std::string());
		detail::call_all_receivers(it->second);

		std::cerr<<"receivers called\n";
		tfinish=std::chrono::high_resolution_clock::now();
		x1=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
		input_time[graph.id()] +=x1;





		// Run graph
		//std::cerr<<"\ntask:"<<task<<std::endl;
		stream<<"graph_id:"<<graph.id()<<" Calling all tasks"<<std::endl;
		std::cerr<<stream.str();
		stream.str(std::string());
		detail::call_all_tasks_pipeline(it->second,nn);
		tstart=std::chrono::high_resolution_clock::now();
		double x2=std::chrono::duration_cast<std::chrono::duration<double>>(tstart-tfinish).count();
		task_time[graph.id()] += x2;



		//Call All Senders
		tstart=std::chrono::high_resolution_clock::now();
		stream<<"graph_id:"<<graph.id()<<" Calling all senders"<<std::endl;
		std::cerr<<stream.str();
		stream.str(std::string());
		transmition_time[graph.id()]+=detail::call_all_senders(it->second);

		std::cerr<<"senders called\n";
		tfinish=std::chrono::high_resolution_clock::now();
		x1=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
		output_time[graph.id()] +=x1;




		// Call output accessors
		double x3=0;
		stream<<"graph_id:"<<graph.id()<<" calling outputs"<<std::endl;
		std::cerr<<stream.str();
		stream.str(std::string());
		detail::call_all_output_node_accessors(it->second);
		tfinish=std::chrono::high_resolution_clock::now();
		x3=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();

		stream<<"Graph"<<graph.id()<<"   Input: "<<x1*1000<<"   Task: "<<x2*1000<<"   Out: "<<x3*1000<<"   Proc: "<<(x2+x3)*1000<<std::endl;
		std::cerr<<stream.str();
		stream.str(std::string());
		output_time[graph.id()] +=x3;
	}*/






    int cc=warmup_n+(num_graphs-1)-graph.id();
    if(!parallel){
    	cc=warmup_n;
    }
    int n=3;
    for(int k=0; k<n;k++)
    {

    	/*I moved it to after loading input
    	 * if(!parallel)
    	{
			std::unique_lock<std::mutex> lck(_mtx);
			if(graph.id()==0 ){
				std::cerr<<"\n\n\n\n\n\n\nfirst graph is ready to run? "<<ready<<"\n\n\n\n\n";
				condVar_serial.wait(lck,[this] {return ready;});
				std::cerr<<"\n\n\n\n\n\n\nfirst graph start processing\n\n\n\n\n";
				ready=false;
			}
    	}*/

    	if(k==cc){
    		reset_timing(graph.id());
    		/*if(!parallel){
    			//start power measurement
    			std::cerr<<"non parallel or start with empty pipeline so just first stage synchronized\n";
    		}*/
    	}


		//If not pipeline (switch mode)
		if(!parallel)
		{
			if(graph.id()==0 ){
				std::unique_lock<std::mutex> lck(_mtx);
				//std::cerr<<"\n\n\n\n\n\n\nfirst graph is ready to run? "<<ready<<"\n\n\n\n\n";
				condVar_serial.wait(lck,[this] {return ready;});
				//std::cerr<<"\n\n\n\n\n\n*****************************\nFirst graph start processing \n**********************************\n\n\n\n";
				t_arriving=std::chrono::high_resolution_clock::now();
				ready=false;
			}
		}

    	// Call input accessors
		auto tstart=std::chrono::high_resolution_clock::now();
		//stream<<"graph_id:"<<graph.id()<<std::endl;
		//std::cerr<<stream.str();
		//stream<<"graph_id:"<<graph.id()<<" calling inputs"<<std::endl;
		//std::cerr<<stream.str();
		//stream.str(std::string());
        detail::call_all_input_node_accessors(it->second);

        //std::cerr<<"inputs called\n";
        auto tfinish=std::chrono::high_resolution_clock::now();
        double x1=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        //std::cerr<<"size of input_time: "<<input_time.size()<<std::endl;

        input_time[graph.id()] +=x1;



        //Call All receivers
        tstart=std::chrono::high_resolution_clock::now();
        //stream<<"graph_id:"<<graph.id()<<" calling receivers"<<std::endl;
        //std::cerr<<stream.str();
        //stream.str(std::string());
		detail::call_all_receivers(it->second);

		//std::cerr<<"receivers called\n";
		tfinish=std::chrono::high_resolution_clock::now();
		x1=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
		input_time[graph.id()] +=x1;




		/*
		 * All stages process the (cc=n_warmup+num_stages-graph_id(stage_id) ) frames and load input and wait
		 * the last stage which reach to this state(which is first stage, set pipeline_ready to true and notify all stages to start with
		 * full pipeline (which all stages already loaded their input. The timings are reset before this loading input(and receivers, but start of power measuerement and ...
		 * are start with task processing for first frame of all stages
		 * Another method, that works when there is no q between stages is that, when the last stage process n_warmup frame, then we can start power measurement
		 *
		 */
		if(k==cc){
			if(measure_when_full && parallel)
			{
				std::unique_lock<std::mutex> lck(_mtx);
				stream<<"\n\n\n\n\ngraph "<<graph.id()<<" wait after set frame "<<cc+1<<" in its input\n\n\n\n";
				std::cerr<<stream.str();
				stream.str(std::string());
				c=c+1;
				std::cerr<<"graph(stage) "<<graph.id()<<" num_stages: "<<num_graphs<<std::endl;
				if(c==num_graphs){
					//start power measurement
					std::cerr<<"stage "<<graph.id()<<" arrived later than all stages for frame "<<k<<"and will set ready to true after 2 seconds\n";
					std::cerr<<"\n\n\n**************\n start running (tasks) all partitions together after warm up\n************\n\n\n";
					std::this_thread::sleep_for(std::chrono::milliseconds(2000));
					pipeline_ready.store(true);
					condVar.notify_all();
				}
				else{
					condVar.wait(lck, [this]{ return (pipeline_ready.load()); });//or return(c==num_graphs)
				}
				lck.unlock();
			}
			//else if(graph.id()==0){
				//start power measurement
				//std::cerr<<"non parallel or start with empty pipeline so just first stage synchronized\n";
			//}
			//If switch mode (not pipeline)
			else{
				if(graph.id()==0){
					std::cerr<<"\n\n\n*****************************\nStart running (tasks) first partition for frame "<<k<<" after warm up\n**********************************\n\n\n";
				}
			}

		}


        // Run graph
		//std::cerr<<"\ntask:"<<task<<std::endl;
		//stream<<"graph_id:"<<graph.id()<<" Calling all tasks"<<std::endl;
		//std::cerr<<stream.str();
		//stream.str(std::string());
		tstart=std::chrono::high_resolution_clock::now();
		detail::call_all_tasks_pipeline(it->second,nn);
		tfinish=std::chrono::high_resolution_clock::now();
		double x2=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish-tstart).count();
		task_time[graph.id()] += x2;



		//Call All Senders
		tstart=std::chrono::high_resolution_clock::now();
		//stream<<"graph_id:"<<graph.id()<<" Calling all senders"<<std::endl;
		//std::cerr<<stream.str();
		//stream.str(std::string());
		transmition_time[graph.id()]+=detail::call_all_senders(it->second);

		//std::cerr<<"senders called\n";
		tfinish=std::chrono::high_resolution_clock::now();
		x1=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
		output_time[graph.id()] +=x1;




        // Call output accessors
		double x3=0;
		//stream<<"graph_id:"<<graph.id()<<" calling outputs"<<std::endl;
		//std::cerr<<stream.str();
		//stream.str(std::string());
        detail::call_all_output_node_accessors(it->second);
        tfinish=std::chrono::high_resolution_clock::now();

        x3=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        output_time[graph.id()] +=x3;

        if(!parallel)
		{
			if(graph.id()==num_graphs-1){
				t_completion=std::chrono::high_resolution_clock::now();
				latency_time+=std::chrono::duration_cast<std::chrono::duration<double>>(t_completion - t_arriving).count();
				std::cerr<<"Frame: "<<k<<" Latency: "<<latency_time<<"\n\n";
				std::unique_lock<std::mutex> lck(_mtx);
				ready=true;
				condVar_serial.notify_all();
			}
		}

        //stream<<"\n\nGraph"<<graph.id()<<"   Input: "<<x1*1000<<"   Task: "<<x2*1000<<"   Out: "<<x3*1000<<"   Proc: "<<(x2+x3)*1000<<"\n\n";
        stream<<"\n\nGraph"<<graph.id()<<"   Input: "<<input_time[graph.id()]*1000<<"   Task: "<<task_time[graph.id()]*1000<<"   Out: "<<output_time[graph.id()]*1000<<"\n\n";

        std::cerr<<stream.str();
        stream.str(std::string());
    }
}

/*If you want to buffer output of a stage if next stage is busy:
 * It also works with no buffer scenario
 */

void GraphManagerPipeline::warmup_and_execute_graph_serial(Graph &graph, int nn)
{

	std::stringstream stream;
	static std::chrono::time_point<std::chrono::high_resolution_clock> t_arriving;
	static std::chrono::time_point<std::chrono::high_resolution_clock> t_completion;

    auto it = _workloads.find(graph.id());
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");

    int	Starting_frame=warmup_n;
    for(int Frame=0; Frame<nn+warmup_n;Frame++)
    {
    	//WarmUp Finished, Reset Timings
    	if(Frame==Starting_frame){
    		//std::cerr<<"Reset timings\n\n";
    		reset_timing(graph.id());
    	}

    	//First Graph waits for Finishing Last Graph
		if(graph.id()==0 ){
			std::unique_lock<std::mutex> lck(_mtx);
			condVar_serial.wait(lck,[this] {return ready;});

			if (-1 == GPIOWrite(POUT, 1)){
				std::cerr<<"could not write 1\n";
			}

			t_arriving=std::chrono::high_resolution_clock::now();
			ready=false;
		}
		//std::cerr<<"graph "<<graph.id()<<" before inputs\n";
		//Inputs
		auto tstart=std::chrono::high_resolution_clock::now();
        detail::call_all_input_node_accessors(it->second);
        auto tfinish=std::chrono::high_resolution_clock::now();
        double t_input=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        input_time[graph.id()] +=t_input;

        //std::cerr<<"graph "<<graph.id()<<" before recs\n";
        //Receivers
        tstart=std::chrono::high_resolution_clock::now();
		detail::call_all_receivers(it->second);
		tfinish=std::chrono::high_resolution_clock::now();
		double t_receive=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
		receive_time[graph.id()] +=t_receive;

		//std::cerr<<"graph "<<graph.id()<<" before tasks\n";
        // Run graph Tasks
		tstart=std::chrono::high_resolution_clock::now();
		detail::call_all_tasks_pipeline(it->second,nn);
		tfinish=std::chrono::high_resolution_clock::now();
		double t_run=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish-tstart).count();
		task_time[graph.id()] += t_run;

		///std::cerr<<"graph "<<graph.id()<<" before sends\n";
		//Senders
		tstart=std::chrono::high_resolution_clock::now();
		transmition_time[graph.id()]+=detail::call_all_senders(it->second);
		tfinish=std::chrono::high_resolution_clock::now();
		auto t_send=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
		send_time[graph.id()] +=t_send;

		//std::cerr<<"graph "<<graph.id()<<" before outs\n";
		//Outputs
		tstart=std::chrono::high_resolution_clock::now();
        detail::call_all_output_node_accessors(it->second);
        tfinish=std::chrono::high_resolution_clock::now();
        double t_out=std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        output_time[graph.id()] +=t_out;

        //std::cerr<<"graph "<<graph.id()<<" before finishs\n";
        //Last graph finished
		if(graph.id()==num_graphs-1){
			t_completion=std::chrono::high_resolution_clock::now();
			auto l=std::chrono::duration_cast<std::chrono::duration<double>>(t_completion - t_arriving).count();
			latency_time+=l;
			//std::cerr<<"\nFrame: "<<Frame<<" Latency: "<<1000*l<<"\n\n";
			std::unique_lock<std::mutex> lck(_mtx);
			ready=true;
			/*if (-1 == GPIOWrite(POUT, 0)){
				std::cerr<<"last graph could not write 0\n";
			}*/
			condVar_serial.notify_all();

		}


        /*stream<<"\n\nGraph"<<graph.id()<<
        		"   Input: "<<(t_input*1000)<<
				//"   Receive: "<<(receive_time[graph.id()]*1000)/num_run<<
        		"   Task: "<<(t_run*1000)<<
				"   send: "<<(t_send*1000)<<
				"   Out: "<<(t_out*1000)<<
				"   Process: "<< ( ( t_input + t_run + t_send + t_out ) * 1000 )  <<
				"\n\n";

        std::cerr<<stream.str();
        stream.str(std::string());*/
    }
    //std::cerr<<"graph "<<graph.id()<<" before end\n";
    //After finishing it sets GPIO to 0 for the experiment
    if(graph.id()==num_graphs-1){
    	std::this_thread::sleep_for(std::chrono::milliseconds(8));
    	if (-1 == GPIOWrite(POUT, 0)){
    		std::cerr<<"could not write 0\n";
    	}
    	std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

}


//Ehsan

void GraphManagerPipeline::print_times(int n)
{
	for(int id=0;id<num_graphs; id++){
		print_times_details(id,n);

		print_times(id, n);

		auto &workload = _workloads.find(id)->second;
		detail::print_NPU_times(workload,n);
		std::cout<<"\n\n********\n\n";


	}


	 std::cerr<<"\n\nAVG Latency: "<<1000*latency_time/n<<"\n\n";
}

void GraphManagerPipeline::print_times(int graph_id, int n)
{
	std::stringstream stream;
	stream<<"\n\nGraph"<<graph_id<<
			"   Input: "<<(input_time[graph_id]*1000)/n<<
			//"   Receive: "<<(receive_time[graph.id()]*1000)/num_run<<
			"   Task: "<<(task_time[graph_id]*1000)/n<<
			"   send: "<<(send_time[graph_id]*1000)/n<<
			"   Out: "<<(output_time[graph_id]*1000)/n<<
			"   Process: "<< ( ( input_time[graph_id] + task_time[graph_id] + send_time[graph_id] + output_time[graph_id] ) * 1000 ) / n <<
			"\n\n";
	std::cout<<stream.str();
}

void GraphManagerPipeline::print_times_details(int graph_id, int n)
{
	auto it = _workloads.find(graph_id);
	ExecutionWorkload *workload = &it->second;
	double sum=0;
	int c=0;
	int l=0;
	double tt=0;
	for(auto &task:workload->tasks){
		if(!task.task){
			//std::cerr<<"nadareeeeeeeee\n";
			continue;
		}
		std::cout<<c++<<"\tLayer Name: "<<task.node->name()
				<<" \t Layer time: "<<task.time(n)
				<<" \t number of inputs: "<<task.node->num_inputs()
				<<" \t input shape: "<<task.node->input(0)->desc().shape
				<<" \t output shape: "<<task.node->output(0)->desc().shape<<std::endl;

		tt+=task.time(n);
		if(task.ending){
			std::cout<<"Layer Number: "<<l<<" \t time: "<<tt<<std::endl;
			tt=0;
			l++;
			std::cout<<"----------------------------\n";
		}
		sum+=task.time(n);
	}
	std::cout<<"\n Sum of Layers time: "<<sum<<std::endl;
}

void GraphManagerPipeline::reset(Graph &graph)
{
	auto it = _workloads.find(graph.id());
	ExecutionWorkload *workload = &it->second;
	for(auto &task:workload->tasks){
		task.reset();
	}
}

void GraphManagerPipeline::print_tasks(){
		//std::vector<std::string> task_names;
	    std::stringstream tasks_names;
	    std::stringstream starting_tasks_names;
	    std::stringstream ending_tasks_names;
	    tasks_names<<"std::string task_names[] = { ";
	    starting_tasks_names<<"std::set<std::string> starting_task_names = { ";
	    ending_tasks_names<<"std::set<std::string> ending_task_names = { ";
	    int num_tasks=0, num_starting_tasks=0, num_ending_tasks=0;
	    char sep='\n';
		for (unsigned int id = 0; id < _workloads.size(); ++id) {
			auto &workload = _workloads.find(id)->second;
			for(auto& task:workload.tasks){
				tasks_names<<"\""<<task.node->name()<<"\","<<sep;
				num_tasks++;
				if(task.starting){
					starting_tasks_names<<"\""<<task.node->name()<<"\","<<sep;
					num_starting_tasks++;
				}
				if(task.ending){
					ending_tasks_names<<"\""<<task.node->name()<<"\","<<sep;
					num_ending_tasks++;
				}
			}
			//std::cerr<<"governor:"<<workload.tasks[n-1].node->name()<<std::endl;
		}
		tasks_names.seekp(-2, std::ios_base::end);
		tasks_names<<" };\n";
		std::cerr<<"Total tasks "<<num_tasks<<std::endl;
		std::cerr<<tasks_names.str();

		starting_tasks_names.seekp(-2, std::ios_base::end);
		starting_tasks_names<<" };\n";
		std::cerr<<"\nStarting tasks "<<num_starting_tasks<<std::endl;
		std::cerr<<starting_tasks_names.str();

		ending_tasks_names.seekp(-2, std::ios_base::end);
		ending_tasks_names<<" };\n";
		std::cerr<<"\nEnding tasks "<<num_ending_tasks<<std::endl;
		std::cerr<<ending_tasks_names.str();
}

/*
starting:
conv2d_58
conv2d_59
conv2d_66
conv2d_67
ending:
conv2d_58/LeakyRelu
Yolo1
conv2d_66/LeakyRelu
Yolo2*/
bool isInList(const std::vector<std::string>& list, const std::string& name) {
    return std::find(list.begin(), list.end(), name) != list.end();
}
void GraphManagerPipeline::set_GPIO_tasks(std::string power_profie_mode){
	std::vector<std::string> not_starting = {"conv2d_58", "conv2d_59", "conv2d_66", "conv2d_67","NPU_YOLOv3_57_57","NPU_YOLOv3_58_58","NPU_YOLOv3_65_65","NPU_YOLOv3_66_66"};
	std::vector<std::string> not_ending = {"conv2d_58/LeakyRelu", "Yolo1", "conv2d_66/LeakyRelu", "Yolo2", "NPU_YOLOv3_57_57","NPU_YOLOv3_58_58","NPU_YOLOv3_65_65","NPU_YOLOv3_66_66"};
	std::cerr<<"set_GPIOs with mode: "<<power_profie_mode<<"\n";
	if(power_profie_mode=="layers"){
		for (unsigned int id = 0; id < _workloads.size(); ++id) {
			auto &workload = _workloads.find(id)->second;
			int n=workload.tasks.size();
			for(auto& task:workload.tasks){
				task.starting_gpio_switch=task.starting;
				task.ending_gpio_switch=task.ending;
				task.profile_layers=true;
				task.profile_transfers=false;
			}
			//Last task will not switch gpio nor set freq (but after output it happens)
			/*if(id==_workloads.size()-1){
				workload.tasks[n-1].ending_gpio_switch=false;
			}*/
			//std::cerr<<"governor:"<<workload.tasks[n-1].node->name()<<std::endl;
		}
	}
	if(power_profie_mode=="transfers"){
		for (unsigned int id = 0; id < _workloads.size(); ++id) {
			auto &workload = _workloads.find(id)->second;
			int n=workload.tasks.size();
			for(auto& task:workload.tasks){
				task.starting_gpio_switch=task.starting;
				task.ending_gpio_switch=task.ending;
				task.profile_layers=false;
				task.profile_transfers=true;
			}
			//std::cerr<<"governor:"<<workload.tasks[n-1].node->name()<<std::endl;
		}
	}
	if(power_profie_mode=="whole"){
		for (unsigned int id = 0; id < _workloads.size(); ++id) {
			auto &workload = _workloads.find(id)->second;
			int n=workload.tasks.size();
			for(auto& task:workload.tasks){
				task.starting_gpio_switch=false;
				task.ending_gpio_switch=false;
				task.profile_layers=false;
				task.profile_transfers=false;
			}
			//std::cerr<<"governor:"<<workload.tasks[n-1].node->name()<<std::endl;
		}
		{
			auto &workload = _workloads.find(0)->second;
			workload.tasks[0].starting_gpio_switch=true;
		}
		{
			auto &workload = _workloads.find(_workloads.size()-1)->second;
			int n=workload.tasks.size();
			std::cerr<<"set ending of GPIO for task "<<workload.tasks[n-1].node->name()<<std::endl;
			workload.tasks[n-1].ending_gpio_switch=true;
		}
	}

	bool simultaneously=false;

	auto &_workload = _workloads.find(0)->second;
	if (_workload.graph->name()=="YOLOv3" && power_profie_mode!="whole"){

		for (unsigned int id = 0; id < _workloads.size(); ++id) {
			auto &workload = _workloads.find(id)->second;
			int n=workload.tasks.size();
			for(auto& task:workload.tasks){
				if (isInList(not_starting, task.node->name())){
					task.starting_gpio_switch=false;
					std::cerr<<"change starting gpio to false for task:"<<task.node->name()<<std::endl;
				}
				if (isInList(not_ending, task.node->name())){
					task.ending_gpio_switch=false;
					std::cerr<<"change ending gpio to false for task:"<<task.node->name()<<std::endl;
				}
			}
		}
	}

	int i=0;
	for (unsigned int id = 0; id < _workloads.size(); ++id) {
		auto &workload = _workloads.find(id)->second;
		int n=workload.tasks.size();
		for(auto& task:workload.tasks){
			std::cerr<<i++<<" task name: "<<task.node->name()<<"\t";
			std::cerr<<"starting gpio: "<<task.starting_gpio_switch<<"\t";
			std::cerr<<"ending gpio: "<<task.ending_gpio_switch<<"\t";
			std::cerr<<"profiling layers: "<<task.profile_layers<<"\t";
			std::cerr<<"profiling transfers: "<<task.profile_transfers<<"\n";
		}
		//std::cerr<<"governor:"<<workload.tasks[n-1].node->name()<<std::endl;
	}


}

void GraphManagerPipeline::extract_governor_tasks(std::string mode){
	std::vector<std::string> not_governor = {"conv2d_58/LeakyRelu", "Yolo1", "conv2d_66/LeakyRelu", "Yolo2","NPU_YOLOv3_57_57","NPU_YOLOv3_58_58","NPU_YOLOv3_65_65","NPU_YOLOv3_66_66"};
	governor_tasks.clear();
	if(mode=="layers"){
		for (unsigned int id = 0; id < _workloads.size(); ++id) {
			auto &workload = _workloads.find(id)->second;
			detail::set_governor_tasks(workload, &governor_tasks);
			/*//Last tasks of the last sub-graph will not change freq nor change GPIO
			//But after calling its outputs it will govern freqs and change GPIO
			if(id==_workloads.size()-1){
				workload.tasks[workload.tasks.size()].governor=false;
			}*/
		}
	}
	if(mode=="graphs"){
		for (unsigned int id = 0; id < _workloads.size(); ++id) {
			auto &workload = _workloads.find(id)->second;
			int n=workload.tasks.size();
			for(auto& task:workload.tasks){
				task.governor=false;
			}
			workload.tasks[n-1].governor=true;
			governor_tasks.push_back(workload.tasks[n-1].node->name());
			//std::cerr<<"governor:"<<workload.tasks[n-1].node->name()<<std::endl;
		}
	}
	if(mode=="PEs"){
		for (unsigned int id = 0; id < _workloads.size(); ++id) {
			auto &workload = _workloads.find(id)->second;
			int n=workload.tasks.size();
			for(auto& task:workload.tasks){
				task.governor=false;
			}
			if(id==_workloads.size()-1){
				workload.tasks[n-1].governor=true;
				governor_tasks.push_back(workload.tasks[n-1].node->name());
			}
		}
	}
	auto &_workload = _workloads.find(0)->second;
	if (_workload.graph->name()=="YOLOv3"){
		for (unsigned int id = 0; id < _workloads.size(); ++id) {
			auto &workload = _workloads.find(id)->second;
			int n=workload.tasks.size();
			for(auto& task:workload.tasks){
				if (isInList(not_governor, task.node->name())){
					task.governor=false;
					std::cerr<<"change governor to false for task:"<<task.node->name()<<std::endl;
					/*auto it = std::find(governor_tasks.begin(), governor_tasks.end(), task.node->name());
					if (it != governor_tasks.end()) {
						governor_tasks.erase(it);
					}*/
				}
			}
		}
	}
	int k=0;
	std::cerr<<governor_tasks.size()<<std::endl;
	for(auto task:governor_tasks){
		std::cerr<<"\n"<<k++<<" gov task: "<<task;
	}

}



std::string removeConsecutiveDuplicates(const std::string& input) {
    std::string result;
    for (size_t i = 0; i < input.length(); ++i) {
        // Only add the current character to result if it's different from the last character in result
        if (result.empty() || input[i] != result.back()) {
            result += input[i];
        }
    }
    return result;
}

void GraphManagerPipeline::set_freqs(std::string freqs, std::string _order, char GPU_Host, char NPU_Host){
	governor_freqs.clear();
	if(freqs==""){
		return;
	}
	if (freqs.size() >= 2 && freqs.front() == '{' && freqs.back() == '}') {
		freqs = freqs.substr(1, freqs.size() - 2);
		if (freqs.size() >= 2 && freqs.front() == '{' && freqs.back() == '}'){
			freqs = freqs.substr(1, freqs.size() - 2);
			extract_governor_tasks("PEs");
			if(freqs=="min" or freqs=="[min]"){
					for(auto task :governor_tasks){
						governor_freqs[task]={0,0,0};
					}
				}
			else if(freqs=="max" or freqs=="[max]"){
					for(auto task :governor_tasks){
						governor_freqs[task]={5,7,4};
					}
				}
				//std::cerr<<"\n\n\n\n*************\nfrerqs are:"<<freqs<<std::endl;
			else{
				int l, b, g;
				std::replace(freqs.begin(), freqs.end(), '-', ' '); // Replace '-' with whitespace
				std::istringstream iss(freqs);
				if (!(iss >> l >> b >> g)) {
					std::cerr << "Parsing error!" << std::endl;
					return;
				}
				governor_freqs[governor_tasks[0]]={l,b,g};
			}
		}
		else{
			extract_governor_tasks("graphs");
			set_freq_map(freqs, removeConsecutiveDuplicates(_order), GPU_Host, NPU_Host);
		}
	}
	else{
		extract_governor_tasks("layers");
		set_freq_map(freqs, _order, GPU_Host, NPU_Host);
	}
	set_governor_freqs();
}
void GraphManagerPipeline::set_freq_map(std::string freqs, std::string _order, char GPU_Host, char NPU_Host){
	if(freqs=="min" or freqs=="[min]"){
		extract_governor_tasks("PEs");
		for(auto task :governor_tasks){
			governor_freqs[task]={0,0,0};
		}
		return;
	}
	if(freqs=="max" or freqs=="[max]"){
		extract_governor_tasks("PEs");
		for(auto task :governor_tasks){
			governor_freqs[task]={5,7,4};
		}
		return;
	}
	//TODO: Set just in use processor to its max fequency
	if(freqs=="max" or freqs=="[max]"){
		int i=0;
		bool Prev_NPU=false;
		for(auto task :governor_tasks){
			if(_order[i]=='L'){
				governor_freqs[task]={arm_compute::graph::ExecutionTask::get_max_l(),0,0};
			}
			if(_order[i]=='B'){
				governor_freqs[task]={0,arm_compute::graph::ExecutionTask::get_max_b(),0};
			}
			if(_order[i]=='G'){
				if(GPU_Host=='B'){
					governor_freqs[task]={0,
										arm_compute::graph::ExecutionTask::get_max_b(),
										arm_compute::graph::ExecutionTask::get_max_g()};
				}
				if(GPU_Host=='L'){
					governor_freqs[task]={arm_compute::graph::ExecutionTask::get_max_l(),
										0,
										arm_compute::graph::ExecutionTask::get_max_g()};
				}
			}
			if(_order[i]=='N'){
				if(NPU_Host=='B'){
					governor_freqs[task]={0,
										arm_compute::graph::ExecutionTask::get_max_b(),
										0};
				}
				if(NPU_Host=='L'){
					governor_freqs[task]={arm_compute::graph::ExecutionTask::get_max_l(),
										0,
										0};
				}
				while( i < (_order.size()-1 ) ){
					if (_order[i]==_order[i+1]){
						i=i+1;
					}
					else{
						break;
					}
				}
			}
			i++;
			//governor_freqs[task]={arm_compute::graph::ExecutionTask::get_max_l(),\
					arm_compute::graph::ExecutionTask::get_max_b(),\
					arm_compute::graph::ExecutionTask::get_max_g()};
		}
		return;
	}

	std::stringstream ss(freqs);
	std::string token;
	//For iterating over orders mapping letters
	int i=_order.size()-1;
	//for iterating over governor_tasks (for consequitive Ns in order there is one gevernor task just)
	int governor_task_index=governor_tasks.size()-1;
	int _method=1;
	char prev_p;
	while (std::getline(ss, token, '-')) {
		//freq_layer[*it++] = std::stoi(token);
		//std::cerr<<"token is:"<<token<<std::endl;
		int j=(i+1)%_order.size();
		int l=0,b=0,g=0;
		char p=_order[j];

		if (token.find('[')!=std::string::npos){
			if(p!='G'){
				std::cerr<<"Error\n\n\n\n";
			}
			token.erase(0, 1);
			token.erase(token.size() - 1, 1);
			std::stringstream t(token);
			char comma;

			(GPU_Host=='B')?t >> g >> comma >> b:t >> g >> comma >> l;
		}
		else{
			if (p=='L'){
				l=std::stoi(token);
				//b=7;
			}
			if (p=='B'){
				b=std::stoi(token);
				//l=5;
			}
			if (p=='N'){
				(NPU_Host=='B')?b=std::stoi(token):l=stoi(token);
				if(_method==2){
					while( i < (_order.size()-1 ) ){
						if (_order[i]==_order[i+1]){
							//std::getline(ss, token, '-');
							i=i+1;
						}
						else{
							break;
						}
					}
				}
				//l=5;
			}
		}
		//std::cerr<<i<<":"<<governor_tasks[i]<<std::endl;
		if(_method==1){
			if(p=='N' && p==prev_p){
				std::cerr<<"set freq skipping layer "<<i<<" which is mapped on NPU"<<std::endl;
			}
			else{
				std::cerr<<"\n** i:"<<i<<" governor index:"<<governor_task_index<<" "<<governor_tasks[governor_task_index]<<std::endl;
				governor_freqs[governor_tasks[governor_task_index]]={l,b,g};
				governor_task_index++;
				governor_task_index=governor_task_index%governor_tasks.size();
			}
			prev_p=p;
		}
		if(_method==2){
			std::cerr<<"**"<<i<<" "<<governor_tasks[i]<<std::endl;
			governor_freqs[governor_tasks[i]]={l,b,g};
		}
		i=i+1;
		i=i%(_order.size());
	}
	/*for(auto it: _end_task_names){
		std::cout<<"layer:"<<it<<"\tLittle:"<<freq_layer[it][0]<<"\tbig:"<<freq_layer[it][1]<<"\tGPU:"<<freq_layer[it][2]<<std::endl;
	}*/
}



void GraphManagerPipeline::set_governor_freqs(){
	int i=0;
	for(auto p : governor_freqs){
		std::cerr<<"governor "<<i++<<" task:"<<p.first<<" freqs:"<<p.second[0]<<","<<p.second[1]<<","<<p.second[2]<<std::endl;
		//std::cerr<<"governor "<<i++<<std::endl;
	}
	for (unsigned int id = 0; id < _workloads.size(); ++id) {
		auto &workload = _workloads.find(id)->second;
		for(auto& task:workload.tasks){
	    	if(!task.task)
	    		continue;
	    	if(task.ending)
	    		std::cerr<<id<<"---- "<<task.node->name()<<"    processor:"<<task.processor<<std::endl;
	    	if(task.governor){
	    		std::cerr<<"setting freq of governor task "<<task.node->name()<<std::endl;
	    		task.set_freq(governor_freqs.at(task.node->name()));
	    	}
		}
	}
}

int GraphManagerPipeline::destroy(){
	int ret=0;
	for (unsigned int id = 0; id < _workloads.size(); ++id) {
		auto &workload = _workloads.find(id)->second;
		ret=ret | detail::NPU_destroy(workload);
	}
	return ret;
}



} // namespace graph
} // namespace arm_compute
