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
#include "arm_compute/graph/nodes/ReceiverNode.h"
#include "arm_compute/graph/nodes/SenderNode.h"

namespace arm_compute
{
namespace graph
{
GraphManagerPipeline::GraphManagerPipeline()
    : GraphManager()
{
    set_num_graphs(1);
    _pipeline_ready.store(false);
    _measure_when_full = true;
    _parallel          = false;
}

void GraphManagerPipeline::finalize_graph(Graph &graph, GraphContext &ctx, PassManager &pm, Target target, std::set<int> *blocking_set, int blocking)
{
    // Check if graph has been registered
    if(_workloads.find(graph.id()) != std::end(_workloads))
    {
        ARM_COMPUTE_ERROR("Graph is already registered!");
    }

    pm.run_type(graph, IGraphMutator::MutationType::IR);

    // Force target to all graph construct
    // TODO (COMPMID-2014) : Support heterogeneous execution
    Target forced_target = target;
    if(!is_target_supported(target))
    {
        //Ehsan
        forced_target = get_default_target();
        ARM_COMPUTE_LOG_GRAPH_INFO("Switching target from " << target << " to " << forced_target << std::endl);
    }
#if My_print > -1
    std::cerr << "Graph id: " << graph.id() << " Target is: " << forced_target << std::endl;
#endif

    force_target_to_graph(graph, forced_target);
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

    detail::configure_all_tensors(graph);

    /* if you enable this(pm.run_type) it leads to segmentation fault at workload.cpp execute_task function in line: task.task->run();*/
    pm.run_type(graph, IGraphMutator::MutationType::Backend);

    if(graph.id() == 2)
    {
        DotGraphPrinter p;
        p.print(graph, std::cout);
    }

    // Perform topological sort
    std::vector<NodeID> topological_sorted_nodes = dfs(graph);

    //It add npu node twice when there are two edges from input to npu node for example
    //GLLLLNNNNNNNNNNLLLLLLLLLNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNLNLLLLLLLLL (graph2)
    //so remove the repatative nodes in this vector when target in npu
    if(forced_target == arm_compute::graph::Target::NPU)
    {
        std::set<int>       seen;
        std::vector<NodeID> uniqueVec;
        for(int elem : topological_sorted_nodes)
        {
            if(seen.insert(elem).second)
            {
                uniqueVec.push_back(elem);
            }
        }
        topological_sorted_nodes.clear();
        for(int elem : uniqueVec)
        {
            topological_sorted_nodes.push_back(elem);
        }

        std::cerr << "topological node counts original graph: " << topological_sorted_nodes.size() << std::endl;
        for(auto &node : topological_sorted_nodes)
        {
            if(node)
                std::cerr << "after topo node " << graph.node(node)->name() << "\n";
        }
    }

    // Validate all nodes
    detail::validate_all_nodes(graph);
    // Configure all nodes

    auto workload = detail::configure_all_nodes_pipeline(graph, ctx, topological_sorted_nodes);
    ARM_COMPUTE_ERROR_ON_MSG(workload.tasks.empty(), "Could not configure all nodes!");
#if My_print > 0
    //Ehsan
    std::cout << "\nGraphManager, outputs size:" << workload.outputs.size() << std::endl;
#endif
    // Allocate const tensors and call accessors

    detail::allocate_const_tensors_pipeline(graph);
    detail::call_all_const_node_accessors(graph);

    int ii = 0;
    if(blocking_set != NULL)
    {
        for(auto &task : workload.tasks)
        {
            if(!task.task)
                continue;
            bool b = false;
            if(blocking_set->find(ii) != blocking_set->end())
            {
                b           = true;
                task.ending = true;
            }
            if(blocking == 1)
            {
                if(blocking_set != NULL and b && target == arm_compute::graph::Target ::CL)
                    task.block = 1;
            }
            if(blocking == 2)
            {
                if(blocking_set != NULL && target == arm_compute::graph::Target ::CL)
                {
                    task.block = 1;
                }
            }

            ii++;
        }
    }

    if(target == arm_compute::graph::Target ::CL)
    {
        workload.tasks[workload.tasks.size() - 1].block = 1;
    }

#if My_print > 0
    //Ehsan
    DotGraphPrinter p;
    p.print(graph, std::cout);
#endif

    if(ctx.config().use_transition_memory_manager)
    {
#if My_print > 0
        //Ehsan
        std::cerr << "transition memory mangaer is used\n";
#endif

        detail::configure_transition_manager(graph, ctx, workload);
    }
    else
    {
        detail::allocate_all_tensors(graph);
    }
    // Finalize Graph context
    ctx.finalize();
    // Register graph
    std::lock_guard<std::mutex> lock(_mtx);
    detail::NPU_set_preallocated_outputs(workload);
    detail::set_ending_tasks(workload);
    blocking = 1;
    for(auto &task : workload.tasks)
    {
        if(!task.task)
            continue;

        if(blocking == 1)
        {
            if(task.ending && target == arm_compute::graph::Target ::CL)
                task.block = 1;
        }
        if(blocking == 2)
        {
            if(target == arm_compute::graph::Target ::CL)
            {
                task.block = 1;
            }
        }

        ii++;
    }

    char processor = 'B';
    if(ctx.config().cluster == 0)
    {
        processor = 'L';
    }
    if(target == arm_compute::graph::Target ::CL)
    {
        processor = 'G';
    }
    if(target == arm_compute::graph::Target ::NPU)
    {
        processor = 'N';
    }
    for(auto &task : workload.tasks)
    {
        task.processor = processor;
    }

    _workloads.insert(std::make_pair(graph.id(), std::move(workload)));
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Created workload for graph with ID : " << graph.id() << std::endl);
}
void GraphManagerPipeline::reset_timing(int graph_id)
{
    auto               it       = _workloads.find(graph_id);
    ExecutionWorkload *workload = &it->second;
    for(auto &task : workload->tasks)
    {
        task.reset();
    }
    _input_time[graph_id]       = 0;
    _receive_time[graph_id]     = 0;
    _task_time[graph_id]        = 0;
    _send_time[graph_id]        = 0;
    _output_time[graph_id]      = 0;
    _transmition_time[graph_id] = 0;
    _latency_time               = 0;
    detail::reset_transmit_timings(it->second);
    detail::reset_NPU_timings(it->second);
}

void GraphManagerPipeline::execute_graph(Graph &graph, int nn)
{
    std::stringstream stream;
    stream << "start of execute graph " << graph.id() << " in graph manager\n";
    std::cerr << stream.str();
    stream.str(std::string());
    auto it = _workloads.find(graph.id());
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");
    //Ehsan measure input, task and output timings:

    unsigned int n = 4;
    for(unsigned int k = 0; k < n; k++)
    {
        if(_measure_when_full)
        {
            if(k == _num_graphs)
            {
                auto t1 = std::chrono::high_resolution_clock::now();
                reset_timing(graph.id());
                auto        t2 = std::chrono::high_resolution_clock::now();
                double      x1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
                std::string s  = "timing reset (for reseting measurement when pipeline is full) took " + std::to_string(x1 * 1000) + "ms\n";
                std::cerr << s;
            }
        }
        // Call input accessors
        auto tstart = std::chrono::high_resolution_clock::now();
        stream << "graph_id:" << graph.id() << " calling inputs" << std::endl;
        std::cerr << stream.str();
        stream.str(std::string());
        detail::call_all_input_node_accessors(it->second);

        std::cerr << "input called\n";
        auto   tfinish = std::chrono::high_resolution_clock::now();
        double x1      = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        std::cerr << "size of input_time: " << _input_time.size() << std::endl;

        _input_time[graph.id()] += x1;

        //Call All receivers
        tstart = std::chrono::high_resolution_clock::now();
        stream << "graph_id:" << graph.id() << " calling receivers" << std::endl;
        std::cerr << stream.str();
        stream.str(std::string());
        detail::call_all_receivers(it->second);

        std::cerr << "receivers called\n";
        tfinish = std::chrono::high_resolution_clock::now();
        x1      = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        _input_time[graph.id()] += x1;

        // Run graph
        stream << "graph_id:" << graph.id() << " Calling all tasks" << std::endl;
        std::cerr << stream.str();
        stream.str(std::string());
        detail::call_all_tasks_pipeline(it->second, nn);
        tstart    = std::chrono::high_resolution_clock::now();
        double x2 = std::chrono::duration_cast<std::chrono::duration<double>>(tstart - tfinish).count();
        _task_time[graph.id()] += x2;

        //Call All Senders
        tstart = std::chrono::high_resolution_clock::now();
        stream << "graph_id:" << graph.id() << " Calling all senders" << std::endl;
        std::cerr << stream.str();
        stream.str(std::string());
        _transmition_time[graph.id()] += detail::call_all_senders(it->second);

        std::cerr << "senders called\n";
        tfinish = std::chrono::high_resolution_clock::now();
        x1      = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        _output_time[graph.id()] += x1;

        // Call output accessors
        double x3 = 0;
        stream << "graph_id:" << graph.id() << " calling outputs" << std::endl;
        std::cerr << stream.str();
        stream.str(std::string());
        detail::call_all_output_node_accessors(it->second);
        tfinish = std::chrono::high_resolution_clock::now();
        x3      = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();

        stream << "Graph" << graph.id() << "   Input: " << x1 * 1000 << "   Task: " << x2 * 1000 << "   Out: " << x3 * 1000 << "   Proc: " << (x2 + x3) * 1000 << std::endl;
        std::cerr << stream.str();
        stream.str(std::string());
        _output_time[graph.id()] += x3;
    }
}

/*
 * If there is no buffer in between stages
 */
void GraphManagerPipeline::warmup_and_execute_graph_no_buffer(Graph &graph, int nn)
{
    // Check if graph is finalized
    //std::stringstream stream;
    //stream<<"start of execute graph "<<graph.id()<<" in graph manager\n";
    //stream.str(std::string());
    auto it = _workloads.find(graph.id());
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");

    int cc = _warmup_n + (_num_graphs - 1) - graph.id();
    int n  = 10;
    for(int k = 0; k < n; k++)
    {
        if(k == cc)
        {
            reset_timing(graph.id());
            //Because there is no buffering so till last stage did not get frame [warmup_n] from previous stage, the previous stage cannot start processing next frame
            if(graph.id() == _num_graphs - 1)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(20000));
            }
            if(!_parallel)
            {
                //start power measurement
            }
        }
        // Call input accessors
        auto tstart = std::chrono::high_resolution_clock::now();
        detail::call_all_input_node_accessors(it->second);

        auto   tfinish = std::chrono::high_resolution_clock::now();
        double x1      = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();

        _input_time[graph.id()] += x1;

        //Call All receivers
        tstart = std::chrono::high_resolution_clock::now();
        detail::call_all_receivers(it->second);

        tfinish = std::chrono::high_resolution_clock::now();
        x1      = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        _input_time[graph.id()] += x1;

        // Run graph
        detail::call_all_tasks_pipeline(it->second, nn);
        tstart    = std::chrono::high_resolution_clock::now();
        double x2 = std::chrono::duration_cast<std::chrono::duration<double>>(tstart - tfinish).count();
        _task_time[graph.id()] += x2;

        //Call All Senders
        tstart = std::chrono::high_resolution_clock::now();
        _transmition_time[graph.id()] += detail::call_all_senders(it->second);
        tfinish = std::chrono::high_resolution_clock::now();
        x1      = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        _output_time[graph.id()] += x1;

        // Call output accessors
        double x3 = 0;
        detail::call_all_output_node_accessors(it->second);
        tfinish = std::chrono::high_resolution_clock::now();
        x3      = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();

        _output_time[graph.id()] += x3;
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
    std::stringstream                                           stream;
    std::chrono::time_point<std::chrono::high_resolution_clock> t_arriving;
    std::chrono::time_point<std::chrono::high_resolution_clock> t_completion;
    //stream<<"start of warmup and execute graph "<<graph.id()<<" in graph manager\n";
    //stream<<"number of workloads: "<<_workloads.size()<<std::endl;
    //std::cerr<<stream.str();
    //stream.str(std::string());
    auto it = _workloads.find(graph.id());
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");

    int cc = _warmup_n + (_num_graphs - 1) - graph.id();
    if(!_parallel)
    {
        cc = _warmup_n;
    }
    int n = 3;
    for(int k = 0; k < n; k++)
    {
        if(k == cc)
        {
            reset_timing(graph.id());
        }

        //If not pipeline (switch mode)
        if(!_parallel)
        {
            if(graph.id() == 0)
            {
                std::unique_lock<std::mutex> lck(_mtx);
                _condVar_serial.wait(lck, [this]
                                     { return _ready; });
                t_arriving = std::chrono::high_resolution_clock::now();
                _ready     = false;
            }
        }

        // Call input accessors
        auto tstart = std::chrono::high_resolution_clock::now();
        detail::call_all_input_node_accessors(it->second);

        auto   tfinish = std::chrono::high_resolution_clock::now();
        double x1      = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();

        _input_time[graph.id()] += x1;

        //Call All receivers
        tstart = std::chrono::high_resolution_clock::now();
        detail::call_all_receivers(it->second);

        //std::cerr<<"receivers called\n";
        tfinish = std::chrono::high_resolution_clock::now();
        x1      = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        _input_time[graph.id()] += x1;

        /*
		 * All stages process the (cc=n_warmup+num_stages-graph_id(stage_id) ) frames and load input and wait
		 * the last stage which reach to this state(which is first stage, set pipeline_ready to true and notify all stages to start with
		 * full pipeline (which all stages already loaded their input. The timings are reset before this loading input(and receivers, but start of power measuerement and ...
		 * are start with task processing for first frame of all stages
		 * Another method, that works when there is no q between stages is that, when the last stage process n_warmup frame, then we can start power measurement
		 *
		 */
        if(k == cc)
        {
            if(_measure_when_full && _parallel)
            {
                std::unique_lock<std::mutex> lck(_mtx);
                _c = _c + 1;
                if(_c == _num_graphs)
                {
                    //start power measurement
                    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                    _pipeline_ready.store(true);
                    _condVar.notify_all();
                }
                else
                {
                    _condVar.wait(lck, [this]
                                  { return (_pipeline_ready.load()); }); //or return(c==num_graphs)
                }
                lck.unlock();
            }
            //If switch mode (not pipeline)
            else
            {
                if(graph.id() == 0)
                {
                }
            }
        }

        // Run graph
        tstart = std::chrono::high_resolution_clock::now();
        detail::call_all_tasks_pipeline(it->second, nn);
        tfinish   = std::chrono::high_resolution_clock::now();
        double x2 = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        _task_time[graph.id()] += x2;

        //Call All Senders
        tstart = std::chrono::high_resolution_clock::now();
        _transmition_time[graph.id()] += detail::call_all_senders(it->second);

        //std::cerr<<"senders called\n";
        tfinish = std::chrono::high_resolution_clock::now();
        x1      = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        _output_time[graph.id()] += x1;

        // Call output accessors
        double x3 = 0;
        detail::call_all_output_node_accessors(it->second);
        tfinish = std::chrono::high_resolution_clock::now();

        x3 = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        _output_time[graph.id()] += x3;

        if(!_parallel)
        {
            if(graph.id() == _num_graphs - 1)
            {
                t_completion = std::chrono::high_resolution_clock::now();
                _latency_time += std::chrono::duration_cast<std::chrono::duration<double>>(t_completion - t_arriving).count();
                std::unique_lock<std::mutex> lck(_mtx);
                _ready = true;
                _condVar_serial.notify_all();
            }
        }
    }
}

/*If you want to buffer output of a stage if next stage is busy:
 * It also works with no buffer scenario
 */

void GraphManagerPipeline::warmup_and_execute_graph_serial(Graph &graph, int nn)
{
    std::stringstream                                                  stream;
    static std::chrono::time_point<std::chrono::high_resolution_clock> t_arriving;
    static std::chrono::time_point<std::chrono::high_resolution_clock> t_completion;

    auto it = _workloads.find(graph.id());
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");

    int Starting_frame = _warmup_n;
    for(int Frame = 0; Frame < nn + _warmup_n; Frame++)
    {
        //WarmUp Finished, Reset Timings
        if(Frame == Starting_frame)
        {
            reset_timing(graph.id());
        }

        //First Graph waits for Finishing Last Graph
        if(graph.id() == 0)
        {
            std::unique_lock<std::mutex> lck(_mtx);
            _condVar_serial.wait(lck, [this]
                                 { return _ready; });

            if(-1 == GPIOWrite(POUT, 1))
            {
                std::cerr << "could not write 1\n";
            }

            t_arriving = std::chrono::high_resolution_clock::now();
            _ready     = false;
        }

        //Inputs
        auto tstart = std::chrono::high_resolution_clock::now();
        detail::call_all_input_node_accessors(it->second);
        auto   tfinish = std::chrono::high_resolution_clock::now();
        double t_input = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        _input_time[graph.id()] += t_input;

        //Receivers
        tstart = std::chrono::high_resolution_clock::now();
        detail::call_all_receivers(it->second);
        tfinish          = std::chrono::high_resolution_clock::now();
        double t_receive = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        _receive_time[graph.id()] += t_receive;

        // Run graph Tasks
        tstart = std::chrono::high_resolution_clock::now();
        detail::call_all_tasks_pipeline(it->second, nn);
        tfinish      = std::chrono::high_resolution_clock::now();
        double t_run = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        _task_time[graph.id()] += t_run;

        //Senders
        tstart = std::chrono::high_resolution_clock::now();
        _transmition_time[graph.id()] += detail::call_all_senders(it->second);
        tfinish     = std::chrono::high_resolution_clock::now();
        auto t_send = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        _send_time[graph.id()] += t_send;

        //Outputs
        tstart = std::chrono::high_resolution_clock::now();
        detail::call_all_output_node_accessors(it->second);
        tfinish      = std::chrono::high_resolution_clock::now();
        double t_out = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
        _output_time[graph.id()] += t_out;

        //Last graph finished
        if(graph.id() == _num_graphs - 1)
        {
            t_completion = std::chrono::high_resolution_clock::now();
            auto l       = std::chrono::duration_cast<std::chrono::duration<double>>(t_completion - t_arriving).count();
            _latency_time += l;
            std::unique_lock<std::mutex> lck(_mtx);
            _ready = true;
            _condVar_serial.notify_all();
        }
    }
    //After finishing it sets GPIO to 0 for the experiment
    if(graph.id() == _num_graphs - 1)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(8));
        if(-1 == GPIOWrite(POUT, 0))
        {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

//Ehsan

void GraphManagerPipeline::print_times(int n)
{
    for(unsigned int id = 0; id < _num_graphs; id++)
    {
        print_times_details(id, n);

        print_times(id, n);

        auto &workload = _workloads.find(id)->second;
        detail::print_NPU_times(workload, n);
        std::cout << "\n\n********\n\n";
    }

    std::cerr << "\n\nAVG Latency: " << 1000 * _latency_time / n << "\n\n";
}

void GraphManagerPipeline::print_times(int graph_id, int n)
{
    std::stringstream stream;
    stream << "\n\nGraph" << graph_id << "   Input: " << (_input_time[graph_id] * 1000) / n << "   Task: " << (_task_time[graph_id] * 1000) / n << "   send: " << (_send_time[graph_id] * 1000) / n << "   Out: " << (_output_time[graph_id] * 1000) / n << "   Process: " << ((_input_time[graph_id] + _task_time[graph_id] + _send_time[graph_id] + _output_time[graph_id]) * 1000) / n << "\n\n";
    std::cout << stream.str();
}

void GraphManagerPipeline::print_times_details(int graph_id, int n)
{
    auto               it       = _workloads.find(graph_id);
    ExecutionWorkload *workload = &it->second;
    double             sum      = 0;
    int                c        = 0;
    int                l        = 0;
    double             tt       = 0;
    for(auto &task : workload->tasks)
    {
        if(!task.task)
        {
            continue;
        }
        std::cout << c++ << "\tLayer Name: " << task.node->name()
                  << " \t Layer time: " << task.time(n)
                  << " \t number of inputs: " << task.node->num_inputs()
                  << " \t input shape: " << task.node->input(0)->desc().shape
                  << " \t output shape: " << task.node->output(0)->desc().shape << std::endl;

        tt += task.time(n);
        if(task.ending)
        {
            std::cout << "Layer Number: " << l << " \t time: " << tt << std::endl;
            tt = 0;
            l++;
            std::cout << "----------------------------\n";
        }
        sum += task.time(n);
    }
    std::cout << "\n Sum of Layers time: " << sum << std::endl;
}

void GraphManagerPipeline::reset(Graph &graph)
{
    auto               it       = _workloads.find(graph.id());
    ExecutionWorkload *workload = &it->second;
    for(auto &task : workload->tasks)
    {
        task.reset();
    }
}

void GraphManagerPipeline::print_tasks()
{
    //std::vector<std::string> task_names;
    std::stringstream tasks_names;
    std::stringstream starting_tasks_names;
    std::stringstream ending_tasks_names;
    tasks_names << "std::string task_names[] = { ";
    starting_tasks_names << "std::set<std::string> starting_task_names = { ";
    ending_tasks_names << "std::set<std::string> ending_task_names = { ";
    int  num_tasks = 0, num_starting_tasks = 0, num_ending_tasks = 0;
    char sep = '\n';
    for(unsigned int id = 0; id < _workloads.size(); ++id)
    {
        auto &workload = _workloads.find(id)->second;
        for(auto &task : workload.tasks)
        {
            tasks_names << "\"" << task.node->name() << "\"," << sep;
            num_tasks++;
            if(task.starting)
            {
                starting_tasks_names << "\"" << task.node->name() << "\"," << sep;
                num_starting_tasks++;
            }
            if(task.ending)
            {
                ending_tasks_names << "\"" << task.node->name() << "\"," << sep;
                num_ending_tasks++;
            }
        }
        //std::cerr<<"governor:"<<workload.tasks[n-1].node->name()<<std::endl;
    }
    tasks_names.seekp(-2, std::ios_base::end);
    tasks_names << " };\n";
    std::cerr << "Total tasks " << num_tasks << std::endl;
    std::cerr << tasks_names.str();

    starting_tasks_names.seekp(-2, std::ios_base::end);
    starting_tasks_names << " };\n";
    std::cerr << "\nStarting tasks " << num_starting_tasks << std::endl;
    std::cerr << starting_tasks_names.str();

    ending_tasks_names.seekp(-2, std::ios_base::end);
    ending_tasks_names << " };\n";
    std::cerr << "\nEnding tasks " << num_ending_tasks << std::endl;
    std::cerr << ending_tasks_names.str();
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
bool isInList(const std::vector<std::string> &list, const std::string &name)
{
    return std::find(list.begin(), list.end(), name) != list.end();
}
void GraphManagerPipeline::set_GPIO_tasks(std::string power_profie_mode)
{
    std::vector<std::string> not_starting = { "conv2d_58", "conv2d_59", "conv2d_66", "conv2d_67", "NPU_YOLOv3_57_57", "NPU_YOLOv3_58_58", "NPU_YOLOv3_65_65", "NPU_YOLOv3_66_66" };
    std::vector<std::string> not_ending   = { "conv2d_58/LeakyRelu", "Yolo1", "conv2d_66/LeakyRelu", "Yolo2", "NPU_YOLOv3_57_57", "NPU_YOLOv3_58_58", "NPU_YOLOv3_65_65", "NPU_YOLOv3_66_66" };
    std::cerr << "set_GPIOs with mode: " << power_profie_mode << "\n";
    if(power_profie_mode == "layers")
    {
        for(unsigned int id = 0; id < _workloads.size(); ++id)
        {
            auto &workload = _workloads.find(id)->second;
            //int   n        = workload.tasks.size();
            for(auto &task : workload.tasks)
            {
                task.starting_gpio_switch = task.starting;
                task.ending_gpio_switch   = task.ending;
                task.profile_layers       = true;
                task.profile_transfers    = false;
            }
        }
    }
    if(power_profie_mode == "transfers")
    {
        for(unsigned int id = 0; id < _workloads.size(); ++id)
        {
            auto &workload = _workloads.find(id)->second;
            //int   n        = workload.tasks.size();
            for(auto &task : workload.tasks)
            {
                task.starting_gpio_switch = task.starting;
                task.ending_gpio_switch   = task.ending;
                task.profile_layers       = false;
                task.profile_transfers    = true;
            }
        }
    }
    if(power_profie_mode == "whole")
    {
        for(unsigned int id = 0; id < _workloads.size(); ++id)
        {
            auto &workload = _workloads.find(id)->second;
            //int   n        = workload.tasks.size();
            for(auto &task : workload.tasks)
            {
                task.starting_gpio_switch = false;
                task.ending_gpio_switch   = false;
                task.profile_layers       = false;
                task.profile_transfers    = false;
            }
        }
        {
            auto &workload                         = _workloads.find(0)->second;
            workload.tasks[0].starting_gpio_switch = true;
        }
        {
            auto &workload                           = _workloads.find(_workloads.size() - 1)->second;
            int   n                                  = workload.tasks.size();
            workload.tasks[n - 1].ending_gpio_switch = true;
        }
    }

    //bool simultaneously = false;

    auto &_workload = _workloads.find(0)->second;
    if(_workload.graph->name() == "YOLOv3" && power_profie_mode != "whole")
    {
        for(unsigned int id = 0; id < _workloads.size(); ++id)
        {
            auto &workload = _workloads.find(id)->second;
            //int   n        = workload.tasks.size();
            for(auto &task : workload.tasks)
            {
                if(isInList(not_starting, task.node->name()))
                {
                    task.starting_gpio_switch = false;
                }
                if(isInList(not_ending, task.node->name()))
                {
                    task.ending_gpio_switch = false;
                }
            }
        }
    }

    int i = 0;
    for(unsigned int id = 0; id < _workloads.size(); ++id)
    {
        auto &workload = _workloads.find(id)->second;
        //int   n        = workload.tasks.size();
        for(auto &task : workload.tasks)
        {
            std::cerr << i++ << " task name: " << task.node->name() << "\t";
            std::cerr << "starting gpio: " << task.starting_gpio_switch << "\t";
            std::cerr << "ending gpio: " << task.ending_gpio_switch << "\t";
            std::cerr << "profiling layers: " << task.profile_layers << "\t";
            std::cerr << "profiling transfers: " << task.profile_transfers << "\n";
        }
        //std::cerr<<"governor:"<<workload.tasks[n-1].node->name()<<std::endl;
    }
}

void GraphManagerPipeline::extract_governor_tasks(std::string mode)
{
    std::vector<std::string> not_governor = { "conv2d_58/LeakyRelu", "Yolo1", "conv2d_66/LeakyRelu", "Yolo2", "NPU_YOLOv3_57_57", "NPU_YOLOv3_58_58", "NPU_YOLOv3_65_65", "NPU_YOLOv3_66_66" };
    _governor_tasks.clear();
    if(mode == "layers")
    {
        for(unsigned int id = 0; id < _workloads.size(); ++id)
        {
            auto &workload = _workloads.find(id)->second;
            detail::set_governor_tasks(workload, &_governor_tasks);
        }
    }
    if(mode == "graphs")
    {
        for(unsigned int id = 0; id < _workloads.size(); ++id)
        {
            auto &workload = _workloads.find(id)->second;
            int   n        = workload.tasks.size();
            for(auto &task : workload.tasks)
            {
                task.governor = false;
            }
            workload.tasks[n - 1].governor = true;
            _governor_tasks.push_back(workload.tasks[n - 1].node->name());
            //std::cerr<<"governor:"<<workload.tasks[n-1].node->name()<<std::endl;
        }
    }
    if(mode == "PEs")
    {
        for(unsigned int id = 0; id < _workloads.size(); ++id)
        {
            auto &workload = _workloads.find(id)->second;
            int   n        = workload.tasks.size();
            for(auto &task : workload.tasks)
            {
                task.governor = false;
            }
            if(id == _workloads.size() - 1)
            {
                workload.tasks[n - 1].governor = true;
                _governor_tasks.push_back(workload.tasks[n - 1].node->name());
            }
        }
    }
    auto &_workload = _workloads.find(0)->second;
    if(_workload.graph->name() == "YOLOv3")
    {
        for(unsigned int id = 0; id < _workloads.size(); ++id)
        {
            auto &workload = _workloads.find(id)->second;
            //int   n        = workload.tasks.size();
            for(auto &task : workload.tasks)
            {
                if(isInList(not_governor, task.node->name()))
                {
                    task.governor = false;
                    std::cerr << "change governor to false for task:" << task.node->name() << std::endl;
                }
            }
        }
    }
    //int k = 0;
}

std::string removeConsecutiveDuplicates(const std::string &input)
{
    std::string result;
    for(size_t i = 0; i < input.length(); ++i)
    {
        // Only add the current character to result if it's different from the last character in result
        if(result.empty() || input[i] != result.back())
        {
            result += input[i];
        }
    }
    return result;
}

void GraphManagerPipeline::set_freqs(std::string freqs, std::string _order, char GPU_Host, char NPU_Host)
{
    _governor_freqs.clear();
    if(freqs == "")
    {
        return;
    }
    if(freqs.size() >= 2 && freqs.front() == '{' && freqs.back() == '}')
    {
        freqs = freqs.substr(1, freqs.size() - 2);
        if(freqs.size() >= 2 && freqs.front() == '{' && freqs.back() == '}')
        {
            freqs = freqs.substr(1, freqs.size() - 2);
            extract_governor_tasks("PEs");
            if(freqs == "min" or freqs == "[min]")
            {
                for(auto task : _governor_tasks)
                {
                    _governor_freqs[task] = { 0, 0, 0 };
                }
            }
            else if(freqs == "max" or freqs == "[max]")
            {
                for(auto task : _governor_tasks)
                {
                    _governor_freqs[task] = { 5, 7, 4 };
                }
            }
            else
            {
                int l, b, g;
                std::replace(freqs.begin(), freqs.end(), '-', ' '); // Replace '-' with whitespace
                std::istringstream iss(freqs);
                if(!(iss >> l >> b >> g))
                {
                    return;
                }
                _governor_freqs[_governor_tasks[0]] = { l, b, g };
            }
        }
        else
        {
            extract_governor_tasks("graphs");
            set_freq_map(freqs, removeConsecutiveDuplicates(_order), GPU_Host, NPU_Host);
        }
    }
    else
    {
        extract_governor_tasks("layers");
        set_freq_map(freqs, _order, GPU_Host, NPU_Host);
    }
    set_governor_freqs();
}
void GraphManagerPipeline::set_freq_map(std::string freqs, std::string _order, char GPU_Host, char NPU_Host)
{
    if(freqs == "min" or freqs == "[min]")
    {
        extract_governor_tasks("PEs");
        for(auto task : _governor_tasks)
        {
            _governor_freqs[task] = { 0, 0, 0 };
        }
        return;
    }
    if(freqs == "max" or freqs == "[max]")
    {
        extract_governor_tasks("PEs");
        for(auto task : _governor_tasks)
        {
            _governor_freqs[task] = { 5, 7, 4 };
        }
        return;
    }
    //TODO: Set just in use processor to its max fequency
    if(freqs == "max" or freqs == "[max]")
    {
        unsigned int i = 0;
        //bool Prev_NPU = false;
        for(auto task : _governor_tasks)
        {
            if(_order[i] == 'L')
            {
                _governor_freqs[task] = { arm_compute::graph::ExecutionTask::get_max_l(), 0, 0 };
            }
            if(_order[i] == 'B')
            {
                _governor_freqs[task] = { 0, arm_compute::graph::ExecutionTask::get_max_b(), 0 };
            }
            if(_order[i] == 'G')
            {
                if(GPU_Host == 'B')
                {
                    _governor_freqs[task] = { 0,
                                              arm_compute::graph::ExecutionTask::get_max_b(),
                                              arm_compute::graph::ExecutionTask::get_max_g() };
                }
                if(GPU_Host == 'L')
                {
                    _governor_freqs[task] = { arm_compute::graph::ExecutionTask::get_max_l(),
                                              0,
                                              arm_compute::graph::ExecutionTask::get_max_g() };
                }
            }
            if(_order[i] == 'N')
            {
                if(NPU_Host == 'B')
                {
                    _governor_freqs[task] = { 0,
                                              arm_compute::graph::ExecutionTask::get_max_b(),
                                              0 };
                }
                if(NPU_Host == 'L')
                {
                    _governor_freqs[task] = { arm_compute::graph::ExecutionTask::get_max_l(),
                                              0,
                                              0 };
                }
                while(i < (_order.size() - 1))
                {
                    if(_order[i] == _order[i + 1])
                    {
                        i = i + 1;
                    }
                    else
                    {
                        break;
                    }
                }
            }
            i++;
        }
        return;
    }

    std::stringstream ss(freqs);
    std::string       token;
    //For iterating over orders mapping letters
    size_t i = _order.size() - 1;
    //for iterating over governor_tasks (for consequitive Ns in order there is one gevernor task just)
    int  governor_task_index = _governor_tasks.size() - 1;
    int  _method             = 1;
    char prev_p              = {};
    while(std::getline(ss, token, '-'))
    {
        int  j = (i + 1) % _order.size();
        int  l = 0, b = 0, g = 0;
        char p = _order[j];

        if(token.find('[') != std::string::npos)
        {
            if(p != 'G')
            {
            }
            token.erase(0, 1);
            token.erase(token.size() - 1, 1);
            std::stringstream t(token);
            char              comma;

            (GPU_Host == 'B') ? t >> g >> comma >> b : t >> g >> comma >> l;
        }
        else
        {
            if(p == 'L')
            {
                l = std::stoi(token);
                //b=7;
            }
            if(p == 'B')
            {
                b = std::stoi(token);
                //l=5;
            }
            if(p == 'N')
            {
                (NPU_Host == 'B') ? b = std::stoi(token) : l = stoi(token);
                if(_method == 2)
                {
                    while(i < (_order.size() - 1))
                    {
                        if(_order[i] == _order[i + 1])
                        {
                            //std::getline(ss, token, '-');
                            i = i + 1;
                        }
                        else
                        {
                            break;
                        }
                    }
                }
                //l=5;
            }
        }
        if(_method == 1)
        {
            if(p == 'N' && p == prev_p)
            {
            }
            else
            {
                _governor_freqs[_governor_tasks[governor_task_index]] = { l, b, g };
                governor_task_index++;
                governor_task_index = governor_task_index % _governor_tasks.size();
            }
            prev_p = p;
        }
        if(_method == 2)
        {
            _governor_freqs[_governor_tasks[i]] = { l, b, g };
        }
        i = i + 1;
        i = i % (_order.size());
    }
}

void GraphManagerPipeline::set_governor_freqs()
{
    for(auto p : _governor_freqs)
    {
    }
    for(unsigned int id = 0; id < _workloads.size(); ++id)
    {
        auto &workload = _workloads.find(id)->second;
        for(auto &task : workload.tasks)
        {
            if(!task.task)
                continue;
            if(task.ending)
                std::cerr << id << "---- " << task.node->name() << "    processor:" << task.processor << std::endl;
            if(task.governor)
            {
                std::cerr << "setting freq of governor task " << task.node->name() << std::endl;
                task.set_freq(_governor_freqs.at(task.node->name()));
            }
        }
    }
}

int GraphManagerPipeline::destroy()
{
    int ret = 0;
    for(unsigned int id = 0; id < _workloads.size(); ++id)
    {
        auto &workload = _workloads.find(id)->second;
        ret            = ret | detail::NPU_destroy(workload);
    }
    return ret;
}

} // namespace graph
} // namespace arm_compute
