/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "arm_compute/graph/nodes/ReceiverNode.h"
#include "arm_compute/graph/nodes/SenderNode.h"
#include "arm_compute/runtime/NPU/NPU.h"
#include <sstream>

#define streamline 0
#include <chrono>
#if streamline > 0
#include "annotate/Sr_ann.c"
#endif
//#include "src/graph/GraphManager.cpp"

#include "arm_compute/graph/detail/ExecutionHelpersPipeline.h"
#include "arm_compute/graph/frontend/IStreamPipeline.h"
#include "utils/main_layer_checker.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/GraphManagerPipeline.h"
#include "arm_compute/graph/TensorPipeline.h"
#include "arm_compute/graph/Utils.h"
#include "arm_compute/graph/backends/BackendRegistry.h"

namespace arm_compute
{
namespace graph
{
namespace detail
{

//std::mutex 								_mtx_backend = {};
ExecutionWorkload configure_all_nodes_pipeline(Graph &g, GraphContext &ctx, const std::vector<NodeID> &node_order)
{
    ExecutionWorkload workload;
    workload.graph = &g;
    workload.ctx   = &ctx;

    // Reserve memory for tasks
    workload.tasks.reserve(node_order.size());

    // Create tasks
    int task_number = 0;
    ////std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    {
        //std::lock_guard<std::mutex> lock(_mtx_backend);
        for(auto &node_id : node_order)
        {
            auto node = g.node(node_id);
            //Ehsan

            /*
			std::cerr<<"\n*******************************\nnode name: "<<node->name()<<" ID: "<<node->id()<<" num inputs: "<<node->num_inputs()<<std::endl<<std::flush;
			for(int k=0; k < node->num_inputs(); k++){
				INode *cc=node->input_edge(k)->producer();
				std::cerr<<"\ninput "<<k<<" node_name: "<<cc->name()<<" ID: "<<cc->id()<<std::endl<<std::flush;
				TensorShape shape=node->input(k)->desc().shape;
				std::cout<<shape<<std::endl;
				//for(int i=0;i<shape.num_dimensions();i++) std::cout<<shape[i]<<'\t'<<std::flush;
				//std::cerr<<"Padding: "<<_padding.left<<_padding.right<<_padding.top<<_padding.bottom<<std::endl;
			}
			*/

            /*
			 ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
								   << node.name()
								   << " Type: " << node.type()
								   << " Target: " << CLTargetInfo::TargetType
								   << " Data Type: " << input0->info()->data_type()
								   << " Input0 shape: " << input0->info()->tensor_shape()
								   << " Input1 shape: " << input1->info()->tensor_shape()
								   << " Input2 shape: " << input2->info()->tensor_shape()
								   << " Output0 shape: " << output0->info()->tensor_shape()
								   << " Output1 shape: " << output1->info()->tensor_shape()
								   << " Output2 shape: " << output2->info()->tensor_shape()
								   << " Output3 shape: " << output3->info()->tensor_shape()
								   << " DetectionPostProcessLayer info: " << detect_info
								   << std::endl);
			 */

            //std::cerr<<"\n\n\n\nnode name: "<<node->name()<<std::endl;
            if(node != nullptr)
            {
                //std::cerr<<"node is not null\n";
                Target assigned_target = node->assigned_target();

                backends::IDeviceBackend  &backend = backends::BackendRegistry::get().get_backend(assigned_target);
                std::unique_ptr<IFunction> func    = backend.configure_node(*node, ctx);
                //std::cerr<<"func is null? "<<(func == nullptr)<<std::endl;
                if(func != nullptr || is_utility_node(node))
                {
                    std::cerr << "Graph (" << g.id() << ") " << "Task " << task_number++ << ": " << node->name() << "\n";
                    workload.tasks.emplace_back(ExecutionTask(std::move(func), node));
                    std::cerr << "task has been emplaced\n";
                }
            }
        }
        //End mutex of backend
    }
    ////std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    // Add inputs and outputs
    for(auto &node : g.nodes())
    {
        if(node != nullptr && node->type() == NodeType::Input)
        {
            //Ehsan
            //std::cout<<"\ninput node name and ID: "<<node->name()<<'_'<<node->id()<<std::endl;

            workload.inputs.push_back(node->output(0));
        }
        if(node != nullptr && node->type() == NodeType::Receiver)
        {
            //Ehsan
            //std::cout<<"\ninput node name and ID: "<<node->name()<<'_'<<node->id()<<std::endl;
            ReceiverNode *rec = dynamic_cast<ReceiverNode *>(node.get());
            rec->get_receiver_tensor()->set_name(rec->common_node_params().name);
            rec->get_receiver_tensor()->set_tensor(rec->output(0));
            //rec->forward_descriptors();
            workload.receivers.push_back(rec->get_receiver_tensor());
        }
        if(node != nullptr && node->type() == NodeType::Sender)
        {
            SenderNode *sender = dynamic_cast<SenderNode *>(node.get());
            sender->get_sender_tensor()->set_name(sender->common_node_params().name);
            //As pm (pass manager in graph manager will change tensor ids it is required to again set them here)
            sender->get_sender_tensor()->set_tensor(sender->input(0));
            workload.senders.push_back(sender->get_sender_tensor());
            //Ehsan
            //std::cout<<"\noutput node name and ID: "<<node->name()<<'_'<<node->id()<<std::endl;

            //continue;
        }

        if(node != nullptr && node->type() == NodeType::Output)
        {
            workload.outputs.push_back(node->input(0));
            //Ehsan
            //std::cout<<"\noutput node name and ID: "<<node->name()<<'_'<<node->id()<<std::endl;

            continue;
        }
    }
    std::stringstream stream;
    stream << "Graph " << g.id() << " input size: " << workload.inputs.size() << " receiver size: " << workload.receivers.size() << " tasks size: " << workload.tasks.size() << " senders size: " << workload.senders.size() << " output size: " << workload.outputs.size() << "\n\n";
    std::cerr << stream.str();
    stream.str(std::string());
    return workload;
}

double call_all_senders(ExecutionWorkload &workload)
{
    double t = 0;
    std::for_each(std::begin(workload.senders), std::end(workload.senders), [&](TensorPipelineSender *sender_tensor)
                  {
        double t_i=sender_tensor->send_data();
        //t=(t_i>t)?t_i:t;//if parallelize senders and receivers
        t+=t_i; });

    return t;
}

bool call_all_receivers(ExecutionWorkload &workload)
{
    bool is_valid = true;
    //Ehsan: size of inputs is 1
    //std::string c;

    //std::cerr<<"number of receivers: "<<workload.receivers.size()<<std::endl;
    //std::string t;
    //std::cin>>t;

    //First just mark receivers as ready then try to receive data one by one (because receive data waits at first receiver till sender just send data
    std::for_each(std::begin(workload.receivers), std::end(workload.receivers), [&](TensorPipelineReceiver *receiver_tensor)
                  {
#if My_print > 0
		std::cerr<<"set Receiver ready"<<std::endl;
		std::cerr<<receiver_tensor->desc().shape <<std::endl;
#endif
		receiver_tensor->set_receiver_ready(); });

    std::for_each(std::begin(workload.receivers), std::end(workload.receivers), [&](TensorPipelineReceiver *receiver_tensor)
                  {
#if My_print > 0
    	std::cerr<<"Receiver"<<std::endl;
    	std::cerr<<receiver_tensor->desc().shape <<std::endl;
#endif
        bool valid_input = (receiver_tensor != nullptr) && receiver_tensor->receive_data();
        is_valid         = is_valid && valid_input; });
    return is_valid;
}

void reset_transmit_timings(ExecutionWorkload &workload)
{
    std::for_each(std::begin(workload.receivers), std::end(workload.receivers), [&](TensorPipelineReceiver *receiver_tensor)
                  { receiver_tensor->reset_timing(); });
    std::for_each(std::begin(workload.senders), std::end(workload.senders), [&](TensorPipelineSender *sender_tensor)
                  { sender_tensor->reset_timing(); });
    return;
}

void reset_NPU_timings(ExecutionWorkload &workload)
{
    std::for_each(std::begin(workload.tasks), std::end(workload.tasks), [&](ExecutionTask &task)
                  {
        if(task.node!= nullptr && task.node->type() == NodeType::NPU){
        	dynamic_cast<arm_compute::NPUBase*>(task.task.get())->reset_timing();

        } });
    return;
}

void NPU_set_preallocated_outputs(ExecutionWorkload &workload)
{
    std::for_each(std::begin(workload.tasks), std::end(workload.tasks), [&](ExecutionTask &task)
                  {
		if(task.node!= nullptr && task.node->type() == NodeType::NPU){
			dynamic_cast<arm_compute::NPUBase*>(task.task.get())->set_preallocated_outputs();
		} });
}

int NPU_destroy(ExecutionWorkload &workload)
{
    int ret = 0;
    std::for_each(std::begin(workload.tasks), std::end(workload.tasks), [&](ExecutionTask &task)
                  {
		if(task.node!= nullptr && task.node->type() == NodeType::NPU){
			ret=ret | dynamic_cast<arm_compute::NPUBase*>(task.task.get())->destroy();
		} });
    return ret;
}

/*void set_ending_tasks(ExecutionWorkload &workload, std::vector<std::string> *ending_tasks){
	//std::vector<std::string> ending_tasks;
	int n=workload.tasks.size();
	workload.tasks[0].starting=true;
	for(int i=0;i<n;i++){
		if(i==n-1){
			std::cerr<<"setting ending for layer: "<<workload.tasks[i].node->name()<<std::endl;
			workload.tasks[i].ending=true;
			if(ending_tasks!=nullptr)
				ending_tasks->push_back(workload.tasks[i].node->name());
		}
		else{
			if(arm_compute::graph::frontend::IStreamPipeline::is_next_layer(workload.tasks[i+1].node->name())){
				std::cerr<<"setting ending for layer: "<<workload.tasks[i].node->name()<<std::endl;
				workload.tasks[i+1].starting=true;
				workload.tasks[i].ending=true;
				if(ending_tasks!=nullptr)
					ending_tasks->push_back(workload.tasks[i].node->name());
			}
		}
	}
	//return ending_tasks;
	return;
}*/

/*
void set_governor_tasks(ExecutionWorkload &workload, std::vector<std::string> *governor_tasks){
	//std::vector<std::string> ending_tasks;
	int n=workload.tasks.size();
	for(int i=0;i<n;i++){
		if(i==n-1){
			workload.tasks[i].governor=true;
			if(governor_tasks!=nullptr)
				governor_tasks->push_back(workload.tasks[i].node->name());
		}
		else{
			if(arm_compute::graph::frontend::IStreamPipeline::is_next_layer(workload.tasks[i+1].node->name())){
				workload.tasks[i].governor=true;
				if(governor_tasks!=nullptr)
					governor_tasks->push_back(workload.tasks[i].node->name());
			}
		}
	}
	//return ending_tasks;
	return;
}*/

void set_ending_tasks(ExecutionWorkload &workload, std::vector<std::string> *ending_tasks)
{
    //std::vector<std::string> ending_tasks;
    int n                      = workload.tasks.size();
    workload.tasks[0].starting = true;
    for(int i = 0; i < n; i++)
    {
        //if(arm_compute::graph::frontend::IStreamPipeline::ending_tasks.count(workload.tasks[i].node->name())>0){
        if(check_ending(workload.graph->name(), workload.tasks[i].node->name()))
        {
            //This is for skipping YOLO consecutive tasks (for a yolo layer, there is 6 consecutive tasks with same names)
            //Notice that this is not a problem when creating graphs in IStreampipeline is_next function as there is just one yolo layer
            if(i < n - 1)
            {
                if(workload.tasks[i].node->name() == workload.tasks[i + 1].node->name())
                {
                    continue;
                }
            } //

            workload.tasks[i].ending = true;
            std::cerr << "setting ending for layer: " << workload.tasks[i].node->name() << std::endl;
            if(ending_tasks != nullptr)
                ending_tasks->push_back(workload.tasks[i].node->name());
            if(i < n - 1)
            {
                workload.tasks[i + 1].starting = true;
            }
        }
    }
    return;
}

//void set_governor_tasks(ExecutionWorkload &workload, std::vector<std::string> *governor_tasks){return;}

void set_governor_tasks(ExecutionWorkload &workload, std::vector<std::string> *governor_tasks)
{
    //std::vector<std::string> ending_tasks;
    int n = workload.tasks.size();
    for(int i = 0; i < n; i++)
    {
        if(workload.tasks[i].ending)
        {
            workload.tasks[i].governor = true;
            if(governor_tasks != nullptr)
                governor_tasks->push_back(workload.tasks[i].node->name());
        }
    }
    /*static int k=0;
	for(auto task:*governor_tasks){
		std::cerr<<"\n"<<k++<<" _gov task: "<<task;
	}*/
    return;
}

void print_NPU_times(ExecutionWorkload &workload, int num_run)
{
    double      input_time = 0, run_time = 0, prof_run_time = 0, output_time = 0;
    std::string name     = "";
    bool        NPU_Node = false;
    std::for_each(std::begin(workload.tasks), std::end(workload.tasks), [&](ExecutionTask &task)
                  {
		if(task.node!= nullptr && task.node->type() == NodeType::NPU){
			NPU_Node=true;
			input_time=dynamic_cast<arm_compute::NPUBase*>(task.task.get())->get_input_time();
			run_time=dynamic_cast<arm_compute::NPUBase*>(task.task.get())->get_run_time();
			prof_run_time=dynamic_cast<arm_compute::NPUBase*>(task.task.get())->get_prof_run_time();
			output_time=dynamic_cast<arm_compute::NPUBase*>(task.task.get())->get_output_time();
			name=task.node->name();
		} });
    if(NPU_Node)
    {
        std::cout << "NPU " << name
                  << "\tAVG_input_time:" << input_time / double(num_run)
                  << "\tAVG_run_time:" << run_time / double(num_run)
                  << "\tAVG_prof_run_time:" << prof_run_time / double(num_run)
                  << "\tAVG_output_time:" << output_time / double(num_run)
                  << "\n";
    }
}

void allocate_const_tensors_pipeline(Graph &g)
{
    for(auto &node : g.nodes())
    {
        if(node != nullptr)
        {
            switch(node->type())
            {
                case NodeType::Const:
                case NodeType::Receiver:
                case NodeType::Input:
                    allocate_all_output_tensors(*node);
                    break;
                case NodeType::Sender:
                case NodeType::Output:
                    allocate_all_input_tensors(*node);
                    break;
                default:
                    break;
            }
        }
    }
}

void call_all_tasks_pipeline(ExecutionWorkload &workload, int nn)
{
    ARM_COMPUTE_ERROR_ON(workload.ctx == nullptr);

    // Acquire memory for the transition buffers
    for(auto &mm_ctx : workload.ctx->memory_managers())
    {
        if(mm_ctx.second.cross_group != nullptr)
        {
            mm_ctx.second.cross_group->acquire();
        }
    }
    // Execute tasks
#if streamline > 0
    ANNOTATE_SETUP;
    ANNOTATE_MARKER_STR("start_running tasks");
    static int cc = 0;
    static int c  = 0;
#endif
    size_t               ii = 0;
    std::stringstream stream;
    //stream<<"--size of tasks: "<<workload.tasks.size()<<std::endl;
    //std::cerr<<stream.str();
    //stream.str(std::string());
    for(auto &task : workload.tasks)
    {
        ii++;
        //std::cerr<<"Graph ("<<workload.graph->id()<<") "<<ii<<"  "<<task.node->name()<<std::endl;
        if(nn == 0 && ii < workload.tasks.size())
        {
            task();
        }
        else
        {
#if streamline > 0
            ANNOTATE_CHANNEL_COLOR(cc, ((c % 2) == 0) ? ANNOTATE_GREEN : ANNOTATE_YELLOW, (std::to_string(c) + " " + task.node->name()).c_str());
#endif
            task(nn);
#if streamline > 0
            if(task.ending)
                c = c + 1;
            ANNOTATE_CHANNEL_END(cc++);
#endif
        }
        auto t0      = std::chrono::high_resolution_clock::now();
        auto nanosec = t0.time_since_epoch();
#if My_print > 0
        std::cout << "Executionhelpers, tasks() time: " << nanosec.count() << std::endl;
#endif
    }

    // Release memory for the transition buffers
    for(auto &mm_ctx : workload.ctx->memory_managers())
    {
        if(mm_ctx.second.cross_group != nullptr)
        {
            mm_ctx.second.cross_group->release();
        }
    }
}

} // namespace detail
} // namespace graph
} // namespace arm_compute
