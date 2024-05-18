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
#include"annotate/streamline_annotate.h"
#include<chrono>
#include "utils/Utils.h"
//For printing shape of a tensor
#include "utils/TypePrinter.h"



#include "arm_compute/graph/TensorPipeline.h"
#include "arm_compute/graph/backends/BackendRegistry.h"

#include "arm_compute/runtime/CL/CLTensor.h"

namespace arm_compute
{
namespace graph
{



//Create and setup an arm_compute::graph::Tensor*
std::unique_ptr<arm_compute::graph::Tensor> create_and_setup_tensor(arm_compute::graph::Tensor* main_tensor){
	//1-Create an empty tensor( even empty tensor descriptor)
	//When GraphBuilder::add_Receiver_node (g.add_node<ReceiverNode>(desc);) --> It calls create_tensor(); in Graph::create_tensor
	//auto tensor = std::make_unique<Tensor>(-1, TensorDescriptor());
	//tensor->desc()=main_tensor->desc();
	//tensor->desc().target = main_tensor->handle()->target;
	//std::cerr<<"Main tensor desc:"<<main_tensor->desc().data_type<<" "<<main_tensor->desc().target<<std::endl;
	auto tensor = std::make_unique<Tensor>(-1, main_tensor->desc());
	//2-In finalizaing the graph in graphmanager in force_target_to_graph(graph, forced_target);
	//It is not needed because we make its instance with main_tansor->desc()
	tensor->desc().target=main_tensor->desc().target;

	//3-In graphmanger it calls detail::configure_all_tensors(graph); which is implemented in ExecutionHelpers.cpp
	if(tensor && tensor->handle() == nullptr)
	{
		Target                         target  = tensor->desc().target;
		backends::IDeviceBackend      &backend = backends::BackendRegistry::get().get_backend(target);
		std::unique_ptr<ITensorHandle> handle  = backend.create_tensor(*tensor);
		ARM_COMPUTE_ERROR_ON_MSG(!handle, "Couldn't create backend handle!");
		tensor->set_handle(std::move(handle));
	}
	//4-In graphManager it calls detail::allocate_const_tensors_pipeline(graph); which for receiver nodes it calls allocate_all_output_tensors(*node); in ExecutionHelpers.cpp
	//if(tensor != nullptr && !tensor->bound_edges().empty())
	if(tensor != nullptr)
	{
		ARM_COMPUTE_ERROR_ON_MSG(!tensor->handle(), "Tensor handle is not configured!");
#if My_print > 0
		//Ehsan

		if(node.type()!=NodeType::Const)
			std::cerr<<i<<"->ExecutionHelpers, Allocating output tensor for input and const node, CLTensor shape:"<<tensor->handle()->tensor().info()->tensor_shape()
				<<" tensor shape:"<<tensor->desc().shape
				<<std::endl;
#endif
		//std::cerr<<"CLTensor shape:"<<tensor->handle()->tensor().info()->tensor_shape()<<" tensor shape:"<<tensor->desc().shape<<std::endl;
		tensor->handle()->allocate();
	}
	return std::move(tensor);
}

//in this function -> graphmanager.cpp: auto workload = detail::configure_all_nodes(graph, ctx, topological_sorted_nodes);
// this line existed -> std::unique_ptr<IFunction> func            = backend.configure_node(*node, ctx);
//Which create a function and in next lines at it to workloads as a task. when creating a function it sets the input and output tensors ( address of tensor.handle.tensor) into the function
// see this for example std::unique_ptr<IFunction> CLFunctionFactory::create(INode *node, GraphContext &ctx) in src/grapoh/backend/cl/clfunctionfactory which for example if we track this :
//return detail::create_activation_layer<CLActivationLayer, CLTargetInfo>(*polymorphic_downcast<ActivationLayerNode *>(node));
//we see that _impl->src = input; this line set the address of input tensor
//Therefore if we just change the address of inner tensor in outer tensor like this:
//auto t2=create_and_setup_tensor(tensor);
//tensor->handle()->set_tensor(t2->handle()->tensor_ptr());
//it does not work because the function (task) when start running get input from the first origin input tensor!



arm_compute::Tensor* create_inner_tensor_cpu(arm_compute::graph::Tensor* _tensor){
	auto tensor=new arm_compute::Tensor;
	tensor->allocator()->init(*(_tensor->handle()->tensor().info()));
	tensor->allocator()->allocate();
	return tensor;
}
arm_compute::CLTensor* create_inner_tensor_gpu(arm_compute::graph::Tensor* _tensor){
	auto tensor=new arm_compute::CLTensor;
	tensor->allocator()->init(*(_tensor->handle()->tensor().info()));
	tensor->allocator()->allocate();
	return tensor;
}

TensorPipelineReceiver::TensorPipelineReceiver()
{
	//receiver_ready.store(false);
	//data_sent.store(false);
	*receiver_ready=false;
	*data_sent=false;
	t_sender_write=0;
	t_sender_transfer=0;
	t_receiver_read=0;
	t_receiver_wait=0;
	num_run=0;
}
bool TensorPipelineReceiver::get_receiver_ready(){
	//return receiver_ready.load();
	return *receiver_ready;
}
bool TensorPipelineReceiver::get_data_sent(){
	//return data_sent.load();
	return *data_sent;
}
void TensorPipelineReceiver::set_tensor(Tensor* t){
	tensor=t;
}
Tensor* TensorPipelineReceiver::get_tensor(){
	return tensor;
}

void TensorPipelineReceiver::set_name(std::string _name){
	name=std::string(_name);
}

void TensorPipelineReceiver::reset_timing(){
	t_sender_write=0;
	t_sender_transfer=0;
	t_receiver_read=0;
	t_receiver_wait=0;
	num_run=0;
}
double TensorPipelineReceiver::get_transmition_time(){
	return t_sender_transfer;
}
double TensorPipelineReceiver::get_receiver_wait_time(){
	return t_receiver_wait;
}
double TensorPipelineReceiver::get_receiver_read_time(){
	return t_receiver_read;
}
double TensorPipelineReceiver::get_sender_write_time(){
	return t_sender_write;
}
int TensorPipelineReceiver::get_graph_id(){
	return graph_id;
}
void TensorPipelineReceiver::set_graph_id(int g_id){
	graph_id=g_id;
}


/*This version is without buffering (which we observed does not work for cnns with branch)
double TensorPipelineReceiver::send_data(Tensor* _tensor){
		{
			std::string s;
			auto tstart=std::chrono::high_resolution_clock::now();
			std::unique_lock<std::mutex> lck(mutex_);

			if(!get_receiver_ready()){
				s="graph:" + std::to_string(graph_id) +name+" waiting for its receiver to get ready\n";
				std::cerr<<s;
			}
			condVar.wait(lck, [this]{ return get_receiver_ready(); });
			auto tend1=std::chrono::high_resolution_clock::now();
			s="graph:" + std::to_string(graph_id) + name+" transferring data\n";
			std::cerr<<s;





			auto t2=create_and_setup_tensor(tensor);
			t2->handle()->map(true);
			t2->handle()->tensor().copy_from(_tensor->handle()->tensor());
			//t2->handle()->unmap();

			//It does not work (explained above)
			//tensor->handle()->set_tensor(t2->handle()->tensor_ptr());

			tensor->handle()->map(true);

			receiver_ready=false;
			//Transfer data
			const auto   output_net  = reinterpret_cast<double *>(_tensor->handle()->tensor().buffer() + _tensor->handle()->tensor().info()->offset_first_element_in_bytes());
			std::cerr<<"graph:" + std::to_string(graph_id) +name +" dfdf\n";
			std::cerr<<"graph:" + std::to_string(graph_id) +name+" _tensor desc: "<<_tensor->desc().shape<<std::endl;
			std::cerr<<"graph:" + std::to_string(graph_id) +name+" tensor desc: "<<tensor->desc().shape<<std::endl;
			std::cerr<<"\n\n\n\n\n\n\n\nsender tensor:  "<<output_net[0]<<","<<output_net[1]<<","<<output_net[2]<<std::endl;
			tensor->handle()->tensor().copy_from(t2->handle()->tensor());
			const auto   output_net2  = reinterpret_cast<double *>(tensor->handle()->tensor().buffer() + tensor->handle()->tensor().info()->offset_first_element_in_bytes());
			std::cerr<<"receiver tensor: "<<output_net2[0]<<","<<output_net2[1]<<","<<output_net2[2]<<"\n\n\n\n\n\n"<<std::endl;
			tensor->handle()->unmap();
			t2->handle()->unmap();
			//t2->handle()->unmap();
			data_sent=true;
			condVar.notify_all();
			s="graph:" + std::to_string(graph_id) +name+" done\n";
			std::cerr<<s;


			lck.unlock();
			auto tend2=std::chrono::high_resolution_clock::now();
			t_sender_wait+=std::chrono::duration_cast<std::chrono::duration<double>>(tend1 - tstart).count();
			double t=std::chrono::duration_cast<std::chrono::duration<double>>(tend2 - tend1).count();
			t_transmition+=t;
			return t;
		}
}*/

double TensorPipelineReceiver::send_data(Tensor* _tensor){
		{

			std::string s;
			double duration_write=0;
			double duration_transfer=0;
			num_run++;
			Frame++;
			auto tstart=std::chrono::high_resolution_clock::now();
			std::unique_lock<std::mutex> lck(mutex_);

			/******************** If is receiver of a NPU ************************/
			if(is_npu){
				//std::cerr<<"sending data to an NPU receiver\n";
				if(!get_receiver_ready() || !NPU_buffer.empty()){
					//add to queue (maybe tensor.map required)
					const auto   output_net  = reinterpret_cast<double *>(_tensor->handle()->tensor().buffer() + _tensor->handle()->tensor().info()->offset_first_element_in_bytes());
					NPU_buffer.emplace(output_net);
					auto tend=std::chrono::high_resolution_clock::now();
					duration_write=1000*(std::chrono::duration_cast<std::chrono::duration<double>>(tend - tstart).count());
					t_sender_write+=duration_write;
					//std::cerr<<name<<" Frame "<<Frame-1<<" sender write time: "<<duration_write<<std::endl;
					//std::cerr<<"Sending to NPU_Receiver of graph:" + std::to_string(graph_id) +"_"+name + " is not ready, it has been put in buffer\n";
				}
				else{
					/*
					s="Sending to NPU_Receiver of graph:" + std::to_string(graph_id) + name+" transferring data directly\n";
					std::cerr<<s;
					*receiver_ready=false;
					//Transfer data
					const auto   output_net  = reinterpret_cast<double *>(_tensor->handle()->tensor().buffer() + _tensor->handle()->tensor().info()->offset_first_element_in_bytes());
					//Here Set the data into input of the NPU


					*data_sent=true;
					condVar.notify_all();*/

					tensor->handle()->map(true);

					*receiver_ready=false;
					//Transfer data
					const auto   output_net  = reinterpret_cast<double *>(_tensor->handle()->tensor().buffer() + _tensor->handle()->tensor().info()->offset_first_element_in_bytes());
					//std::cerr<<"graph:" + std::to_string(graph_id) +"_"+name+" _tensor desc: "<<_tensor->desc().shape<<std::endl;
					//std::cerr<<"graph:" + std::to_string(graph_id) +"_"+name+" tensor desc: "<<tensor->desc().shape<<std::endl;
					//std::cerr<<"\n\n\n\n\n\n\n\nsender tensor:  "<<output_net[0]<<","<<output_net[1]<<","<<output_net[2]<<std::endl;
					tensor->handle()->tensor().copy_from(_tensor->handle()->tensor());
					const auto   output_net2  = reinterpret_cast<double *>(tensor->handle()->tensor().buffer() + tensor->handle()->tensor().info()->offset_first_element_in_bytes());
					//std::cerr<<"receiver tensor: "<<output_net2[0]<<","<<output_net2[1]<<","<<output_net2[2]<<"\n\n\n\n\n\n"<<std::endl;
					tensor->handle()->unmap();

					*data_sent=true;
					condVar.notify_all();
					auto tend=std::chrono::high_resolution_clock::now();
					duration_transfer=1000*(std::chrono::duration_cast<std::chrono::duration<double>>(tend - tstart).count());
					t_sender_transfer+=duration_transfer;
					//std::cerr<<name<<" Frame "<<Frame-1<<" sender transfer time: "<<duration_transfer<<std::endl;
					//s="graph:" + std::to_string(graph_id) +name+" done\n";
					//std::cerr<<s;



				}
			}
			/********************* If is not receiver of a NPU *******************/
			else{
				if(!get_receiver_ready() || !buffer.empty()){
					//add to queue
					auto t2=create_and_setup_tensor(tensor);
					t2->handle()->map(true);
					t2->handle()->tensor().copy_from(_tensor->handle()->tensor());
					t2->handle()->unmap();
					buffer.emplace(std::move(t2));
					auto tend=std::chrono::high_resolution_clock::now();
					duration_write=1000*(std::chrono::duration_cast<std::chrono::duration<double>>(tend - tstart).count());
					t_sender_write+=duration_write;
					//std::cerr<<name<<" Frame "<<Frame-1<<" sender write time: "<<duration_write<<std::endl;
					//std::cerr<<"Sending to rec of  graph:" + std::to_string(graph_id) +name + " is not ready, it has been put in buffer\n";
				}
				else{

					//s="Sending to rec of graph:" + std::to_string(graph_id) +"_"+ name+" transferring data directly\n";
					//std::cerr<<s;



					//t2->handle()->unmap();

					//It does not work (explained above)
					//tensor->handle()->set_tensor(t2->handle()->tensor_ptr());

					tensor->handle()->map(true);

					//receiver_ready.store(false);
					*receiver_ready=false;
					//Transfer data
					const auto   output_net  = reinterpret_cast<double *>(_tensor->handle()->tensor().buffer() + _tensor->handle()->tensor().info()->offset_first_element_in_bytes());
					//std::cerr<<"graph:" + std::to_string(graph_id) +"_"+name+" _tensor desc: "<<_tensor->desc().shape<<std::endl;
					//std::cerr<<"graph:" + std::to_string(graph_id) +"_"+name+" tensor desc: "<<tensor->desc().shape<<std::endl;
					//std::cerr<<"\n\n\n\n\n\n\n\nsender tensor:  "<<output_net[0]<<","<<output_net[1]<<","<<output_net[2]<<std::endl;
					tensor->handle()->tensor().copy_from(_tensor->handle()->tensor());
					const auto   output_net2  = reinterpret_cast<double *>(tensor->handle()->tensor().buffer() + tensor->handle()->tensor().info()->offset_first_element_in_bytes());
					//std::cerr<<"receiver tensor: "<<output_net2[0]<<","<<output_net2[1]<<","<<output_net2[2]<<"\n\n\n\n\n\n"<<std::endl;

					tensor->handle()->unmap();

					//t2->handle()->unmap();
					//data_sent.store(true);
					*data_sent=true;
					condVar.notify_all();
					auto tend=std::chrono::high_resolution_clock::now();
					duration_transfer=1000*(std::chrono::duration_cast<std::chrono::duration<double>>(tend - tstart).count());
					t_sender_transfer+=duration_transfer;
					//std::cerr<<name<<" Frame "<<Frame-1<<" sender transfer time: "<<duration_transfer<<std::endl;
					//s="graph:" + std::to_string(graph_id) +name+" done\n";
					//std::cerr<<s;
				}
			}

			lck.unlock();
			/*auto tend2=std::chrono::high_resolution_clock::now();
			t_sender_wait+=std::chrono::duration_cast<std::chrono::duration<double>>(tend1 - tstart).count();
			double t=std::chrono::duration_cast<std::chrono::duration<double>>(tend2 - tend1).count();
			t_transmition+=t;
			return t;*/
			return 0;
		}
}
/*
double TensorPipelineReceiver::send_data(double* _npu_output){
		{
			std::cerr<<"Sending data from npu sender with size "<<sizeof(_npu_output)<<std::endl;
			std::string s;
			auto tstart=std::chrono::high_resolution_clock::now();
			auto tend1=std::chrono::high_resolution_clock::now();
			//if(graph_id==2 or graph_id==5){
				//std::cerr<<"sending data to "<<graph_id<<std::endl;
				//std::string sss;
				//std::cin>>sss;
			//}
			std::unique_lock<std::mutex> lck(mutex_);

			// If is receiver of a NPU
			if(is_npu){
				if(!get_receiver_ready() || !NPU_buffer.empty()){
					//add to queue
					NPU_buffer.emplace(_npu_output);
					std::cerr<<"Sending to NPU_Receiver of graph:" + std::to_string(graph_id) +name + " is not ready, it has been put in buffer\n";
				}
				else{

					s="Sending to NPU_Receiver of graph:" + std::to_string(graph_id) + name+" transferring data directly\n";
					std::cerr<<s;
					*receiver_ready=false;
					//Transfer data
					//Here Set the data into input of the NPU


					*data_sent=true;
					condVar.notify_all();

				}
			}
			// If is not receiver of an NPU
			else{
				if(!get_receiver_ready() || !buffer.empty()){
					//add to queue
					auto t2=create_and_setup_tensor(tensor);
					t2->handle()->map(true);


					//rknn_output* data=NPU_Sender->get_output();
					//Input_data=(float*)data[0].buf;
					//std::cerr<<"hereee\n";
					auto tstart=std::chrono::high_resolution_clock::now();
					if(_Transpose){
						utils::fill_tensor_array2<float,ITensor>(t2->handle()->tensor(),(float*)(_npu_output),Input_size);
					}
					else{
						utils::fill_tensor_array<float,ITensor>(t2->handle()->tensor(),(float*)(_npu_output),Input_size);
					}

					auto tfinish=std::chrono::high_resolution_clock::now();
#if NPU_Debug
					double cost0 = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
					std::cerr<<"Transfer time (transpose = "<<_Transpose<<"): "<<cost0<<std::endl;
#endif


					t2->handle()->unmap();
					buffer.emplace(std::move(t2));
					std::cerr<<"Sending to rec of  graph:" + std::to_string(graph_id) +name + " is not ready, it has been put in buffer\n";
				}
				else{

					s="Sending to rec of graph:" + std::to_string(graph_id) + name+" transferring data directly\n";
					std::cerr<<s;



					//t2->handle()->unmap();

					//It does not work (explained above)
					//tensor->handle()->set_tensor(t2->handle()->tensor_ptr());

					tensor->handle()->map(true);

					//receiver_ready.store(false);
					*receiver_ready=false;
					//Transfer data
					auto tstart=std::chrono::high_resolution_clock::now();
					if(_Transpose){
						utils::fill_tensor_array2<float,ITensor>(tensor->handle()->tensor(),(float*)(_npu_output),Input_size);
					}
					else{
						utils::fill_tensor_array<float,ITensor>(tensor->handle()->tensor(),(float*)(_npu_output),Input_size);
					}

					auto tfinish=std::chrono::high_resolution_clock::now();
//#if NPU_Debug
					double cost0 = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
					std::cerr<<"Transfer time (transpose = "<<_Transpose<<"): "<<cost0<<std::endl;
//#endif
					tensor->handle()->unmap();

					//t2->handle()->unmap();
					//data_sent.store(true);
					*data_sent=true;
					condVar.notify_all();
					//s="graph:" + std::to_string(graph_id) +name+" done\n";
					//std::cerr<<s;
				}
			}

			lck.unlock();
			//auto tend2=std::chrono::high_resolution_clock::now();
			//t_sender_wait+=std::chrono::duration_cast<std::chrono::duration<double>>(tend1 - tstart).count();
			//double t=std::chrono::duration_cast<std::chrono::duration<double>>(tend2 - tend1).count();
			//t_transmition+=t;
			//return t;
			return 0;
		}
}
*/
void TensorPipelineReceiver::wait_for_receiver(){
	{
		std::unique_lock<std::mutex> lck(mutex_);
		condVar.wait(lck, [this]{ return get_receiver_ready(); });
		lck.unlock();
	}
	return;
}
void TensorPipelineReceiver::signal_receiver(){
	{
			std::unique_lock<std::mutex> lck(mutex_);
			//data_sent.store(true);
			*data_sent=true;
			//receiver_ready.store(false);
			*receiver_ready=false;
			condVar.notify_all();
			lck.unlock();
	}
}
/*This is the implementation without buffering
bool TensorPipelineReceiver::receive_data(){
	{
		//std::lock_guard<std::mutex> lck(mutex_);
		std::string s;
		s="graph:" + std::to_string(graph_id) +name+" setting ready for getting data\n";
		std::cerr<<s;
		auto tstart=std::chrono::high_resolution_clock::now();
		std::unique_lock<std::mutex> lck(mutex_);
		receiver_ready = true;
		condVar.notify_all();
		if(!get_data_sent()){
			s="graph:" + std::to_string(graph_id) + name+" waiting for sender to send the data\n";
			std::cerr<<s;
		}
		condVar.wait(lck, [this]{ return get_data_sent(); });
		data_sent=false;
		s="graph:" + std::to_string(graph_id) + name+" Receiver done\n";
		std::cerr<<s;
		lck.unlock();
		auto tend=std::chrono::high_resolution_clock::now();
		t_receiver_wait+=std::chrono::duration_cast<std::chrono::duration<double>>(tend - tstart).count();
	}
	return true;
}
*/
bool TensorPipelineReceiver::receive_data(){
	{
		//std::lock_guard<std::mutex> lck(mutex_);
		//std::string s;
		//s="graph:" + std::to_string(graph_id) +name+" setting ready for getting data\n";
		//std::cerr<<s;
		//num_run++;
		double duration_wait=0;
		double duration_read=0;
		auto tstart=std::chrono::high_resolution_clock::now();

		//std::chrono::time_point<std::chrono::high_resolution_clock> tend;
		std::unique_lock<std::mutex> lck(mutex_);
		/******************** If is receiver of a NPU ************************/
		if(is_npu){
			if (NPU_buffer.empty() || get_data_sent()){
				//std::cerr<<"NPU receiver of graph:" + std::to_string(graph_id) +"_"+name + "nothing in buffer\n";
				//receiver_ready.store(true);
				//*receiver_ready=true;
				//condVar.notify_all();
				if(!get_data_sent()){
					//s="NPU receiver graph:" + std::to_string(graph_id) + "_"+name+" waiting for sender to send the data\n";
					//std::cerr<<s;
				}
				//tensor->handle()->map(false);
				condVar.wait(lck, [this]{ return get_data_sent(); });
				auto tend=std::chrono::high_resolution_clock::now();
				duration_wait=1000*(std::chrono::duration_cast<std::chrono::duration<double>>(tend - tstart).count());
				t_receiver_wait+=duration_wait;
				//std::cerr<<name<<" Frame "<<Frame-1<<" receiver wait time: "<<duration_wait<<std::endl;
				//data_sent.store(false);
				*data_sent=false;
				//*receiver_ready=false;
				//s="NPU Receiver graph:" + std::to_string(graph_id) + "_"+name+"transfered, Receiver done\n";
				//std::cerr<<s;
			}
			else{
				*receiver_ready=false;
				std::string s;
				/*tensor->handle()->map(true);
				buffer.front()->handle()->map(true);
				tensor->handle()->tensor().copy_from(buffer.front()->handle()->tensor());
				buffer.front()->handle()->unmap();
				buffer.pop();
				tensor->handle()->unmap();*/
				//Here Read from double* NPU_buffer into the inupt of the NPU
				auto tend=std::chrono::high_resolution_clock::now();
				duration_read=1000*(std::chrono::duration_cast<std::chrono::duration<double>>(tend - tstart).count());
				t_receiver_read+=duration_read;
				//std::cerr<<name<<" Frame "<<Frame-1<<" receiver read time: "<<duration_wait<<std::endl;
				//std::cerr<<"NPU Receiver graph:" + std::to_string(graph_id) +"_"+name + "read data from buffer\n";
			}
		}

		/********************* If is not receiver of a NPU *******************/
		else{
			if (buffer.empty() || get_data_sent()){
				//std::cerr<<"graph:" + std::to_string(graph_id) +"_"+name + "nothing in buffer\n";
				//receiver_ready.store(true);
				//*receiver_ready=true;
				//condVar.notify_all();
				if(!get_data_sent()){
					//s="graph:" + std::to_string(graph_id) + "_"+name+" waiting for sender to send the data\n";
					//std::cerr<<s;
				}
				condVar.wait(lck, [this]{ return get_data_sent(); });
				auto tend=std::chrono::high_resolution_clock::now();
				duration_wait=1000*(std::chrono::duration_cast<std::chrono::duration<double>>(tend - tstart).count());
				t_receiver_wait+=duration_wait;
				//std::cerr<<name<<" Frame "<<Frame-1<<" receiver wait time: "<<duration_wait<<std::endl;
				//data_sent.store(false);
				*data_sent=false;
				//*receiver_ready=false;
				//s="graph:" + std::to_string(graph_id) + "_"+name+"transfered, Receiver done\n";
				//std::cerr<<s;
			}
			else{
				*receiver_ready=false;
				std::string s;
				tensor->handle()->map(true);
				buffer.front()->handle()->map(true);
				tensor->handle()->tensor().copy_from(buffer.front()->handle()->tensor());
				buffer.front()->handle()->unmap();
				buffer.pop();
				tensor->handle()->unmap();
				auto tend=std::chrono::high_resolution_clock::now();
				duration_read=1000*(std::chrono::duration_cast<std::chrono::duration<double>>(tend - tstart).count());
				t_receiver_read+=duration_read;
				//std::cerr<<name<<" Frame "<<Frame-1<<" receiver read time: "<<duration_wait<<std::endl;
				//std::cerr<<"graph:" + std::to_string(graph_id) +"_"+name + "read data from buffer\n";
			}
		}
		lck.unlock();


		//auto tend=std::chrono::high_resolution_clock::now();
		//t_receiver_wait+=std::chrono::duration_cast<std::chrono::duration<double>>(tend - tstart).count();
	}
	return true;
}

void TensorPipelineReceiver::set_receiver_ready(){
	//auto s="graph:" + std::to_string(graph_id) + "_"+name+" set receiver ready\n";
	//std::cerr<<s;
	//receiver_ready.store(true);
	*receiver_ready=true;
}



void TensorPipelineSender::add_receiver(TensorPipelineReceiver* d){
	receivers.emplace_back(d);
}
std::vector<TensorPipelineReceiver*> TensorPipelineSender::get_dest(){
	return receivers;
}


void TensorPipelineSender::set_tensor(Tensor* t){
	tensor=t;
}
Tensor* TensorPipelineSender::get_tensor(){
	return tensor;
}
void TensorPipelineSender::set_name(std::string _name){
	name=std::string(_name);
}
int TensorPipelineSender::get_graph_id(){
	return graph_id;
}
void TensorPipelineSender::set_graph_id(int g_id){
	graph_id=g_id;
}


bool TensorPipelineSender::send_data(){
		//std::string s;
		//s="graph:" + std::to_string(graph_id) +"_"+name+ " before check dest_tensor\n";
		//std::cerr<<s;
		double duration_sender_sending=0;
		auto start=std::chrono::high_resolution_clock::now();
		tensor->handle()->map(true);
		for(auto rec:receivers){
			/*if(is_npu){
				double* output=nullptr;
				//output=get npu output
				rec->send_data(output);
			}
			else{
				rec->send_data(tensor);
			}*/
			rec->send_data(tensor);
			//std::cerr<<"graph:" + std::to_string(graph_id) +name +" send to "+rec->name()+" done!"<<std::endl;
		}
		tensor->handle()->unmap();
		auto end=std::chrono::high_resolution_clock::now();
		num_run++;
		Frame++;
		duration_sender_sending=1000*(std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());
		//std::cerr<<name<<" Frame "<<Frame-1<<" sender whole sending time: "<<duration_sender_sending<<std::endl;
		sending_time+=duration_sender_sending;
		//s="graph:" + std::to_string(graph_id) +"_"+name+" after check dest_tensor\n";
		//std::cerr<<s;
		return true;

	}

} // namespace graph
} // namespace arm_compute
