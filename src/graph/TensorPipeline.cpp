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
//#include"annotate/streamline_annotate.h"
#include "utils/Utils.h"
#include <chrono>
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
std::unique_ptr<arm_compute::graph::Tensor> create_and_setup_tensor(arm_compute::graph::Tensor *main_tensor)
{
    //1-Create an empty tensor( even empty tensor descriptor)
    //When GraphBuilder::add_Receiver_node (g.add_node<ReceiverNode>(desc);) --> It calls create_tensor(); in Graph::create_tensor
    //auto tensor = std::make_unique<Tensor>(-1, TensorDescriptor());
    //tensor->desc()=main_tensor->desc();
    //tensor->desc().target = main_tensor->handle()->target;
    //std::cerr<<"Main tensor desc:"<<main_tensor->desc().data_type<<" "<<main_tensor->desc().target<<std::endl;
    auto tensor = std::make_unique<Tensor>(-1, main_tensor->desc());
    //2-In finalizaing the graph in graphmanager in force_target_to_graph(graph, forced_target);
    //It is not needed because we make its instance with main_tansor->desc()
    tensor->desc().target = main_tensor->desc().target;

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

        if(node.type() != NodeType::Const)
            std::cerr << i << "->ExecutionHelpers, Allocating output tensor for input and const node, CLTensor shape:" << tensor->handle()->tensor().info()->tensor_shape()
                      << " tensor shape:" << tensor->desc().shape
                      << std::endl;
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

arm_compute::Tensor *create_inner_tensor_cpu(arm_compute::graph::Tensor *_tensor)
{
    auto tensor = new arm_compute::Tensor;
    tensor->allocator()->init(*(_tensor->handle()->tensor().info()));
    tensor->allocator()->allocate();
    return tensor;
}
arm_compute::CLTensor *create_inner_tensor_gpu(arm_compute::graph::Tensor *_tensor)
{
    auto tensor = new arm_compute::CLTensor;
    tensor->allocator()->init(*(_tensor->handle()->tensor().info()));
    tensor->allocator()->allocate();
    return tensor;
}

TensorPipelineReceiver::TensorPipelineReceiver()
{
    //receiver_ready.store(false);
    //data_sent.store(false);
    _buffer            = {};
    *_receiver_ready   = false;
    *_data_sent        = false;

    _t_sender_write    = 0;
    _t_sender_transfer = 0;
    _t_receiver_read   = 0;
    _t_receiver_wait   = 0;
    _num_run           = 0;
}
bool TensorPipelineReceiver::get_receiver_ready()
{
    //return receiver_ready.load();
    return *_receiver_ready;
}
bool TensorPipelineReceiver::get_data_sent()
{
    //return data_sent.load();
    return *_data_sent;
}
void TensorPipelineReceiver::set_tensor(Tensor *t)
{
    _tensor = t;
}
Tensor *TensorPipelineReceiver::get_tensor()
{
    return _tensor;
}

void TensorPipelineReceiver::set_name(std::string _name)
{
    _name = std::string(_name);
}

void TensorPipelineReceiver::reset_timing()
{
    _t_sender_write    = 0;
    _t_sender_transfer = 0;
    _t_receiver_read   = 0;
    _t_receiver_wait   = 0;
    _num_run           = 0;
}
double TensorPipelineReceiver::get_transmition_time()
{
    return _t_sender_transfer;
}
double TensorPipelineReceiver::get_receiver_wait_time()
{
    return _t_receiver_wait;
}
double TensorPipelineReceiver::get_receiver_read_time()
{
    return _t_receiver_read;
}
double TensorPipelineReceiver::get_sender_write_time()
{
    return _t_sender_write;
}
int TensorPipelineReceiver::get_graph_id()
{
    return _graph_id;
}
void TensorPipelineReceiver::set_graph_id(int g_id)
{
    _graph_id = g_id;
}

double TensorPipelineReceiver::send_data(Tensor *_tensor)
{
    {
        std::string s;
        double      duration_write    = 0;
        double      duration_transfer = 0;
        _num_run++;
        _Frame++;
        auto                         tstart = std::chrono::high_resolution_clock::now();
        std::unique_lock<std::mutex> lck(_mutex_);

        /******************** If is receiver of a NPU ************************/
        if(_is_npu)
        {
            //std::cerr<<"sending data to an NPU receiver\n";
            if(!get_receiver_ready() || !_NPU_buffer.empty())
            {
                //add to queue (maybe tensor.map required)
                const auto output_net = reinterpret_cast<double *>(_tensor->handle()->tensor().buffer() + _tensor->handle()->tensor().info()->offset_first_element_in_bytes());
                _NPU_buffer.emplace(output_net);
                auto tend      = std::chrono::high_resolution_clock::now();
                duration_write = 1000 * (std::chrono::duration_cast<std::chrono::duration<double>>(tend - tstart).count());
                _t_sender_write += duration_write;
            }
            else
            {
               
                _tensor->handle()->map(true);

                *_receiver_ready = false;
                //Transfer data
                const auto output_net = reinterpret_cast<double *>(_tensor->handle()->tensor().buffer() + _tensor->handle()->tensor().info()->offset_first_element_in_bytes());
                _tensor->handle()->tensor().copy_from(_tensor->handle()->tensor());
                const auto output_net2 = reinterpret_cast<double *>(_tensor->handle()->tensor().buffer() + _tensor->handle()->tensor().info()->offset_first_element_in_bytes());
                _tensor->handle()->unmap();

                *_data_sent = true;
                _condVar.notify_all();
                auto tend         = std::chrono::high_resolution_clock::now();
                duration_transfer = 1000 * (std::chrono::duration_cast<std::chrono::duration<double>>(tend - tstart).count());
                _t_sender_transfer += duration_transfer;
            }
        }
        /********************* If is not receiver of a NPU *******************/
        else
        {
            if(!get_receiver_ready() || !_buffer.empty())
            {
                //add to queue
                auto t2 = create_and_setup_tensor(_tensor);
                t2->handle()->map(true);
                t2->handle()->tensor().copy_from(_tensor->handle()->tensor());
                t2->handle()->unmap();
                _buffer.emplace(std::move(t2));
                auto tend      = std::chrono::high_resolution_clock::now();
                duration_write = 1000 * (std::chrono::duration_cast<std::chrono::duration<double>>(tend - tstart).count());
                _t_sender_write += duration_write;
            }
            else
            {
                _tensor->handle()->map(true);

                //receiver_ready.store(false);
                *_receiver_ready = false;
                //Transfer data
                const auto output_net = reinterpret_cast<double *>(_tensor->handle()->tensor().buffer() + _tensor->handle()->tensor().info()->offset_first_element_in_bytes());
                _tensor->handle()->tensor().copy_from(_tensor->handle()->tensor());
                const auto output_net2 = reinterpret_cast<double *>(_tensor->handle()->tensor().buffer() + _tensor->handle()->tensor().info()->offset_first_element_in_bytes());
                
                _tensor->handle()->unmap();

                //t2->handle()->unmap();
                //data_sent.store(true);
                *_data_sent = true;
                _condVar.notify_all();
                auto tend         = std::chrono::high_resolution_clock::now();
                duration_transfer = 1000 * (std::chrono::duration_cast<std::chrono::duration<double>>(tend - tstart).count());
                _t_sender_transfer += duration_transfer;
                
            }
        }

        lck.unlock();
        return 0;
    }
}
void TensorPipelineReceiver::wait_for_receiver()
{
    {
        std::unique_lock<std::mutex> lck(_mutex_);
        _condVar.wait(lck, [this]
                     { return get_receiver_ready(); });
        lck.unlock();
    }
    return;
}
void TensorPipelineReceiver::signal_receiver()
{
    {
        std::unique_lock<std::mutex> lck(_mutex_);
        //data_sent.store(true);
        *_data_sent = true;
        //receiver_ready.store(false);
        *_receiver_ready = false;
        _condVar.notify_all();
        lck.unlock();
    }
}
bool TensorPipelineReceiver::receive_data()
{
    {
        double duration_wait = 0;
        double duration_read = 0;
        auto   tstart        = std::chrono::high_resolution_clock::now();

        std::unique_lock<std::mutex> lck(_mutex_);
        /******************** If is receiver of a NPU ************************/
        if(_is_npu)
        {
            if(_NPU_buffer.empty() || get_data_sent())
            {
                if(!get_data_sent())
                {
                }
                _condVar.wait(lck, [this]
                             { return get_data_sent(); });
                auto tend     = std::chrono::high_resolution_clock::now();
                duration_wait = 1000 * (std::chrono::duration_cast<std::chrono::duration<double>>(tend - tstart).count());
                _t_receiver_wait += duration_wait;
                //data_sent.store(false);
                *_data_sent = false;
            }
            else
            {
                *_receiver_ready = false;
                std::string s;
                //Here Read from double* NPU_buffer into the inupt of the NPU
                auto tend     = std::chrono::high_resolution_clock::now();
                duration_read = 1000 * (std::chrono::duration_cast<std::chrono::duration<double>>(tend - tstart).count());
                _t_receiver_read += duration_read;
                
			}
        }

        /********************* If is not receiver of a NPU *******************/
        else
        {
            if(_buffer.empty() || get_data_sent())
            {
                if(!get_data_sent())
                {
                }
                _condVar.wait(lck, [this]
                             { return get_data_sent(); });
                auto tend     = std::chrono::high_resolution_clock::now();
                duration_wait = 1000 * (std::chrono::duration_cast<std::chrono::duration<double>>(tend - tstart).count());
                _t_receiver_wait += duration_wait;
                *_data_sent = false;
            }
            else
            {
                *_receiver_ready = false;
                std::string s;
                _tensor->handle()->map(true);
                _buffer.front()->handle()->map(true);
                _tensor->handle()->tensor().copy_from(_buffer.front()->handle()->tensor());
                _buffer.front()->handle()->unmap();
                _buffer.pop();
                _tensor->handle()->unmap();
                auto tend     = std::chrono::high_resolution_clock::now();
                duration_read = 1000 * (std::chrono::duration_cast<std::chrono::duration<double>>(tend - tstart).count());
                _t_receiver_read += duration_read;
            }
        }
        lck.unlock();

    }
    return true;
}

void TensorPipelineReceiver::set_receiver_ready()
{
    *_receiver_ready = true;
}

void TensorPipelineSender::add_receiver(TensorPipelineReceiver *d)
{
    _receivers.emplace_back(d);
}
std::vector<TensorPipelineReceiver *> TensorPipelineSender::get_dest()
{
    return _receivers;
}

void TensorPipelineSender::set_tensor(Tensor *t)
{
    _tensor = t;
}
Tensor *TensorPipelineSender::get_tensor()
{
    return _tensor;
}
void TensorPipelineSender::set_name(std::string _name)
{
    _name = std::string(_name);
}
int TensorPipelineSender::get_graph_id()
{
    return _graph_id;
}
void TensorPipelineSender::set_graph_id(int g_id)
{
    _graph_id = g_id;
}

bool TensorPipelineSender::send_data()
{
    double duration_sender_sending = 0;
    auto   start                   = std::chrono::high_resolution_clock::now();
    _tensor->handle()->map(true);
    for(auto rec : _receivers)
    {
        rec->send_data(_tensor);
    }
    _tensor->handle()->unmap();
    auto end = std::chrono::high_resolution_clock::now();
    _num_run++;
    _Frame++;
    duration_sender_sending = 1000 * (std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());
    _sending_time += duration_sender_sending;
    return true;
}

} // namespace graph
} // namespace arm_compute
