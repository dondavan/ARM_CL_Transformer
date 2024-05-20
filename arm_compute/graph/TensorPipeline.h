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
#ifndef ARM_COMPUTE_GRAPH_TENSOR_Pipeline_H
#define ARM_COMPUTE_GRAPH_TENSOR_Pipeline_H

#include "arm_compute/graph/Tensor.h"
#include <condition_variable>
#include <thread>

#include "arm_compute/runtime/Tensor.h"
#include <queue>

namespace arm_compute
{
namespace graph
{
/** Tensor object **/

class TensorPipelineReceiver
{
    public:
    /** Default constructor
     *
     * @param[in] id   Tensor ID
     * @param[in] desc Tensor information
     */
    TensorPipelineReceiver();

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    TensorPipelineReceiver(const arm_compute::graph::TensorPipelineReceiver &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    TensorPipelineReceiver &operator=(const arm_compute::graph::TensorPipelineReceiver &) = delete;

    double send_data(Tensor *_tensor);
    double send_data(double *_npu_output);
    void   wait_for_receiver();
    void   signal_receiver();
    bool   receive_data();

    bool    get_receiver_ready();
    void    set_receiver_ready();
    bool    get_data_sent();
    void    set_tensor(Tensor *t);
    Tensor *get_tensor();

    void set_name(std::string _name);

    void   reset_timing();
    double get_transmition_time();
    double get_sender_write_time();
    double get_receiver_wait_time();
    double get_receiver_read_time();

    int  get_graph_id();
    void set_graph_id(int g_id);
    void set_is_npu(bool is_npu)
    {
        _is_npu = is_npu;
    }

    private:
    Tensor                                                 *_tensor = nullptr;
    std::queue<std::unique_ptr<arm_compute::graph::Tensor>> _buffer;
    std::queue<double *>                                    _NPU_buffer;
    std::mutex                                              _mutex_;
    std::condition_variable                                 _condVar;
    bool                                                   *_receiver_ready = new bool(false);
    bool                                                   *_data_sent      = new bool(false);

    //For NPU
    bool         _is_npu     = false;
    unsigned int _Input_size = 0;
    bool         _Transpose = true;

    std::string _name;
    int         _graph_id;
    double      _t_sender_write    = 0;
    double      _t_sender_transfer = 0;
    double      _t_receiver_read   = 0;
    double      _t_receiver_wait   = 0;
    int         _num_run           = 0;
    int         _Frame             = 0;
};

class TensorPipelineSender
{
    public:
    /** Default constructor
     *
     * @param[in] id   Tensor ID
     * @param[in] desc Tensor information
     */
    void                                  add_receiver(TensorPipelineReceiver *d);
    std::vector<TensorPipelineReceiver *> get_dest();
    bool                                  send_data();

    void    set_tensor(Tensor *t);
    Tensor *get_tensor();
    void    set_name(std::string _name);
    int     get_graph_id();
    void    set_graph_id(int g_id);
    void    set_is_npu(bool _is_npu)
    {
        _is_npu = _is_npu;
    }
    double get_sending_time()
    {
        return _sending_time;
    }
    void reset_timing()
    {
        _sending_time = _num_run = 0;
    }

    private:
    Tensor *_tensor = nullptr;
    //vector of receivers instead of one receiver
    std::vector<TensorPipelineReceiver *> _receivers;
    std::string                           _name;
    int                                   _graph_id;
    bool                                  _is_npu       = false;
    double                                _sending_time = 0;
    int                                   _num_run      = 0;
    int                                   _Frame        = 0;
};

class TensorPipelineNPU : public Tensor
{
    public:
    bool my_call_accessor() override
    {
        return true;
    }
    virtual ~TensorPipelineNPU()
    {
    }
};

} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_TENSOR_Pipeline_H */
