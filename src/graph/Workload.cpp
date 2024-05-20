/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "arm_compute/graph/Workload.h"

#include "arm_compute/graph/INode.h"
#include "arm_compute/graph/ITensorHandle.h"
#include "arm_compute/graph/nodes/PrintLayerNode.h"

#include "support/Cast.h"

//Ehsan
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "utils/DVFS/DVFS.h"
#include "utils/Power/Power.h"

namespace arm_compute
{
namespace graph
{
//DVFS ExecutionTask::dvfs;
DVFS dvfs;
void ExecutionTask::init()
{
    dvfs.init();
}
void ExecutionTask::finish()
{
    dvfs.finish();
}

void ExecutionTask::apply_freq(std::string _name)
{
    ARM_COMPUTE_UNUSED(_name);
    if(governor)
    {
        dvfs.commit_freq(LittleFreq, bigFreq, GPUFreq);
    }
}

void ExecutionTask::switch_GPIO_starting()
{
    if(starting_gpio_switch)
    {
        if(-1 == GPIOWrite(POUT, 0))
            std::cerr << "Could not write 0 to GPIO\n";
    }
}
void ExecutionTask::switch_GPIO_ending()
{
    if(ending_gpio_switch)
    {
        if(-1 == GPIOWrite(POUT, 1))
            std::cerr << "Could not write 1 to GPIO\n";
    }
}

int ExecutionTask::get_max_l()
{
    return dvfs.get_max_l();
}
int ExecutionTask::get_max_b()
{
    return dvfs.get_max_b();
}
int ExecutionTask::get_max_g()
{
    return dvfs.get_max_g();
}

/*
void ExecutionTask::operator()()
{
    TaskExecutor::get().execute_function(*this);
}
*/
void ExecutionTask::operator()()
{
    switch_GPIO_starting();
    if(!profile_transfers)
    {
        TaskExecutor::get().execute_function(*this);
    }
    else
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(8));
    }
    apply_freq();
    switch_GPIO_ending();
    if(profile_layers)
    {
        if(ending_gpio_switch)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(8));
        }
    }
}

void ExecutionTask::operator()(int nn)
{
    switch_GPIO_starting();
    if(!profile_transfers)
    {
        t += TaskExecutor::get().execute_function2(*this, nn);
    }
    else
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(8));
    }
    apply_freq();
    switch_GPIO_ending();
    if(profile_layers)
    {
        if(ending_gpio_switch)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(8));
        }
    }
}

double ExecutionTask::time(int n)
{
    if(n != 0)
        return t / n;
    else
        return t;
}

void ExecutionTask::reset()
{
    t = 0;
    n = 0;
}

void execute_task(ExecutionTask &task)
{
    if(task.task)
    {
        task.task->run();
    }
#ifdef ARM_COMPUTE_ASSERTS_ENABLED
    else if(task.node->type() == NodeType::PrintLayer)
    {
        auto print_node   = utils::cast::polymorphic_downcast<PrintLayerNode *>(task.node);
        auto input_handle = print_node->input(0)->handle();
        auto transform    = print_node->transform();

        input_handle->map(true);
        ITensor *input = transform ? transform(&input_handle->tensor()) : &input_handle->tensor();
        input->print(print_node->stream(), print_node->format_info());
        input_handle->unmap();
    }
#endif // ARM_COMPUTE_ASSERTS_ENABLED
}

double execute_task2(ExecutionTask &task, int nn)
{
    double t = 0;
    if(task.task)
    {
        auto start = std::chrono::high_resolution_clock::now();
        task.task->run();
        if(task.block)
        {
            arm_compute::CLScheduler::get().queue().finish();
        }
        auto finish = std::chrono::high_resolution_clock::now();
        t           = 1000 * (std::chrono::duration_cast<std::chrono::duration<double>>(finish - start).count());
    }
#ifdef ARM_COMPUTE_ASSERTS_ENABLED
    // COMPMID-3012 - Hide the printing logic from the execute_task method in the graph API
    else if(task.node->type() == NodeType::PrintLayer)
    {
        auto print_node   = dynamic_cast<PrintLayerNode *>(task.node);
        auto input_handle = print_node->input(0)->handle();
        auto transform    = print_node->transform();

        input_handle->map(true);
        ITensor *input = transform ? transform(&input_handle->tensor()) : &input_handle->tensor();
        input->print(print_node->stream(), print_node->format_info());
        input_handle->unmap();
    }
#endif // ARM_COMPUTE_ASSERTS_ENABLED
    return t;
}

void ExecutionTask::prepare()
{
    if(task)
    {
        task->prepare();
    }
}

TaskExecutor::TaskExecutor()
    : execute_function(execute_task), execute_function2(execute_task2)
{
}

TaskExecutor &TaskExecutor::get()
{
    static TaskExecutor executor;
    return executor;
}
} // namespace graph
} // namespace arm_compute
