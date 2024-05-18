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
#ifndef ARM_COMPUTE_GRAPH_DETAIL_EXECUTION_HELPERS_PIPELINE_H
#define ARM_COMPUTE_GRAPH_DETAIL_EXECUTION_HELPERS_PIPELINE_H

#include "arm_compute/graph/Types.h"
#include "arm_compute/graph/detail/ExecutionHelpers.h"


namespace arm_compute
{
namespace graph
{
// Forward declarations
class Graph;
class GraphContext;
struct ExecutionWorkloadPipeline;
class Tensor;
class INode;

namespace detail
{

ExecutionWorkload configure_all_nodes_pipeline(Graph &g, GraphContext &ctx, const std::vector<NodeID> &node_order);

double call_all_senders(ExecutionWorkload &workload);

bool call_all_receivers(ExecutionWorkload &workload);

void allocate_const_tensors_pipeline(Graph &g);

void call_all_tasks_pipeline(ExecutionWorkload &workload,int n=0);

void reset_transmit_timings(ExecutionWorkload &workload);
void reset_NPU_timings(ExecutionWorkload &workload);
void NPU_set_preallocated_outputs(ExecutionWorkload &workload);
int NPU_destroy(ExecutionWorkload &workload);
void print_NPU_times(ExecutionWorkload &workload, int num_run);
void set_ending_tasks(ExecutionWorkload &workload, std::vector<std::string>* ending_tasks=nullptr);
void set_governor_tasks(ExecutionWorkload &workload, std::vector<std::string>* governor_tasks=nullptr);
//void set_GPIO_tasks(ExecutionWorkload &workload, std::string mode, std::vector<std::string>* GPIO_tasks=nullptr);
//void set_tasks_freqs(ExecutionWorkload &workload, std::vector<std::array<int, 3>> _freqs, bool repeat);

} // namespace detail
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_DETAIL_EXECUTION_HELPERS_H */
