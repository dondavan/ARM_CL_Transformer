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
#include "arm_compute/graph/GraphManager.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/Logger.h"
#include "arm_compute/graph/PassManager.h"
#include "arm_compute/graph/TypePrinter.h"
#include "arm_compute/graph/Utils.h"
#include "arm_compute/graph/algorithms/TopologicalSort.h"
#include "arm_compute/graph/detail/CrossLayerMemoryManagerHelpers.h"
#include "arm_compute/graph/detail/ExecutionHelpers.h"

#include "src/common/utils/Log.h"

#ifdef MEASURE_TIME
#include <chrono>
#include <fstream>
#endif

namespace arm_compute
{
namespace graph
{
GraphManager::GraphManager()
    : _workloads()
{
}

void GraphManager::finalize_graph(Graph &graph, GraphContext &ctx, PassManager &pm, Target target)
{
    ARM_COMPUTE_LOG_INFO_WITH_FUNCNAME_ACL("Initiate graph configuration!");

    // Check if graph has been registered
    if(_workloads.find(graph.id()) != std::end(_workloads))
    {
        ARM_COMPUTE_ERROR("Graph is already registered!");
    }

    // Apply IR mutating passes
    pm.run_type(graph, IGraphMutator::MutationType::IR);

    // Force target to all graph construct
    Target forced_target = target;

    switch (target)
    {
    case Target::UNSPECIFIED:
        std::cout << "src/graph/GraphManager.cpp Target::UNSPECIFIED " << std::endl;
        break;
    case Target::NEON:
        std::cout << "src/graph/GraphManager.cpp Target::NEON " << std::endl;
        break;
    case Target::CL:
        std::cout << "src/graph/GraphManager.cpp Target::CL " << std::endl;
        break;
    case Target::CLVK:
        std::cout << "src/graph/GraphManager.cpp Target::CLVK " << std::endl;
        break;
    
    default:
        break;
    }

    // In case CLVK is selected, use the CL backend and
    // update config
    if(target == Target::CLVK)
    {
        forced_target       = Target::CL;
        GraphConfig config  = ctx.config();
        config.backend_type = CLBackendType::Clvk;

        ctx.set_config(config);
    }

    if(!is_target_supported(target))
    {
        forced_target = get_default_target();
        ARM_COMPUTE_LOG_GRAPH_INFO("Switching target from " << target << " to " << forced_target << std::endl);
        std::cout << "Switching target from " << target << " to " << forced_target << std::endl;
    }
    force_target_to_graph(graph, forced_target);

    // Setup backend context
    setup_requested_backend_context(ctx, forced_target);

    // Configure all tensors
    detail::configure_all_tensors(graph);

    // Apply backend mutating passes
    pm.run_type(graph, IGraphMutator::MutationType::Backend);

    // Perform topological sort
    std::vector<NodeID> topological_sorted_nodes = dfs(graph);

    // Validate all nodes
    detail::validate_all_nodes(graph);

    // Configure all nodes
    auto workload = detail::configure_all_nodes(graph, ctx, topological_sorted_nodes);
    ARM_COMPUTE_ERROR_ON_MSG(workload.tasks.empty(), "Could not configure all nodes!");

    // Allocate const tensors and call accessors

    std::cout << "allocate_const_tensors start" << std::endl;
    detail::allocate_const_tensors(graph);
    std::cout << "allocate_const_tensors end" << std::endl;
    std::cout << "call_all_const_node_accessors start" << std::endl;
    detail::call_all_const_node_accessors(graph);

    std::cout << "call_all_const_node_accessors end" << std::endl;

    std::cout << "prepare_all_tasks start" << std::endl;
    // Prepare graph
    detail::prepare_all_tasks(workload);
    std::cout << "prepare_all_tasks end" << std::endl;


    std::cout << "Setup tensor memory start" << std::endl;
    // Setup tensor memory (Allocate all tensors or setup transition manager)
    if(ctx.config().use_transition_memory_manager)
    {
        std::cout << "use_transition_memory_manager" << std::endl;
        detail::configure_transition_manager(graph, ctx, workload);
    }
    else
    {
        std::cout << "allocate_all_tensors" << std::endl;
        detail::allocate_all_tensors(graph);
    }
    std::cout << "Setup tensor memory end" << std::endl;

    // Finalize Graph context
    ctx.finalize();

    // Register graph
    _workloads.insert(std::make_pair(graph.id(), std::move(workload)));
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Created workload for graph with ID : " << graph.id() << std::endl);
}

void GraphManager::execute_graph(Graph &graph)
{
    ARM_COMPUTE_LOG_INFO_WITH_FUNCNAME_ACL("Initiate graph execution!");

    // Check if graph is finalized
    auto it = _workloads.find(graph.id());
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");

    while(true)
    {
#ifdef MEASURE_TIME
        auto input_start_time = std::chrono::high_resolution_clock::now();
#endif
    std::cout << "call_all_input_node_accessors start" << std::endl;
        // Call input accessors
        if(!detail::call_all_input_node_accessors(it->second))
        {
            return;
        }
    std::cout << "call_all_input_node_accessors end" << std::endl;
#ifdef MEASURE_TIME
        auto   input_end_time  = std::chrono::high_resolution_clock::now();
        double input_cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(input_end_time - input_start_time).count();

        std::ofstream measure_out("measure_output.txt", std::ios::app);
        measure_out.precision(5);
        measure_out << std::scientific << "Input cost: " << input_cost_time << std::endl;

        std::cout.precision(5);
        std::cout << "Input cost: " << input_cost_time << std::endl;
#endif

#ifdef MEASURE_TIME
        auto all_task_start_time = std::chrono::high_resolution_clock::now();
#endif
std::cout << "call_all_tasks start" << std::endl;
        // Run graph
        detail::call_all_tasks(it->second);
        std::cout << "call_all_tasks end" << std::endl;
#ifdef MEASURE_TIME
        auto   all_task_end_time  = std::chrono::high_resolution_clock::now();
        double all_task_cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(all_task_end_time - all_task_start_time).count();

        measure_out.precision(5);
        measure_out << std::scientific << "All_task cost: " << all_task_cost_time << std::endl;

        std::cout.precision(5);
        std::cout << "All_task cost: " << all_task_cost_time << std::endl;
#endif

#ifdef MEASURE_TIME
        auto output_start_time = std::chrono::high_resolution_clock::now();
#endif
std::cout << "call_all_output_node_accessors start" << std::endl;
        // Call output accessors
        if(!detail::call_all_output_node_accessors(it->second))
        {
            return;
        }
std::cout << "call_all_output_node_accessors end" << std::endl;
#ifdef MEASURE_TIME
        auto   output_end_time  = std::chrono::high_resolution_clock::now();
        double output_cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(output_end_time - output_start_time).count();

        measure_out.precision(5);
        measure_out << std::scientific << "Output cost: " << output_cost_time << std::endl;

        std::cout.precision(5);
        std::cout << "Output cost: " << output_cost_time << std::endl;
        measure_out.close();
#endif
    }
}

void GraphManager::invalidate_graph(Graph &graph)
{
    auto it = _workloads.find(graph.id());
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");

    _workloads.erase(it);
}
} // namespace graph
} // namespace arm_compute
