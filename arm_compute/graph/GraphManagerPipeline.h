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
#ifndef ARM_COMPUTE_GRAPH_GRAPH_MANAGER_PIPELINE_H
#define ARM_COMPUTE_GRAPH_GRAPH_MANAGER_PIPELINE_H


#include "arm_compute/graph/GraphManager.h"
//#include "arm_compute/graph/WorkloadPipeline.h"

#include<chrono>
#include "arm_compute/graph/printers/DotGraphPrinter.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/Logger.h"
#include "arm_compute/graph/PassManager.h"
#include "arm_compute/graph/TypePrinter.h"
#include "arm_compute/graph/Utils.h"
#include "arm_compute/graph/detail/CrossLayerMemoryManagerHelpers.h"
#include "arm_compute/graph/detail/ExecutionHelpersPipeline.h"
#include "arm_compute/graph/algorithms/TopologicalSort.h"



namespace arm_compute
{
namespace graph
{
// Forward declaration
class Graph;
class GraphContext;
class PassManager;

/** Graph manager class
 *
 * Manages a list of graphs along with their resources
 */


class GraphManagerPipeline:public GraphManager
{
public:
    /** Default Constructor **/
	GraphManagerPipeline();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
	GraphManagerPipeline(const GraphManagerPipeline &) = delete;
    /** Default move constructor */
	GraphManagerPipeline(GraphManagerPipeline &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
	GraphManagerPipeline &operator=(const GraphManagerPipeline &) = delete;
    /** Default move assignment operator */
	GraphManagerPipeline &operator=(GraphManagerPipeline &&) = default;
    /** Finalizes a given graph
     *
     * @warning At this given time finalize_graph will alter the passed graph,
     *          plan is to avoid by copying the graph structure,
     *          or provide another entry-point for this functionality as it will increase the memory requirements
     *
     * @param[in] graph  Graph to finalize
     * @param[in] ctx    Graph context
     * @param[in] pm     Pass manager to use for any optimization passes
     * @param[in] target Execution target (Single target execution is currently supported)
     */
    //void finalize_graph(Graph &graph, GraphContext &ctx, PassManager &pm, Target target, std::set<int> *b=NULL, int blocking=0);
    /** Executes a graph
     *
     * @param[in] graph Graph to execute
     */
    //Ehsan
    void execute_graph(Graph &graph);
    void execute_graph(Graph &graph, int nn=0);
    //if there is no buffer in between stages
    void warmup_and_execute_graph_no_buffer(Graph &graph, int nn);
    //If you want to buffer output of a stage if next stage is busy:
    void warmup_and_execute_graph_pipeline(Graph &graph, int nn);
    void warmup_and_execute_graph_serial(Graph &graph, int nn);
    //void execute_graph(Graph &graph, bool annotate, int nn=0);
    //void execute_graph(Graph &graph, double &in, double &task, double &out, int nn=0);
    //void execute_graph(Graph &graph, double &in, double &task, double &out, bool annotate, int nn=0);
    /** Invalidates the graph execution workload
     *
     * @param[in] graph Graph to invalidate
     */
    //void invalidate_graph(Graph &graph);

    //Ehsan
    //void print_times(Graph &graph, int n);
    //void reset(Graph &graph);

    void set_input_time(int target, double t){
    	input_time[target]=t;
    }
    void set_task_time(int target, double t){
        task_time[target]=t;
    }
    void set_output_time(int target, double t){
        output_time[target]=t;
    }


    double get_input_time(int target){
    	return input_time[target];
    }
    double get_task_time(int target){
    	return task_time[target];
    }
    double get_output_time(int target){
    	return output_time[target];
    }

    void print_times(int n);
    void print_times(int graph_id, int n);
    void print_times_details(int graph_id, int n);
    void reset(Graph &graph);
    void set_num_graphs(int n){
    	num_graphs=n;
    	input_time.resize(n);
    	receive_time.resize(n);
    	task_time.resize(n);
    	send_time.resize(n);
    	output_time.resize(n);
    	transmition_time.resize(n);
    }
    void print_tasks();
    void reset_timing(int graph_id);

    void finalize_graph(Graph &graph, GraphContext &ctx, PassManager &pm, Target target, std::set<int> *blocking_set=nullptr, int blocking=0);

    bool* get_ready(){
    	return &ready;
    }
    std::vector<std::string>& get_input_list(){
    	return input_images;
    }

    std::map<GraphID, ExecutionWorkload>& get_workloads(){
    	return _workloads;
    }

    void set_freqs(std::string freqs, std::string _order, char GPU_Host, char NPU_Host);
    void extract_governor_tasks(std::string);
    void set_freq_map(std::string freqs, std::string _order, char GPU_Host, char NPU_Host);
    void set_governor_freqs();
    int destroy();

    void set_GPIO_tasks(std::string power_profie_mode);


private:

    std::map<GraphID, ExecutionWorkload> 	_workloads = {}; /**< Graph workloads */

    std::vector<double> 					input_time;
    std::vector<double> 					receive_time;
    std::vector<double> 					task_time;
    std::vector<double> 					output_time;
    std::vector<double> 					send_time;
    std::vector<double>						transmition_time;
    double									latency_time=0;
    int 									num_graphs=1;
    int										warmup_n=3;
	bool									parallel=true;
	bool									measure_when_full=true&&parallel;
	std::mutex 								_mtx = {};
	std::condition_variable 				condVar;
	std::condition_variable 				condVar_serial;
	bool									ready=true;
	int										c=0;
	std::atomic<bool>						pipeline_ready;
	std::vector<std::string>				input_images;

	std::vector<std::string>				governor_tasks;
	std::map<std::string, std::array<int, 3>> governor_freqs;
	//std::map<std::string, std::array<int, 3>> graphs_freqs;
	//std::map<std::string, std::array<int, 3>> PEs_freqs;

};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_GRAPH_MANAGER_H */
