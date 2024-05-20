/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#ifndef __UTILS_UTILS_PIPELINE_H__
#define __UTILS_UTILS_PIPELINE_H__

#include "arm_compute/graph.h"
#include "utils/CommonGraphOptions.h"
#include "utils/Utils.h"

//#include "arm_compute/graph.h"

namespace arm_compute
{
namespace utils
{

typedef std::vector<std::string> stringvec;
void                             read_directory(const std::string &name, stringvec &v);

// Helper function to convert a string to lowercase
std::string toLowerCase(std::string str);
// Function to remove "net" from the end of a string if it exists
std::string removeNetFromEnd(std::string str);

//std::unordered_set<std::string> get_end_task_names(std::string graph_name="alex");
/** Abstract Example class.
 *
 * All examples have to inherit from this class.
 */
class Example_Pipeline : public Example
{
    public:
    Example_Pipeline(int _id, std::string name)
        //:	cmd_parser(), common_opts(cmd_parser), common_params(), graph(_id,std::move(_name))
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, name), _name(name)
    {
        graph::frontend::IStreamPipeline::_graph_name = toLowerCase(name);
        //graph::frontend::IStreamPipeline::ending_tasks=get_end_task_names(graph::frontend::IStreamPipeline::graph_name);
    }

    void init(int _id, char _PE, int start, int end, char _host_PE)
    {
        assert(_id == id);
        start_indx = start;
        end_indx   = end;
        PE         = _PE;
        if(PE == 'B' || PE == 'L')
        {
            target = arm_compute::graph::Target ::NEON;
        }
        if(PE == 'G')
        {
            target = arm_compute::graph::Target ::CL;
        }
        host_PE = _host_PE;
        //npu_init_context(NPU_index);
        return;
    }

    int init(int argc, char **argv)
    {
        cmd_parser.parse(argc, argv);
        cmd_parser.validate();

        // Consume common parameters
        common_params = consume_common_graph_parameters(common_opts);

        // Return when help menu is requested
        if(common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return -1;
        }

        //Provide directory of images in addition to one image file as image argument
        bool      imgs = !(common_params.image.empty());
        stringvec images_list;
        size_t    image_index = 0;
        if(imgs)
        {
            //read_directory(common_params.image, images_list);
            //std::cerr<<"[UtilsPipeline.h] image directory is: "<<common_params.image<<std::endl;
            read_directory(common_params.image, graph.manager()->get_input_list());
            std::cout << graph.manager()->get_input_list().size() << " Input images are read from " << common_params.image << std::endl;
            //common_params.image = images_list[image_index];
            //common_params.image = graph.manager()->get_input_list()[0];
            common_params.image = common_params.image + "/";
            std::cout << "[UtilsPipeline.h] image directory is: " << common_params.image << std::endl;
        }
        // Print parameter values
        std::cout << common_params << std::endl;

        graph.set_num_runs(common_params.n);

        return 0;
        //example_pipeline->common_opts=common_opts;
    }
    int config_pipeline()
    {
        set_common_params(common_params);

        std::string order  = common_params.order;
        int         Layers = order.size();
        if(Layers == 0)
        {
            order = 'B';
        }
        int  id    = 0;
        char PE    = order[0];
        int  start = 0;
        int  end   = 0;
        for(int i = 1; i < Layers; i++)
        {
            if(order[i] != PE)
            {
                end = i - 1;
                //id=example_pipeline->graph.get_next_id();
                char Host_PE = PE;
                if(PE == 'G')
                {
                    Host_PE = common_params.gpu_host;
                }
                if(PE == 'N')
                {
                    Host_PE = common_params.npu_host;
                }
                add_graph(start, end, PE, Host_PE);
                start = i;
                PE    = order[i];
            }
            if(i == Layers - 1)
            {
                end          = i;
                char Host_PE = PE;
                if(PE == 'G')
                {
                    Host_PE = common_params.gpu_host;
                }
                if(PE == 'N')
                {
                    Host_PE = common_params.npu_host;
                }
                add_graph(start, end, PE, Host_PE);
            }
        }

        return 0;
    }

    void set_start(int start)
    {
        start_indx = start;
    }
    void set_end(int end)
    {
        end_indx = end;
    }
    void set_PE(char _PE)
    {
        PE = _PE;
    }
    void set_host(char _host_PE)
    {
        host_PE = _host_PE;
    }
    int get_start()
    {
        return start_indx;
    }
    int get_end()
    {
        return end_indx;
    }
    char get_PE()
    {
        return PE;
    }
    char get_host()
    {
        return host_PE;
    }
    int get_id()
    {
        return id;
    }

    bool do_setup_pipeline(int argc, char **argv);

    /** Default destructor. */
    virtual ~Example_Pipeline() = default;

    void add_graph(int start, int end, char PE, char Host_PE)
    {
        graph.add_graph(start, end, PE, Host_PE);
        return;
    }
    void set_common_params(CommonGraphParams _common_params)
    {
        graph.set_common_params(_common_params);
    }

    void initialize_dvfs();
    int  get_max_l()
    {
        return arm_compute::graph::ExecutionTask::get_max_l();
    }
    int get_max_b()
    {
        return arm_compute::graph::ExecutionTask::get_max_b();
    }
    int get_max_g()
    {
        return arm_compute::graph::ExecutionTask::get_max_g();
    }

    void initialize_power_measurement();

    std::string name()
    {
        return _name;
        //return graph.name();
    }
    CommonGraphParams get_common_params()
    {
        return common_params;
    }

    void do_teardown() override
    {
        graph.manager()->destroy();
    }

    /*std::vector<std::unique_ptr<arm_compute::graph::Graph>> get_graphs(){
    	return graph.get_graphs();
    }
    std::map<arm_compute::graph::GraphID, arm_compute::graph::ExecutionWorkload>& get_workloads(){
        	return graph.manager()->get_workloads();
    }
    void extract_ending_tasks(){
    	auto &workloads=get_workloads();
    	for (unsigned int id = 0; id < workloads.size(); ++id) {
    		auto it = workloads.find(id);
    		auto &workload = it->second;

    	}
    }*/
    void set_freqs(std::string freqs)
    {
        graph.manager()->set_freqs(freqs, common_params.order, common_params.gpu_host, common_params.npu_host);
    }

    void set_GPIOs(std::string power_profie_mode)
    {
        graph.manager()->set_GPIO_tasks(power_profie_mode);
    }
    void print_tasks()
    {
        graph.manager()->print_tasks();
    }

    /*CommonGraphParams  				common_params;
	CommonGraphOptions 				common_opts;
	CommandLineParser  				cmd_parser;*/
    protected:
    CommandLineParser               cmd_parser;
    CommonGraphOptions              common_opts;
    CommonGraphParams               common_params;
    graph::frontend::StreamPipeline graph;
    int                             start_indx;
    int                             end_indx;
    char                            PE;
    int                             id;
    char                            host_PE;
    arm_compute::graph::Target      target;
    std::string                     _name;
    //rknn_context 					ctx;
    //
};

/** Run an example and handle the potential exceptions it throws
 *
 * @param[in] argc    Number of command line arguments
 * @param[in] argv    Command line arguments
 * @param[in] example Example to run
 */
int run_example_pipeline(int argc, char **argv, std::unique_ptr<Example_Pipeline> example);

template <typename T>
int run_example_pipeline(int argc, char **argv)
{
    std::unique_ptr<Example_Pipeline> example_pipeline = std::make_unique<T>();
    /*CommandLineParser  cmd_parser;
	CommonGraphOptions common_opts(cmd_parser);
	CommonGraphParams  common_params;*/
    // Parse arguments
    int r;

    r = example_pipeline->init(argc, argv);
    if(r < 0)
    {
        std::cerr << "error in initializing the options\n";
        return r;
    }
    example_pipeline->config_pipeline();
    if(r < 0)
    {
        std::cerr << "error in configuring the pipeline\n";
        return r;
    }

    //Setup GPIO for sending (start and end) signals to power manager
#if Power_Measurement
    if(-1 == GPIOExport(POUT))
    {
        std::cerr << "Could not Export GPIO\n";
        return (-1);
    }
    if(-1 == GPIODirection(POUT, OUT))
    {
        std::cerr << "Could not set GPIO direction\n";
        return (-2);
    }
    if(-1 == GPIOWrite(POUT, 0))
    {
        std::cerr << "Could not write 0 to GPIO\n";
        return (-3);
    }
#endif

    return run_example_pipeline(argc, argv, std::move(example_pipeline));
}

} // namespace utils
} // namespace arm_compute
#endif /* __UTILS_UTILS_H__*/
