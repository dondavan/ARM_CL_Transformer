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

#include "utils/Utils.h"
#include "utils/CommonGraphOptions.h"
#include "arm_compute/graph.h"


namespace arm_compute
{
namespace utils
{


typedef std::vector<std::string> stringvec;
void read_directory(const std::string& name, stringvec& v);

/** Abstract Example class.
 *
 * All examples have to inherit from this class.
 */
class Example_Pipeline: public Example
{
public:
	Example_Pipeline(int _id, char _PE, int start, int end, char _host_PE)
		: common_params(), graph(_id, "YOLOv3"), start_indx(start), end_indx(end), PE(_PE), host_PE(_host_PE)
	{
		id = _id;
	}
	Example_Pipeline(int _id=0)
		: common_params(), graph(_id, "YOLOv3")
	{
		id = _id;
	}
	void init(int _id, char _PE, int start, int end, char _host_PE)
	{
		assert(_id == id);
		start_indx = start;
		end_indx = end;
		PE = _PE;
		if (PE=='B' || PE=='L'){
			target=arm_compute::graph::Target ::NEON;
		}
		if (PE=='G'){
			target=arm_compute::graph::Target ::CL;
		}
		host_PE = _host_PE;
		//npu_init_context(NPU_index);
		return;
	}

	void set_start(int start){
		start_indx = start;
	}
	void set_end(int end){
		end_indx = end;
	}
	void set_PE(char _PE){
		PE = _PE;
	}
	void set_host(char _host_PE){
		host_PE = _host_PE;
	}
	int get_start(){
		return start_indx;
	}
	int get_end(){
		return end_indx;
	}
	char get_PE(){
		return PE;
	}
	char get_host(){
		return host_PE;
	}
	int get_id(){
		return id;
	}

	bool do_setup_pipeline(int argc, char **argv);

    /** Default destructor. */
    virtual ~Example_Pipeline() = default;
protected:


	CommonGraphParams  			common_params;
	graph::frontend::Stream	graph;
	int								start_indx;
	int 							end_indx;
	char 							PE;
	int 							id;
	char 							host_PE;
	arm_compute::graph::Target		target;
	//rknn_context 					ctx;
	//
};


//template <typename T>
class Pipeline_Manager{
public:
	int add_stage(std::unique_ptr<Example_Pipeline> stage, int id, int start, int end, char PE, CommonGraphParams common_params){
		char host_PE = PE;
		if (PE == 'G'){
			host_PE = common_params.gpu_host;
		}
		if (PE == 'N'){
			host_PE = common_params.npu_host;
		}
		std::cerr<<"Initializing pipeline stage-"<<stage->get_id()<<" on PE: "<<PE<<
				"  For layers "<<start<<"-"<<end<<" and Host PE is "<<host_PE<<std::endl;
		stage->init(id, PE, start, end, host_PE);
		stages.push_back(std::move(stage));
		return 0;
	}
	void add_stage(std::unique_ptr<Example_Pipeline> stage){
			stages.push_back(std::move(stage));
		}
	int get_next_id(){
		return stages.size();
	}
	/*std::unique_ptr<Example_Pipeline> get_stage(int i){
		return stages[i];
	}
	std::vector<std::unique_ptr<Example_Pipeline>> get_stages(){
		return stages;
	}*/

private:
	std::vector<std::unique_ptr<Example_Pipeline>> stages;
	CommonGraphParams  common_params;
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
	CommandLineParser  cmd_parser;
	CommonGraphOptions common_opts(cmd_parser);
	CommonGraphParams  common_params;
	// Parse arguments
	cmd_parser.parse(argc, argv);
	cmd_parser.validate();

	// Consume common parameters
	common_params = consume_common_graph_parameters(common_opts);

	// Return when help menu is requested
	if(common_params.help)
	{
		cmd_parser.print_help(argv[0]);
		return false;
	}

	//Provide directory of images in addition to one image file as image argument
	bool imgs = !(common_params.image.empty());
	stringvec images_list;
	size_t image_index = 0;
	if(imgs){
		read_directory(common_params.image, images_list);
		std::cerr<<images_list.size()<<" Input images are read from "<<common_params.image<<std::endl;
		common_params.image = images_list[image_index];
	}

	Pipeline_Manager pipeline_manager;
	std::string order = common_params.order;
	int Layers = order.size();
	if (Layers == 0){
		order='B';
	}
	int id = 0;
	char PE = order[0];
	int start = 0;
	int end = 0;
	for(int i=1 ; i < Layers ; i++){
		if(order[i] != PE){
			end = i-1;
			id=pipeline_manager.get_next_id();
			pipeline_manager.add_stage(std::make_unique<T>(id), id, start, end, PE, common_params);
			start = i;
			PE = order[i];
		}
		if(i == Layers-1){
			end = i;
			id=pipeline_manager.get_next_id();
			pipeline_manager.add_stage(std::make_unique<T>(id), id, start, end, PE, common_params);
		}
	}




	/*
	 * bool create_pipeline(){
			std::string order = common_params.order;
			int Layers = order.size();
			if (Layers == 0){
				return true;
			}
			//Pipeline_Manager<T> pipeline_manager;
			int id = 0;
			char PE = order[0];
			int start = 0;
			int end = 0;
			for(int i=1 ; i < Layers ; i++){
				if(order[i] != PE){
					end = i;
					id = create_stage(start, end, PE, common_params);
					start = i;
					PE = order[i];
				}
				if(i == Layers-1){
					end = i;
					id = create_stage(start, end, PE, common_params);
				}
			}
			return true;
		}
	 */


	//Setup GPIO for sending (start and end) signals to power manager
#if Power_Measurement
	if (-1 == GPIOExport(POUT)){
		std::cerr<<"Could not Export GPIO\n";
		return(-1);
	}
	if (-1 == GPIODirection(POUT, OUT)){
		std::cerr<<"Could not set GPIO direction\n";
		return(-2);
	}
	if (-1 == GPIOWrite(POUT, 0)){
		std::cerr<<"Could not write 0 to GPIO\n";
		return(-3);
	}
#endif

	// Print parameter values
	std::cout << common_params << std::endl;





	/*
	int n_l=13;
	std::cerr<<"Number of Layers: "<<n_l<<std::endl;
	std::string lbl=common_params.labels;
	if(common_params.order.size()==1){
		common_params.order=std::string(n_l, common_params.order[0]);
	}
	if(common_params.order[1]=='-'){
		common_params.order=std::string(common_params.partition_point,common_params.order[0])+
				std::string(common_params.partition_point2-common_params.partition_point,common_params.order[2])+
				std::string(n_l-common_params.partition_point2,common_params.order[4]);
	}
	std::string order=common_params.order;

	int Layers=order.size();
	//Another option to parse is string.find_first_of and string.find_last_of
	int g=0;
	//NPU
	int start_N=-1;
	int end_N=-1;
	for(int i=0;i<Layers;i++){
		if (order[i]=='N'){
			if (start_N==-1){
				start_N=i;
				end_N=i;
			}
			else{
				end_N=i;
			}
		}
	}
	std::string NPU_Model_Name='_'+std::to_string(start_N+1)+'_'+std::to_string(end_N+1)+".rknn";
	for(int i=0;i<Layers;i++){
		if(i==0){
			if (order[i]=='B'){
				targets.push_back(arm_compute::graph::Target ::NEON);
				classes.push_back(1);
			}
			if (order[i]=='L'){
				targets.push_back(arm_compute::graph::Target ::NEON);
				classes.push_back(0);
			}
			if (order[i]=='G'){
				targets.push_back(arm_compute::graph::Target ::CL);
				classes.push_back(2);
			}
			if (order[i]=='N'){
				//targets.push_back(arm_compute::graph::Target ::NEON);
				//classes.push_back(3);
				rknn_context ctx=0;
				NPU_Contexts.push_back(ctx);
			}
			if (order[i]!='-' && order[i]!='N' ){
				graphs.push_back(new Stream(g,"GoogleNet"));
				gr_layer[i]=g;
			}
			if(order[i]=='-'){
				gr_layer[i]=-1;
			}
			if(order[i]=='N'){
				gr_layer[i]=-2;
			}
		}

		else if (order[i]!=order[i-1]){
			if(order[i]=='-'){
				gr_layer[i]=-1;
			}
			else if(order[i]=='N'){
				gr_layer[i]=-2;
				rknn_context ctx=0;
				NPU_Contexts.push_back(ctx);
			}
			else{
				if (order[i]=='B'){
					targets.push_back(arm_compute::graph::Target ::NEON);
					classes.push_back(1);
				}
				if (order[i]=='L'){
					targets.push_back(arm_compute::graph::Target ::NEON);
					classes.push_back(0);
				}
				if (order[i]=='G'){
					targets.push_back(arm_compute::graph::Target ::CL);
					classes.push_back(2);
				}

				graphs.push_back(new Stream(g+1,"GoogleNet"));
				gr_layer[i]=graphs.size()-1;
				g=graphs.size()-1;
			}

		}

		else{
			if(order[i]!='-' && order[i]!='N')
				gr_layer[i]=g;
			if(order[i]=='-')
				gr_layer[i]=-1;
			if(order[i]=='N')
				gr_layer[i]=-2;
		}
	}
	for(int i=0;i<Layers;i++){
		//std::cerr<<i<<"\t"<<gr_layer[i]<<std::endl;
		if(order[i]=='-' || order[i]=='N'){
			dump_graph=new Stream(1000,"GoogleNEt");
			break;
		}
	}
#if NPU_Debug
	std::cerr<<"graph layers:\n";
	for(int i=0;i<Layers;i++){
				std::cerr<<i<<"\t"<<gr_layer[i]<<std::endl;
	}
#endif
	////per_frame=(graphs.size()>1);
	//for(int i=0;i<8;i++){
	//	std::cout<<"Layer:"<<i<<'\t'<<"graph:"<<gr_layer[i]<<'\t'<<"class:"<<classes[gr_layer[i]]<<'\t'<<"target:"<<int(targets[gr_layer[i]])<<std::endl;
	}//


	cpu_set_t set;
	CPU_ZERO(&set);
	//NPU:
	//
	if(gr_layer[Layer]>0){
		//CPU_SET(host_core[classes[gr_layer[Layer]]],&set);
		set_cores(&set,one_master_core,host_core[classes[gr_layer[Layer]]]);
		ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");
	}
	std::cout << common_params << std::endl;

	annotate=common_params.annotate;
	//ann=annotate;
	save_model=common_params.save;

	//If subgraph is dummy
	if(gr_layer[Layer]==-1 ){
		sub_graph=dump_graph;
		common_params.target=arm_compute::graph::Target ::NEON;
	}

	//If subgraph is NPU
	else if(gr_layer[Layer]==-2){
		sub_graph=dump_graph;
		common_params.target=arm_compute::graph::Target ::NEON;
		NPU_index++;
		npu_init_context(NPU_index);
#if NPU_Debug
		std::cerr<<"Setup: init npu model\n";
#endif
		//Input_Accessor=get_input_accessor(common_params, std::move(preprocessor), true, NPU_Contexts[NPU_index]).get();
		Input_Accessor=get_input_accessor(common_params, std::move(preprocessor), true, &NPU_Contexts[NPU_index],tensor_shape.total_size()).release();
		im_acc=dynamic_cast<ImageAccessor*>(Input_Accessor);

		arm_compute::TensorInfo info(input_descriptor.shape,1,input_descriptor.data_type,operation_layout);
		Input_tensor.allocator()->init(info);
		Input_tensor.allocator()->allocate();

	}
	//If subgraph is real
	else{
		sub_graph=(graphs[gr_layer[Layer]]);
		common_params.target=targets[gr_layer[Layer]];
	}
	*/
//***************************************************************























    //return run_example_pipeline(argc, argv, std::make_unique<T>());
    return run_example_pipeline(argc, argv, std::make_unique<T>());
}


} // namespace utils
} // namespace arm_compute
#endif /* __UTILS_UTILS_H__*/
