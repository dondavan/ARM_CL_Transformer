/*
 * NPURockPi.cpp
 *
 *  Created on: Jul 7, 2023
 *      Author: ehsan
 */
//#include "rockx.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <algorithm>
#include <bitset>
//#include <android/log.h>
#include <chrono>

//#include "arm_compute/runtime/NPU/rknn_api.h"


#include "arm_compute/runtime/NPU/NPU.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "utils/Utils.h"

namespace arm_compute
{
//class NPUTemplate;
//class NPURockPi;
//using NPUType=RockPi;
const NPUTypes NPUType = NPUTypes::RockPi;
template <>
NPU<NPUType>::NPU(int _id)
{
	//std::cerr<<"Rockpi npu\n";
	id=_id;
	NPU_Data._NPU_Context=new rknn_context(id);
	preallocated_output=!_Transpose;
	std::cerr<<"Creating a RockPi NPU node...\n";

}

template <>
NPU<NPUType>::NPU(NPU<NPUType> &&) = default;

template <>
NPU<NPUType> &NPU<NPUType>::operator=(NPU<NPUTypes::RockPi> &&) = default;

template <>
NPU<NPUType>::~NPU()                               = default;


template <>
void NPU<NPUType>::set_preallocated_outputs(){
	/*int i=0;
	for(auto out:outputs){
		rknn_output output;
		int ret = rknn_query(*NPU_Data._NPU_Context, RKNN_QUERY_OUTPUT_ATTR, &(NPU_Data.output_attr[i]), sizeof(rknn_tensor_attr));
		if(ret < 0) {
			printf("Error: Query Output fail! ret=%d\n",ret);
			//return -1;
		}

		if(out->info()->tensor_shape().total_size()!=NPU_Data.output_attr[i].n_elems){
			std::cerr<<"NPU "<<_name<<" Output size missmatch\n";
			std::cerr<<"NPU "<<_name<<" Expected Output size: "<<out->info()->tensor_shape().total_size()<<" Model Output size: "<<NPU_Data.output_attr[i].n_elems<<std::endl;
			//std::cerr<<out->info()->total_size()<<std::endl;
			//Output_size=NPU_Data.output_attr.n_elems;
		}
		else{
			std::cerr<<"NPU "<<_name<<" Output size match with model: "<<NPU_Data.output_attr[i].n_elems<<std::endl;
			//std::cerr<<out->info()->total_size()<<std::endl; this is 4*out->info()->tensor_shape().total_size()
		}

		output.want_float = true;
		output.index=i;
		if(preallocated_output){
			output.is_prealloc = true;
			if (!out || !out->buffer() || !out->info()) {
				std::cerr << "Null pointer detected!" << std::endl;
				preallocated_output=false;
				//return; // or handle the error accordingly
			}
			else{
				output.buf=reinterpret_cast<void *>(out->buffer()+out->info()->offset_first_element_in_bytes());
				output.size=out->info()->total_size();
			}
		}
		else{
			output.is_prealloc = false;
		}

		NPU_Data.outputs[i]=output;
		i++;
	}*/
	rknn_input_output_num io_num;
	int ret = rknn_query(*(NPU_Data._NPU_Context), RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
	for(int i=0;i<io_num.n_output;i++){
		NPU_Data.output_attr[i].index=i;
		ret = rknn_query(*NPU_Data._NPU_Context, RKNN_QUERY_OUTPUT_ATTR, &(NPU_Data.output_attr[i]),sizeof(rknn_tensor_attr));
		std::cerr<<"\nNPU Model "<<_name<<" Output:"<<i
				<<"\tNum elements:"<<NPU_Data.output_attr[i].n_elems
				<<"\tSize:"<<NPU_Data.output_attr[i].size
				<<"\tName:"<<NPU_Data.output_attr[i].name
				<<"\tFmt:"<<NPU_Data.output_attr[i].fmt
				<<"\tQnt type:"<<NPU_Data.output_attr[i].qnt_type
				<<"\tType:"<<NPU_Data.output_attr[i].type
				<<std::endl;
		for(int j=0;j<outputs.size();j++){
			std::cerr<<"Explore output "<<j<<" n_elements: "<<outputs[j]->info()->tensor_shape().total_size()<<std::endl;
			//if(NPU_Data.outputs[j].index==-1){
			if(NPU_Data.outputs[j].buf==nullptr){
				if (outputs[j]->info()->tensor_shape().total_size()==NPU_Data.output_attr[i].n_elems){
					NPU_Data.outputs[j].index=i;
					NPU_Data.outputs[j].want_float = true;
					NPU_Data.outputs[j].is_prealloc = false;
					//NPU_Data.outputs[j].size=outputs[j]->info()->total_size();
					std::cerr<<"Is match with sub graph output: "<<j<<std::endl;
					if(preallocated_output){
						NPU_Data.outputs[j].is_prealloc = true;
						if (!outputs[j] || !outputs[j]->buffer() || !outputs[j]->info()) {
							std::cerr << "Null pointer detected!" << std::endl;
							preallocated_output=false;
							//return; // or handle the error accordingly
						}
						else{
							NPU_Data.outputs[j].buf=reinterpret_cast<void *>(outputs[j]->buffer()+outputs[j]->info()->offset_first_element_in_bytes());
							NPU_Data.outputs[j].size=outputs[j]->info()->total_size();
						}
					}
					else{
						NPU_Data.outputs[j].is_prealloc = false;
					}

					break;
				}
			}
		}

	}
}



template <>
Status NPU<NPUType>::validate(const ITensorInfo *input, const ITensorInfo *output)
{
	return Status{};
    //return opencl::ClActivation::validate(input, output, act_info);
}

template <>
void NPU<NPUType>::prepare()
{
	std::string t;
	//std::cerr<<"prepare\n";

	return;
}


template <>
void NPU<NPUType>::configure(std::string name, std::vector<arm_compute::ITensor *> _inputs, std::vector<arm_compute::ITensor *> _outputs)
{
	inputs=_inputs;
	outputs=_outputs;
	_name=name.erase(0,4);
	//std::cerr<<_inputs[0]->info()->total_size()<<std::endl;
	ARM_COMPUTE_ERROR_ON_NULLPTR(input);
    //configure(input, output);
	std::cerr<<"NPU "<<_name<<" Number of inputs: "<<inputs.size()<<" and number of outputs: "<<outputs.size()<<std::endl;



	//std::string model_name="/data/data/com.termux/files/home/ARMCL-RockPi/graphs/Google_8_12.rknn";
	//std::string model_name="/data/data/com.termux/files/home/ARMCL-RockPi/graphs/"+name+".rknn";
	std::string model_name="/data/local/ARM-CO-UP/graphs/"+_name+".rknn";
	std::cerr<<"Model name: "<<model_name<<std::endl;
	FILE *fp = fopen(model_name.c_str(), "rb");
	if(fp == NULL) {
		//LOGE("fopen %s fail!\n", mParamPath);
		printf("fopen %s failed!\n", model_name.c_str());
		return ;
	}
	fseek(fp, 0, SEEK_END);
	int model_len = ftell(fp);
	void *model = malloc(model_len);
	fseek(fp, 0, SEEK_SET);
	if(model_len != fread(model, 1, model_len, fp)) {
		//LOGE("fread %s fail!\n", mParamPath);
		printf("fread %s fail!\n", model_name.c_str());
		free(model);
		fclose(fp);
		return ;
	}
	std::cerr<<"NPU "<<model_name<<" Reading model...\n";
	fclose(fp);

	// RKNN_FLAG_ASYNC_MASK: enable async mode to use NPU efficiently.
	//int ret = rknn_init(&ctx, model, model_len, RKNN_FLAG_PRIOR_MEDIUM|RKNN_FLAG_ASYNC_MASK);
	//rknn_context ctx = 0;
	//int ret = rknn_init(NPU_Data._NPU_Context, model, model_len, (RKNN_FLAG_PRIOR_MEDIUM | RKNN_FLAG_COLLECT_PERF_MASK));

	//int ret = rknn_init(NPU_Data._NPU_Context, model, model_len, RKNN_FLAG_PRIOR_MEDIUM);

	//int ret = rknn_init(NPU_Data._NPU_Context, model, model_len, RKNN_FLAG_COLLECT_PERF_MASK);
	//| RKNN_FLAG_COLLECT_PERF_MASK

	bool enable_op_profiling=false;
	//RKNN_FLAG_ASYNC_MASK this flag will cause get_output to not block (think of using it in pipeline mode)
	auto init_flag =  RKNN_FLAG_PRIOR_HIGH |
                   (enable_op_profiling ? RKNN_FLAG_COLLECT_PERF_MASK : 0);
	int ret = rknn_init(NPU_Data._NPU_Context, model, model_len,init_flag);
	if(ret<0){
		std::cerr<<"Error in initializing NPU context...\n";
		return;
	}
	std::cerr<<"NPU "<<model_name<<" Initialized\n";
	//int ret = rknn_init(&ctx, model, model_len, RKNN_FLAG_COLLECT_PERF_MASK);
	free(model);


	/*_rknn_sdk_version version;
	ret = rknn_query(*NPU_Data._NPU_Context, RKNN_QUERY_SDK_VERSION, &version,sizeof(version));
	if(ret < 0) {
		printf("rknn_query fail! ret=%d\n",ret);
	}
	printf("api version:%s \t drive version:%s \n",version.api_version,version.drv_version);*/

	rknn_input_output_num io_num;
	ret = rknn_query(*(NPU_Data._NPU_Context), RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
	printf("model input num: %d, output num: %d\n", io_num.n_input,
	io_num.n_output);
	NPU_Data.inputs.resize(inputs.size());
	NPU_Data.input_attr.resize(inputs.size());
	//memset(NPU_Data.input_attr, 0, sizeof(NPU_Data.input_attr));
	NPU_Data.outputs.resize(outputs.size());
	NPU_Data.output_attr.resize(outputs.size());

	/*for(auto &in :NPU_Data.inputs){
		in.index=-1;
	}
	for(auto &out :NPU_Data.outputs){
		out.index=-1;
	}*/

	std::cerr<<"\nNumber of NPU model outputs: "<<io_num.n_output<<" number of NPU subgraph outputs: "<<outputs.size();
	std::cerr<<"\nNumber of NPU model inputs: "<<io_num.n_input<<" number of NPU subgraph inputs: "<<inputs.size();
	for(int i=0;i<io_num.n_input;i++){
		NPU_Data.input_attr[i].index=i;
		ret = rknn_query(*NPU_Data._NPU_Context, RKNN_QUERY_INPUT_ATTR, &(NPU_Data.input_attr[i]),sizeof(rknn_tensor_attr));
		std::cerr<<"\nNPU Model "<<_name<<" Input:"<<i
				<<"\tNum elements:"<<NPU_Data.input_attr[i].n_elems
				<<"\tSize:"<<NPU_Data.input_attr[i].size
				<<"\tName:"<<NPU_Data.input_attr[i].name
				<<"\tFmt:"<<NPU_Data.input_attr[i].fmt
				<<"\tQnt type:"<<NPU_Data.input_attr[i].qnt_type
				<<"\tType:"<<NPU_Data.input_attr[i].type
				<<std::endl;
		bool match=false;
		for(int j=0;j<inputs.size();j++){
			std::cerr<<"Explre input "<<j<<" n_elements: "<<inputs[j]->info()->tensor_shape().total_size()<<std::endl;
			//if(NPU_Data.inputs[j].index==-1){
			if(NPU_Data.inputs[j].buf==nullptr){
				if (inputs[j]->info()->tensor_shape().total_size()==NPU_Data.input_attr[i].n_elems){
					NPU_Data.inputs[j].index=i;
					NPU_Data.inputs[j].pass_through=Pass;
					NPU_Data.inputs[j].fmt=RKNN_TENSOR_NHWC;
					//Be careful that inputs[j]->info()->total_size() is the size in byte
					//And inputs[j]->info()->tensor_shape().total_size() is the number of elements (the above is 4*n_elements if float)
					NPU_Data.inputs[j].size=inputs[j]->info()->total_size();
					NPU_Data.inputs[j].type=RKNN_TENSOR_FLOAT32;
					std::cerr<<"Is match with sub graph input: "<<j<<std::endl;
					match=true;
					break;
				}
			}
		}
		if(!match){
			std::cerr<<"\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<
					"\nError: Could not find the input match for input "<<i<<" of the NPU model\n"<<
							"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n";
		}

	}
	//fmt==0:RKNN_TENSOR_NCHW    fmt==1:RKNN_TENSOR_NHWC
	for(int i=0;i<io_num.n_output;i++){
		NPU_Data.output_attr[i].index=i;
		ret = rknn_query(*NPU_Data._NPU_Context, RKNN_QUERY_OUTPUT_ATTR, &(NPU_Data.output_attr[i]),sizeof(rknn_tensor_attr));
		std::cerr<<"\nNPU Model "<<_name<<" Output:"<<i
				<<"\tNum elements:"<<NPU_Data.output_attr[i].n_elems
				<<"\tSize:"<<NPU_Data.output_attr[i].size
				<<"\tName:"<<NPU_Data.output_attr[i].name
				<<"\tFmt:"<<NPU_Data.output_attr[i].fmt
				<<"\tQnt type:"<<NPU_Data.output_attr[i].qnt_type
				<<"\tType:"<<NPU_Data.output_attr[i].type
				<<std::endl;
		bool match=false;
		for(int j=0;j<outputs.size();j++){
			std::cerr<<"Explore output "<<j<<" n_elements: "<<outputs[j]->info()->tensor_shape().total_size()<<std::endl;
			//if(NPU_Data.outputs[j].index==-1){
			if(NPU_Data.outputs[j].buf==nullptr){
				if (outputs[j]->info()->tensor_shape().total_size()==NPU_Data.output_attr[i].n_elems){
					NPU_Data.outputs[j].index=i;
					NPU_Data.outputs[j].want_float = true;
					NPU_Data.outputs[j].is_prealloc = false;
					//NPU_Data.outputs[j].size=outputs[j]->info()->total_size();
					std::cerr<<"Is match with sub graph output: "<<j<<std::endl;
					match=true;
					break;
				}
			}
		}
		if(!match){
			std::cerr<<"\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<
					"\nError: Could not find the input match for input "<<i<<" of the NPU model\n"<<
							"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n";
		}

	}





	/*int i=0;
	for(auto in:inputs){
		rknn_input input;
		input.index=0;
		input.pass_through = Pass;
		input.fmt = RKNN_TENSOR_NHWC;
		//if(in->buffer()==nullptr){
		//	std::string sss;
		//	std::cerr<<"hh\n";
		//	std::cin>>sss;
		//}
		//input.buf = static_cast<void*>(in->buffer());
		input.size = in->info()->total_size();
		//std::cerr<<"Input size: "<<in->info()->tensor_shape().total_size()<<std::endl;
		//std::cerr<<in->info()->total_size()<<std::endl;
		input.type = RKNN_TENSOR_FLOAT32;
		//NPU_Data.inputs.emplace_back(input);
		NPU_Data.inputs[i]=input;
		i++;
	}
	std::cerr<<"NPU "<<_name<<" Inputs have been set up.\n";

	i=0;
	for(auto out:_outputs){
		rknn_output output;
		int ret = rknn_query(*NPU_Data._NPU_Context, RKNN_QUERY_OUTPUT_ATTR, &(NPU_Data.output_attr[i]), sizeof(rknn_tensor_attr));
		if(ret < 0) {
			printf("Error: Query Output fail! ret=%d\n",ret);
			//return -1;
		}

		if(out->info()->tensor_shape().total_size()!=NPU_Data.output_attr[i].n_elems){
			std::cerr<<"NPU "<<_name<<" Output size missmatch\n";
			std::cerr<<"NPU "<<_name<<" Expected Output size: "<<out->info()->tensor_shape().total_size()<<" Model Output size: "<<NPU_Data.output_attr[i].n_elems<<std::endl;
			//std::cerr<<out->info()->total_size()<<std::endl;
			//Output_size=NPU_Data.output_attr.n_elems;
		}
		else{
			std::cerr<<"NPU "<<_name<<" Output size match with model: "<<NPU_Data.output_attr[i].n_elems<<std::endl;
			//std::cerr<<out->info()->total_size()<<std::endl; this is 4*out->info()->tensor_shape().total_size()
		}

		output.want_float = true;
		output.index=i;
		output.is_prealloc = false;
		NPU_Data.outputs[i]=output;
		i++;
	}*/
}


template <>
void NPU<NPUType>::run()
{
	//std::cerr<<"Running NPU "<<_name<<"\n";
	auto start=std::chrono::high_resolution_clock::now();
	int i=0;
	for(auto in:inputs){
		const auto   pointer  = reinterpret_cast<double *>(in->buffer() + in->info()->offset_first_element_in_bytes());
		NPU_Data.inputs[i].buf=pointer;
		//auto t=(double*)(NPU_Data.inputs[i].buf);
		//std::cerr<<"input "<<i<<" of NPU, total size: "<<NPU_Data.inputs[i].size<<" first values: "<<t[0]<<", "<<t[1]<<", "<<t[2]<<std::endl;

		const auto   output_net2  = reinterpret_cast<double *>(in->buffer() + in->info()->offset_first_element_in_bytes());
		//std::cerr<<"input "<<i<<" of NPU, total size: "<<NPU_Data.inputs[i].size<<std::endl;
		//std::cerr<<" first values: "<<output_net2[0]<<", "<<output_net2[1]<<", "<<output_net2[2]<<std::endl;
		i++;
	}
	//First set input of the model
	int ret=rknn_inputs_set(*(NPU_Data._NPU_Context), NPU_Data.inputs.size(), &NPU_Data.inputs[0]);
	if(ret < 0){
		printf("Error: Loading inputs fail! ret=%d\n", ret);
		return;
	}
	//std::cerr<<"NPU "<<_name<<" Loads NPU Inputs\n";
	auto end=std::chrono::high_resolution_clock::now();

	ret = rknn_run(*(NPU_Data._NPU_Context), NULL);
	if(ret<0){
		std::cerr<<"Error "<<ret<<" running NPU "<<_name<<" part with id: "<<id<<std::endl;
	}
	//std::cerr<<"NPU "<<_name<<" run model done\n";

	/*i=0;
	for(auto out:outputs){
		if(out->buffer()==nullptr){
			std::cerr<<"haa\n";
			std::string gg;
			std::cin>>gg;
		}
		NPU_Data.outputs[i].buf=static_cast<void*>(out->buffer());
		i++;
	}*/
	ret = rknn_outputs_get(*NPU_Data._NPU_Context, NPU_Data.outputs.size(), &NPU_Data.outputs[0], NULL);
	if(ret < 0) {
		printf("NPU get output fail! ret=%d\n",ret);
		return;
	}

	auto end2=std::chrono::high_resolution_clock::now();
	//std::cerr<<"npu set output done\n";
	i=0;
	if(!preallocated_output){
		for(auto out:outputs){
			auto Output_data=(float*)(NPU_Data.outputs[i].buf);
			//std::cerr<<"output "<<i<<" of NPU with the size: "<<NPU_Data.outputs[i].size<<std::endl;
			if(_Transpose){
				utils::fill_tensor_array2<float,ITensor>(*out,(float*)(Output_data),out->info()->total_size());
			}
			else{
				utils::fill_tensor_array<float,ITensor>(*out,(float*)(Output_data),out->info()->total_size());
			}
			i++;
		}
	}
	auto end3=std::chrono::high_resolution_clock::now();

	/*for(int k=0;k<NPU_Data.outputs.size();k++){
		rknn_outputs_release(*(NPU_Data._NPU_Context), 1, &NPU_Data.outputs[k]);
	}*/
	rknn_outputs_release(*NPU_Data._NPU_Context, NPU_Data.outputs.size(), &NPU_Data.outputs[0]);



	num_run++;
	double t_input=1000*(std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());
	double t_run=1000*(std::chrono::duration_cast<std::chrono::duration<double>>(end2 - end).count());
	double t_output=1000*(std::chrono::duration_cast<std::chrono::duration<double>>(end3 - end2).count());
	input_time+=t_input;
	run_time+=t_run;
	output_time+=t_output;
	//std::cerr<<"Timing of NPU part "<<_name<<" Frame:"<<num_run<<"  input_time: "<<t_input<<", run_time: "<<t_run<<", output_time: "<<t_output<<"\n";
	//rknn_outputs_release(ctx, 1, outputs);
	//consider preallocate approach
	bool enable_op_profiling=false;
	if(enable_op_profiling){
		rknn_perf_detail perf_detail;
		ret = rknn_query(*NPU_Data._NPU_Context, RKNN_QUERY_PERF_DETAIL, &perf_detail,sizeof(rknn_perf_detail));
		if(ret < 0) {
			printf("rknn_query fail! ret=%d\n",ret);
		}
		printf("%s", perf_detail.perf_data);
	}


	_rknn_perf_run run_time;
	ret = rknn_query(*NPU_Data._NPU_Context, RKNN_QUERY_PERF_RUN, &run_time,sizeof(run_time));
	if(ret < 0) {
		printf("error rknn_query fail! ret=%d\n",ret);
	}
	//printf("\nRun_time:%ld\n",run_time.run_duration);
	prof_run_time+=run_time.run_duration;






	//rknn_outputs_release(*NPU_Data._NPU_Context, 1, &NPU_Data.outputs[0]);
	//std::cerr<<"npu done\n";
	//std::string t;
	//std::cin>>t;

}



template <>
int NPU<NPUType>::destroy(){
	int ret=rknn_destroy(*(NPU_Data._NPU_Context));
	return ret;
}
} // namespace arm_compute
