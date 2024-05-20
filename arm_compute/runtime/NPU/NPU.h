/*
 * NPU.h
 *
 *  Created on: Jul 7, 2023
 *      Author: ehsan
 */

#ifndef ARM_COMPUTE_RUNTIME_NPU_NPU_H_
#define ARM_COMPUTE_RUNTIME_NPU_NPU_H_



#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IRuntimeContext.h"
//#include "arm_compute/graph/Types.h"

#include <memory>



#include "rknn_api.h"

namespace arm_compute
{
// Forward declarations
class ITensor;
class ITensorInfo;

/** Basic function to run @ref cpu::kernels::CpuActivationKernel
 *
 * @note The function simulates an activation layer with the specified activation function.
 */

enum class NPUTypes{
	RockPi,
	Khadas,
	temp,
};
const NPUTypes selectedNPU = NPUTypes::RockPi;

template <NPUTypes NPUType>
struct SelectDataStructure {
    using DataType = void;
};

/*********RockPi ***************/
struct RockPiData{
	rknn_context*				_NPU_Context=NULL;
	std::vector<rknn_input>		inputs;
	std::vector<rknn_output> 	outputs;
	std::vector<rknn_tensor_attr> 			input_attr;
	std::vector<rknn_tensor_attr> 			output_attr;
	rknn_tensor_format 			fmt=RKNN_TENSOR_NHWC;
	rknn_tensor_type 			type=RKNN_TENSOR_FLOAT32;
};

template <>
struct SelectDataStructure<NPUTypes::RockPi> {
    using DataType = RockPiData;
};
/***************************************/


/******************** TempNPU ***************/
struct TempData {
    // Define Temp-specific data structure
    // ...
};

template <>
struct SelectDataStructure<NPUTypes::temp> {
    using DataType = TempData;
};
/**********************************************/

class NPUBase : public IFunction {
public:
    virtual ~NPUBase() {}
    virtual void reset_timing() = 0; // if this is a common function
    virtual double get_input_time()=0;
    virtual double get_run_time()=0;
    virtual double get_prof_run_time() =0;
    virtual double get_output_time()=0;
    virtual void set_preallocated_outputs()=0;
    virtual int destroy()=0;
    // ... other common functions ...
};

template <NPUTypes NPUType>
class NPU : public NPUBase
{
public:
    /** Constructor
     *
     * @param[in] ctx Runtime context to be used by the function
     */
	NPU(int _id);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
	NPU(const NPU &) = delete;
    /** Default move constructor */
	NPU(NPU &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
	NPU &operator=(const NPU &) = delete;
    /** Default move assignment operator */
	NPU &operator=(NPU &&);
    /** Destructor */
    ~NPU();
    /** [NEActivationLayer snippet] **/
    /** Set the input and output tensor.
     *
     * @note If the output tensor is a nullptr or is equal to the input, the activation function will be performed in-place
     *
     * @param[in, out] input           Source tensor. In case of @p output tensor = nullptr, this tensor will store the result
     *                                 of the activation function. Data types supported: QASYMM8/QASYMM8_SIGNED/QSYMM16/F16/F32.
     * @param[out]     output          Destination tensor. Data type supported: same as @p input
     * @param[in]      activation_info Activation layer parameters.
     */
    void configure(ITensor *input, ITensor *output);
    //template <typename TargetInfo>
    //void configure(std::vector<typename TargetInfo::TensorType *> inputs, std::vector<typename TargetInfo::TensorType *> inputs);
    void configure(std::string name, std::vector<arm_compute::ITensor *> inputs, std::vector<arm_compute::ITensor *> outputs);

    /** [NEActivationLayer snippet] **/
    /** Static function to check if given info will lead to a valid configuration of @ref NEActivationLayer
     *
     * @param[in] input    Source tensor info. In case of @p output tensor info = nullptr, this tensor will store the result
     *                     of the activation function. Data types supported: QASYMM8/QASYMM8_SIGNED/QSYMM16/F16/F32.
     * @param[in] output   Destination tensor info. Data type supported: same as @p input
     * @param[in] act_info Activation layer information.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden
    void run() override;

    void reset_timing(){
    	//std::cerr<<"NPU reset is calllling\n\n\n\n";
    	input_time=run_time=prof_run_time=output_time=num_run=0;
    }

    double get_input_time() {return input_time;};
    double get_run_time() {return run_time;};
    double get_prof_run_time() {double result = static_cast<float>(prof_run_time) / 1000.0f; return result;};
    double get_output_time() {return output_time;};

    void set_preallocated_outputs() override{};

    void prepare() override;
    int destroy() override{return 0;};
protected:
    int									id;
    std::string							_name;
    std::vector<arm_compute::ITensor *> inputs;
    std::vector<arm_compute::ITensor *> outputs;
    typename SelectDataStructure<NPUType>::DataType NPU_Data;
    bool						_already_loaded;
	std::string			   		_filename;
	bool                  		_bgr;
	unsigned int				Input_size=0;
	bool						Pass=false;
	double						input_time=0;
	double						run_time=0;
	long int					prof_run_time=0;
	double						output_time=0;
	int							num_run=0;
	bool						preallocated_output=false;
	bool						_Transpose=true;


private:
    //struct Impl;
    //std::unique_ptr<Impl> _impl;
};


} // namespace arm_computes








#endif /* ARM_COMPUTE_RUNTIME_NPU_NPU_H_ */
