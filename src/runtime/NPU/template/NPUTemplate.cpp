/*
 * NPUTemplate.cpp
 *
 *  Created on: Jul 7, 2023
 *      Author: ehsan
 */

#include "arm_compute/runtime/NPU/NPU.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

namespace arm_compute
{

const NPUTypes NPUType = NPUTypes::temp;
template <>
NPU<NPUType>::NPU(int _id)
{
	id=_id;
	std::cerr<<"creating a template NPU node...\n";
}

template <>
NPU<NPUType>::NPU(NPU<NPUType> &&) = default;

template <>
NPU<NPUType> &NPU<NPUType>::operator=(NPU<NPUType> &&) = default;

template <>
NPU<NPUType>::~NPU()                               = default;

template <>
void NPU<NPUType>::configure(ITensor *input, ITensor *output)
{
}

template <>
Status NPU<NPUType>::validate(const ITensorInfo *input, const ITensorInfo *output)
{
	return Status{};
}

template <>
void NPU<NPUType>::prepare()
{
	std::cerr<<"preparing npu\n";

}

template <>
void NPU<NPUType>::run()
{
	std::cerr<<"running npu\n";

}
} // namespace arm_compute
