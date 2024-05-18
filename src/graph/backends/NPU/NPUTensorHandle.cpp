/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "arm_compute/graph/backends/NPU/NPUTensorHandle.h"

#include "arm_compute/runtime/MemoryGroup.h"
#include "support/Cast.h"

namespace arm_compute
{
namespace graph
{
namespace backends
{
NPUTensorHandle::NPUTensorHandle(const ITensorInfo &info)
    //: _tensor()
{
    //_tensor.allocator()->init(info);
	_tensor=new arm_compute::Tensor();
	_tensor->allocator()->init(info);
}

void NPUTensorHandle::allocate()
{
    //_tensor.allocator()->allocate();
	_tensor->allocator()->allocate();
}

void NPUTensorHandle::free()
{
    //_tensor.allocator()->free();
	_tensor->allocator()->free();
}

void NPUTensorHandle::manage(IMemoryGroup *mg)
{
    if(mg != nullptr)
    {
        //mg->manage(&_tensor);
    	mg->manage(_tensor);
    }
}

void NPUTensorHandle::map(bool blocking)
{
    ARM_COMPUTE_UNUSED(blocking);
}

void NPUTensorHandle::unmap()
{
}

void NPUTensorHandle::release_if_unused()
{
    // TODO (geopin01): Release tensor only if all sub-tensors are marked as not used
    /*if(!_tensor.is_used())
    {
        _tensor.allocator()->free();
    }*/
	if(!_tensor->is_used())
	{
		_tensor->allocator()->free();
	}
}

const arm_compute::ITensor &NPUTensorHandle::tensor() const
{
	//std::cerr<<"const tensor in neon\n";
    //return _tensor;
	return *_tensor;
}

arm_compute::ITensor &NPUTensorHandle::tensor()
{
	//std::cerr<<"tensor in neon\n";
    //return _tensor;
    return *_tensor;
}

ITensorHandle *NPUTensorHandle::parent_handle()
{
    return this;
}

bool NPUTensorHandle::is_subtensor() const
{
    return false;
}

Target NPUTensorHandle::target() const
{
    return Target::NEON;
}

//Ehsan
void NPUTensorHandle::set_tensor(arm_compute::ITensor* _t){
	_tensor=dynamic_cast<arm_compute::Tensor*>(_t);
}
//Ehsan
arm_compute::ITensor *NPUTensorHandle::tensor_ptr(){
	return _tensor;
}

} // namespace backends
} // namespace graph
} // namespace arm_compute
