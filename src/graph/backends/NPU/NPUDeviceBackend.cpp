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
#ifndef My_print
#include "arm_compute/gl_vs.h"
#endif

#include "arm_compute/graph/backends/NPU/NPUDeviceBackend.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/INode.h"
#include "arm_compute/graph/Logger.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/graph/backends/BackendRegistrar.h"
#include "arm_compute/graph/backends/NPU/NPUFunctionFactory.h"
#include "arm_compute/graph/backends/NPU/NPUNodeValidator.h"
#include "arm_compute/graph/backends/NPU/NPUSubTensorHandle.h"
#include "arm_compute/graph/backends/NPU/NPUTensorHandle.h"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/Allocator.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/IWeightsManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"
#include "arm_compute/runtime/OffsetLifetimeManager.h"
#include "arm_compute/runtime/PoolManager.h"
#include "arm_compute/runtime/Scheduler.h"

#include "support/ToolchainSupport.h"

//Ehsan
//#include "arm_compute/gl_vs.h"

namespace arm_compute
{
namespace graph
{
namespace backends
{
/** Register NPU backend */
static detail::BackendRegistrar<NPUDeviceBackend> NPUDeviceBackend_registrar(Target::NPU);

NPUDeviceBackend::NPUDeviceBackend()
    : _allocator()
{
	//std::cerr<<"NPU backend created\n";
	//std::string s;
	//std::cin>>s;
}

void NPUDeviceBackend::initialize_backend()
{
    //Nothing to do
	std::cerr<<"NPU backend initialize...\n";
}

void NPUDeviceBackend::release_backend_context(GraphContext &ctx)
{
    //Nothing to do
    ARM_COMPUTE_UNUSED(ctx);
}

void NPUDeviceBackend::setup_backend_context(GraphContext &ctx)
{
    //std::cerr<<"NPU backend setup \n";
    return;
}

bool NPUDeviceBackend::is_backend_supported()
{
    return true;
}

IAllocator *NPUDeviceBackend::backend_allocator()
{
    return &_allocator;
}

std::unique_ptr<ITensorHandle> NPUDeviceBackend::create_tensor(const Tensor &tensor)
{
    // Get tensor descriptor
    const TensorDescriptor &tensor_desc = tensor.desc();
    ARM_COMPUTE_ERROR_ON(tensor_desc.target != Target::NPU);

    // Create backend tensor handle
    TensorInfo info(tensor_desc.shape, 1, tensor_desc.data_type, tensor_desc.quant_info);
    info.set_data_layout(tensor_desc.layout);
    return std::make_unique<NPUTensorHandle>(info);
}



std::unique_ptr<arm_compute::IFunction> NPUDeviceBackend::configure_node(INode &node, GraphContext &ctx)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Configuring Neon node with ID : " << node.id() << std::endl);
    ARM_COMPUTE_ERROR_ON(node.assigned_target() != Target::NEON);

    // Configure node
    //std::cerr<<"NPU backend configurte node\n";
    auto func = NPUFunctionFactory::create(&node, ctx);
    if(func!=nullptr)
    	func->prepare();
    return std::move(func);
}

arm_compute::Status NPUDeviceBackend::validate_node(INode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Validating Neon node with ID : " << node.id() << std::endl);
    ARM_COMPUTE_ERROR_ON(node.assigned_target() != Target::NPU);

    return NPUNodeValidator::validate(&node);
}

std::shared_ptr<arm_compute::IMemoryManager> NPUDeviceBackend::create_memory_manager(MemoryManagerAffinity affinity)
{
    std::shared_ptr<ILifetimeManager> lifetime_mgr = nullptr;
    if(affinity == MemoryManagerAffinity::Buffer)
    {
        lifetime_mgr = std::make_shared<BlobLifetimeManager>();
    }
    else
    {
        lifetime_mgr = std::make_shared<OffsetLifetimeManager>();
    }
    auto pool_mgr = std::make_shared<PoolManager>();
    auto mm       = std::make_shared<MemoryManagerOnDemand>(lifetime_mgr, pool_mgr);

    return mm;
}

std::shared_ptr<arm_compute::IWeightsManager> NPUDeviceBackend::create_weights_manager()
{
    auto weights_mgr = std::make_shared<IWeightsManager>();
    return weights_mgr;
}
} // namespace backends
} // namespace graph
} // namespace arm_compute
