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
#include "arm_compute/graph/backends/NPU/NPUFunctionFactory.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/Logger.h"
#include "arm_compute/graph/TypePrinter.h"
#include "arm_compute/graph/backends/FunctionHelpers.h"
#include "arm_compute/graph/backends/Utils.h"
#include "arm_compute/graph/nodes/Nodes.h"
#include "arm_compute/runtime/CPP/CPPFunctions.h"
//#include "arm_compute/runtime/NPU/NPUFunctions.h"
#include "arm_compute/runtime/NPU/NPU.h"
#include "support/Cast.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute::utils::cast;

namespace arm_compute
{
namespace graph
{
namespace backends
{
/** Target specific information structure used to pass information to the layer templates */
struct NPUTargetInfo
{
    using TensorType         = arm_compute::ITensor;
    using SrcTensorType      = const arm_compute::ITensor;
    using TensorConcreteType = arm_compute::Tensor;
    static Target TargetType;
};

Target NPUTargetInfo::TargetType = Target::NPU;






std::unique_ptr<IFunction> NPUFunctionFactory::create(INode *node, GraphContext &ctx)
{
    if(node == nullptr)
    {
        return nullptr;
    }

    NodeType type = node->type();
    //std::cerr<<"type of node:"<<node->name()<<" is "<<type<<std::endl;
    switch(type)
    {
        case NodeType::NPU:
            return detail::create_npu_function<NPU<selectedNPU>, NPUTargetInfo>(*polymorphic_downcast<NPUNode *>(node));
        default:
            return nullptr;
    }
}
} // namespace backends
} // namespace graph
} // namespace arm_compute
