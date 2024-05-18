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
#include "arm_compute/graph/backends/NPU/NPUNodeValidator.h"

#include "arm_compute/graph/backends/ValidateHelpers.h"
#include "arm_compute/graph/nodes/Nodes.h"

#include "arm_compute/runtime/CPP/CPPFunctions.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "src/core/NEON/kernels/NEConvertFullyConnectedWeightsKernel.h"
#include "src/core/NEON/kernels/NEConvertQuantizedSignednessKernel.h"
#include "src/core/NEON/kernels/NEGEMMInterleave4x4Kernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpMatrixMultiplyKernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpOffsetContributionKernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpOffsetContributionOutputStageKernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpReductionKernel.h"
#include "src/core/NEON/kernels/NEGEMMMatrixAdditionKernel.h"
#include "src/core/NEON/kernels/NEGEMMMatrixMultiplyKernel.h"
#include "src/core/NEON/kernels/NEGEMMTranspose1xWKernel.h"
#include "src/core/NEON/kernels/NEQLSTMLayerNormalizationKernel.h"
#include "src/core/NEON/kernels/NEWeightsReshapeKernel.h"
#include "support/Cast.h"

using namespace arm_compute::utils::cast;

namespace arm_compute
{
namespace graph
{
namespace backends
{

Status NPUNodeValidator::validate(INode *node)
{
	return Status{};
	/*
    if(node == nullptr)
    {
        return Status{};
    }

    NodeType type = node->type();
    switch(type)
    {
        case NodeType::NPU:
            return detail::validate_arg_min_max_layer<NEArgMinMaxLayer>(*polymorphic_downcast<ArgMinMaxLayerNode *>(node));
        case NodeType::BoundingBoxTransformLayer:
            return ARM_COMPUTE_CREATE_ERROR(arm_compute::ErrorCode::RUNTIME_ERROR, "Unsupported operation : BoundingBoxTransformLayer");
        default:
            return Status{};
    }
    */
}
} // namespace backends
} // namespace graph
} // namespace arm_compute
