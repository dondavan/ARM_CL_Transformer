#include "src/gpu/cl/operators/ClLinear.h"

#include "src/common/utils/Log.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "arm_compute/core/KernelDescriptors.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "src/gpu/cl/utils/ClAuxTensorHandler.h"

#include "src/gpu/cl/kernels/ClLinearKernel.h"
#include "src/runtime/heuristics/matmul_native/ClMatMulNativeKernelConfig.h"
#include "src/runtime/heuristics/matmul_native/IClMatMulNativeKernelConfig.h"
#include "src/runtime/CL/gemm_auto_heuristics/CLGEMMAutoHeuristics.h"


#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Log.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/ITensorAllocator.h"

#include "src/common/utils/Log.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/core/utils/helpers/float_ops.h"
#include "src/gpu/cl/IClKernel.h"
#include "src/gpu/cl/utils/ClAuxTensorHandler.h"
#include "src/runtime/CL/gemm/CLGEMMKernelSelection.h"
#include "src/runtime/CL/gemm_auto_heuristics/CLGEMMAutoHeuristics.h"
#include "support/Cast.h"
#include "utils/TypePrinter.h"


namespace arm_compute
{
namespace opencl
{

using namespace arm_compute::misc::shape_calculator;
using namespace arm_compute::cl_gemm;
using namespace arm_compute::experimental;
using namespace arm_compute::utils::cast;
using namespace arm_compute::opencl::kernels;
void ClLinear::configure(const ClCompileContext &compile_context,
                         ITensorInfo      *a,
                         ITensorInfo      *b,
                         ITensorInfo      *c,
                         ITensorInfo            *d,
                         float                   alpha,
                         float beta, const LinearLayerInfo &linear_info)
{
    ARM_COMPUTE_LOG_PARAMS(a, b, c, d, alpha, beta, linear_info);
    ARM_COMPUTE_UNUSED(a, b, c, d, alpha);
    ARM_COMPUTE_UNUSED(linear_info);
    ARM_COMPUTE_UNUSED(beta);



    auto kernl = std::make_unique<kernels::ClGemmMatrixMultiplyNativeKernel>();

    DataType           data_type               = a->data_type();
    bool               reinterpret_input_as_3d = false;
    const unsigned int m          = reinterpret_input_as_3d ? (a->dimension(1) * a->dimension(2)) : a->dimension(1);
    const unsigned int n          = b->dimension(0);
    const unsigned int k          = a->dimension(0);
    const unsigned int batch_size = reinterpret_input_as_3d ? a->dimension(3) : a->dimension(2);
    const int          depth_output_gemm3d = 1;
    const GPUTarget    gpu_target          = CLScheduler::get().target();
    bool               broadcast_bias      = false;

    GEMMKernelInfo kernel_info;
    kernel_info.m                       = m;
    kernel_info.n                       = n;
    kernel_info.k                       = k;
    kernel_info.depth_output_gemm3d     = depth_output_gemm3d;
    kernel_info.reinterpret_input_as_3d = reinterpret_input_as_3d;
    kernel_info.broadcast_bias          = broadcast_bias;

    // Set the target for the kernels
    kernl->set_target(gpu_target);

    auto config = auto_heuristics::select_mlgo_gemm_config_reshaped_only_rhs(
        auto_heuristics::CommonQuery{gpu_target, data_type, m, n, k, batch_size});

    // Configure and tune matrix multiply kernel
    kernl->configure(compile_context, a, b, c, d, alpha, beta, config.lhs_info, config.rhs_info,
                                 kernel_info);

    _kernel = std::move(kernl);
}

Status
ClLinear::validate(const ITensorInfo *a,
                   const ITensorInfo *b,
                   const ITensorInfo *c,
                   ITensorInfo       *d,
                   float              alpha,
                   float beta, const LinearLayerInfo &linear_info)
{
    ARM_COMPUTE_UNUSED(a);
    ARM_COMPUTE_UNUSED(b);
    ARM_COMPUTE_UNUSED(c);
    ARM_COMPUTE_UNUSED(d);
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_UNUSED(beta);
    ARM_COMPUTE_UNUSED(linear_info);
    return Status{};
}

void ClLinear::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
    auto a = tensors.get_const_tensor(ACL_SRC_0);
    auto b = tensors.get_const_tensor(ACL_SRC_1);
    auto c = tensors.get_const_tensor(ACL_SRC_2);
    auto d = tensors.get_tensor(ACL_DST);
    ARM_COMPUTE_UNUSED(a, b, c, d);
}

} // namespace opencl
} // namespace arm_compute
