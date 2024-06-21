#include "arm_compute/runtime/CL/functions/CLSimpleForwardLayer.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Validate.h"

#include "src/common/utils/Log.h"
#include "src/core/CL/ICLKernel.h"

#include "src/gpu/cl/operators/ClSimpleForward.h"

#ifdef MEASURE_TIME
#include <chrono>
#include <fstream>
#endif

namespace arm_compute
{

struct CLSimpleForwardLayer::Impl
{
    const ICLTensor                         *src1{ nullptr };
    const ICLTensor                         *src2{ nullptr };
    const ICLTensor                         *src3{ nullptr };
    ICLTensor                               *dst1{ nullptr };
    ICLTensor                               *dst2{ nullptr };
    ICLTensor                               *dst3{ nullptr };
    std::unique_ptr<opencl::ClSimpleForward> kernel{ nullptr };
};

CLSimpleForwardLayer::CLSimpleForwardLayer()
    : _impl(std::make_unique<Impl>())
{
}
CLSimpleForwardLayer::~CLSimpleForwardLayer() = default;
void CLSimpleForwardLayer::configure(const ICLTensor        *src1,
                                     const ICLTensor        *src2,
                                     const ICLTensor        *src3,
                                     ICLTensor              *dst1,
                                     ICLTensor              *dst2,
                                     ICLTensor              *dst3)
{
    configure(CLKernelLibrary::get().get_compile_context(), src1, src2, src3, dst1, dst2, dst3);
}
void CLSimpleForwardLayer::configure(const CLCompileContext &compile_context,
                                     const ICLTensor        *src1,
                                     const ICLTensor        *src2,
                                     const ICLTensor        *src3,
                                     ICLTensor              *dst1,
                                     ICLTensor              *dst2,
                                     ICLTensor              *dst3)
{
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

    ARM_COMPUTE_ERROR_ON_NULLPTR(src1);
    ARM_COMPUTE_ERROR_ON_NULLPTR(src2);
    ARM_COMPUTE_ERROR_ON_NULLPTR(src3);
    ARM_COMPUTE_ERROR_ON_NULLPTR(dst1);
    ARM_COMPUTE_ERROR_ON_NULLPTR(dst2);
    ARM_COMPUTE_ERROR_ON_NULLPTR(dst3);

    _impl->src1 = src1;
    _impl->src2 = src2;
    _impl->src3 = src3;

    _impl->dst1 = dst1;
    _impl->dst2 = dst2;
    _impl->dst3 = dst3;

    _impl->kernel = std::make_unique<opencl::ClSimpleForward>();
    _impl->kernel->configure(compile_context, src1->info(), src2->info(), src3->info(), dst1->info(), dst2->info(), dst3->info());

#ifdef MEASURE_TIME
    auto          end_time  = std::chrono::high_resolution_clock::now();
    double        cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt", std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "CLSimpleForwardLayer::configure cost: " << cost_time << std::endl;
    measure_out.close();

    std::cout.precision(5);
    std::cout << std::scientific << "CLSimpleForwardLayer::configure cost: " << cost_time << std::endl;
#endif
}

void CLSimpleForwardLayer::run()
{
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src1);
    pack.add_tensor(TensorType::ACL_DST_0, _impl->dst1);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->src2);
    pack.add_tensor(TensorType::ACL_DST_1, _impl->dst2);
    pack.add_tensor(TensorType::ACL_SRC_2, _impl->src3);
    pack.add_tensor(TensorType::ACL_DST_2, _impl->dst3);

    _impl->kernel->run(pack);

#ifdef MEASURE_TIME
    auto          end_time  = std::chrono::high_resolution_clock::now();
    double        cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt", std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "CLSimpleForwardLayer::run cost: " << cost_time << std::endl;
    measure_out.close();

    std::cout.precision(5);
    std::cout << std::scientific << "CLSimpleForwardLayer::run cost: " << cost_time << std::endl;
#endif
}

} // namespace arm_compute