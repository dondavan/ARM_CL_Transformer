#include "arm_compute/runtime/NEON/functions/NESimpleForwardLayer.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/Tensor.h"

#include "src/common/utils/Log.h"
#include "src/cpu/operators/CpuSimpleForward.h"

namespace arm_compute
{

struct  NESimpleForwardLayer::Impl
{
    const ITensor *src1{nullptr};
    const ITensor *src2{nullptr};
    const ITensor *src3{nullptr};
    ITensor *dst1{nullptr};
    ITensor *dst2{nullptr};
    ITensor *dst3{nullptr};
    std::unique_ptr<cpu::CpuSimpleForward>     kernel{nullptr};
};

NESimpleForwardLayer::NESimpleForwardLayer() : _impl(std::make_unique<Impl>())
{
}
NESimpleForwardLayer::~NESimpleForwardLayer() = default;

void NESimpleForwardLayer::configure(const ITensor *src1,
                                     const ITensor *src2,
                                     const ITensor *src3,
                                     ITensor *dst1,
                                     ITensor *dst2,
                                     ITensor *dst3)
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

    _impl->kernel = std::make_unique<cpu::CpuSimpleForward>();
    _impl->kernel->configure(src1->info(),src2->info(),src3->info(),dst1->info(),dst2->info(),dst3->info());

#ifdef MEASURE_TIME
    auto   end_time  = std::chrono::high_resolution_clock::now();
    double cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt",std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "NESimpleForwardLayer::configure cost: " << cost_time << std::endl;
    measure_out.close();

    std::cout.precision(5);
    std::cout << std::scientific << "NESimpleForwardLayer::configure cost: " << cost_time << std::endl;
#endif
}

void NESimpleForwardLayer::run()
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
    auto   end_time  = std::chrono::high_resolution_clock::now();
    double cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt",std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "NESimpleForwardLayer::run cost: " << cost_time << std::endl;
    measure_out.close();

    std::cout.precision(5);
    std::cout << std::scientific << "NESimpleForwardLayer::run cost: " << cost_time << std::endl;
#endif
}

} // namespace arm_compute
