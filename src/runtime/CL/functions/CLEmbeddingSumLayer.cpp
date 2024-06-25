#include "arm_compute/runtime/CL/functions/CLEmbeddingSumLayer.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Validate.h"

#include "src/core/CL/ICLKernel.h"
#include "src/gpu/cl/operators/ClEmbedSum.h"
#include "src/gpu/cl/operators/ClAdd.h"

#ifdef MEASURE_TIME
#include <chrono>
#include <fstream>
#endif

namespace arm_compute
{

struct CLEmbeddingSumLayer::Impl
{
    const ICLTensor                    *token{ nullptr };
    const ICLTensor                    *segment{ nullptr };
    const ICLTensor                    *position{ nullptr };
    ICLTensor                          *dst{ nullptr };
    IRuntimeContext                    *ctx{ nullptr };
    std::unique_ptr<opencl::ClAdd> op_1{ nullptr };
    std::unique_ptr<opencl::ClAdd> op_2{ nullptr };
};

CLEmbeddingSumLayer::CLEmbeddingSumLayer()
    : _impl(std::make_unique<Impl>())
{
}

CLEmbeddingSumLayer::~CLEmbeddingSumLayer() = default;

void CLEmbeddingSumLayer::configure(ICLTensor                *token,
                                    ICLTensor                *segment,
                                    ICLTensor                *position,
                                    ICLTensor                *output,
                                    const EmbeddingLayerInfo &emb_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), token, segment, position, output, emb_info);
}

void CLEmbeddingSumLayer::configure(const CLCompileContext   &compile_context,
                                    ICLTensor                *token,
                                    ICLTensor                *segment,
                                    ICLTensor                *position,
                                    ICLTensor                *output,
                                    const EmbeddingLayerInfo &emb_info)
{
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

    _impl->token    = token;
    _impl->segment  = segment;
    _impl->position = position;
    _impl->dst      = output;

    std::cout << "src/runtime/CL/functions/CLEmbeddingSumLayer.cpp configure start" << std::endl;
    
    _impl->op_1 = std::make_unique<opencl::ClAdd>();
    _impl->op_1->configure(compile_context,
                         token->info(),
                         segment->info(),
                         output->info(),
                         emb_info.c_policy());
    _impl->op_2 = std::make_unique<opencl::ClAdd>();
    _impl->op_2->configure(compile_context,
                         output->info(),
                         position->info(),
                         output->info(),
                         emb_info.c_policy());  
    std::cout << "src/runtime/CL/functions/CLEmbeddingSumLayer.cpp end" << std::endl;

#ifdef MEASURE_TIME
    auto          end_time  = std::chrono::high_resolution_clock::now();
    double        cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt", std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "CLEmbeddingSumLayer::configure cost: " << cost_time << std::endl;
    measure_out.close();

    std::cout.precision(5);
    std::cout << std::scientific << "CLEmbeddingSumLayer::configure cost: " << cost_time << std::endl;
#endif
}

void CLEmbeddingSumLayer::prepare()
{
}

void CLEmbeddingSumLayer::run()
{
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

    std::cout << "src/runtime/CL/functions/CLEmbeddingSumLayer.cpp run start" << std::endl;

    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->token);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->segment);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);

    _impl->op_1->run(pack);

    pack.add_tensor(TensorType::ACL_SRC_0, _impl->dst);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->position);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);

    _impl->op_2->run(pack);

    std::cout << "src/runtime/CL/functions/CLEmbeddingSumLayer.cpp run end" << std::endl;
#ifdef MEASURE_TIME
    auto          end_time  = std::chrono::high_resolution_clock::now();
    double        cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt", std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "CLEmbeddingSumLayer::run cost: " << cost_time << std::endl;
    measure_out.close();

    std::cout.precision(5);
    std::cout << std::scientific << "CLEmbeddingSumLayer::run cost: " << cost_time << std::endl;
#endif
}

} // namespace arm_compute