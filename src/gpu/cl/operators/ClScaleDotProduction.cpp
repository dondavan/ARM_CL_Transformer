#include "src/gpu/cl/operators/ClScaleDotProduction.h"

#include "arm_compute/runtime/CL/CLScheduler.h"

#include "src/common/IOperator.h"
#include "src/common/utils/LegacySupport.h"
#include "src/common/utils/Log.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "src/gpu/cl/utils/ClAuxTensorHandler.h"

#include "src/gpu/cl/kernels/ClVectorizeKernel.h"

#include "src/runtime/heuristics/matmul_native/ClMatMulNativeKernelConfig.h"
#include "src/runtime/heuristics/matmul_native/IClMatMulNativeKernelConfig.h"

namespace arm_compute
{
namespace opencl
{

void ClScaleDotProduction::configure(const ClCompileContext                     &compile_context,
                                     const ITensorInfo                          *query,
                                     const ITensorInfo                          *key,
                                     const ITensorInfo                          *value,
                                     ITensorInfo                                *output,
                                     const ScaleDotProductionAttentionLayerInfo &info)
{
    ARM_COMPUTE_LOG_PARAMS(key, value, query, output);
    ARM_COMPUTE_UNUSED(compile_context, query, key, value, output, info);

    // Query multi-Head reshape
    TensorShape query_reshape = TensorShape(query->tensor_shape().x() / info.h(),
                                            info.h(),
                                            query->tensor_shape().y(),
                                            1);
    _reshaped_query           = query->clone()->set_tensor_shape(query_reshape);
    TensorShape query_permute = TensorShape(query->tensor_shape().x() / info.h(),
                                            query->tensor_shape().y(),
                                            info.h(),
                                            1);
    _permuted_query           = query->clone()->set_tensor_shape(query_permute);

    auto query_reshape_kernel = std::make_unique<kernels::ClReshapeKernel>();
    query_reshape_kernel->configure(compile_context, query, &_reshaped_query);
    _query_reshape_kernel = std::move(query_reshape_kernel);

    auto query_permute_kernel = std::make_unique<kernels::ClPermuteKernel>();
    query_permute_kernel->configure(compile_context, &_reshaped_query, &_permuted_query, PermutationVector(0U, 2U, 1U));
    _query_permute_kernel = std::move(query_permute_kernel);

    // Key multi-Head reshape
    TensorShape key_reshape = TensorShape(key->tensor_shape().x() / info.h(),
                                          info.h(),
                                          key->tensor_shape().y(),
                                          1);
    _reshaped_key           = key->clone()->set_tensor_shape(key_reshape);
    TensorShape key_permute = TensorShape(key->tensor_shape().x() / info.h(),
                                          key->tensor_shape().y(),
                                          info.h(),
                                          1);
    _permuted_key           = key->clone()->set_tensor_shape(key_permute);

    auto key_reshape_kernel = std::make_unique<kernels::ClReshapeKernel>();
    key_reshape_kernel->configure(compile_context, key, &_reshaped_key);
    _key_reshape_kernel = std::move(key_reshape_kernel);

    auto key_permute_kernel = std::make_unique<kernels::ClPermuteKernel>();
    key_permute_kernel->configure(compile_context, &_reshaped_key, &_permuted_key, PermutationVector(0U, 2U, 1U));
    _key_permute_kernel = std::move(key_permute_kernel);

    // Value multi-Head reshape
    TensorShape value_reshape = TensorShape(value->tensor_shape().x() / info.h(),
                                            info.h(),
                                            value->tensor_shape().y(),
                                            1);
    _reshaped_value           = value->clone()->set_tensor_shape(value_reshape);
    TensorShape value_permute = TensorShape(value->tensor_shape().x() / info.h(),
                                            value->tensor_shape().y(),
                                            info.h(),
                                            1);
    _permuted_value           = value->clone()->set_tensor_shape(value_permute);

    auto value_reshape_kernel = std::make_unique<kernels::ClReshapeKernel>();
    value_reshape_kernel->configure(compile_context, value, &_reshaped_value);
    _value_reshape_kernel = std::move(value_reshape_kernel);

    auto value_permute_kernel = std::make_unique<kernels::ClPermuteKernel>();
    value_permute_kernel->configure(compile_context, &_reshaped_value, &_permuted_value, PermutationVector(0U, 2U, 1U));
    _value_permute_kernel = std::move(value_permute_kernel);

    // Pretranspose Key
    auto key_transpose_kernel = std::make_unique<kernels::ClTransposeKernel>();
    key_transpose_kernel->configure(compile_context, &_permuted_key, &_transposed_key);
    _key_transpose_kernel = std::move(key_transpose_kernel);

    // Specify whether transpose weights is necessary in matmul info
    const MatMulInfo mat_info_qk = MatMulInfo();

    // Note: MatMul does not need offset negation unlike gemm
    // 1. Change shape when calling matmul to fit batch expectations.
    //_lhs_to_use = src->clone()->set_tensor_shape(get_reshaped_matmul_tensor(_lhs_to_use.tensor_shape()));

    // 2. Use heuristics to get kernel info object
    const GPUTarget                                         gpu_target = CLScheduler::get().target();
    std::unique_ptr<cl_matmul::IClMatMulNativeKernelConfig> kernel_config_qk =
        cl_matmul::ClMatMulNativeKernelConfigurationFactory::create(gpu_target);
    MatMulKernelInfo mm_kernel_info_qk = kernel_config_qk->configure(&_permuted_query, &_transposed_key, mat_info_qk);

    // Matrix multiply compute multi-head attention between Query and Key
    auto        product_mm_kernel = std::make_unique<kernels::ClLinearKernel>();
    const float scale             = 1.0f / sqrt(info.d_model() / info.h());
    product_mm_kernel->set_target(gpu_target);
    product_mm_kernel->configure(compile_context, &_permuted_query, &_transposed_key, nullptr, &_scaled_query_key, scale, 1, mm_kernel_info_qk);
    _product_mm_kernel = std::move(product_mm_kernel);

    //  Softmax of previous product
    SoftmaxKernelInfo softmax_info{1.0f, false, query->data_type(), 0};
    auto softmax_kernel = std::make_unique<kernels::ClSoftmaxKernel>();
    softmax_kernel->configure(compile_context,_scaled_query_key, _softmaxed_product, softmax_info);
    _softmax_kernel = std::move(softmax_kernel);


    // Specify whether transpose weights is necessary in matmul info
    const MatMulInfo mat_info_pv = MatMulInfo();

    // Note: MatMul does not need offset negation unlike gemm
    // 1. Change shape when calling matmul to fit batch expectations.
    //_lhs_to_use = src->clone()->set_tensor_shape(get_reshaped_matmul_tensor(_lhs_to_use.tensor_shape()));

    // 2. Use heuristics to get kernel info object
    std::unique_ptr<cl_matmul::IClMatMulNativeKernelConfig> kernel_config_pv =
        cl_matmul::ClMatMulNativeKernelConfigurationFactory::create(gpu_target);
    MatMulKernelInfo mm_kernel_info_pv = kernel_config_pv->configure(&_softmaxed_product, &_permuted_value, mat_info_pv);

    //  Multiply between scaled product and value
    auto context_mm_kernel = std::make_unique<kernels::ClLinearKernel>();
    context_mm_kernel->set_target(gpu_target);
    context_mm_kernel->configure(compile_context, &_softmaxed_product, &_permuted_value, nullptr, &_gemmed_context, 1.0f, 1, mm_kernel_info_pv);
    _context_mm_kernel = std::move(context_mm_kernel);


    // Concat multi-Head reshape
    TensorShape concat_permute = TensorShape(query->tensor_shape().x() / info.h(),
                                             info.h(),
                                             query->tensor_shape().y(),
                                             1);
    _permuted_concat           = query->clone()->set_tensor_shape(concat_permute);
    
    auto concat_permute_kernel       = std::make_unique<kernels::ClPermuteKernel>();
    concat_permute_kernel->configure(compile_context, &_gemmed_context, &_permuted_concat, PermutationVector(0U, 2U, 1U));
    _concat_permute_kernel = std::move(concat_permute_kernel);

    auto concat_reshape_kernel = std::make_unique<kernels::ClReshapeKernel>();
    concat_reshape_kernel->configure(compile_context, &_permuted_concat, output);
    _concat_reshape_kernel = std::move(concat_reshape_kernel);

    //auto product_mm_kernel = std::make_unique<kernels::ClMatMulNativeKernel>();
    //const int   m      = _permuted_query.dimension(1);
    //const int   n      = _transposed_key.dimension(0);
    //const int   k      = _permuted_query.dimension(0);
    //const float scale  = 1.0f / sqrt(info.d_model() / info.h());
    //product_mm_kernel->configure(compile_context, &_permuted_query, &_permuted_key,nullptr, output, mm_kernel_info);
    //_product_mm_kernel = std::move(product_mm_kernel);

    /*
    _query_permute_func = std::make_unique<CpuPermute>();
    _query_permute_func->configure(&_reshaped_query, &_permuted_query, PermutationVector(0U, 2U, 1U));
    
    // Query multi-Head reshape
    TensorShape query_reshape = TensorShape(query->tensor_shape().x() / info.h(),
                                            info.h(),
                                            query->tensor_shape().y(),
                                            1);
    _reshaped_query           = query->clone()->set_tensor_shape(query_reshape);
    TensorShape query_permute = TensorShape(query->tensor_shape().x() / info.h(),
                                            query->tensor_shape().y(),
                                            info.h(),
                                            1);
    _permuted_query           = query->clone()->set_tensor_shape(query_permute);
    _query_reshape_kernel     = std::make_unique<kernels::CpuReshapeKernel>();
    _query_reshape_kernel->configure(query, &_reshaped_query);
    _query_permute_func = std::make_unique<CpuPermute>();
    _query_permute_func->configure(&_reshaped_query, &_permuted_query, PermutationVector(0U, 2U, 1U));

    // Key multi-Head reshape
    TensorShape key_reshape = TensorShape(key->tensor_shape().x() / info.h(),
                                          info.h(),
                                          key->tensor_shape().y(),
                                          1);
    _reshaped_key           = key->clone()->set_tensor_shape(key_reshape);
    TensorShape key_permute = TensorShape(key->tensor_shape().x() / info.h(),
                                          key->tensor_shape().y(),
                                          info.h(),
                                          1);
    _permuted_key           = key->clone()->set_tensor_shape(key_permute);
    _key_reshape_kernel     = std::make_unique<kernels::CpuReshapeKernel>();
    _key_reshape_kernel->configure(key, &_reshaped_key);
    _key_permute_func = std::make_unique<CpuPermute>();
    _key_permute_func->configure(&_reshaped_key, &_permuted_key, PermutationVector(0U, 2U, 1U));
    // Pretranspose Key, K=K^T
    _key_transpose_func = std::make_unique<CpuTranspose>();
    _key_transpose_func->configure(&_permuted_key, &_transposed_key);

    // Value multi-Head reshape
    TensorShape value_reshape = TensorShape(value->tensor_shape().x() / info.h(),
                                            info.h(),
                                            value->tensor_shape().y(),
                                            1);
    _reshaped_value           = value->clone()->set_tensor_shape(value_reshape);
    TensorShape value_permute = TensorShape(value->tensor_shape().x() / info.h(),
                                            value->tensor_shape().y(),
                                            info.h(),
                                            1);
    _permuted_value           = value->clone()->set_tensor_shape(value_permute);
    _value_reshape_kernel     = std::make_unique<kernels::CpuReshapeKernel>();
    _value_reshape_kernel->configure(value, &_reshaped_value);
    _value_permute_func = std::make_unique<CpuPermute>();
    _value_permute_func->configure(&_reshaped_value, &_permuted_value, PermutationVector(0U, 2U, 1U));

    // Configure interleave kernel
    _query_interleave_kernel = std::make_unique<cpu::kernels::CpuGemmInterleave4x4Kernel>();
    _query_interleave_kernel->configure(&_permuted_query, &_tmp_query);
    _aux_mem[InterleavedLHS] =
        experimental::MemoryInfo(offset_int_vec(InterleavedLHS), experimental::MemoryLifetime::Persistent, _tmp_query.total_size());

    // Configure rhs transpose1xw kernel
    _key_transpose1xW_kernel = std::make_unique<cpu::kernels::CpuGemmTranspose1xWKernel>();
    _key_transpose1xW_kernel->configure(&_transposed_key, &_tmp_key);
    _aux_mem[Transposed1xWRHS] =
        experimental::MemoryInfo(offset_int_vec(Transposed1xWRHS), experimental::MemoryLifetime::Persistent, _tmp_key.total_size());

    // Matrix multiply compute multi-head attention between Query and Key
    _product_mm_kernel = std::make_unique<cpu::kernels::CpuGemmMatrixMultiplyKernel>();
    const int   m      = _permuted_query.dimension(1);
    const int   n      = _transposed_key.dimension(0);
    const int   k      = _permuted_query.dimension(0);
    const float scale  = 1.0f / sqrt(info.d_model() / info.h());
    _product_mm_kernel->configure(&_tmp_query, &_tmp_key, &_scaled_query_key, scale, true, GEMMReshapeInfo(m, n, k));

    //  Softmax of previous product
    _softmax_func = std::make_unique<cpu::CpuSoftmaxGeneric>();
    _softmax_func->configure(&_scaled_query_key, &_softmaxed_product);

    // Configure interleave kernel
    _product_interleave_kernel = std::make_unique<cpu::kernels::CpuGemmInterleave4x4Kernel>();
    _product_interleave_kernel->configure(&_softmaxed_product, &_interleaved_product);
    _aux_mem[InterleavedProduct] =
        experimental::MemoryInfo(offset_int_vec(InterleavedProduct), experimental::MemoryLifetime::Persistent, _interleaved_product.total_size());

    // Configure rhs transpose1xw kernel
    _value_transpose1xW_kernel = std::make_unique<cpu::kernels::CpuGemmTranspose1xWKernel>();
    _value_transpose1xW_kernel->configure(&_permuted_value, &_transposed1xW_value);
    _aux_mem[Transposed1xWValue] =
        experimental::MemoryInfo(offset_int_vec(Transposed1xWValue), experimental::MemoryLifetime::Persistent, _transposed1xW_value.total_size());

    //  Multiply between scaled product and value
    _context_mm_kernel = std::make_unique<cpu::kernels::CpuGemmMatrixMultiplyKernel>();
    const int m1       = _softmaxed_product.dimension(1);
    const int n1       = _permuted_value.dimension(0);
    const int k1       = _softmaxed_product.dimension(0);
    _context_mm_kernel->configure(&_interleaved_product, &_permuted_value, &_gemmed_context, 1.0f, true, GEMMReshapeInfo(m1, n1, k1));

    // Concat multi-Head reshape

    TensorShape concat_permute = TensorShape(query->tensor_shape().x() / info.h(),
                                             info.h(),
                                             query->tensor_shape().y(),
                                             1);
    _permuted_concat           = query->clone()->set_tensor_shape(concat_permute);
    _concat_permute_func       = std::make_unique<CpuPermute>();
    _concat_permute_func->configure(&_gemmed_context, &_permuted_concat, PermutationVector(0U, 2U, 1U));

    _concat_reshape_kernel = std::make_unique<kernels::CpuReshapeKernel>();
    _concat_reshape_kernel->configure(&_permuted_concat, output);
    */
}

Status
ClScaleDotProduction::validate(const ITensorInfo *query, const ITensorInfo *key, const ITensorInfo *value, ITensorInfo *output)
{
    ARM_COMPUTE_UNUSED(key);
    ARM_COMPUTE_UNUSED(value);
    ARM_COMPUTE_UNUSED(query);
    ARM_COMPUTE_UNUSED(output);
    return Status{};
}

void ClScaleDotProduction::run(ITensorPack &tensors)
{
    ARM_COMPUTE_UNUSED(tensors);

    auto query = tensors.get_const_tensor(ACL_SRC_0);
    auto key   = tensors.get_const_tensor(ACL_SRC_1);
    auto value  = tensors.get_const_tensor(ACL_SRC_2);
    auto output = tensors.get_tensor(ACL_DST);

    CLAuxTensorHandler reshaped_query(offset_int_vec(QueryReshape), _reshaped_query, tensors);
    CLAuxTensorHandler permuted_query(offset_int_vec(QueryPermute), _permuted_query, tensors);
    CLAuxTensorHandler reshaped_key(offset_int_vec(KeyReshape), _reshaped_key, tensors);
    CLAuxTensorHandler permuted_key(offset_int_vec(KeyPermute), _permuted_key, tensors);
    CLAuxTensorHandler reshaped_value(offset_int_vec(ValueReshape), _reshaped_value, tensors);
    CLAuxTensorHandler permuted_value(offset_int_vec(ValuePermute), _permuted_value, tensors);

    CLAuxTensorHandler transposed_key(offset_int_vec(KeyTranspose), _transposed_key, tensors);

    CLAuxTensorHandler scaled_query_key(offset_int_vec(QueryKeyScale), _scaled_query_key, tensors);
    CLAuxTensorHandler softmaxed_product(offset_int_vec(Softmax), _softmaxed_product, tensors);
    CLAuxTensorHandler gemmed_context(offset_int_vec(GemmedContext), _gemmed_context, tensors);

    CLAuxTensorHandler permuted_concat(offset_int_vec(ConcatPermute), _permuted_concat, tensors);

    // Run Query multi-Head reshape
    ITensorPack query_reshape_pack{ { ACL_SRC_0, query }, { ACL_DST, reshaped_query.get() } };
    CLScheduler::get().enqueue_op(*_query_reshape_kernel, query_reshape_pack, true);
    ITensorPack query_permute_pack{ { ACL_SRC, reshaped_query.get() }, { ACL_DST, permuted_query.get() } };
    CLScheduler::get().enqueue_op(*_query_permute_kernel, query_permute_pack, true);

    // Run Key multi-Head reshape
    ITensorPack key_reshape_pack{ { ACL_SRC_0, key }, { ACL_DST, reshaped_key.get() } };
    CLScheduler::get().enqueue_op(*_key_reshape_kernel, key_reshape_pack, true);
    ITensorPack key_permute_pack{ { ACL_SRC, reshaped_key.get() }, { ACL_DST, permuted_key.get() } };
    CLScheduler::get().enqueue_op(*_key_permute_kernel, key_permute_pack, true);
    
    // Run Value multi-Head reshape
    ITensorPack value_reshape_pack{ { ACL_SRC_0, value }, { ACL_DST, reshaped_value.get() } };
    CLScheduler::get().enqueue_op(*_value_reshape_kernel, value_reshape_pack, true);
    ITensorPack value_permute_pack{ { ACL_SRC, reshaped_value.get() }, { ACL_DST, permuted_value.get() } };
    CLScheduler::get().enqueue_op(*_value_permute_kernel, value_permute_pack, true);

    // Run Key pre-transpose
    ITensorPack key_transpose_pack{ { ACL_SRC, permuted_key.get() }, { ACL_DST, transposed_key.get() } };
    CLScheduler::get().enqueue_op(*_key_transpose_kernel, key_transpose_pack, true);

    // Run matrix multiply compute multi-head attention between Query and Key
    ITensorPack gemm_QK_pack{ { ACL_SRC_0, permuted_query.get() }, { ACL_SRC_1, transposed_key.get() }, { ACL_DST, scaled_query_key.get() } };
    CLScheduler::get().enqueue_op(*_product_mm_kernel, gemm_QK_pack, true);

    // Softmax scaled product
    ITensorPack softmax_pack = { { ACL_SRC, scaled_query_key.get() }, { ACL_DST, softmaxed_product.get() } };
    CLScheduler::get().enqueue_op(*_softmax_kernel, softmax_pack, true);

    // Run matrix multiply compute multi-head attention between Context and Value
    ITensorPack gemm_context_pack{ { ACL_SRC_0, softmaxed_product.get() }, { ACL_SRC_1, permuted_value.get() }, { ACL_DST, gemmed_context.get() } };
    CLScheduler::get().enqueue_op(*_context_mm_kernel, gemm_context_pack, true);

    // Concat all attention head together
    ITensorPack concat_permute_pack{ { ACL_SRC, gemmed_context.get() }, { ACL_DST, permuted_concat.get() } };
    CLScheduler::get().enqueue_op(*_concat_permute_kernel, concat_permute_pack, true);

    ITensorPack concat_reshape_pack{ { ACL_SRC_0, permuted_concat.get() }, { ACL_DST, output } };
    CLScheduler::get().enqueue_op(*_concat_reshape_kernel, concat_reshape_pack, true);


    /*
    CpuAuxTensorHandler reshaped_query(offset_int_vec(QueryReshape), _reshaped_query, tensors);
    CpuAuxTensorHandler permuted_query(offset_int_vec(QueryPermute), _permuted_query, tensors);
    CpuAuxTensorHandler reshaped_key(offset_int_vec(KeyReshape), _reshaped_key, tensors);
    CpuAuxTensorHandler permuted_key(offset_int_vec(KeyPermute), _permuted_key, tensors);
    CpuAuxTensorHandler transposed_key(offset_int_vec(KeyTranspose), _transposed_key, tensors);
    CpuAuxTensorHandler reshaped_value(offset_int_vec(ValueReshape), _reshaped_value, tensors);
    CpuAuxTensorHandler permuted_value(offset_int_vec(ValuePermute), _permuted_value, tensors);
    CpuAuxTensorHandler permuted_concat(offset_int_vec(ConcatPermute), _permuted_concat, tensors);

    CpuAuxTensorHandler scaled_query_key(offset_int_vec(QueryKeyScale), _scaled_query_key, tensors);
    CpuAuxTensorHandler interleaved_query(offset_int_vec(InterleavedLHS), _tmp_query, tensors, true);
    CpuAuxTensorHandler transposed1xw_key(offset_int_vec(Transposed1xWRHS), _tmp_key, tensors, true);
    CpuAuxTensorHandler softmaxed_product(offset_int_vec(Softmax), _softmaxed_product, tensors);
    CpuAuxTensorHandler interleaved_product(offset_int_vec(InterleavedProduct), _interleaved_product, tensors, true);
    CpuAuxTensorHandler transposed1xW_value(offset_int_vec(Transposed1xWValue), _transposed1xW_value, tensors, true);
    CpuAuxTensorHandler gemmed_context(offset_int_vec(GemmedContext), _gemmed_context, tensors);

    // Run Query multi-Head reshape
    ITensorPack query_reshape_pack{ { ACL_SRC_0, query }, { ACL_DST, reshaped_query.get() } };
    const auto  query_split_dimension = _query_reshape_kernel->get_split_dimension();
    NEScheduler::get().schedule_op(_query_reshape_kernel.get(), query_split_dimension, _query_reshape_kernel->window(), query_reshape_pack);

    ITensorPack query_permute_pack{ { ACL_SRC, reshaped_query.get() }, { ACL_DST, permuted_query.get() } };
    _query_permute_func->run(query_permute_pack);

    // Run Key multi-Head reshape
    ITensorPack key_reshape_pack{ { ACL_SRC_0, key }, { ACL_DST, reshaped_key.get() } };
    const auto  key_split_dimension = _key_reshape_kernel->get_split_dimension();
    NEScheduler::get().schedule_op(_key_reshape_kernel.get(), key_split_dimension, _key_reshape_kernel->window(), key_reshape_pack);

    ITensorPack key_permute_pack{ { ACL_SRC, reshaped_key.get() }, { ACL_DST, permuted_key.get() } };
    _key_permute_func->run(key_permute_pack);

    ITensorPack key_transpose_pack{ { ACL_SRC, permuted_key.get() }, { ACL_DST, transposed_key.get() } };
    _key_transpose_func->run(key_transpose_pack);

    // Run Value multi-Head reshape
    ITensorPack value_reshape_pack{ { ACL_SRC_0, value }, { ACL_DST, reshaped_value.get() } };
    const auto  value_split_dimension = _value_reshape_kernel->get_split_dimension();
    NEScheduler::get().schedule_op(_value_reshape_kernel.get(), value_split_dimension, _value_reshape_kernel->window(), value_reshape_pack);

    ITensorPack value_permute_pack{ { ACL_SRC, reshaped_value.get() }, { ACL_DST, permuted_value.get() } };
    _value_permute_func->run(value_permute_pack);

    // Run interleave kernel
    ITensorPack interleave_pack{ { ACL_SRC, permuted_query.get() }, { ACL_DST, interleaved_query.get() } };
    NEScheduler::get().schedule_op(_query_interleave_kernel.get(), Window::DimY, _query_interleave_kernel->window(),
                                   interleave_pack);

    // Run transpose1xw kernel
    ITensorPack transpose_pack{ { ACL_SRC, transposed_key.get() }, { ACL_DST, transposed1xw_key.get() } };
    NEScheduler::get().schedule_op(_key_transpose1xW_kernel.get(), Window::DimY,
                                   _key_transpose1xW_kernel->window(), transpose_pack);

    // Run matrix multiply compute multi-head attention between Query and Key
    ITensorPack gemm_QK_pack{ { ACL_SRC_0, interleaved_query.get() }, { ACL_SRC_1, transposed1xw_key.get() }, { ACL_DST, scaled_query_key.get() } };
    NEScheduler::get().schedule_op(_product_mm_kernel.get(), Window::DimZ, _product_mm_kernel->window(), gemm_QK_pack);

    ITensorPack softmax_pack = { { ACL_SRC, scaled_query_key.get() }, { ACL_DST, softmaxed_product.get() } };
    _softmax_func->run(softmax_pack);

    // Run interleave kernel
    ITensorPack interleave_product_pack{ { ACL_SRC, softmaxed_product.get() }, { ACL_DST, interleaved_product.get() } };
    NEScheduler::get().schedule_op(_product_interleave_kernel.get(), Window::DimY, _product_interleave_kernel->window(),
                                   interleave_product_pack);

    // Run transpose1xw kernel
    ITensorPack transpose_value_pack{ { ACL_SRC, permuted_value.get() }, { ACL_DST, transposed1xW_value.get() } };
    NEScheduler::get().schedule_op(_value_transpose1xW_kernel.get(), Window::DimY,
                                   _value_transpose1xW_kernel->window(), transpose_value_pack);

    // Run matrix multiply compute multi-head attention between Query and Key
    ITensorPack gemm_context_pack{ { ACL_SRC_0, interleaved_product.get() }, { ACL_SRC_1, transposed1xW_value.get() }, { ACL_DST, gemmed_context.get() } };
    NEScheduler::get().schedule_op(_context_mm_kernel.get(), Window::DimZ, _context_mm_kernel->window(), gemm_context_pack);

    // Concat all attention head together
    ITensorPack concat_permute_pack{ { ACL_SRC, gemmed_context.get() }, { ACL_DST, permuted_concat.get() } };
    _concat_permute_func->run(concat_permute_pack);

    ITensorPack concat_reshape_pack{ { ACL_SRC_0, permuted_concat.get() }, { ACL_DST, output } };
    const auto  concat_split_dimension = _concat_reshape_kernel->get_split_dimension();
    NEScheduler::get().schedule_op(_concat_reshape_kernel.get(), concat_split_dimension, _concat_reshape_kernel->window(), concat_reshape_pack);
    */
}

experimental::MemoryRequirements ClScaleDotProduction::workspace() const
{
    return _aux_mem;
}

} // namespace opencl
} // namespace arm_compute
