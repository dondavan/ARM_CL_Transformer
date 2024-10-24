#include "src/cpu/kernels/CpuTokenEmbedKernel.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"

#include "src/core/common/Registrars.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/tokenembed/list.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{

namespace
{
static const std::vector<CpuTokenEmbedKernel::TKEMBKernel> available_kernels = {
    /*
#ifdef ARM_COMPUTE_ENABLE_SVE
    // TBA
#endif // ARM_COMPUTE_ENABLE_SVE

#ifdef __aarch64__
    // TBA
#endif // __aarch64__
*/
    {"neon_fp32_token_embedding", [](const TokenEmbedKernelDataTypeISASelectorData &data) { return data.dt == DataType::F32; },
     REGISTER_FP32_NEON(arm_compute::cpu::neon_token_embed_char_2_float32)},

};
}

void CpuTokenEmbedKernel::configure(const ITensorInfo *src, const ITensorInfo *vocab, ITensorInfo *dst, EmbeddingLayerInfo tkemb_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src);
    ARM_COMPUTE_ERROR_ON_NULLPTR(vocab);
    
    const auto uk = CpuTokenEmbedKernel::get_implementation(
        TokenEmbedKernelDataTypeISASelectorData{dst->data_type(), CPUInfo::get().get_isa()}
    );

    const TensorShape dst_shape(src->tensor_shape().x(),vocab->tensor_shape().y());
    // Configure output tensor info.
    auto_init_if_empty(*dst, TensorInfo(*vocab->clone()).set_tensor_shape(dst_shape));

    ARM_COMPUTE_ERROR_ON_NULLPTR(uk);

    _run_method = uk->ukernel;
    _tkemb_info = tkemb_info;
    _name       = std::string("CpuTokenEmbedKernel").append("/").append(uk->name);

    Window win;

    win = calculate_max_window(*dst, Steps());
    ICPPKernel::configure(win);
    
}

Status CpuTokenEmbedKernel::validate(const ITensorInfo *src,  const ITensorInfo *vocab, ITensorInfo *dst, EmbeddingLayerInfo tkemb_info)
{
    ARM_COMPUTE_UNUSED(src);
    ARM_COMPUTE_UNUSED(vocab);
    ARM_COMPUTE_UNUSED(dst);
    ARM_COMPUTE_UNUSED(tkemb_info);
    return Status{};
}


size_t CpuTokenEmbedKernel::get_mws(const CPUInfo &platform, size_t thread_count) const
{
    ARM_COMPUTE_UNUSED(thread_count);
    ARM_COMPUTE_UNUSED(platform);

    if (_split_dimension == Window::DimX)
    {
        // Don't split the work load too small if the tensor has been reinterpreted as 1D.
        // This number is loosely chosen as threading overhead in each platform varies wildly.
        return 1536;
    }
    return default_mws;
}

void CpuTokenEmbedKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{

    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    ARM_COMPUTE_ERROR_ON(tensors.empty());
    ARM_COMPUTE_ERROR_ON(_run_method == nullptr);

    const ITensor *src   = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const ITensor *vocab = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    ITensor       *dst   = tensors.get_tensor(TensorType::ACL_DST);

    _run_method(src, vocab, dst, _tkemb_info, window);
}

const char *CpuTokenEmbedKernel::name() const
{
    return _name.c_str();
}

const std::vector<CpuTokenEmbedKernel::TKEMBKernel> &CpuTokenEmbedKernel::get_available_kernels()
{
    return available_kernels ;
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute