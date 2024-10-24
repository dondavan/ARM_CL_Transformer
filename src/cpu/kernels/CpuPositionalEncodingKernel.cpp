#include "src/cpu/kernels/CpuPositionalEncodingKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/Validate.h"

#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <cmath>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{

namespace
{
/**  Compute Transformer positional encoding
 *
 *  @note math: PE(pos,2i)   = sin(pos/10000^(2i/dmodel))
 *              PE(pos,2i+1) = cos(pos/10000^(2i/dmodel))
 */
template <typename T>
void run_positional_encoding(const Window &window, ITensor *src, ITensor *dst, const unsigned int d_model)
{
    ARM_COMPUTE_ERROR_ON_MSG(d_model%2!=0, "Model depth (d_model) must be dividable by 2");

    std::cout << "src/cpu/kernels/CpuPositionalEncodingKernel.cpp" << std::endl;

    Window win = window;
    win.set(Window::DimX, Window::Dimension(0,1,1));
    win.set(Window::DimY, Window::Dimension(0,1,1));
    /* token sequence */
    const unsigned int window_start_x   = static_cast<unsigned int>(window.x().start());
    const unsigned int window_end_x     = static_cast<unsigned int>(window.x().end());

    unsigned int token_offset;

    Iterator src_iter(src,win);
    Iterator dst_iter(dst,win);
    const auto src_ptr  = reinterpret_cast<T *>(src_iter.ptr());
    const auto dst_ptr  = reinterpret_cast<T *>(dst_iter.ptr());

    T PE_2i,PE_2i1;
    execute_window_loop(win,
        [&](const Coordinates &)
        {
            for(unsigned int pos = window_start_x; pos < window_end_x; pos++)
            {
                token_offset = pos * d_model;
                for(unsigned int i = 0; i < d_model ; i+=2)
                {
                    double div_term = exp(i * -log(10000.0) / d_model);
                    PE_2i   = sin(pos * div_term);
                    PE_2i1  = cos(pos * div_term);

                    *(dst_ptr + token_offset + i)       = *(src_ptr + token_offset + i)   + PE_2i;
                    *(dst_ptr + token_offset + i+1)     = *(src_ptr + token_offset + i+1) + PE_2i1;
                }
            }
        }
    ,src_iter);

}

}

void CpuPositionalEncodingKernel::configure(const ITensorInfo *src, ITensorInfo *dst, const unsigned int d_model)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    _d_model = d_model;

    // Configure output tensor info.
    auto_init_if_empty(*dst, TensorInfo(*src->clone()));

    // Configure kernel window
    Window win = calculate_max_window(*src, Steps());
    ICpuKernel::configure(win);
}


Status CpuPositionalEncodingKernel::validate(const ITensorInfo *src, const ITensorInfo *dst, const unsigned int d_model)
{
    ARM_COMPUTE_UNUSED(src);
    ARM_COMPUTE_UNUSED(dst);
    ARM_COMPUTE_UNUSED(d_model);

    return Status{};
}

void CpuPositionalEncodingKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);

    auto src = tensors.get_tensor(TensorType::ACL_SRC);
    auto dst = tensors.get_tensor(TensorType::ACL_DST);

    run_positional_encoding<float>(window, src, dst, _d_model);
}

const char * CpuPositionalEncodingKernel::name() const
{
    return "CpuPositionalEncodingKernel";
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
