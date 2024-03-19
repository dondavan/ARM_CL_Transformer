#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"

namespace arm_compute
{
namespace cpu
{
void neon_token_embed_char_2_float32(const ITensor *src, const ITensor *vocab, ITensor *dst, const TokenEmbeddingLayerInfo &tkemb_info, const Window &window)
{
    std::cout << "src/cpu/kernels/tokenembed/generic/neon/fp32.cpp: neon_token_embed_char_2_float32" << std::endl;
    std::cout << tkemb_info.d_model() << std::endl;

    Window win = window;
    win.set(Window::DimX, Window::Dimension(0,1,1));
    const unsigned int window_start_x   = static_cast<unsigned int>(window.x().start());
    const unsigned int window_end_x     = src->info()->tensor_shape().x();
    unsigned int       x                = window_start_x;

    const unsigned int dst_start_y      = static_cast<unsigned int>(window.y().start());
    const unsigned int dst_end_y        = dst->info()->tensor_shape().y();
    //unsigned int       y                = dst_start_y;

    unsigned int offset_src;

    std::cout << "Tensor shape" << std::endl;
    std::cout << src->info()->tensor_shape().x() << std::endl;
    std::cout << dst->info()->tensor_shape().x() << std::endl;
    std::cout << dst->info()->tensor_shape().y() << std::endl;
    std::cout << tkemb_info.d_model() << std::endl;
    std::cout << tkemb_info.d_vocab() << std::endl;
    
    Iterator src_iter(src,win);
    Iterator dst_iter(dst,win);
    Iterator vocab_iter(vocab,win);

    const auto src_ptr      = reinterpret_cast<unsigned int *>(src_iter.ptr());
    const auto dst_ptr      = reinterpret_cast<float *>(dst_iter.ptr());
    const auto vocab_ptr    = reinterpret_cast<float *>(vocab_iter.ptr());

    std::cout << "YeaHhhhhhhhhhhh " << std::endl;
    execute_window_loop(win,
        [&](const Coordinates &)
        {
            for(; x < window_end_x; x++)
            {
                std::cout << *(src_ptr+x) << std::endl;

                std::cout << *(vocab_ptr) << std::endl;
                offset_src = dst_start_y+  0 * dst_end_y - 1;
                std::cout << *(vocab_ptr + offset_src) << std::endl;
                offset_src = dst_start_y+  1 * dst_end_y - 1;
                std::cout << *(vocab_ptr + offset_src) << std::endl;
                
                //offset_dst = x * dst_end_y - 1;

                //std::memcpy(dst_ptr + offset_dst, src_ptr, dst_end_y * sizeof( *src_ptr));
            }

            std::cout << *(vocab_ptr) << std::endl;
        },vocab_iter);

}

} // namespace cpu
} // namespace arm_compute
