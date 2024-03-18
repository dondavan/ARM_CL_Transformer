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

    std::cout << src->info()->id() << std::endl;
    std::cout << vocab->info()->id() << std::endl;
    std::cout << dst->info()->id() << std::endl;
    std::cout << tkemb_info.d_vocab() << std::endl;
    std::cout << window.DimX << std::endl;

    Window win = window;

    Iterator src_iter(src,win);
    Iterator vocab_iter(vocab,win);

    const auto src_ptr      = reinterpret_cast<char *>(vocab_iter.ptr());
    const auto vocab_ptr    = reinterpret_cast<float *>(vocab_iter.ptr());

    std::cout << "YeaHhhhhhhhhhhh " << std::endl;
    execute_window_loop(win,
        [&](const Coordinates &){
            std::cout << *(src_ptr) << std::endl;
            std::cout << *(vocab_ptr) << std::endl;
        },vocab_iter);

}

} // namespace cpu
} // namespace arm_compute
