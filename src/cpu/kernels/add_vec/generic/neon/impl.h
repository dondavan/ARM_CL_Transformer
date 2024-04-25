#ifndef SRC_CORE_NEON_KERNELS_ADD_VEC_IMPL_H
#define SRC_CORE_NEON_KERNELS_ADD_VEC_IMPL_H
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/Traits.h"
#include "arm_compute/core/Window.h"

#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/helpers/Utils.h"

namespace arm_compute
{
namespace cpu
{
template <typename ScalarType>
void add_vec_same_neon(
    const ITensor *src0, const ITensor *src1, ITensor *dst, const ConvertPolicy &policy, const Window &window)
{
    std::cout << "Add veccccccccccccccccccccccccccccccccccccccc  " << std::endl;
    /** SIMD vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_bitvector_tag_t<ScalarType, wrapper::traits::BitWidth::W128>;

    // Create input windows
    Window input1_win = window.broadcast_if_dimension_le_one(src0->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(src1->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    constexpr int window_step_x         = 16 / sizeof(ScalarType);
    const auto    window_start_x        = static_cast<int>(window.x().start());
    const auto    window_end_x          = static_cast<int>(window.x().end());
    const bool    is_broadcast_across_x = src0->info()->tensor_shape().x() != src1->info()->tensor_shape().x();

    if (is_broadcast_across_x)
    {
        const bool     is_broadcast_input_2 = input2_win.x().step() == 0;
        Window         broadcast_win        = is_broadcast_input_2 ? input2_win : input1_win;
        Window         non_broadcast_win    = !is_broadcast_input_2 ? input2_win : input1_win;
        const ITensor *broadcast_tensor     = is_broadcast_input_2 ? src1 : src0;
        const ITensor *non_broadcast_tensor = !is_broadcast_input_2 ? src1 : src0;

        // Clear X Dimension on execution window as we handle manually
        non_broadcast_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator broadcast_input(broadcast_tensor, broadcast_win);
        Iterator non_broadcast_input(non_broadcast_tensor, non_broadcast_win);
        Iterator output(dst, win);

        execute_window_loop(
            win,
            [&](const Coordinates &)
            {
                const auto non_broadcast_input_ptr = reinterpret_cast<const ScalarType *>(non_broadcast_input.ptr());
                const auto output_ptr              = reinterpret_cast<ScalarType *>(output.ptr());

                const ScalarType broadcast_value     = *reinterpret_cast<const ScalarType *>(broadcast_input.ptr());
                const auto       broadcast_value_vec = wrapper::vdup_n(broadcast_value, ExactTagType{});

                // Compute S elements per iteration
                int x = window_start_x;
                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                {
                    const auto non_broadcast_v = wrapper::vloadq(non_broadcast_input_ptr + x);
                    const auto res             = (policy == ConvertPolicy::SATURATE)
                                                     ? wrapper::vqadd(broadcast_value_vec, non_broadcast_v)
                                                     : wrapper::vadd(broadcast_value_vec, non_broadcast_v);
                    wrapper::vstore(output_ptr + x, res);
                }

                // Compute left-over elements
                for (; x < window_end_x; ++x)
                {
                    const auto non_broadcast_v = *(non_broadcast_input_ptr + x);
                    *(output_ptr + x)          = (policy == ConvertPolicy::SATURATE)
                                                     ? wrapper::add_sat(broadcast_value, non_broadcast_v)
                                                     : broadcast_value + non_broadcast_v;
                }
            },
            broadcast_input, non_broadcast_input, output);
    }
    else
    {
        // Clear X Dimension on execution window as we handle manually
        input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
        input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator input1(src0, input1_win);
        Iterator input2(src1, input2_win);
        Iterator output(dst, win);

        execute_window_loop(
            win,
            [&](const Coordinates &)
            {
                const auto input1_ptr = reinterpret_cast<const ScalarType *>(input1.ptr());
                const auto input2_ptr = reinterpret_cast<const ScalarType *>(input2.ptr());
                const auto output_ptr = reinterpret_cast<ScalarType *>(output.ptr());

                // Compute S elements per iteration
                int x = window_start_x;
                for (; x <= (window_end_x - window_step_x); x += window_step_x)
                {
                    const auto val1 = wrapper::vloadq(input1_ptr + x);
                    const auto val2 = wrapper::vloadq(input2_ptr + x);
                    const auto res =
                        (policy == ConvertPolicy::SATURATE) ? wrapper::vqadd(val1, val2) : wrapper::vadd(val1, val2);
                    wrapper::vstore(output_ptr + x, res);
                }

                // Compute left-over elements
                for (; x < window_end_x; ++x)
                {
                    const auto val1 = *(input1_ptr + x);
                    const auto val2 = *(input2_ptr + x);
                    *(output_ptr + x) =
                        (policy == ConvertPolicy::SATURATE) ? wrapper::add_sat(val1, val2) : val1 + val2;
                }
            },
            input1, input2, output);
    }
}

} // namespace cpu
} // namespace arm_compute
#endif // SRC_CORE_NEON_KERNELS_ADD_IMPL_H
