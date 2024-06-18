#include "src/gpu/cl/kernels/ClVectorizeKernel.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"

#include "src/core/CPP/Validate.h"
#include "src/core/common/Registrars.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/vectorize/list.h"

#include "support/Cast.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{

namespace
{


/**  Vectorize pretrained position embedding*/
template <typename T>
void run_vectorize(const Window &window, const ITensor *src, const ITensor *vector, ITensor *dst)
{
    /* Runtime reshape valid tensor region if input has been reshaped during preprocess */
    size_t reshape_input_x = src->info()->valid_region().shape.x();
    if(src->info()->tensor_shape().x() != reshape_input_x)
    {
        dst->info()->set_valid_region(dst->info()->valid_region().set(0,0,reshape_input_x));
    }
    Window win(window);

    const unsigned int window_start_x   = static_cast<unsigned int>(win.x().start());
    const unsigned int window_end_x     = static_cast<unsigned int>(win.x().end());

    const unsigned int vector_depth     = vector->info()->tensor_shape().x(); 

    unsigned int offset_vector,offset_dst;

    win.set(Window::DimX, Window::Dimension(0,1,1));
    Iterator src_iter(src,win);
    Iterator dst_iter(dst,win);
    Iterator vector_iter(vector,win);

    const auto src_ptr      = reinterpret_cast<unsigned int *>(src_iter.ptr());
    const auto dst_ptr      = reinterpret_cast<float *>(dst_iter.ptr());
    const auto vector_ptr   = reinterpret_cast<float *>(vector_iter.ptr());
    execute_window_loop(win,
        [&](const Coordinates &)
        {
            for(unsigned int x = window_start_x; x < window_end_x; x++)
            {
                offset_dst     = x * vector_depth;
                offset_vector  = *(src_ptr+x) * vector_depth;
                std::memcpy(dst_ptr + offset_dst, vector_ptr + offset_vector, (vector_depth) * sizeof(*vector_ptr));
            }
            
        }, src_iter);
}

} // namespace

void ClVectorizeKernel::configure(const CLCompileContext &compile_context, const ITensorInfo *src, const ITensorInfo *vector, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src);
    ARM_COMPUTE_ERROR_ON_NULLPTR(vector);
    ARM_COMPUTE_UNUSED(compile_context);


    // Configure output tensor info.
    const TensorShape dst_shape(vector->tensor_shape().x(), src->tensor_shape().x());
    if(dst->tensor_shape().total_size() == 0)
    {
        auto_init_if_empty(*dst, TensorInfo(*vector->clone()).set_tensor_shape(dst_shape));
    }
    else
    {
        dst->set_tensor_shape(dst_shape);
    }

    // Configure kernel window
    Window win = calculate_max_window(*src, Steps());
    ICLKernel::configure_internal(win);
}

Status ClVectorizeKernel::validate(const ITensorInfo *src, const ITensorInfo *vector, ITensorInfo *dst)
{
    ARM_COMPUTE_UNUSED(src);
    ARM_COMPUTE_UNUSED(vector);
    ARM_COMPUTE_UNUSED(dst);
    return Status{};
}

void ClVectorizeKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);
    ARM_COMPUTE_UNUSED(queue);

    ARM_COMPUTE_ERROR_ON(tensors.empty());
    const auto src =
        utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_0));
    const auto vector = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_1));
    auto dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    run_vectorize<float>(window, src, vector, dst);
}

} // namespace kernels
} // namespace opencl
} // namespace arm_compute