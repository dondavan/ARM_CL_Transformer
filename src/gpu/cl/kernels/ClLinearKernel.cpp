#include "src/gpu/cl/kernels/ClLinearKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/utils/ActivationFunctionUtils.h"
#include "arm_compute/core/utils/StringUtils.h"
#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "src/common/utils/Log.h"
#include "src/core/CL/CLUtils.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/gpu/cl/kernels/gemm/ClGemmHelpers.h"
#include "src/gpu/cl/kernels/helpers/MatMulKernelHelpers.h"
#include "support/Cast.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{

void ClLinearKernel::configure(const CLCompileContext &compile_context,
                               ITensorInfo            *lhs,
                               ITensorInfo            *rhs,
                               ITensorInfo            *bias,
                               ITensorInfo            *dst,
                               float                   alpha,
                               float                   beta,
                               const MatMulKernelInfo &matmul_kernel_info)
{
    ARM_COMPUTE_UNUSED(compile_context);
    ARM_COMPUTE_UNUSED(alpha,beta);

    std::cout << "src/gpu/cl/kernels/ClLinearKernel.cpp configure start" << std::endl;

    // dst tensor auto initialization if not yet initialized
    auto_init_if_empty(*dst, lhs->clone()->set_tensor_shape(misc::shape_calculator::compute_matmul_shape(
                                 lhs->tensor_shape(), rhs->tensor_shape(), matmul_kernel_info)));

    const int  m       = dst->dimension(1);
    const int  n       = dst->dimension(0);
    const int  k       = matmul_kernel_info.adj_lhs ? lhs->tensor_shape().y() : lhs->tensor_shape().x();
    const bool adj_lhs = matmul_kernel_info.adj_lhs;

    int m0 = adj_lhs ? adjust_vec_size(matmul_kernel_info.m0, m) : std::min(matmul_kernel_info.m0, m);
    int n0 = adjust_vec_size(matmul_kernel_info.n0, n);

    _export_rhs_to_cl_image = matmul_kernel_info.export_rhs_to_cl_image && !rhs->lock_paddings();

    // Configure kernel window
    Window win = calculate_max_window(*dst, Steps(n0, m0));
    win        = win.collapse(win, Window::DimZ);
    IClKernel::configure_internal(win);

    // Calculate partial (store instead of load) M0 and partial N0 for the partial blocks at the end of a row/column if any. This is to avoid padding.
    const unsigned int partial_store_m0 = m % m0;
    const unsigned int partial_store_n0 = n % n0;

    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(lhs->data_type()));
    build_opts.add_option("-DM0=" + support::cpp11::to_string(m0));
    build_opts.add_option("-DN0=" + support::cpp11::to_string(n0));
    build_opts.add_option("-DK0=" + support::cpp11::to_string(matmul_kernel_info.k0));
    build_opts.add_option("-DPARTIAL_STORE_M0=" + support::cpp11::to_string(partial_store_m0));
    build_opts.add_option("-DPARTIAL_STORE_N0=" + support::cpp11::to_string(partial_store_n0));
    build_opts.add_option("-DK=" + support::cpp11::to_string(k));
    build_opts.add_option_if(bias != nullptr, "-DBIAS");
    build_opts.add_option_if_else(_export_rhs_to_cl_image, "-DRHS_TENSOR_TYPE=IMAGE", "-DRHS_TENSOR_TYPE=BUFFER");

    // Define values for activation function

    std::string kernel_name("mat_mul_native");
    kernel_name += matmul_kernel_info.adj_lhs ? "_t" : "_nt";
    kernel_name += matmul_kernel_info.adj_rhs ? "_t" : "_nt";

    // A macro guard to compile ONLY the kernel of interest
    build_opts.add_option("-D" + upper_string(kernel_name));

    if (_export_rhs_to_cl_image)
    {
        gemm::update_padding_for_cl_image(rhs);
    }

    // Create kernel
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(lhs->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(m);
    _config_id += "_";
    _config_id += support::cpp11::to_string(n);
    _config_id += "_";
    _config_id += support::cpp11::to_string(k);
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(2));
    _config_id += "_";
    _config_id += support::cpp11::to_string(_export_rhs_to_cl_image);
    _config_id += "_";
    _config_id += support::cpp11::to_string(m0);
    _config_id += "_";
    _config_id += support::cpp11::to_string(n0);
    _config_id += "_";
    _config_id += support::cpp11::to_string(matmul_kernel_info.k0);
    std::cout << "src/gpu/cl/kernels/ClLinearKernel.cpp configure end" << std::endl;
}

Status ClLinearKernel::validate(const ITensorInfo *src, const ITensorInfo *vector, ITensorInfo *dst)
{
    ARM_COMPUTE_UNUSED(src);
    ARM_COMPUTE_UNUSED(vector);
    ARM_COMPUTE_UNUSED(dst);
    return Status{};
}

void ClLinearKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    std::cout << "src/gpu/cl/kernels/ClLinearKernel.cpp run start" << std::endl;

    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(tensors.empty());

    const ICLTensor *lhs   = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_0));
    const ICLTensor *rhs   = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_1));
    const ICLTensor *bias  = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_2)); // nullptr if bias is not present
    ICLTensor       *dst   = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    // Set srcs
    Window       window_collapsed = window.collapse(ICLKernel::window(), Window::DimZ);

    std::cout << "window_collapsed x " << window_collapsed.x().end() << std::endl;
    std::cout << "window_collapsed y " << window_collapsed.y().end() << std::endl;
    std::cout << "window_collapsed z " << window_collapsed.z().end() << std::endl;

    std::cout << "lhs->info()->strides_in_bytes().x() " << lhs->info()->strides_in_bytes().x() << std::endl;
    std::cout << "lhs->info()->strides_in_bytes().y() " << lhs->info()->strides_in_bytes().y() << std::endl;
    std::cout << "lhs->info()->strides_in_bytes().z() " << lhs->info()->strides_in_bytes().z() << std::endl;

    std::cout << "lhs->info()->strides_in_bytes().x() " << lhs->info()->strides_in_bytes().x() << std::endl;
    std::cout << "lhs->info()->strides_in_bytes().y() " << lhs->info()->strides_in_bytes().y() << std::endl;
    std::cout << "lhs->info()->strides_in_bytes().z() " << lhs->info()->strides_in_bytes().z() << std::endl;

    std::cout << "dst->info()->strides_in_bytes().x() " << dst->info()->strides_in_bytes().x() << std::endl;
    std::cout << "dst->info()->strides_in_bytes().y() " << dst->info()->strides_in_bytes().y() << std::endl;
    std::cout << "dst->info()->strides_in_bytes().z() " << dst->info()->strides_in_bytes().z() << std::endl;

    unsigned int idx = 0;
    add_3d_tensor_nhw_argument(idx, lhs);
    add_3d_tensor_nhw_argument(idx, rhs);
    if(bias != nullptr)
    {
        add_3d_tensor_nhw_argument(idx, bias);
    }
    add_3d_tensor_nhw_argument(idx, dst);

    enqueue(queue, *this, window_collapsed, lws_hint());

    //add_3D_tensor_argument(idx, src, window);
    //add_3D_tensor_argument(idx, vector, window);
    //add_3D_tensor_argument(idx, dst, window);

    std::cout << "src/gpu/cl/kernels/ClLinearKernel.cpp run end" << std::endl;
}

} // namespace kernels
} // namespace opencl
} // namespace arm_compute