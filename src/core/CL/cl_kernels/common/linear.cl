#include "activation_float_helpers.h"
#include "helpers.h"
#include "tile_helpers.h"

#define UNROLL_INCR(idx, step, macro) idx += (step); (macro)

#define LOOP_UNROLLING_1(idx, step, macro) (macro)
#define LOOP_UNROLLING_2(idx, step, macro) LOOP_UNROLLING_1(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_3(idx, step, macro) LOOP_UNROLLING_2(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_4(idx, step, macro) LOOP_UNROLLING_3(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_5(idx, step, macro) LOOP_UNROLLING_4(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_6(idx, step, macro) LOOP_UNROLLING_5(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_7(idx, step, macro) LOOP_UNROLLING_6(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_8(idx, step, macro) LOOP_UNROLLING_7(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_9(idx, step, macro) LOOP_UNROLLING_8(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_10(idx, step, macro) LOOP_UNROLLING_9(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_11(idx, step, macro) LOOP_UNROLLING_10(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_12(idx, step, macro) LOOP_UNROLLING_11(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_13(idx, step, macro) LOOP_UNROLLING_12(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_14(idx, step, macro) LOOP_UNROLLING_13(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_15(idx, step, macro) LOOP_UNROLLING_14(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_16(idx, step, macro) LOOP_UNROLLING_15(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_17(idx, step, macro) LOOP_UNROLLING_16(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_18(idx, step, macro) LOOP_UNROLLING_17(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_19(idx, step, macro) LOOP_UNROLLING_18(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_20(idx, step, macro) LOOP_UNROLLING_19(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_21(idx, step, macro) LOOP_UNROLLING_20(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_22(idx, step, macro) LOOP_UNROLLING_21(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_23(idx, step, macro) LOOP_UNROLLING_22(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_24(idx, step, macro) LOOP_UNROLLING_23(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_25(idx, step, macro) LOOP_UNROLLING_24(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_26(idx, step, macro) LOOP_UNROLLING_25(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_27(idx, step, macro) LOOP_UNROLLING_26(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_28(idx, step, macro) LOOP_UNROLLING_27(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_29(idx, step, macro) LOOP_UNROLLING_28(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_30(idx, step, macro) LOOP_UNROLLING_29(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_31(idx, step, macro) LOOP_UNROLLING_30(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_32(idx, step, macro) LOOP_UNROLLING_31(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_33(idx, step, macro) LOOP_UNROLLING_32(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_34(idx, step, macro) LOOP_UNROLLING_33(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_35(idx, step, macro) LOOP_UNROLLING_34(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_36(idx, step, macro) LOOP_UNROLLING_35(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_37(idx, step, macro) LOOP_UNROLLING_36(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_38(idx, step, macro) LOOP_UNROLLING_37(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_39(idx, step, macro) LOOP_UNROLLING_38(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_40(idx, step, macro) LOOP_UNROLLING_39(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_41(idx, step, macro) LOOP_UNROLLING_40(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_42(idx, step, macro) LOOP_UNROLLING_41(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_43(idx, step, macro) LOOP_UNROLLING_42(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_44(idx, step, macro) LOOP_UNROLLING_43(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_45(idx, step, macro) LOOP_UNROLLING_44(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_46(idx, step, macro) LOOP_UNROLLING_45(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_47(idx, step, macro) LOOP_UNROLLING_46(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_48(idx, step, macro) LOOP_UNROLLING_47(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_49(idx, step, macro) LOOP_UNROLLING_48(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_50(idx, step, macro) LOOP_UNROLLING_49(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_51(idx, step, macro) LOOP_UNROLLING_50(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_52(idx, step, macro) LOOP_UNROLLING_51(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_53(idx, step, macro) LOOP_UNROLLING_52(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_54(idx, step, macro) LOOP_UNROLLING_53(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_55(idx, step, macro) LOOP_UNROLLING_54(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_56(idx, step, macro) LOOP_UNROLLING_55(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_57(idx, step, macro) LOOP_UNROLLING_56(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_58(idx, step, macro) LOOP_UNROLLING_57(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_59(idx, step, macro) LOOP_UNROLLING_58(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_60(idx, step, macro) LOOP_UNROLLING_59(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_61(idx, step, macro) LOOP_UNROLLING_60(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_62(idx, step, macro) LOOP_UNROLLING_61(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_63(idx, step, macro) LOOP_UNROLLING_62(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_64(idx, step, macro) LOOP_UNROLLING_63(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_65(idx, step, macro) LOOP_UNROLLING_64(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_66(idx, step, macro) LOOP_UNROLLING_65(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_67(idx, step, macro) LOOP_UNROLLING_66(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_68(idx, step, macro) LOOP_UNROLLING_67(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_69(idx, step, macro) LOOP_UNROLLING_68(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_70(idx, step, macro) LOOP_UNROLLING_69(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_71(idx, step, macro) LOOP_UNROLLING_70(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_72(idx, step, macro) LOOP_UNROLLING_71(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_73(idx, step, macro) LOOP_UNROLLING_72(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_74(idx, step, macro) LOOP_UNROLLING_73(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_75(idx, step, macro) LOOP_UNROLLING_74(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_76(idx, step, macro) LOOP_UNROLLING_75(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_77(idx, step, macro) LOOP_UNROLLING_76(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_78(idx, step, macro) LOOP_UNROLLING_77(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_79(idx, step, macro) LOOP_UNROLLING_78(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_80(idx, step, macro) LOOP_UNROLLING_79(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_81(idx, step, macro) LOOP_UNROLLING_80(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_82(idx, step, macro) LOOP_UNROLLING_81(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_83(idx, step, macro) LOOP_UNROLLING_82(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_84(idx, step, macro) LOOP_UNROLLING_83(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_85(idx, step, macro) LOOP_UNROLLING_84(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_86(idx, step, macro) LOOP_UNROLLING_85(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_87(idx, step, macro) LOOP_UNROLLING_86(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_88(idx, step, macro) LOOP_UNROLLING_87(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_89(idx, step, macro) LOOP_UNROLLING_88(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_90(idx, step, macro) LOOP_UNROLLING_89(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_91(idx, step, macro) LOOP_UNROLLING_90(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_92(idx, step, macro) LOOP_UNROLLING_91(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_93(idx, step, macro) LOOP_UNROLLING_92(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_94(idx, step, macro) LOOP_UNROLLING_93(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_95(idx, step, macro) LOOP_UNROLLING_94(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_96(idx, step, macro) LOOP_UNROLLING_95(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_97(idx, step, macro) LOOP_UNROLLING_96(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_98(idx, step, macro) LOOP_UNROLLING_97(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_99(idx, step, macro) LOOP_UNROLLING_98(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_100(idx, step, macro) LOOP_UNROLLING_99(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_101(idx, step, macro) LOOP_UNROLLING_100(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_102(idx, step, macro) LOOP_UNROLLING_101(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_103(idx, step, macro) LOOP_UNROLLING_102(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_104(idx, step, macro) LOOP_UNROLLING_103(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_105(idx, step, macro) LOOP_UNROLLING_104(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_106(idx, step, macro) LOOP_UNROLLING_105(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_107(idx, step, macro) LOOP_UNROLLING_106(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_108(idx, step, macro) LOOP_UNROLLING_107(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_109(idx, step, macro) LOOP_UNROLLING_108(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_110(idx, step, macro) LOOP_UNROLLING_109(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_111(idx, step, macro) LOOP_UNROLLING_110(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_112(idx, step, macro) LOOP_UNROLLING_111(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_113(idx, step, macro) LOOP_UNROLLING_112(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_114(idx, step, macro) LOOP_UNROLLING_113(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_115(idx, step, macro) LOOP_UNROLLING_114(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_116(idx, step, macro) LOOP_UNROLLING_115(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_117(idx, step, macro) LOOP_UNROLLING_116(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_118(idx, step, macro) LOOP_UNROLLING_117(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_119(idx, step, macro) LOOP_UNROLLING_118(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_120(idx, step, macro) LOOP_UNROLLING_119(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_121(idx, step, macro) LOOP_UNROLLING_120(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_122(idx, step, macro) LOOP_UNROLLING_121(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_123(idx, step, macro) LOOP_UNROLLING_122(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_124(idx, step, macro) LOOP_UNROLLING_123(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_125(idx, step, macro) LOOP_UNROLLING_124(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_126(idx, step, macro) LOOP_UNROLLING_125(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_127(idx, step, macro) LOOP_UNROLLING_126(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_128(idx, step, macro) LOOP_UNROLLING_127(idx, step, macro); UNROLL_INCR(idx, step, macro)

#define LOOP_UNROLLING_STR(type, idx, start, step, num, macro) \
    {                                                          \
        type idx = start;                                      \
        LOOP_UNROLLING_##num(idx, step, macro);                \
    }

#define LOOP_UNROLLING(type, idx, start, step, num, macro) LOOP_UNROLLING_STR(type, idx, start, step, num, macro)

#if defined(LINEAR)
/** This OpenCL kernel performs the batch matrix multiplication (BatchMatMul): LHS non-transposed, RHS non-transposed - buffer only
 *
 * @note the "batch" here expresses the number of matrix multiplications to run in parallel. However, it
 *       should NOT be confused with the batch size of the model. For NHWC the "batch" is the "H" dimension
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=float)
 * @note The block's dimensions used for the LHS and RHS matrices (M0, N0 and K0) must be passed at compile time using -DN0, -DM0 and -DK0 (e.g. -DN0=8, -DM0=4, -DK0=4).
 * @note The fused activation function used should be passed with -DACTIVATION_TYPE, -DA_VAL and -DB_VAL are used for min and max output bounded activation functions.
 * @note The number of leftover outputs rows/columns must be passed using -DPARTIAL_STORE_N0 and -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_N0=2, -DPARTIAL_STORE_M0=3)
 * @note The dimension K must be passed at compile time using -DK (e.g. -DK=6)
 * @note The tensor type ("BUFFER" or "IMAGE") of the rhs tensor must be passed at compile time using -DRHS_TENSOR_TYPE (e.g. -DRHS_TENSOR_TYPE=BUFFER)
 * @note The kernel name in uppercase must be passed at compile time (e.g. -DMAT_MUL_NATIVE_NT_NT)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 > 0
 *  - N0 = 1, 2, 3, 4, 8, 16 (only 4, 8, 16 if RHS_TENSOR_TYPE=IMAGE)
 *  - K0 = 1, 2, 3, 4, 8, 16
 * @note Values > 8 for M0 are not expected to be efficient
 *
 * @param[in]  lhs_ptr                            Pointer to the lhs matrix. Supported data types: F32/F16
 * @param[in]  lhs_stride_y                       Stride of the lhs matrix in Y (2nd) dimension (in bytes)
 * @param[in]  lhs_stride_z                       Stride of the lhs tensor in Z (3rd) dimension (in bytes)
 * @param[in]  lhs_w                              The width of the lhs tensor
 * @param[in]  lhs_h                              The height of the lhs tensor
 * @param[in]  lhs_n                              Number of the matrices (buffers) in the batch
 * @param[in]  lhs_offset_first_element_in_bytes  The offset of the first element in the lhs matrix
 * @param[in]  rhs_img                            (Optional) Read only cl_image object for the rhs tensor. Included when RHS_TENSOR_TYPE=IMAGE
 * @param[in]  rhs_ptr                            Pointer to the rhs matrix. Supported data types: same as @p lhs_ptr
 * @param[in]  rhs_stride_y                       Stride of the rhs matrix in Y (2nd) dimension (in bytes)
 * @param[in]  rhs_stride_z                       Stride of the rhs tensor in Z (3rd) dimension (in bytes)
 * @param[in]  rhs_w                              The width of the rhs tensor
 * @param[in]  rhs_h                              The height of the rhs tensor
 * @param[in]  rhs_n                              Number of the matrices (buffers) in the batch
 * @param[in]  rhs_offset_first_element_in_bytes  The offset of the first element in the rhs matrix
 * @param[in]  bias_ptr                           (Optional) Pointer to the bias tensor. Supported data type: same as @p lhs_ptr
 * @param[in]  bias_stride_y                      (Optional) Stride of the bias tensor in Y dimension (in bytes)
 * @param[in]  bias_stride_z                      (Optional) Stride of the bias tensor in Z dimension (in bytes)
 * @param[in]  bias_w                             (Optional) The size of the width dimension of the bias tensor
 * @param[in]  bias_h                             (Optional) The size of the height dimension of the bias tensor
 * @param[in]  bias_n                             (Optional) The size of the depth dimension of the bias tensor
 * @param[in]  bias_offset_first_element_in_bytes (Optional) The offset of the first element in the bias tensor
 * @param[out] dst_ptr                            Pointer to the dst matrix. Supported data types: same as @p lhs_ptr
 * @param[in]  dst_stride_y                       Stride of the dst matrix in Y (2nd) dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the dst tensor in Z (3rd) dimension (in bytes)
 * @param[in]  dst_w                              The width of the dst tensor
 * @param[in]  dst_h                              The height of the dst tensor
 * @param[in]  dst_n                              Number of the matrices (buffers) in the batch
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the dst matrix
 */
__kernel void linear(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, RHS_TENSOR_TYPE),
#ifdef BIAS
    TENSOR3D_T(bias, BUFFER),
#endif // defined(BIAS)
    TENSOR3D_T(dst, BUFFER))
{
    const uint x = GET_SPATIAL_IDX(0, N0, PARTIAL_STORE_N0);
    const uint y = GET_SPATIAL_IDX(1, M0, PARTIAL_STORE_M0);
    const uint z = GET_SPATIAL_IDX(2, 1, 0);

    // Compute LHS/RHS/DST matrix address
    lhs_offset_first_element_in_bytes += y * lhs_stride_y + z * lhs_stride_z;
    dst_offset_first_element_in_bytes += x * sizeof(DATA_TYPE) + y * dst_stride_y + z * dst_stride_z;

    // Initialize the accumulators
    TILE(DATA_TYPE, M0, N0, acc);

    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        acc[i].v = 0.f;
    });

    const int rhs_z = z * rhs_h;
    int       k;
    for(k = 0; k <= K - K0; k += K0)
    {
        TILE(DATA_TYPE, M0, K0, a);
        TILE(DATA_TYPE, K0, N0, b);

        LOOP_UNROLLING(int, i, 0, 1, M0,
        {
            a[i].v = 0.f;
        });

        LOOP_UNROLLING(int, i, 0, 1, K0,
        {
            b[i].v = 0.f;
        });

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, M0, K0, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, K0, N0, RHS_TENSOR_TYPE, rhs, x, k + rhs_z, 1, rhs_stride_y, b);

        T_MMUL(DATA_TYPE, DATA_TYPE, DATA_TYPE, M0, N0, K0, NT, NT, a, b, acc);

        lhs_offset_first_element_in_bytes += K0 * sizeof(DATA_TYPE);
    }

#if K % K0 != 0
    /* Leftover Loop */
    for(; k < K; ++k)
    {
        TILE(DATA_TYPE, M0, 1, a);
        TILE(DATA_TYPE, 1, N0, b);

        LOOP_UNROLLING(int, i, 0, 1, M0,
        {
            a[i].v = 0.f;
        });

        LOOP_UNROLLING(int, i, 0, 1, 1,
        {
            b[i].v = 0.f;
        });

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, M0, 1, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, 1, N0, BUFFER, rhs, x, k + rhs_z, 1, rhs_stride_y, b);

        T_MMUL(DATA_TYPE, DATA_TYPE, DATA_TYPE, M0, N0, 1, NT, NT, a, b, acc);

        lhs_offset_first_element_in_bytes += 1 * sizeof(DATA_TYPE);
    }
#endif // K % K0 != 0

    const bool x_cond = PARTIAL_STORE_N0 != 0 && get_global_id(0) == 0;
    const bool y_cond = PARTIAL_STORE_M0 != 0 && get_global_id(1) == 0;

    TILE(int, M0, 1, indirect_buffer);
    LOOP_UNROLLING(int, _i, 0, 1, M0,
    {
        indirect_buffer[_i].v = min(_i, select(M0 - 1, PARTIAL_STORE_M0 - 1, y_cond));
    });

#ifdef BIAS
    perform_bias_addition(bias_ptr, bias_offset_first_element_in_bytes, acc, x);
#endif // defined(BIAS)

    T_ACTIVATION(DATA_TYPE, M0, N0, ACTIVATION_TYPE, A_VAL, B_VAL, acc, acc);

    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, M0, N0, PARTIAL_STORE_N0, BUFFER, dst, 0, dst_stride_y, x_cond, acc, indirect_buffer);
}
#endif // defined(LINEAR)