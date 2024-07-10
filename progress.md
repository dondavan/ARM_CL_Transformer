Vanilla_Transformer
|
|- Graph Representation ----> examples/graph_vanilla_transformer.cpp
|
|
|- Graph -------------------> arm_compute/graph/Types.h Add support layer type enum
|        |
|        |------------------> arm_compute/graph/frontend/Layers.h
|        |                                               |-> TokenEmbeddingLayer
|        |                                               |-> PositionalEncodingLayer(Add sequence information)
|        |                                                   TODO : moveable? data type
|        |                                               |-> MultiHeadAttentionLayer(Wraps over Scale Dot Production)
|        |                                               |-> ScaleDotProductionAttentionLayer
|        |                                               |-> LayerNormLayer
|        |                                               |-> FeedForwardLayer
|        |
|        |-----------------> arm_compute/graph/GraphBuilder.h
|                                              |-> add_tkemb_node
|
|
|- Nodes ------------------> arm_compute/graph/nodes/nodes.h: Include all nodes 
|        |
|        | ----------------> arm_compute/graph/nodes/TokenEmbeddingLayerNode.h
|        |                                     |--> /PositionalEncodingNode.h
|        |                                     |--> /MultiHeadAttentionNode.h
|        |                                     |--> /ScaleDotProductionAttentionNode.h
|        |                                     |--> /LinearLayerNode.h
|        |                                     |--> /SimpleForwardLayerNode.h: TODO forward_descriptors
|        |                                     |--> /LayerNormNode.h
|        |                                     |--> /FeedForwardNode.h
|        |
|        |
|        |-----------------> src/graph/nodes/TokenEmbeddingNode.cpp
|                                      |--> /PositionalEncodingNode.cpp
|                                      |--> /MultiHeadAttentionNode.cpp
|                                      |--> /ScaleDotProductionAttentionNode.cpp
|                                      |     _input_edges.resize(1, EmptyEdgeID);
|                                      |     _outputs.resize(1, NullTensorID);
|                                      |
|                                      |--> /SimpleForwardLayerNode.cpp
|                                      |--> /LinearLayerNode.cpp
|                                      |--> /LayerNormNode.cpp
|                                      |--> /FeedForwardNode.cpp
|
|
|- Function ----------------> src/graph/backends/NEON/NEFunctionFactory.cpp
|           |                                         |-> NodeType::TokenEmbeddingLayer
|           |
|           |
|           |---------------> arm_compute/runtime/NEON/functions/NETokenEmbeddingLayer.h
|           |                                          |------> /NEPositionalEncodingLayer.h
|           |                                          |------> /NEMultiHeadAttentionLayer.h
|           |                                          |------> /NEScaleDotProductionAttentionLayer.h
|           |                                          |------> /NELayerNormLayer.h
|           |                                          |------> /NEFeedForwardLayer.h
|           |
|           |
|           |---------------> src/runtime/NEON/functions/NETokenEmbeddingLayer.cpp
|                                              |         |-> TODO: validate
|                                              |------> /NEPositionalEncodingLayer.cpp
|                                              |------> /NEMultiHeadAttentionLayer.cpp
|                                              |------> /NEScaleDotProductionAttentionLayer.cpp
|                                              |------> /NELayerNormLayer.cpp
|                                              |------> /NEFeedForwardLayer.cpp
|
|
|- Core --------------------> arm_compute/core/Types.h
|       |                                      |-> TokenEmbeddingLayerInfo
|       |                                      |-> PositionalEncodingLayerInfo
|       |                                      |-> MultiHeadAttentionLayerInfo
|       |                                      |-> ScaleDotProductionAttentionLayerInfo
|       |                                      |-> LayerNormLayerInfo
|       |                                      |-> FeedForwardLayerInfo
|       |
|       |-------------------> arm_compute/core/CoreTypes.h
|       |                                      |-> TextFormat: utf-8
|       |
|       |-------------------> arm_compute/core/TensorInfo.h
|       |                                      |-> TensorInfo: 1D tensor info for text input 
|       |
|       |-------------------> src/core/TensorInfo.cpp
|       |                              |-> TensorInfo: wrapper over init
|       |                              |-> init
|       |
|       |-------------------> arm_compute/core/utils/DataTypeUtils.h
|                                                    |-> data_type_from_format: configure tensor data type
|
|
|- Operator ----------------> src/cpu/operators/CpuTokenEmbed.h.cpp
|
|
|- Kernel ------------------> Token Embedding
|         |                   |-src/cpu/kernels/CpuTokenEmbedKernel.h.cpp -> Have been replaced by CpuVectorizeKernel.h.cpp
|         |                   |                 |-> using dst::datatype for kernel selection.
|         |                   |                     #TODO: data compability
|         |                   |-src/cpu/kernels/tokenembed/generic/neon/fp32.cpp
|         |                                                             |-> Improve kernel using intrinsics
|         |
|         |-----------------> Linear Layer
|         |                   |-src/cpu/kernels/CpuLinearKernel.h.cpp
|         |                                           |-> Interface for the kernel to perform linear operation 
|         |                   
|         |
|         |-----------------> src/cpu/kernels/CpuKernelSelectionTypes.h:
|                                             |->TokenEmbedKernelDataTypeISASelectorData & Ptr: 
|                                                For selecting kernel implmentation
|           
|
|- Utils -------------------> utils/GraphUtils.h
         |                          |-> TextAccessor
         |                          |-> get_input_accessor : add txt reader
         |                          |-> WordPiecePreprocessor: constructor, preprcessor, preprocessor_typed
         |------------------> utils/GraphUtils.cpp
         |                          |-> TextAccessor
         |                          |-> get_input_accessor : add txt reader
         |                          |-> WordPiecePreprocessor: constructor, preprcessor, preprocessor_typed     
         |                              TODO: Currenting read file in using unsigned char, but processing using F32.
         |                                      Datatype mismatch.
         |                              TODO: Implement maximum input length(throughout), and implement padding. 
         |                                  Currently manual add space.
         |
         |
         |------------------> utils/TextLoader.h
         |                          |-> TextDataFeeder
         |                          |-> TXTLoader
         |                          |-> TextLoaderFactory
         |                          |-> ITextLoader   
         |
         |
         |------------------> utils/CommonGraphOptions.h
         |                          |-> add text type
         |
         |------------------> utils/Utils.h
         |                          |-> TextType
         |                          |-> parse_txt_header: TO BE USE?
         |                          |-> get_text_type_from_file
         |                              Note: NPY loader reinterpert char
         |------------------> utils/Utils.cpp
                                    |-> parse_txt_header: TO BE USE?
                                    |-> get_text_type_from_file: TODO: now just return Default UTF-8


Program structure

Input
  |-->WordPiecePreprocessor: Word token to numerical representation
  |-->atoiPreprocessor: Get sentence segmentation and convert into numerical
  |
Embedding
  |-->TokenEmbeddingLayerNode --> cpu::CpuTokenEmbed --> kernels::CpuVectorizeKernel : Numerical token into pretrained vector
  |                                                  |-> kernels::CpuPositionalEncodingKernel : Compute positional encoding 
  |
  |-->SegmentEmbeddingLayerNode --> cpu::CpuSegmentEmbed --> kernels::CpuVectorizeKernel : Segemnt token into pretrained vector
  |
  |-->EltwiseLayerNode --> EltwiseOperation::Add: Sum all three token embedding, segemnt embedding and positional embedding
  |
Linear
  |--> LinearLayerNode: Input vector * value weight(pre-trained) + value bias(pre-trained)
  |--> LinearLayerNode: Input vector * key weight(pre-trained) + key bias(pre-trained)
  |--> LinearLayerNode: Input vector * query weight(pre-trained) + query bias(pre-trained)
  |
  |--> SimpleForwardLayerNode : hold all three above output tensor in one node
  |
Attention
  |
  |


Potential problem:
            1.utils/GraphUtils.cpp: Text Preprocess input/output, configure, runtime tensor shape may mismatch
              src/cpu/kernels/tokenembed/generic/neon/fp32.cpp: neon_token_embed_char_2_float32:
                  (const unsigned int window_end_x     = src->info()->tensor_shape().x();)
          
Compatability:
            1: All function only support NEON right now.
            2. Input only support UTF-8 encoding (U8) input
            3. Interpret token_embedding npy in float right now 
            4. data layout ND
            5. Linear Layer Now only have "src/core/NEON/kernels" implementation
            6. Update arm_compute/graph/nodes/NodesFwd.h
            7. Only support lower case input eg. "I, i" 1045 
            8. Add ARM_COMPUTE_LOG_GRAPH_INFO to FunctionHelpers.h

Functionality:
            1. Segment token, currently only support 1 sentence input
            2. Token vectorize
            3. Postion Embedding from pretained model dont really need src input :
                        src/cpu/kernels/CpuPositionEmbeddingKernel.cpp
            4. Pytorch positional embdding is implemented using pretrained model, but this calcualtes.
            5. Deallocate Simple forward original output tensor
            6. Potential: src/cpu/operators/CpuScaleDotProduction.cpp Run tensor pack re indexed
            7. CpuGemmMatrixMultiplyKernel Requires 4*4transpose and 1w transpose
            8. src/cpu/operators/CpuScaleDotProduction.cpp: {ACL_SRC, const_cast<const ITensor*>(scaled_output.get())} causes 
                    free(): invalid next size (normal) Aborted
            9. There is difference from pytroch bert, which has at the end of embedding
                      embeddings = self.LayerNorm(embeddings)
                      embeddings = self.dropout(embeddings)
                while out inplementation does not
            10. segemnt embedding using verctorize kernel, produeces (*,2) shape, should be (*,1)


Optimization: 
            1. window collapse
            2. Every kernel
            3. GEMM run optimized


Input                                                           char U8                          (Len_seq, ...)
  |             
utils/GraphUtils.cpp(preprocess)                            U8 -> unsigned int                  (Len_seq*, ...) *reshape
  |
Token Embedding                                            unsigned int -> FP32                 (Len_seq*, d_model, ...)
  |
Query,Key,Value                                               FP32 -> FP32                      (Len_seq*, d_model, ...)
  |
Scale Dot Production                                          FP32 -> FP32


output_descriptor shape:  13 768 1 1 1 1
DataLayout: NCHW
DataType::F32

Tensor Shape

Input                               (Len_seq, ...)
Vocabulary                          (d_vocab, d_model, ...)
Query,Key,Value Weight              (d_model, d_model, ...)
Query,Key,Value Bias                (1, d_model, ...)


Pytroch modificationL:
Embedding:
embeddings = self.LayerNorm(embeddings)
embeddings = self.dropout(embeddings)

Self Attention:
attention_probs = self.dropout(attention_probs)

BertSelfOutput
hidden_states = self.dense(hidden_states)
hidden_states = self.dropout(hidden_states)

Layer Norm class LayerNorm(Module):
self.elementwise_affine = elementwise_affine -> False

Intermedia 

Output
hidden_states = self.dropout(hidden_states)
hidden_states = self.LayerNorm(hidden_states + input_tensor)




Original modified: 
src/cpu/kernels/CpuGemmMatrixMultiplyKernel.cpp : ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(rhs, &tensor_info_reshaped1);


# CL Support

# Modified on original
arm_compute/core/CL/CLDevice.h  
      is_non_uniform_workgroup_supported
src/gpu/cl/kernels/ClElementwiseKernel.cpp::configure_window_arithmetic_commonadd 
      boradcast z for dst 

## Layer
NETokenEmbeddingLayer, NESegmentEmbeddingLayer, NEPositionEmbeddingLayer -> ?
NEEmbeddingSumLayer -> CLEmbeddingSumLayer
NELayerNormLayer -> CLLayerNormLayer
NEScaleDotProductionAttentionLayer -> CLScaleDotProductionAttentionLayer
NESimpleForwardLayer -> CLSimpleForwardLayer
NELinearLayer -> ClLinearLayer
NEEltwiseFunctions -> ClEltwiseFunctions

## Operation
cpu::CpuTokenEmbed
  ->kernels::CpuVectorizeKernel

cpu::CpuSegmentEmbed
  ->kernels::CpuVectorizeKernel

cpu::CpuPositionEmbed
  ->kernels::CpuPositionEmbeddingKernel

cpu::CpuEmbedSum
  ->kernels::CpuAddKernel -> ClElementwiseKernel

NESimpleForwardLayer : copy tensor

cpu::CpuLayerNorm
  ->kernels::CpuLayerNormKernel (This is 100% missing)

cpu::CpuScaleDotProduction
  ->CpuPermute -> ClPermuteKernel
  ->kernels::CpuReshapeKernel -> ClReshapeKernel          |
  ->cpu::kernels::CpuGemmInterleave4x4Kernel              |
  ->cpu::kernels::CpuGemmTranspose1xWKernel               | These 4 are for matrix multiplcation
  ->cpu::kernels::CpuGemmMatrixMultiplyKernel             |

cpu::CpuLinear
  ->CpuTranspose -> ClTransposeKernel                     |
  ->cpu::kernels::CpuGemmMatrixMultiplyKernel             |
  ->cpu::kernels::CpuGemmInterleave4x4Kernel              | These 4 are for matrix multiplcation
  ->cpu::kernels::CpuGemmTranspose1xWKernel               |
  ->cpu::kernels::CpuAddVecKernel -> ClElementwiseKernel

cpu::CpuAdd
  ->kernels::CpuAddKernel -> ClElementwiseKernel


src/runtime/CL/CLScheduler.cpp

src/core/CL/CLHelpers.cpp

src/core/CL/CLCompileContext.cpp


Linear

a->tensor_shape().x() 768
a->tensor_shape().y() 7
a->tensor_shape().z() 1
b->tensor_shape().x() 768
b->tensor_shape().y() 768
b->tensor_shape().z() 1
c->tensor_shape().x() 768
c->tensor_shape().y() 1
c->tensor_shape().z() 1
d->tensor_shape().x() 768
d->tensor_shape().y() 7
d->tensor_shape().z() 1

a->tensor_shape().x() 768
a->tensor_shape().y() 7
a->tensor_shape().z() 1
b->tensor_shape().x() 768
b->tensor_shape().y() 3072
b->tensor_shape().z() 1
c->tensor_shape().x() 3072
c->tensor_shape().y() 1
c->tensor_shape().z() 1
d->tensor_shape().x() 768
d->tensor_shape().y() 7
d->tensor_shape().z() 1


Permute
lhs->info().x() 256
lhs->info().y() 2
lhs->info().z() 12
rhs->tensor_shape().x() 256
rhs->tensor_shape().y() 2
rhs->tensor_shape().z() 12
dst->tensor_shape().x() 7
dst->tensor_shape().y() 7
dst->tensor_shape().z() 12
lhs->info().x() 28
lhs->info().y() 2
lhs->info().z() 12
rhs->tensor_shape().x() 64
rhs->tensor_shape().y() 7
rhs->tensor_shape().z() 12
dst->tensor_shape().x() 64
dst->tensor_shape().y() 7
dst->tensor_shape().z() 12




[ 3.156732e-02, -4.113317e-02, -5.644031e-02,  ..., 2.102060e-03, 4.447279e-03, 2.194647e-02, ] 
[ -1.289720e-02, 1.915356e-02, -3.366558e-02,  ..., 3.856649e-02, 3.858358e-02, -6.709838e-03, ] 
[ -5.459049e-02, -6.004878e-03, -4.968532e-03,  ..., -1.994273e-02, 6.275434e-02, -6.078227e-02, ] 
[ 1.144635e-02, 7.319955e-03, -1.315639e-02,  ..., 1.274683e-02, -3.624680e-03, 8.325529e-03, ] 
[ 6.947308e-04, -1.461485e-02, -3.047169e-02,  ..., 1.212882e-02, 4.256184e-02, 3.591327e-02, ] 
[ -2.340063e-02, -9.955178e-03, -2.701394e-02,  ..., 1.355235e-02, 3.684345e-02, 2.063839e-02, ] 
[ -7.663916e-03, -1.663753e-02, -1.233587e-02,  ..., -3.097108e-02, 1.237562e-02, -6.427724e-03, ]


/*
 * Copyright (c) 2023 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "helpers.h"
#include "tile_helpers.h"

#ifdef BIAS
// This function performs in-place bias addition for float and half datatypes when bias is enabled.
// Note The tile's dimensions used for the LHS and RHS matrices (M0, N0) must be passed at compile time using -DN0, -DM0 (e.g. -DN0=8, -DM0=4).
inline void perform_bias_addition(uchar *bias_ptr, uint bias_offset_first_element_in_bytes,  uint x)
{
    TILE(DATA_TYPE, 1, N0, bias_tile);

    // below expands to use bias_ptr and bias_offset_first_element_in_bytes
    T_LOAD(DATA_TYPE, 1, N0, BUFFER, bias, x, 0, 1, 0, bias_tile);

    // c = c + bias[broadcasted]
    //T_ELTWISE_BROADCAST_ADD_X(DATA_TYPE, M0, N0, acc, bias_tile, acc);
}
#endif // defined(BIAS)

#if defined(MAT_MUL_MMUL_HUGH)
/** This OpenCL kernel performs the batch matrix multiplication (BatchMatMul) using MMUL: LHS non-transposed, RHS non-transposed - buffer only
 *
 * @note the "batch" here expresses the number of matrix multiplications to run in parallel. However, it
 *       should NOT be confused with the batch size of the model. For NHWC the "batch" is the "H" dimension
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=float)
 * @note The tile's dimensions used for the LHS and RHS matrices (M0, N0 and K0) must be passed at compile time using -DN0, -DM0 and -DK0 (e.g. -DN0=8, -DM0=4, -DK0=1).
 * @note The number of leftover outputs rows/columns must be passed using -DN0_LEFTOVER and -DM0_LEFTOVER (e.g. -DN0_LEFTOVER=2, -DM0_LEFTOVER=3)
 * @note The MMUL block dimension (MMUL_M0, MMUL_N0, MMUL_K0) must be passed at compile time using -DMMUL_M0, -DMMUL_N0 and -DMMUL_K0 (e.g. -DMMUL_M0=4, -DMMUL_N0=4, -DMMUL_K0=4).
 * @note The kernel name in uppercase must be passed at compile time (e.g. -DMAT_MUL_NATIVE_MMUL_NT_NT)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 > 0
 *  - N0 = 1, 2, 3, 4, 8, 16
 *  - K0 = 1
 * @note Values > 8 for M0 are not expected to be efficient
 *
 * @param[in]  lhs_ptr                            Pointer to the lhs matrix. Supported data types: F32/F16
 * @param[in]  lhs_stride_y                       Stride of the lhs matrix in Y (2nd) dimension (in bytes)
 * @param[in]  lhs_stride_z                       Stride of the lhs tensor in Z (3rd) dimension (in bytes)
 * @param[in]  lhs_w                              The width of the lhs tensor
 * @param[in]  lhs_h                              The height of the lhs tensor
 * @param[in]  lhs_n                              Number of the matrices (buffers) in the batch
 * @param[in]  lhs_offset_first_element_in_bytes  The offset of the first element in the lhs matrix
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
 * @param[in]  M                                  Number of rows in LHS matrix
 * @param[in]  N                                  Number of columns in RHS matrix
 * @param[in]  K                                  Number of columns in LHS matrix and rows in RHS matrix, which is multiple of MMUL_K0.
 */
 //mat_mul_native_mmul_nt_nt
__kernel void mat_mul_mmul_hugh(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, RHS_TENSOR_TYPE),
#ifdef BIAS
    TENSOR3D_T(bias, BUFFER),
#endif // defined(BIAS)
    TENSOR3D_T(dst, BUFFER))
{
    uint x = GET_SPATIAL_IDX(0, N0, PARTIAL_STORE_N0);
    uint y = GET_SPATIAL_IDX(1, M0, PARTIAL_STORE_M0);
    uint z = GET_SPATIAL_IDX(2, 1, 0);
    
    // Compute LHS/RHS/DST matrix address
    lhs_offset_first_element_in_bytes += y * lhs_stride_y + z * lhs_stride_z;
    dst_offset_first_element_in_bytes += x * sizeof(DATA_TYPE) + y * dst_stride_y + z * dst_stride_z;

    // Initialize the accumulators
    TILE(DATA_TYPE, M0, N0, acc);

    //T_LOAD(DATA_TYPE, M0, N0, BUFFER, lhs, x, 0, 1, lhs_stride_y, acc);
    
    /*
    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        acc[i].v = x;
    })
    

    LOOP_UNROLLING(int, _m, 0, 1, M0,
    {
        LOOP_UNROLLING(int, _n, 0, 1, N0,
        {
            acc[_m].s[_n] = 1.f;
        })
    })*/

    for(int _m = 0; _m < M0; _m++)
    {
        acc[_m].s[0] = 0.f;
        acc[_m].s[1] = 0.f;
        acc[_m].v = 0.f;
    }

    const int rhs_z = z * rhs_h;
    int       k;
    for(k = 0; k <= K - K0; k += K0)
    {
        
        /*
        LOOP_UNROLLING(int, _m, 0, 1, M0,
        {
            LOOP_UNROLLING(int, _k, 0, 1, K0,
            {
                a[_m].s[_k] = 1.f;
            })
        })
        LOOP_UNROLLING(int, _n, 0, 1, N0,
        {
            LOOP_UNROLLING(int, _k, 0, 1, K0,
            {
                b[_n].s[_k] = 1.f;
            })
        })*/
 
        TILE(DATA_TYPE, M0, K0, a);
        TILE(DATA_TYPE, N0, K0, b);

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, M0, K0, BUFFER, lhs, k, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, N0, K0, RHS_TENSOR_TYPE, rhs, k, x + rhs_z, 1, rhs_stride_y, b);

        for(int _m = 0; _m < M0; _m++)
        {
            a[_m].s[0] = a[_m].v.s0;
            a[_m].s[1] = a[_m].v.s1;
            a[_m].s[2] = a[_m].v.s2;
            a[_m].s[3] = a[_m].v.s3;

            a[_m].s[4] = a[_m].v.s4;
            a[_m].s[5] = a[_m].v.s5;
            a[_m].s[6] = a[_m].v.s6;
            a[_m].s[7] = a[_m].v.s7;
        }

        for(int _n = 0; _n < N0; _n++)
        {
            b[_n].s[0] = b[_n].v.s0;
            b[_n].s[1] = b[_n].v.s1;
            b[_n].s[2] = b[_n].v.s2;
            b[_n].s[3] = b[_n].v.s3;

            b[_n].s[4] = b[_n].v.s4;
            b[_n].s[5] = b[_n].v.s5;
            b[_n].s[6] = b[_n].v.s6;
            b[_n].s[7] = b[_n].v.s7;
        }


        //T_MMUL(DATA_TYPE, DATA_TYPE, DATA_TYPE, M0, N0, K0, NT, T, a, b, acc);
        
        /*
        LOOP_UNROLLING(int, _m, 0, 1, M0,
        {
            LOOP_UNROLLING(int, _n, 0, 1, N0,
            {
                LOOP_UNROLLING(int, _k, 0, 1, K0,
                {
                    acc[_m].s[_n] = fma((DATA_TYPE)(a[_m].s[_k]), (DATA_TYPE)(b[_n].s[_k]), acc[_m].s[_n]);
                })
            })
        }) */
        
        /*
        LOOP_UNROLLING(int, _m, 0, 1, M0,
        {
            acc[_m].s[0] = fma((DATA_TYPE)(a[_m].s[0]), (DATA_TYPE)(b[0].s[0]), acc[_m].s[0]);
            acc[_m].s[0] = fma((DATA_TYPE)(a[_m].s[1]), (DATA_TYPE)(b[0].s[1]), acc[_m].s[0]);
            acc[_m].s[0] = fma((DATA_TYPE)(a[_m].s[2]), (DATA_TYPE)(b[0].s[2]), acc[_m].s[0]);
            acc[_m].s[0] = fma((DATA_TYPE)(a[_m].s[3]), (DATA_TYPE)(b[0].s[3]), acc[_m].s[0]);
            acc[_m].s[0] = fma((DATA_TYPE)(a[_m].s[4]), (DATA_TYPE)(b[0].s[4]), acc[_m].s[0]);
            acc[_m].s[0] = fma((DATA_TYPE)(a[_m].s[5]), (DATA_TYPE)(b[0].s[5]), acc[_m].s[0]);
            acc[_m].s[0] = fma((DATA_TYPE)(a[_m].s[6]), (DATA_TYPE)(b[0].s[6]), acc[_m].s[0]);
            acc[_m].s[0] = fma((DATA_TYPE)(a[_m].s[7]), (DATA_TYPE)(b[0].s[7]), acc[_m].s[0]);

            acc[_m].s[1] = fma((DATA_TYPE)(a[_m].s[0]), (DATA_TYPE)(b[1].s[0]), acc[_m].s[1]);
            acc[_m].s[1] = fma((DATA_TYPE)(a[_m].s[1]), (DATA_TYPE)(b[1].s[1]), acc[_m].s[1]);
            acc[_m].s[1] = fma((DATA_TYPE)(a[_m].s[2]), (DATA_TYPE)(b[1].s[2]), acc[_m].s[1]);
            acc[_m].s[1] = fma((DATA_TYPE)(a[_m].s[3]), (DATA_TYPE)(b[1].s[3]), acc[_m].s[1]);
            acc[_m].s[1] = fma((DATA_TYPE)(a[_m].s[4]), (DATA_TYPE)(b[1].s[4]), acc[_m].s[1]);
            acc[_m].s[1] = fma((DATA_TYPE)(a[_m].s[5]), (DATA_TYPE)(b[1].s[5]), acc[_m].s[1]);
            acc[_m].s[1] = fma((DATA_TYPE)(a[_m].s[6]), (DATA_TYPE)(b[1].s[6]), acc[_m].s[1]);
            acc[_m].s[1] = fma((DATA_TYPE)(a[_m].s[7]), (DATA_TYPE)(b[1].s[7]), acc[_m].s[1]);

        }) 
        
        LOOP_UNROLLING(int, _m, 0, 1, M0,
        {
            acc[_m].s[0] = fma((DATA_TYPE)(a[_m].s[0]), (DATA_TYPE)(b[0].s[0]), acc[_m].s[0]);
            acc[_m].s[0] = fma((DATA_TYPE)(a[_m].s[1]), (DATA_TYPE)(b[0].s[1]), acc[_m].s[0]);
            acc[_m].s[0] = fma((DATA_TYPE)(a[_m].s[2]), (DATA_TYPE)(b[0].s[2]), acc[_m].s[0]);
            acc[_m].s[0] = fma((DATA_TYPE)(a[_m].s[3]), (DATA_TYPE)(b[0].s[3]), acc[_m].s[0]);
            acc[_m].s[0] = fma((DATA_TYPE)(a[_m].s[4]), (DATA_TYPE)(b[0].s[4]), acc[_m].s[0]);
            acc[_m].s[0] = fma((DATA_TYPE)(a[_m].s[5]), (DATA_TYPE)(b[0].s[5]), acc[_m].s[0]);
            acc[_m].s[0] = fma((DATA_TYPE)(a[_m].s[6]), (DATA_TYPE)(b[0].s[6]), acc[_m].s[0]);
            acc[_m].s[0] = fma((DATA_TYPE)(a[_m].s[7]), (DATA_TYPE)(b[0].s[7]), acc[_m].s[0]);

            acc[_m].s[1] = fma((DATA_TYPE)(a[_m].s[0]), (DATA_TYPE)(b[1].s[0]), acc[_m].s[1]);
            acc[_m].s[1] = fma((DATA_TYPE)(a[_m].s[1]), (DATA_TYPE)(b[1].s[1]), acc[_m].s[1]);
            acc[_m].s[1] = fma((DATA_TYPE)(a[_m].s[2]), (DATA_TYPE)(b[1].s[2]), acc[_m].s[1]);
            acc[_m].s[1] = fma((DATA_TYPE)(a[_m].s[3]), (DATA_TYPE)(b[1].s[3]), acc[_m].s[1]);
            acc[_m].s[1] = fma((DATA_TYPE)(a[_m].s[4]), (DATA_TYPE)(b[1].s[4]), acc[_m].s[1]);
            acc[_m].s[1] = fma((DATA_TYPE)(a[_m].s[5]), (DATA_TYPE)(b[1].s[5]), acc[_m].s[1]);
            acc[_m].s[1] = fma((DATA_TYPE)(a[_m].s[6]), (DATA_TYPE)(b[1].s[6]), acc[_m].s[1]);
            acc[_m].s[1] = fma((DATA_TYPE)(a[_m].s[7]), (DATA_TYPE)(b[1].s[7]), acc[_m].s[1]);

        }) */
        
        //LOOP_UNROLLING_HUGH(int, caonima, 0, 1, M0,LOOP_UNROLLING_HUGH(int, nimasile, 0, 1, N0, acc[caonima].s[nimasile] = fma((DATA_TYPE)(a[caonima].s[0]), (DATA_TYPE)(b[nimasile].s[0]), acc[caonima].s[nimasile]);acc[caonima].s[nimasile] = fma((DATA_TYPE)(a[caonima].s[1]), (DATA_TYPE)(b[nimasile].s[1]), acc[caonima].s[nimasile]);acc[caonima].s[nimasile] = fma((DATA_TYPE)(a[caonima].s[2]), (DATA_TYPE)(b[nimasile].s[2]), acc[caonima].s[nimasile]);acc[caonima].s[nimasile] = fma((DATA_TYPE)(a[caonima].s[3]), (DATA_TYPE)(b[nimasile].s[3]), acc[caonima].s[nimasile]);acc[caonima].s[nimasile] = fma((DATA_TYPE)(a[caonima].s[4]), (DATA_TYPE)(b[nimasile].s[4]), acc[caonima].s[nimasile]);acc[caonima].s[nimasile] = fma((DATA_TYPE)(a[caonima].s[5]), (DATA_TYPE)(b[nimasile].s[5]), acc[caonima].s[nimasile]);acc[caonima].s[nimasile] = fma((DATA_TYPE)(a[caonima].s[6]), (DATA_TYPE)(b[nimasile].s[6]), acc[caonima].s[nimasile]);acc[caonima].s[nimasile] = fma((DATA_TYPE)(a[caonima].s[7]), (DATA_TYPE)(b[nimasile].s[7]), acc[caonima].s[nimasile]);))
        for(int _m = 0; _m < M0; _m++)
        {
            for(int _nimasile = 0; _nimasile < N0; _nimasile++)
            {
                acc[_m].s[_nimasile] = fma((DATA_TYPE)(a[_m].s[0]), (DATA_TYPE)(b[_nimasile].s[0]), acc[_m].s[_nimasile]);
            }
        }
        /*
        LOOP_UNROLLING(int, _m, 0, 1, M0,
        {
            LOOP_UNROLLING(int, _k, 0, 1, K0,
            {
                acc[_m].s[_n] = fma((DATA_TYPE)(a[_m].s[_k]), (DATA_TYPE)(b[_n].s[_k]), acc[_m].s[_n]);
            })

        })
        */
        /*
        TILE(DATA_TYPE, M0, K0, a);
        TILE(DATA_TYPE, K0, N0, b);

        T_LOAD(DATA_TYPE, M0, K0, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, K0, N0, RHS_TENSOR_TYPE, rhs, x, k + rhs_z, 1, rhs_stride_y, b);

        for(int _m = 0; _m < M0; _m++)
        {
            a[_m].s[0] = a[_m].v.s0;
            a[_m].s[1] = a[_m].v.s1;
            a[_m].s[2] = a[_m].v.s2;
            a[_m].s[3] = a[_m].v.s3;

            a[_m].s[4] = a[_m].v.s4;
            a[_m].s[5] = a[_m].v.s5;
            a[_m].s[6] = a[_m].v.s6;
            a[_m].s[7] = a[_m].v.s7;
        }

        for(int _k = 0; _k < K0; _k++)
        {
            b[_k].s[0] = b[_k].v.s0;
            b[_k].s[1] = b[_k].v.s1;
        }

        LOOP_UNROLLING(int, _m, 0, 1, M0,
        {
            acc[_m].s[0] = fma((DATA_TYPE)(a[_m].s[0]), (b[0].s[0]), acc[_m].s[0]);
            acc[_m].s[1] = fma((DATA_TYPE)(a[_m].s[0]), (b[0].s[1]), acc[_m].s[1]);
            acc[_m].s[0] = fma((DATA_TYPE)(a[_m].s[1]), (b[1].s[0]), acc[_m].s[0]);
            acc[_m].s[1] = fma((DATA_TYPE)(a[_m].s[1]), (b[1].s[1]), acc[_m].s[1]);
            acc[_m].s[0] = fma((DATA_TYPE)(a[_m].s[2]), (b[2].s[0]), acc[_m].s[0]);
            acc[_m].s[1] = fma((DATA_TYPE)(a[_m].s[2]), (b[2].s[1]), acc[_m].s[1]);
            acc[_m].s[0] = fma((DATA_TYPE)(a[_m].s[3]), (b[3].s[0]), acc[_m].s[0]);
            acc[_m].s[1] = fma((DATA_TYPE)(a[_m].s[3]), (b[3].s[1]), acc[_m].s[1]);
            
            acc[_m].s[0] = fma((DATA_TYPE)(a[_m].s[4]), (b[4].s[0]), acc[_m].s[0]);
            acc[_m].s[1] = fma((DATA_TYPE)(a[_m].s[4]), (b[4].s[1]), acc[_m].s[1]);
            acc[_m].s[0] = fma((DATA_TYPE)(a[_m].s[5]), (b[5].s[0]), acc[_m].s[0]);
            acc[_m].s[1] = fma((DATA_TYPE)(a[_m].s[5]), (b[5].s[1]), acc[_m].s[1]);
            acc[_m].s[0] = fma((DATA_TYPE)(a[_m].s[6]), (b[6].s[0]), acc[_m].s[0]);
            acc[_m].s[1] = fma((DATA_TYPE)(a[_m].s[6]), (b[6].s[1]), acc[_m].s[1]);
            acc[_m].s[0] = fma((DATA_TYPE)(a[_m].s[7]), (b[7].s[0]), acc[_m].s[0]);
            acc[_m].s[1] = fma((DATA_TYPE)(a[_m].s[7]), (b[7].s[1]), acc[_m].s[1]);
        })   
        */
        
        
        //lhs_offset_first_element_in_bytes += K0 * sizeof(DATA_TYPE);
    }

    const bool x_cond = PARTIAL_STORE_N0 != 0 && get_global_id(0) == 0;
    const bool y_cond = PARTIAL_STORE_M0 != 0 && get_global_id(1) == 0;

    TILE(int, M0, 1, indirect_buffer);
    LOOP_UNROLLING(int, _i, 0, 1, M0,
    {
        indirect_buffer[_i].v = min(_i, select(M0 - 1, PARTIAL_STORE_M0 - 1, y_cond));
    });

#ifdef BIAS
    TILE(DATA_TYPE, 1, N0, bias_tile);

    // below expands to use bias_ptr and bias_offset_first_element_in_bytes
    T_LOAD(DATA_TYPE, 1, N0, BUFFER, bias, x, 0, 1, 0, bias_tile);

    bias_tile[0].s[0] = bias_tile[0].v.s0;
    bias_tile[0].s[1] = bias_tile[0].v.s1;

    LOOP_UNROLLING(int, _m, 0, 1, M0,
    {
        acc[_m].s[0] += bias_tile[0].s[0];
        acc[_m].s[1] += bias_tile[0].s[1];
    }) 
#endif // defined(BIAS)

    LOOP_UNROLLING(int, _ib_i, 0, 1, M0,
    {
        *((__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + (0) * sizeof(DATA_TYPE) + (indirect_buffer[M0 - 1 - _ib_i].v) * dst_stride_y)) = acc[M0 - 1 - _ib_i].s[0] * ALPHA + BETA;
        *((__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + (1) * sizeof(DATA_TYPE) + (indirect_buffer[M0 - 1 - _ib_i].v) * dst_stride_y)) = acc[M0 - 1 - _ib_i].s[1] * ALPHA + BETA;
    })

    //T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, M0, N0, PARTIAL_STORE_N0, BUFFER, dst, 0, dst_stride_y, x_cond, acc, indirect_buffer);
    
    /*
    if(x_cond)
    {
        LOOP_UNROLLING(int, _i, 0, 1, M0,
        {
            VSTORE_PARTIAL(N0, PARTIAL_STORE_N0)(CONVERT(acc[M0 - 1 - _i].v, VEC_DATA_TYPE(DATA_TYPE, N0)), 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + (0) * sizeof(DATA_TYPE) + (indirect_buffer[M0 - 1 - _i].v) * dst_stride_y));
        })
    }
    else
    {
        LOOP_UNROLLING(int, _i, 0, 1, M0,
        {
            VSTORE(N0)(CONVERT(acc[M0 - 1 - _i].v, VEC_DATA_TYPE(DATA_TYPE, N0)), 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + (0) * sizeof(DATA_TYPE) + (indirect_buffer[M0 - 1 - _i].v) * dst_stride_y));
        })
    }*/
}
#endif // defined(MAT_MUL_MMUL_HUGH)