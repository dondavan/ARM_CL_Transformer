# mat_mul_mmul_hugh.cl


#if defined(MAT_MUL_MMUL_HUGH_NT_NT)
__kernel void mat_mul_mmul_hugh_nt_nt(
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
    TILE(DATA_TYPE, M0, N0, ret);
    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        ret[i].v = 0.f;
    })

    //HUGH_2D(DATA_TYPE, M0, N0, acc);
    //T_LOAD_HUGH(DATA_TYPE, M0, N0, BUFFER, lhs, 0, 0, 1, lhs_stride_y, acc);

    const int rhs_z = z * rhs_h;
    int       k;
    for(k = 0; k <= K - K0; k += K0)
    {
        //TILE(DATA_TYPE, M0, K0, a);
        TILE(DATA_TYPE, K0, N0, b);

        //LOOP_UNROLLING(int, i, 0, 1, M0,{a[i].v = 0.f;})

        LOOP_UNROLLING(int, i, 0, 1, K0,
        {
            b[i].v = 0.f;
        })

        HUGH_2D(DATA_TYPE, M0, K0, a);
        //HUGH_2D(DATA_TYPE, K0, N0, b);

        T_LOAD_HUGH(DATA_TYPE, M0, K0, BUFFER, lhs, k, 0, 1, lhs_stride_y, a);
        //T_LOAD_HUGH(DATA_TYPE, K0, N0, RHS_TENSOR_TYPE, rhs, x, k + rhs_z, 1, rhs_stride_y, b);

        // Load tile from the lhs/rhs tensors
        //T_LOAD(DATA_TYPE, M0, K0, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, K0, N0, RHS_TENSOR_TYPE, rhs, x, k + rhs_z, 1, rhs_stride_y, b);

        //T_MMUL(DATA_TYPE, DATA_TYPE, DATA_TYPE, M0, N0, K0, NT, NT, a, b, acc);
        LOOP_UNROLLING(int, _m, 0, 1, M0,
        {
            LOOP_UNROLLING(int, _k, 0, 1, K0,
            {
                ret[_m].v = fma((DATA_TYPE)(HUGH_2D_ACCESS(a, _m, _k, K0)), b[_k].v, ret[_m].v);
            })
        }) 

        //lhs_offset_first_element_in_bytes += K0 * sizeof(DATA_TYPE);
    }

#if K % K0 != 0
    /* Leftover Loop */
    for(; k < K; ++k)
    {
        //TILE(DATA_TYPE, M0, K0, a);
        TILE(DATA_TYPE, 1, N0, b);

        //LOOP_UNROLLING(int, i, 0, 1, M0,{a[i].v = 0.f;})

        LOOP_UNROLLING(int, i, 0, 1, 1,
        {
            b[i].v = 0.f;
        })

        HUGH_2D(DATA_TYPE, M0, 1, a);
        //HUGH_2D(DATA_TYPE, K0, N0, b);

        T_LOAD_HUGH(DATA_TYPE, M0, 1, BUFFER, lhs, k, 0, 1, lhs_stride_y, a);
        //T_LOAD_HUGH(DATA_TYPE, K0, N0, RHS_TENSOR_TYPE, rhs, x, k + rhs_z, 1, rhs_stride_y, b);

        // Load tile from the lhs/rhs tensors
        //T_LOAD(DATA_TYPE, M0, K0, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, 1, N0, RHS_TENSOR_TYPE, rhs, x, k + rhs_z, 1, rhs_stride_y, b);

        //T_MMUL(DATA_TYPE, DATA_TYPE, DATA_TYPE, M0, N0, K0, NT, NT, a, b, acc);
        LOOP_UNROLLING(int, _m, 0, 1, M0,
        {
            LOOP_UNROLLING(int, _k, 0, 1, 1,
            {
                ret[_m].v = fma((DATA_TYPE)(HUGH_2D_ACCESS(a, _m, _k, 1)), b[_k].v, ret[_m].v);
            })
        }) 
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
    TILE(DATA_TYPE, 1, N0, bias_tile);

    // below expands to use bias_ptr and bias_offset_first_element_in_bytes
    T_LOAD(DATA_TYPE, 1, N0, BUFFER, bias, x, 0, 1, 0, bias_tile);
    
    LOOP_UNROLLING(int, _m, 0, 1, M0,
    {
        ret[_m].v+=bias_tile[0].v; 
    }) 

#endif // defined(BIAS)


    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, M0, N0, PARTIAL_STORE_N0, BUFFER, dst, 0, dst_stride_y, x_cond, ret, indirect_buffer);
}
#endif // defined(MAT_MUL_MMUL_HUGH_NT_NT)