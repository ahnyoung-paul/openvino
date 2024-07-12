// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

// JIT Parameters:
// SIMD         - sub-group size/simd width, one of {8, 16};
// TILE_B       - number of batches processed by each work-item;
// TILE_OFM     - number of output features calculated by work-item, one of {1, 2, 4, 8};
// TILE_IFM     - number of input features loaded from input by work-item, one of {1, 2, 4, 8};
// TILE_K       - number of input features loaded from weights, one of {1, 2, 4, 8};
// TILE_K_OFM   - must be equal to TILE_OFM * TILE_K and less or equal to 8;
// DISPATCH_FSV - output coordinates for each sub-group are calculated from linearized coordinates
// DISPATCH_BSV   as if they laid in bs_fs_bsv_fsv format, these macros describe fsv and bsv factors;

#define INPUT_LOAD_SIZE                     4

#if FC_KERNEL_DYNAMIC_QUANTIZE
KERNEL(quantize_input)(
    const __global INPUT0_TYPE* input,
    __global char* quantized_input,
    __global INPUT0_TYPE* de_quan_scale) {
    const uint offset = get_global_id(0);

    uint input_offset = offset * QUANTIZE_GROUP_SIZE;
    half4 input_0[8];
    char4 quantized_value[8];
    half  max[8];

    unroll_for (uint i = 0 ; i < 8 ; ++i) {
        input_0[i] = vload4(0, &input[input_offset + i * 4]);
        max[i] = fmax(fmax(fabs(input_0[i][0]), fabs(input_0[i][1])), fmax(fabs(input_0[i][2]), fabs(input_0[i][3])));
    }

    half max_value = fmax(fmax(fmax(max[0], max[1]), fmax(max[2], max[3])),
                            fmax(fmax(max[4], max[5]), fmax(max[6], max[7])));

    half quan_scale = max_value / 128;

    unroll_for (uint i = 0 ; i < 8 ; ++i) {
        quantized_value[i] = CAT(convert_, MAKE_VECTOR_TYPE(char, INPUT_LOAD_SIZE))(input_0[i] / (half4)quan_scale);
        vstore4(quantized_value[i], 0, &quantized_input[input_offset + i * 4]);
    }

    de_quan_scale[offset] = quan_scale;
}
#else  // !FC_KERNEL_DYNAMIC_QUANTIZE

// Verify JIT parameters.
#if SIMD != 8 && SIMD != 16
#   error "fully_connected_gpu_bf_tiled.cl - SIMD must be one of {8, 16}"
#endif

#if TILE_OFM != 1 && TILE_OFM != 2 && TILE_OFM != 4 && TILE_OFM != 8
#   error "fully_connected_gpu_bf_tiled.cl - TILE_OFM must be one of {1, 2, 4, 8}"
#endif

#if TILE_IFM != 1 && TILE_IFM != 2 && TILE_IFM != 4 && TILE_IFM != 8
#   error "fully_connected_gpu_bf_tiled.cl - TILE_IFM must be one of {1, 2, 4, 8}"
#endif

#if TILE_K != 1 && TILE_K != 2 && TILE_K != 4 && TILE_K != 8
#   error "fully_connected_gpu_bf_tiled.cl - TILE_K must be one of {1, 2, 4, 8}"
#endif

#if TILE_K_OFM != (TILE_K * TILE_OFM) || TILE_K_OFM > 8
#   error "fully_connected_gpu_bf_tiled.cl - TILE_K_OFM must be equal to TILE_K * TILE_OFM and at most 8"
#endif

#if COMPRESSED_WEIGHTS_INT4
#   if TILE_K_OFM != TILE_K_OFM_PACKED * 2
#       error "fully_connected_gpu_bf_tiled.cl - TILE_K_OFM must be divisible by 2 for 4-bit compressed case"
#   endif
#   if FILTER_LAYOUT_OS_IS_YX_OSV32_ISV2 && TILE_K != 4 && TILE_K != 2 && TILE_K != 1
#       error "fully_connected_gpu_bf_tiled.cl - TILE_K must be one of {1, 2, 4}"
#   endif
#endif
#if TILE_K == 4 && COMPRESSED_WEIGHTS_INT4 && FILTER_LAYOUT_OS_IS_YX_OSV32_ISV2
// Data stored in memory : f0k0k1|f16k0k1|f0k2k3|f16k2k3
// => unpack as f0k0k1|f0k2k3|f16k0k1|f16k2k3 so that the weight access order is preserved 
#define UNPACK_INT4 UNPACK_INT4x2_OSV32_ISV2
#define UNPACK_TRANSPOSED_INT4 UNPACK_INT4x2_OSV32_ISV2
#else
#define UNPACK_INT4 UNPACK_INT4x2
#define UNPACK_TRANSPOSED_INT4 UNPACK_TRANSPOSED_INT4x2
#endif
// Macros for vectorized types.
#define INPUT_VEC_TYPE             MAKE_VECTOR_TYPE(INPUT0_TYPE, TILE_IFM)
#define ACCUMULATOR_VEC_TYPE       MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, TILE_OFM)
#define FILTER_VEC_TYPE            MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, TILE_K_OFM)
#define FILTER_PACKED_VEC_TYPE     MAKE_VECTOR_TYPE(FILTER_TYPE, TILE_K_OFM_PACKED)
#define BIAS_VEC_TYPE              MAKE_VECTOR_TYPE(BIAS_TYPE, TILE_OFM)
#define OUTPUT_VEC_TYPE            MAKE_VECTOR_TYPE(OUTPUT_TYPE, TILE_OFM)
#define ACTIVATION_VEC_TYPE        MAKE_VECTOR_TYPE(ACTIVATION_TYPE, TILE_OFM)
#define TO_OUTPUT_VEC_TYPE(x)      CAT(convert_, OUTPUT_VEC_TYPE)(x)
#define TO_ACTIVATION_VEC_TYPE(x)  CAT(convert_, ACTIVATION_VEC_TYPE)(x)
#define TO_FILTER_VEC_TYPE(x)      CAT(convert_, FILTER_VEC_TYPE)(x)
#define TO_ACCUMULATOR_VEC_TYPE(x) CAT(convert_, ACCUMULATOR_VEC_TYPE)(x)

#define INPUT_BLOCK_READ(ptr, offset)        BLOCK_READN(INPUT0_TYPE, TILE_IFM, ptr, offset)
#define FILTER_BLOCK_READ(ptr, offset)       BLOCK_READN(FILTER_TYPE, TILE_K_OFM_PACKED, ptr, offset)
#define BIAS_BLOCK_READ(ptr, offset)         BLOCK_READN(BIAS_TYPE, TILE_OFM, ptr, offset)
#define OUTPUT_BLOCK_WRITE(ptr, offset, val) BLOCK_WRITEN(OUTPUT_TYPE, TILE_OFM, ptr, offset, val)

#define SLM_FILTER_VEC          MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, TILE_OFM)
#define SLM_FILTER_PACKED_VEC   MAKE_VECTOR_TYPE(FILTER_TYPE, FILTER_LOAD_BLOCK_SIZE)
#define SLM_FILTER_UNPACKED_VEC MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, FILTER_ELEMENTS_PER_LOAD)


// Check alignment restrictions for using block writes on output.
#define USE_BLOCK_WRITE ((OUTPUT_TYPE_SIZE * TILE_OUT_B_PITCH) % 16 == 0 && (OUTPUT_TYPE_SIZE * OUTPUT_OFFSET) % 16 == 0)


#if !REALIGN_FP16_OFFSET
#   if OUTPUT_3D
#       define MAIN_LOOP_ELEMENTS_COUNT  INPUT0_SIZE_Y
#   else
#       define MAIN_LOOP_ELEMENTS_COUNT  INPUT0_ELEMENTS_COUNT
#   endif
#else
// For REALIGN_FP16_OFFSET one feature is processed separately before entering main loop to correct alignment.
#   if OUTPUT_3D
#       define MAIN_LOOP_ELEMENTS_COUNT  (INPUT0_SIZE_Y - 1)
#   else
#       define MAIN_LOOP_ELEMENTS_COUNT  (INPUT0_ELEMENTS_COUNT - 1)
#   endif
#endif

#if OUTPUT_3D
#   define INPUT_ELEMENTS_COUNT INPUT0_SIZE_Y
#else
#   define INPUT_ELEMENTS_COUNT INPUT0_ELEMENTS_COUNT
#endif

#if IS_DYNAMIC && COMPRESSED_WEIGHTS_INT4
#pragma disable_includes_optimization
#define FORCED_TILE_B 1
#include "include/fully_connected_gpu_bf_tiled_common.cl"
#undef FORCED_TILE_B

#define FORCED_TILE_B 2
#include "include/fully_connected_gpu_bf_tiled_common.cl"
#undef FORCED_TILE_B

#define FORCED_TILE_B 3
#include "include/fully_connected_gpu_bf_tiled_common.cl"
#undef FORCED_TILE_B

#define FORCED_TILE_B 4
#include "include/fully_connected_gpu_bf_tiled_common.cl"
#undef FORCED_TILE_B

#define FORCED_TILE_B 5
#include "include/fully_connected_gpu_bf_tiled_common.cl"
#undef FORCED_TILE_B

#define FORCED_TILE_B 6
#include "include/fully_connected_gpu_bf_tiled_common.cl"
#undef FORCED_TILE_B

#define FORCED_TILE_B 7
#include "include/fully_connected_gpu_bf_tiled_common.cl"
#undef FORCED_TILE_B
#pragma enable_includes_optimization
#endif

inline void FUNC(fc_bf_tiled_kernel_default)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
#if DECOMPRESSION_SCALE_TERM
    const __global DECOMPRESSION_SCALE_TYPE* decompression_scale,
#endif
#if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
    const __global DECOMPRESSION_ZP_TYPE* decompression_zp,
#endif
    __global OUTPUT_TYPE* output,
    const __global FILTER_TYPE* weights
#if USE_SLM
    , __local ACCUMULATOR_TYPE* wei_local_mem
#endif
#if BIAS_TERM
    , const __global BIAS_TYPE* biases
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
#if USE_SLM
    uint gid = (uint)get_group_id(0);
    uint local_id = (uint)get_local_id(2);
#else
    uint gid = (uint)get_group_id(0);
#endif
    uint sglid = (uint)get_sub_group_local_id();

    // Dispatch as bs_fs_bsv_fsv, where bsv = DISPATCH_BSV and fsv = DISPATCH_FSV.
    // This allows more fine grained control over dispatch order than using work-groups and
    // avoids requirement of threads being available for whole work-group.
    // It could hovewer have some drawbacks like not providing physical locality or not using
    // full dispatch pipeline.
    uint feature_mini_block = gid % DISPATCH_FSV;
    uint batch_mini_block = gid / DISPATCH_FSV % DISPATCH_BSV;
    uint feature_mega_block = gid / (DISPATCH_FSV * DISPATCH_BSV) % (CEIL_DIV(TILE_OUT_F_NUM, TILE_OFM * SIMD) / DISPATCH_FSV);
    uint batch_mega_block = gid / (DISPATCH_FSV * DISPATCH_BSV * CEIL_DIV(TILE_OUT_F_NUM, TILE_OFM * SIMD) / DISPATCH_FSV);

#if USE_SLM
    uint out_f = gid * (TILE_OFM * SIMD);
    uint out_b = LWS_BATCHES * TILE_B * (uint)get_group_id(2) + local_id * TILE_B;
#else
    uint out_f = (feature_mega_block * DISPATCH_FSV + feature_mini_block) * (TILE_OFM * SIMD);
    uint out_b = ((batch_mega_block * DISPATCH_BSV + batch_mini_block) * TILE_B);
#endif

    ACCUMULATOR_VEC_TYPE acc[TILE_B] = { };
    INPUT_VEC_TYPE       in_0[TILE_B] = { };
    ACCUMULATOR_VEC_TYPE acc_ref[TILE_B] = { };
    ACCUMULATOR_TYPE val_acc = 0.f;
    ACCUMULATOR_TYPE val_ref_acc = 0.f;
#define NUM_ARR 6144
    ACCUMULATOR_TYPE acc_var_arr[NUM_ARR] = { };
    uint arr_idx = 0;

#if !USE_SLM
    FILTER_VEC_TYPE wei = 0;
#endif


#if OUTPUT_3D
    uint out_b0 = out_b / OUTPUT_FEATURE_NUM;
    uint out_b1 = out_b % OUTPUT_FEATURE_NUM;
    uint input_offset = out_b0 * INPUT0_BATCH_PITCH + out_b1 * INPUT0_FEATURE_PITCH + INPUT0_OFFSET;
#else
    uint input_offset = out_b * TILE_IN_B_PITCH + INPUT0_OFFSET;
#endif

#if COMPRESSED_WEIGHTS_INT4
    #if TILE_OFM == 1 && FILTER_LAYOUT_OS_IS_YX_OSV32_ISV2
    const int power_of_two_for_simd = 4;
    const int power_of_two_for_osv = 5;
    const uint osv32_weight_base = (( (int) (out_f >> power_of_two_for_osv) ) << power_of_two_for_osv);
    const uint osv_weight_stride = (INPUT_ELEMENTS_COUNT >> 1);
    const uint out_f_offset = (int)((out_f >> power_of_two_for_simd) & 0x1) << power_of_two_for_simd;
    // out_f(32) : 32 + osv_weight_stride + 0;
    // out_f(48) : 32 + osv_weight_stride + 16;
    // out_f(64) : 64 + osv_weight_stride + 0;
    // ...
    uint weights_offset =  osv32_weight_base * osv_weight_stride + out_f_offset;
    #else
    uint weights_offset = out_f * (INPUT_ELEMENTS_COUNT / 2);
    #endif
#else
    uint weights_offset = out_f * INPUT_ELEMENTS_COUNT;
#endif

#if COMPRESSED_WEIGHTS && DECOMPRESSION_SCALE_GROUPS_NUM == 1
    #if DECOMPRESSION_SCALE_LENGTH > 1 && DECOMPRESSION_SCALE_LENGTH % (TILE_OFM * SIMD) == 0
        ACCUMULATOR_VEC_TYPE d_scale = TO_ACCUMULATOR_VEC_TYPE(BLOCK_READN(DECOMPRESSION_SCALE_TYPE, TILE_OFM, decompression_scale, out_f));
    #elif DECOMPRESSION_SCALE_LENGTH > 1 && DECOMPRESSION_SCALE_LENGTH % (TILE_OFM * SIMD) != 0
        ACCUMULATOR_VEC_TYPE d_scale = 0;
        unroll_for(uint of = 0; of < TILE_OFM; ++of) {
            uint offset = out_f + of*SIMD + get_sub_group_local_id();
            if (offset < DECOMPRESSION_SCALE_LENGTH)
                ((ACCUMULATOR_TYPE*)(&d_scale))[of] = decompression_scale[offset];
        }
    #else
        ACCUMULATOR_VEC_TYPE d_scale = decompression_scale[0];
    #endif

    ACCUMULATOR_TYPE* d_scales = (ACCUMULATOR_TYPE*)(&d_scale);
#endif

#if COMPRESSED_WEIGHTS && DECOMPRESSION_ZP_TERM && DECOMPRESSION_ZP_GROUPS_NUM == 1 && !DECOMPRESSION_ZP_SCALAR
    #if DECOMPRESSION_ZP_LENGTH > 1 && DECOMPRESSION_ZP_LENGTH % (TILE_OFM * SIMD) == 0
        ACCUMULATOR_VEC_TYPE d_zp = TO_ACCUMULATOR_VEC_TYPE(BLOCK_READN(DECOMPRESSION_ZP_TYPE, TILE_OFM, decompression_zp, out_f));
    #elif DECOMPRESSION_ZP_LENGTH > 1 && DECOMPRESSION_ZP_LENGTH % (TILE_OFM * SIMD) != 0
        ACCUMULATOR_VEC_TYPE d_zp = 0;
        unroll_for(uint of = 0; of < TILE_OFM; ++of) {
            uint offset = out_f + of*SIMD + get_sub_group_local_id();
            if (offset < DECOMPRESSION_ZP_LENGTH)
                ((ACCUMULATOR_TYPE*)(&d_zp))[of] = decompression_zp[offset];
        }
    #else
        ACCUMULATOR_VEC_TYPE d_zp = decompression_zp[0];
    #endif
    ACCUMULATOR_TYPE* d_zps = (ACCUMULATOR_TYPE*)(&d_zp);
#endif

#if REALIGN_FP16_OFFSET
    // For fp16 we need to ensure that all block reads are aligned to 4 byte (2 words) boundary.
    // To do this solve first input feature separately.
    {
        INPUT0_TYPE tmp_input = input[input_offset + get_sub_group_local_id() % TILE_B * TILE_IN_B_PITCH];
        ACCUMULATOR_VEC_TYPE tmp_wei = TO_ACCUMULATOR_VEC_TYPE(BLOCK_READN(FILTER_TYPE, TILE_OFM, weights, weights_offset));
        #if COMPRESSED_WEIGHTS
            tmp_wei = (tmp_wei - d_zp) * d_scale;
        #endif
        unroll_for(uint bi = 0; bi < TILE_B; ++bi) {
            acc[bi] = _sub_group_shuffle(tmp_input, bi) * tmp_wei;
        }
        weights_offset += TILE_OFM * SIMD;
        input_offset += 1;
    }
#endif
    // =====================================================================================================================================
    // Main computation loop
    uint iterations = MAIN_LOOP_ELEMENTS_COUNT / (TILE_IFM * SIMD);
    __attribute__((opencl_unroll_hint(1)))
    for (uint ni = 0; ni < iterations; ++ni) {
        // Load input.
        #define LOAD_IN_0(bi) do {                                  \
                in_0[bi] = INPUT_BLOCK_READ(input, input_offset);   \
                input_offset += TILE_IN_B_PITCH;                    \
            } while (false)

        CONST_LOOP(TILE_B, LOAD_IN_0);
        #undef LOAD_IN_0
        input_offset += TILE_IFM * SIMD - TILE_IN_B_PITCH * TILE_B;
        // NOTE: Manually unrolling multiplication loop leads to lower register pressure and allows for bigger block sizes,
        //       but significantly degrades readability and generality of code.
        //       It doesn't also show noticable performance improvement on tested configurations.
        // #if DECOMPRESSION_SCALE_POST_OP
            ACCUMULATOR_VEC_TYPE acc_tmp[TILE_B] = { };
        // #endif

        #if USE_SLM && COMPRESSED_WEIGHTS_INT4
            #if TILE_OFM != 2
            #error "FC bf_tiled kernel: can't use SLM optimization with TILE_OFM != 2"
            #endif

            // Skip first barrier synchronization if there is only single outer loop iteration.
            #if MAIN_LOOP_ELEMENTS_COUNT / (TILE_IFM * SIMD) > 1
                barrier(CLK_LOCAL_MEM_FENCE);
            #endif

            __local SLM_FILTER_VEC* slm_wei_vec = (__local SLM_FILTER_VEC*)wei_local_mem;

            uint weights_idx = weights_offset + local_id * SIMD * FILTER_LOAD_ITERS * FILTER_LOAD_BLOCK_SIZE;
            uint wei_local_idx = local_id * SIMD * FILTER_LOAD_ITERS * FILTER_LOAD_BLOCK_SIZE + sglid;

            unroll_for(uint load_iter = 0; load_iter < FILTER_LOAD_ITERS; ++load_iter) {
                SLM_FILTER_PACKED_VEC wei_packed = BLOCK_READN(FILTER_TYPE, FILTER_LOAD_BLOCK_SIZE, weights, weights_idx);
                SLM_FILTER_UNPACKED_VEC wei_unpacked = UNPACK_INT4(ACCUMULATOR_TYPE, *((INT4_PACKED_TYPE_PRELOAD*)&wei_packed));
                ACCUMULATOR_TYPE* w = (ACCUMULATOR_TYPE*)(&wei_unpacked);
                unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
                    unroll_for(uint kii = 0; kii < FILTER_LOAD_BLOCK_SIZE; ++kii) {
                        const uint offset_ofm = out_f + fi*SIMD + sglid;
                        const uint offset_ifm = ni * TILE_IFM * SIMD + local_id * FILTER_LOAD_ITERS * FILTER_LOAD_BLOCK_SIZE + load_iter * FILTER_LOAD_BLOCK_SIZE + kii;
                        #if !DECOMPRESSION_SCALE_POST_OP
                            #if DECOMPRESSION_SCALE_GROUPS_NUM > 1
                                const uint scale_offset = (offset_ofm % DECOMPRESSION_SCALE_BATCH_NUM) * DECOMPRESSION_SCALE_BATCH_PITCH  +
                                                          (offset_ifm / DECOMPRESSION_SCALE_GROUP_SIZE) * DECOMPRESSION_SCALE_FEATURE_PITCH;
                                ACCUMULATOR_TYPE ds = decompression_scale[scale_offset];
                            #else
                                ACCUMULATOR_TYPE ds = d_scales[fi % DECOMPRESSION_SCALE_LENGTH];
                            #endif
                        #else
                            ACCUMULATOR_TYPE ds = ACCUMULATOR_VAL_ONE;
                        #endif

                        #if DECOMPRESSION_ZP_TERM
                            #if DECOMPRESSION_ZP_SCALAR
                                ACCUMULATOR_TYPE dzp = DECOMPRESSION_ZP_VALUE;
                            #elif DECOMPRESSION_ZP_GROUPS_NUM > 1
                                const uint zp_offset = (offset_ofm % DECOMPRESSION_ZP_BATCH_NUM) * DECOMPRESSION_ZP_BATCH_PITCH +
                                                       (offset_ifm / DECOMPRESSION_ZP_GROUP_SIZE) * DECOMPRESSION_ZP_FEATURE_PITCH;
                                ACCUMULATOR_TYPE dzp = decompression_zp[zp_offset];
                            #else
                                ACCUMULATOR_TYPE dzp = d_zps[fi % DECOMPRESSION_ZP_LENGTH];
                            #endif
                        #else
                            ACCUMULATOR_TYPE dzp = ACCUMULATOR_VAL_ZERO;
                        #endif
                        w[W_IDX] = (w[W_IDX] - dzp) * ds;
                    }
                }

                #define STORE_TO_SLM(vec2) slm_wei_vec[wei_local_idx] = vec2; wei_local_idx += SIMD;

                #if FILTER_LOAD_BLOCK_SIZE == 2
                    STORE_TO_SLM(wei_unpacked.s01);
                    STORE_TO_SLM(wei_unpacked.s23);
                #elif FILTER_LOAD_BLOCK_SIZE == 4
                    STORE_TO_SLM(wei_unpacked.s01);
                    STORE_TO_SLM(wei_unpacked.s23);
                    STORE_TO_SLM(wei_unpacked.s45);
                    STORE_TO_SLM(wei_unpacked.s67);
                #elif FILTER_LOAD_BLOCK_SIZE == 8
                    STORE_TO_SLM(wei_unpacked.s01);
                    STORE_TO_SLM(wei_unpacked.s23);
                    STORE_TO_SLM(wei_unpacked.s45);
                    STORE_TO_SLM(wei_unpacked.s67);
                    STORE_TO_SLM(wei_unpacked.s89);
                    STORE_TO_SLM(wei_unpacked.sab);
                    STORE_TO_SLM(wei_unpacked.scd);
                    STORE_TO_SLM(wei_unpacked.sef);
                #else
                    #error "FC bf_tiled kernel: unsupported FILTER_LOAD_BLOCK_SIZE for SLM kernel"
                #endif

                #undef STORE_TO_SLM

                weights_idx += SIMD * FILTER_LOAD_BLOCK_SIZE;
            }

            wei_local_idx = sglid;

            barrier(CLK_LOCAL_MEM_FENCE);
        #endif
        unroll_for(uint ki = 0; ki < (TILE_IFM * SIMD) / TILE_K; ++ki) {
            #if COMPRESSED_WEIGHTS_INT4
                #if USE_SLM
                    FILTER_VEC_TYPE wei = 0;
                    #define LOAD_FROM_SLM(vec2) vec2 = slm_wei_vec[wei_local_idx]; wei_local_idx += SIMD;
                    #if TILE_K == 1
                        LOAD_FROM_SLM(wei.s01);
                    #elif TILE_K == 2
                        LOAD_FROM_SLM(wei.s01);
                        LOAD_FROM_SLM(wei.s23);
                    #elif TILE_K == 4
                        LOAD_FROM_SLM(wei.s01);
                        LOAD_FROM_SLM(wei.s23);
                        LOAD_FROM_SLM(wei.s45);
                        LOAD_FROM_SLM(wei.s67);
                    #else
                    #error "FC bf_tiled kernel: unsupported TILE_K size for SLM kernel"
                    #endif
                    #undef LOAD_FROM_SLM
                #else
                    FILTER_PACKED_VEC_TYPE wei_packed = FILTER_BLOCK_READ(weights, weights_offset);
                    wei = UNPACK_INT4(ACCUMULATOR_TYPE, *((INT4_PACKED_TYPE*)&wei_packed));
                #endif
            #else
                wei = TO_FILTER_VEC_TYPE(FILTER_BLOCK_READ(weights, weights_offset));
            #endif

            #if COMPRESSED_WEIGHTS && !USE_SLM
                ACCUMULATOR_TYPE* w = (ACCUMULATOR_TYPE*)(&wei);
                unroll_for(uint kii = 0; kii < TILE_K; ++kii) {
                    unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
                        const uint offset_ofm = out_f + fi*SIMD + sglid;
                        #if !DECOMPRESSION_SCALE_POST_OP
                            // Apply scales before FMA to avoid FP16 overflow in case of INT8
                            #if DECOMPRESSION_SCALE_GROUPS_NUM > 1
                                const uint scale_offset = (offset_ofm % DECOMPRESSION_SCALE_BATCH_NUM) * DECOMPRESSION_SCALE_BATCH_PITCH  +
                                                        ((kii + ki*TILE_K + ni*TILE_IFM*SIMD) / DECOMPRESSION_SCALE_GROUP_SIZE)*DECOMPRESSION_SCALE_FEATURE_PITCH;
                                ACCUMULATOR_TYPE ds = decompression_scale[scale_offset];
                            #else
                                ACCUMULATOR_TYPE ds = d_scales[fi % DECOMPRESSION_SCALE_LENGTH];
                            #endif
                        #else
                            ACCUMULATOR_TYPE ds = ACCUMULATOR_VAL_ONE;
                        #endif

                        #if DECOMPRESSION_ZP_TERM
                            #if DECOMPRESSION_ZP_SCALAR
                                ACCUMULATOR_TYPE dzp = DECOMPRESSION_ZP_VALUE;
                            #elif DECOMPRESSION_ZP_GROUPS_NUM > 1
                                const uint zp_offset = (offset_ofm % DECOMPRESSION_ZP_BATCH_NUM) * DECOMPRESSION_ZP_BATCH_PITCH +
                                                    ((kii + ki*TILE_K + ni*TILE_IFM*SIMD) / DECOMPRESSION_ZP_GROUP_SIZE) * DECOMPRESSION_ZP_FEATURE_PITCH;
                                ACCUMULATOR_TYPE dzp = decompression_zp[zp_offset];
                            #else
                                ACCUMULATOR_TYPE dzp = d_zps[fi % DECOMPRESSION_ZP_LENGTH];
                            #endif
                        #else
                            ACCUMULATOR_TYPE dzp = ACCUMULATOR_VAL_ZERO;
                        #endif
                        w[W_IDX] = (w[W_IDX] - dzp) * ds;
                    }
                }
            #endif

            unroll_for (uint kii = 0; kii < TILE_K; ++kii) {
                const uint total_k = ki * TILE_K + kii;
                unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
                    INPUT0_TYPE in_val = _sub_group_shuffle(((INPUT0_TYPE*)(&in_0[bi]))[total_k / SIMD], total_k % SIMD);
                    unroll_for (uint fi = 0; fi < TILE_OFM; ++fi) {
#if DECOMPRESSION_SCALE_POST_OP
                    half weight = ((ACCUMULATOR_TYPE*)(&wei))[W_IDX];
                    #if TILE_OFM > 1
                        ((ACCUMULATOR_TYPE*)(&acc_tmp[bi]))[fi] += in_val * weight;
                    #else
                        acc_tmp[bi] += in_val * weight;
                    #endif
#else
                    #if TILE_OFM > 1
                        ACCUMULATOR_TYPE tmp_val = in_val * ((ACCUMULATOR_TYPE*)(&wei))[W_IDX];
                        ((ACCUMULATOR_TYPE*)(&acc[bi]))[fi] += tmp_val;
                        ((ACCUMULATOR_TYPE*)(&acc_tmp[bi]))[fi] += tmp_val;
                        if (get_global_id(0) == 1 && get_global_id(1) == 0 && get_global_id(2) == 0 && bi == 7 && fi == 0) {
                            val_acc += tmp_val;
                            acc_var_arr[arr_idx] = tmp_val;
                            arr_idx += 1;
                        }
                    #else
                        acc[bi] += in_val * ((ACCUMULATOR_TYPE*)(&wei))[W_IDX];
                    #endif
#endif
                    }
                }
            }
            #if TILE_OFM == 1 && FILTER_LAYOUT_OS_IS_YX_OSV32_ISV2
            weights_offset += TILE_K_OFM_PACKED * 2 * SIMD;
            #else
            weights_offset += TILE_K_OFM_PACKED * SIMD;
            #endif


#if DECOMPRESSION_SCALE_POST_OP && (TILE_IFM * SIMD > DECOMPRESSION_SCALE_GROUP_SIZE)
            unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
                unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
                    const uint offset_ofm = out_f + fi*SIMD + sglid;

                    #if DECOMPRESSION_SCALE_GROUPS_NUM > 1
                        const uint scale_offset = (offset_ofm % DECOMPRESSION_SCALE_BATCH_NUM) * DECOMPRESSION_SCALE_BATCH_PITCH +
                                                ((ni*TILE_IFM*SIMD + ki*TILE_K) / DECOMPRESSION_SCALE_GROUP_SIZE)*DECOMPRESSION_SCALE_FEATURE_PITCH;
                        ACCUMULATOR_TYPE ds = decompression_scale[scale_offset];
                    #else
                        ACCUMULATOR_TYPE ds = d_scales[fi % DECOMPRESSION_SCALE_LENGTH];
                    #endif
                    #if TILE_OFM > 1
                    ((ACCUMULATOR_TYPE*)(&acc[bi]))[fi] += ((ACCUMULATOR_TYPE*)(&acc_tmp[bi]))[fi] * ds;
                    #else
                    acc[bi] += acc_tmp[bi] * ds;
                    acc_tmp[bi] = 0;
                    #endif
                }
            }
#endif
        }
#if DECOMPRESSION_SCALE_POST_OP && (TILE_IFM * SIMD <= DECOMPRESSION_SCALE_GROUP_SIZE)
        unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
            unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
                const uint offset_ofm = out_f + fi*SIMD + sglid;

                #if DECOMPRESSION_SCALE_GROUPS_NUM > 1
                    const uint scale_offset = (offset_ofm % DECOMPRESSION_SCALE_BATCH_NUM) * DECOMPRESSION_SCALE_BATCH_PITCH +
                                              ((ni*TILE_IFM*SIMD) / DECOMPRESSION_SCALE_GROUP_SIZE)*DECOMPRESSION_SCALE_FEATURE_PITCH;
                    ACCUMULATOR_TYPE ds = decompression_scale[scale_offset];
                #else
                    ACCUMULATOR_TYPE ds = d_scales[fi % DECOMPRESSION_SCALE_LENGTH];
                #endif
                #if TILE_OFM > 1
                ((ACCUMULATOR_TYPE*)(&acc[bi]))[fi] += ((ACCUMULATOR_TYPE*)(&acc_tmp[bi]))[fi] * ds;
                #else
                acc[bi] += acc_tmp[bi] * ds;
                #endif
            }
        }
#endif
#if !DECOMPRESSION_SCALE_POST_OP
        unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
            unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
                #if TILE_OFM > 1
                ACCUMULATOR_TYPE tmp_val2 = ((ACCUMULATOR_TYPE*)(&acc_tmp[bi]))[fi];
                ((ACCUMULATOR_TYPE*)(&acc_ref[bi]))[fi] += tmp_val2;
                if (get_global_id(0) == 1 && get_global_id(1) == 0 && get_global_id(2) == 0 && bi == 7 && fi == 0) {
                    val_ref_acc += tmp_val2;
                }
                #else
                acc_ref[bi] += acc_tmp[bi];
                #endif
            }
        }
#endif
    }
// #if !DECOMPRESSION_SCALE_POST_OP
//     if (get_global_id(0) < 3 && get_global_id(1) < 3 && get_global_id(2) < 3) {
//         unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
//             unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
//                 ACCUMULATOR_TYPE ref_val = ((ACCUMULATOR_TYPE*)(&acc_ref[bi]))[fi];
//                 ACCUMULATOR_TYPE act_val = ((ACCUMULATOR_TYPE*)(&acc[bi]))[fi];
//                 if (ref_val != act_val) {
//                     #if DECOMPRESSION_SCALE_POST_OP
//                         bool scale_post_op = true;
//                     #else
//                         bool scale_post_op = false;
//                     #endif
//                     uint gid0 = (uint)get_global_id(0);
//                     uint gid1 = (uint)get_global_id(1);
//                     uint gid2 = (uint)get_global_id(2);
//                     printf("[%d,%d,%d][%d,%d], scale_post_op(%d) acc_ref{%f} = acc{%f}\n",
//                         gid0, gid1, gid2, bi, fi, scale_post_op, ref_val, act_val);
//                 }
//             }
//         }
//     }
// #endif
    {
        if (get_global_id(0) == 1 && get_global_id(1) == 0 && get_global_id(2) == 0) {
            half val0 = 2047.f;
            half val1 = val0 + 1.f;
            half val2 = val1 + 1.f;
            half val3 = val1 + val1;
            #if DECOMPRESSION_SCALE_POST_OP
                bool scale_post_op = true;
            #else
                bool scale_post_op = false;
            #endif
            printf("\n[calc] TILE_K:%d, TILE_B:%d, TILE_OFM:%d, TILE_IFM:%d, SIMD:%d, "
                "iterations:%d, DECOMPRESSION_SCALE_GROUP_SIZE:%d, scale_post_op:%d, acc:%f, acc_ref:%f"
                "[val0:%f,(val1=val0+1):%f,(val2=val1+1):%f, (val3=val1+val1):%f]\n"
                , TILE_K, TILE_B, TILE_OFM, TILE_IFM, SIMD, iterations, DECOMPRESSION_SCALE_GROUP_SIZE,
                 scale_post_op, ((ACCUMULATOR_TYPE*)(&acc[7]))[0], ((ACCUMULATOR_TYPE*)(&acc_ref[7]))[0],
                 (float)val0, (float)val1, (float)val2, (float)val3);
            printf("1.val_acc[%f], val_ref_acc[%f]\n", val_acc, val_ref_acc);
#if !DECOMPRESSION_SCALE_POST_OP
            {
                half sum1 = 0.f;
                half sum2 = 0.f;
                uint idx = 0;
                __attribute__((opencl_unroll_hint(1)))
                for (uint ni = 0; ni < iterations; ++ni) {
                    half temp = 0.f;
                    unroll_for(uint ki = 0; ki < (TILE_IFM * SIMD) / TILE_K; ++ki) {
                        unroll_for (uint kii = 0; kii < TILE_K; ++kii) {
                            const uint total_k = ki * TILE_K + kii;
                            sum1 += acc_var_arr[idx];
                            temp += acc_var_arr[idx];
                            idx += 1;
                        }
                    }
                    sum2 += temp;
                }
                printf("2.Check sum sum1[%f] vs sum2[%f]\n", sum1, sum2);

                // printf("half acc_var_arr[%d]={", NUM_ARR);
                // for (uint i = 0; i < NUM_ARR; i++) {
                //     printf("%ff,", acc_var_arr[i]);
                //     if (i != 0 && i % 200 == 0)
                //         printf("\n");
                // }
                // printf("};\n");
            }
            {
                half acc_var_arr2[6144]={0.000641f,-0.000326f,0.000176f,-0.000000f,0.001503f,0.000215f,0.000165f,0.000000f,0.000000f,-0.000765f,0.000092f,0.000397f,0.001147f,0.000187f,0.000328f,-0.000000f,-0.000103f,0.000100f,0.000718f,0.000116f,0.000582f,0.000442f,0.000355f,0.000202f,0.000453f,0.000476f,-0.001279f,0.000873f,-0.000027f,-0.001253f,-0.000692f,0.000194f,-0.000732f,-0.000468f,0.000000f,0.000946f,0.000968f,0.001018f,0.000481f,-0.000304f,0.001697f,0.000567f,-0.000269f,0.000638f,0.001662f,-0.000119f,-0.000381f,-0.001328f,0.001801f,-0.000385f,0.000256f,0.000000f,-0.000279f,0.001322f,0.000532f,-0.000779f,0.000477f,-0.000221f,-0.000158f,0.000465f,-0.000355f,0.000175f,0.000348f,-0.001643f,0.000071f,-0.000132f,0.000893f,-0.000160f,0.000174f,-0.000000f,0.000464f,0.002926f,0.000401f,-0.001472f,0.000185f,-0.001700f,-0.000094f,-0.001824f,0.000602f,0.001463f,0.000772f,-0.002047f,0.000061f,-0.001320f,-0.000308f,-0.000556f,-0.000000f,-0.001630f,0.000444f,-0.000138f,-0.000520f,-0.000287f,-0.000451f,0.000000f,0.000535f,-0.000194f,-0.000278f,0.000854f,0.000389f,0.000695f,-0.000382f,0.000368f,-0.000745f,0.001045f,0.000244f,0.000022f,0.001573f,-0.000361f,0.000000f,-0.000000f,0.003164f,0.000576f,0.000000f,-0.000384f,0.000000f,0.000482f,0.000126f,-0.000222f,0.000222f,-0.001290f,-0.000268f,-0.000604f,0.000880f,-0.000648f,-0.000000f,0.000000f,-0.000000f,0.000315f,-0.000000f,-0.000508f,-0.000753f,-0.000103f,-0.001484f,-0.000000f,0.000000f,0.000385f,-0.000256f,-0.000000f,-0.000000f,0.000063f,-0.000080f,0.000049f,0.000002f,0.000000f,0.001040f,-0.000353f,0.000774f,0.001236f,0.000062f,0.000267f,-0.001631f,-0.000867f,0.000165f,-0.000112f,0.000763f,0.000011f,-0.000311f,0.000450f,0.000236f,0.004761f,-0.000577f,0.000520f,0.006706f,0.000000f,0.008308f,-0.000234f,-0.000483f,0.001078f,-0.000670f,0.000000f,-0.000322f,0.002453f,0.000997f,0.000018f,-0.000388f,0.000214f,-0.000329f,-0.001011f,0.000375f,0.002359f,-0.000214f,-0.000643f,0.000000f,0.000396f,-0.000726f,-0.000606f,0.000586f,0.001657f,0.000054f,0.001526f,0.000000f,0.000492f,-0.000505f,-0.000001f,0.000160f,0.000817f,0.000461f,0.001052f,0.000617f,0.000537f,-0.000246f,
                    -0.000331f,0.001740f,-0.000465f,0.000026f,0.000601f,-0.000464f,0.000984f,-0.000000f,0.000000f,-0.000342f,0.000331f,0.000676f,-0.000000f,-0.000000f,0.000007f,0.000176f,0.001630f,-0.002069f,-0.000218f,0.000112f,-0.000797f,0.000057f,-0.000514f,-0.000000f,0.005005f,0.000636f,-0.000118f,0.001450f,0.000258f,0.000576f,0.000303f,0.000527f,-0.000298f,0.000000f,0.001070f,0.000953f,-0.000167f,0.000464f,-0.000000f,-0.000000f,-0.000251f,-0.002405f,0.000000f,0.000079f,0.000000f,0.000131f,-0.000654f,-0.000048f,0.006454f,0.000282f,0.001345f,0.000498f,0.000800f,0.000057f,-0.000362f,0.001980f,0.001538f,-0.000000f,-0.000319f,0.000647f,0.000000f,-0.000000f,0.000244f,0.000991f,-0.000000f,-0.000000f,-0.000000f,-0.000078f,0.000000f,-0.000150f,0.000163f,-0.001039f,-0.009140f,-0.000079f,-0.000121f,-0.000135f,-0.000212f,-0.001451f,-0.000222f,-0.000147f,0.000866f,0.011391f,0.000697f,0.009468f,-0.000531f,0.000152f,0.000276f,-0.000231f,-0.001735f,-0.000615f,0.003027f,-0.000000f,0.000312f,-0.000000f,0.000570f,0.000125f,-0.000873f,0.000770f,-0.000278f,-0.000230f,0.001685f,0.001550f,-0.000222f,-0.000135f,-0.000669f,0.000677f,0.000004f,-0.000297f,-0.000000f,-0.002600f,0.000364f,-0.000378f,-0.001808f,-0.000195f,0.000389f,-0.000021f,0.001010f,0.000883f,0.000382f,0.000283f,0.001075f,-0.000094f,0.000000f,0.001487f,-0.000216f,0.000762f,0.000184f,0.000547f,-0.000989f,0.003845f,0.001217f,-0.000062f,-0.000097f,0.001141f,-0.000073f,-0.000153f,0.000589f,0.001175f,0.000542f,0.000000f,-0.000017f,-0.000247f,0.000630f,0.000278f,-0.000161f,-0.001267f,-0.000300f,0.000260f,0.000082f,0.009598f,0.000783f,-0.000466f,-0.000000f,0.001597f,0.000602f,-0.000085f,0.000669f,-0.000423f,-0.000000f,0.000200f,-0.000420f,0.001329f,-0.001668f,0.000242f,-0.000000f,0.000000f,0.001564f,0.000275f,-0.000650f,0.000452f,0.000160f,-0.000014f,0.000344f,-0.000064f,-0.000292f,0.000924f,0.000827f,0.000341f,0.000466f,0.001227f,-0.000304f,0.000029f,0.000052f,-0.000091f,0.000645f,-0.000000f,-0.000000f,-0.000426f,-0.000040f,-0.000000f,0.000530f,-0.000000f,-0.001118f,0.000320f,0.000000f,0.000540f,-0.000534f,0.000043f,0.000000f,0.000019f,
                    -0.000509f,-0.000218f,-0.000151f,0.001105f,0.001052f,-0.000394f,-0.000000f,-0.000793f,-0.000217f,-0.000207f,-0.000255f,0.000972f,0.000455f,-0.000006f,-0.000000f,0.000090f,0.000000f,0.000624f,-0.000009f,-0.000000f,-0.000057f,0.000137f,-0.000000f,0.000848f,0.000136f,0.000000f,-0.000773f,0.000045f,-0.000328f,-0.000630f,-0.000771f,-0.002052f,0.000141f,0.000000f,-0.000257f,0.001249f,0.000349f,-0.006660f,0.000520f,-0.000000f,0.000123f,0.000009f,0.000189f,0.000435f,-0.000582f,0.000000f,0.001536f,0.000671f,-0.000000f,0.001040f,-0.000399f,0.000503f,0.000000f,0.000178f,-0.000163f,-0.000143f,0.005337f,0.000100f,-0.000093f,0.000334f,-0.000273f,0.001802f,-0.000118f,0.000000f,-0.001764f,0.000136f,-0.000000f,-0.001160f,-0.002251f,-0.000350f,0.002668f,0.001495f,0.000154f,0.000369f,0.000071f,0.000106f,-0.000000f,-0.001874f,-0.000817f,0.000797f,0.000690f,0.001155f,0.000526f,0.001104f,0.000000f,0.000093f,-0.000000f,0.000425f,0.008186f,-0.000000f,-0.000748f,0.000000f,-0.000199f,0.173218f,-0.000924f,0.000097f,-0.000309f,-0.001104f,-0.000234f,-0.000000f,0.001250f,-0.000355f,-0.000191f,-0.000165f,0.000000f,-0.000000f,0.000468f,0.000571f,-0.001085f,0.000000f,-0.000166f,-0.000000f,0.000494f,0.000862f,-0.000130f,-0.000000f,0.000114f,0.000811f,-0.000059f,-0.000233f,0.001639f,-0.000323f,0.000089f,-0.000122f,0.000142f,-0.000266f,0.000327f,0.000050f,0.000164f,-0.000066f,-0.000060f,-0.000558f,0.000564f,-0.000061f,0.000135f,-0.000178f,-0.000562f,-0.000209f,-0.000256f,0.002281f,0.000115f,0.002090f,0.001317f,-0.000597f,-0.000708f,-0.000638f,-0.000221f,0.000711f,0.000658f,0.001254f,0.001197f,0.000343f,0.000297f,-0.000180f,-0.000723f,0.000000f,-0.000000f,-0.000570f,-0.000001f,0.000171f,-0.000020f,-0.000863f,0.000800f,0.000876f,-0.000122f,0.000000f,0.000796f,-0.000544f,-0.000656f,-0.000080f,0.000221f,0.001884f,-0.000278f,0.000140f,0.000101f,0.000394f,-0.000206f,0.000102f,0.002060f,0.000115f,0.000163f,-0.000228f,0.000934f,-0.000000f,0.000072f,0.000502f,0.000334f,-0.000204f,0.000000f,0.000407f,-0.000347f,-0.000494f,0.000196f,-0.000320f,0.000520f,0.000394f,0.000347f,0.000067f,0.000000f,-0.000261f,
                    -0.000680f,0.000886f,0.000351f,-0.000195f,0.000508f,-0.000009f,-0.000415f,0.002270f,-0.000042f,0.001098f,0.000196f,-0.000095f,0.000087f,0.000492f,0.000155f,-0.001172f,-0.000312f,0.000376f,0.000916f,0.000032f,-0.000000f,0.000262f,-0.000378f,0.000084f,0.000607f,-0.000127f,0.000211f,0.000257f,0.000289f,0.700684f,0.006351f,0.000607f,-0.000249f,-0.000274f,-0.000274f,0.000951f,0.000397f,-0.000418f,-0.000000f,-0.000382f,0.000749f,0.000345f,-0.000000f,-0.000156f,0.000046f,-0.000390f,-0.000384f,0.000582f,-0.000584f,0.000413f,0.001526f,0.000389f,0.000950f,0.000542f,-0.003010f,-0.000877f,0.000032f,-0.000306f,-0.000283f,-0.000097f,-0.000270f,-0.001265f,0.001453f,0.000000f,0.000612f,0.000056f,-0.001223f,-0.001437f,-0.002005f,0.001105f,-0.002584f,0.002726f,0.002846f,0.000632f,0.000219f,0.000053f,0.000472f,-0.000121f,0.000444f,0.000782f,-0.000000f,-0.000134f,0.002312f,-0.000319f,-0.000000f,-0.000471f,0.000000f,-0.002094f,0.000139f,-0.000042f,0.001167f,0.000518f,-0.000091f,-0.000123f,0.000157f,-0.000759f,0.000234f,0.000000f,-0.000902f,-0.000000f,-0.000000f,0.000087f,0.000122f,0.000000f,-0.000044f,0.000692f,-0.000323f,0.000193f,-0.000968f,0.000723f,0.000000f,0.000205f,0.001522f,-0.000075f,0.000000f,0.001086f,-0.000338f,0.002125f,0.002872f,0.000509f,0.000005f,-0.000000f,0.000365f,0.000231f,0.000161f,-0.000201f,-0.002144f,0.000540f,0.000000f,-0.002237f,0.000498f,0.002081f,0.000238f,-0.000350f,0.000000f,0.000146f,-0.000363f,0.000356f,-0.000254f,0.000549f,0.000096f,-0.000371f,-0.000293f,-0.000028f,-0.000000f,-0.000397f,0.000932f,0.003109f,0.000041f,-0.001254f,0.000414f,-0.000280f,0.000410f,-0.000000f,-0.000302f,0.011330f,-0.000107f,-0.000074f,-0.001053f,-0.000285f,-0.000530f,-0.000316f,-0.000000f,0.001433f,0.000577f,-0.000000f,0.000162f,0.000806f,-0.000118f,0.000037f,0.000607f,-0.000983f,-0.001205f,0.000488f,0.000282f,0.000125f,0.000288f,-0.000304f,-0.000000f,0.000080f,-0.000000f,0.001143f,-0.001396f,0.001566f,0.001278f,-0.000103f,-0.001292f,0.000279f,0.004322f,0.000491f,0.000732f,0.000000f,0.000123f,-0.000306f,-0.000000f,-0.000208f,-0.000022f,-0.000031f,0.000758f,0.001993f,
                    -0.000190f,0.002415f,-0.000126f,-0.000736f,-0.000000f,0.001819f,-0.000147f,-0.000049f,-0.001065f,0.000505f,-0.000000f,-0.000000f,-0.000000f,0.000020f,0.000532f,-0.000128f,0.000764f,0.000469f,0.000345f,-0.000264f,0.000106f,-0.000126f,-0.000018f,0.000231f,0.000038f,0.013023f,0.000594f,0.000000f,0.000000f,-0.000000f,-0.000141f,-0.000000f,-0.000378f,-0.000614f,0.001456f,0.001131f,-0.000943f,-0.003719f,0.000000f,0.000490f,0.000597f,0.000019f,0.000649f,0.000154f,0.001251f,-0.000547f,0.000198f,-0.001282f,-0.000714f,-0.000207f,-0.000359f,-0.001420f,0.000000f,-0.001767f,-0.000088f,0.002512f,0.001342f,-0.000000f,0.000907f,0.000000f,0.001078f,0.002670f,0.000000f,-0.000304f,0.000234f,0.000501f,0.000457f,0.000873f,0.000998f,-0.000855f,0.000033f,-0.000191f,-0.000000f,0.000316f,-0.000072f,-0.000297f,0.000041f,0.002460f,0.000162f,-0.001698f,0.001864f,0.000000f,0.000933f,0.000239f,-0.000129f,0.000708f,-0.000847f,0.000431f,-0.000753f,0.000329f,0.000000f,0.000000f,0.000000f,0.000199f,0.001232f,0.000481f,0.000408f,0.000149f,-0.000696f,-0.000000f,0.000563f,0.000000f,-0.000147f,0.000314f,0.000129f,-0.000535f,0.000253f,0.000428f,-0.000371f,-0.000037f,0.000577f,0.000236f,0.000245f,-0.000484f,0.000735f,0.001681f,-0.000037f,0.001938f,0.000373f,0.002800f,0.000000f,-0.000824f,-0.000282f,-0.000015f,-0.001261f,0.001299f,-0.000611f,0.001105f,0.000101f,0.000193f,0.000584f,0.000159f,0.000000f,0.000000f,-0.001880f,0.000000f,0.000036f,-0.000000f,0.000334f,0.000000f,0.000000f,0.000556f,0.000705f,-0.000412f,0.000000f,-0.001945f,0.000029f,-0.000000f,-0.000168f,-0.000624f,0.000354f,-0.000372f,0.008209f,0.000505f,-0.000705f,0.000066f,0.000744f,-0.000101f,0.000000f,0.000121f,0.000223f,0.000268f,-0.000000f,-0.000191f,-0.005104f,-0.000120f,0.000000f,0.000306f,0.000675f,-0.001177f,-0.000124f,0.000687f,0.000632f,-0.000306f,-0.000040f,-0.000522f,0.000624f,0.000449f,0.000161f,0.000813f,-0.000006f,0.000222f,0.000735f,0.000001f,-0.000725f,-0.000000f,0.000453f,0.000560f,0.000187f,-0.000407f,-0.000689f,-0.000000f,-0.000538f,0.000737f,0.007656f,0.000562f,-0.000157f,-0.000282f,0.000000f,0.001157f,
                    0.001420f,0.001243f,0.003092f,0.000039f,0.000408f,0.000668f,-0.000321f,-0.001228f,0.000200f,-0.000022f,-0.001376f,-0.000000f,0.000065f,-0.000230f,0.000000f,-0.000572f,0.000306f,-0.000000f,-0.000180f,-0.001853f,-0.000654f,0.000201f,-0.000979f,0.000077f,0.000473f,0.000000f,-0.000521f,-0.000977f,0.001208f,-0.000032f,0.000367f,0.000225f,0.000339f,0.000491f,-0.000000f,0.001346f,-0.000059f,0.000039f,0.000176f,0.000709f,0.000390f,0.033417f,0.000636f,0.000003f,-0.000144f,0.000000f,0.001304f,-0.000000f,-0.001721f,-0.000214f,-0.000213f,0.000000f,0.001469f,0.000113f,0.001186f,0.000250f,0.000055f,-0.000299f,0.000218f,0.000000f,-0.000213f,0.000294f,0.000127f,0.000344f,-0.000000f,-0.000064f,-0.000211f,-0.000000f,-0.000357f,0.000000f,-0.000272f,-0.000562f,0.000180f,0.000465f,-0.000190f,-0.000323f,-0.000000f,0.000158f,0.000002f,-0.000232f,0.000054f,-0.000000f,0.000619f,0.003450f,-0.001088f,-0.000000f,-0.001205f,0.001471f,0.002235f,-0.000097f,0.000847f,0.000321f,-0.000947f,0.000757f,-0.002958f,0.000812f,0.000769f,0.000041f,-0.000127f,-0.000000f,-0.000602f,-0.000973f,0.000665f,-0.001974f,-0.000000f,-0.000000f,-0.000128f,-0.000039f,-0.000510f,-0.000709f,-0.000211f,-0.000000f,-0.001292f,-0.001461f,0.001449f,0.000312f,-0.001525f,0.001801f,0.000756f,0.000000f,-0.001235f,-0.001155f,-0.002531f,-0.000104f,-0.000009f,-0.001113f,-0.000388f,0.000000f,-0.000649f,-0.000262f,-0.000831f,-0.000140f,0.000130f,-0.000460f,-0.001396f,0.000067f,-0.000059f,0.001021f,0.000564f,-0.000458f,-0.000376f,-0.000561f,0.000566f,-0.000519f,0.000268f,0.000068f,0.001543f,0.004131f,0.000432f,0.000113f,-0.000124f,0.000648f,0.000381f,0.001477f,-0.000186f,0.001852f,-0.000000f,-0.000293f,-0.000117f,-0.000405f,-0.001531f,-0.000000f,0.000149f,0.000289f,0.000000f,0.000000f,0.015167f,0.000265f,-0.000821f,0.000000f,0.001574f,-0.000000f,0.000056f,-0.000923f,-0.001677f,-0.001867f,-0.000555f,-0.000344f,-0.000201f,-0.000000f,-0.000000f,-0.000000f,0.004734f,0.000448f,-0.000553f,0.000355f,0.000204f,0.000111f,0.000165f,0.000000f,-0.000887f,0.000126f,-0.000150f,0.000261f,-0.000854f,-0.000417f,0.000046f,0.003426f,0.000579f,-0.000322f,
                    0.000408f,-0.000000f,0.000094f,0.000343f,-0.000889f,0.001013f,-0.000601f,0.000310f,0.000377f,0.000496f,0.000959f,-0.000011f,0.000388f,-0.000083f,-0.000000f,0.000213f,0.000000f,-0.000080f,0.000257f,0.000441f,0.002003f,0.000245f,-0.000000f,0.000064f,0.001790f,0.000310f,-0.000281f,-0.000000f,-0.000403f,-0.000000f,0.000318f,0.000000f,0.000825f,-0.000742f,-0.000000f,0.009033f,0.001087f,-0.000000f,0.000331f,-0.000000f,0.000000f,0.000439f,-0.000409f,0.000000f,-0.000088f,-0.001184f,-0.000139f,-0.000249f,-0.000408f,0.000000f,0.000129f,0.001745f,-0.000085f,0.016785f,0.000060f,-0.000260f,0.000934f,0.000150f,0.000000f,0.000000f,-0.000015f,0.000408f,0.000017f,0.000862f,0.000108f,-0.000108f,0.000350f,0.000025f,-0.000611f,-0.000776f,0.000530f,0.000198f,-0.000000f,0.000000f,-0.000332f,0.000443f,0.014366f,-0.000196f,0.000539f,-0.000316f,0.000012f,0.000089f,-0.000032f,0.000540f,-0.001429f,0.000056f,-0.000418f,0.001237f,-0.000098f,-0.000950f,-0.000827f,0.000724f,-0.000000f,-0.000351f,-0.000548f,-0.000313f,-0.000277f,0.000000f,0.000374f,-0.000067f,0.003859f,0.002890f,-0.000450f,0.000869f,0.000729f,-0.000000f,-0.000920f,0.000471f,-0.000678f,-0.000236f,0.000120f,0.001550f,-0.000006f,0.001454f,-0.000536f,-0.000314f,-0.000000f,0.000602f,-0.000000f,-0.000701f,-0.001197f,0.000000f,0.000484f,0.000462f,-0.000558f,0.000618f,-0.000222f,0.000165f,0.004868f,0.000000f,-0.000000f,0.000016f,-0.000210f,-0.000263f,0.000386f,-0.000309f,0.000014f,-0.000000f,0.000882f,0.001418f,-0.000328f,-0.000000f,-0.000242f,-0.000501f,0.000488f,0.000000f,-0.000000f,-0.000000f,0.000041f,-0.000691f,0.000511f,0.001126f,0.000421f,0.000721f,0.000835f,0.000822f,-0.000384f,0.000583f,-0.000000f,0.000282f,0.000000f,0.004124f,-0.000811f,0.000000f,0.000261f,-0.000407f,-0.001011f,0.000520f,-0.000943f,0.000834f,0.000046f,0.000166f,-0.002050f,0.001585f,0.000017f,-0.000000f,0.000000f,0.000454f,-0.000234f,0.000611f,-0.000000f,0.000177f,0.000345f,0.000918f,-0.000000f,0.000556f,0.001987f,0.000000f,0.000853f,0.000157f,-0.000295f,0.019089f,0.000953f,0.000044f,-0.000838f,-0.000668f,0.000293f,0.000000f,-0.000050f,-0.000262f,
                    -0.000443f,-0.000449f,-0.000200f,0.000369f,-0.000906f,0.216309f,-0.000125f,0.000000f,-0.000042f,-0.000465f,0.000983f,0.000807f,-0.000000f,-0.000839f,0.000000f,-0.000360f,0.000230f,0.000348f,-0.000968f,0.000660f,0.000161f,0.000000f,0.000131f,0.000380f,0.000048f,-0.000235f,-0.000159f,-0.002792f,0.000000f,-0.001375f,0.000811f,0.000029f,-0.000216f,0.001209f,0.000330f,0.000015f,-0.002708f,0.000448f,0.000999f,-0.000219f,-0.000000f,-0.000216f,0.000441f,-0.000417f,-0.000596f,0.000499f,-0.000735f,-0.000000f,0.025757f,-0.000000f,-0.000377f,-0.000008f,-0.001637f,-0.000000f,0.000357f,0.003891f,0.000363f,0.000000f,0.000471f,0.001694f,0.000220f,0.000220f,0.000839f,-0.000698f,-0.000074f,0.000175f,0.000134f,-0.001049f,0.000000f,0.000528f,0.000878f,-0.000190f,0.000330f,0.000120f,0.000359f,0.000379f,0.000066f,0.000241f,-0.000079f,-0.000096f,-0.002527f,0.001579f,0.000179f,0.001924f,0.000417f,-0.000486f,0.000469f,0.001222f,0.000000f,-0.000000f,0.000457f,-0.000540f,-0.000054f,0.000198f,-0.000020f,-0.000895f,-0.000462f,0.000187f,-0.000345f,-0.000664f,-0.000547f,0.000143f,0.000023f,-0.000214f,-0.000469f,-0.000491f,-0.000257f,-0.001111f,-0.000569f,-0.000399f,-0.000185f,0.000937f,0.000069f,-0.000239f,0.000001f,-0.000064f,0.000000f,0.000142f,0.001122f,0.000348f,-0.000000f,0.000000f,-0.000221f,-0.000387f,-0.001424f,0.002522f,-0.001461f,-0.000155f,0.000459f,-0.001436f,-0.000000f,0.000136f,0.000021f,0.000036f,-0.000176f,0.000281f,0.000426f,0.000000f,0.000110f,0.000186f,0.000784f,-0.000000f,0.000393f,-0.000381f,0.000000f,0.001321f,0.000513f,0.000000f,-0.000206f,-0.001364f,0.000000f,0.000277f,-0.000000f,-0.000538f,-0.000155f,-0.000000f,-0.000137f,0.000367f,0.000069f,-0.000000f,-0.000658f,0.000368f,-0.000663f,0.001846f,-0.000142f,-0.000685f,-0.001310f,-0.001232f,0.000029f,-0.000037f,-0.000028f,0.000000f,0.001301f,0.000007f,-0.000121f,-0.000000f,0.047180f,0.000978f,0.000737f,-0.000023f,-0.000000f,0.000137f,-0.002361f,-0.000881f,-0.000137f,0.000000f,0.000151f,0.001828f,0.002048f,-0.000000f,0.000427f,0.000048f,-0.000301f,0.000715f,0.000381f,0.000000f,0.000165f,0.000255f,-0.000092f,-0.000000f,
                    0.000439f,0.000616f,-0.000184f,0.000427f,0.000042f,0.001883f,0.000858f,0.001003f,-0.000514f,0.000172f,-0.000377f,0.000000f,-0.000540f,0.002361f,0.000481f,-0.000009f,-0.000000f,0.000132f,0.000000f,0.000489f,0.000894f,-0.000000f,-0.000635f,0.000000f,-0.000720f,0.000493f,0.000112f,0.006130f,-0.000262f,0.000201f,0.000206f,-0.000707f,0.000033f,0.000324f,-0.000000f,-0.000700f,0.000301f,0.000122f,0.000860f,-0.000157f,-0.000417f,0.000235f,-0.000340f,0.000021f,0.000013f,-0.000123f,-0.000120f,0.000270f,-0.000180f,0.001243f,0.001283f,0.000000f,0.006805f,0.000141f,-0.000272f,-0.000770f,-0.000292f,0.001656f,-0.000245f,-0.001616f,0.000875f,-0.000035f,-0.000007f,0.000008f,-0.000319f,0.000560f,-0.000173f,0.002796f,0.000298f,-0.000289f,-0.000799f,0.002638f,-0.000307f,-0.000000f,0.000000f,0.000440f,-0.000000f,0.000823f,-0.000050f,0.000812f,0.000000f,0.000000f,0.001103f,-0.000000f,0.000000f,0.002022f,0.000113f,0.000000f,0.000337f,-0.001052f,0.000345f,0.000000f,0.001040f,0.000110f,0.000968f,0.000000f,-0.000340f,0.000000f,-0.000648f,-0.000000f,0.000067f,0.000262f,-0.000423f,0.000264f,0.000672f,-0.000000f,0.001414f,0.012474f,0.000000f,0.000000f,-0.000325f,0.000183f,-0.000494f,-0.000113f,0.000057f,0.000539f,-0.001107f,-0.000660f,0.000886f,-0.000548f,-0.000000f,-0.000000f,0.000161f,0.000178f,-0.002335f,0.000000f,-0.000000f,-0.000000f,-0.000187f,0.001627f,-0.000639f,0.000722f,-0.000158f,-0.001049f,0.000317f,-0.000000f,0.000000f,-0.000163f,0.003515f,0.000257f,-0.000192f,0.001292f,0.001142f,0.000245f,0.000000f,0.001881f,0.000000f,-0.001064f,0.000443f,-0.000000f,-0.001523f,-0.000544f,-0.001088f,-0.000241f,-0.000049f,0.000017f,-0.000044f,-0.000233f,0.000835f,0.000354f,-0.000542f,0.000331f,0.000340f,0.000170f,0.001588f,-0.000000f,-0.000405f,0.000918f,0.001245f,-0.000463f,0.000041f,-0.000094f,-0.000008f,-0.000270f,-0.000399f,-0.000000f,-0.000187f,-0.000342f,0.001418f,0.000051f,-0.000904f,-0.002047f,0.000016f,0.000178f,0.000030f,0.000418f,0.000009f,0.000000f,0.000299f,-0.000069f,-0.000000f,0.000996f,0.001372f,-0.000040f,-0.000175f,-0.000347f,-0.000186f,0.001167f,0.002188f,0.000591f,
                    0.000368f,-0.000007f,0.000000f,0.001725f,-0.000002f,-0.001493f,-0.000356f,-0.000000f,0.000580f,0.000000f,-0.000000f,0.000000f,0.004440f,-0.000537f,-0.000388f,0.000144f,0.000108f,-0.000000f,0.001062f,-0.000579f,-0.000000f,0.000247f,-0.000000f,0.000315f,-0.000104f,-0.001192f,0.000114f,0.000455f,0.002310f,0.000566f,0.000388f,-0.000000f,-0.000042f,0.000000f,0.000534f,0.000014f,0.000665f,0.000004f,0.000064f,-0.000172f,-0.000216f,-0.000183f,0.002016f,-0.000283f,0.000407f,0.000000f,0.000261f,0.000225f,0.001617f,0.066162f,-0.000041f,0.000000f,-0.000056f,-0.000017f,-0.000152f,0.000039f,-0.000496f,-0.000215f,-0.000603f,-0.000319f,-0.000297f,-0.002457f,-0.000128f,0.001142f,0.000895f,-0.001428f,0.000000f,-0.000347f,0.000701f,0.000722f,-0.000872f,0.000000f,-0.001972f,0.000127f,0.000000f,0.000242f,0.001134f,-0.000159f,-0.000142f,-0.000689f,-0.000127f,-0.000283f,0.000185f,-0.000087f,-0.000182f,-0.000955f,-0.000596f,-0.000017f,0.000370f,0.000400f,-0.000000f,-0.000459f,-0.000449f,0.000000f,0.000715f,0.000710f,0.000391f,0.000000f,0.002420f,-0.000395f,-0.000314f,-0.000148f,-0.000718f,-0.000311f,-0.000276f,0.000457f,-0.000207f,0.002146f,0.000680f,-0.000550f,-0.000868f,0.000292f,-0.000404f,0.000000f,0.000008f,-0.000049f,-0.000071f,0.000140f,0.000316f,0.000222f,-0.000222f,-0.000422f,-0.001001f,0.000506f,-0.000201f,-0.000000f,0.000477f,0.000000f,-0.000252f,0.002531f,-0.000000f,0.001258f,-0.000000f,0.000782f,0.000202f,0.002367f,0.000272f,0.001245f,0.000108f,0.000214f,0.000156f,-0.000299f,-0.000130f,-0.000101f,0.003130f,-0.000939f,0.000002f,-0.000327f,-0.000491f,0.000048f,0.000255f,0.000252f,0.001058f,0.000000f,-0.000000f,-0.000000f,-0.000125f,-0.000484f,0.000718f,-0.000081f,-0.003632f,0.000098f,0.106873f,-0.000501f,-0.000083f,-0.000089f,-0.000054f,0.000048f,0.000070f,0.000257f,0.001449f,-0.000305f,-0.000281f,-0.000094f,-0.000069f,0.000492f,0.000000f,0.000251f,0.000120f,-0.003361f,0.000149f,-0.000933f,0.001290f,-0.000203f,-0.000273f,-0.000095f,0.001295f,0.000397f,-0.000414f,0.000037f,0.001457f,0.001119f,0.000139f,0.000000f,-0.000032f,0.002008f,-0.000148f,-0.000232f,-0.001205f,0.000165f,
                    -0.000021f,0.000474f,-0.000415f,0.000222f,-0.000311f,0.002398f,0.004299f,0.000000f,-0.000216f,0.000506f,-0.000000f,0.000077f,-0.000992f,-0.000473f,-0.001681f,-0.000162f,-0.000000f,-0.000726f,-0.000115f,0.000655f,0.000066f,-0.000513f,0.000793f,0.000088f,-0.000177f,0.000284f,0.000000f,-0.000247f,-0.000000f,-0.000640f,0.000000f,-0.000000f,-0.001040f,0.000253f,-0.000100f,-0.002441f,-0.000000f,-0.000123f,0.000249f,-0.000103f,-0.000935f,0.000229f,-0.000255f,0.000101f,0.000187f,0.000716f,-0.000017f,-0.000873f,-0.000000f,0.000013f,-0.000000f,-0.000240f,0.000000f,-0.000072f,-0.000669f,0.000245f,-0.000934f,-0.000884f,0.001157f,-0.000000f,-0.000626f,0.000903f,-0.000073f,0.000272f,-0.000000f,0.001894f,-0.000048f,-0.000930f,0.000038f,0.000001f,-0.000000f,0.001422f,0.000062f,-0.000147f,0.000704f,0.000086f,-0.000027f,0.000478f,0.000340f,0.000000f,0.002077f,0.000001f,0.000000f,-0.000019f,0.001011f,-0.000380f,0.000803f,-0.000299f,-0.000000f,-0.001102f,-0.000063f,0.000552f,-0.000183f,-0.001549f,-0.000000f,0.000296f,0.003540f,-0.000394f,0.000359f,0.000000f,0.000105f,0.000195f,-0.000599f,0.002871f,-0.000549f,0.000368f,0.000000f,0.000092f,-0.000029f,0.000047f,0.000261f,0.000094f,0.000154f,0.000000f,0.000259f,-0.000000f,-0.000104f,-0.000196f,-0.000237f,-0.000089f,-0.000000f,-0.000327f,-0.000194f,0.000541f,0.000000f,0.000509f,-0.000581f,-0.000076f,0.000215f,0.000807f,-0.000034f,0.002832f,0.000805f,0.000816f,0.000490f,-0.000000f,-0.000136f,0.000872f,0.004852f,-0.000741f,0.000140f,0.000846f,0.000966f,-0.000584f,0.000443f,0.000856f,0.001603f,0.000344f,-0.001466f,0.000390f,-0.000168f,-0.000329f,0.001369f,0.000529f,0.000207f,-0.001362f,0.000135f,0.000000f,0.000000f,-0.001728f,0.000000f,-0.000000f,0.000381f,-0.000000f,0.000186f,0.000003f,0.000002f,0.000000f,0.000141f,-0.000000f,0.000992f,0.000425f,-0.000019f,0.000637f,0.000394f,-0.000157f,-0.000139f,-0.000609f,0.002230f,-0.000467f,0.000002f,-0.001044f,0.000446f,-0.000000f,0.000352f,0.000052f,-0.000387f,0.000113f,0.000514f,-0.000370f,-0.000746f,-0.000278f,0.000133f,0.000205f,-0.001127f,0.000888f,0.000084f,-0.000684f,0.000214f,0.000128f,
                    0.000848f,0.000728f,-0.000133f,-0.000536f,0.000778f,0.000664f,0.000385f,0.009399f,0.000166f,-0.000659f,0.000207f,0.000696f,-0.000648f,-0.000699f,-0.002819f,-0.001145f,-0.000138f,-0.000117f,-0.000527f,0.000738f,0.000000f,0.001118f,-0.000000f,-0.000578f,0.000249f,0.000362f,0.000515f,-0.000924f,-0.000074f,-0.000352f,-0.000033f,-0.001880f,0.000751f,0.000383f,-0.001088f,0.000385f,-0.000219f,-0.000399f,-0.001333f,-0.000000f,0.014969f,0.000174f,-0.000384f,0.000353f,0.000226f,-0.000009f,0.000043f,0.000007f,-0.000000f,-0.000143f,0.002165f,-0.000001f,0.001302f,0.000140f,-0.000325f,-0.000541f,-0.000005f,0.000666f,0.000162f,0.000353f,0.000085f,0.000809f,0.002712f,-0.001192f,0.001047f,-0.000105f,-0.002228f,-0.000347f,-0.000517f,-0.000500f,-0.000000f,-0.000221f,-0.001064f,0.000000f,-0.001090f,0.000093f,0.000139f,0.000524f,0.000242f,0.000350f,0.001138f,0.000369f,0.000954f,-0.000057f,-0.000935f,-0.001020f,0.000366f,-0.001153f,-0.000000f,-0.000000f,0.000029f,0.000395f,-0.000104f,-0.000277f,0.000310f,-0.000330f,-0.000563f,0.000335f,-0.002439f,0.000078f,-0.000274f,0.000073f,-0.000194f,-0.000020f,0.003531f,0.000123f,0.000715f,-0.001363f,-0.001387f,0.002262f,0.000000f,-0.000489f,-0.000272f,0.000220f,0.001230f,0.000500f,0.000626f,0.000618f,0.001920f,0.000184f,-0.000466f,0.000154f,-0.000801f,0.000832f,-0.001386f,-0.000152f,-0.000000f,-0.000000f,0.000661f,0.001743f,-0.000245f,-0.000334f,-0.000023f,0.000220f,0.001596f,0.000381f,0.001266f,-0.000312f,0.000450f,0.000649f,0.001839f,-0.002436f,-0.000336f,0.000684f,0.000437f,0.000655f,-0.000000f,0.000000f,0.000467f,-0.000034f,-0.000572f,-0.001253f,0.000118f,-0.000132f,0.000006f,-0.000071f,0.000070f,0.000000f,-0.000080f,-0.000845f,-0.000982f,0.000000f,-0.000077f,-0.000142f,0.001520f,-0.000176f,0.000566f,-0.000511f,-0.000275f,0.000105f,-0.000000f,-0.000080f,0.001537f,-0.000310f,-0.000055f,-0.000000f,0.000414f,0.001705f,-0.000196f,0.000099f,-0.000453f,-0.000216f,-0.000168f,0.000328f,-0.000100f,-0.000000f,-0.000525f,-0.001198f,-0.000727f,0.000282f,-0.000200f,-0.000348f,0.000225f,-0.000047f,-0.000037f,-0.000299f,0.000535f,0.000262f,0.000988f,-0.000000f,
                    0.000393f,-0.000000f,-0.000463f,-0.000372f,0.000498f,0.000211f,0.000217f,-0.000664f,-0.000473f,-0.000154f,-0.000798f,0.000346f,0.000086f,0.001365f,0.000871f,-0.000000f,0.000000f,-0.000000f,0.000288f,0.000000f,0.000321f,-0.002365f,-0.000181f,0.001457f,0.000000f,0.000774f,0.000540f,0.000214f,-0.000213f,-0.000280f,-0.000600f,-0.000099f,0.000282f,0.000424f,0.000248f,0.000000f,0.001911f,0.000044f,0.000264f,0.000025f,0.000380f,-0.000000f,0.000304f,0.000441f,-0.002399f,0.000607f,0.000626f,-0.000419f,-0.000033f,0.000182f,0.000224f,-0.000237f,0.000080f,0.000501f,0.000387f,-0.000095f,0.000287f,0.000000f,0.000032f,-0.000216f,0.000353f,0.000358f,0.000000f,0.000046f,0.000200f,0.001922f,0.000186f,-0.002325f,0.000654f,0.000451f,0.000370f,0.000813f,0.000371f,0.000000f,0.000479f,-0.000251f,0.000365f,-0.000000f,0.001438f,-0.001675f,0.000432f,0.000157f,0.000156f,-0.000017f,-0.000000f,0.001469f,-0.000029f,-0.000535f,0.001140f,-0.000023f,-0.000471f,-0.000247f,0.000120f,-0.000000f,-0.000634f,0.000165f,0.000043f,0.000229f,-0.000826f,-0.000838f,-0.000298f,0.000029f,0.000321f,-0.000024f,-0.000710f,-0.000000f,0.000000f,0.000024f,0.000388f,-0.000311f,-0.000004f,0.000000f,-0.000105f,0.004570f,-0.000196f,0.001721f,0.000837f,0.001817f,-0.000903f,-0.000000f,-0.000000f,-0.000721f,-0.000131f,0.000000f,0.013649f,0.000008f,-0.000083f,0.000259f,0.000259f,-0.000021f,-0.000335f,0.000556f,-0.000234f,0.001370f,0.000438f,0.001074f,0.000299f,0.001145f,0.000991f,0.000940f,-0.000001f,-0.000000f,0.000152f,-0.000748f,0.000367f,0.000391f,-0.000229f,-0.000019f,0.000597f,-0.000350f,0.002563f,0.000090f,0.000102f,0.000062f,0.000092f,-0.000235f,-0.000000f,-0.000245f,-0.000486f,-0.000990f,-0.000122f,0.000104f,-0.000058f,-0.000090f,-0.000787f,-0.000661f,-0.000226f,0.000146f,0.000000f,0.000406f,-0.000000f,0.000000f,-0.000352f,0.000382f,0.000036f,-0.000350f,0.000184f,-0.001779f,-0.000000f,-0.000569f,0.000296f,-0.000401f,0.000149f,-0.000000f,-0.000988f,0.001358f,-0.000000f,0.000623f,-0.000775f,0.000541f,-0.000312f,-0.000437f,0.001817f,0.000971f,-0.000000f,0.000072f,-0.000745f,0.001266f,-0.000332f,-0.001106f,
                    0.001207f,0.000661f,-0.000615f,-0.000113f,0.000000f,-0.000091f,-0.000000f,0.003286f,0.001000f,0.000027f,0.000794f,-0.001164f,0.000371f,-0.000225f,-0.000794f,0.000488f,0.001399f,0.000808f,0.000379f,-0.000000f,-0.000396f,-0.000000f,0.000081f,-0.000129f,0.000000f,0.000000f,-0.000040f,-0.000765f,-0.000087f,-0.000137f,0.000017f,0.002836f,-0.000051f,0.002048f,-0.000000f,0.000000f,0.001714f,-0.000498f,0.000437f,0.000824f,-0.000000f,-0.000531f,0.000000f,0.000579f,-0.007381f,-0.000635f,0.002066f,-0.000955f,-0.000082f,-0.000108f,0.000755f,-0.000025f,-0.000132f,-0.000320f,0.000000f,-0.001346f,-0.001831f,-0.000981f,0.000422f,-0.000279f,-0.000000f,0.001266f,0.000965f,0.000361f,-0.000000f,0.001723f,-0.000372f,-0.000645f,0.001544f,0.000181f,0.002071f,0.000203f,-0.000570f,0.000233f,-0.000000f,-0.000220f,0.000772f,0.001138f,0.000138f,0.000029f,0.002733f,0.000660f,-0.000204f,0.000411f,0.000582f,-0.000000f,0.001337f,-0.000072f,0.000537f,0.000082f,-0.001040f,0.000000f,0.000000f,-0.000000f,-0.001373f,0.000935f,0.000525f,-0.000000f,-0.000620f,-0.001846f,-0.000901f,0.000913f,-0.000990f,0.001225f,0.000008f,0.000185f,0.000380f,-0.000249f,-0.000000f,0.000000f,0.000412f,0.000436f,0.000041f,-0.000000f,0.000313f,0.000498f,0.001095f,0.000663f,-0.000000f,0.000830f,-0.000401f,0.001311f,0.000621f,-0.000916f,0.000462f,0.001400f,0.000340f,0.003336f,0.001106f,0.000349f,0.001058f,0.000400f,0.001904f,-0.000363f,0.000323f,-0.000567f,0.000000f,0.000069f,-0.000051f,0.001110f,0.000464f,0.000391f,0.001460f,-0.001339f,0.000615f,-0.000000f,0.000000f,-0.000102f,0.000294f,0.000422f,-0.000087f,0.001247f,0.000382f,-0.000451f,0.000332f,-0.000416f,0.000837f,-0.000321f,0.000636f,0.000139f,-0.000959f,0.002935f,0.000000f,0.000445f,-0.000598f,-0.000000f,-0.000013f,-0.000762f,0.000223f,0.003532f,-0.000082f,-0.000382f,-0.000112f,-0.000101f,0.000491f,-0.000000f,-0.000311f,-0.000000f,-0.000400f,0.000000f,0.000050f,-0.000826f,0.000000f,-0.000000f,-0.000366f,0.001790f,-0.000348f,0.000000f,0.001921f,-0.000976f,0.000584f,-0.001104f,0.000138f,-0.000101f,-0.000000f,-0.000386f,-0.001117f,-0.000000f,-0.000804f,-0.000750f,
                    -0.001001f,0.001598f,-0.000000f,-0.000577f,0.001426f,0.000471f,-0.000000f,-0.000865f,-0.001413f,0.001453f,0.000000f,0.000582f,-0.000296f,0.000983f,-0.000091f,0.001507f,0.021805f,-0.000023f,-0.000751f,0.000372f,0.000334f,0.000389f,0.002430f,-0.000000f,0.000362f,-0.000000f,0.000281f,0.000392f,-0.001282f,-0.000292f,0.000008f,0.234009f,0.000482f,-0.000032f,0.000157f,0.000431f,0.000557f,0.000108f,-0.000319f,-0.000173f,0.001276f,0.000000f,-0.000954f,0.001265f,0.000678f,-0.000050f,-0.000000f,-0.001035f,-0.000247f,0.000074f,-0.000422f,0.000531f,-0.000222f,0.000808f,0.016174f,-0.000151f,0.000621f,-0.001322f,0.001707f,0.000507f,-0.000026f,-0.001101f,0.001500f,-0.000444f,-0.000127f,-0.000366f,0.000000f,0.000569f,0.000326f,0.000865f,-0.000000f,0.000092f,-0.000000f,0.000703f,-0.000129f,-0.000306f,0.000373f,0.000391f,-0.000000f,-0.001327f,-0.001166f,0.004654f,-0.000308f,0.000397f,0.000404f,-0.000491f,-0.000178f,0.000313f,0.000246f,0.000705f,-0.000818f,0.000775f,0.000000f,-0.000026f,0.000811f,-0.000243f,0.000713f,-0.000521f,0.004162f,-0.000303f,-0.000024f,0.000616f,-0.000000f,0.000469f,0.000162f,-0.000220f,0.001756f,0.000000f,-0.000235f,0.000290f,-0.000449f,-0.000493f,-0.000104f,0.000711f,0.008759f,0.000000f,0.000000f,-0.000363f,0.000980f,-0.000000f,0.000381f,0.000953f,-0.000449f,0.000089f,0.000332f,-0.000284f,0.000547f,0.001097f,-0.000525f,0.001017f,0.001225f,-0.000643f,0.000451f,0.002754f,-0.000193f,-0.000796f,0.000756f,-0.000630f,-0.000096f,0.000403f,-0.000055f,-0.000862f,-0.000793f,0.000000f,0.000230f,-0.000356f,0.000208f,0.000155f,-0.000035f,0.000088f,-0.000049f,-0.000080f,0.000614f,-0.000269f,-0.001082f,-0.001227f,0.001062f,-0.000000f,0.000706f,-0.000129f,-0.000000f,-0.000013f,0.000287f,-0.001146f,0.000000f,0.000367f,-0.000065f,-0.000414f,-0.001393f,-0.000000f,0.000000f,-0.000464f,0.001869f,-0.000265f,-0.000459f,0.000247f,0.002470f,0.001555f,-0.000746f,-0.000045f,0.000000f,0.000386f,0.000418f,0.000350f,0.000066f,-0.001513f,-0.000115f,-0.000090f,-0.000389f,0.000870f,0.000873f,0.000520f,-0.000771f,0.000083f,0.000546f,-0.000162f,-0.000262f,-0.000028f,-0.000387f,0.000182f,
                    0.000351f,0.000241f,-0.000432f,0.000322f,0.000473f,0.000060f,-0.000529f,-0.002186f,-0.001147f,-0.000343f,0.000275f,-0.002518f,0.000000f,0.000268f,-0.000000f,-0.001677f,-0.000365f,0.002275f,-0.000041f,0.000059f,0.000081f,0.000031f,-0.000899f,-0.000241f,0.000000f,0.000144f,0.000000f,0.000087f,0.000237f,0.001770f,0.000000f,-0.000130f,0.003075f,0.000000f,-0.001523f,-0.001786f,0.000312f,0.000575f,0.001591f,-0.000385f,0.002180f,0.000000f,0.001064f,-0.000370f,0.000240f,0.001180f,0.000151f,0.000340f,0.000199f,0.000809f,0.000245f,-0.000390f,0.000487f,-0.002001f,0.000237f,0.000920f,0.000000f,-0.000234f,0.000069f,0.000775f,0.000000f,-0.000000f,0.000041f,-0.000167f,0.000704f,0.000000f,0.000554f,0.000882f,-0.000125f,-0.000146f,0.000501f,-0.000862f,-0.001321f,0.000426f,0.000118f,0.003147f,0.001136f,0.000000f,0.000540f,0.001238f,0.001263f,-0.000204f,0.000028f,0.000030f,0.000000f,0.000000f,-0.000175f,0.000677f,-0.000385f,-0.000344f,0.000000f,0.000262f,-0.001550f,0.000205f,0.000097f,0.001577f,0.000320f,0.000005f,0.000252f,-0.000000f,-0.000529f,0.000437f,-0.001024f,0.001621f,-0.004318f,0.000662f,0.000268f,-0.000903f,-0.000191f,-0.000407f,0.000223f,-0.000296f,0.001970f,0.000000f,0.000487f,0.000000f,-0.000183f,0.000367f,-0.000148f,0.000603f,0.000625f,0.000000f,0.000417f,-0.000501f,0.000000f,0.000255f,0.000542f,0.000365f,-0.001449f,0.000634f,0.000000f,-0.000459f,0.000269f,0.000072f,-0.000601f,-0.000000f,0.000000f,0.000764f,-0.000000f,0.000466f,0.000529f,0.000000f,0.002748f,0.000000f,-0.000334f,0.000269f,0.001772f,0.000700f,-0.000000f,-0.001621f,0.001705f,0.000561f,0.000415f,0.001034f,0.001019f,0.000535f,0.000000f,-0.000252f,-0.001968f,-0.000000f,0.000199f,-0.000005f,-0.000459f,0.000356f,-0.002373f,0.000050f,0.000000f,0.000216f,0.001194f,0.000127f,-0.000377f,0.000913f,-0.000880f,-0.000138f,0.024734f,-0.000000f,0.000335f,-0.000000f,-0.000000f,0.000516f,-0.000368f,0.000659f,-0.000000f,0.000145f,0.000436f,-0.000060f,-0.000063f,-0.000413f,0.178223f,0.000021f,0.000573f,0.000390f,0.000286f,-0.000000f,0.000413f,0.000669f,0.000466f,-0.001186f,0.000000f,-0.001157f,
                    -0.000306f,0.002626f,-0.000435f,-0.000000f,-0.000053f,0.000644f,-0.000409f,-0.000134f,-0.000478f,-0.000175f,0.000221f,0.000444f,-0.000164f,-0.000432f,0.000602f,-0.003241f,0.001238f,-0.000057f,-0.001462f,0.000036f,0.000093f,0.000792f,-0.001329f,0.000036f,-0.000310f,0.000834f,-0.000015f,0.000686f,0.000000f,-0.000032f,0.000428f,0.000503f,-0.000548f,0.001116f,0.001261f,0.000322f,-0.000176f,0.001125f,0.000000f,-0.000482f,-0.001231f,0.001896f,0.000409f,0.000672f,0.000197f,0.000230f,-0.000000f,-0.000210f,-0.000100f,0.000014f,-0.000201f,0.000316f,-0.000000f,0.000161f,0.000285f,-0.000349f,0.000327f,-0.000359f,0.000000f,-0.000315f,0.000428f,0.000297f,-0.000105f,0.000594f,0.000552f,-0.001888f,0.000112f,0.000463f,0.000059f,0.000336f,0.000565f,-0.000181f,-0.000572f,-0.000000f,-0.000000f,0.000646f,-0.001030f,0.000150f,0.000137f,-0.000000f,-0.001859f,0.000170f,0.000500f,0.000081f,-0.000000f,0.001681f,-0.000000f,0.000770f,0.000229f,0.000198f,0.000321f,-0.000061f,-0.001327f,0.000052f,0.000160f,0.000284f,-0.000000f,-0.000813f,0.013596f,-0.000628f,0.000250f,-0.000027f,-0.000000f,-0.000179f,0.000155f,0.000116f,0.000439f,-0.000206f,0.002193f,-0.002415f,-0.000284f,0.000621f,0.000161f,0.000593f,-0.000325f,-0.000275f,0.000896f,0.001046f,0.000156f,-0.000597f,0.000053f,-0.000027f,-0.000694f,0.000623f,-0.000170f,0.000077f,0.001187f,0.000496f,-0.000667f,0.000084f,-0.000123f,0.000640f,0.000172f,-0.000470f,0.001222f,0.000565f,-0.000049f,-0.000165f,-0.000019f,-0.000218f,0.000000f,0.001105f,-0.000023f,-0.000000f,0.000769f,0.000000f,0.000001f,-0.000963f,-0.000626f,-0.000606f,-0.000000f,-0.000284f,-0.000833f,0.000000f,0.000203f,0.000177f,-0.000144f,-0.000391f,-0.000000f,0.000000f,-0.001134f,-0.001736f,-0.000151f,-0.000453f,-0.000031f,-0.000124f,-0.000253f,0.002161f,0.000000f,0.000592f,0.000048f,0.001359f,-0.000561f,-0.002340f,0.001422f,-0.000104f,-0.000024f,0.000936f,0.001965f,-0.000000f,-0.000977f,-0.000476f,-0.000000f,-0.000094f,-0.000405f,-0.000000f,0.006256f,0.000129f,0.000144f,-0.000174f,0.001146f,0.000199f,-0.000383f,0.001628f,0.001466f,-0.000549f,-0.001266f,-0.000382f,-0.001609f,0.001630f,
                    -0.001228f,-0.002317f,0.000033f,0.001936f,0.001186f,-0.000000f,0.000000f,0.000077f,-0.000108f,0.000072f,-0.000725f,-0.000798f,0.000313f,0.000400f,-0.000419f,-0.003405f,-0.000214f,0.000962f,-0.000193f,0.000040f,0.000452f,-0.000874f,0.000727f,0.000536f,0.001602f,-0.000494f,-0.001404f,-0.000118f,-0.000286f,-0.002316f,0.000772f,-0.000137f,-0.000479f,-0.000903f,-0.001332f,-0.001451f,-0.000530f,0.000603f,-0.002243f,0.000408f,0.000591f,0.000394f,-0.000285f,-0.000000f,-0.001516f,0.000186f,-0.000440f,0.001585f,-0.000944f,-0.000789f,-0.002134f,-0.000126f,-0.000437f,0.000093f,-0.001254f,-0.000000f,0.000192f,0.000906f,-0.000115f,0.000000f,0.000000f,-0.000261f,-0.000005f,0.000045f,-0.000320f,-0.000212f,0.000172f,0.000044f,-0.000324f,0.000328f,-0.000073f,-0.000477f,0.001532f,-0.000090f,0.001298f,0.000231f,0.001381f,-0.000000f,-0.000666f,0.000161f,0.000134f,0.000168f,-0.001843f,-0.000034f,0.000268f,0.000867f,0.000216f,0.000343f,-0.000000f,-0.003063f,-0.000032f,-0.000912f,0.000457f,-0.000391f,0.000049f,-0.000478f,0.000078f,-0.000000f,-0.000000f,0.001767f,0.000864f,0.002422f,0.000159f,0.000547f,-0.000156f,0.000000f,-0.000026f,-0.001245f,-0.000224f,0.000108f,0.000234f,0.001753f,0.000380f,0.000325f,0.000507f,0.000268f,0.000103f,0.000172f,0.002329f,-0.000547f,0.000046f,0.000000f,0.001220f,-0.000186f,-0.001673f,0.000065f,-0.000639f,0.000172f,1.239258f,0.000000f,0.000351f,-0.000206f,0.000051f,0.001283f,0.000360f,0.000240f,-0.000307f,-0.000996f,0.000264f,-0.000074f,-0.000000f,0.001062f,-0.000419f,-0.000875f,-0.000181f,-0.000183f,0.000382f,0.000827f,0.000741f,-0.000229f,0.000525f,0.001090f,-0.001548f,0.000402f,-0.001586f,-0.000004f,-0.000000f,-0.000937f,0.000266f,-0.000000f,-0.000979f,0.001507f,-0.000053f,0.000056f,0.001295f,0.000561f,-0.000129f,-0.000557f,0.000020f,0.001174f,-0.000268f,0.000030f,-0.000903f,0.000054f,-0.000456f,-0.002190f,0.000119f,0.000892f,0.000000f,0.000000f,-0.000515f,0.014954f,0.000414f,-0.000549f,0.000006f,0.002607f,-0.003361f,-0.000859f,0.000026f,0.000394f,-0.000077f,-0.000758f,0.007950f,0.000467f,0.000345f,-0.000468f,-0.000474f,-0.000460f,-0.000464f,0.000000f,
                    0.000346f,0.001453f,-0.000194f,0.000277f,0.000084f,-0.000000f,0.000000f,-0.000049f,-0.000247f,0.000000f,-0.001568f,-0.000355f,-0.000591f,0.000417f,0.000728f,-0.000298f,-0.000000f,-0.001698f,0.000027f,0.000832f,0.001651f,0.000000f,0.001567f,-0.000587f,0.000465f,0.000428f,0.000266f,0.000314f,-0.000000f,-0.001326f,0.000052f,0.000142f,0.000909f,0.000609f,-0.000652f,-0.000866f,0.000065f,0.000915f,-0.000000f,-0.000091f,-0.000000f,0.000314f,0.010864f,-0.000411f,-0.000122f,0.000953f,0.001280f,0.000000f,0.000813f,0.000361f,-0.000296f,0.000112f,-0.000000f,-0.000445f,0.005054f,0.000082f,-0.000000f,0.000047f,-0.001716f,-0.000291f,-0.000888f,0.000287f,0.012802f,0.000053f,-0.000794f,0.001135f,-0.000000f,-0.001078f,-0.000004f,0.000276f,0.000504f,-0.000370f,0.000350f,-0.000233f,-0.000435f,0.000000f,-0.000634f,0.000000f,-0.000264f,-0.000000f,0.000184f,0.000111f,0.000349f,-0.000532f,-0.000259f,-0.000000f,0.000411f,0.000536f,-0.000000f,-0.000000f,-0.000113f,-0.000464f,-0.000950f,-0.001149f,0.000000f,0.000103f,-0.001588f,-0.000000f,-0.000431f,-0.001163f,0.000226f,-0.000213f,0.000000f,0.000000f,0.000600f,-0.000271f,0.002266f,-0.000427f,0.000000f,-0.000118f,-0.000000f,-0.000121f,-0.001390f,0.000579f,0.000475f,-0.001596f,-0.000415f,-0.000000f,0.000135f,-0.000107f,0.000005f,0.000259f,-0.000316f,0.000792f,0.000110f,-0.000172f,0.000093f,0.000837f,-0.000196f,0.000196f,-0.000126f,0.000451f,0.001742f,0.000000f,0.000223f,0.002417f,0.000358f,0.000341f,-0.001658f,0.001090f,-0.000027f,-0.000386f,0.000678f,0.001307f,0.000025f,0.000000f,-0.000000f,0.000510f,-0.000772f,-0.000000f,0.000530f,-0.000301f,-0.000115f,-0.000000f,0.001403f,0.000927f,0.001358f,0.000826f,0.000687f,0.001922f,-0.001850f,0.000000f,0.000022f,0.002939f,-0.000217f,-0.000142f,-0.000742f,-0.000514f,0.003731f,-0.000000f,0.000133f,-0.000889f,-0.000907f,-0.000673f,0.000126f,-0.000010f,0.000800f,0.000000f,0.000290f,0.001491f,0.000208f,-0.000526f,-0.000000f,0.000619f,0.000794f,0.000242f,0.000746f,-0.000023f,-0.000947f,0.002748f,-0.001045f,0.000591f,-0.000283f,0.000611f,-0.000325f,-0.001488f,-0.000142f,-0.000000f,-0.000573f,-0.000535f,
                    0.000607f,0.001012f,-0.000000f,-0.000175f,-0.000000f,0.000679f,-0.000486f,-0.000000f,-0.000645f,0.002579f,0.000543f,0.000000f,0.000000f,-0.000561f,-0.000695f,-0.000000f,0.000530f,0.001375f,-0.000000f,-0.000116f,0.009949f,0.001501f,0.000056f,0.000521f,-0.000000f,0.000322f,-0.000581f,-0.000010f,0.000000f,0.000258f,0.000234f,0.000632f,0.000000f,-0.000000f,0.000553f,-0.000381f,0.000198f,0.000011f,0.002100f,-0.002817f,-0.001290f,0.000000f,-0.000000f,-0.000000f,-0.000723f,0.000340f,0.001665f,0.003563f,0.000020f,-0.000363f,0.000984f,-0.000241f,0.000000f,0.000487f,-0.002993f,0.000000f,-0.000306f,-0.000909f,-0.000628f,0.000072f,-0.000062f,0.000398f,0.000448f,0.000000f,0.000103f,0.001752f,0.000914f,-0.000305f,0.000908f,-0.000000f,-0.000431f,-0.000223f,0.000000f,-0.000288f,0.000080f,-0.000145f,-0.000585f,-0.000871f,0.000112f,0.000567f,0.000102f,0.000622f,0.000348f,0.000000f,-0.000229f,-0.000730f,-0.000339f,-0.000000f,-0.000000f,-0.000417f,-0.000674f,0.002476f,-0.000598f,-0.000271f,-0.000000f,0.000755f,-0.000347f,0.001043f,-0.000088f,0.000000f,0.000318f,0.000179f,0.000409f,0.000913f,0.007008f,0.000338f,0.000427f,-0.000256f,-0.000000f,-0.000000f,0.000023f,0.000557f,-0.000139f,-0.000000f,0.000288f,0.000529f,0.000000f,0.000666f,-0.000302f,0.000438f,-0.000862f,-0.000350f,0.000107f,0.000476f,-0.000000f,0.000598f,0.000007f,-0.000462f,-0.001009f,0.000000f,-0.000103f,-0.000000f,0.001554f,-0.000214f,0.000204f,-0.000283f,-0.002453f,0.000010f,-0.000495f,-0.000116f,0.000071f,-0.001261f,0.000000f,-0.000332f,-0.000000f,-0.000772f,-0.000615f,0.000369f,-0.000000f,-0.000785f,-0.000027f,-0.000838f,0.000323f,0.000391f,-0.000902f,0.000137f,-0.000187f,-0.000627f,-0.000222f,0.000000f,0.002457f,0.184570f,0.000416f,-0.000077f,-0.000662f,0.001065f,0.000678f,0.000023f,-0.000016f,-0.000638f,0.000000f,0.000159f,0.000047f,-0.000244f,-0.001257f,0.000103f,-0.000000f,-0.000746f,0.000217f,0.000362f,0.000489f,-0.000216f,0.001024f,0.001406f,0.000000f,-0.000231f,0.000058f,-0.000508f,0.000428f,0.000390f,-0.000117f,0.000181f,-0.000360f,0.000844f,0.000180f,-0.000865f,0.000871f,0.000294f,0.000000f,-0.000156f,
                    0.000391f,-0.001909f,0.000708f,-0.000166f,-0.000486f,0.002451f,-0.000144f,-0.001119f,0.000656f,0.000777f,-0.000257f,0.000136f,-0.000407f,0.000000f,-0.001064f,0.000375f,-0.000725f,0.000477f,0.000453f,0.001616f,-0.000000f,-0.000172f,0.001092f,0.000000f,-0.000520f,0.000558f,-0.000606f,-0.000617f,0.000016f,0.002001f,0.000162f,-0.000000f,0.000108f,0.007942f,0.000032f,0.000000f,0.000740f,0.000056f,-0.000216f,0.000232f,-0.000000f,-0.000000f,0.000444f,0.000333f,0.000193f,0.000000f,0.000236f,0.000000f,-0.000557f,0.000000f,-0.000371f,0.000247f,-0.000985f,0.000709f,-0.000000f,-0.001097f,-0.000000f,0.000385f,0.000325f,-0.000016f,0.000122f,0.000178f,-0.000116f,0.000043f,0.000169f,0.002178f,0.000410f,0.000761f,-0.000159f,0.000325f,0.000000f,0.000439f,-0.000185f,0.000930f,-0.001195f,0.000378f,0.000744f,0.000268f,0.000000f,0.001564f,0.000447f,-0.000113f,0.000428f,-0.000378f,-0.000345f,-0.000315f,0.001019f,-0.000771f,-0.000000f,-0.000806f,-0.000841f,0.001020f,-0.000060f,-0.003910f,0.000414f,-0.000000f,0.000092f,0.001176f,0.001289f,0.000236f,0.000166f,0.000454f,-0.000138f,0.000000f,0.000139f,0.000153f,0.000000f,-0.000000f,-0.000220f,0.000145f,0.000143f,0.000372f,0.000312f,-0.000000f,-0.000249f,-0.001184f,0.000909f,0.000736f,0.003464f,-0.000000f,-0.000657f,-0.000484f,0.000167f,0.000376f,0.000056f,0.000654f,-0.000319f,-0.000000f,-0.000277f,0.000000f,-0.000000f,0.000043f,-0.000076f,-0.000000f,0.000680f,0.000063f,0.000904f,0.000951f,-0.000000f,-0.000000f,0.001062f,-0.000000f,0.000118f,-0.000274f,0.000000f,-0.000139f,0.000000f,-0.000995f,-0.000000f,0.000005f,-0.000000f,-0.001521f,-0.000645f,-0.000115f,0.001889f,-0.000644f,0.007153f,0.000602f,0.000768f,-0.000089f,-0.000703f,-0.000400f,0.000483f,-0.001184f,0.000652f,-0.000000f,0.000000f,0.000046f,0.000398f,0.000961f,0.000000f,-0.000000f,-0.000064f,-0.000122f,0.000044f,0.000523f,0.000000f,-0.000213f,0.000286f,0.000753f,-0.000379f,0.000823f,0.000015f,-0.000561f,-0.000875f,0.000171f,0.020416f,-0.000232f,0.000079f,-0.000483f,-0.000018f,-0.001497f,0.000706f,-0.000691f,-0.000200f,-0.000442f,-0.000310f,0.000000f,0.000095f,0.000243f,
                    0.000000f,0.002857f,0.000282f,0.000626f,-0.000022f,-0.000000f,0.001094f,0.000195f,-0.001320f,0.107361f,-0.000000f,0.000445f,0.000118f,-0.000325f,-0.000000f,0.000803f,-0.001790f,0.000017f,-0.000095f,-0.002146f,0.000605f,-0.000053f,-0.000751f,-0.000940f,-0.000000f,0.001535f,-0.000331f,-0.000386f,-0.000036f,0.002106f,0.000247f,0.000018f,-0.000727f,-0.000624f,-0.000555f,-0.000000f,0.000817f,-0.000000f,-0.000000f,0.000528f,-0.001344f,-0.000788f,0.004906f,-0.000000f,-0.000106f,0.001728f,-0.000401f,0.000929f,-0.000229f,0.000650f,0.001180f,0.002880f,0.000000f,0.001618f,0.000436f,0.000133f,0.000267f,-0.003078f,-0.000458f,-0.000272f,0.002317f,0.000000f,0.000850f,-0.000014f,0.000000f,-0.000000f,0.000093f,0.000320f,0.003613f,-0.001118f,-0.001031f,0.002607f,-0.000043f,0.000000f,0.000005f,0.001443f,0.000576f,0.001699f,-0.000511f,0.000000f,-0.001403f,0.000000f,0.000352f,0.000000f,-0.000855f,0.000187f,-0.000000f,-0.000000f,0.109314f,0.000000f,-0.000471f,-0.000644f,-0.000466f,0.000000f,-0.000422f,-0.000000f,0.001330f,0.000000f,-0.000050f,-0.000416f,-0.000524f,0.000013f,0.000393f,-0.000000f,0.000302f,0.000452f,-0.000000f,0.000000f,0.000499f,-0.000631f,-0.000385f,-0.000000f,-0.000070f,0.000092f,0.002377f,0.000791f,0.000000f,0.001861f,0.000000f,-0.000184f,0.005169f,-0.000099f,-0.001707f,-0.000000f,0.000219f,0.000322f,0.000916f,-0.000000f,0.000973f,0.000116f,-0.000257f,0.001045f,-0.000383f,0.000097f,-0.000507f,0.001427f,0.000217f,-0.000000f,-0.001074f,0.000679f,0.000479f,0.000597f,0.000731f,-0.000106f,-0.000147f,0.001725f,0.000926f,-0.000421f,0.000498f,0.000715f,0.000287f,0.000394f,-0.000413f,0.000656f,0.000432f,-0.000628f,0.001422f,-0.000079f,-0.000000f,0.026581f,0.002409f,0.000765f,-0.001136f,-0.000224f,0.000116f,-0.000485f,0.000421f,0.000278f,0.001251f,-0.000000f,-0.000000f,-0.000114f,0.000672f,0.000771f,-0.000160f,-0.000853f,-0.000113f,-0.000012f,-0.000201f,0.000489f,-0.000812f,0.000498f,0.000000f,0.000484f,-0.000345f,0.000536f,0.000441f,0.002264f,0.000000f,-0.000440f,-0.002708f,-0.000622f,0.000924f,0.000224f,0.001499f,-0.000031f,0.000000f,0.000139f,-0.001027f,0.000120f,
                    0.000286f,0.000479f,0.000559f,0.000492f,-0.000215f,0.002028f,0.000000f,0.000107f,0.000145f,-0.001693f,0.000819f,-0.000249f,-0.002594f,-0.000000f,-0.000000f,-0.000946f,-0.000065f,-0.000119f,0.000389f,0.001073f,0.001611f,-0.000233f,-0.000615f,0.000000f,0.000512f,-0.000428f,-0.000382f,0.000588f,0.000000f,0.000450f,-0.000387f,0.000381f,-0.000546f,-0.000000f,-0.000129f,0.000000f,-0.001612f,0.000000f,0.000322f,-0.000136f,0.001190f,-0.000350f,0.000655f,0.000274f,0.000607f,0.000000f,-0.000660f,-0.000000f,0.000215f,-0.000000f,-0.001466f,-0.000343f,-0.000000f,0.001454f,-0.000942f,-0.000702f,-0.000821f,-0.000587f,0.000421f,-0.000506f,0.067993f,0.000220f,-0.000529f,0.000630f,-0.000267f,0.001254f,0.003431f,0.000260f,0.000092f,0.001698f,0.001150f,0.000228f,-0.000021f,0.001300f,-0.000145f,0.000890f,0.000319f,0.000000f,0.001187f,0.000000f,-0.000189f,-0.001677f,0.000224f,0.000324f,0.000506f,0.000139f,0.000299f,-0.000000f,0.000570f,-0.000320f,0.000286f,0.000138f,0.000000f,-0.000425f,0.003487f,-0.000374f,-0.000761f,0.004368f,0.001585f,0.000000f,-0.000265f,-0.000299f,0.000680f,-0.000447f,-0.000085f,0.213867f,-0.000000f,0.000672f,-0.000317f,-0.000115f,0.002047f,-0.000000f,0.000656f,-0.002102f,-0.000441f,-0.000641f,0.000372f,0.000025f,-0.000000f,0.001432f,-0.000040f,0.000118f,-0.000000f,0.000059f,0.000368f,-0.000300f,0.000140f,0.000405f,0.000348f,-0.000533f,0.000064f,-0.000000f,-0.000375f,0.000215f,0.000000f,0.000451f,-0.000151f,0.001627f,-0.000249f,-0.000383f,0.000363f,-0.000326f,0.000261f,0.000647f,0.002150f,0.000396f,0.000293f,-0.000090f,-0.000844f,0.000000f,-0.000000f,-0.000000f,0.000165f,0.000139f,0.001204f,-0.000000f,-0.000365f,-0.000347f,0.000638f,-0.000000f,-0.000546f,-0.000256f,0.000284f,-0.000000f,-0.000186f,0.000398f,0.000000f,-0.000050f,-0.000380f,-0.000000f,0.000438f,0.000291f,-0.000534f,0.001183f,-0.001150f,-0.000548f,0.000080f,-0.000646f,0.000285f,-0.000813f,0.002825f,0.000959f,-0.000994f,0.002762f,0.000282f,0.000662f,-0.000363f,0.000000f,0.000000f,-0.000161f,0.000213f,-0.000253f,-0.000224f,0.000638f,0.000690f,-0.000239f,-0.000000f,0.000360f,-0.000825f,0.000274f,
                    0.000319f,0.001196f,-0.001965f,-0.000737f,-0.000000f,-0.000013f,-0.000142f,-0.000083f,-0.001130f,-0.000190f,0.000340f,0.001015f,0.000000f,0.001334f,-0.000072f,-0.000529f,0.000838f,0.000765f,0.000083f,0.000000f,0.001265f,-0.001841f,0.000118f,0.002398f,0.000455f,0.000477f,0.000100f,0.000000f,0.000111f,0.000191f,-0.000070f,0.000000f,0.000414f,-0.001377f,-0.000638f,-0.000628f,0.001661f,-0.000284f,-0.000146f,-0.001164f,-0.000062f,0.000000f,-0.000199f,0.000244f,-0.000595f,0.000000f,0.000000f,0.000046f,-0.001088f,-0.000933f,0.000135f,0.000495f,-0.000115f,0.000764f,0.000364f,0.000006f,-0.000217f,0.000034f,-0.000348f,-0.000080f,0.000259f,0.000896f,0.000445f,-0.000293f,-0.000050f,0.000682f,-0.000359f,0.000958f,0.000139f,0.000592f,-0.003540f,-0.003139f,-0.000638f,-0.000525f,0.000356f,0.000493f,0.000214f,0.003450f,-0.000000f,0.001569f,-0.000000f,-0.001370f,0.000380f,-0.000885f,-0.000000f,-0.000201f,-0.000112f,-0.000522f,0.000799f,0.002827f,0.000186f,-0.000000f,-0.000297f,-0.000593f,-0.000000f,-0.000937f,0.000275f,0.000354f,0.000744f,-0.000207f,-0.000160f,0.001865f,-0.000492f,-0.000827f,-0.001242f,0.000741f,-0.000000f,-0.000614f,-0.000239f,0.002581f,0.000495f,-0.000478f,-0.000631f,-0.000055f,0.000000f,0.000908f,-0.001039f,0.000399f,-0.000416f,0.000737f,-0.000288f,0.000017f,-0.000166f,0.000099f,-0.000425f,0.000148f,0.000704f,0.000077f,0.004543f,-0.003674f,0.000000f,-0.000619f,0.000000f,-0.000183f,0.000427f,0.000362f,0.000000f,-0.000010f,0.000024f,-0.000115f,0.001015f,0.000233f,0.000136f,0.000030f,-0.000000f,0.000514f,-0.000690f,-0.000171f,-0.000120f,-0.000059f,0.000105f,0.000072f,0.000415f,-0.000513f,-0.000269f,-0.000152f,0.000262f,0.000000f,-0.000127f,-0.001283f,-0.000516f,0.000074f,0.000000f,0.000206f,-0.000152f,-0.000010f,-0.001541f,-0.000402f,-0.000456f,-0.000160f,0.000377f,0.000092f,-0.000212f,-0.000190f,-0.000086f,-0.001088f,0.000266f,0.000000f,-0.000995f,0.000205f,-0.000080f,0.003454f,-0.000485f,-0.000746f,-0.000235f,0.000339f,0.001345f,-0.000040f,0.000193f,-0.000667f,0.000409f,-0.000135f,-0.000089f,-0.000108f,-0.000281f,0.068054f,0.000314f,0.001976f,0.000298f,-0.000625f,
                    0.001031f,-0.000859f,0.000529f,0.000635f,-0.001348f,-0.000029f,-0.000799f,0.000865f,-0.000405f,-0.000278f,-0.000000f,0.000267f,-0.000150f,-0.000090f,-0.001451f,-0.000106f,0.000309f,0.000589f,-0.000106f,0.000000f,-0.000667f,0.000000f,0.000000f,0.000516f,0.000061f,-0.000872f,0.002512f,0.000612f,-0.000000f,0.000445f,0.001168f,-0.000402f,0.000224f,-0.001390f,-0.000280f,0.000148f,0.000727f,-0.000252f,-0.000694f,0.000160f,0.000187f,-0.000192f,-0.002409f,-0.000000f,0.000079f,-0.000122f,0.000000f,0.000000f,-0.000138f,-0.000223f,-0.000777f,-0.000032f,0.000505f,0.000074f,0.000214f,0.000411f,0.000347f,-0.001254f,-0.000000f,0.000000f,-0.000026f,-0.001093f,-0.001447f,-0.000071f,0.001581f,-0.000000f,0.000000f,-0.000671f,0.000156f,-0.000000f,0.001547f,-0.000981f,0.000407f,-0.000000f,0.000297f,-0.000779f,-0.000551f,-0.000677f,0.000503f,0.001194f,0.000627f,0.000523f,0.000255f,-0.001199f,0.001451f,-0.000108f,-0.000131f,0.000063f,0.000219f,-0.000253f,0.000200f,-0.000000f,0.000183f,-0.000265f,0.000000f,0.001809f,-0.001706f,0.000000f,0.000454f,0.000755f,0.001475f,-0.000426f,0.000000f,0.000467f,0.001117f,0.000646f,-0.000000f,-0.000494f,0.000078f,0.002794f,0.001646f,-0.000345f,-0.001285f,-0.000031f,-0.001010f,-0.000468f,0.000354f,0.000129f,-0.000492f,0.000025f,-0.000044f,-0.000033f,-0.000133f,0.003527f,-0.000680f,0.000540f,0.000833f,-0.000025f,0.000249f,-0.000036f,0.000000f,-0.000216f,-0.000618f,0.000204f,-0.000000f,0.003323f,0.000230f,0.000195f,-0.000130f,0.000000f,0.000437f,0.001410f,0.001213f,0.000047f,-0.000123f,-0.000000f,-0.000388f,-0.001305f,-0.000000f,-0.001726f,0.000595f,0.002165f,-0.000953f,-0.000000f,-0.001594f,-0.001989f,-0.000396f,-0.000000f,0.001521f,0.000638f,-0.000309f,-0.000000f,0.000454f,-0.000092f,0.000000f,-0.000181f,-0.000026f,-0.000000f,-0.000089f,-0.000343f,0.000364f,0.001266f,0.001177f,-0.000379f,0.000222f,0.000040f,-0.000416f,0.000119f,-0.000939f,-0.001220f,0.000823f,0.000387f,-0.000279f,0.000154f,-0.000272f,0.000323f,0.001060f,-0.001216f,-0.000000f,-0.001171f,0.000000f,0.000285f,-0.000046f,-0.000018f,0.000155f,0.000282f,0.001121f,0.000310f,0.006908f,0.000447f,
                    -0.000167f,-0.001019f,0.000906f,0.000000f,-0.000000f,-0.000000f,-0.000430f,-0.000048f,0.000903f,0.000000f,-0.000453f,0.000000f,-0.000375f,-0.000556f,-0.000778f,-0.000128f,0.001360f,0.000260f,0.000443f,-0.000328f,0.000350f,-0.000647f,-0.002113f,-0.000000f,-0.001422f,0.000489f,-0.000851f,-0.000146f,-0.000534f,-0.000223f,-0.000064f,0.000716f,0.000019f,-0.001241f,0.002783f,0.000112f,0.000358f,0.000000f,0.002298f,-0.000182f,0.000832f,-0.000147f,-0.000347f,0.000357f,0.001104f,-0.000058f,0.000000f,0.000207f,-0.000176f,0.000732f,-0.000224f,0.000000f,0.000365f,0.000060f,0.000226f,-0.000476f,0.000566f,0.000078f,-0.000065f,0.000237f,-0.000340f,0.000281f,0.000000f,-0.000899f,-0.000505f,0.000752f,-0.000000f,0.000285f,0.000294f,0.001452f,-0.000622f,0.002251f,0.000884f,-0.000244f,0.000198f,-0.000144f,-0.000620f,-0.000256f,-0.000405f,-0.000349f,-0.000086f,0.001054f,-0.000239f,-0.000216f,-0.000084f,-0.000000f,0.001275f,0.001794f,0.003820f,0.001413f,-0.000068f,0.000661f,-0.000116f,-0.000920f,0.000059f,-0.000746f,0.000356f,-0.000480f,0.000074f,0.000828f,0.000361f,0.001074f,0.001289f,0.000389f,-0.000215f,0.001045f,0.003250f,0.000302f,0.000738f,0.000059f,0.000174f,0.000599f,0.000022f,0.001505f,0.003756f,0.000600f,0.000623f,0.000909f,-0.000000f,-0.001657f,0.000000f,-0.000089f,0.001599f,0.000021f,0.000507f,-0.000000f,-0.000073f,0.000000f,0.000000f,0.000689f,0.000419f,-0.000227f,0.001599f,0.000034f,0.000183f,0.000839f,0.000895f,-0.000395f,0.001163f,-0.000586f,0.001925f,0.000049f,0.000118f,0.013283f,0.000350f,-0.000135f,-0.000102f,-0.000062f,0.000000f,0.002735f,-0.000236f,0.000143f,0.000676f,-0.000230f,0.000316f,0.000000f,0.000014f,-0.000379f,-0.000440f,0.000619f,-0.000237f,0.000000f,0.000866f,-0.000421f,0.000000f,-0.001550f,-0.000125f,-0.000678f,-0.002804f,-0.001179f,-0.002039f,0.000000f,0.000000f,0.000211f,0.000000f,0.000000f,-0.000415f,0.001620f,-0.000047f,0.000000f,0.002520f,-0.000000f,-0.000192f,0.000646f,-0.000049f,0.000070f,0.000488f,-0.001279f,-0.000121f,0.000214f,0.003725f,0.000000f,0.000550f,0.000519f,-0.000038f,0.000110f,-0.000693f,-0.000439f,-0.000594f,0.000334f,
                    0.000104f,0.000299f,0.001292f,0.001101f,-0.000000f,-0.000280f,0.000000f,0.000000f,-0.000331f,0.001797f,0.000622f,0.000335f,0.000582f,-0.000456f,-0.000000f,-0.001052f,0.000690f,-0.000187f,0.000158f,-0.000390f,-0.000206f,-0.000425f,-0.000000f,-0.000262f,-0.000222f,-0.000165f,0.000000f,0.001188f,0.000828f,-0.000502f,0.000000f,0.001132f,0.000278f,-0.001550f,0.000504f,-0.000652f,0.000335f,0.000092f,-0.000000f,-0.000000f,0.000124f,0.002649f,-0.000000f,0.000540f,0.000002f,-0.002722f,0.000659f,0.000000f,0.004269f,0.000332f,-0.000444f,-0.000469f,-0.000417f,-0.000000f,-0.000044f,0.000000f,0.000000f,0.000586f,-0.000144f,-0.000073f,0.000100f,0.000025f,-0.000000f,0.000034f,-0.000291f,0.002115f,0.000911f,-0.000285f,0.000000f,-0.000048f,0.000306f,0.000077f,-0.000296f,-0.001576f,0.000031f,-0.000342f,0.000408f,-0.000000f,0.000000f,0.000624f,0.000133f,0.000000f,0.000450f,0.001016f,-0.000000f,-0.000985f,0.000588f,0.000111f,0.000128f,0.000338f,-0.000000f,0.000919f,-0.000000f,0.000227f,0.000442f,-0.000120f,-0.000304f,0.000632f,-0.000439f,0.000220f,0.000000f,-0.001243f,-0.000039f,-0.000402f,-0.000098f,0.000476f,0.000000f,-0.000000f,-0.001370f,-0.000520f,-0.000000f,0.000266f,-0.000013f,-0.000220f,0.001890f,0.000193f,0.001383f,-0.000642f,0.000000f,-0.000393f,-0.000000f,0.000000f,0.000339f,-0.000744f,0.000105f,0.000152f,0.000533f,0.000772f,-0.000553f,-0.000000f,0.000528f,-0.001853f,-0.000022f,0.001376f,-0.000000f,0.000000f,0.000698f,-0.000293f,0.033905f,-0.000038f,0.000157f,0.003681f,-0.000264f,-0.001221f,-0.000000f,-0.000401f,-0.000752f,0.000000f,-0.000876f,-0.000135f,-0.000559f,0.000000f,0.000659f,0.000549f,-0.000721f,0.000000f,0.000133f,-0.000125f,0.000034f,-0.000885f,-0.000035f,0.000658f,-0.000678f,-0.000443f,-0.000030f,-0.000146f,-0.000551f,-0.000487f,-0.000184f,0.000678f,0.000390f,0.000315f,0.000000f,0.000000f,-0.000014f,-0.000000f,-0.000550f,-0.000611f,-0.000137f,-0.000424f,-0.000000f,0.000124f,0.000271f,-0.000117f,0.001192f,0.000674f,0.000250f,-0.000154f,0.000063f,-0.001001f,0.002247f,-0.000499f,0.000994f,-0.004002f,0.000193f,-0.000569f,0.000232f,0.000609f,0.000251f,0.000830f,
                    0.000507f,-0.000696f,-0.000527f,0.000685f,-0.002657f,0.000000f,-0.000322f,-0.000323f,0.000000f,0.001019f,0.002363f,-0.000204f,-0.002541f,-0.000000f,0.000331f,-0.001792f,0.001474f,-0.000382f,-0.001202f,0.001205f,-0.000708f,-0.000072f,-0.000155f,0.000491f,0.000063f,-0.001204f,0.000525f,-0.000211f,-0.000434f,-0.000299f,0.000680f,-0.000178f,0.000541f,0.000110f,-0.000758f,-0.000789f,0.001419f,0.000122f,-0.000501f,0.000525f,-0.000076f,0.001485f,0.001576f,-0.000213f,0.000244f,0.001040f,0.000032f,-0.000340f,-0.000859f,-0.000452f,0.000418f,0.000456f,0.000000f,-0.000265f,0.000453f,0.000601f,-0.000545f,0.000151f,0.000322f,0.000364f,0.000119f,0.000885f,-0.000479f,0.000415f,-0.000371f,0.000119f,0.000137f,0.000081f,-0.000067f,0.000000f,0.000406f,0.000000f,0.000790f,-0.000369f,0.000539f,-0.000000f,-0.000469f,-0.000461f,-0.000303f,0.001603f,0.000605f,0.000633f,0.001421f,-0.001662f,-0.000182f,0.000120f,-0.000036f,-0.000097f,0.000232f,-0.000615f,-0.000678f,0.000341f,-0.000867f,-0.000152f,0.001267f,-0.000670f,0.000232f,-0.000047f,-0.000000f,-0.000332f,0.001030f,-0.000579f,0.006699f,-0.006481f,-0.002453f,-0.000393f,0.002151f,-0.000000f,0.000098f,0.011810f,-0.001071f,0.000246f,0.000299f,-0.000056f,0.000770f,-0.000506f,0.000222f,0.001137f,-0.001750f,-0.000000f,0.000000f,-0.000566f,-0.001047f,-0.000770f,-0.000146f,0.000000f,-0.000869f,0.000657f,0.000030f,-0.000004f,-0.000301f,-0.000343f,0.000319f,-0.000195f,-0.000000f,0.000909f,-0.000228f,0.002069f,-0.001587f,0.000352f,0.000827f,-0.000502f,0.000282f,-0.000000f,-0.000000f,-0.000000f,-0.000350f,-0.001339f,-0.000123f,0.000181f,0.000350f,0.000860f,-0.000467f,-0.000762f,0.000739f,0.003004f,0.000296f,0.000000f,-0.000000f,0.000520f,0.000177f,-0.000092f,-0.000127f,-0.000000f,0.005840f,-0.000153f,-0.000243f,0.000000f,0.003756f,0.000154f,0.000208f,0.000000f,-0.000070f,0.000545f,-0.000372f,-0.000237f,-0.001052f,0.000172f,-0.000554f,0.000000f,-0.000326f,0.000650f,-0.000192f,0.000204f,0.000005f,0.000186f,0.000047f,0.000421f,-0.000490f,-0.000000f,0.000459f,0.000644f,0.000258f,-0.000913f,-0.000483f,0.000045f,-0.000000f,-0.000248f,0.000322f,-0.001740f,
                    0.001169f,0.000168f,0.000500f,0.000000f,0.000000f,0.000604f,0.000000f,0.000000f,-0.000473f,0.002729f,-0.000116f,-0.001003f,0.000382f,-0.000213f,-0.000119f,0.000368f,0.000350f,0.001284f,-0.000863f,0.000628f,0.001219f,-0.000144f,-0.000833f,0.000387f,0.000000f,0.000596f,0.000448f,-0.000762f,-0.000000f,-0.000087f,-0.001297f,0.000434f,-0.000519f,-0.000006f,0.000720f,-0.000163f,-0.000000f,-0.001081f,0.000505f,0.000113f,-0.000553f,0.000000f,-0.000875f,-0.000112f,-0.000546f,-0.000443f,-0.000235f,0.000000f,-0.000305f,0.000514f,0.000264f,-0.000593f,-0.000391f,-0.000154f,-0.000144f,0.000309f,0.000366f,-0.001020f,0.001106f,0.000804f,0.000946f,-0.001038f,-0.000696f,-0.000247f,0.000129f,0.004917f,0.000257f,-0.000712f,0.000000f,-0.000334f,-0.000135f,0.000000f,0.000540f,0.000437f,-0.000000f,-0.000114f,0.046539f,0.000525f,0.000000f,0.000000f,0.000129f,-0.000317f,-0.000190f,0.000327f,0.000734f,0.002848f,-0.000599f,0.000105f,0.000000f,-0.000287f,0.000668f,-0.000059f,0.003160f,-0.000922f,-0.000397f,-0.000073f,0.000097f,-0.000479f,0.000937f,-0.000381f,0.000202f,-0.000136f,0.000000f,0.000000f,0.003267f,0.000028f,-0.000136f,0.000535f,-0.000753f,0.000000f,0.000360f,-0.000299f,-0.001027f,-0.001056f,0.000257f,0.000001f,-0.000000f,0.003241f,0.000379f,0.000197f,-0.000485f,-0.000472f,0.001948f,0.000000f,-0.000603f,0.000412f,0.001087f,0.001646f,0.001459f,0.000937f,-0.001011f,0.001466f,0.000304f,-0.000218f,-0.000000f,-0.000256f,0.000357f,0.000277f,0.000000f,0.000420f,0.000000f,0.000329f,-0.001644f,0.000000f,-0.001905f,-0.001328f,0.000214f,-0.000576f,0.000501f,-0.000000f,0.000825f,0.001347f,0.000000f,-0.000098f,0.000158f,0.000052f,-0.001580f,0.000000f,0.001038f,0.000423f,-0.000000f,-0.000446f,0.000271f,0.000805f,-0.000353f,0.000527f,0.000265f,0.003620f,-0.000163f,0.000091f,0.000536f,0.001041f,0.000203f,0.000592f,-0.000242f,0.000000f,-0.001410f,0.000257f,-0.000247f,-0.000181f,-0.000103f,-0.000200f,-0.000000f,-0.000000f,0.000310f,-0.000293f,-0.000494f,-0.000484f,-0.000870f,0.000000f,0.001208f,-0.000230f,-0.000000f,-0.000371f,0.001972f,0.000344f,0.000342f,-0.000278f,-0.000597f,-0.000000f,
                    -0.000332f,-0.000347f,-0.000000f,-0.000000f,0.002548f,-0.000121f,0.000006f,-0.000891f,-0.000901f,-0.000080f,-0.000000f,0.000079f,-0.000503f,-0.000000f,0.001618f,0.002111f,-0.000330f,-0.000235f,0.000017f,0.000495f,0.001271f,0.000000f,0.000457f,-0.000000f,0.000332f,0.000690f,-0.000591f,0.000000f,-0.000164f,-0.000244f,0.000035f,0.001530f,0.000103f,-0.000108f,-0.000551f,0.000800f,-0.000521f,-0.000587f,-0.001179f,0.000020f,0.000196f,0.001382f,0.000484f,-0.000253f,0.001094f,-0.001014f,-0.000000f,0.000000f,-0.000226f,0.000079f,0.000292f,0.000000f,-0.000071f,0.000324f,0.000455f,0.000205f,0.000411f,-0.000157f,0.000902f,-0.000947f,-0.000699f,-0.001837f,0.000148f,-0.000000f,0.001656f,0.000762f,-0.002701f,-0.000362f,0.002102f,0.000017f,-0.001532f,-0.000000f,0.000422f,0.000498f,-0.000058f,-0.001094f,0.003344f,-0.001373f,0.000000f,0.000257f,0.000235f,-0.000000f,-0.000309f,0.000059f,-0.000000f,0.000373f,0.000525f,0.001069f,0.000577f,0.000273f,-0.000160f,-0.000000f,0.001537f,-0.001878f,-0.000118f,0.000228f,-0.000092f,-0.000627f,0.002022f,0.000022f,0.000153f,-0.001369f,0.000242f,-0.000065f,0.001358f,-0.000273f,0.000106f,0.000483f,0.000353f,0.000000f,0.000862f,-0.000052f,-0.000259f,0.000644f,-0.000043f,0.001466f,-0.000090f,0.000382f,0.000099f,0.000000f,0.000356f,-0.000317f,0.000645f,0.000000f,-0.001290f,0.000217f,0.000250f,0.000212f,-0.000037f,0.001006f,-0.000000f,0.000008f,0.001755f,0.000467f,-0.000191f,0.000896f,-0.000531f,0.000744f,-0.001710f,0.000788f,-0.000978f,0.000093f,0.001918f,0.000000f,0.001148f,-0.001588f,0.000123f,-0.000350f,-0.000000f,-0.000000f,0.000143f,-0.000083f,-0.000523f,0.001004f,-0.000365f,0.001415f,0.000941f,-0.000630f,-0.000341f,0.000198f,0.000000f,0.000119f,0.001439f,0.000640f,0.001301f,0.001192f,0.002079f,0.000076f,-0.000044f,0.000000f,0.000609f,0.000000f,0.000197f,-0.000000f,-0.000437f,-0.000000f,-0.000469f,-0.000000f,0.000833f,0.000113f,0.000824f,-0.000852f,-0.000296f,0.000000f,-0.000000f,0.000262f,0.000421f,-0.000042f,0.003515f,-0.000200f,0.000181f,0.000319f,-0.000816f,0.000187f,0.014336f,0.000000f,-0.000700f,-0.000362f,-0.000278f,0.000053f,
                    0.000021f,0.000893f,0.000834f,0.000765f,-0.001699f,0.000013f,0.000000f,-0.004784f,0.000322f,0.000525f,0.000000f,-0.000000f,-0.000089f,0.002968f,-0.000558f,0.000070f,0.000291f,-0.000124f,-0.000026f,0.000203f,0.000709f,-0.000327f,0.114075f,0.000130f,-0.000287f,-0.000060f,-0.000301f,0.033081f,0.000183f,-0.001087f,-0.000218f,0.003389f,0.001100f,0.000381f,-0.000887f,0.000016f,-0.000000f,0.000240f,-0.000000f,0.000291f,-0.000059f,-0.003006f,0.000324f,-0.000697f,0.000787f,-0.000775f,0.000078f,-0.000461f,-0.000120f,0.001340f,-0.000198f,-0.000302f,0.000504f,0.000798f,0.000185f,0.000820f,-0.001308f,-0.000130f,-0.000346f,0.000153f,0.000448f,-0.000392f,0.000072f,0.000556f,0.000000f,-0.000000f,-0.000200f,0.000864f,-0.001100f,0.000109f,-0.000303f,-0.000000f,-0.005653f,0.000698f,0.000951f,-0.001178f,-0.000567f,0.000193f,0.000000f,0.000831f,0.001461f,0.000495f,0.000847f,0.000000f,-0.001231f,0.000262f,0.000181f,0.000880f,0.000471f,0.000684f,-0.000196f,0.001532f,-0.000930f,-0.002333f,-0.000260f,0.001986f,-0.000290f,0.000534f,0.002054f,-0.000163f,0.000450f,0.000000f,-0.000209f,-0.000254f,-0.000422f,-0.000451f,-0.000066f,0.000070f,-0.000000f,-0.000029f,0.000345f,0.002066f,0.000299f,0.000323f,-0.000342f,0.001264f,0.000412f,-0.000331f,-0.000159f,0.001341f,-0.000170f,-0.000253f,0.000426f,-0.000000f,0.001675f,-0.000000f,0.000656f,0.000839f,-0.000714f,-0.000000f,0.000273f,0.000329f,-0.000375f,0.001361f,0.000533f,-0.001837f,-0.000375f,0.000362f,-0.000102f,0.000359f,-0.000197f,0.000876f,0.000600f};

                half sum1 = 0.f;
                half sum2 = 0.f;
                uint idx = 0;
                __attribute__((opencl_unroll_hint(1)))
                for (uint ni = 0; ni < iterations; ++ni) {
                    half temp = 0.f;
                    unroll_for(uint ki = 0; ki < (TILE_IFM * SIMD) / TILE_K; ++ki) {
                        unroll_for (uint kii = 0; kii < TILE_K; ++kii) {
                            const uint total_k = ki * TILE_K + kii;
                            sum1 += acc_var_arr2[idx];
                            temp += acc_var_arr2[idx];
                            idx += 1;
                        }
                    }
                    sum2 += temp;
                }
                printf("3.Check sum sum1[%f] vs sum2[%f]\n", sum1, sum2);
            }
#endif
        }
    }
    // =====================================================================================================================================
    // Leftovers
#if MAIN_LOOP_ELEMENTS_COUNT % (TILE_IFM * SIMD) != 0
    // Handle leftovers in normal case without alignment correction.
    #define LEFTOVER_IFM               (MAIN_LOOP_ELEMENTS_COUNT % (TILE_IFM * SIMD))
    {
        #define LOAD_IN_0(bi) do {                                  \
                in_0[bi] = INPUT_BLOCK_READ(input, input_offset);   \
                input_offset += TILE_IN_B_PITCH;                    \
            } while (false)

        CONST_LOOP(TILE_B, LOAD_IN_0);
        #undef LOAD_IN_0
        input_offset += TILE_IFM * SIMD - TILE_IN_B_PITCH * TILE_B;
        unroll_for(uint ki = 0; ki < CEIL_DIV(LEFTOVER_IFM, TILE_K); ++ki) {
            #if USE_SLM
                FILTER_VEC_TYPE wei = 0;
            #endif

            #if COMPRESSED_WEIGHTS_INT4
                FILTER_PACKED_VEC_TYPE wei_packed = FILTER_BLOCK_READ(weights, weights_offset);
                wei = UNPACK_INT4(ACCUMULATOR_TYPE, *((INT4_PACKED_TYPE*)&wei_packed));
            #else
                wei = TO_FILTER_VEC_TYPE(FILTER_BLOCK_READ(weights, weights_offset));
            #endif

            #if COMPRESSED_WEIGHTS
                ACCUMULATOR_TYPE* w = (ACCUMULATOR_TYPE*)(&wei);
                unroll_for(uint kii = 0; kii < TILE_K; ++kii) {
                    unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
                        uint offset_ofm = out_f + fi*SIMD + get_sub_group_local_id();
                        #if DECOMPRESSION_SCALE_GROUPS_NUM > 1
                            const uint scale_offset = (offset_ofm % DECOMPRESSION_SCALE_BATCH_NUM) * DECOMPRESSION_SCALE_BATCH_PITCH +
                                                      ((kii + ki*TILE_K + iterations*TILE_IFM*SIMD) / DECOMPRESSION_SCALE_GROUP_SIZE)*DECOMPRESSION_SCALE_FEATURE_PITCH;
                            ACCUMULATOR_TYPE ds = decompression_scale[scale_offset];
                        #else
                            ACCUMULATOR_TYPE ds = d_scales[fi % DECOMPRESSION_SCALE_LENGTH];
                        #endif

                        #if DECOMPRESSION_ZP_TERM
                            #if DECOMPRESSION_ZP_SCALAR
                                ACCUMULATOR_TYPE dzp = DECOMPRESSION_ZP_VALUE;
                            #elif DECOMPRESSION_ZP_GROUPS_NUM > 1
                                const uint zp_offset = (offset_ofm % DECOMPRESSION_ZP_BATCH_NUM) * DECOMPRESSION_ZP_BATCH_PITCH +
                                                    ((kii + ki*TILE_K + iterations*TILE_IFM*SIMD) / DECOMPRESSION_ZP_GROUP_SIZE) * DECOMPRESSION_ZP_FEATURE_PITCH;
                                ACCUMULATOR_TYPE dzp = decompression_zp[zp_offset];
                            #else
                                ACCUMULATOR_TYPE dzp = d_zps[fi % DECOMPRESSION_ZP_LENGTH];
                            #endif
                        #else
                            ACCUMULATOR_TYPE dzp = ACCUMULATOR_VAL_ZERO;
                        #endif
                        w[W_IDX] = (w[W_IDX] - dzp) * ds;
                    }
                }
            #endif
            #if TILE_OFM == 1 && FILTER_LAYOUT_OS_IS_YX_OSV32_ISV2
            weights_offset += TILE_K_OFM_PACKED * SIMD * 2;
            #else
            weights_offset += TILE_K_OFM_PACKED * SIMD;
            #endif

            unroll_for (uint kii = 0; kii < TILE_K; ++kii) {
                unroll_for (uint fi = 0; fi < TILE_OFM; ++fi) {
                    unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
                        const uint total_k = ki * TILE_K + kii;
                        if (total_k < LEFTOVER_IFM) {
                            INPUT0_TYPE in_val = _sub_group_shuffle(((INPUT0_TYPE*)(&in_0[bi]))[total_k / SIMD], total_k % SIMD);
                            #if TILE_OFM > 1
                            ((ACCUMULATOR_TYPE*)(&acc[bi]))[fi] += in_val * ((ACCUMULATOR_TYPE*)(&wei))[W_IDX];
                            #else
                            acc[bi] += in_val * ((ACCUMULATOR_TYPE*)(&wei))[W_IDX];
                            #endif
                        }
                    }
                }
            }
        }
    }
    #undef LEFTOVER_IFM
#endif // MAIN_LOOP_ELEMENTS_COUNT % (TILE_IFM * SIMD) != 0
    // =====================================================================================================================================
    // Post-processing: bias, activation, fused-ops
    ACTIVATION_VEC_TYPE activated[TILE_B] = { };
    for (uint bi = 0; bi < TILE_B; ++bi) {
        activated[bi] = TO_ACTIVATION_VEC_TYPE(acc[bi]);
    }

// #if BIAS_TERM
//     #if TILE_OUT_F_NUM % (TILE_OFM * SIMD) == 0
//         BIAS_VEC_TYPE bias = BIAS_BLOCK_READ(biases, out_f);
//     #else
//         BIAS_VEC_TYPE bias = 0;
//         unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
//             ((BIAS_TYPE*)(&bias))[fi] = biases[out_f + sglid + fi * SIMD];
//         }
//     #endif
//     unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
//         activated[bi] += TO_ACTIVATION_VEC_TYPE(bias);
//     }
// #endif

    OUTPUT_VEC_TYPE result[TILE_B] = { };
// #if HAS_FUSED_OPS
//     unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
//     #if TILE_OFM > 1
//         unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
//             FUSED_OPS_VEC;
//             result[bi][fi] = FUSED_OPS_RESULT_VEC;
//         }
//     #else
//         FUSED_OPS_SCALAR;
//         result[bi] = FUSED_OPS_RESULT_SCALAR;
//     #endif // TILE_OFM > 1
//     }
// #else
    unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
        result[bi] = TO_OUTPUT_VEC_TYPE(ACTIVATION_TYPED(activated[bi], ACTIVATION_PARAMS_TYPED));
    }
// #endif
    // =====================================================================================================================================
    // Write results
    uint output_offset = out_f * TILE_OUT_F_PITCH + out_b * TILE_OUT_B_PITCH + OUTPUT_OFFSET;

    if (USE_BLOCK_WRITE && (TILE_OUT_F_NUM % (TILE_OFM * SIMD) == 0 || out_f + (TILE_OFM * SIMD) <= TILE_OUT_F_NUM)) {
#if IS_DYNAMIC
        #define WRITE_OUTPUT(bi) do {                                       \
                if (bi + out_b < BATCH_SIZE)                                \
                    OUTPUT_BLOCK_WRITE(output, output_offset, result[bi]);  \
                output_offset += TILE_OUT_B_PITCH;                          \
            } while (false)
#else
        #define WRITE_OUTPUT(bi) do {                                       \
                OUTPUT_BLOCK_WRITE(output, output_offset, result[bi]);      \
                output_offset += TILE_OUT_B_PITCH;                          \
            } while (false)
#endif
        CONST_LOOP(TILE_B, WRITE_OUTPUT);
        #undef WRITE_OUTPUT
    } else {
        output_offset += sglid;
        for (uint bi = 0; bi < TILE_B; ++bi) {
            for (uint fi = 0; fi < TILE_OFM; ++fi) {
                const bool should_write =
#if IS_DYNAMIC
                    bi + out_b < BATCH_SIZE &&
#endif
                    (TILE_OUT_F_NUM % (TILE_OFM * SIMD) == 0 ||
                    out_f + fi * SIMD + sglid < TILE_OUT_F_NUM);
                if (should_write) {
                    output[output_offset] = ((OUTPUT_TYPE*)(&result[bi]))[fi];
                }
                output_offset += SIMD;
            }
            output_offset += TILE_OUT_B_PITCH - TILE_OFM * SIMD;
        }
    }
    // =====================================================================================================================================
}

// Dyc Quantize
#if USE_SLM && DYNAMIC_QUANTIZE
#define PACKED_DQ_TYPE                      int
#define DQ_VEC_TYPE                         MAKE_VECTOR_TYPE(DQ_TYPE, TILE_IFM)
#define DQ_SLM_FILTER_VEC                   MAKE_VECTOR_TYPE(DQ_TYPE, 4)
#define DQ_SLM_FILTER_PACKED_VEC            MAKE_VECTOR_TYPE(FILTER_TYPE, FILTER_LOAD_BLOCK_SIZE)
#define DQ_SLM_FILTER_UNPACKED_VEC          MAKE_VECTOR_TYPE(DQ_TYPE, FILTER_ELEMENTS_PER_LOAD)
#define DQ_FILTER_VEC_TYPE                  MAKE_VECTOR_TYPE(DQ_TYPE, TILE_K_OFM)

#define TO_DQ_TYPE(x)                       CAT(CAT(convert_, DQ_TYPE),_sat)(x)
#define TO_DQ_VEC_TYPE(x)                   CAT(convert_, DQ_VEC_TYPE)(x)
#define TO_DQ_SLM_FILTER_UNPACKED_VEC(x)  CAT(convert_, DQ_SLM_FILTER_UNPACKED_VEC)(x)
#define TO_DQ_FILTER_VEC_TYPE(x)            CAT(convert_, DQ_FILTER_VEC_TYPE)(x)

#define AS_TYPE_N_(type, n, x)  as_##type##n(x)
#define AS_TYPE_N(type, n, x)   AS_TYPE_N_(type, n, x)
#define AS_DQ_TYPE_4(x)         AS_TYPE_N(DQ_TYPE, INPUT_LOAD_SIZE, x)

inline void FUNC(fc_bf_tiled_kernel_dyn_quan)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global char* quantized_input,
    __global INPUT0_TYPE* scale,
#if DECOMPRESSION_SCALE_TERM
    const __global DECOMPRESSION_SCALE_TYPE* decompression_scale,
#endif
#if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
    const __global DECOMPRESSION_ZP_TYPE* decompression_zp,
#endif
    __global OUTPUT_TYPE* output,
    const __global FILTER_TYPE* weights
    , __local int* wei_local_mem
#if BIAS_TERM
    , const __global BIAS_TYPE* biases
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
    uint gid = (uint)get_group_id(0);
    uint local_id = (uint)get_local_id(2);
    uint sglid = (uint)get_sub_group_local_id();

    // Dispatch as bs_fs_bsv_fsv, where bsv = DISPATCH_BSV and fsv = DISPATCH_FSV.
    // This allows more fine grained control over dispatch order than using work-groups and
    // avoids requirement of threads being available for whole work-group.
    // It could hovewer have some drawbacks like not providing physical locality or not using
    // full dispatch pipeline.
    uint feature_mini_block = gid % DISPATCH_FSV;
    uint batch_mini_block = gid / DISPATCH_FSV % DISPATCH_BSV;
    uint feature_mega_block = gid / (DISPATCH_FSV * DISPATCH_BSV) % (CEIL_DIV(TILE_OUT_F_NUM, TILE_OFM * SIMD) / DISPATCH_FSV);
    uint batch_mega_block = gid / (DISPATCH_FSV * DISPATCH_BSV * CEIL_DIV(TILE_OUT_F_NUM, TILE_OFM * SIMD) / DISPATCH_FSV);

    FILTER_VEC_TYPE wei = 0;

    uint out_f = gid * (TILE_OFM * SIMD);
    uint out_b = LWS_BATCHES * TILE_B * (uint)get_group_id(2) + local_id * TILE_B;

#if OUTPUT_3D
    uint out_b0 = out_b / OUTPUT_FEATURE_NUM;
    uint out_b1 = out_b % OUTPUT_FEATURE_NUM;
    uint input_offset = out_b0 * INPUT0_BATCH_PITCH + out_b1 * INPUT0_FEATURE_PITCH + INPUT0_OFFSET;
#else
    uint input_offset = out_b * TILE_IN_B_PITCH + INPUT0_OFFSET;
#endif

    uint weights_offset = out_f * (INPUT_ELEMENTS_COUNT / 2);

    ACCUMULATOR_VEC_TYPE    acc[TILE_B] = { };

    // Dynamic Quantize
    MAKE_VECTOR_TYPE(DQ_TYPE, INPUT_LOAD_SIZE)      tiled_input_0[HALF_TILE_B] = { };   // Load 4 linear inputs for packing
    PACKED_DQ_TYPE                                  packed_in_0[HALF_TILE_B] = { };     // Packing char4 inputs to 1 integer
    INPUT0_TYPE                                     de_quantize_scale[TILE_B];

#if COMPRESSED_WEIGHTS && DECOMPRESSION_SCALE_GROUPS_NUM == 1
    #if DECOMPRESSION_SCALE_LENGTH > 1 && DECOMPRESSION_SCALE_LENGTH % (TILE_OFM * SIMD) == 0
        ACCUMULATOR_VEC_TYPE d_scale = TO_ACCUMULATOR_VEC_TYPE(BLOCK_READN(DECOMPRESSION_SCALE_TYPE, TILE_OFM, decompression_scale, out_f));
    #elif DECOMPRESSION_SCALE_LENGTH > 1 && DECOMPRESSION_SCALE_LENGTH % (TILE_OFM * SIMD) != 0
        ACCUMULATOR_VEC_TYPE d_scale = 0;
        unroll_for(uint of = 0; of < TILE_OFM; ++of) {
            uint offset = out_f + of*SIMD + get_sub_group_local_id();
            if (offset < DECOMPRESSION_SCALE_LENGTH)
                ((ACCUMULATOR_TYPE*)(&d_scale))[of] = decompression_scale[offset];
        }
    #else
        ACCUMULATOR_VEC_TYPE d_scale = decompression_scale[0];
    #endif

    ACCUMULATOR_TYPE* d_scales = (ACCUMULATOR_TYPE*)(&d_scale);
#endif

#if COMPRESSED_WEIGHTS && DECOMPRESSION_ZP_TERM && DECOMPRESSION_ZP_GROUPS_NUM == 1 && !DECOMPRESSION_ZP_SCALAR
    #if DECOMPRESSION_ZP_LENGTH > 1 && DECOMPRESSION_ZP_LENGTH % (TILE_OFM * SIMD) == 0
        ACCUMULATOR_VEC_TYPE d_zp = TO_ACCUMULATOR_VEC_TYPE(BLOCK_READN(DECOMPRESSION_ZP_TYPE, TILE_OFM, decompression_zp, out_f));
    #elif DECOMPRESSION_ZP_LENGTH > 1 && DECOMPRESSION_ZP_LENGTH % (TILE_OFM * SIMD) != 0
        ACCUMULATOR_VEC_TYPE d_zp = 0;
        unroll_for(uint of = 0; of < TILE_OFM; ++of) {
            uint offset = out_f + of*SIMD + get_sub_group_local_id();
            if (offset < DECOMPRESSION_ZP_LENGTH)
                ((ACCUMULATOR_TYPE*)(&d_zp))[of] = decompression_zp[offset];
        }
    #else
        ACCUMULATOR_VEC_TYPE d_zp = decompression_zp[0];
    #endif
    ACCUMULATOR_TYPE* d_zps = (ACCUMULATOR_TYPE*)(&d_zp);
#endif

    // =====================================================================================================================================
    // Main computation loop
    const uint iterations = MAIN_LOOP_ELEMENTS_COUNT / (TILE_IFM * SIMD);
    // Each sub-group loads 2 Batch 
    uint idx_sglid = (sglid * TILE_K) % QUANTIZE_GROUP_SIZE;       // same index for sglid 0~7 : to tile_k direction
    uint batch_sglid = (sglid * TILE_K) / QUANTIZE_GROUP_SIZE;     // 0 to 1 : to batch direction

    __attribute__((opencl_unroll_hint(1)))
    for (uint ni = 0; ni < iterations; ++ni) {
        uint in_offset = input_offset + (idx_sglid + batch_sglid * TILE_IN_B_PITCH);
        uint scale_offset = input_offset / QUANTIZE_GROUP_SIZE;
        for (uint bi = 0; bi < HALF_TILE_B; ++bi) {
            // Load quantizing info from pre-quantizing kernel
            tiled_input_0[bi] = vload4(0, &quantized_input[in_offset]);
            de_quantize_scale[bi * 2] = scale[scale_offset];
            de_quantize_scale[bi * 2 + 1] = scale[scale_offset+ (TILE_IN_B_PITCH/QUANTIZE_GROUP_SIZE)];

            // Packing : Get 4(B)x4(K) integer vector (packing to 4x1 vector)
            packed_in_0[bi] = as_int(tiled_input_0[bi]);

            // Next batch
            in_offset += (TILE_IN_B_PITCH * 2);
            scale_offset += (TILE_IN_B_PITCH/QUANTIZE_GROUP_SIZE * 2);
        }

        input_offset += TILE_IFM * SIMD;

        // Packing
        MAKE_VECTOR_TYPE(int, TILE_B) acc_tmp[TILE_OFM] = { };

        #if TILE_OFM != 2
        #error "FC bf_tiled kernel: can't use SLM optimization with TILE_OFM != 2"
        #endif

        // Skip first barrier synchronization if there is only single outer loop iteration.
        #if MAIN_LOOP_ELEMENTS_COUNT / (TILE_IFM * SIMD) > 1
            barrier(CLK_LOCAL_MEM_FENCE);
        #endif

        __local int* char_slm_weight = (__local int*)wei_local_mem;

        uint weights_idx = weights_offset + local_id * SIMD * FILTER_LOAD_ITERS * FILTER_LOAD_BLOCK_SIZE;
        uint wei_local_idx = local_id * SIMD * FILTER_LOAD_ITERS * (FILTER_LOAD_BLOCK_SIZE/2) + sglid * 2;

        // DECOMPRESSION_SCALE_POST_OP SHOULD be enabled for dynamic quantize FC : scale is ACCUMULATOR_VAL_ONE
        unroll_for(uint load_iter = 0; load_iter < FILTER_LOAD_ITERS; ++load_iter) {
            SLM_FILTER_PACKED_VEC wei_packed = BLOCK_READN(FILTER_TYPE, FILTER_LOAD_BLOCK_SIZE, weights, weights_idx);
            DQ_SLM_FILTER_UNPACKED_VEC dq_wei_unpacked = UNPACK_TRANSPOSED_INT4(DQ_TYPE, *((uint4x8_t *)&wei_packed));

            // Calculate zero-point and scale only for DECOMPRESSION_SCALE_POST_OP enabled
            #if DECOMPRESSION_ZP_TERM
                #if DECOMPRESSION_ZP_SCALAR
                    DQ_SLM_FILTER_UNPACKED_VEC dzp = (DQ_SLM_FILTER_UNPACKED_VEC)(DECOMPRESSION_ZP_VALUE);
                #elif DECOMPRESSION_ZP_GROUPS_NUM > 1
                    DQ_SLM_FILTER_UNPACKED_VEC dzp;
                    unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
                        unroll_for(uint kii = 0; kii < FILTER_LOAD_BLOCK_SIZE; ++kii) {
                            const uint offset_ofm = out_f + fi*SIMD + sglid;
                            const uint offset_ifm = ni * TILE_IFM * SIMD + local_id * FILTER_LOAD_ITERS * FILTER_LOAD_BLOCK_SIZE + load_iter * FILTER_LOAD_BLOCK_SIZE + kii;
                            const uint zp_offset = (offset_ofm % DECOMPRESSION_ZP_BATCH_NUM) * DECOMPRESSION_ZP_BATCH_PITCH +
                                                    (offset_ifm / DECOMPRESSION_ZP_GROUP_SIZE) * DECOMPRESSION_ZP_FEATURE_PITCH;
                            dzp[W_IDX] = decompression_zp[zp_offset];
                        }
                    }
                #else
                    DQ_SLM_FILTER_UNPACKED_VEC dzp = (DQ_SLM_FILTER_UNPACKED_VEC)(d_zps[0]);
                #endif
            #else
                DQ_SLM_FILTER_UNPACKED_VEC dzp = (DQ_SLM_FILTER_UNPACKED_VEC)(ACCUMULATOR_VAL_ZERO);
            #endif

            // Calculate weight : w = (w - dzp) * ds
            dq_wei_unpacked -= dzp;

            #if FILTER_LOAD_BLOCK_SIZE == 2
                DQ_SLM_FILTER_VEC wei_1 = {dq_wei_unpacked.s01, dq_wei_unpacked.s23};
                char_slm_weight[wei_local_idx] = as_int(wei_1);
            #elif FILTER_LOAD_BLOCK_SIZE == 4
                DQ_SLM_FILTER_VEC wei_1 = {dq_wei_unpacked.s01, dq_wei_unpacked.s23};
                char_slm_weight[wei_local_idx] = as_int(wei_1);
                DQ_SLM_FILTER_VEC wei_2 = {dq_wei_unpacked.s45, dq_wei_unpacked.s67};
                char_slm_weight[wei_local_idx+1] = as_int(wei_2);
            #elif FILTER_LOAD_BLOCK_SIZE == 8
                DQ_SLM_FILTER_VEC wei_1 = {dq_wei_unpacked.s01, dq_wei_unpacked.s23};
                char_slm_weight[wei_local_idx] = as_int(wei_1);
                DQ_SLM_FILTER_VEC wei_2 = {dq_wei_unpacked.s45, dq_wei_unpacked.s67};
                char_slm_weight[wei_local_idx+1] = as_int(wei_2);
                DQ_SLM_FILTER_VEC wei_3 = {dq_wei_unpacked.s89, dq_wei_unpacked.sab};
                char_slm_weight[wei_local_idx+2] = as_int(wei_3);
                DQ_SLM_FILTER_VEC wei_4 = {dq_wei_unpacked.scd, dq_wei_unpacked.sef};
                char_slm_weight[wei_local_idx+3] = as_int(wei_4);
            #else
                #error "FC bf_tiled kernel: unsupported FILTER_LOAD_BLOCK_SIZE for SLM kernel"
            #endif

            wei_local_idx += SIMD * (FILTER_LOAD_BLOCK_SIZE/2);
            weights_idx += SIMD * FILTER_LOAD_BLOCK_SIZE;
        }

        wei_local_idx = sglid * 2;

        barrier(CLK_LOCAL_MEM_FENCE);

        unroll_for(uint ki = 0; ki < (TILE_IFM * SIMD) / TILE_K; ++ki) {
            #if TILE_K != 4
                #error "FC bf_tiled kernel: unsupported TILE_K size for SLM kernel"
            #endif

            // Compute input * weight : packed char4 type
            char8 weight = vload8(0, (__local char *)(&char_slm_weight[wei_local_idx + 16*2*ki]));
            char4 first_weight = weight.s0123;
            char4 second_weight = weight.s4567;
            unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
                char4 input_val = as_char4(_sub_group_shuffle(packed_in_0[bi / 2], (bi % 2) * 8 + ki));
                acc_tmp[0][bi] = imad_SW(acc_tmp[0][bi], input_val, first_weight);
                acc_tmp[1][bi] = imad_SW(acc_tmp[1][bi], input_val, second_weight);
            }

            weights_offset += TILE_K_OFM_PACKED * SIMD;

            #if DECOMPRESSION_SCALE_POST_OP && (TILE_IFM * SIMD > DECOMPRESSION_SCALE_GROUP_SIZE)
                unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
                    unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
                        const uint offset_ofm = out_f + fi*SIMD + sglid;

                        #if DECOMPRESSION_SCALE_GROUPS_NUM > 1
                            const uint scale_offset = (offset_ofm % DECOMPRESSION_SCALE_BATCH_NUM) * DECOMPRESSION_SCALE_BATCH_PITCH +
                                                    ((ni*TILE_IFM*SIMD + ki*TILE_K) / DECOMPRESSION_SCALE_GROUP_SIZE)*DECOMPRESSION_SCALE_FEATURE_PITCH;
                            ACCUMULATOR_TYPE ds = decompression_scale[scale_offset];
                        #else
                            ACCUMULATOR_TYPE ds = d_scales[fi % DECOMPRESSION_SCALE_LENGTH];
                        #endif

                        ((ACCUMULATOR_TYPE*)(&acc[bi]))[fi] += convert_half(((int *)(&acc_tmp[fi]))[bi]) * ds * de_quantize_scale[bi];
                        acc_tmp[fi][bi] = 0;
                    }
                }
            #endif
        }  // Whole tile_k elements of each iteration : ki

        #if DECOMPRESSION_SCALE_POST_OP && (TILE_IFM * SIMD <= DECOMPRESSION_SCALE_GROUP_SIZE)
            const uint ni_offset = ((ni*TILE_IFM*SIMD) / DECOMPRESSION_SCALE_GROUP_SIZE)*DECOMPRESSION_SCALE_FEATURE_PITCH;
            unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
                unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
                    const uint offset_ofm = out_f + fi*SIMD + sglid;

                    #if DECOMPRESSION_SCALE_GROUPS_NUM > 1
                        const uint scale_offset = (offset_ofm % DECOMPRESSION_SCALE_BATCH_NUM) * DECOMPRESSION_SCALE_BATCH_PITCH + ni_offset;
                        ACCUMULATOR_TYPE ds = decompression_scale[scale_offset];
                    #else
                        ACCUMULATOR_TYPE ds = d_scales[fi % DECOMPRESSION_SCALE_LENGTH];
                    #endif

                    ((ACCUMULATOR_TYPE*)(&acc[bi]))[fi] += convert_half(((int *)(&acc_tmp[fi]))[bi]) * ds * de_quantize_scale[bi];
                }
            }
        #endif
    }  // Main compute loop : ni

    // =====================================================================================================================================
    // Post-processing: bias, activation, fused-ops
    ACTIVATION_VEC_TYPE activated[TILE_B] = { };
    for (uint bi = 0; bi < TILE_B; ++bi) {
        activated[bi] = TO_ACTIVATION_VEC_TYPE(acc[bi]);
    }

#if BIAS_TERM
    #if TILE_OUT_F_NUM % (TILE_OFM * SIMD) == 0
        BIAS_VEC_TYPE bias = BIAS_BLOCK_READ(biases, out_f);
    #else
        BIAS_VEC_TYPE bias = 0;
        unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
            ((BIAS_TYPE*)(&bias))[fi] = biases[out_f + sglid + fi * SIMD];
        }
    #endif
    unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
        activated[bi] += TO_ACTIVATION_VEC_TYPE(bias);
    }
#endif

    OUTPUT_VEC_TYPE result[TILE_B] = { };
#if HAS_FUSED_OPS
    unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
    #if TILE_OFM > 1
        unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
            FUSED_OPS_VEC;
            result[bi][fi] = FUSED_OPS_RESULT_VEC;
        }
    #else
        FUSED_OPS_SCALAR;
        result[bi] = FUSED_OPS_RESULT_SCALAR;
    #endif // TILE_OFM > 1
    }
#else
    unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
        result[bi] = TO_OUTPUT_VEC_TYPE(ACTIVATION_TYPED(activated[bi], ACTIVATION_PARAMS_TYPED));
    }
#endif

    // =====================================================================================================================================
    // Write results
    uint output_offset = out_f * TILE_OUT_F_PITCH + out_b * TILE_OUT_B_PITCH + OUTPUT_OFFSET;

    if (USE_BLOCK_WRITE && (TILE_OUT_F_NUM % (TILE_OFM * SIMD) == 0 || out_f + (TILE_OFM * SIMD) <= TILE_OUT_F_NUM)) {
#if IS_DYNAMIC
        #define WRITE_OUTPUT(bi) do {                                       \
                if (bi + out_b < BATCH_SIZE)                                \
                    OUTPUT_BLOCK_WRITE(output, output_offset, result[bi]);  \
                output_offset += TILE_OUT_B_PITCH;                          \
            } while (false)
#else
        #define WRITE_OUTPUT(bi) do {                                       \
                OUTPUT_BLOCK_WRITE(output, output_offset, result[bi]);      \
                output_offset += TILE_OUT_B_PITCH;                          \
            } while (false)
#endif
        CONST_LOOP(TILE_B, WRITE_OUTPUT);
        #undef WRITE_OUTPUT
    } else {
        output_offset += sglid;

        for (uint bi = 0; bi < TILE_B; ++bi) {
            for (uint fi = 0; fi < TILE_OFM; ++fi) {
                const bool should_write =
#if IS_DYNAMIC
                    bi + out_b < BATCH_SIZE &&
#endif
                    (TILE_OUT_F_NUM % (TILE_OFM * SIMD) == 0 ||
                    out_f + fi * SIMD + sglid < TILE_OUT_F_NUM);
                if (should_write) {
                    output[output_offset] = ((OUTPUT_TYPE*)(&result[bi]))[fi];
                }
                output_offset += SIMD;
            }
            output_offset += TILE_OUT_B_PITCH - TILE_OFM * SIMD;
        }
    }
    // =====================================================================================================================================
}
#endif

REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(fc)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
#if DECOMPRESSION_SCALE_TERM
    const __global DECOMPRESSION_SCALE_TYPE* decompression_scale,
#endif
#if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
    const __global DECOMPRESSION_ZP_TYPE* decompression_zp,
#endif
    __global OUTPUT_TYPE* output,
    const __global FILTER_TYPE* weights
#if BIAS_TERM
    , const __global BIAS_TYPE* biases
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
#if DYNAMIC_QUANTIZE
    , __global char* quantized_input
    , __global INPUT0_TYPE* de_quan_scale
#endif
) {
#if USE_SLM
    #if DYNAMIC_QUANTIZE
        __local int dq_wei_local_mem[SIMD * TILE_OFM * SIMD];
    #else
        __local ACCUMULATOR_TYPE wei_local_mem[TILE_IFM * SIMD * TILE_OFM * SIMD];
    #endif
#endif
#if IS_DYNAMIC && COMPRESSED_WEIGHTS_INT4
    const int batch_size = BATCH_SIZE;
    if (batch_size == 1) {
        FUNC_CALL(fc_bf_tiled_kernel_forced_tile_b1)(
            OPTIONAL_SHAPE_INFO_TENSOR
            input,
        #if DECOMPRESSION_SCALE_TERM
            decompression_scale,
        #endif
        #if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
            decompression_zp,
        #endif
            output,
            weights
        #if BIAS_TERM
            , biases
        #endif
        #if HAS_FUSED_OPS_DECLS
            , FUSED_OPS_ARGS
        #endif
        );
    } else if (batch_size == 2) {
        FUNC_CALL(fc_bf_tiled_kernel_forced_tile_b2)(
            OPTIONAL_SHAPE_INFO_TENSOR
            input,
        #if DECOMPRESSION_SCALE_TERM
            decompression_scale,
        #endif
        #if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
            decompression_zp,
        #endif
            output,
            weights
        #if BIAS_TERM
            , biases
        #endif
        #if HAS_FUSED_OPS_DECLS
            , FUSED_OPS_ARGS
        #endif
        );
    } else if (batch_size == 3) {
        FUNC_CALL(fc_bf_tiled_kernel_forced_tile_b3)(
            OPTIONAL_SHAPE_INFO_TENSOR
            input,
        #if DECOMPRESSION_SCALE_TERM
            decompression_scale,
        #endif
        #if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
            decompression_zp,
        #endif
            output,
            weights
        #if BIAS_TERM
            , biases
        #endif
        #if HAS_FUSED_OPS_DECLS
            , FUSED_OPS_ARGS
        #endif
        );
    } else if (batch_size == 4) {
        FUNC_CALL(fc_bf_tiled_kernel_forced_tile_b4)(
            OPTIONAL_SHAPE_INFO_TENSOR
            input,
        #if DECOMPRESSION_SCALE_TERM
            decompression_scale,
        #endif
        #if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
            decompression_zp,
        #endif
            output,
            weights
        #if BIAS_TERM
            , biases
        #endif
        #if HAS_FUSED_OPS_DECLS
            , FUSED_OPS_ARGS
        #endif
        );
    } else if (batch_size == 5) {
        FUNC_CALL(fc_bf_tiled_kernel_forced_tile_b5)(
            OPTIONAL_SHAPE_INFO_TENSOR
            input,
        #if DECOMPRESSION_SCALE_TERM
            decompression_scale,
        #endif
        #if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
            decompression_zp,
        #endif
            output,
            weights
        #if BIAS_TERM
            , biases
        #endif
        #if HAS_FUSED_OPS_DECLS
            , FUSED_OPS_ARGS
        #endif
        );
    } else if (batch_size == 6) {
        FUNC_CALL(fc_bf_tiled_kernel_forced_tile_b6)(
            OPTIONAL_SHAPE_INFO_TENSOR
            input,
        #if DECOMPRESSION_SCALE_TERM
            decompression_scale,
        #endif
        #if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
            decompression_zp,
        #endif
            output,
            weights
        #if BIAS_TERM
            , biases
        #endif
        #if HAS_FUSED_OPS_DECLS
            , FUSED_OPS_ARGS
        #endif
        );
    } else if (batch_size == 7) {
        FUNC_CALL(fc_bf_tiled_kernel_forced_tile_b7)(
            OPTIONAL_SHAPE_INFO_TENSOR
            input,
        #if DECOMPRESSION_SCALE_TERM
            decompression_scale,
        #endif
        #if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
            decompression_zp,
        #endif
            output,
            weights
        #if BIAS_TERM
            , biases
        #endif
        #if HAS_FUSED_OPS_DECLS
            , FUSED_OPS_ARGS
        #endif
        );
    } else {
        #if USE_SLM && DYNAMIC_QUANTIZE
            FUNC_CALL(fc_bf_tiled_kernel_dyn_quan)(
                OPTIONAL_SHAPE_INFO_TENSOR
                input,
                quantized_input,
                de_quan_scale,
            #if DECOMPRESSION_SCALE_TERM
                decompression_scale,
            #endif
            #if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
                decompression_zp,
            #endif
                output,
                weights
                , dq_wei_local_mem
            #if BIAS_TERM
                , biases
            #endif
            #if HAS_FUSED_OPS_DECLS
                , FUSED_OPS_ARGS
            #endif
            );
        #else
            FUNC_CALL(fc_bf_tiled_kernel_default)(
                OPTIONAL_SHAPE_INFO_TENSOR
                input,
            #if DECOMPRESSION_SCALE_TERM
                decompression_scale,
            #endif
            #if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
                decompression_zp,
            #endif
                output,
                weights
            #if USE_SLM
                , wei_local_mem
            #endif
            #if BIAS_TERM
                , biases
            #endif
            #if HAS_FUSED_OPS_DECLS
                , FUSED_OPS_ARGS
            #endif
            );
        #endif
    }
#else
    #if USE_SLM && DYNAMIC_QUANTIZE
        FUNC_CALL(fc_bf_tiled_kernel_dyn_quan)(
            OPTIONAL_SHAPE_INFO_TENSOR
            input,
            quantized_input,
            de_quan_scale,
        #if DECOMPRESSION_SCALE_TERM
            decompression_scale,
        #endif
        #if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
            decompression_zp,
        #endif
            output,
            weights
            , dq_wei_local_mem
        #if BIAS_TERM
            , biases
        #endif
        #if HAS_FUSED_OPS_DECLS
            , FUSED_OPS_ARGS
        #endif
        );
    #else
        FUNC_CALL(fc_bf_tiled_kernel_default)(
            OPTIONAL_SHAPE_INFO_TENSOR
            input,
        #if DECOMPRESSION_SCALE_TERM
            decompression_scale,
        #endif
        #if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
            decompression_zp,
        #endif
            output,
            weights
        #if USE_SLM
            , wei_local_mem
        #endif
        #if BIAS_TERM
            , biases
        #endif
        #if HAS_FUSED_OPS_DECLS
            , FUSED_OPS_ARGS
        #endif
        );
    #endif
#endif
}
#endif  // !FC_KERNEL_DYNAMIC_QUANTIZE

#undef INPUT_VEC_TYPE
#undef ACCUMULATOR_VEC_TYPE
#undef FILTER_VEC_TYPE
#undef BIAS_VEC_TYPE
#undef OUTPUT_VEC_TYPE
#undef ACTIVATION_VEC_TYPE
#undef TO_OUTPUT_VEC_TYPE
#undef TO_ACTIVATION_VEC_TYPE

#undef INPUT_BLOCK_READ
#undef FILTER_BLOCK_READ
#undef BIAS_BLOCK_READ
#undef OUTPUT_BLOCK_WRITE

#undef USE_BLOCK_WRITE

#undef MAIN_LOOP_ELEMENTS_COUNT
