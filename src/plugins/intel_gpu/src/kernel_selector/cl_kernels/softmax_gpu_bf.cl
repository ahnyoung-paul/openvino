// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"

#if SUBGROUP_BLOCK_SIZE == 1
#define BLOCK_READ(ptr, offset) DT_INPUT_BLOCK_READ(ptr, offset)
#define BLOCK_WRITE(ptr, offset, val) DT_OUTPUT_BLOCK_WRITE(ptr, offset, val)
#define BLOCK_TYPE INPUT0_TYPE
#else
#define BLOCK_READ(ptr, offset) CAT(DT_INPUT_BLOCK_READ, SUBGROUP_BLOCK_SIZE)(ptr, offset)
#define BLOCK_WRITE(ptr, offset, val) CAT(DT_OUTPUT_BLOCK_WRITE, SUBGROUP_BLOCK_SIZE)(ptr, offset, val)
#define BLOCK_TYPE MAKE_VECTOR_TYPE(INPUT0_TYPE, SUBGROUP_BLOCK_SIZE)
#endif

#if IS_DYNAMIC
#define CALC_POWER(n) ({uint pos = 0; uint i = n; do { i >>= 1; ++pos; } while (i); --pos;})
#endif

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
KERNEL (softmax_gpu_continuous_bfyx)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
    const uint data_set_idx = get_global_id(1);     // in processing of which data set this WI participates?
    const uint workers_per_data_set = LWS;          // how many WI participates in processing of one data set
    const uint in_data_set_idx = get_global_id(0);  // this WI's id in group of items processing single data set
    const uint data_set_size = DATA_SET_SIZE;       // how many elements are in one data set
    const uint data_sets_count = DATA_SETS_COUNT;   // how many data sets are in the processing payload
#if !IS_DYNAMIC
    const uint items_num = ITEMS_NUM;               // how many elements are processed per one WI
    const uint leftovers = LEFTOVERS;
#else
    // since workers_per_data_set is calculated by power of 2
    // items_num can be calculated by dividing data_set_size by power of 2
    const uint power = CALC_POWER(workers_per_data_set);
    const uint items_num = data_set_size>>power;
    const uint leftovers = data_set_size-(items_num<<power);
#endif
    const uint odd = data_set_idx - ((data_set_idx>>1)<<1) ;
    uint leftover_offsets = 0;
    uint in_data_offset = workers_per_data_set * items_num + in_data_set_idx;
    if (odd == 1)
    {
        leftover_offsets = leftovers;
        in_data_offset = in_data_set_idx;
    }
    // if (in_data_set_idx >= leftovers)
    //     in_data_offset = 1000;

    const uint data_set_offset = data_set_idx * data_set_size;
    const uint subgroup_offset = get_sub_group_id() * get_sub_group_size() * items_num;

    INPUT0_TYPE my_chunk[STACK_SIZE];
    INPUT0_TYPE my_maximum = -UNIT_VAL_MAX;
    INPUT0_TYPE my_sum = UNIT_VAL_ZERO;

    __local INPUT0_TYPE lg_storage[SLM_SIZE];

    uint i=0;
    if (workers_per_data_set > SUB_GROUP_SIZE)
    {
        // printf("%d,%d,%d,%d,%d,%d,%d,%d,%d\n", data_set_idx, in_data_set_idx, get_sub_group_id(), (data_set_offset + subgroup_offset + leftover_offsets), data_set_offset, subgroup_offset, leftover_offsets, odd, in_data_offset);
        for (; i<items_num - (items_num % SUBGROUP_BLOCK_SIZE); i+=SUBGROUP_BLOCK_SIZE)
        {
            BLOCK_TYPE vec_tmp = BLOCK_READ(input, data_set_offset + subgroup_offset + leftover_offsets + i * get_sub_group_size());
#if SUBGROUP_BLOCK_SIZE == 1
            my_maximum = max(my_maximum, vec_tmp);
            my_chunk[i] = vec_tmp;
#else
            for (int j = 0; j < SUBGROUP_BLOCK_SIZE; j++)
            {
                INPUT0_TYPE tmp = vec_tmp[j];
                my_maximum = max(my_maximum, tmp);
                my_chunk[i+j] = tmp;
            }
#endif
        }
    }

    for (; i<items_num; i++)
    {
        INPUT0_TYPE tmp = input[data_set_offset + subgroup_offset + leftover_offsets + get_sub_group_local_id() + i * get_sub_group_size()];
        my_maximum = max(my_maximum, tmp);
        my_chunk[i] = tmp;
    }
    const uint debug_offset = data_set_offset + subgroup_offset + leftover_offsets + get_sub_group_local_id();
    uint leftover_off = data_set_offset + in_data_offset;
    if (in_data_set_idx < leftovers)
    {
        INPUT0_TYPE tmp = input[data_set_offset + in_data_offset];
        my_maximum = max(my_maximum, tmp);
        my_chunk[items_num] = tmp;
    } else {
        my_chunk[items_num] = -UNIT_VAL_MAX;
        leftover_off = 0;
    }
    my_maximum = sub_group_reduce_max(my_maximum);

    if (get_sub_group_local_id() == 0)
        lg_storage[get_sub_group_id()] = my_maximum;

    // if (data_set_idx == 3 && in_data_set_idx == 0)
    // {
    //     printf("sub_group_id, in_data_set_idx, sub_group_local_id, data_set_offset, debug_offset, items_count, leftover_off,"
    //         "01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25\n");
    // }
    barrier(CLK_LOCAL_MEM_FENCE);
    // if (data_set_idx == 3)
    // {
    //     printf("%d, %d, %d, %d, %d, %d, %d,"
    //             "%f, %f, %f, %f, %f, %f," 
    //             "%f, %f, %f, %f, %f, %f,"
    //             "%f, %f, %f, %f, %f, %f," 
    //             "%f, %f, %f, %f, %f, %f, %f\n", 
    //             get_sub_group_id(), in_data_set_idx, get_sub_group_local_id(), data_set_offset, debug_offset, (items_num * get_sub_group_size() * get_sub_group_size()), leftover_off,
    //             my_chunk[0], my_chunk[1], my_chunk[2], my_chunk[3], my_chunk[4], my_chunk[5],
    //             my_chunk[6], my_chunk[7], my_chunk[8], my_chunk[9], my_chunk[10], my_chunk[11],
    //             my_chunk[12], my_chunk[13], my_chunk[14], my_chunk[15], my_chunk[16], my_chunk[17],
    //             my_chunk[18], my_chunk[19], my_chunk[20], my_chunk[21], my_chunk[22], my_chunk[23], my_chunk[24]);
    // }

    if (in_data_set_idx == 0)
    {
        for (uint i=1; i<get_num_sub_groups(); ++i)
            my_maximum = max(my_maximum, lg_storage[i]);

        lg_storage[0] = my_maximum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //my_maximum from this point is in fact global maximum
    my_maximum = lg_storage[0];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i=0; i<items_num; ++i)
    {
        INPUT0_TYPE tmp = native_exp(my_chunk[i] - my_maximum);
        my_sum += tmp;
        my_chunk[i] = tmp;
    }

    if (in_data_set_idx < leftovers)
    {
        INPUT0_TYPE tmp = native_exp(my_chunk[items_num] - my_maximum);
        my_sum += tmp;
        my_chunk[items_num] = tmp;
    }

    my_sum = sub_group_reduce_add(my_sum);

    if (get_sub_group_local_id() == 0)
        lg_storage[get_sub_group_id()] = my_sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    // if (data_set_idx == 3 && in_data_set_idx == 0)
    //     printf("data_set_idx, in_data_set_idx, get_sub_group_id(),offsets,data_set_offset, subgroup_offset, leftover_offsets, odd, in_data_offset,my_maximum,my_sum,lg_storage[subgroup_id]\n");
    if (in_data_set_idx == 0)
    {
        for (uint i=1; i<get_num_sub_groups(); ++i)
            my_sum += lg_storage[i];

        // if (data_set_idx == 3)
        // {
        //     printf("%d, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", get_sub_group_id(), my_maximum, my_sum, lg_storage[0], lg_storage[1], lg_storage[2], lg_storage[3], lg_storage[4], lg_storage[5], lg_storage[6], lg_storage[7]);
        // }
        lg_storage[0] = my_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    my_sum = lg_storage[0];
    // if (data_set_idx == 3)
    //     printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%f,%f,%f\n", data_set_idx, in_data_set_idx, get_sub_group_id(), 
    //         (data_set_offset + subgroup_offset + leftover_offsets), data_set_offset, subgroup_offset, leftover_offsets, odd, in_data_offset,my_maximum,my_sum,lg_storage[get_sub_group_id()]);
    i=0;
#if HAS_FUSED_OPS
    if (workers_per_data_set > SUB_GROUP_SIZE)
    {
        for (; i < items_num - (items_num % SUBGROUP_BLOCK_SIZE); i+=SUBGROUP_BLOCK_SIZE)
        {
            BLOCK_TYPE vec_tmp;
#if SUBGROUP_BLOCK_SIZE == 1
            ACTIVATION_TYPE dequantized = my_chunk[i] / my_sum;
            FUSED_OPS_MAIN;
            vec_tmp = FUSED_OPS_RESULT_MAIN;
#else
            for (int j = 0; j < SUBGROUP_BLOCK_SIZE; j++)
            {
                ACTIVATION_TYPE dequantized = my_chunk[i + j] / my_sum;
                FUSED_OPS_MAIN;
                vec_tmp[j] = FUSED_OPS_RESULT_MAIN;
            }
#endif
            BLOCK_WRITE(output, data_set_offset + subgroup_offset + leftover_offsets + i * get_sub_group_size(), vec_tmp);
        }
    }
    for (; i<items_num; i++)
    {
        ACTIVATION_TYPE dequantized = my_chunk[i] / my_sum;
        FUSED_OPS_MAIN;
        output[data_set_offset + subgroup_offset + leftover_offsets + get_sub_group_local_id() + i * get_sub_group_size()] = FUSED_OPS_RESULT_MAIN;
    }
    if (in_data_set_idx < leftovers)
    {
        ACTIVATION_TYPE dequantized = my_chunk[items_num] / my_sum;
        FUSED_OPS_LEFTOVERS;
        output[data_set_offset + in_data_offset] = FUSED_OPS_RESULT_LEFTOVERS;
    }
#else
    if (workers_per_data_set > SUB_GROUP_SIZE)
    {
        for (; i<items_num - (items_num % SUBGROUP_BLOCK_SIZE); i+=SUBGROUP_BLOCK_SIZE)
        {
            BLOCK_TYPE vec_tmp;
#if SUBGROUP_BLOCK_SIZE == 1
            vec_tmp = ACTIVATION(my_chunk[i] / my_sum, ACTIVATION_PARAMS);
#else
            for (int j = 0; j < SUBGROUP_BLOCK_SIZE; j++)
                vec_tmp[j] = ACTIVATION(my_chunk[i + j] / my_sum, ACTIVATION_PARAMS);
#endif
            BLOCK_WRITE(output, data_set_offset + subgroup_offset + leftover_offsets + i * get_sub_group_size(), vec_tmp);
        }
    }
    for (; i < items_num; i++)
    {
        output[data_set_offset + subgroup_offset + leftover_offsets + get_sub_group_local_id() + i * get_sub_group_size()] = ACTIVATION(my_chunk[i] / my_sum, ACTIVATION_PARAMS);
    }
    if (in_data_set_idx < leftovers)
        output[data_set_offset + in_data_offset] = ACTIVATION(my_chunk[items_num] / my_sum, ACTIVATION_PARAMS);
#endif
}
#ifdef CALC_POWER
#undef CALC_POWER
#endif
#undef BLOCK_READ
#undef BLOCK_WRITE
#undef BLOCK_TYPE

