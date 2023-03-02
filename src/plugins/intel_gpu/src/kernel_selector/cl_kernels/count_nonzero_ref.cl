// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

#define INPUT0_GET_INDEX1(idx_order) INPUT0_GET_INDEX(idx_order)

KERNEL (count_nonzero_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    volatile __global uint* tmp_buffer,
    volatile __global OUTPUT_TYPE* output)
{
    const uint gdim0 = (uint)get_global_id(0);
    const uint gdim1 = (uint)get_global_id(1);
    const uint gdim2 = (uint)get_global_id(2);
    const uint ldim0 = (uint)get_local_id(0);
    const uint ldim1 = (uint)get_local_id(1);
    const uint ldim2 = (uint)get_local_id(2);
    const uint group_id = (uint)get_group_id(2);
    local uint local_count;


    if (ldim0 == 0 && ldim1 == 0 && ldim2 == 0) {
        tmp_buffer[group_id] = 0;
        local_count = 0;
    }

    work_group_barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    #if INPUT0_DIMS == 6
        #define INPUT_ORDER b,f,w,z,y,x
        const uint x = gdim0 % INPUT0_SIZE_X;
        const uint y = gdim0 / INPUT0_SIZE_X;
        const uint z = gdim1 % INPUT0_SIZE_Z;
        const uint w = gdim1 / INPUT0_SIZE_Z;
    #elif INPUT0_DIMS == 5
        #define INPUT_ORDER b,f,z,y,x
        const uint x = gdim0;
        const uint y = gdim1 % INPUT0_SIZE_Y;
        const uint z = gdim1 / INPUT0_SIZE_Y;
    #elif INPUT0_DIMS == 4
        #define INPUT_ORDER b,f,y,x
        const uint x = gdim0;
        const uint y = gdim1;
    #endif
    const uint f = gdim2 % INPUT0_FEATURE_NUM;
    const uint b = gdim2 / INPUT0_FEATURE_NUM;

    uint count = (input[INPUT0_GET_INDEX1(INPUT_ORDER)] == INPUT0_VAL_ZERO) ? 0 : 1;
    count = sub_group_reduce_add(count);

    if (get_sub_group_local_id() == 0) {
        atomic_add(&local_count, count);
        // printf("** nonzero_count[Step01][%3d] : local_count[%3d] = count %3d(%3d) - subgroup_size(%3d)\n", get_global_size(2), group_id, local_count, count, get_sub_group_size());
    }

    work_group_barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    if (ldim0 == 0 && ldim1 == 0 && ldim2 == 0) {
        tmp_buffer[group_id] = count;
        // printf("** nonzero_count[Step02][%3d] : tmp_buffer[%3d] = %3d(%3d) - subgroup_size(%3d)\n", get_global_size(2), group_id, tmp_buffer[group_id], local_count, get_sub_group_size());
    }

    work_group_barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    if (gdim0 == 0 && gdim1 == 0 && gdim2 == 0) {
        output[0] = 0;
        uint group_size = (uint)get_num_groups(2);
        for (uint gid = 0; gid < group_size; gid++ ) {
            output[0] = output[0] + tmp_buffer[gid];
            // const uint cgid = gid;
            // printf("nonzero_count - GlobalWG[%3d,%3d,%3d],LocalWG[%3d,%3d,%3d],group_size[%3d],get_global_size(%3d)tem_buffer[%3d](%3d),output[%3d]\n",gdim0, gdim1, gdim2, ldim0, ldim1, ldim2, group_size, get_global_size(2), cgid, tmp_buffer[cgid], output[0]);
        }
    }

// printf("nonzero_count - GlobalWG[%3d,%3d,%3d],LocalWG[%3d,%3d,%3d],Subgroup[%3d][D: %3d] -- get_global_size[%3d,%3d,%3d], get_local_size[%3d,%3d,%3d], get_num_groups[%3d,%3d,%3d], get_group_id[%3d,%3d,%3d]\n",
//             gdim0, gdim1, gdim2, local0, local1, local2, get_sub_group_local_id(), shape_info[0],
//             get_global_size(0),get_global_size(1),get_global_size(2),get_local_size(0),get_local_size(1),get_local_size(2),get_num_groups(0),get_num_groups(1),get_num_groups(2),get_group_id(0),get_group_id(1),get_group_id(2));
}

#undef INPUT0_GET_INDEX1
#undef INPUT_ORDER
