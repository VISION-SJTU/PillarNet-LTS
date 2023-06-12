#ifndef PILLAR_QUERY_OPS_GPU_H
#define PILLAR_QUERY_OPS_GPU_H

#include "cuda_utils.h"


int create_point_pillar_index_stack_wrapper(at::Tensor pts_xy_tensor, at::Tensor pts_batch_cnt_tensor,
										    at::Tensor pillars_mask_tensor, at::Tensor point_pillar_index_tensor);

void create_point_pillar_index_stack_kernel_launcher(int N, int B, int H, int W,
												     const int *pts_xy, const int *pts_batch_cnt, 
                                                     bool *pillar_mask, int *point_pillar_index);

int create_pillar_indices_wrapper(at::Tensor pillars_position_tensor, at::Tensor pillar_indices_tensor);

void create_pillar_indices_kernel_launcher(int B, int H, int W, const int *pillars_position, int *pillar_indices);

#endif //PILLAR_QUERY_OPS_GPU_H
