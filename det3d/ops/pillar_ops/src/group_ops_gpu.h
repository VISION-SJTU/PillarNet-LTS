#ifndef PILLAR_GROUP_OPS_GPU_H
#define PILLAR_GROUP_OPS_GPU_H

#include "cuda_utils.h"

int gather_indice_wrapper(at::Tensor index_tensor, at::Tensor indices_tensor, at::Tensor outs_tensor);
int gather_feature_wrapper(at::Tensor index_tensor, at::Tensor features_tensor, at::Tensor outs_tensor);
int gather_feature_grad_wrapper(at::Tensor grad_outs_tensor, at::Tensor index_tensor, at::Tensor grad_features_tensor);

void gather_indice_kernel_launcher(int L, const int *index, const int *indices, int *outs);
void gather_feature_kernel_launcher(int L, int C, const int *index, const float *features, float *outs);
void gather_feature_grad_kernel_launcher(int L, int C, const int *index, const float *grad_outs, float *grad_features);


#endif //PILLAR_GROUP_OPS_GPU_H