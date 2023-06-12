#ifndef PILLAR_GROUP_OPS_GPU_H
#define PILLAR_GROUP_OPS_GPU_H


#include "cuda_utils.h"


int flatten_indice_pairs_wrapper(at::Tensor indice_pairs_tensor, at::Tensor position_tensor,
                                 at::Tensor first_indices_tensor, at::Tensor second_indices_tensor);
void flatten_indice_paris_kernel_launcher(int N, int K, const int *indicePairs, const int *position,
                                          int *firstIndices, int *secondIndices);

int gather_feature_wrapper(at::Tensor set_indices_tensor, at::Tensor features_tensor, at::Tensor new_features_tensor);
int gather_feature_grad_wrapper(at::Tensor set_indices_tensor, at::Tensor grad_out_tensor, at::Tensor grad_features_tensor);
void gather_feature_kernel_launcher(int L, int C, const int *set_indices, const float *features, float *out);
void gather_feature_grad_kernel_launcher(int L, int C, const int *set_indices, const float *outGrad, float *inGrad);


int kernel_average_concatenate_wrapper(int kernel, at::Tensor features_tensor, at::Tensor new_xy_tensor, at::Tensor query_xy_tensor,
                                       at::Tensor query_indices_tensor, at::Tensor is_empty_tensor, at::Tensor out_tensor);
int kernel_average_concatenate_grad_wrapper(int kernel, at::Tensor grad_out_tensor, at::Tensor new_xy_tensor, at::Tensor query_xy_tensor,
                                            at::Tensor query_indices_tensor, at::Tensor is_empty_tensor, at::Tensor grad_features_tensor);
void kernel_average_concatenate_grad_kernel_launcher(int N, int M, int K, int C, int kernel, const float *grad_out, const float *new_xy,
                                                     const int *query_xy, const int *query_indices, const uint8_t *is_empty, float *grad_features);
void kernel_average_concatenate_kernel_launcher(int N, int M, int K, int C, int kernel, const float *features, const float *new_xy,
                                                const int *query_xy, const int *query_indices, const uint8_t *is_empty, float *out);

#endif //PILLAR_GROUP_OPS_GPU_H
