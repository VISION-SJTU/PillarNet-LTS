#include "group_ops_gpu.h"


int flatten_indice_pairs_wrapper(at::Tensor indice_pairs_tensor, at::Tensor position_tensor,
                                 at::Tensor first_indices_tensor, at::Tensor second_indices_tensor){
    // indice_pairs_tensor: (N, K)  position_tensor: (N*K) first_indices_tensor: (L,) second_indices_tensor: (L,)

    CHECK_INPUT(indice_pairs_tensor);
    CHECK_INPUT(position_tensor);
    CHECK_INPUT(first_indices_tensor);
    CHECK_INPUT(second_indices_tensor);

    int N = indice_pairs_tensor.size(0);
    int K = indice_pairs_tensor.size(1);

    const int *indicePairs = indice_pairs_tensor.data_ptr<int>();
    const int *position = position_tensor.data_ptr<int>();
    int *firstIndices = first_indices_tensor.data_ptr<int>();
    int *secondIndices = second_indices_tensor.data_ptr<int>();

    flatten_indice_paris_kernel_launcher(N, K, indicePairs, position, firstIndices, secondIndices);

    return 1;
}


int gather_feature_wrapper(at::Tensor set_indices_tensor, at::Tensor features_tensor, at::Tensor new_features_tensor){

    CHECK_INPUT(set_indices_tensor);
    CHECK_INPUT(features_tensor);
    CHECK_INPUT(new_features_tensor);

    int L = set_indices_tensor.size(0);
    int C = features_tensor.size(1);

    const int *set_indices = set_indices_tensor.data_ptr<int>();
    const float *features = features_tensor.data_ptr<float>();
    float *new_features = new_features_tensor.data_ptr<float>();

    gather_feature_kernel_launcher(L, C, set_indices, features, new_features);

    return 1;
}

int gather_feature_grad_wrapper(at::Tensor set_indices_tensor, at::Tensor grad_out_tensor, at::Tensor grad_features_tensor){

    CHECK_INPUT(set_indices_tensor);
    CHECK_INPUT(grad_out_tensor);
    CHECK_INPUT(grad_features_tensor);

    int L = set_indices_tensor.size(0);
    int C = grad_features_tensor.size(1);

    const int *set_indices = set_indices_tensor.data_ptr<int>();
    const float *outGrad = grad_out_tensor.data_ptr<float>();
    float *inGrad = grad_features_tensor.data_ptr<float>();

    gather_feature_grad_kernel_launcher(L, C, set_indices, outGrad, inGrad);

    return 1;
}


int kernel_average_concatenate_wrapper(int kernel, at::Tensor features_tensor, at::Tensor new_xy_tensor, at::Tensor query_xy_tensor,
									   at::Tensor query_indices_tensor, at::Tensor is_empty_tensor, at::Tensor out_tensor) {
	CHECK_INPUT(features_tensor);
	CHECK_INPUT(new_xy_tensor);
	CHECK_INPUT(query_xy_tensor);
	CHECK_INPUT(query_indices_tensor);
	CHECK_INPUT(is_empty_tensor);
	CHECK_INPUT(out_tensor);

	const float *features = features_tensor.data_ptr<float>();
	const float *new_xy = new_xy_tensor.data_ptr<float>();
	const int *query_xy = query_xy_tensor.data_ptr<int>();
	const int *query_indices = query_indices_tensor.data_ptr<int>();
	const uint8_t *is_empty = is_empty_tensor.data_ptr<uint8_t>();
	float *out = out_tensor.data_ptr<float>();

	int N = features_tensor.size(0);
	int C = features_tensor.size(1);

	int M = query_indices_tensor.size(0);
	int K = query_indices_tensor.size(1);

	kernel_average_concatenate_kernel_launcher(N, M, K, C, kernel, features, new_xy, query_xy, query_indices, is_empty, out);

	return 1;
}


int kernel_average_concatenate_grad_wrapper(int kernel, at::Tensor grad_out_tensor, at::Tensor new_xy_tensor, at::Tensor query_xy_tensor,
                                            at::Tensor query_indices_tensor, at::Tensor is_empty_tensor, at::Tensor grad_features_tensor) {

	CHECK_INPUT(new_xy_tensor);
	CHECK_INPUT(query_xy_tensor);
	CHECK_INPUT(query_indices_tensor);
	CHECK_INPUT(is_empty_tensor);
	CHECK_INPUT(grad_out_tensor);
	CHECK_INPUT(grad_features_tensor);

	const float *new_xy = new_xy_tensor.data_ptr<float>();
	const int *query_xy = query_xy_tensor.data_ptr<int>();
	const int *query_indices = query_indices_tensor.data_ptr<int>();
	const uint8_t *is_empty = is_empty_tensor.data_ptr<uint8_t>();
	const float *grad_out = grad_out_tensor.data_ptr<float>();
	float *grad_features = grad_features_tensor.data_ptr<float>();

	int N = grad_features_tensor.size(0);
	int C = grad_features_tensor.size(1);

	int M = query_indices_tensor.size(0);
	int K = query_indices_tensor.size(1);

	kernel_average_concatenate_grad_kernel_launcher(N, M, K, C, kernel, grad_out, new_xy, query_xy, query_indices, is_empty, grad_features);

	return 1;
}

