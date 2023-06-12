#include "group_ops_gpu.h"


int gather_indice_wrapper(at::Tensor index_tensor, at::Tensor indices_tensor, at::Tensor outs_tensor){

    CHECK_INPUT(index_tensor);
    CHECK_INPUT(indices_tensor);
    CHECK_INPUT(outs_tensor);

    int L = index_tensor.size(0);

    const int *index = index_tensor.data_ptr<int>();
    const int *indices = indices_tensor.data_ptr<int>();
    int *outs = outs_tensor.data_ptr<int>();

    gather_indice_kernel_launcher(L, index, indices, outs);

    return 1;
}


int gather_feature_wrapper(at::Tensor index_tensor, at::Tensor features_tensor, at::Tensor outs_tensor){

    CHECK_INPUT(index_tensor);
    CHECK_INPUT(features_tensor);
    CHECK_INPUT(outs_tensor);

    int L = index_tensor.size(0);
    int C = features_tensor.size(1);

    const int *index = index_tensor.data_ptr<int>();
    const float *features = features_tensor.data_ptr<float>();
    float *outs = outs_tensor.data_ptr<float>();

    gather_feature_kernel_launcher(L, C, index, features, outs);

    return 1;
}


int gather_feature_grad_wrapper(at::Tensor grad_outs_tensor, at::Tensor index_tensor, at::Tensor grad_features_tensor){

    CHECK_INPUT(index_tensor);
    CHECK_INPUT(grad_outs_tensor);
    CHECK_INPUT(grad_features_tensor);

    int L = index_tensor.size(0);
    int C = grad_features_tensor.size(1);

    const int *index = index_tensor.data_ptr<int>();
    const float *grad_outs = grad_outs_tensor.data_ptr<float>();
    float *grad_features = grad_features_tensor.data_ptr<float>();

    gather_feature_grad_kernel_launcher(L, C, index, grad_outs, grad_features);

    return 1;
}
