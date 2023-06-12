//
// Created by sgs on 2021/6/21.
//
#include "scatter_ops_gpu.h"


int scatter_max_wrapper(at::Tensor index_tensor, at::Tensor src_tensor, at::Tensor arg_tensor, at::Tensor out_tensor){
    // src: (L, C)   index: (L)  arg: (M, C)  out: (M, C)
    CHECK_INPUT(src_tensor);
    CHECK_INPUT(index_tensor);
    CHECK_INPUT(arg_tensor);
    CHECK_INPUT(out_tensor);

    int L = src_tensor.size(0);
    int C = src_tensor.size(1);

    const float *src = src_tensor.data_ptr<float>();
    const int *index = index_tensor.data_ptr<int>();
    int * arg = arg_tensor.data_ptr<int>();
    float *out = out_tensor.data_ptr<float>();

    scatter_max_kernel_launcher(L, C, index, src, arg, out);
    return 1;
}

int scatter_max_grad_wrapper(at::Tensor grad_out_tensor, at::Tensor arg_tensor, at::Tensor grad_src_tensor){
    // arg: (M, C)   grad_out: (M, C)  grad_src: (L, C)
    CHECK_INPUT(arg_tensor);
    CHECK_INPUT(grad_out_tensor);
    CHECK_INPUT(grad_src_tensor);

    int M = grad_out_tensor.size(0);
    int C = grad_out_tensor.size(1);

    const int *arg = arg_tensor.data_ptr<int>();
    const float *grad_out = grad_out_tensor.data_ptr<float>();
    float *grad_src = grad_src_tensor.data_ptr<float>();

    scatter_max_grad_kernel_launcher(M, C, grad_out, arg, grad_src);
    return 1;
}
