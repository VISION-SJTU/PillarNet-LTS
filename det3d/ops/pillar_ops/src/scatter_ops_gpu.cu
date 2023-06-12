//
// Created by sgs on 2021/6/21.
//

#include "atomics.cuh"
#include "scatter_ops_gpu.h"


#define THREADS_PER_BLOCK 512
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))


__global__ void scatter_max_kernel(int L, int C, const int *index, const float *src, float *out){
    // src: (L, C)   index: (L)  arg: (M, C)  out: (M, C)
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= L * C) return;

    int pid = tid / C;
    int cid = tid % C;

    atomMax(out + index[pid] * C + cid, src[tid]);
}

__global__ void scatter_arg_max_kernel(int L, int C, const int *index, const float *src, const float *out, int *arg){
    // src: (L, C)   index: (L)  arg: (M, C)  out: (M, C)
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= L * C) return;
    
    int pid = tid / C;
    int cid = tid % C;

    int oid = index[pid] * C + cid;
    if (abs(src[tid] - out[oid]) < 1e-5){
        arg[oid] = tid;
    }
}

__global__ void scatter_max_grad_kernel(int M, int C, const float *grad_out, const int *arg, float *grad_src){
    // arg: (M, C)  grad_out: (M, C) grad_src: (L, C)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M * C) return;

    if (arg[tid] < 0) return;
    grad_src[arg[tid]] = grad_out[tid];
}


void scatter_max_kernel_launcher(int L, int C, const int *index, const float *src, int *arg, float *out) {
    cudaError_t err;
    dim3 blocks(DIVUP(L * C, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    scatter_max_kernel<<<blocks, threads>>>(L, C, index, src, out);
    scatter_arg_max_kernel<<<blocks, threads>>>(L, C, index, src, out, arg);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed (scatter_max_kernel or scatter_arg_max_kernel) : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void scatter_max_grad_kernel_launcher(int M, int C, const float *grad_out, const int *arg, float *grad_src) {
    cudaError_t err;
    dim3 blocks(DIVUP(M * C, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    scatter_max_grad_kernel<<<blocks, threads>>>(M, C, grad_out, arg, grad_src);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed (scatter_max_grad_kernel) : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
