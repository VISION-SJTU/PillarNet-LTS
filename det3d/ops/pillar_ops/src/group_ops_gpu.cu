#include "group_ops_gpu.h"
#include "atomics.cuh"

#define THREADS_PER_BLOCK 512
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))


__global__ void gather_indice_kernel(int L, const int *index, const int *indices, int *outs){
    // index: (L,)  
	// indices: (N, )
    // outs: (L, )

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= L) return;

    outs[tid] = indices[index[tid]];
}


__global__ void gather_feature_kernel(int L, int C, const int *index, const float *features, float *outs){
    // index: (L,)  
	// features: (N, C)
    // outs: (L, C)

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= L) return;

    outs += tid * C;
    features += index[tid] * C;
    for (int c = 0; c < C; ++c){
        outs[c] = features[c];
    }
}

__global__ void gather_feature_grad_kernel(int L, int C, const int *index, const float *grad_outs, float *grad_features){
    // index: (L,)
	// grad_outs: (L, C)
    // grad_features: (N, C)

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= L) return;

    grad_outs += tid * C;
    grad_features += index[tid] * C;
    for (int c = 0; c < C; ++c){
        atomicAdd(grad_features + c, grad_outs[c]);
    }
}


void gather_indice_kernel_launcher(int L, const int *index, const int *indices, int *outs) {
    cudaError_t err;
    dim3 blocks(DIVUP(L, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    gather_indice_kernel<<<blocks, threads>>>(L, index, indices, outs);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


void gather_feature_kernel_launcher(int L, int C, const int *index, const float *features, float *outs){
    cudaError_t err;
    dim3 blocks(DIVUP(L, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    gather_feature_kernel<<<blocks, threads>>>(L, C, index, features, outs);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void gather_feature_grad_kernel_launcher(int L, int C, const int *index, const float *grad_outs, float *grad_features){
    cudaError_t err;
    dim3 blocks(DIVUP(L, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    gather_feature_grad_kernel<<<blocks, threads>>>(L, C, index, grad_outs, grad_features);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

