#include "group_ops_gpu.h"
#include "atomics.cuh"

#define THREADS_PER_BLOCK 512
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))



__global__ void flattenIndicePairsKernel(int N, int K, const int *indicePairs, const int *position,
                                         int *firstIndices, int *secondIndices) {
    // indicePairs: (N, K) -1[none]  position: (N*K, )  -1[none]
    // firstIndices: (L,)            secondIndices: (L,)

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * K) return;

    indicePairs += tid;
    position += tid;
    if (indicePairs[0] < 0) return;

    int index = position[0];
    firstIndices[index] = tid / K;
    secondIndices[index] = indicePairs[0];
}

void flatten_indice_paris_kernel_launcher(int N, int K, const int *indicePairs, const int *position,
                                          int *firstIndices, int *secondIndices) {
    cudaError_t err;

    dim3 blocks(DIVUP(N*K, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    flattenIndicePairsKernel<<<blocks, threads>>>(N, K, indicePairs, position, firstIndices, secondIndices);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed (flattenIndicePairsKernel): %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void gather_feature_kernel(int L, int C, const int *set_indices, const float *features, float *out){
    // set_indices: (L,)         features: (N, C)
    // out: (L, C)

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= L) return;

    set_indices += tid;
    out += tid * C;
    features += set_indices[0] * C;
    for (int c = 0; c < C; ++c){
        out[c] = features[c];
    }
}

__global__ void gather_feature_grad_kernel(int L, int C, const int *set_indices, const float *outGrad, float *inGrad){
    // set_indices: (L,)         grad_features: (L, C)
    // grad_out: (N, C)

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= L) return;

    set_indices += tid;
    outGrad += tid * C;
    inGrad += set_indices[0] * C;
    for (int c = 0; c < C; ++c){
        atomicAdd(inGrad + c, outGrad[c]);
    }
}

void gather_feature_kernel_launcher(int L, int C, const int *set_indices, const float *features, float *out){
    cudaError_t err;
    dim3 blocks(DIVUP(L, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    gather_feature_kernel<<<blocks, threads>>>(L, C, set_indices, features, out);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void gather_feature_grad_kernel_launcher(int L, int C, const int *set_indices, const float *outGrad, float *inGrad){
    cudaError_t err;
    dim3 blocks(DIVUP(L, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    gather_feature_grad_kernel<<<blocks, threads>>>(L, C, set_indices, outGrad, inGrad);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}



__global__ void kernel_average_concatenate_kernel(int N, int M, int K, int C, int kernel, const float *features, const float *new_xy,
                                                  const int *query_xy, const int *query_indices, const uint8_t *is_empty, float *out) {
	// features: (N, C)     new_xy: (M, 2) 
	// query_xy: (M, K, 2)  query_indices: (M, K)  is_empty: (M, K)
	// out: (M, C*2)

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= M * C) return;

	int mid = tid / C;
	int cid = tid % C;

	new_xy += mid * 2;
	query_xy += mid * K * 2;
	query_indices += mid * K;
	is_empty += mid * K;
	out += mid * C * 2 + cid * 2;

	const float cx = new_xy[0];
	const float cy = new_xy[1];

	int cnt = 0;
	float sum1 = 0, sum2 = 0;
	for (int j = 0; j < K; ++j) {
		if (is_empty[j]) continue;

		cnt++;
		const int idx = query_indices[j];
		const float cf = features[idx * C + cid];

		sum1 += (query_xy[j*2 + 0] + 0.5 - cx) * cf / kernel;
		sum2 += (query_xy[j*2 + 1] + 0.5 - cy) * cf / kernel;
	}

	if (cnt > 0) {
		out[0] = sum1 / cnt;
		out[1] = sum2 / cnt;
	}
}


void kernel_average_concatenate_kernel_launcher(int N, int M, int K, int C, int kernel, const float *features, const float *new_xy,
												const int *query_xy, const int *query_indices, const uint8_t *is_empty, float *out) {
	cudaError_t err;
	dim3 blocks(DIVUP(M * C, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
	dim3 threads(THREADS_PER_BLOCK);

	kernel_average_concatenate_kernel<<<blocks, threads>>>(N, M, K, C, kernel, features,
										       new_xy, query_xy, query_indices, is_empty, out);

	err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
		exit(-1);
	}
}


__global__ void kernel_average_concatenate_grad_kernel(int N, int M, int K, int C, int kernel, const float *grad_out, const float *new_xy,
                                                       const int *query_xy, const int *query_indices, const uint8_t *is_empty, float *grad_features) {
	// features: (N, C)    new_xy: (M, 2)
	// query_xy: (M, K, 2) query_indices: (M, K)  is_empty: (M, K)
	// out: (M, C*2)

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= M * C) return;

	int mid = tid / C;
	int cid = tid % C;

	new_xy += mid * 2;
	query_xy += mid * K * 2;
	query_indices += mid * K;
	is_empty += mid * K;
	grad_out += mid * C * 2 + cid * 2;

	const float cx = new_xy[0];
	const float cy = new_xy[1];

	int cnt = 0;
	for (int j = 0; j < K; ++j) {
		if (is_empty[j]) continue;
		cnt++;
	}

	if (cnt == 0) {
		cnt = 1;
	}

	for (int j = 0; j < K; ++j) {
		if (is_empty[j]) continue;

		const float gff = (query_xy[j*2 + 0] + 0.5 - cx) / kernel * grad_out[0] + \
						  (query_xy[j*2 + 1] + 0.5 - cy) / kernel * grad_out[1];
						  
		const int idx = query_indices[j];
		atomAdd(grad_features + idx * C + cid, gff / cnt);
	}
}


void kernel_average_concatenate_grad_kernel_launcher(int N, int M, int K, int C, int kernel, const float *grad_out, const float *new_xy,
                                                     const int *query_xy, const int *query_indices, const uint8_t *is_empty, float *grad_features) {
	cudaError_t err;
	dim3 blocks(DIVUP(M * C, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
	dim3 threads(THREADS_PER_BLOCK);

	kernel_average_concatenate_grad_kernel<<<blocks, threads>>>(N, M, K, C, kernel, grad_out,
	                                                            new_xy, query_xy, query_indices, is_empty, grad_features);

	err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
		exit(-1);
	}
}
