//
// Created by sgs on 2021/7/13.
//

#include "atomics.cuh"
#include "pillar_ops_gpu.h"


#define THREADS_PER_BLOCK 512
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))


__global__ void createPointPillarIndexStackKernel(int N, int B, int H, int W,
												  const int *pts_xy, const int *pts_batch_cnt, 
                                                  bool *pillar_mask, int *point_pillar_index) {
	// pts_xy: (N1+N2..., 2)
    // pts_batch_cnt: (N1, N2, ...)
    // pillar_mask: (B, H, W)
    // point_pillar_index: (N1+N2..., ) default -1
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    pts_xy += tid * 2;
    int bid = 0, pts_cnt = pts_batch_cnt[0];
    for (int b = 1; b < B; ++b){
    	if (tid < pts_cnt) break;
    	pts_cnt += pts_batch_cnt[b];
    	bid = b;
    }

    int xid = pts_xy[0];
    int yid = pts_xy[1];

    if (xid < 0 || xid >= W || yid < 0 || yid >= H) return;

    int idx = bid * H * W + yid * W + xid;
    pillar_mask[idx] = 1;
    point_pillar_index[tid] = idx;
}


void create_point_pillar_index_stack_kernel_launcher(int N, int B, int H, int W,
												     const int *pts_xy, const int *pts_batch_cnt, 
                                                     bool *pillar_mask, int *point_pillar_index) {
    cudaError_t err;
    dim3 blocks(DIVUP(N, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    createPointPillarIndexStackKernel<<<blocks, threads>>>(N, B, H, W, pts_xy, pts_batch_cnt, 
                                                           pillar_mask, point_pillar_index);

    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void createPillarIndicesKernel(int B, int H, int W, const int *pillars_position, int *pillar_indices){
    // pillars_position: (B*H*W)
	// pillar_indices: (L, 3)  [bid, yid, xid]   

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= B * H * W) return;

	pillars_position += tid;
	if (pillars_position[0] < 0) return;

	int bid = tid / (H * W);
    int yid = (tid / W) % H;
	int xid = tid % W;

	const int idx = pillars_position[0] * 3;
	pillar_indices[idx + 0] = bid;
	pillar_indices[idx + 1] = yid;
	pillar_indices[idx + 2] = xid;
}


void create_pillar_indices_kernel_launcher(int B, int H, int W, const int *pillars_position, int *pillar_indices){
	cudaError_t err;
	dim3 blocks(DIVUP(B * H * W, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
	dim3 threads(THREADS_PER_BLOCK);

	createPillarIndicesKernel<<<blocks, threads>>>(B, H, W, pillars_position, pillar_indices);

	// cudaDeviceSynchronize();  // for using printf in kernel function
	err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
		exit(-1);
	}
}
