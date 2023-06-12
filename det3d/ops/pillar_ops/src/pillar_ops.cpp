//
// Created by sgs on 2021/7/13.
//

#include "pillar_ops_gpu.h"

//#define assert_expr(x) if(!(x)) { assert_fail(#x, __FILE__, __LINE__, ""); }
//#define assert_expr_msg(x, msg) if(!(x)) { assert_fail(#x, __FILE__, __LINE__, msg); }
//void assert_fail(const char *str, const char *file, int line, std::string msg) {
//	std::cerr << "Assertion failed " << str << " " << file << ":" << line << " " << msg << std::endl;
//	abort();
//}


int create_point_pillar_index_stack_wrapper(at::Tensor pts_xy_tensor, at::Tensor pts_batch_cnt_tensor,
										    at::Tensor pillars_mask_tensor, at::Tensor point_pillar_index_tensor) {
    CHECK_INPUT(pts_xy_tensor);
	CHECK_INPUT(pts_batch_cnt_tensor);
    CHECK_INPUT(pillars_mask_tensor);
    CHECK_INPUT(point_pillar_index_tensor);

    int N = pts_xy_tensor.size(0);
    int B = pillars_mask_tensor.size(0);
    int H = pillars_mask_tensor.size(1);
    int W = pillars_mask_tensor.size(2);

    const int *pts_xy = pts_xy_tensor.data_ptr<int>();
    const int *pts_batch_cnt = pts_batch_cnt_tensor.data_ptr<int>();
    bool *pillars_mask = pillars_mask_tensor.data_ptr<bool>();
    int *point_pillar_index = point_pillar_index_tensor.data_ptr<int>();

    create_point_pillar_index_stack_kernel_launcher(N, B, H, W, pts_xy, pts_batch_cnt, 
                                                    pillars_mask, point_pillar_index);

    return 1;
}


int create_pillar_indices_wrapper(at::Tensor pillars_position_tensor, at::Tensor pillar_indices_tensor) {
    // pillars_position_tensor: (B, H, W)
    // pillar_indices_tensor: (M, 3) [byx]    
	CHECK_INPUT(pillars_position_tensor);
	CHECK_INPUT(pillar_indices_tensor);

	int B = pillars_position_tensor.size(0);
	int H = pillars_position_tensor.size(1);
	int W = pillars_position_tensor.size(2);

	const int *pillars_position = pillars_position_tensor.data_ptr<int>();
	int *pillar_indices = pillar_indices_tensor.data_ptr<int>();

	create_pillar_indices_kernel_launcher(B, H, W, pillars_position, pillar_indices);

	return 1;
}