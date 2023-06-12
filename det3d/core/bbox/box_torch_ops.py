import numpy as np
import torch
from torch import stack as tstack
from det3d.ops.iou3d_nms import iou3d_nms_cuda
from det3d.ops.iou3d_nms import to_pcdet


def torch_to_np_dtype(ttype):
    type_map = {
        torch.float16: np.dtype(np.float16),
        torch.float32: np.dtype(np.float32),
        torch.float16: np.dtype(np.float64),
        torch.int32: np.dtype(np.int32),
        torch.int64: np.dtype(np.int64),
        torch.uint8: np.dtype(np.uint8),
    }
    return type_map[ttype]


def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
        dtype (output dtype, optional): Defaults to np.float32

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    dtype = torch_to_np_dtype(dims.dtype)
    if isinstance(origin, float):
        origin = [origin] * ndim
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1
    ).astype(dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start from minimum point
    # for 3d boxes, please draw them by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dtype)
    corners_norm = torch.from_numpy(corners_norm).type_as(dims)
    corners = dims.view(-1, 1, ndim) * corners_norm.view(1, 2 ** ndim, ndim)
    return corners


def corners_2d(dims, origin=0.5):
    """generate relative 2d box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, 2]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
        dtype (output dtype, optional): Defaults to np.float32

    Returns:
        float array, shape=[N, 4, 2]: returned corners.
        point layout: x0y0, x0y1, x1y1, x1y0
    """
    return corners_nd(dims, origin)


def corner_to_standup_nd(boxes_corner):
    ndim = boxes_corner.shape[2]
    standup_boxes = []
    for i in range(ndim):
        standup_boxes.append(torch.min(boxes_corner[:, :, i], dim=1)[0])
    for i in range(ndim):
        standup_boxes.append(torch.max(boxes_corner[:, :, i], dim=1)[0])
    return torch.stack(standup_boxes, dim=1)


def rotation_3d_in_axis(points, angles, axis=0):
    # points: [N, point_size, 3]
    # angles: [N]
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = tstack(
            [
                tstack([rot_cos, zeros, -rot_sin]),
                tstack([zeros, ones, zeros]),
                tstack([rot_sin, zeros, rot_cos]),
            ]
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = tstack(
            [
                tstack([rot_cos, -rot_sin, zeros]),
                tstack([rot_sin, rot_cos, zeros]),
                tstack([zeros, zeros, ones]),
            ]
        )
    elif axis == 0:
        rot_mat_T = tstack(
            [
                tstack([zeros, rot_cos, -rot_sin]),
                tstack([zeros, rot_sin, rot_cos]),
                tstack([ones, zeros, zeros]),
            ]
        )
    else:
        raise ValueError("axis should in range")
    # print(points.shape, rot_mat_T.shape)
    return torch.einsum("aij,jka->aik", points, rot_mat_T)

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  -sina, zeros,
        sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot


def rotate_points2d_along_z(points, angle):
    """
    Args:
        points: (B, N, 2 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    rot_matrix = torch.stack((
        cosa,  -sina,
        sina, cosa
    ), dim=1).view(-1, 2, 2).float()
    points_rot = torch.matmul(points[:, :, 0:2], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 2:]), dim=-1)
    return points_rot


def rotation_2d(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    rot_mat_T = torch.stack([tstack([rot_cos, -rot_sin]), tstack([rot_sin, rot_cos])])
    return torch.einsum("aij,jka->aik", (points, rot_mat_T))


def center_to_corner_box3d(centers, dims, angles, origin=(0.5, 0.5, 0.5), axis=1):
    """convert kitti locations, dimensions and angles to corners

    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.view(-1, 1, 3)
    return corners


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners

    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.

    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.view(-1, 1, 2)
    return corners


def center_to_grid_box2d(centers, dims, angles=None, grid_size=(6, 6)):
    """convert kitti locations, dimensions and angles to grid points

    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.

    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = get_dense_roi_grid_points(dims, grid_size)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.view(-1, 1, 2)
    return corners

def get_dense_roi_grid_points(dims, grid_size):
    batch_size_rcnn = dims.shape[0]

    faked_features = dims.new_ones((grid_size[0], grid_size[1]))
    dense_idx = faked_features.nonzero()  # (6x6, 2) [x_idx, y_idx]
    dense_idx = dense_idx.unsqueeze(0).repeat(batch_size_rcnn, 1, 1).float()  # (B*6x6x6, 2)

    dims = dims.view(batch_size_rcnn, 1, -1)
    roi_grid_points = (dense_idx + 0.5) / torch.tensor(grid_size, device=dims.device) * dims \
                      - (dims / 2)  # (B, 6x6, 2)
    return roi_grid_points


def project_to_image(points_3d, proj_mat):
    points_num = list(points_3d.shape)[:-1]
    points_shape = np.concatenate([points_num, [1]], axis=0).tolist()
    points_4 = torch.cat(
        [points_3d, torch.ones(*points_shape).type_as(points_3d)], dim=-1
    )
    # point_2d = points_4 @ tf.transpose(proj_mat, [1, 0])
    point_2d = torch.matmul(points_4, proj_mat.t())
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    return point_2d_res


def camera_to_lidar(points, r_rect, velo2cam):
    num_points = points.shape[0]
    points = torch.cat([points, torch.ones(num_points, 1).type_as(points)], dim=-1)
    lidar_points = points @ torch.inverse((r_rect @ velo2cam).t())
    return lidar_points[..., :3]


def lidar_to_camera(points, r_rect, velo2cam):
    num_points = points.shape[0]
    points = torch.cat([points, torch.ones(num_points, 1).type_as(points)], dim=-1)
    camera_points = points @ (r_rect @ velo2cam).t()
    return camera_points[..., :3]


def box_camera_to_lidar(data, r_rect, velo2cam):
    xyz = data[..., 0:3]
    l, h, w = data[..., 3:4], data[..., 4:5], data[..., 5:6]
    r = data[..., 6:7]
    xyz_lidar = camera_to_lidar(xyz, r_rect, velo2cam)
    return torch.cat([xyz_lidar, w, l, h, r], dim=-1)


def box_lidar_to_camera(data, r_rect, velo2cam):
    xyz_lidar = data[..., 0:3]
    w, l, h = data[..., 3:4], data[..., 4:5], data[..., 5:6]
    r = data[..., 6:7]
    xyz = lidar_to_camera(xyz_lidar, r_rect, velo2cam)
    return torch.cat([xyz, l, h, w, r], dim=-1)


def rotate_nms_pcdet(box_preds, score_preds, iou_preds, label_preds, rectifier, 
                     nms_thresh, pre_maxsize=None, post_max_size=None, use_rectify=False):
    # transform back to pcdet's coordinate
    box_preds_pcdet = to_pcdet(box_preds.clone())
    
    rect_scores = torch.pow(score_preds, 1 - rectifier) * torch.pow(iou_preds, rectifier)
    
    order = rect_scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    box_preds_pcdet = box_preds_pcdet[order].contiguous()
    keep = torch.LongTensor(box_preds_pcdet.size(0))

    if len(box_preds_pcdet) == 0:
        num_out =0
    else:
        num_out = iou3d_nms_cuda.nms_gpu(box_preds_pcdet, keep, nms_thresh)

    selected = order[keep[:num_out].cuda()].contiguous()
    if post_max_size is not None:
        selected = selected[:post_max_size]
    
    if use_rectify:
        return box_preds[selected], rect_scores[selected], label_preds[selected]
    
    return box_preds[selected], score_preds[selected], label_preds[selected]


def rotate_class_nms_pcdet(box_preds, score_preds, iou_preds, label_preds, nms_thresh,
                           rectifiers, pre_maxsize, post_max_size, use_rectify=False):
    
    assert isinstance(rectifiers, list)
    assert isinstance(nms_thresh, list)
    assert isinstance(pre_maxsize, list)
    assert isinstance(post_max_size, list)
    
    num_classes = len(rectifiers)
    assert len(rectifiers) == len(nms_thresh) == len(pre_maxsize) == len(post_max_size)
    
    box_preds_list, scores_list, labels_list = [], [], []
    for k in range(num_classes):
        mask = label_preds == k
        box_preds_cls = box_preds[mask]
        scores_cls = score_preds[mask]
        ious_cls = iou_preds[mask]
        labels_cls = label_preds[mask]
        
        if isinstance(use_rectify, list) or isinstance(use_rectify, tuple):
            cur_use_rectify = use_rectify[k]
        else:
            cur_use_rectify = use_rectify

        selected_boxes, selected_scores, selected_labels = rotate_nms_pcdet(
            box_preds_cls, scores_cls, ious_cls, labels_cls, rectifiers[k], 
            nms_thresh[k], pre_maxsize[k], post_max_size[k], use_rectify=cur_use_rectify)

        box_preds_list.append(selected_boxes)
        scores_list.append(selected_scores)
        labels_list.append(selected_labels)

    return torch.cat(box_preds_list, dim=0), \
            torch.cat(scores_list, dim=0), \
            torch.cat(labels_list, dim=0)
            

# following functions is targeted for R-CNN
def rotate_nms_pcdetv1(box_preds, score_preds, label_preds, nms_thresh):
    # transform back to pcdet's coordinate
    box_preds_pcdet = box_preds[:, [0, 1, 2, 4, 3, 5, -1]].clone()
    box_preds_pcdet[:, -1] = -box_preds_pcdet[:, -1] - np.pi / 2 
        
    order = score_preds.sort(0, descending=True)[1]

    box_preds_pcdet = box_preds_pcdet[order].contiguous()
    keep = torch.LongTensor(box_preds_pcdet.size(0))

    if len(box_preds_pcdet) == 0:
        num_out =0
    else:
        num_out = iou3d_nms_cuda.nms_gpu(box_preds_pcdet, keep, nms_thresh)

    selected = order[keep[:num_out].cuda()].contiguous()
    
    return box_preds[selected], score_preds[selected], label_preds[selected]


def rotate_class_nms_pcdetv1(num_class, box_preds, score_preds, label_preds, nms_thresh):
    
    assert isinstance(nms_thresh, list)

    box_preds_list, scores_list, labels_list = [], [], []
    for k in range(num_class):
        mask = label_preds == k
        box_preds_cls = box_preds[mask]
        scores_cls = score_preds[mask]
        labels_cls = label_preds[mask]

        selected_boxes, selected_scores, selected_labels = rotate_nms_pcdetv1(
            box_preds_cls, scores_cls, labels_cls, nms_thresh[k])

        box_preds_list.append(selected_boxes)
        scores_list.append(selected_scores)
        labels_list.append(selected_labels)

    return torch.cat(box_preds_list, dim=0), \
            torch.cat(scores_list, dim=0), \
            torch.cat(labels_list, dim=0)