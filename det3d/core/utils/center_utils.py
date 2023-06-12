# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou and Tianwei Yin 
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch import nn
from .circle_nms_jit import circle_nms

def gaussian_radius(det_size, min_overlap=0.5):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _circle_nms(boxes, min_radius, post_max_size=83):
    """
    NMS according to center distance
    """
    keep = np.array(circle_nms(boxes.cpu().numpy(), thresh=min_radius))[:post_max_size]

    keep = torch.from_numpy(keep).long().to(boxes.device)

    return keep 


def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)
    Returns:
    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


def _bboxes_to_corners2d(center, dim):
    corners_norm = torch.tensor([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]],
                                dtype=torch.float32, device=dim.device)
    corners = dim.view([-1, 1, 2]) * corners_norm.view([1, 4, 2])
    corners = corners + center.view(-1, 1, 2)
    return corners


def bbox3d_overlaps_iou(pred_boxes, gt_boxes):
    assert pred_boxes.shape[0] == gt_boxes.shape[0]

    qcorners = _bboxes_to_corners2d(pred_boxes[:, :2], pred_boxes[:, 3:5])
    gcorners = _bboxes_to_corners2d(gt_boxes[:, :2], gt_boxes[:, 3:5])

    inter_max_xy = torch.minimum(qcorners[:, 2], gcorners[:, 2])
    inter_min_xy = torch.maximum(qcorners[:, 0], gcorners[:, 0])

    # calculate area
    volume_pred_boxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
    volume_gt_boxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

    inter_h = torch.minimum(gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]) - \
              torch.maximum(gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5])
    inter_h = torch.clamp(inter_h, min=0)

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    volume_inter = inter[:, 0] * inter[:, 1] * inter_h
    volume_union = volume_gt_boxes + volume_pred_boxes - volume_inter

    ious = volume_inter / volume_union
    ious = torch.clamp(ious, min=0, max=1.0)
    return ious


def bbox3d_overlaps_giou(pred_boxes, gt_boxes):
    assert pred_boxes.shape[0] == gt_boxes.shape[0]

    qcorners = _bboxes_to_corners2d(pred_boxes[:, :2], pred_boxes[:, 3:5])
    gcorners = _bboxes_to_corners2d(gt_boxes[:, :2], gt_boxes[:, 3:5])

    inter_max_xy = torch.minimum(qcorners[:, 2], gcorners[:, 2])
    inter_min_xy = torch.maximum(qcorners[:, 0], gcorners[:, 0])
    out_max_xy = torch.maximum(qcorners[:, 2], gcorners[:, 2])
    out_min_xy = torch.minimum(qcorners[:, 0], gcorners[:, 0])

    # calculate area
    volume_pred_boxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
    volume_gt_boxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

    inter_h = torch.minimum(gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]) - \
              torch.maximum(gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5])
    inter_h = torch.clamp(inter_h, min=0)

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    volume_inter = inter[:, 0] * inter[:, 1] * inter_h
    volume_union = volume_gt_boxes + volume_pred_boxes - volume_inter

    outer_h = torch.maximum(gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]) - \
              torch.minimum(gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5])
    outer_h = torch.clamp(outer_h, min=0)
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    closure = outer[:, 0] * outer[:, 1] * outer_h

    gious = volume_inter / volume_union - (closure - volume_union) / closure
    gious = torch.clamp(gious, min=-1.0, max=1.0)
    return gious


def bbox3d_overlaps_diou(pred_boxes, gt_boxes):
    assert pred_boxes.shape[0] == gt_boxes.shape[0]

    qcorners = _bboxes_to_corners2d(pred_boxes[:, :2], pred_boxes[:, 3:5])
    gcorners = _bboxes_to_corners2d(gt_boxes[:, :2], gt_boxes[:, 3:5])

    inter_max_xy = torch.minimum(qcorners[:, 2], gcorners[:, 2])
    inter_min_xy = torch.maximum(qcorners[:, 0], gcorners[:, 0])
    out_max_xy = torch.maximum(qcorners[:, 2], gcorners[:, 2])
    out_min_xy = torch.minimum(qcorners[:, 0], gcorners[:, 0])

    # calculate area
    volume_pred_boxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
    volume_gt_boxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

    inter_h = torch.minimum(pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5]) - \
              torch.maximum(pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5])
    inter_h = torch.clamp(inter_h, min=0)

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    volume_inter = inter[:, 0] * inter[:, 1] * inter_h
    volume_union = volume_gt_boxes + volume_pred_boxes - volume_inter

    # boxes_iou3d_gpu(pred_boxes, gt_boxes)
    inter_diag = torch.pow(gt_boxes[:, 0:3] - pred_boxes[:, 0:3], 2).sum(-1)

    outer_h = torch.maximum(gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]) - \
              torch.minimum(gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5])
    outer_h = torch.clamp(outer_h, min=0)
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = outer[:, 0] ** 2 + outer[:, 1] ** 2 + outer_h ** 2

    dious = volume_inter / volume_union - inter_diag / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)

    return dious


def set_by_task_cfg(test_cfg, task_num_classes):
    """
    -if single float, copy for each task head;
    -if list, re-organize for multiple task heads.

    :param test_cfg:
      nms=dict(
            use_rotate_nms=False,
            # nms_pre_max_size=4096,
            # nms_post_max_size=500,
            # nms_iou_threshold=0.7,
            use_multi_class_nms=True,
            nms_pre_max_size=[2048, 1024, 1024],
            nms_post_max_size=[300, 150, 150],
            nms_iou_threshold=[0.8, 0.55, 0.55],
        ),
        rectifier=[0, 0, 0],
        score_threshold=0.1,
        :param task_num_classes: [int] * num_classes
    :return:
    """
    def _param_org(param):
        if isinstance(param, float) or isinstance(param, int):
            return param

        assert isinstance(param, list) or isinstance(param, tuple)
        assert len(param) == sum(task_num_classes)

        ret_list = [[]] * len(task_num_classes)
        flag = 0
        for k, num in enumerate(task_num_classes):
            ret_list[k] = list(param[flag:flag+num])
            flag += num
        return ret_list

    if test_cfg.get('rectifier', False):
        test_cfg['rectifier'] = _param_org(test_cfg['rectifier'])
        
    if test_cfg.get('use_rectify', False):
        test_cfg['use_rectify'] = _param_org(test_cfg['use_rectify'])

    test_cfg['nms']['nms_pre_max_size'] = _param_org(test_cfg['nms']['nms_pre_max_size'])
    test_cfg['nms']['nms_post_max_size'] = _param_org(test_cfg['nms']['nms_post_max_size'])
    test_cfg['nms']['nms_iou_threshold'] = _param_org(test_cfg['nms']['nms_iou_threshold'])

    return test_cfg