import torch

from det3d.models.registry import POINT_HEAD
from .point_head_template import PointHeadTemplate


def enlarge_box3d(boxes3d, extra_width=[0.2, 0.2, 0.2]):
    large_boxes3d = boxes3d.clone()
    large_boxes3d[:, 3:6] += boxes3d.new_tensor(extra_width)[None, :]
    return large_boxes3d


@POINT_HEAD.register_module
class PointHead(PointHeadTemplate):
    def __init__(self, num_class, input_channels, model_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class, **kwargs)
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) or (N1 + N2 + N3 + ..., 3) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict['point_coords']

        batch_size = input_dict['batch_size']
        if point_coords[0].shape[-1] == 2:
            gt_boxes = input_dict['roi_targets_dict']['gt_of_rois_src']
            targets_dict = self.assign_stack_targets_2d(
                points=point_coords, gt_boxes=gt_boxes
            )
        # elif isinstance(point_coords, tuple) and point_coords[0].shape[-1] == 3:
        #     gt_boxes = input_dict['gt_box']
        #     assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        #
        #     extend_gt_boxes = enlarge_box3d(
        #         gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        #     ).view(batch_size, -1, gt_boxes.shape[-1])
        #     targets_dict = self.assign_stack_targets_3d(
        #         points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
        #         set_ignore_flag=True, use_ball_constraint=False,
        #         ret_part_labels=False
        #     )
        else:
            raise NotImplementedError

        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()

        point_loss = point_loss_cls
        tb_dict.update(tb_dict_1)
        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        point_features = batch_dict['point_features']
        point_cls_preds = self.cls_layers(point_features.view(-1, point_features.shape[-1]))  # (total_points, num_class)

        ret_dict = {
            'point_cls_preds': point_cls_preds,
        }

        batch_dict['point_cls_scores'] = torch.sigmoid(point_cls_preds)

        if self.model_cfg.get('ATT_MODEL', False):
            point_features = point_features * batch_dict['point_cls_scores'].view(*point_features.shape[:-1], 1)
            batch_dict['point_features'] = point_features

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
        self.forward_ret_dict = ret_dict

        return batch_dict
