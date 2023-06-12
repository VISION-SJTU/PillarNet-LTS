import torch 
from torch import nn
from ..registry import DETECTORS
from .base import BaseDetector
from .. import builder


@DETECTORS.register_module
class PillarRCNN(BaseDetector):
    def __init__(self,
                 first_stage_cfg,
                 second_stage_modules,
                 roi_head, 
                 freeze=False,
                 point_head=None,
                 **kwargs):
        super(PillarRCNN, self).__init__()
        self.single_det = builder.build_detector(first_stage_cfg, **kwargs)

        if freeze:
            print("Freeze First Stage Network")
            # we train the model in two steps 
            self.single_det = self.single_det.freeze()
            
        self.bbox_head = self.single_det.bbox_head
        self.test_cfg = self.single_det.test_cfg
        self.num_classes = sum(self.single_det.num_classes)
        
        first_cfg = dict(backbone_channels=self.single_det.backbone.backbone_channels,
                         backbone_strides=self.single_det.backbone.backbone_strides)
        
        self.second_stage = nn.ModuleList()
        # can be any number of modules 
        # bird eye view, cylindrical view, image, multiple timesteps, etc.. 
        for module in second_stage_modules:
            module.update(first_cfg)
            self.second_stage.append(builder.build_second_stage_module(module))
        
        self.point_head = point_head
        if point_head is not None:
            self.point_head = builder.build_point_head(point_head)
        self.roi_head = builder.build_roi_head(roi_head)

    def combine_loss(self, one_stage_loss, point_loss, roi_loss, tb_dict):
        for i in range(len(one_stage_loss['loss'])):
            one_stage_loss['roi_reg_loss'].append(tb_dict['rcnn_loss_reg'])
            one_stage_loss['roi_cls_loss'].append(tb_dict['rcnn_loss_cls'])

        one_stage_loss['loss'][0] += (roi_loss) + (point_loss)
        return one_stage_loss

    def reorder_first_stage_prediction(self, first_pred, example):
        batch_size = len(first_pred)
        box_length = first_pred[0]['box3d_lidar'].shape[1]

        NMS_POST_MAXSIZE = self.single_det.NMS_POST_MAXSIZE
        rois = first_pred[0]['box3d_lidar'].new_zeros((batch_size, NMS_POST_MAXSIZE, box_length))
        roi_scores = first_pred[0]['scores'].new_zeros((batch_size, NMS_POST_MAXSIZE))
        roi_labels = first_pred[0]['label_preds'].new_zeros((batch_size, NMS_POST_MAXSIZE), dtype=torch.long)

        for i in range(batch_size):
            # basically move rotation to position 6, so now the box is 7 + C . C is 2 for nuscenes to
            # include velocity target
            box_preds = first_pred[i]['box3d_lidar']
            num_obj = box_preds.shape[0]

            if self.roi_head.code_size == 9:
                # x, y, z, w, l, h, rotation_y, velocity_x, velocity_y
                box_preds = box_preds[:, [0, 1, 2, 3, 4, 5, 8, 6, 7]]

            rois[i, :num_obj] = box_preds
            roi_labels[i, :num_obj] = first_pred[i]['label_preds'] + 1
            roi_scores[i, :num_obj] = first_pred[i]['scores']
            # roi_features[i, :num_obj] = features[i].view(num_obj, -1)

        example['rois'] = rois 
        example['roi_labels'] = roi_labels 
        example['roi_scores'] = roi_scores  
        # example['roi_features'] = roi_features

        example['has_class_labels']= True 

        return example

    def forward(self, example, return_loss=True, **kwargs):
        batch_size = len(example['metadata'])
        example['batch_size'] = batch_size

        out = self.single_det.forward_two_stage(example, return_loss, **kwargs)
        one_stage_pred, bev_features, backbone_features, one_stage_loss = out
        example['bev_feature'] = bev_features[-1]
        example['backbone_features'] = backbone_features

        if self.roi_head.code_size == 7 and return_loss is True:
            # drop velocity 
            example['gt_boxes_and_cls'] = example['gt_boxes_and_cls'][:, :, [0, 1, 2, 3, 4, 5, 6, -1]]
        
        # from tools.visual import draw_scenes
        # points = example["points"][0].cpu().numpy()
        # gt_boxes = example["gt_boxes_and_cls"][0][:10, :7].cpu().numpy()
        # gt_boxes2 = example["gt_box"][0][0, :10].cpu().numpy()
        # draw_scenes(points, gt_boxes=gt_boxes)

        example = self.reorder_first_stage_prediction(first_pred=one_stage_pred, example=example)

        if self.training:
            targets_dict = self.roi_head.assign_targets(example)
            example['rois'] = targets_dict['rois']
            example['roi_labels'] = targets_dict['roi_labels']
            example['roi_scores'] = targets_dict['roi_scores']
            example['roi_targets_dict'] = targets_dict

        for module in self.second_stage:
            example = module.forward(example)

        if self.point_head is not None:
            example = self.point_head(example)

        # final classification / regression 
        batch_dict = self.roi_head(example, training=return_loss)

        if return_loss:
            roi_loss, tb_dict = self.roi_head.get_loss()

            point_loss = 0
            if self.point_head is not None:
                point_loss, tb_dict = self.point_head.get_loss(tb_dict)

            return self.combine_loss(one_stage_loss, point_loss, roi_loss, tb_dict)
        else:
            return self.post_process(batch_dict)
    
    def post_process(self, batch_dict):
        pred_dicts = [] 
        
        batch_size = batch_dict['batch_size']
        for index in range(batch_size):
            box_preds = batch_dict['batch_box_preds'][index]
            cls_preds = batch_dict['batch_cls_preds'][index]  # this is the predicted iou 
            label_preds = batch_dict['roi_labels'][index]

            if box_preds.shape[-1] == 9:
                # move rotation to the end (the create submission file will take elements from 0:6 and -1) 
                box_preds = box_preds[:, [0, 1, 2, 3, 4, 5, 7, 8, 6]]

            scores = torch.sqrt(torch.sigmoid(cls_preds).view(-1) * batch_dict['roi_scores'][index].view(-1)) 
            mask = (label_preds != 0).view(-1)
            
            # remove box dim < 0
            mask = mask & (box_preds[:, 3:6] > 0).all(1)

            box_preds = box_preds[mask, :]
            scores = scores[mask]
            labels = label_preds[mask] - 1

            # currently don't need nms 
            pred_dict = {
                'box3d_lidar': box_preds,
                'scores': scores,
                'label_preds': labels,
                "metadata": batch_dict["metadata"][index]
            }

            pred_dicts.append(pred_dict)

        return pred_dicts 
