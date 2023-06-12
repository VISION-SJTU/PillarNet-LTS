import copy, logging
from collections import defaultdict
from typing import Dict, List, Tuple
import torch
from torch import nn
from det3d.core import box_torch_ops
from det3d.torchie.cnn import kaiming_init
from det3d.models.losses.centernet_loss import FastFocalLoss, RegLoss, IouLoss, IouRegLoss
from det3d.models.utils import Sequential
from ..registry import HEADS
from det3d.core.utils.circle_nms_jit import circle_nms


class SepHead(nn.Module):
    def __init__(self, 
                 in_channels, 
                 heads, 
                 head_conv=64, 
                 init_bias=-2.19, 
                 **kwargs):
        super().__init__(**kwargs)

        self.heads = heads 
        for head in self.heads:
            classes, num_conv = self.heads[head]

            fc = Sequential()
            for i in range(num_conv-1):
                fc.add(nn.Conv2d(
                    in_channels, head_conv, 3, stride=1, padding=1, bias=True))
                fc.add(nn.BatchNorm2d(head_conv, momentum=0.01, eps=1e-3))
                fc.add(nn.ReLU())

            fc.add(nn.Conv2d(
                head_conv, classes, 3, stride=1, padding=1, bias=True))    

            if 'hm' in head:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)

            self.__setattr__(head, fc)
        
    def forward(self, x):
        ret_dict = dict()        
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict


@HEADS.register_module
class CenterHead(nn.Module):
    def __init__(self, 
                 tasks: List[dict],
                 in_channels: List,
                 code_weights: List,
                 common_heads=dict(),
                 logger=None,
                 share_channel=64,
                 reg_iou=None,
                 pillar_size=0.1,
                 point_cloud_range=[-75.2, -75.2, -2, 75.2, 75.2, 4]):
        super().__init__()

        self.num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.task_strides = [int(t.stride) for t in tasks]
        self.code_weights = code_weights 
        self.pillar_size = pillar_size
        self.point_cloud_range = point_cloud_range
        
        # create task-to-stride mapping
        tmp_list = list(set(self.task_strides)); tmp_list.sort()
        tmp_list = tmp_list[::-1]
        assert len(in_channels) == len(tmp_list)
        self.task_idx = [tmp_list.index(item) for item in self.task_strides]
        
        self.crit = FastFocalLoss()
        self.crit_reg = RegLoss()

        self.use_iou = 'iou' in common_heads
        self.use_reg_iou = reg_iou is not None
        if self.use_iou:
            self.crit_iou = IouLoss()
        if self.use_reg_iou:
            self.crit_reg_iou = IouRegLoss(reg_iou)

        self.box_n_dim = 9 if 'vel' in common_heads else 7  
        self.use_direction_classifier = False 

        if not logger:
            logger = logging.getLogger("CenterHead")
        self.logger = logger

        logger.info(f"num_classes: {self.num_classes}")

        # a shared convolution 
        self.share_convs = nn.ModuleList()
        for channels in in_channels:
            self.share_convs.append(nn.Sequential(
                nn.Conv2d(channels, share_channel, 3, padding=1, bias=True),
                nn.BatchNorm2d(share_channel, momentum=0.01, eps=1e-3),
                nn.ReLU()))

        self.task_heads = nn.ModuleList()
        for num_cls in self.num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(hm=(num_cls, 2)))
            self.task_heads.append(SepHead(share_channel, heads))

        logger.info("Finish CenterHead Initialization")

    def forward(self, x: Tuple):
        assert len(x) == len(self.share_convs)
        
        share_feats = []
        for k, share_conv in enumerate(self.share_convs):
            share_feats.append(share_conv(x[k]))
        
        ret_dicts = []
        for k, task_head in enumerate(self.task_heads):
            ret_dicts.append(task_head(share_feats[self.task_idx[k]]))

        return ret_dicts

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
        return y

    def loss(self, example, preds_dicts, train_cfg, **kwargs):
        
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            # convert (B C H W) to (B H W C )
            for key, val in preds_dict.items():
                preds_dict[key] = val.permute(0, 2, 3, 1).contiguous()
                
            preds_dict['hm'] = self._sigmoid(preds_dict['hm'])
            hm_loss = self.crit(preds_dict['hm'], example['hm'][task_id], example['ind'][task_id], 
                                example['mask'][task_id], example['cat'][task_id])

            target_box = example['anno_box'][task_id]
            # reconstruct the anno_box from multiple reg heads
            if 'vel' in preds_dict:
                preds_dict['anno_box'] = torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                    preds_dict['vel'], preds_dict['rot']), dim=-1)  
            else:
                preds_dict['anno_box'] = torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                    preds_dict['rot']), dim=-1)   
                target_box = target_box[..., [0, 1, 2, 3, 4, 5, -2, -1]] # remove vel target                     
        
            ret = {}
            
            # Regression loss for dimension, offset, height, rotation            
            box_loss = self.crit_reg(preds_dict['anno_box'], example['mask'][task_id], 
                                     example['ind'][task_id], target_box)

            loc_loss = (box_loss * box_loss.new_tensor(self.code_weights)).sum()

            loss = hm_loss * train_cfg.hm_weight + loc_loss * train_cfg.bbox_weight
            
            ret.update({'hm_loss': hm_loss.detach().cpu(), 
                        'loc_loss': loc_loss, 'loc_loss_elem': box_loss.detach().cpu(),
                        'num_positive': example['mask'][task_id].float().sum()})

            if self.use_iou or self.use_reg_iou:
                # Only dimensions between [0.3, 25] is valid
                batch_dim = torch.exp(preds_dict['dim'].clamp(min=-1.2, max=3.2))
                batch_reg = preds_dict['reg']
                batch_hei = preds_dict['height']
                batch_rots = preds_dict['rot'][..., 0:1]
                batch_rotc = preds_dict['rot'][..., 1:2]
                batch_rot = torch.atan2(batch_rots, batch_rotc)

                batch, H, W, _ = batch_dim.size()
                
                ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
                ys = ys.view(1, H, W).repeat(batch, 1, 1).to(batch_dim)
                xs = xs.view(1, H, W).repeat(batch, 1, 1).to(batch_dim)

                xs = xs.view(batch, H, W, 1) + batch_reg[:, :, :, 0:1]
                ys = ys.view(batch, H, W, 1) + batch_reg[:, :, :, 1:2]

                xs = xs * self.task_strides[task_id] * self.pillar_size + self.point_cloud_range[0]
                ys = ys * self.task_strides[task_id] * self.pillar_size + self.point_cloud_range[1]
                batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_rot], dim=-1)

            if self.use_iou:
                iou_loss = self.crit_iou(preds_dict['iou'], example['mask'][task_id], 
                                         example['ind'][task_id], batch_box_preds.detach(), 
                                         example['gt_box'][task_id])
                loss = loss + iou_loss * train_cfg.iou_weight
                ret.update({'iou_loss': iou_loss.detach().cpu()})

            if self.use_reg_iou:
                reg_iou_loss = self.crit_reg_iou(batch_box_preds, example['mask'][task_id], 
                                                 example['ind'][task_id], example['gt_box'][task_id])
                loss = loss + reg_iou_loss * train_cfg.reg_iou_weight
                ret.update({'reg_iou_loss': reg_iou_loss.detach().cpu()})

            ret.update({'loss': loss})
            rets.append(ret)
        
        """convert batch-key to key-batch
        """
        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)

        return rets_merged

    @torch.no_grad()
    def predict(self, example, preds_dicts, test_cfg, **kwargs):
        """decode, nms, then return the detection result. Additionaly support double flip testing 
        """
        # get loss info
        rets = []
        metas = []

        double_flip = test_cfg.get('double_flip', False)

        for task_id, preds_dict in enumerate(preds_dicts):
            # convert N C H W to N H W C 
            for key, val in preds_dict.items():
                preds_dict[key] = val.permute(0, 2, 3, 1).contiguous()

            batch_size = preds_dict['hm'].shape[0]

            if double_flip:
                assert batch_size % 4 == 0, print(batch_size)
                batch_size = int(batch_size / 4)
                for k in preds_dict.keys():
                    # transform the prediction map back to their original coordinate befor flipping
                    # the flipped predictions are ordered in a group of 4. The first one is the original pointcloud
                    # the second one is X flip pointcloud(y=-y), the third one is Y flip pointcloud(x=-x), and the last one is 
                    # X and Y flip pointcloud(x=-x, y=-y).
                    # Also please note that pytorch's flip function is defined on higher dimensional space, so dims=[2] means that
                    # it is flipping along the axis with H length(which is normaly the Y axis), however in our traditional word, it is flipping along
                    # the X axis. The below flip follows pytorch's definition yflip(y=-y) xflip(x=-x)
                    _, H, W, C = preds_dict[k].shape
                    preds_dict[k] = preds_dict[k].reshape(int(batch_size), 4, H, W, C)
                    preds_dict[k][:, 1] = torch.flip(preds_dict[k][:, 1], dims=[1]) 
                    preds_dict[k][:, 2] = torch.flip(preds_dict[k][:, 2], dims=[2])
                    preds_dict[k][:, 3] = torch.flip(preds_dict[k][:, 3], dims=[1, 2])

            if "metadata" not in example or len(example["metadata"]) == 0:
                meta_list = [None] * batch_size
            else:
                meta_list = example["metadata"]
                if double_flip:
                    meta_list = meta_list[:4*int(batch_size):4]

            batch_hm = torch.sigmoid(preds_dict['hm'])
            # Only dimensions between [0.3, 25] is valid
            batch_dim = torch.exp(preds_dict['dim'].clamp(min=-1.2, max=3.2))
            
            if 'iou' in preds_dict.keys():
                batch_iou = (preds_dict['iou'].squeeze(dim=-1) + 1) * 0.5
                batch_iou = batch_iou.type_as(batch_dim)
                batch_iou = torch.clamp(batch_iou, min=0, max=1.)
            else:
                batch_iou = torch.ones((batch_hm.shape[0], batch_hm.shape[1], batch_hm.shape[2]),
                                        dtype=batch_dim.dtype).to(batch_hm.device)

            batch_rots = preds_dict['rot'][..., 0:1]
            batch_rotc = preds_dict['rot'][..., 1:2]
            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']

            if double_flip:
                batch_hm = batch_hm.mean(dim=1)
                batch_iou = batch_iou.mean(dim=1)
                batch_hei = batch_hei.mean(dim=1)
                batch_dim = batch_dim.mean(dim=1)

                # y = -y reg_y = 1-reg_y
                batch_reg[:, 1, ..., 1] = 1 - batch_reg[:, 1, ..., 1]
                batch_reg[:, 2, ..., 0] = 1 - batch_reg[:, 2, ..., 0]

                batch_reg[:, 3, ..., 0] = 1 - batch_reg[:, 3, ..., 0]
                batch_reg[:, 3, ..., 1] = 1 - batch_reg[:, 3, ..., 1]
                batch_reg = batch_reg.mean(dim=1)

                # first yflip 
                # y = -y theta = pi -theta
                # sin(pi-theta) = sin(theta) cos(pi-theta) = -cos(theta)
                # batch_rots[:, 1] the same
                batch_rotc[:, 1] *= -1

                # then xflip x = -x theta = 2pi - theta
                # sin(2pi - theta) = -sin(theta) cos(2pi - theta) = cos(theta)
                # batch_rots[:, 2] the same
                batch_rots[:, 2] *= -1

                # double flip 
                batch_rots[:, 3] *= -1
                batch_rotc[:, 3] *= -1

                batch_rotc = batch_rotc.mean(dim=1)
                batch_rots = batch_rots.mean(dim=1)

            batch_rot = torch.atan2(batch_rots, batch_rotc)
            batch, H, W, num_cls = batch_hm.size()

            ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
            ys = ys.view(1, H, W).repeat(batch, 1, 1).to(batch_hm)
            xs = xs.view(1, H, W).repeat(batch, 1, 1).to(batch_hm)
            xs = xs.view(batch, H, W, 1) + batch_reg[:, :, :, 0:1]
            ys = ys.view(batch, H, W, 1) + batch_reg[:, :, :, 1:2]
            xs = xs * self.task_strides[task_id] * self.pillar_size + self.point_cloud_range[0]
            ys = ys * self.task_strides[task_id] * self.pillar_size + self.point_cloud_range[1]

            if 'vel' in preds_dict:
                batch_vel = preds_dict['vel']
                if double_flip:
                    batch_vel[:, 1, ..., 1] *= -1  # flip vy
                    batch_vel[:, 2, ..., 0] *= -1  # flip vx
                    batch_vel[:, 3] *= -1
                    batch_vel = batch_vel.mean(dim=1)
                batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_vel, batch_rot], dim=-1)
            else: 
                batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_rot], dim=-1)

            metas.append(meta_list)
            rets.append(self.post_processing(task_id, batch_box_preds, batch_hm, batch_iou, test_cfg))

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ["box3d_lidar", "scores"]:
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k in ["label_preds"]:
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    ret[k] = torch.cat([ret[i][k] for ret in rets])

            ret['metadata'] = metas[0][i]
            ret_list.append(ret)

        return ret_list 

    @torch.no_grad()
    def post_processing(self, task_id, batch_box_preds, batch_hm, batch_iou, test_cfg):
        batch_size = len(batch_hm)
        
        post_center_range = test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range, dtype=batch_hm.dtype, device=batch_hm.device)

        prediction_dicts = []
        for i in range(batch_size):
            box_preds = batch_box_preds[i]
            hm_preds = batch_hm[i]
            ious = batch_iou[i]
            scores, labels = torch.max(hm_preds, dim=-1)

            score_mask = scores > test_cfg.score_threshold
            distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(-1) \
                & (box_preds[..., :3] <= post_center_range[3:]).all(-1)

            mask = distance_mask & score_mask
            box_preds = box_preds[mask]
            scores = scores[mask]
            labels = labels[mask]
            ious = ious[mask]
            
            if test_cfg.get('circular_nms', False):
                centers = box_preds[:, [0, 1]] 
                boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                selected = _circle_nms(boxes, min_radius=test_cfg.min_radius[task_id],
                                       post_max_size=test_cfg.nms.nms_post_max_size[task_id])

                selected_boxes = box_preds[selected]
                selected_scores = scores[selected]
                selected_labels = labels[selected]
            elif test_cfg.nms.get('use_rotate_nms', False):
                selected_boxes, selected_scores, selected_labels = box_torch_ops.rotate_nms_pcdet(
                    box_preds, scores, ious, labels,
                    rectifier=test_cfg.rectifier,
                    nms_thresh=test_cfg.nms.nms_iou_threshold,
                    pre_maxsize=test_cfg.nms.nms_pre_max_size,
                    post_max_size=test_cfg.nms.nms_post_max_size)
            elif test_cfg.nms.get('use_multi_class_nms', False):
                selected_boxes, selected_scores, selected_labels = box_torch_ops.rotate_class_nms_pcdet(
                    box_preds, scores, ious, labels,
                    rectifiers=test_cfg.rectifier[task_id],
                    use_rectify=test_cfg.use_rectify[task_id] if test_cfg.get('use_rectify') else False,
                    nms_thresh=test_cfg.nms.nms_iou_threshold[task_id],
                    pre_maxsize=test_cfg.nms.nms_pre_max_size[task_id],
                    post_max_size=test_cfg.nms.nms_post_max_size[task_id])
            else:
                raise NotImplementedError

            prediction_dict = {
                'box3d_lidar': selected_boxes,
                'scores': selected_scores,
                'label_preds': selected_labels
            }

            prediction_dicts.append(prediction_dict)

        return prediction_dicts
    


import numpy as np 
def _circle_nms(boxes, min_radius, post_max_size=83):
    """
    NMS according to center distance
    """
    keep = np.array(circle_nms(boxes.cpu().numpy(), thresh=min_radius))[:post_max_size]

    keep = torch.from_numpy(keep).long().to(boxes.device)

    return keep  
