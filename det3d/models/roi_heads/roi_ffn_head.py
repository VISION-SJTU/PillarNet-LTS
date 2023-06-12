import torch
import torch.nn as nn
from .roi_head_template import RoIHeadTemplate
from ..registry import ROI_HEAD
from ..utils.norm import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN


@ROI_HEAD.register_module
class RoIFFNHead(RoIHeadTemplate):
    def __init__(self, model_cfg, 
                num_cls_fcs=1,
                num_reg_fcs=1,
                num_iou_fcs=1,
                feedforward_channels=2048,
                content_dim=256,
                num_ffn_fcs=2,
                dropout=0.0,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                num_class=1, 
                code_size=7, 
                test_cfg=None,
                init_bias=-2.19,
                add_box_param=False):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.test_cfg = test_cfg 
        self.code_size = code_size
        self.add_box_param = add_box_param
        
        self.init_bias = init_bias
        self.use_iou = num_iou_fcs > 0
        
        self.ffn = FFN(
            content_dim,
            feedforward_channels,
            num_ffn_fcs,
            act_cfg=ffn_act_cfg,
            dropout=dropout)
        self.ffn_norm = build_norm_layer(dict(type='LN'), content_dim)[1]
        
        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(nn.Linear(content_dim, content_dim, bias=True))
            self.cls_fcs.append(build_norm_layer(dict(type='LN'), content_dim)[1])
            self.cls_fcs.append(nn.ReLU())
        self.fc_cls = nn.Linear(content_dim, self.num_class)
        
        self.reg_fcs = nn.ModuleList()
        for _ in range(num_reg_fcs):
            self.reg_fcs.append(nn.Linear(content_dim, content_dim, bias=True))
            self.reg_fcs.append(build_norm_layer(dict(type='LN'), content_dim)[1])
            self.reg_fcs.append(nn.ReLU())
        self.fc_reg = nn.Linear(content_dim, code_size)
        
        if self.use_iou > 0:
            self.iou_fcs = nn.ModuleList()
            for _ in range(num_iou_fcs):
                self.iou_fcs.append(nn.Linear(content_dim, content_dim, bias=True))
                self.iou_fcs.append(build_norm_layer(dict(type='LN'), content_dim)[1])
                self.iou_fcs.append(nn.ReLU())
            self.fc_iou = nn.Linear(content_dim, 1)
        
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
                nn.init.xavier_uniform_(m.weight)
                
        nn.init.zeros_(self.fc_reg.weight)
        nn.init.zeros_(self.fc_reg.bias)
        nn.init.constant_(self.fc_cls.bias, self.init_bias)
        
        if self.use_iou > 0:
            nn.init.zeros_(self.fc_iou.weight)
            nn.init.zeros_(self.fc_iou.bias)

    def forward(self, batch_dict, training=True):

        if training:
            targets_dict = batch_dict.get('roi_targets_dict', None)
            if targets_dict is None:
                targets_dict = self.assign_targets(batch_dict)
                batch_dict['rois'] = targets_dict['rois']
                batch_dict['roi_labels'] = targets_dict['roi_labels']
                batch_dict['roi_scores'] = targets_dict['roi_scores']

        # RoI aware pooling
        if self.add_box_param:
            batch_dict['roi_features'] = torch.cat([batch_dict['roi_features'], 
                batch_dict['rois'], batch_dict['roi_scores'].unsqueeze(-1)], dim=-1)

        batch_dict['roi_features'] = batch_dict['roi_features'].view(
                                            -1, batch_dict['roi_features'].shape[-1])
        
        # FFN
        query_content = self.ffn_norm(self.ffn(batch_dict['roi_features']))

        cls_feat = query_content
        reg_feat = query_content

        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat)

        rcnn_cls = self.fc_cls(cls_feat)
        rcnn_reg = self.fc_reg(reg_feat)
        
        if self.use_iou:
            iou_feat = query_content
            for iou_layer in self.iou_fcs:
                iou_feat = iou_layer(iou_feat)
            rcnn_iou = self.fc_iou(iou_feat)

        if not training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            
            if self.use_iou > 0:
                targets_dict['rcnn_iou'] = rcnn_iou

                batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                    batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg)
                
                targets_dict['batch_box_preds'] = batch_box_preds

            self.forward_ret_dict = targets_dict
        
        return batch_dict
