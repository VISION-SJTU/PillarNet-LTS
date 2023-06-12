from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from det3d.core.utils.center_utils import set_by_task_cfg


@DETECTORS.register_module
class PillarNet(SingleStageDetector):
    def __init__(self,
                 reader,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        if isinstance(self.test_cfg.nms.nms_post_max_size, list):
            self.NMS_POST_MAXSIZE = sum(self.test_cfg.nms.nms_post_max_size)
        else:
            self.NMS_POST_MAXSIZE = self.test_cfg.nms.nms_post_max_size

        self.num_classes = self.bbox_head.num_classes
        self.test_cfg = set_by_task_cfg(self.test_cfg, self.bbox_head.num_classes)
        
    def extract_feat(self, data):
        sp_tensor = self.reader(data)
        pillar_features = self.backbone(sp_tensor)
        if self.with_neck:
            bev_features = self.neck(pillar_features)

        return bev_features, pillar_features

    def forward(self, example, return_loss=True, **kwargs):
        batch_size = len(example['metadata'])

        data = dict(
            points=example["points"],
            batch_size=batch_size,
        )

        bev_features, _ = self.extract_feat(data)
        preds = self.bbox_head(bev_features)

        if return_loss:
            return self.bbox_head.loss(example, preds, self.train_cfg)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        batch_size = len(example['metadata'])

        data = dict(
            points=example["points"],
            batch_size=batch_size,
        )
        
        # from tools.visual import draw_scenes
        # points = example["points"][0].cpu().numpy()
        # gt_boxes = example["gt_boxes_and_cls"][0][:10, :7].cpu().numpy()
        # gt_boxes2 = example["gt_box"][0][0, :10].cpu().numpy()
        # draw_scenes(points, gt_boxes=gt_boxes)
        
        bev_features, backbone_features = self.extract_feat(data)
        preds = self.bbox_head(bev_features)

        # manual deepcopy ...
        new_preds = []
        for pred in preds:
            new_pred = {} 
            for k, v in pred.items():
                new_pred[k] = v.detach()
            new_preds.append(new_pred)

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        if return_loss:
            return boxes, bev_features, backbone_features, \
                    self.bbox_head.loss(example, preds, self.train_cfg)
        else:
            return boxes, bev_features, backbone_features, None
