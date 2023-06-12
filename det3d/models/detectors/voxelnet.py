from ..registry import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module
class VoxelNet(SingleStageDetector):
    def __init__(self,
                 reader,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(VoxelNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        
    def extract_feat(self, data):
        sp_tensor = self.reader(data)
        x, voxel_feature = self.backbone(sp_tensor)

        if self.with_neck:
            x = self.neck(x)

        return x, voxel_feature

    def forward(self, example, return_loss=True, **kwargs):
        x, _ = self.extract_feat(example)
        preds = self.bbox_head(x)

        if return_loss:
            return self.bbox_head.loss(example, preds, self.train_cfg)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        x, voxel_feature = self.extract_feat(example)
        bev_feature = x 
        preds, final_feat = self.bbox_head(x)

        if return_loss:
            # manual deepcopy ...
            new_preds = []
            for pred in preds:
                new_pred = {} 
                for k, v in pred.items():
                    new_pred[k] = v.detach()
                new_preds.append(new_pred)

            boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

            return boxes, bev_feature, voxel_feature, final_feat, \
                    self.bbox_head.loss(example, preds, self.train_cfg)
        else:
            boxes = self.bbox_head.predict(example, preds, self.test_cfg)
            
            return boxes, bev_feature, voxel_feature, final_feat, None 