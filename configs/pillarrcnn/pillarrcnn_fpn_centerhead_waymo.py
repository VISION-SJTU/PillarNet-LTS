import itertools
import logging

tasks = [
    dict(stride=8, class_names=['VEHICLE']),
    dict(stride=4, class_names=['PEDESTRIAN', 'CYCLIST']),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

pillar_size = 0.1
point_cloud_range = [-75.2, -75.2, -2, 75.2, 75.2, 4]
post_center_limit_range=[-80, -80, -10.0, 80, 80, 10.0]


# model settings
model = dict(
    type='PillarRCNN',
    freeze=False,
    first_stage_cfg=dict(
        type="PillarNet",
        reader=dict(
            type="DynamicPFE",
            in_channels=5,
            num_filters=(32, ),
            pillar_size=pillar_size,
            pc_range=point_cloud_range,
        ),
        backbone=dict(type="PillarResNet18", in_channels=32),
        neck=dict(
            type="RPNG",
            layer_nums=[5, 5],
            num_filters=[256, 128],
            in_channels=[256, 256, 128],
            logger=logging.getLogger("RPN"),
        ),
        bbox_head=dict(
            type="CenterHead",
            tasks=tasks,
            in_channels=[256, 128], # stride: channels
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2)}, # (output_channel, num_conv)
            reg_iou='GIoU',
            pillar_size=pillar_size,
            point_cloud_range=point_cloud_range
        )
    ),
    second_stage_modules=[
        dict(
            type="BEVStrideFeature",
            feature_sources=['conv3'],
            grid_size=7,
            out_stride=4,
            in_channels=128,
            share_channels=64,
            pillar_size=pillar_size,
            pc_range=point_cloud_range,
        )
    ],
    point_head=dict(
        type="PointHead",
        in_channels=64,
        num_class=1,
        model_cfg=dict(
            CLASS_AGNOSTIC=True,
            CLS_FC=[256, 256],
            TARGET_CONFIG=dict(
                GT_EXTRA_WIDTH=[0.2, 0.2, 0.2]
            ),
            LOSS_CONFIG=dict(
                LOSS_REG='smooth-l1',
                LOSS_WEIGHTS={'point_cls_weight': 1.0}
            )
        ),
    ),
    roi_head=dict(
        type="RoIMIXHead",
        in_channels=64,
        mixer_type='',  # ResMLP   MLPMixer
        num_patches=7 * 7,
        model_cfg=dict(
            CLASS_AGNOSTIC=True,
            SHARED_FC=[256, 256],
            CLS_FC=[256, 256],
            REG_FC=[256, 256],
            DP_RATIO=0.3,

            TARGET_CONFIG=dict(
                ROI_PER_IMAGE=128,
                FG_RATIO=0.5,
                SAMPLE_ROI_BY_EACH_CLASS=True,
                CLS_SCORE_TYPE='roi_iou',
                CLS_FG_THRESH=0.7,
                CLS_BG_THRESH=0.25,
                CLS_BG_THRESH_LO=0.1,
                HARD_BG_RATIO=0.8,
                REG_FG_THRESH=0.5
            ),
            LOSS_CONFIG=dict(
                CLS_LOSS='BinaryCrossEntropy',
                REG_LOSS='L1',
                LOSS_WEIGHTS={
                    'rcnn_cls_weight': 1.0,
                    'rcnn_reg_weight': 1.0,
                    'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                }
            ),
        ),
        code_size=7
    )
)


train_cfg = dict(
    assigner=dict(
        target_assigner=dict(tasks=tasks),
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        pc_range=point_cloud_range,
        pillar_size=pillar_size),
    hm_weight=1,
    bbox_weight=2,
    iou_weight=1,
    reg_iou_weight=2
)


test_cfg = dict(
    nms=dict(
        use_multi_class_nms=True,
        nms_pre_max_size=[2048, 1024, 1024],
        nms_post_max_size=[200, 150, 150],
        nms_iou_threshold=[0.8, 0.55, 0.55],
    ),
    rectifier=[0., 0., 0.],
    score_threshold=0.1,
    post_center_limit_range=post_center_limit_range,
)


# dataset settings
dataset_type = "WaymoDataset"
nsweeps = 1
data_root = "data/Waymo/"

db_sampler = dict(
    type="GT-AUG",
    enable=False,
    rate=1.0,
    global_random_rotation_range_per_object=[0, 0],
    db_info_path=data_root + "dbinfos_train_1sweeps_withvelo.pkl",
    sample_groups=[
        dict(VEHICLE=15),
        dict(PEDESTRIAN=10),
        dict(CYCLIST=10),
    ],
    db_prep_steps=[
        dict(
            filter_by_min_num_points=dict(
                VEHICLE=5,
                PEDESTRIAN=5,
                CYCLIST=5)),
        dict(filter_by_difficulty=[-1],),
    ],
)

train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    global_rot_noise=[-0.78539816, 0.78539816],
    global_scale_noise=[0.95, 1.05],
    db_sampler=db_sampler,
    class_names=class_names,
)
val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Reformat"),
]

train_anno = data_root + "infos_train_01sweeps_filter_zero_gt.pkl"
val_anno = data_root + "infos_val_01sweeps_filter_zero_gt.pkl"
test_anno = data_root + "infos_test_01sweeps_filter_zero_gt.pkl"

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
        load_interval=1,
        use_cbgs=True,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        ann_file=val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        ann_file=test_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.01, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)

log_config = dict(
    interval=5,
    hooks=[dict(type="TextLoggerHook")],
)

# runtime settings
total_epochs = 36
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None 
resume_from = None  
workflow = [('train', 1)]
