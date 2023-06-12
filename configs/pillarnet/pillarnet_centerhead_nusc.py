import itertools
import logging

DOUBLE_FLIP = False

tasks = [
    dict(stride=8, class_names=["car"]),
    dict(stride=8, class_names=["truck", "construction_vehicle"]),
    dict(stride=8, class_names=["bus", "trailer"]),
    dict(stride=8, class_names=["barrier"]),
    dict(stride=8, class_names=["motorcycle", "bicycle"]),
    dict(stride=8, class_names=["pedestrian", "traffic_cone"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

pillar_size = 0.075
point_cloud_range = [-54, -54, -5.0, 54, 54, 3.0]
post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]

# model settings
model = dict(
    type="PillarNet",
    pretrained=None,
    reader=dict(
        type="DynamicPFE",
        in_channels=5,
        num_filters=(32, ),
        pillar_size=pillar_size,
        pc_range=point_cloud_range,
    ),
    backbone=dict(type="PillarResNet18", in_channels=32),
    neck=dict(
        type="RPNV1",
        layer_nums=[5, 5],
        num_filters=256,
        in_channels=[256, 256],
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="CenterHead",
        tasks=tasks,
        in_channels=[256], # stride: channels
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2), 'vel': (2, 2)}, # (output_channel, num_conv)
        reg_iou='GIoU',
        pillar_size=pillar_size,
        point_cloud_range=point_cloud_range
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
    bbox_weight=0.25,
    iou_weight=1,
    reg_iou_weight=0.25,
)

test_cfg = dict(
    nms=dict(
        use_rotate_nms=True,
        nms_pre_max_size=1000,
        nms_post_max_size=83,
        nms_iou_threshold=0.2,
    ),
    # rectifier=[0.68, 0.71, 0.65],
    # rectifier=[0, 0, 0],
    rectifier=0,
    score_threshold=0.1,
    double_flip=DOUBLE_FLIP,
    post_center_limit_range=post_center_limit_range,
)


# dataset settings
dataset_type = "NuScenesDataset"
nsweeps = 10
data_root = "data/nuScenes/"

db_sampler = dict(
    type="GT-AUG",
    enable=False,
    rate=1.0,
    global_random_rotation_range_per_object=[0, 0],
    db_info_path=data_root + "dbinfos_train_10sweeps_withvelo.pkl",
    sample_groups=[
        dict(car=2),
        dict(truck=3),
        dict(construction_vehicle=7),
        dict(bus=4),
        dict(trailer=6),
        dict(barrier=2),
        dict(motorcycle=6),
        dict(bicycle=6),
        dict(pedestrian=2),
        dict(traffic_cone=2),
    ],
    db_prep_steps=[
        dict(
            filter_by_min_num_points=dict(
                car=5,
                truck=5,
                bus=5,
                trailer=5,
                construction_vehicle=5,
                traffic_cone=5,
                barrier=5,
                motorcycle=5,
                bicycle=5,
                pedestrian=5,
            )
        ),
        dict(filter_by_difficulty=[-1],),
    ],
)

train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    global_rot_noise=[-0.78539816, 0.78539816],
    global_scale_noise=[0.9, 1.1],
    global_translate_std=0.5,
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
    dict(type="DoubleFlip") if DOUBLE_FLIP else dict(type="Empty"),
    dict(type="Reformat", double_flip=DOUBLE_FLIP),
]

train_anno = data_root + "infos_train_10sweeps_withvelo_filter_True.pkl"
val_anno = data_root + "infos_val_10sweeps_withvelo_filter_True.pkl"
test_anno = data_root + "infos_test_10sweeps_withvelo_filter_True.pkl"

data = dict(
    samples_per_gpu=4,
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
        test_mode=False,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
        version='v1.0-test'
    ),
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.001, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)

log_config = dict(
    interval=5,
    hooks=[dict(type="TextLoggerHook")],
)

# runtime settings
total_epochs = 20
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None 
workflow = [('train', 1)]