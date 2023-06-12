import itertools
import logging

from det3d.utils.config_tool import get_downsample_factor
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

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

# model settings
model = dict(
    type="PillarNet",
    pretrained=None,
    reader=dict(type="Identity", pc_range=[-54, -54, -5.0, 54, 54, 3.0], num_input_features=2),
    backbone=dict(
        type="SpMiddlePillarEncoder18", ds_factor=8, double=2,
        pc_range=[-54, -54, -5.0, 54, 54, 3.0],
        pillar_cfg=dict(
            pool0=dict(bev=0.075 / 2),
            pool1=dict(bev=0.075),
            pool2=dict(bev=0.075 * 2),
            pool3=dict(bev=0.075 * 4),
            pool4=dict(bev=0.075 * 8),
        ),
    ),
    neck=dict(
        type="RPNV2",
        layer_nums=[5, 5],
        ds_layer_strides=[1, 2],
        ds_num_filters=[256, 256],
        us_layer_strides=[1, 2],
        us_num_filters=[128, 128],
        num_input_features=[128, 256],   # [256, 256]
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="CenterHead",
        in_channels=256,
        tasks=tasks,
        dataset='nuscenes',
        weight=0.25,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim': (3, 2), 'rot': (2, 2), 'vel': (2, 2)},
        share_conv_channel=64,
        dcn_head=False,
    ),
)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    pc_range=[-54, -54, -5.0, 54, 54, 3.0],
    voxel_size=[0.075, 0.075, 0.2],
)


train_cfg = dict(assigner=assigner)

test_cfg = dict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=83,
        nms_iou_threshold=0.2,
    ),
    score_threshold=0.1,
    pc_range=[-54, -54],
    rectifier=0,
    out_size_factor=get_downsample_factor(model),
    voxel_size=[0.075, 0.075],
    double_flip=DOUBLE_FLIP
)

# dataset settings
dataset_type = "NuScenesDataset"
nsweeps = 10
data_root = "data/nuScenes"

db_sampler = dict(
    type="GT-AUG",
    enable=False,
    db_info_path="data/nuScenes/dbinfos_train_10sweeps_withvelo.pkl",
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
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
    extra_scale=1.,
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

voxel_generator = dict(
    range=[-54, -54, -5.0, 54, 54, 3.0],
    voxel_size=[0.075, 0.075, 0.2],
    max_points_in_voxel=10,
    max_voxel_num=[120000, 160000],
    double_flip=DOUBLE_FLIP,
    skip=True,
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
    # dict(type='PointCloudCollect', keys=['points', 'voxels', 'annotations', 'calib']),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="DoubleFlip") if DOUBLE_FLIP else dict(type="Empty"),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat", double_flip=DOUBLE_FLIP),
]

train_anno = "data/nuScenes/infos_train_10sweeps_withvelo_filter_True.pkl"
val_anno = "data/nuScenes/infos_val_10sweeps_withvelo_filter_True.pkl"
test_anno = "data/nuScenes/infos_test_10sweeps_withvelo_filter_True.pkl"

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
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
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 20
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None 
workflow = [('train', 1)]
