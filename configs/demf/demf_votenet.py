_base_ = [
    '../_base_/datasets/sunrgbd-3d-10class.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py',
    '../deformdetr/imvotenet_image.py'
]

# may also use your own pre-trained image branch
load_from = '/path/to/pre-trained/image/branch'

class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

lr = 0.008  # max learning rate
optimizer = dict(
    type='AdamW', lr=lr, weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'decoder': dict(lr_mult=0.05, decay_mult=1.0),
        },
    ),
)

model = dict(
    type='DeMFVoteNet',
    img_encoder=dict(
        type='DeformableDetrEncoder',
        encoder=dict(
            type='DetrTransformerEncoder',
            num_layers=6,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiScaleDeformableAttention', embed_dims=256),
                feedforward_channels=1024,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        num_feature_levels=4,
        embed_dims=256,
    ),
    pts_backbone=dict(
        type='PointNet2SASSG',
        in_channels=4,
        num_points=(2048, 1024, 512, 256),
        radius=(0.2, 0.4, 0.8, 1.2),
        num_samples=(64, 32, 16, 16),
        sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                     (128, 128, 256)),
        fp_channels=((256, 256), (256, 256)),
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='PointSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=True)),
    pts_bbox_head=dict(
        type='DeMFVoteHead',
        pred_layer_cfg=dict(
            in_channels=256, shared_conv_channels=(128, 128), bias=True,
            conv_pred_layers=2),
        decoder=dict(
            type='DeMFTransformerDecoderLayer',
            num_layers=1,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.4),
                    dict(
                        type='MultiScaleDeformableAttention',
                        num_heads=8,
                        num_levels=4,
                        num_points=2,
                        dropout=0.4,
                        embed_dims=256)
                ],
                feedforward_channels=1024,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                    'ffn', 'norm'),
            ),
            posembed=dict(
                input_channel=6,
                num_pos_feats=256,
            ),
        ),
        num_classes=10,
        bbox_coder=dict(
            type="ClassAgnosticBBoxCoderGFReg",
            num_dir_bins=12,
            with_rot=True,
            num_sizes=10,
            mean_sizes=[[2.114256, 1.620300, 0.927272],
                        [0.791118, 1.279516, 0.718182],
                        [0.923508, 1.867419, 0.845495],
                        [0.591958, 0.552978, 0.827272],
                        [0.699104, 0.454178, 0.75625],
                        [0.69519, 1.346299, 0.736364],
                        [0.528526, 1.002642, 1.172878],
                        [0.500618, 0.632163, 0.683424],
                        [0.404671, 1.071108, 1.688889],
                        [0.76584, 1.398258, 0.472728]]
        ),
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        objectness_loss=dict(
            type='CrossEntropyLoss',
            class_weight=[0.2, 0.8],
            reduction='sum',
            loss_weight=5.0),
        dir_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        dir_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=10.0),
        size_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        size_res_loss=dict(
            type='SmoothL1Loss', 
            reduction='sum', 
            loss_weight=10.0,
            beta=0.0625,
        ),
        center_loss=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, reduction='sum', loss_weight=10.0),
        iou_loss=dict(
            type='AxisAlignedIoULoss', 
            reduction='sum', 
            loss_weight=12.0 / 3.0,
        ),
        semantic_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        vote_module_cfg=dict(
            in_channels=256,
            vote_per_seed=1,
            gt_per_seed=3,
            conv_channels=(256, 256),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            norm_feats=True,
            vote_loss=dict(
                type='ChamferDistance',
                mode='l1',
                reduction='none',
                loss_dst_weight=10.0)),
        vote_aggregation_cfg=dict(
            type='PointSAModule',
            num_point=256,
            radius=0.3,
            num_sample=16,
            mlp_channels=[256, 256, 256, 256],
            use_xyz=True,
            normalize_xyz=True),
    ),
    num_sampled_seed=1024,
    freeze_img_branch=True,

    train_cfg=dict(
        pts=dict(
            pos_distance_thr=0.3, 
            neg_distance_thr=0.6, 
            sample_mod='seed',
        )
    ),
    test_cfg=dict(
        img_rcnn=dict(score_thr=0.1),
        pts=dict(
            ensemble_layers=[0, 1],
            sample_mod='seed',
            nms_thr=0.25,
            score_thr=0.05,
            per_class_proposal=True))
)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations3D'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
    ),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.523599, 0.523599],
        scale_ratio_range=[0.85, 1.15],
        shift_height=True),
    dict(type='PointSample', num_points=20000),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'points', 'gt_bboxes_3d',
            'gt_labels_3d'
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
            ),
            dict(type='PointSample', num_points=20000),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img', 'points'])
        ]),
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['img', 'points'])
]

data = dict(
    train=dict(dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
evaluation = dict(
    pipeline=eval_pipeline,
    interval=36,
)
find_unused_parameters = True
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
