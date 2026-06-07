_base_ = [
    '../_base_/default_runtime.py'
]

data_root = '/root/autodl-tmp/dataset/coco/'
log_interval = 100



# dataset settings
dataset_type = 'OVCocoDataset'

image_size = (1024, 1024)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals', num_max_proposals=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=image_size, keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'img_no_normalize', 'proposals', 'gt_bboxes', 'gt_labels'])
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=image_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img', 'img_no_normalize']),
            dict(type='Collect', keys=['img', 'img_no_normalize'])
        ])
]


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='OVCocoDataset',
        ann_file=data_root + 'annotations/instances_train2017.48.json',
        img_prefix=data_root + 'train2017/',
        proposal_file='/root/mmdetection/ovd_resources/coco_proposal_train_object_centric.pkl',
        pipeline=train_pipeline),
    val=dict(
        type='OVCocoDataset',
        ann_file=data_root + 'annotations/instances_val2017.65.min.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type='OVCocoDataset',
        ann_file=data_root + 'annotations/instances_val2017.65.min.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
)




num_stages = 6
num_query = 300
QUERY_DIM = 256
FEAT_DIM = 256
FF_DIM = 2048
TEXT_DIM = 512


model = dict(
    type='DmzDETR',
    embedding_file='/root/mmdetection/ovd_resources/coco_detpro_category_embeddings_vit-b-32.pt',
    dataset='ov_coco',
    num_use_pseudo_box_epoch=8,
    base_ind_file=None,
    add_pseudo_box_to_rpn=True,
    

    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        # norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='DmzDETRHead',
        num_query=300,
        num_classes=65,
        in_channels=2048,
        sync_cls_avg_factor=True,
        as_two_stage=True,
        
        num_stages=num_stages,
        stage_loss_weights=[1] * 6,
        content_dim=256,
        text_dim=512,
        clip_model_path='/root/mmdetection/ovd_resources/CLIP_ViT-B-32.pt',
        max_pseudo_box_num=5,
        cls_tau=50,
        skd_tau=20,
        rkd_tau=5,
        pre_extracted_clip_text_feat='/root/mmdetection/ovd_resources/coco_proposals_text_embedding10/',
        use_pseudo_box=True,
        split_visual_text=True,
        use_text_space_rkd_loss=True,
        loss_visual_skd=dict(type='CrossEntropyLoss', loss_weight=0.5),
        loss_visual_rkd=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=5.0),
        use_image_level_distill=True,
        num_additional_padding_prompts=32,
        novel_obj_queue_dict=dict(
            names=['novel_obj', 'obj_query_0', 'obj_query_1', 'obj_query_2', 'obj_query_3', 'obj_query_4', 'obj_query_5', 'clip_query_0', 'clip_query_1', 'clip_query_2', 'clip_query_3', 'clip_query_4', 'clip_query_5', 'encoder_image_query', 'clip_image_query'],
            lengths=[2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 512, 512], 
            emb_dim=TEXT_DIM),
        
        
        transformer=dict(
            type='DmzDetrTransformer',
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
            decoder=dict(
                type='DmzDetrTransformerDecoder',
                num_layers=6,   
                return_intermediate=True,
                
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    num_cls_fcs=1,
                    text_dims=512,
                    cls_predictor_cfg=dict(type='Linear'),
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))
            )),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=300))


# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/instances_train2017.48.json',
#         img_prefix=data_root + 'train2017/',
#         proposal_file='/root/mmdetection/ovd_resources/coco_proposal_train_object_centric.pkl',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/instances_val2017.65.min.json',
#         img_prefix=data_root + 'val2017/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/instances_val2017.65.min.json',
#         img_prefix=data_root + 'val2017/',
#         pipeline=test_pipeline),
# )

# optimizer
optimizer = dict(
    type='AdamW',
    lr=2.5e-5,
    weight_decay=0.005,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        },
        norm_decay_mult=0.,
        bypass_duplicate=True
    ))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.001,
    step=[37]
)


# custom hooks
custom_hooks = [dict(type='SetEpochInfoHook')]

runner = dict(type='EpochBasedRunner', max_epochs=47)
evaluation = dict(metric=['bbox'], interval=1)
checkpoint_config = dict(interval=1, create_symlink=False, max_keep_ckpts=20)

dist_params = dict(backend='nccl')
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (2 samples per GPU)
# auto_scale_lr = dict(base_batch_size=16)


# log_config = dict(
#     interval=log_interval,
#     hooks=[
#         dict(type='TextLoggerHook'),
#         # dict(type='TensorboardLoggerHook')
#     ]
# )

# def __date():
#     import datetime
#     return datetime.datetime.now().strftime('%m%d_%H%M')

# postfix = '_' + __date()

# find_unused_parameters = True

# Enable FP16 training
# fp16 = dict(loss_scale='dynamic')