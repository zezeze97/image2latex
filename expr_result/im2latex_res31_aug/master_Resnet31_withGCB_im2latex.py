checkpoint_config = dict(interval=10)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'expr_result/im2latex_res31_aug/epoch_5.pth'
resume_from = None
workflow = [('train', 1)]
alphabet_file = 'data/im2latex_data/master_data/keys.txt'
alphabet_len = 533
max_seq_len = 200
start_end_same = False
label_convertor = dict(
    type='MasterConvertor',
    dict_file='data/im2latex_data/master_data/keys.txt',
    max_seq_len=200,
    start_end_same=False,
    with_unknown=True)
PAD = 536
model = dict(
    type='MASTER',
    backbone=dict(
        type='ResNetExtra',
        input_dim=3,
        gcb_config=dict(
            ratio=0.0625,
            headers=1,
            att_scale=False,
            fusion_type='channel_add',
            layers=[False, True, True, True]),
        layers=[1, 2, 5, 3]),
    encoder=dict(
        type='PositionalEncoding', d_model=512, dropout=0.2, max_len=5000),
    decoder=dict(
        type='MasterDecoder',
        N=3,
        decoder=dict(
            self_attn=dict(headers=8, d_model=512, dropout=0.0),
            src_attn=dict(headers=8, d_model=512, dropout=0.0),
            feed_forward=dict(d_model=512, d_ff=2024, dropout=0.0),
            size=512,
            dropout=0.0),
        d_model=512),
    loss=dict(type='MASTERTFLoss', ignore_index=536, reduction='mean'),
    label_convertor=dict(
        type='MasterConvertor',
        dict_file='data/im2latex_data/master_data/keys.txt',
        max_seq_len=200,
        start_end_same=False,
        with_unknown=True),
    max_seq_len=200)
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CaptionAug', args=[dict(cls='Affine', rotate=[-10, 10])]),
    dict(
        type='ResizeOCR',
        height=128,
        min_width=128,
        max_width=512,
        keep_aspect_ratio=True),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'text', 'valid_ratio'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=128,
        min_width=128,
        max_width=512,
        keep_aspect_ratio=True),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'valid_ratio',
            'img_norm_cfg', 'ori_filename'
        ])
]
dataset_type = 'OCRDataset'
img_prefix1 = 'data/im2latex_data/formula_images_processed/'
train_anno_file1 = 'data/im2latex_data/master_data/train.txt'
train1 = dict(
    type='OCRDataset',
    img_prefix='data/im2latex_data/formula_images_processed/',
    ann_file='data/im2latex_data/master_data/train.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='Im2TokenTextLineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator='	')),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='CaptionAug', args=[dict(cls='Affine', rotate=[-10, 10])]),
        dict(
            type='ResizeOCR',
            height=128,
            min_width=128,
            max_width=512,
            keep_aspect_ratio=True),
        dict(type='ToTensorOCR'),
        dict(type='NormalizeOCR', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        dict(
            type='Collect',
            keys=['img'],
            meta_keys=[
                'filename', 'ori_shape', 'img_shape', 'text', 'valid_ratio'
            ])
    ],
    test_mode=False)
test_img_prefix = 'data/im2latex_data/formula_images_processed/'
test_ann_files = dict(
    table_Rec_val_debug_0='data/im2latex_data/master_data/test.txt')
testset = [
    dict(
        type='OCRDataset',
        img_prefix='data/im2latex_data/formula_images_processed/',
        ann_file='data/im2latex_data/master_data/test.txt',
        loader=dict(
            type='HardDiskLoader',
            repeat=1,
            parser=dict(
                type='Im2TokenTextLineStrParser',
                keys=['filename', 'text'],
                keys_idx=[0, 1],
                separator='	')),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='ResizeOCR',
                height=128,
                min_width=128,
                max_width=512,
                keep_aspect_ratio=True),
            dict(type='ToTensorOCR'),
            dict(
                type='NormalizeOCR', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5,
                                                                0.5]),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'valid_ratio',
                    'img_norm_cfg', 'ori_filename'
                ])
        ],
        dataset_info='table_Rec_val_debug_0',
        test_mode=True)
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8,
    train=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix='data/im2latex_data/formula_images_processed/',
                ann_file='data/im2latex_data/master_data/train.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='Im2TokenTextLineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator='	')),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(
                        type='CaptionAug',
                        args=[dict(cls='Affine', rotate=[-10, 10])]),
                    dict(
                        type='ResizeOCR',
                        height=128,
                        min_width=128,
                        max_width=512,
                        keep_aspect_ratio=True),
                    dict(type='ToTensorOCR'),
                    dict(
                        type='NormalizeOCR',
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=[
                            'filename', 'ori_shape', 'img_shape', 'text',
                            'valid_ratio'
                        ])
                ],
                test_mode=False)
        ]),
    val=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix='data/im2latex_data/formula_images_processed/',
                ann_file='data/im2latex_data/master_data/test.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='Im2TokenTextLineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator='	')),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(
                        type='ResizeOCR',
                        height=128,
                        min_width=128,
                        max_width=512,
                        keep_aspect_ratio=True),
                    dict(type='ToTensorOCR'),
                    dict(
                        type='NormalizeOCR',
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=[
                            'filename', 'ori_shape', 'img_shape',
                            'valid_ratio', 'img_norm_cfg', 'ori_filename'
                        ])
                ],
                dataset_info='table_Rec_val_debug_0',
                test_mode=True)
        ]),
    test=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix='data/im2latex_data/formula_images_processed/',
                ann_file='data/im2latex_data/master_data/test.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='Im2TokenTextLineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator='	')),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(
                        type='ResizeOCR',
                        height=128,
                        min_width=128,
                        max_width=512,
                        keep_aspect_ratio=True),
                    dict(type='ToTensorOCR'),
                    dict(
                        type='NormalizeOCR',
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=[
                            'filename', 'ori_shape', 'img_shape',
                            'valid_ratio', 'img_norm_cfg', 'ori_filename'
                        ])
                ],
                dataset_info='table_Rec_val_debug_0',
                test_mode=True)
        ]))
optimizer = dict(type='Ranger', lr=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.3333333333333333,
    step=[80, 100])
total_epochs = 100
evaluation = dict(interval=10, metric='acc')
fp16 = dict(loss_scale='dynamic')
work_dir = './expr_result/im2latex_res31_aug/'
gpu_ids = range(0, 1)
