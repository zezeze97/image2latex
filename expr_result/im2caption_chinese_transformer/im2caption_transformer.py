checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'expr_result/im2caption_chinese_transformer/pretrained.pth'
resume_from = None
workflow = [('train', 1)]
alphabet_file = '/home/zhangzr/im2caption_data/chinese_caption/chinese_vocab.txt'
alphabet_len = 2530
max_seq_len = 40
start_end_same = False
label_convertor = dict(
    type='MasterConvertor',
    dict_file='/home/zhangzr/im2caption_data/chinese_caption/chinese_vocab.txt',
    max_seq_len=40,
    start_end_same=False,
    with_unknown=True)
PAD = 2533
model = dict(
    type='MASTER',
    backbone=dict(
        type='SimpleProjection', input_format='RGB', downsample=(32, 32)),
    encoder=dict(
        type='MasterTFEncoder',
        n_layers=6,
        n_head=8,
        d_k=64,
        d_v=64,
        d_model=512,
        d_inner=2048,
        dropout=0.0,
        max_len=5000,
        position_dropout=0.0),
    decoder=dict(
        type='MasterDecoder',
        N=6,
        decoder=dict(
            self_attn=dict(headers=8, d_model=512, dropout=0.0),
            src_attn=dict(headers=8, d_model=512, dropout=0.0),
            feed_forward=dict(d_model=512, d_ff=2048, dropout=0.0),
            size=512,
            dropout=0.0),
        d_model=512),
    loss=dict(type='MASTERTFLoss', ignore_index=2533, reduction='mean'),
    label_convertor=dict(
        type='MasterConvertor',
        dict_file=
        '/home/zhangzr/im2caption_data/chinese_caption/chinese_vocab.txt',
        max_seq_len=40,
        start_end_same=False,
        with_unknown=True),
    max_seq_len=40)
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='CaptionAug',
        args=[['Fliplr', 0.5], {
            'cls': 'Crop',
            'percent': (0, 0.2)
        }]),
    dict(
        type='ResizeOCR',
        height=384,
        min_width=384,
        max_width=384,
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
        height=384,
        min_width=384,
        max_width=384,
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
img_prefix1 = '/home/zhangzr/im2caption_data/chinese_caption/ai_challenger_caption_train_20170902/caption_train_images_20170902/'
train_anno_file1 = '/home/zhangzr/im2caption_data/chinese_caption/train.txt'
train1 = dict(
    type='OCRDataset',
    img_prefix=
    '/home/zhangzr/im2caption_data/chinese_caption/ai_challenger_caption_train_20170902/caption_train_images_20170902/',
    ann_file='/home/zhangzr/im2caption_data/chinese_caption/train.txt',
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
            args=[['Fliplr', 0.5], {
                'cls': 'Crop',
                'percent': (0, 0.2)
            }]),
        dict(
            type='ResizeOCR',
            height=384,
            min_width=384,
            max_width=384,
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
test_img_prefix = '/home/zhangzr/im2caption_data/chinese_caption/ai_challenger_caption_validation_20170910/caption_validation_images_20170910/'
test_ann_files = dict(
    table_Rec_val_debug_0=
    '/home/zhangzr/im2caption_data/chinese_caption/mini_val.txt')
testset = [
    dict(
        type='OCRDataset',
        img_prefix=
        '/home/zhangzr/im2caption_data/chinese_caption/ai_challenger_caption_validation_20170910/caption_validation_images_20170910/',
        ann_file='/home/zhangzr/im2caption_data/chinese_caption/mini_val.txt',
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
                height=384,
                min_width=384,
                max_width=384,
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
    samples_per_gpu=128,
    workers_per_gpu=4,
    train=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix=
                '/home/zhangzr/im2caption_data/chinese_caption/ai_challenger_caption_train_20170902/caption_train_images_20170902/',
                ann_file=
                '/home/zhangzr/im2caption_data/chinese_caption/train.txt',
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
                        args=[['Fliplr', 0.5], {
                            'cls': 'Crop',
                            'percent': (0, 0.2)
                        }]),
                    dict(
                        type='ResizeOCR',
                        height=384,
                        min_width=384,
                        max_width=384,
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
                img_prefix=
                '/home/zhangzr/im2caption_data/chinese_caption/ai_challenger_caption_validation_20170910/caption_validation_images_20170910/',
                ann_file=
                '/home/zhangzr/im2caption_data/chinese_caption/mini_val.txt',
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
                        height=384,
                        min_width=384,
                        max_width=384,
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
                img_prefix=
                '/home/zhangzr/im2caption_data/chinese_caption/ai_challenger_caption_validation_20170910/caption_validation_images_20170910/',
                ann_file=
                '/home/zhangzr/im2caption_data/chinese_caption/mini_val.txt',
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
                        height=384,
                        min_width=384,
                        max_width=384,
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
optimizer = dict(type='Ranger', lr=0.001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.3333333333333333,
    step=[20, 30])
total_epochs = 30
evaluation = dict(interval=1, metric='acc')
fp16 = dict(loss_scale='dynamic')
work_dir = 'expr_result/im2caption_chinese_transformer/'
gpu_ids = range(0, 2)
