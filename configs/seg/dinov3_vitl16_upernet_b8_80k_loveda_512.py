_base_ = [
    "mmseg::_base_/default_runtime.py",
    "mmseg::_base_/datasets/loveda.py",
    "mmseg::_base_/schedules/schedule_80k.py",
]

custom_imports = dict(imports=["src.dinov3_backbone"], allow_failed_imports=False)

crop_size = (512, 512)
data_preprocessor = dict(
    type="SegDataPreProcessor",
    size=crop_size,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
)
norm_cfg = dict(type="SyncBN", requires_grad=True)
checkpoint_file = "weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
# checkpoint_file = "weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="DINOv3ViTBackbone",
        repo_or_dir="dinov3",
        model_name="dinov3_vitl16",
        out_indices=(7, 11, 15, 23),
        patch_size=16,
        fp16=True,
        frozen=True,
        init_cfg=dict(type="Pretrained", checkpoint=checkpoint_file),
    ),
    neck=dict(
        type="Feature2Pyramid",
        embed_dim=1024,
        rescales=[4, 2, 1, 0.5],
        norm_cfg=norm_cfg,
    ),
    decode_head=dict(
        type="UPerHead",
        in_channels=[1024, 1024, 1024, 1024],
        num_classes=7,
        ignore_index=255,
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)


train_dataloader = dict(batch_size=8, num_workers=8)


# optimizer
optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=6e-5, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            "pos_embed": dict(decay_mult=0.0),
            "cls_token": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=80000,
        by_epoch=False,
    ),
]

default_hooks = dict(checkpoint=dict(interval=8000, max_keep_ckpts=1, save_best="mIoU"))
