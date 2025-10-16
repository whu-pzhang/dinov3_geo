# DINOv3 in dense prediction task

## Results

### ADE20K

| Method  | Backbone   | Arch  | mIoU  | config                                                             |
| ------- | ---------- | ----- | ----- | ------------------------------------------------------------------ |
| UPerNet | DINOv3 Web | ViT-L | 52.65 | [config](./configs/seg/dinov3_vitl16_upernet_b8_80k_ade20k_512.py) |



### LoveDA

| Method      | Backbone   | Arch  | mIoU  | config                                                                 |
| ----------- | ---------- | ----- | ----- | ---------------------------------------------------------------------- |
| UPerNet     | DINOv3 Sat | ViT-L | 51.98 | [config](./configs/seg/dinov3_vitl16_upernet_b8_80k_loveda_512.py)     |
| UPerNet     | DINOv3 Web | ViT-L | 53.72 |                                                                        |
| Mask2former | DINOv3 Web | ViT-L | TBD   | [config](./configs/seg/dinov3_vitl16_mask2former_b8_80k_loveda_512.py) |