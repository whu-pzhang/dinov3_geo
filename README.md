# DINOv3 in GeoAI

| Method      | Backbone   | Arch  | FT  | LoveDA mIoU | config                                                                 |
| ----------- | ---------- | ----- | --- | ----------- | ---------------------------------------------------------------------- |
| UPerNet     | DINOv3 Sat | ViT-L | N   | 51.98       | [config](./configs/seg/dinov3_vitl16_upernet_b8_80k_loveda_512.py)     |
| UPerNet     | DINOv3 Web | ViT-L | N   | 53.72       |                                                                        |
| Mask2former | DINOv3 Web | ViT-L | N   | TBD         | [config](./configs/seg/dinov3_vitl16_mask2former_b8_80k_loveda_512.py) |