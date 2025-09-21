import math

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from transformers.models.dinov3_vit.image_processing_dinov3_vit_fast import (
    DINOv3ViTImageProcessorFast,  # noqa
)


def pad_to_multiple(pil_img: Image.Image, multiple: int = 16, fill=(0, 0, 0)):
    """
    Pad a PIL image on right/bottom so that (H, W) are multiples of `multiple`.

    Args:
        pil_img: Input PIL Image
        multiple: The padding multiple (default: 16)
        fill: RGB fill color for padding (default: black)

    Returns:
        padded_img: Padded PIL Image
        pad_box: (left, top, right, bottom) padding sizes
    """
    W, H = pil_img.size
    W_pad = math.ceil(W / multiple) * multiple
    H_pad = math.ceil(H / multiple) * multiple

    if (W_pad, H_pad) == (W, H):
        return pil_img, (0, 0, 0, 0)

    # 直接用 expand 来减少拷贝（比新建大图再 paste 效率高）
    pad_left, pad_top = 0, 0
    pad_right, pad_bottom = W_pad - W, H_pad - H
    padded_img = Image.new(pil_img.mode, (W_pad, H_pad), fill)
    padded_img.paste(pil_img, (0, 0))

    return padded_img, (pad_left, pad_top, pad_right, pad_bottom)


def preprocess_image_no_resize(
    pil_img: Image.Image,
    multiple: int = 16,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """
    Preprocess PIL image without resizing:
      1. Pad (right/bottom) to make H,W multiples of `multiple`
      2. Convert to tensor
      3. Normalize with ImageNet stats

    Returns:
        dict: {"pixel_values": tensor (1,3,H,W)}
        disp_np: numpy array for visualization (H,W,3)
        pad_box: padding sizes (l,t,r,b)
    """
    img_padded, pad_box = pad_to_multiple(pil_img, multiple=multiple)

    transform = T.Compose(
        [
            T.ToTensor(),  # (C,H,W), float32, [0,1]
            T.Normalize(mean=mean, std=std),
        ]
    )

    pixel_tensor = transform(img_padded).unsqueeze(0)  # (1,3,H,W)
    disp_np = np.array(img_padded, dtype=np.uint8)  # for visualization

    return {"pixel_values": pixel_tensor}, disp_np, pad_box


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)

pretrained_model_name = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(pretrained_model_name, do_resize=False)
model = AutoModel.from_pretrained(
    pretrained_model_name,
    device_map="auto",
)
model.eval()

patch_size = 16

# inputs = processor(images=image, return_tensors="pt").to(model.device)
inputs, disp_np, _ = preprocess_image_no_resize(image, multiple=patch_size)
pixel_values = inputs["pixel_values"].to(model.device)

with torch.inference_mode():
    outputs = model(pixel_values=pixel_values)

print(outputs.keys())

pooled_output = outputs.pooler_output
hidden_states = outputs.last_hidden_state.squeeze(0)
print("Pooled output shape:", pooled_output.shape)
print("Hidden states shape:", hidden_states.shape)
