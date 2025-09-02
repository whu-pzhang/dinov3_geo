import torch
import torch.nn as nn
from mmengine.logging import MMLogger
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from mmengine.runner.checkpoint import _load_checkpoint


@MODELS.register_module()
class DINOv3Backbone(BaseModule):
    def __init__(
        self,
        *,
        repo_or_dir: str,
        model_name: str = "dinov3_vitl16",
        out_indices=(7, 11, 15, 23),
        patch_size: int = 16,
        fp16: bool = False,
        frozen: bool = False,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.vit = torch.hub.load(
            repo_or_dir,
            model_name,
            source="local",
            weights=init_cfg.get("checkpoint", None),
        )
        self.out_indices = tuple(out_indices)
        self.embed_dim = getattr(self.vit, "embed_dim", 1024)
        self.out_channels = [self.embed_dim] * len(self.out_indices)
        self.frozen = frozen
        self.fp16 = fp16

        if frozen:
            for p in self.vit.parameters():
                p.requires_grad_(False)

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        """Load ViT from init_cfg checkpoint, then init only scale_ops."""
        # 1) fingerprint
        wt_sum = sum(p.detach().abs().mean() for p in self.vit.parameters())
        logger.info(f"[DINOv3] pre-load weight abs-mean fingerprint: {wt_sum:.3f}")

        # 2) load checkpoint (MMEngine already passed us the path via init_cfg)
        logger.info(f"Loading from {self.init_cfg['checkpoint']}")
        ckpt = _load_checkpoint(self.init_cfg["checkpoint"], map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        missing, unexpected = self.vit.load_state_dict(sd, strict=False)
        logger.info(
            f"[DINOv3] Loaded ViT: missing_keys={missing}, unexpected_keys={unexpected}"
        )

        wt_sum = sum(p.detach().abs().mean() for p in self.vit.parameters())
        logger.info(f"[DINOv3] post-load weight abs-mean fingerprint: {wt_sum:.3f}")

    def forward(self, x):
        """
        if self.training:
            print("=== ViT parameter requires_grad states ===")
            for name, p in self.vit.named_parameters():
                print(f"{name:60s}  requires_grad={p.requires_grad}")
            print("==========================================")
        """
        device = x.device.type
        with torch.amp.autocast(device, dtype=torch.float16):
            feats = self.vit.get_intermediate_layers(
                x,
                n=self.out_indices,
                reshape=True,
                norm=True,
                return_class_token=False,
            )
        feats = [f.float() for f in feats]

        return feats


class Norm2d(nn.Module):
    """LayerNorm on a (N,C,H,W) tensor."""

    def __init__(self, c: int):
        super().__init__()
        self.ln = nn.LayerNorm(c, eps=1e-6)

    def forward(self, x):
        # (N,C,H,W) → (N,H,W,C) → LN → back to (N,C,H,W)
        x = self.ln(x.permute(0, 2, 3, 1))
        return x.permute(0, 3, 1, 2).contiguous()


@MODELS.register_module()
class Feature2Pyramid(BaseModule):
    def __init__(self, embed_dim, out_channels, rescales=[4, 2, 1, 0.5]):
        super().__init__()
        self.rescales = rescales
        self.ops = nn.ModuleList()
        for r in rescales:
            if r == 4:
                self.ops.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            embed_dim, embed_dim, kernel_size=2, stride=2, bias=False
                        ),
                        nn.BatchNorm2d(embed_dim),
                        nn.GELU(),
                        nn.ConvTranspose2d(
                            embed_dim, embed_dim, kernel_size=2, stride=2
                        ),
                    )
                )
            elif r == 2:
                self.ops.append(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)
                )
            elif r == 1:
                self.ops.append(nn.Identity())
            elif r == 0.5:
                self.ops.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                raise KeyError(f"Invalid rescale factor: {r}")

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, Norm2d):
                nn.init.ones_(m.ln.weight)
                nn.init.zeros_(m.ln.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, inputs):
        assert len(inputs) == len(self.rescales)
        outs = []
        for i, feat in enumerate(inputs):
            x = self.ops[i](feat)
            outs.append(x)
        return tuple(outs)
