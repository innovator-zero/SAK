from functools import partial
from typing import Dict, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer, resize_pos_embed

from models.backbones.model_set import hf_names, out_indices_cfg, timm_names
from utils import global_print


def preprocess(x: torch.Tensor, new_size: int) -> torch.Tensor:
    # Resize
    oldh, oldw = x.shape[-2:]
    scale = new_size * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    x = F.interpolate(x, (newh, neww), mode="bicubic", align_corners=False)

    # Pad
    padh = new_size - newh
    padw = new_size - neww
    x = F.pad(x, (0, padw, 0, padh))

    return x


# Modified from timm.models.vision_transformer._convert_openai_clip
def _convert_openai_clip(
    state_dict: Dict[str, torch.Tensor],
    model: VisionTransformer,
    prefix: str = "visual.",
) -> Dict[str, torch.Tensor]:
    out_dict = {}
    swaps = [
        ("conv1", "patch_embed.proj"),
        ("positional_embedding", "pos_embed"),
        ("transformer.resblocks.", "blocks."),
        ("ln_pre", "norm_pre"),
        ("ln_post", "norm"),
        ("ln_", "norm"),
        ("in_proj_", "qkv."),
        ("out_proj", "proj"),
        ("mlp.c_fc", "mlp.fc1"),
        ("mlp.c_proj", "mlp.fc2"),
    ]
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k.replace("module.", "")
        if not k.startswith(prefix):
            continue
        k = k.replace(prefix, "")
        for sp in swaps:
            k = k.replace(sp[0], sp[1])

        if k == "proj":
            continue  # remove proj layer
        elif k == "class_embedding":
            k = "cls_token"
            v = v.unsqueeze(0).unsqueeze(1)
        elif k == "pos_embed":
            v = v.unsqueeze(0)
            if v.shape[1] != model.pos_embed.shape[1]:
                # To resize pos embedding when using model at different size from pretrained weights
                v = resize_pos_embed(
                    v,
                    model.pos_embed,
                    (0 if getattr(model, "no_embed_class") else getattr(model, "num_prefix_tokens", 1)),
                    model.patch_embed.grid_size,
                )
        out_dict[k] = v
    return out_dict


# Enable SAM to work with different image size
def _convert_sam(
    state_dict: Dict[str, torch.Tensor],
    new_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    out_dict = {}
    for k, v in state_dict.items():
        if k == "pos_embed":
            # Resize pos embedding
            if v.shape[1:3] != new_state_dict[k].shape[1:3]:
                v = F.interpolate(
                    v.permute(0, 3, 1, 2),
                    size=new_state_dict[k].shape[1:3],
                    mode="bicubic",
                    antialias=False,
                )
                v = v.permute(0, 2, 3, 1)
        elif "rel_pos" in k:
            # Interpolate rel pos if needed.
            max_rel_dist = new_state_dict[k].shape[0]
            if v.shape[0] != max_rel_dist:
                v = F.interpolate(
                    v.reshape(1, v.shape[0], -1).permute(0, 2, 1),
                    size=max_rel_dist,
                    mode="linear",
                )
                v = v.reshape(-1, max_rel_dist).permute(1, 0)
        out_dict[k] = v
    return out_dict


class TimmEncoder(nn.Module):

    def __init__(
        self,
        backbone_type: str,
        img_size: Tuple[int, int],
        lora_config: dict = None,
        resize: bool = True,
        freeze: bool = True,
        pretrain: bool = True,
        checkpoint_path: str = None,
    ):
        super().__init__()
        assert backbone_type in timm_names, f"Model {backbone_type} not found in timm models"
        timm_name = timm_names[backbone_type]
        log = f"{backbone_type}: Loading {timm_name} backbone from timm, "

        if "sam" in backbone_type and resize:
            log += "Resize image to 1024x1024, "
            img_size = 1024
            self.preprocess = partial(preprocess, new_size=img_size)
        else:
            self.preprocess = None

        if "swin" in backbone_type:
            out_indices = [0, 1, 2, 3]
        else:
            out_indices = out_indices_cfg[backbone_type.split("_")[1]]

        if backbone_type in hf_names:
            timm_pretrain = False
        elif "sam" in backbone_type and not resize:
            timm_pretrain = False
        else:
            timm_pretrain = pretrain

        self.model = timm.create_model(
            timm_name,
            img_size=img_size,
            num_classes=0,
            global_pool="",
            pretrained=timm_pretrain,
            checkpoint_path=checkpoint_path,
        )
        self.embed_dim = self.model.embed_dim

        # Load pretrained weights from huggingface
        if pretrain and backbone_type in hf_names:
            log += "Loading " + hf_names[backbone_type] + " weights from huggingface, "
            state_dict = torch.hub.load_state_dict_from_url(
                "https://huggingface.co/" + hf_names[backbone_type] + "/resolve/main/open_clip_pytorch_model.bin",
                file_name=hf_names[backbone_type].split("/")[-1] + ".pth",
                map_location="cpu",
                weights_only=True,
            )
            state_dict = _convert_openai_clip(state_dict, self.model)
            self.model.load_state_dict(state_dict)

        # Load pretrained weight for SAM
        if "sam" in backbone_type and not resize:
            log += "Loading SAM weights for different image size, "
            state_dict = timm.create_model(
                timm_name,
                num_classes=0,
                global_pool="",
                pretrained=True,
            ).state_dict()
            state_dict = _convert_sam(state_dict, self.model.state_dict())
            self.model.load_state_dict(state_dict)

        # Remove useless layers
        if "dinov2" in backbone_type or "lip" in backbone_type or "vit" in backbone_type:
            self.model.norm = None
        elif "sam" in backbone_type:
            self.model.neck = None

        if "sam" in backbone_type or "swin" in backbone_type:
            self.model.forward = partial(
                self.model.forward_intermediates,
                indices=out_indices,
                norm=False,
                intermediates_only=True,
            )
        else:
            self.model.forward = partial(
                self.model.forward_intermediates,
                indices=out_indices,
                return_prefix_tokens=False,
                norm=False,
                intermediates_only=True,
            )

        if lora_config:
            log += "Use LoRA for backbone"
            from peft import LoraConfig, get_peft_model

            self.model = get_peft_model(self.model, LoraConfig(**lora_config))
            self.model.print_trainable_parameters()
        elif freeze:
            log += "Freeze backbone"
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            log += "Tune backbone"

        global_print(log)

    def forward(self, x: torch.Tensor):
        if self.preprocess is not None:
            x = self.preprocess(x=x)
        return self.model(x)
