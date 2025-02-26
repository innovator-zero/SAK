import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.utils.configs import get_output_num
from models.backbones.model_set import hf_names, timm_names
from models.heads import DecodeHead


def build_model(
    arch: str,
    img_size: tuple,
    backbone_args: dict,
    dataname: str = None,
    tasks: list = None,
) -> nn.Module:
    """
    Initialize the model
    """

    backbone = get_backbone(img_size=img_size, tasks=tasks, **backbone_args)

    if arch == "backbone":
        return backbone
    elif arch == "mt":
        assert dataname is not None and tasks is not None
        heads = get_head(img_size=img_size, embed_dim=backbone.embed_dim, tasks=tasks, dataname=dataname)
        model = MultiTaskModel(backbone, heads, tasks)
        return model
    else:
        raise NotImplementedError


def get_backbone(backbone_type: str, img_size: tuple, tasks: list = None, **args) -> nn.Module:
    """Return backbone"""

    if backbone_type == "sak":
        from models.backbones.sak import SAK

        backbone = SAK(img_size=img_size, tasks=tasks, **args)
    elif backbone_type in ["radio_v2.5-b", "radio_v2.5-l"]:
        from models.backbones.radio import RADIO

        backbone = RADIO(backbone_type=backbone_type, **args)
    elif backbone_type in timm_names:
        from models.backbones.timm_encoders import TimmEncoder

        backbone = TimmEncoder(backbone_type=backbone_type, img_size=img_size, **args)
    elif backbone_type in hf_names:
        from models.backbones.hf_encoders import HfEncoder

        backbone = HfEncoder(backbone_type=backbone_type, img_size=img_size, **args)
    else:
        raise NotImplementedError

    return backbone


def get_head(img_size: tuple, embed_dim: int, tasks: list, dataname: str, **args) -> nn.ModuleDict:
    """Return heads"""

    heads = nn.ModuleDict()
    for task in tasks:
        heads[task] = DecodeHead(
            img_size=img_size, in_channels=embed_dim, num_classes=get_output_num(task, dataname), **args
        )

    return heads


class MultiTaskModel(nn.Module):
    """Multi-Task model with shared encoder + task-specific heads"""

    def __init__(self, backbone: nn.Module, heads: nn.ModuleDict, tasks: list) -> None:
        super().__init__()
        self.backbone = backbone
        self.heads = heads
        self.tasks = tasks

    def forward(self, x: torch.Tensor):
        out = {}
        img_size = x.size()[2:]

        if self.backbone.__class__.__name__ == "SAK":
            aligned_feas_dict, out_feas_dict = self.backbone(x)
            # aligned_feas_dict: dict of {tea_name: list of [B, C_T, H, W]}
            # out_feas_dict: dict of {task: list of [B, C, H, W]}

            for task in self.tasks:
                out[task] = F.interpolate(self.heads[task](out_feas_dict[task]), img_size, mode="bilinear")

            return aligned_feas_dict, out
        else:
            encoder_output = self.backbone(x)
            for task in self.tasks:
                out[task] = F.interpolate(self.heads[task](encoder_output), img_size, mode="bilinear")

            return out
