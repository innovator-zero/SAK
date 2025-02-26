from typing import Tuple

import torch
import torch.nn as nn

from models.backbones.model_set import hf_names, out_indices_cfg


class HfEncoder(nn.Module):

    def __init__(
        self,
        backbone_type: str,
        img_size: Tuple[int, int],
        freeze: bool = True,
        lora_config: dict = None,
    ):
        super().__init__()
        assert (
            backbone_type in hf_names
        ), f"Model {backbone_type} not found in huggingface models"
        model_name = hf_names[backbone_type]
        log = f"{backbone_type}: Loading {model_name} backbone from huggingface, "

        self.out_indices = out_indices_cfg[backbone_type.split("_")[1]]

        if "siglip" in backbone_type:
            from transformers import SiglipVisionModel

            self.model = SiglipVisionModel.from_pretrained(model_name)
            self.model.vision_model.head = nn.Identity()
            self.model.vision_model.post_layernorm = nn.Identity()
            self.embed_dim = self.model.config.hidden_size
        elif "theia" in backbone_type:
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True
            ).backbone.model
            self.model.layernorm = nn.Identity()
            self.embed_dim = self.model.config.hidden_size

        else:
            raise NotImplementedError

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

        print(log)

    def forward(self, x: torch.Tensor):
        B, _, height, width = x.shape
        patch_size = self.model.config.patch_size
        H, W = height // patch_size, width // patch_size

        y = self.model(x, output_hidden_states=True, interpolate_pos_encoding=True)
        hidden_states = y.hidden_states[1:]  # remove patch embeddings

        features = []
        for i in self.out_indices:
            if hidden_states[i].shape[1] != H * W:
                fea = hidden_states[i][:, 1:]
            else:
                fea = hidden_states[i]
            features.append(fea.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous())

        return features
