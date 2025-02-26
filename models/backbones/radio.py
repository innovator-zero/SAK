import torch
import torch.nn as nn

from .model_set import out_indices_cfg


class RADIO(nn.Module):
    def __init__(self, backbone_type: str, freeze: bool = True):
        super().__init__()

        log = f"{backbone_type}: Loading from torchhub, "

        if backbone_type == "radio_v2.5-b":
            self.out_indices = out_indices_cfg["base"]
        elif backbone_type == "radio_v2.5-l":
            self.out_indices = out_indices_cfg["large"]
        else:
            raise NotImplementedError

        self.model = torch.hub.load(
            "NVlabs/RADIO",
            "radio_model",
            version=backbone_type,
            progress=True,
            skip_validation=True,
        )
        self.model.input_conditioner = torch.nn.Identity()
        self.embed_dim = self.model.model.embed_dim

        if freeze:
            log += "Freeze backbone"
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            log += "Tune backbone"

        print(log)

    def forward(self, x):
        _, out = self.model.forward_intermediates(x, indices=self.out_indices)

        return out[1]
