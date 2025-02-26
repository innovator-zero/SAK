import torch
import torch.nn as nn
import torch.nn.functional as F


class DecodeHead(nn.Module):
    """
    Simple MLP & Conv head
    """

    def __init__(
        self,
        img_size,
        in_channels,
        num_classes,
        kernel_size=3,
    ):
        super().__init__()
        self.opr_size = (img_size[0] // 4, img_size[1] // 4)
        self.in_channels = in_channels
        self.embed_dim = in_channels // 4

        self.linears = nn.ModuleList()
        for _ in range(4):
            self.linears.append(nn.Linear(in_channels, self.embed_dim))

        padding = kernel_size // 2
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )

        self.last_conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        feas = []
        for i in range(4):
            B, C, H, W = inputs[i].shape
            assert C == self.in_channels, "Input feature dimension mismatch!"
            fea = inputs[i].flatten(2).transpose(1, 2)  # B, h*W, C

            # dimension reduction
            fea = self.linears[i](fea)
            # B, h*w, C/4 => B, C/4, h, w
            fea = fea.permute(0, 2, 1).reshape(B, self.embed_dim, H, W)
            fea = F.interpolate(fea, size=self.opr_size, mode="bilinear", align_corners=False)
            feas.append(fea)

        x = self.linear_fuse(torch.cat(feas, dim=1).contiguous())  # B, C, H/4, W/4
        x = self.last_conv(x)
        return x
