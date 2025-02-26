import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.timm_encoders import out_indices_cfg
from utils import global_print


class Adapter(nn.Module):

    def __init__(self, input_dim: int, down_ratio: int, output_dim: int = None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        hidden_dim = int(input_dim // down_ratio)
        self.down = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden_dim, output_dim)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        residual = x
        x = self.down(x)
        x = self.act(x)
        x = self.up(x)
        x = x * self.scale
        return x + residual


class GlobalRouter(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, input_size: tuple):
        super().__init__()

        self.input_size = input_size
        self.module = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, output_dim),
        )

    def forward(self, x):
        H, W = self.input_size
        x = self.module(x)  # B, L, C'
        # Global pooling
        x = torch.mean(x, dim=1)  # B, C'
        # B, C' -> B, C', H, W
        x = x.reshape(x.shape[0], x.shape[1], 1, 1).expand(-1, -1, H, W)
        return x


class MoR(nn.Module):
    """
    Our proposed Mixture-of-Representations.
    :param int input_dim: Input dimension.
    :param tuple input_size: Input representation size (H, W).
    :param list tea_names: List of teacher names.
    :param bool noisy_gating: Use noisy gating.
    """

    def __init__(
        self,
        input_dim: int,
        input_size: tuple,
        tea_names: list,
        noisy_gating: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.tea_names = tea_names
        self.noisy_gating = noisy_gating

        if noisy_gating:
            router_dim = (len(tea_names) + 1) * 2
        else:
            router_dim = len(tea_names) + 1

        self.router = GlobalRouter(input_dim, router_dim, input_size)

    def forward(self, vit_fea, stu_fea_dict):
        H, W = self.input_size
        B = vit_fea.shape[0]

        logits = self.router(vit_fea)  # B, num_experts(*2), H, W

        if self.noisy_gating:
            clean_logits, raw_noise_stddev = logits.chunk(2, dim=1)
            if self.training:
                noise_stddev = F.softplus(raw_noise_stddev) + 1e-2
                eps = torch.randn_like(clean_logits)
                noisy_logits = clean_logits + eps * noise_stddev
                logits = noisy_logits
            else:
                logits = clean_logits

        probs = F.softmax(logits, dim=1)  # B, num_experts, H, W

        # Get feature of each expert
        fea_list = [vit_fea.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()]
        for tea_name in self.tea_names:  # maintain order
            stu_fea = stu_fea_dict[tea_name]
            fea_list.append(stu_fea.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous())

        assert len(fea_list) == probs.shape[1]
        fea = torch.stack(fea_list, dim=1)  # B, num_experts, C, H, W
        fea = fea * probs.unsqueeze(2)  # B, num_experts, C, H, W
        fea = fea.sum(dim=1)  # B, C, H, W

        return fea


class SAK(nn.Module):
    """
    Our proposed SAK framework.
    :param tuple img_size: Input image size (H, W).
    :param str vit_name: ViT backbone (Teacher-Agnostic Stem) name.
    :param dict tea_dims: Teacher dimensions.
    :param int down_ratio: Down ratio for Teacher-Specific Adapter Path modules.
    :param bool aligner: Use aligners to mitigate the dimension differences between student and teacher.
    :param list tasks: List of tasks.
    :param str router_type: Router type "mor" or "none".
    :param bool noisy_gating: Use noisy gating.
    :param bool freeze_vit: Freeze ViT backbone (TAS).
    :param bool freeze: Freeze all except MoR Routers (TAS and TSAP).
    :param dict lora_config: LoRA config.
    """

    def __init__(
        self,
        img_size: tuple,
        vit_name: str,
        tea_dims: dict,
        down_ratio: int = 4,
        aligner: bool = False,
        tasks: list = None,
        router_type: str = "mor",
        noisy_gating: bool = True,
        freeze_vit: bool = False,
        freeze: bool = False,
        lora_config: dict = None,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        # ViT backbone (Teacher-Agnostic Stem)
        if vit_name == "vit_small":
            from models.backbones.vit import vit_small_patch16_384

            self.vit = vit_small_patch16_384(img_size=img_size, pretrained=True)
        elif vit_name == "vit_base":
            from models.backbones.vit import vit_base_patch16_384

            self.vit = vit_base_patch16_384(img_size=img_size, pretrained=True)
        elif vit_name == "vit_large":
            from models.backbones.vit import vit_large_patch16_384

            self.vit = vit_large_patch16_384(img_size=img_size, pretrained=True)
        else:
            raise NotImplementedError
        self.vit.norm = None

        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False
        elif lora_config:
            from peft import LoraConfig, get_peft_model

            self.vit = get_peft_model(self.vit, LoraConfig(**lora_config))
            if tasks:
                global_print("Full fine-tune ViT and LoRA in stage 2")
                for param in self.vit.parameters():
                    param.requires_grad = True
            self.vit.print_trainable_parameters()

        self.embed_dim = self.vit.embed_dim
        self.out_indices = out_indices_cfg[vit_name.split("_")[1]]
        self.fea_size = (img_size[0] // 16, img_size[1] // 16)

        self.tea_names = tea_dims.keys()
        self.tasks = tasks

        # Teacher-Specific Adapter Path
        self.tsap = nn.ModuleDict()
        for tea_name in self.tea_names:
            self.tsap[tea_name] = nn.ModuleList()
            self.tsap[tea_name].append(Adapter(self.embed_dim, down_ratio))  # for patch embed

        self.out_norms = nn.ModuleDict()

        # Teacher-Specific Aligners
        if aligner:
            self.ts_aligners = nn.ModuleDict()
        else:
            for tea_dim in tea_dims.values():
                if tea_dim == 0:
                    global_print("Aligners are disabled without distillation")
                    break
                assert tea_dim == self.embed_dim, "Aligners are disabled, teacher dim must match student dim"
            self.ts_aligners = None

        # Mixture-of-Representations Routers
        if router_type == "none":
            self.mor = None
        else:
            assert router_type == "mor"
            self.mor = nn.ModuleDict() if tasks else None

        for i in range(len(self.vit.blocks)):
            # A group of adapters for each block
            for tea_name in self.tea_names:
                self.tsap[tea_name].append(Adapter(self.embed_dim, down_ratio))

            if i in self.out_indices:
                # A group of output norms for each adapter
                for tea_name in self.tea_names:
                    self.out_norms[str(i) + "_" + tea_name] = nn.LayerNorm(self.embed_dim)

                    # A group of aligners for each output block
                    if aligner:
                        self.ts_aligners[str(i) + "_" + tea_name] = nn.Linear(self.embed_dim, tea_dims[tea_name])

                # A group of MoR for each task
                if tasks and router_type != "none":
                    for task in self.tasks:
                        self.mor[str(i) + "_" + task] = MoR(
                            self.embed_dim,
                            self.fea_size,
                            self.tea_names,
                            noisy_gating,
                        )

        # Init weights for adapters and aligners
        self.tsap.apply(self._init_weights)
        self.out_norms.apply(self._init_weights)

        if self.ts_aligners:
            self.ts_aligners.apply(self._init_weights)

        if self.mor:
            self.mor.apply(self._init_weights)

        if freeze:
            for name, param in self.named_parameters():
                if not "mor" in name:
                    param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        aligned_feas_dict = {tea_name: [] for tea_name in self.tea_names}
        # Dict of list of aligned student features for distillation

        if self.tasks is not None:
            out_feas_dict = {task: [] for task in self.tasks}
            # Dict of list of output features for task decoder

        # Forward pass
        B = x.shape[0]
        H, W = self.fea_size
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)

        residuals_dict = {}  # Dict of residuals for each adapter
        for tea_name in self.tea_names:
            residuals_dict[tea_name] = self.tsap[tea_name][0](x)

        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)

            # Get adapter features for each teacher
            for tea_name in self.tea_names:
                residuals_dict[tea_name] = self.tsap[tea_name][i + 1](residuals_dict[tea_name] + x)

            if i in self.out_indices:
                stu_fea_dict = {}
                # Dict of student features in this block {tea_name: [B, L, C]}
                for tea_name in self.tea_names:
                    # Student out feature = residual
                    stu_fea = residuals_dict[tea_name][:, self.vit.num_prefix_tokens :]
                    stu_fea = self.out_norms[str(i) + "_" + tea_name](stu_fea)
                    stu_fea_dict[tea_name] = stu_fea

                    # Align embed dim to teacher's
                    if self.ts_aligners:
                        aligned_fea = self.ts_aligners[str(i) + "_" + tea_name](stu_fea)
                        # dict of {tea_name: [B, L, C_T]}
                    else:
                        aligned_fea = stu_fea

                    aligned_fea = aligned_fea.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                    aligned_feas_dict[tea_name].append(aligned_fea)

                # Output features for each task
                if self.tasks is not None:
                    # Get ViT feature tokens
                    vit_fea = x[:, self.vit.num_prefix_tokens :]

                    # Get MoR output features for each task
                    if self.mor is not None:
                        for task in self.tasks:
                            out_fea = self.mor[str(i) + "_" + task](vit_fea, stu_fea_dict)
                            out_feas_dict[task].append(out_fea)
                    else:
                        # No MoR, sum all student features
                        out_fea = vit_fea.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

                        for tea_name in self.tea_names:  # maintain order
                            stu_fea = stu_fea_dict[tea_name]
                            out_fea += stu_fea.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                        for task in self.tasks:
                            out_feas_dict[task].append(out_fea)

        if self.tasks is not None:
            return aligned_feas_dict, out_feas_dict
        else:
            return aligned_feas_dict
