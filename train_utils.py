import torch
from timm.layers import resample_abs_pos_embed
from timm.scheduler.cosine_lr import CosineLRScheduler


class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, max_iterations, gamma=0.9, min_lr=0.0, last_epoch=-1):
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # slight abuse: last_epoch refers to last iteration
        factor = (1 - self.last_epoch / float(self.max_iterations)) ** self.gamma
        return [(base_lr - self.min_lr) * factor + self.min_lr for base_lr in self.base_lrs]


def get_optimizer_scheduler(config, model):
    """
    Get optimizer and scheduler for model
    """
    params = model.parameters()

    if config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=float(config["lr"]),
            momentum=0.9,
            weight_decay=float(config["weight_decay"]),
        )

    elif config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(params, lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))

    elif config["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(params, lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))

    else:
        raise NotImplementedError("Invalid optimizer %s!" % config["optimizer"])

    if config["scheduler"] == "poly":
        # Operate in each iteration
        assert config["max_iters"] is not None
        scheduler = PolynomialLR(
            optimizer=optimizer,
            max_iterations=int(config["max_iters"]),
            gamma=0.9,
            min_lr=0,
        )

    elif config["scheduler"] == "cosine":
        # Operate in each epoch
        assert config["max_epochs"] is not None
        assert config["warmup_epochs"] is not None
        max_epochs = int(config["max_epochs"])
        warmup_epochs = int(config["warmup_epochs"])
        scheduler = CosineLRScheduler(
            optimizer=optimizer,
            t_initial=max_epochs - warmup_epochs,
            lr_min=1.25e-6,
            warmup_t=warmup_epochs,
            warmup_lr_init=1.25e-7,
            warmup_prefix=True,
        )

    else:
        raise NotImplementedError("Invalid scheduler %s!" % config["scheduler"])

    return optimizer, scheduler


def cal_params(model):
    tea_params = 0
    tea_trainable_params = 0
    stu_params = 0
    stu_trainable_params = 0
    stu_tas_params = 0
    stu_tsap_params = 0
    stu_aligner_params = 0
    stu_mor_params = 0
    stu_head_params = 0

    for name, param in model.named_parameters():
        if "teacher" in name:
            tea_params += param.numel()
            if param.requires_grad:
                tea_trainable_params += param.numel()
        else:
            stu_params += param.numel()
            if param.requires_grad:
                stu_trainable_params += param.numel()

            if "vit" in name:
                stu_tas_params += param.numel()
            elif "tsap" in name:
                stu_tsap_params += param.numel()
            elif "ts_aligners" in name:
                stu_aligner_params += param.numel()
            elif "mor" in name:
                stu_mor_params += param.numel()
            elif "head" in name:
                stu_head_params += param.numel()

    # Print a table
    print("--- Number of parameters ---")
    print(f"Teachers:     {tea_params/1e6:>10.2f}M")
    print(f"Trainable:    {tea_trainable_params/1e6:>10.2f}M")
    print(f"Student:      {stu_params/1e6:>10.2f}M")
    print(f"Trainable:    {stu_trainable_params/1e6:>10.2f}M")
    print(f"TAS:          {stu_tas_params/1e6:>10.2f}M")
    print(f"TSAP:         {stu_tsap_params/1e6:>10.2f}M")
    print(f"Aligners:     {stu_aligner_params/1e6:>10.2f}M")
    print(f"MoR:          {stu_mor_params/1e6:>10.2f}M")
    print(f"Heads:        {stu_head_params/1e6:>10.2f}M")

    return stu_params


def update_weights(model, state_dict):
    update_state_dict = model.state_dict()
    for k in update_state_dict.keys():
        # kk = k.replace("module.", "")  # model is wrapped by DDP
        if k in state_dict:  # same arch
            old_k = k
        elif k.replace("backbone.", "") in state_dict:  # distilled in backbone, now multi-task model
            old_k = k.replace("backbone.", "")
        else:
            continue

        if "pos_embed" in k:
            v = state_dict[old_k]
            # # Resize pos embedding
            if v.shape[1] != update_state_dict[k].shape[1]:
                if hasattr(model, "backbone"):
                    backbone = model.backbone
                else:
                    backbone = model
                v = resample_abs_pos_embed(
                    v,
                    new_size=backbone.vit.patch_embed.grid_size,
                    num_prefix_tokens=1,
                    interpolation="bicubic",
                    antialias=False,
                    verbose=True,
                )
            update_state_dict[k] = v
        else:
            update_state_dict[k] = state_dict[old_k]

    model.load_state_dict(update_state_dict)
