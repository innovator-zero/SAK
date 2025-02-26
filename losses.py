import torch
import torch.nn as nn
import torch.nn.functional as F

# Loss functions and hyperparameters
PASCAL_LOSS_CONFIG = {
    "semseg": {"loss_function": "CELoss", "weight": 1},
    "human_parts": {"loss_function": "CELoss", "weight": 2},
    "normals": {
        "loss_function": "L1Loss",
        "parameters": {"normalize": True},
        "weight": 10,
    },
    "sal": {"loss_function": "CELoss", "parameters": {"balanced": True}, "weight": 5},
    "edge": {
        "loss_function": "BalancedBCELoss",
        "parameters": {"pos_weight": 0.95},
        "weight": 50,
    },
}

NYUD_LOSS_CONFIG = {
    "semseg": {"loss_function": "CELoss", "weight": 1},
    "normals": {
        "loss_function": "L1Loss",
        "parameters": {"normalize": True},
        "weight": 10,
    },
    "edge": {
        "loss_function": "BalancedBCELoss",
        "parameters": {"pos_weight": 0.95},
        "weight": 50,
    },
    "depth": {"loss_function": "L1Loss", "weight": 1},
}


class BalancedBCELoss(nn.Module):
    # Edge Detection

    def __init__(self, pos_weight=0.95, ignore_index=255):
        super().__init__()
        self.pos_weight = pos_weight
        self.ignore_index = ignore_index

    def forward(self, output, label):
        mask = label != self.ignore_index
        masked_output = torch.masked_select(output, mask)  # 1-d tensor
        masked_label = torch.masked_select(label, mask)  # 1-d tensor

        # pos weight: w, neg weight: 1-w
        w = torch.tensor(self.pos_weight, device=output.device)
        factor = 1.0 / (1 - w)
        loss = F.binary_cross_entropy_with_logits(
            masked_output, masked_label, pos_weight=w * factor
        )
        loss /= factor

        return loss


class CELoss(nn.Module):
    # Semantic Segmentation, Human Parts Segmentation, Saliency Detection

    def __init__(self, balanced=False, ignore_index=255):
        super(CELoss, self).__init__()
        self.ignore_index = ignore_index
        self.balanced = balanced

    def forward(self, output, label):
        label = torch.squeeze(label, dim=1).long()

        if self.balanced:
            mask = label != self.ignore_index
            masked_label = torch.masked_select(label, mask)
            assert torch.max(masked_label) < 2  # binary

            num_labels_neg = torch.sum(1.0 - masked_label)
            num_total = torch.numel(masked_label)
            pos_weight = num_labels_neg / num_total
            class_weight = torch.stack((1.0 - pos_weight, pos_weight), dim=0)
            loss = F.cross_entropy(
                output,
                label,
                weight=class_weight,
                ignore_index=self.ignore_index,
                reduction="sum",
            )
        else:
            loss = F.cross_entropy(
                output, label, ignore_index=self.ignore_index, reduction="sum"
            )

        n_valid = (label != self.ignore_index).sum()
        loss /= max(n_valid, 1)

        return loss


class L1Loss(nn.Module):
    # Normals Estimation, Depth Estimation

    def __init__(self, normalize=False, ignore_index=255):
        super(L1Loss, self).__init__()
        self.normalize = normalize
        self.ignore_index = ignore_index

    def forward(self, output, label):
        if self.normalize:
            # Normalize to unit vector
            output = F.normalize(output, p=2, dim=1)

        mask = (label != self.ignore_index).all(dim=1, keepdim=True)
        masked_output = torch.masked_select(output, mask)
        masked_label = torch.masked_select(label, mask)

        loss = F.l1_loss(masked_output, masked_label, reduction="sum")
        n_valid = torch.sum(mask).item()
        loss /= max(n_valid, 1)

        return loss


def get_loss_functions(task_loss_config):
    """
    Get loss function for each task
    """

    key2loss = {
        "CELoss": CELoss,
        "BalancedBCELoss": BalancedBCELoss,
        "L1Loss": L1Loss,
    }

    # Get loss function for each task
    loss_fx = key2loss[task_loss_config["loss_function"]]
    if "parameters" in task_loss_config:
        loss_ft = loss_fx(**task_loss_config["parameters"])
    else:
        loss_ft = loss_fx()

    return loss_ft


class MultiTaskLoss(nn.Module):
    """
    Multi-Task loss with different loss functions and weights
    """

    def __init__(self, tasks, loss_ft, loss_weights):
        super(MultiTaskLoss, self).__init__()
        assert set(tasks) == set(loss_ft.keys())
        assert set(tasks) == set(loss_weights.keys())
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    def forward(self, pred, gt, tasks):
        out = {t: self.loss_weights[t] * self.loss_ft[t](pred[t], gt[t]) for t in tasks}
        out["total"] = torch.sum(torch.stack([out[t] for t in tasks]))

        return out


def get_criterion(dataname, tasks):
    if dataname == "pascalcontext":
        losses_config = PASCAL_LOSS_CONFIG
    elif dataname == "nyud":
        losses_config = NYUD_LOSS_CONFIG
    else:
        raise NotImplementedError

    loss_ft = torch.nn.ModuleDict(
        {task: get_loss_functions(losses_config[task]) for task in tasks}
    )
    loss_weights = {task: losses_config[task]["weight"] for task in tasks}

    return MultiTaskLoss(tasks, loss_ft, loss_weights)


class DistillLoss(nn.Module):
    """
    Distillation loss
    """

    def __init__(self, loss_type):
        super(DistillLoss, self).__init__()
        self.loss_type = loss_type
        if loss_type == "cos+l1":
            self.l1_loss = nn.SmoothL1Loss()
            self.cos_loss = nn.CosineEmbeddingLoss()
            self.cos_target = torch.tensor(1.0, requires_grad=False)
        elif loss_type == "cos":
            self.cos_loss = nn.CosineEmbeddingLoss()
            self.cos_target = torch.tensor(1.0, requires_grad=False)
        elif loss_type == "l2":
            self.l2_loss = nn.MSELoss()
        else:
            raise NotImplementedError

    def forward(self, tea_feas_dict, aligned_feas_dict):
        out = {}
        for tea_name in tea_feas_dict.keys():
            tea_fea_list = tea_feas_dict[tea_name]
            stu_fea_list = aligned_feas_dict[tea_name]
            # 4 levels
            for i, (tea_fea, stu_fea) in enumerate(zip(tea_fea_list, stu_fea_list)):
                assert len(tea_fea.shape) == len(stu_fea.shape)

                # Resize to match the larger resolution between teacher and student
                if tea_fea.shape[2:] != stu_fea.shape[2:]:
                    if tea_fea.shape[2] > stu_fea.shape[2]:
                        stu_fea = F.interpolate(
                            stu_fea,
                            size=tea_fea.shape[2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    else:
                        tea_fea = F.interpolate(
                            tea_fea,
                            size=stu_fea.shape[2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                assert tea_fea.shape == stu_fea.shape

                if self.loss_type == "cos+l1":
                    l1_loss = self.l1_loss(stu_fea, tea_fea)
                    target = self.cos_target.repeat(stu_fea.shape[0]).to(stu_fea.device)
                    cos_loss = self.cos_loss(
                        stu_fea.flatten(1), tea_fea.flatten(1), target
                    )
                    out[tea_name + "_level" + str(i + 1)] = (
                        0.9 * cos_loss + 0.1 * l1_loss
                    )
                elif self.loss_type == "cos":
                    target = self.cos_target.repeat(stu_fea.shape[0]).to(stu_fea.device)
                    out[tea_name + "_level" + str(i + 1)] = self.cos_loss(
                        stu_fea.flatten(1), tea_fea.flatten(1), target
                    )
                elif self.loss_type == "l2":
                    out[tea_name + "_level" + str(i + 1)] = self.l2_loss(
                        stu_fea, tea_fea
                    )

        # Sum up all KD losses
        out["total"] = torch.sum(torch.stack([out[t] for t in out.keys()]))

        return out
