import torch
import torch.nn as nn

from models.build_models import build_model
from losses import DistillLoss


class MTMT_Distiller(nn.Module):
    """Multi-Task Multi-Teacher Distiller"""

    def __init__(
        self,
        img_size: tuple,
        tea_configs: dict,
        stu_config: dict,
        loss_type: str = "cos+l1",
        task_out: bool = False,
        stu_criterion: nn.Module = None,
        dataname: str = None,
        tasks: list = None,
    ) -> None:
        super().__init__()
        self.kd_criterion = DistillLoss(loss_type)
        self.task_out = task_out
        if task_out:
            assert stu_criterion is not None
            assert tasks is not None
            self.stu_criterion = stu_criterion
            self.tasks = tasks

        # Build teachers
        self.teachers = nn.ModuleDict()
        tea_dims = {}
        for tea_name, tea_config in tea_configs.items():
            tea_model = build_model(arch="backbone", img_size=img_size, backbone_args=tea_config)
            # Freeze teacher
            tea_model.eval()
            for param in tea_model.parameters():
                param.requires_grad = False

            self.teachers[tea_name] = tea_model
            if hasattr(tea_model.model, "embed_dim"):
                tea_dims[tea_name] = tea_model.model.embed_dim
            elif hasattr(tea_model.model, "config"):
                tea_dims[tea_name] = tea_model.model.config.hidden_size
            else:
                raise RuntimeError

        # Build student
        assert stu_config["backbone"]["backbone_type"] == "sak"
        stu_config["backbone"]["tea_dims"] = tea_dims
        if task_out:
            # multi-task model
            self.student = build_model(
                arch="mt",
                img_size=img_size,
                backbone_args=stu_config["backbone"],
                dataname=dataname,
                tasks=tasks,
            )
        else:
            # backbone only
            self.student = build_model(
                arch="backbone",
                img_size=img_size,
                backbone_args=stu_config["backbone"],
            )

    def forward(self, batch):
        self.student.train()

        if isinstance(batch, dict):
            images = batch["image"]
        else:
            images = batch

        tea_feas_dict = {}
        with torch.no_grad():
            for tea_name, tea_model in self.teachers.items():
                tea_feas_dict[tea_name] = tea_model(images)

        if self.task_out:
            aligned_feas_dict, stu_outputs = self.student(images)
            kd_loss_dict = self.kd_criterion(tea_feas_dict, aligned_feas_dict)
            task_loss_dict = self.stu_criterion(stu_outputs, batch, self.tasks)

            return kd_loss_dict, task_loss_dict
        else:
            aligned_feas_dict = self.student(images)
            kd_loss_dict = self.kd_criterion(tea_feas_dict, aligned_feas_dict)

            return kd_loss_dict

    def forward_val(self, batch):
        self.student.eval()

        if isinstance(batch, dict):
            images = batch["image"]
        else:
            images = batch

        tea_feas_dict = {}
        with torch.no_grad():
            for tea_name, tea_model in self.teachers.items():
                tea_feas_dict[tea_name] = tea_model(images)

        if self.task_out:
            aligned_feas_dict, stu_outputs = self.student(images)
            kd_loss_dict = self.kd_criterion(tea_feas_dict, aligned_feas_dict)

            return kd_loss_dict, stu_outputs
        else:
            aligned_feas_dict = self.student(images)
            kd_loss_dict = self.kd_criterion(tea_feas_dict, aligned_feas_dict)

            return kd_loss_dict
