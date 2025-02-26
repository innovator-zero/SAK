import os

import numpy as np
import torch
import torch.nn.functional as F


class RunningMeter(object):

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_dir(directory):
    """
    Create required directory if it does not exist
    """

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def create_results_dir(results_dir, exp_name):
    """
    Create required results directory if it does not exist
    :param str results_dir: Directory to create subdirectory in
    :param str exp_name: Name of experiment to be used in the directory created
    :return: Path of experiment directory and checkpoint directory
    """

    exp_dir = os.path.join(results_dir, exp_name)
    create_dir(results_dir)
    create_dir(exp_dir)

    return exp_dir


def create_pred_dir(results_dir, exp_name, tasks):
    """
    Create required prediction directory if it does not exist
    :param str results_dir: Directory to create subdirectory in
    :param str exp_name: Name of experiment to be used in the directory created
    :param list tasks: List of tasks
    :return: Path of checkpoint directory and prediction dictionary
    """

    exp_dir = os.path.join(results_dir, exp_name)
    pred_dir = os.path.join(exp_dir, "predictions")
    create_dir(pred_dir)

    for task in tasks:
        task_dir = os.path.join(pred_dir, task)
        create_dir(task_dir)
        if task == "edge":
            create_dir(os.path.join(task_dir, "img"))

    return exp_dir, pred_dir


def get_loss_metric(loss_meter, tasks, prefix):
    """
    Get loss statistics
    :param dict loss_meter: Loss meter
    :param str tasks: List of tasks
    :param str prefix: Prefix for the loss, train or val
    :return: Loss statistics
    """

    statistics = {prefix + "/" + "loss_sum": 0.0}

    for task in tasks:
        statistics[prefix + "/" + "loss_sum"] += loss_meter[task].avg
        statistics[prefix + "/" + task] = loss_meter[task].avg
        loss_meter[task].reset()

    return statistics


def to_cuda(batch):
    """
    Move batch to GPU
    :param dict batch: Input batch
    :return: Batch on GPU
    """

    if type(batch) is dict:
        out = {}
        for k, v in batch.items():
            if k == "meta":
                out[k] = v
            else:
                out[k] = to_cuda(v)
        return out
    elif type(batch) is torch.Tensor:
        return batch.cuda(non_blocking=True)
    elif type(batch) is list:
        return [to_cuda(v) for v in batch]
    else:
        return batch


def get_output(output, task):
    """
    Get output prediction in the required range and format
    :param Tensor output: Output tensor
    :param str task: Task
    :return: Tensor
    """

    if task == "normals":
        output = output.permute(0, 2, 3, 1)
        output = (F.normalize(output, p=2, dim=3) + 1.0) * 255 / 2.0

    elif task in {"semseg", "human_parts"}:
        output = output.permute(0, 2, 3, 1)
        _, output = torch.max(output, dim=3)

    elif task in {"edge"}:
        output = output.permute(0, 2, 3, 1)
        output = torch.sigmoid(output).squeeze(-1) * 255

    elif task in {"sal"}:
        output = output.permute(0, 2, 3, 1)
        output = F.softmax(output, dim=3)[:, :, :, 1] * 255

    elif task in {"depth"}:
        output.clamp_(min=0.0)
        output = output.permute(0, 2, 3, 1).squeeze(-1)

    else:
        raise NotImplementedError

    return output


def global_print(s):
    if "RANK" in os.environ:
        if int(os.environ["RANK"]) == 0:
            print(s)
    else:
        print(s)


def set_seed(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
