import numpy as np
import torch


class DepthMeter(object):

    def __init__(self):
        self.total_rmses = 0.0
        self.n_valid = 0.0
        self.max_depth = 80.0
        self.min_depth = 0.0

    @torch.no_grad()
    def update(self, pred, gt):
        pred, gt = pred.squeeze(), gt.squeeze()

        # Determine valid mask
        mask = torch.logical_and(gt < self.max_depth, gt > self.min_depth)
        self.n_valid += mask.float().sum().item()  # Valid pixels per image

        gt[gt <= 0] = 1e-9
        pred[pred <= 0] = 1e-9

        rmse_tmp = torch.pow(gt[mask] - pred[mask], 2)
        self.total_rmses += rmse_tmp.sum().item()

    def reset(self):
        self.total_rmses = 0.0
        self.n_valid = 0.0

    def get_score(self):
        eval_result = {"RMSE": np.sqrt(self.total_rmses / self.n_valid)}

        return eval_result
