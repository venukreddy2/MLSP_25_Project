import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    def __init__(self, config, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):
        super(MultiTaskLoss, self).__init__()
        self.config = config
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights
    
    def forward(self, pred, gt):
        out = {task: self.loss_ft[task](pred[task], gt[task]) for task in self.tasks}
        out['total'] = torch.sum(torch.stack([self.loss_weights[t] * out[t] for t in self.tasks]))

        return out
    
class MultiTaskDistillationLoss(nn.Module):
    def __init__(self, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):
        super().__init__()
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    
    def forward(self, pred, gt):
        total = 0.
        out = {}

        # Losses initial task predictions (deepsup)
        for task in self.tasks:
            loss_ = self.loss_ft[task](pred[task], gt[task])
            out['initial_%s' %(task)] = loss_
            total += self.loss_weights[task] * loss_

        # Losses at output  
        for task in self.tasks:
            pred_, gt_ = pred[task], gt[task]
            loss_ = self.loss_ft[task](pred_, gt_)
            out[task] = loss_
            total += self.loss_weights[task] * loss_

        out['total'] = total

        return out
