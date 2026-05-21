import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        bce_loss = binary_cross_entropy_with_logits(pred, target, reduction="none")
        pred_s = torch.sigmoid(pred)
        p_t = pred_s * target + (1 - pred_s) * (1 - target)
        if self.alpha != 0.5:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        else:
            alpha_t = 1
        loss = alpha_t * torch.pow(1 - p_t, self.gamma) * bce_loss
        return loss.mean()


class HMapLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(HMapLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        diff = torch.abs(target - torch.sigmoid(pred))
        pos_mask = target >= 0.004
        neg_mask = target < 0.004

        pos_loss = -torch.pow(diff[pos_mask], self.gamma) * torch.log(1 - diff[pos_mask] + 1e-14) * target[pos_mask]
        neg_loss = -torch.pow(diff[neg_mask], self.gamma) * torch.log(1 - diff[neg_mask] + 1e-14)
        pos_ratio = len(pos_loss) / (len(pos_loss) + len(neg_loss))
        alpha = max(self.alpha, pos_ratio)

        loss = alpha * torch.mean(pos_loss) + (1 - alpha) * torch.mean(neg_loss)
        return loss, torch.mean(pos_loss), torch.mean(neg_loss)
