import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import Module, MSELoss, L1Loss

from .bcpd import square_distance

if __name__=='__main__':
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lib import pointnet2_utils as pointutils
else:

    from .lib import pointnet2_utils as pointutils

class PermLoss(nn.Module):
    """
    Cross entropy loss between two permutations.
    cal avg loss, used by RGM.
    """
    def __init__(self):
        super(PermLoss, self).__init__()

    def forward(self, pred_perm, gt_perm, pred_ns, gt_ns):
        batch_num = pred_perm.shape[0]

        pred_perm = pred_perm.to(dtype=torch.float32)

        # In case of dustbins, M+1,N+1 should not be considered

        # assert torch.all((pred_perm[:,:-1,:-1] >= 0) * (pred_perm[:,:-1,:-1] <= 1))
        assert torch.all((pred_perm >= 0) * (pred_perm <= 1))
        assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        assert pred_perm.shape == gt_perm.shape

        loss = torch.tensor(0.).to(pred_perm.device)
        n_sum = torch.zeros_like(loss)

        for b in range(batch_num):
            loss += F.binary_cross_entropy(
                pred_perm[b, :pred_ns[b], :gt_ns[b]],
                gt_perm[b, :pred_ns[b], :gt_ns[b]],
                reduction='sum')

            n_sum += pred_ns[b].to(n_sum.dtype).to(pred_perm.device)

        return loss / n_sum


class PermLossOcclusion(nn.Module):
    """
    Cross entropy loss between two permutations.
    cal avg loss. It contrains extra term 'doubling' the cost 
    non-matches classification.
    """
    def __init__(self):
        super(PermLossOcclusion, self).__init__()

    def forward(self, pred_perm, gt_perm, pred_ns, gt_ns):
        batch_num = pred_perm.shape[0]

        if not torch.all((pred_perm >= 0) * (pred_perm <= 1)):
            print('max min', torch.max(pred_perm), torch.min(pred_perm))
        assert torch.all((pred_perm >= 0) * (pred_perm <= 1))
        assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        assert pred_perm.shape == gt_perm.shape

        loss = torch.tensor(0.).to(pred_perm.device)
        n_sum = torch.zeros_like(loss)

        pred_match = torch.clamp(torch.sum(pred_perm, dim=-1), min=0, max=1)

        gt_match = torch.sum(gt_perm, dim=-1)

        for b in range(batch_num):
            loss += F.binary_cross_entropy(
                pred_perm[b, :pred_ns[b], :gt_ns[b]],
                gt_perm[b, :pred_ns[b], :gt_ns[b]],
                reduction='sum')

            # Consider twice the cost of correctly matching source into target
            # this is due to improve the model on matching complete to partial
            # cases
            loss += F.binary_cross_entropy(
                pred_match[b, :pred_ns[b]],
                gt_match[b, :pred_ns[b]],
                reduction='sum')

            n_sum += pred_ns[b].to(n_sum.dtype).to(pred_perm.device)

        return loss / n_sum

