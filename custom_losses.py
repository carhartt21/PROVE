"""Custom loss wrappers for PROVE."""

from mmseg.registry import MODELS
from mmseg.models.losses import BoundaryLoss
import torch
import torch.nn.functional as F


@MODELS.register_module()
class BoundaryLossIgnoreWeight(BoundaryLoss):
    """BoundaryLoss wrapper that ignores per-pixel weight inputs."""

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        return super().forward(pred, target)


@MODELS.register_module()
class SegBoundaryLoss(torch.nn.Module):
    """Boundary loss derived from segmentation logits and labels.

    This computes a binary boundary target from labels and a differentiable
    boundary prediction from segmentation probabilities.
    """

    def __init__(self, loss_weight: float = 1.0, ignore_index: int = 255, loss_name: str = 'loss_boundary'):
        super().__init__()
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.loss_name_ = loss_name

    def forward(self, seg_logits: torch.Tensor, seg_gt: torch.Tensor, **kwargs) -> torch.Tensor:
        # seg_logits: (N, C, H, W), seg_gt: (N, H, W)
        probs = torch.softmax(seg_logits, dim=1)
        prob_max = probs.max(dim=1, keepdim=True).values

        # Differentiable boundary estimate from local variation
        max_pool = F.max_pool2d(prob_max, kernel_size=3, stride=1, padding=1)
        min_pool = -F.max_pool2d(-prob_max, kernel_size=3, stride=1, padding=1)
        boundary_pred = (max_pool - min_pool).clamp(0, 1)

        # Boundary target from label changes in 4-neighborhood
        gt = seg_gt
        if gt.ndim == 4:
            gt = gt.squeeze(1)
        gt = gt.long()
        valid = gt != self.ignore_index

        def shift(t, dx, dy):
            return F.pad(t, (1, 1, 1, 1), mode='replicate')[:, 1 + dy:t.shape[1] + 1 + dy, 1 + dx:t.shape[2] + 1 + dx]

        gt_up = shift(gt, 0, -1)
        gt_down = shift(gt, 0, 1)
        gt_left = shift(gt, -1, 0)
        gt_right = shift(gt, 1, 0)

        boundary_gt = (gt != gt_up) | (gt != gt_down) | (gt != gt_left) | (gt != gt_right)
        boundary_gt = boundary_gt & valid
        boundary_gt = boundary_gt.unsqueeze(1).float()

        valid_mask = valid.unsqueeze(1)
        if valid_mask.any():
            loss = F.binary_cross_entropy(boundary_pred[valid_mask], boundary_gt[valid_mask], reduction='mean')
        else:
            loss = boundary_pred.sum() * 0.0

        return self.loss_weight * loss

    @property
    def loss_name(self):
        return self.loss_name_
