from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Core Losses
# ---------------------------
class WeightedCrossEntropy(nn.Module):
    """
    Weighted CE for class imbalance with optional label smoothing.
    weights: Tensor[10] with per-class weights (higher for rare classes).
    label_smoothing: float in [0, 1] to prevent overconfident predictions.
    """
    def __init__(self, class_weights: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("w", class_weights if class_weights is not None else None)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # targets: int64 labels in [0..9]
        return F.cross_entropy(logits, targets, weight=self.w, label_smoothing=self.label_smoothing)


class EarthMoversDistance(nn.Module):
    """
    EMD between predicted and target categorical distributions using CDF L2.
    For a single-label target, we one-hot then take the cumulative sum.
    """
    def __init__(self, num_classes: int = 10, reduction: str = "mean"):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, K]; targets: int64 [B] or one-hot [B, K]
        B, K = logits.shape
        probs = F.softmax(logits, dim=-1)

        if targets.dim() == 1:
            t = torch.zeros_like(probs).scatter_(1, targets.view(-1, 1), 1.0)
        else:
            t = targets

        Fp = torch.cumsum(probs, dim=1)
        Ft = torch.cumsum(t, dim=1)
        emd = torch.mean((Fp - Ft) ** 2, dim=1)  # per-sample
        if self.reduction == "mean":
            return emd.mean()
        elif self.reduction == "sum":
            return emd.sum()
        return emd  # 'none'


class BCEWithLogitsLossSafe(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, targets.float())


class MSE(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


# ---------------------------
# Composite Objective
# L = CE + α*EMD + β_edge*BCE + β_center*MSE + λ_W * ||θ||^2 (optional)
# Note: Prefer AdamW weight_decay for L2; the explicit term is optional.
# ---------------------------
class CompositeLoss(nn.Module):
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        alpha_emd: float = 0.7,
        beta_edge: float = 0.05,
        beta_center: float = 0.1,
        l2_lambda: float = 0.0,  # usually keep 0 and rely on AdamW(weight_decay)
        label_smoothing: float = 0.0,  # label smoothing for CE loss
    ):
        super().__init__()
        self.ce = WeightedCrossEntropy(class_weights, label_smoothing=label_smoothing)
        self.emd = EarthMoversDistance(num_classes=10, reduction="mean")
        self.edge = BCEWithLogitsLossSafe()
        self.center = MSE()
        self.alpha_emd = alpha_emd
        self.beta_edge = beta_edge
        self.beta_center = beta_center
        self.l2_lambda = l2_lambda

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        model_params: Optional[Sequence[torch.nn.Parameter]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        outputs: {"logits": [B,10], "edge_logit": [B], "center":[B,2]}
        targets: {"grade": [B], "edge": [B], "center":[B,2]}
        """
        logits = outputs["logits"]
        edge_logit = outputs["edge_logit"]
        center_pred = outputs["center"]

        grade = targets["grade"].long()      # [B]
        edge_t = targets["edge"].float()     # [B] (0/1)
        center_t = targets["center"].float() # [B,2]

        loss_ce = self.ce(logits, grade)
        loss_emd = self.emd(logits, grade)
        loss_edge = self.edge(edge_logit, edge_t)
        loss_center = self.center(center_pred, center_t)

        loss = loss_ce + self.alpha_emd * loss_emd \
               + self.beta_edge * loss_edge \
               + self.beta_center * loss_center

        l2 = torch.tensor(0.0, device=logits.device)
        if self.l2_lambda > 0.0 and model_params is not None:
            l2 = sum((p.pow(2).sum() for p in model_params if p.requires_grad)) * self.l2_lambda
            loss = loss + l2

        return {
            "loss": loss,
            "loss_ce": loss_ce.detach(),
            "loss_emd": loss_emd.detach(),
            "loss_edge": loss_edge.detach(),
            "loss_center": loss_center.detach(),
            "loss_l2": l2.detach(),
        }


# ---------------------------
# Metrics (for logging)
# ---------------------------
@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pred = torch.argmax(logits, dim=1)
    return (pred == targets).float().mean()

@torch.no_grad()
def mae_expected_grade(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    classes = torch.arange(logits.size(1), device=logits.device, dtype=probs.dtype) + 1.0
    expected = (probs * classes.unsqueeze(0)).sum(dim=1)  # [B]
    return (expected - (targets.float() + 1.0)).abs().mean()  # targets 0..9 -> grades 1..10

@torch.no_grad()
def quadratic_weighted_kappa(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """
    Differentiable-ish QWK estimator for logging (CPU-friendly).
    """
    # Confusion matrix
    pred = torch.argmax(logits, dim=1)
    O = torch.zeros((num_classes, num_classes), device=logits.device)
    for i in range(num_classes):
        for j in range(num_classes):
            O[i, j] = torch.sum((targets == i) & (pred == j)).float()

    # Expected matrix E from histogram outer product
    t_hist = torch.bincount(targets, minlength=num_classes).float()
    p_hist = torch.bincount(pred, minlength=num_classes).float()
    E = torch.ger(t_hist, p_hist) / max(1.0, torch.sum(t_hist))

    # Quadratic weights
    W = torch.zeros((num_classes, num_classes), device=logits.device)
    for i in range(num_classes):
        for j in range(num_classes):
            W[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum().clamp_min(1e-9)
    kappa = 1.0 - num / den
    return kappa


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int = 10) -> dict:
    """
    Compute all evaluation metrics.

    Args:
        predictions: Predicted class indices [B]
        targets: True class indices [B]
        num_classes: Number of classes (default 10 for grades 1-10)

    Returns:
        Dictionary with accuracy, mae, and qwk
    """
    # Convert predictions to one-hot for compatibility with existing functions
    logits = torch.zeros(len(predictions), num_classes)
    logits.scatter_(1, predictions.unsqueeze(1), 1.0)

    acc = accuracy_top1(logits, targets).item()
    mae = mae_expected_grade(logits, targets).item()
    qwk = quadratic_weighted_kappa(logits, targets, num_classes).item()

    return {
        'accuracy': acc,
        'mae': mae,
        'qwk': qwk
    }
