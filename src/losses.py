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


class CORALLoss(nn.Module):
    """
    CORAL: Consistent Rank Logits for Ordinal Regression

    Instead of treating grades as independent classes, CORAL learns
    cumulative thresholds: P(grade > k) for k=1,2,...,9

    For grade 8 (0-indexed as 7):
    - P(grade > 0) = 1  (yes, grade is > 1)
    - P(grade > 1) = 1  (yes, grade is > 2)
    - ...
    - P(grade > 7) = 0  (no, grade is not > 8)
    - P(grade > 8) = 0  (no, grade is not > 9)

    Args:
        num_classes: Number of ordinal classes (10 for PSA grades 1-10)
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, num_classes: int = 10, reduction: str = "mean"):
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1  # 9 thresholds for 10 classes
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, num_classes-1] cumulative binary logits
            targets: [B] integer labels in [0, num_classes-1]

        Returns:
            CORAL loss (scalar if reduction='mean')
        """
        # Create cumulative labels: "is grade > k?"
        # For target grade 7 (8 in 1-10 scale):
        # cum_labels = [1, 1, 1, 1, 1, 1, 1, 0, 0]
        #              >0 >1 >2 >3 >4 >5 >6 >7 >8

        batch_size = targets.size(0)
        cum_labels = torch.zeros(batch_size, self.num_thresholds,
                                  dtype=torch.float32, device=targets.device)

        for i in range(self.num_thresholds):
            cum_labels[:, i] = (targets > i).float()

        # Binary cross-entropy for each threshold
        loss = F.binary_cross_entropy_with_logits(logits, cum_labels, reduction='none')

        # Sum across thresholds, then reduce across batch
        loss = loss.sum(dim=1)  # [B]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def create_soft_ordinal_labels(targets: torch.Tensor, num_classes: int = 10,
                                 sigma: float = 1.0) -> torch.Tensor:
    """
    Create soft labels for ordinal classification using Gaussian kernel.

    Instead of hard one-hot labels, blend adjacent grades with a Gaussian.
    For grade 8 with sigma=1.0:
    - Hard: [0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0]
    - Soft: [0.01, 0.01, 0.03, 0.05, 0.09, 0.15, 0.24, 0.24, 0.15, 0.09]

    Args:
        targets: [B] integer labels in [0, num_classes-1]
        num_classes: Number of ordinal classes
        sigma: Gaussian kernel width (higher = more smoothing)

    Returns:
        [B, num_classes] soft label distributions
    """
    batch_size = targets.size(0)
    soft_labels = torch.zeros(batch_size, num_classes, device=targets.device)

    for i in range(batch_size):
        target_class = targets[i].item()
        for j in range(num_classes):
            # Gaussian centered at target class
            distance = (j - target_class) ** 2
            soft_labels[i, j] = torch.exp(torch.tensor(-distance / (2 * sigma ** 2)))

        # Normalize to sum to 1
        soft_labels[i] /= soft_labels[i].sum()

    return soft_labels


def coral_logits_to_predictions(cumulative_logits: torch.Tensor) -> torch.Tensor:
    """
    Convert CORAL cumulative logits to predicted class indices.

    Args:
        cumulative_logits: [B, num_classes-1] cumulative binary logits

    Returns:
        [B] predicted class indices in [0, num_classes-1]
    """
    # Get cumulative probabilities P(y > k)
    cum_probs = torch.sigmoid(cumulative_logits)  # [B, num_classes-1]

    # Predicted class = number of thresholds exceeded
    # Count how many P(y > k) >= 0.5
    predictions = (cum_probs >= 0.5).sum(dim=1).long()  # [B]

    return predictions


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
        use_coral: bool = False,  # Use CORAL ordinal regression instead of CE+EMD
        num_classes: int = 10,
    ):
        super().__init__()
        self.use_coral = use_coral
        self.num_classes = num_classes

        if use_coral:
            # CORAL replaces both CE and EMD
            self.coral = CORALLoss(num_classes=num_classes, reduction="mean")
        else:
            # Standard losses
            self.ce = WeightedCrossEntropy(class_weights, label_smoothing=label_smoothing)
            self.emd = EarthMoversDistance(num_classes=num_classes, reduction="mean")

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
        outputs: {"logits": [B,10] or [B,9] if CORAL, "edge_logit": [B], "center":[B,2]}
        targets: {"grade": [B], "edge": [B], "center":[B,2]}
        """
        logits = outputs["logits"]
        edge_logit = outputs["edge_logit"]
        center_pred = outputs["center"]

        grade = targets["grade"].long()      # [B]
        edge_t = targets["edge"].float()     # [B] (0/1)
        center_t = targets["center"].float() # [B,2]

        if self.use_coral:
            # CORAL loss for ordinal classification
            loss_grade = self.coral(logits, grade)
            loss_ce = torch.tensor(0.0, device=logits.device)
            loss_emd = torch.tensor(0.0, device=logits.device)
        else:
            # Standard CE + EMD
            loss_ce = self.ce(logits, grade)
            loss_emd = self.emd(logits, grade)
            loss_grade = loss_ce + self.alpha_emd * loss_emd

        loss_edge = self.edge(edge_logit, edge_t)
        loss_center = self.center(center_pred, center_t)

        loss = loss_grade + self.beta_edge * loss_edge + self.beta_center * loss_center

        l2 = torch.tensor(0.0, device=logits.device)
        if self.l2_lambda > 0.0 and model_params is not None:
            l2 = sum((p.pow(2).sum() for p in model_params if p.requires_grad)) * self.l2_lambda
            loss = loss + l2

        return {
            "loss": loss,
            "loss_ce": loss_ce.detach(),
            "loss_emd": loss_emd.detach(),
            "loss_grade": loss_grade.detach() if self.use_coral else loss_ce.detach(),
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
