"""
Binary Survival Classifier for Trajectory Prediction
=====================================================

This module implements a neural network classifier that predicts whether molecular
beam trajectories will survive passage through an electrostatic lens or collide with
apertures. It uses advanced techniques to handle severe class imbalance (typical
survival rates: 5-30%) and improve decision boundary accuracy.

Key Features
------------
1. **Focal Loss**: Addresses class imbalance by down-weighting easy examples
2. **Hard-Negative Mining**: Focuses training on difficult boundary cases
3. **Temperature Calibration**: Produces well-calibrated probability estimates
4. **Early Stopping**: Prevents overfitting via validation monitoring

Architecture
------------
- Input: Augmented features (12-D: raw + physics features)
- Hidden layers: 3-4 layers with ReLU activations and dropout
- Output: Single sigmoid probability (survival probability)
- Temperature parameter: Learned post-training for calibration

Training Strategy
-----------------
The classifier uses a sophisticated training pipeline:
1. **Focal loss** to handle 70-95% negative (collision) examples
2. **Dynamic hard-negative mining** each epoch to focus on decision boundary
3. **AdamW optimizer** with weight decay for regularization
4. **Gradient clipping** to stabilize training
5. **Temperature scaling** on validation set for probability calibration

Performance Expectations
------------------------
On well-separated data:
- Accuracy: 95-99%
- ROC-AUC: 0.97-0.995
- Precision/Recall: 90-98%

On difficult cases (overlapping distributions):
- Accuracy: 85-92%
- ROC-AUC: 0.85-0.93
- Calibration important for uncertainty quantification

Usage Example
-------------
>>> from classifier_code import train_classifier, Standardizer, evaluate_classifier
>>> from data_creation import build_raw_dataset_full
>>> from physics_augment import augment_with_physics
>>>
>>> # Load training data
>>> data_train = build_raw_dataset_full(...)
>>> X_train = augment_with_physics(data_train.X, R=0.022, L=0.6, ...)
>>>
>>> # Standardize features
>>> std = Standardizer(X_train)
>>> mu, sd = std.mu, std.sd
>>>
>>> # Train classifier
>>> model = train_classifier(
>>>     data_tr_full=data_train,
>>>     data_va_full=data_valid,
>>>     mu=mu, sd=sd,
>>>     device='cuda',
>>>     epochs=30,
>>>     gamma=2.0,  # Focal loss strength
>>>     patience=5
>>> )
>>>
>>> # Calibrate temperature
>>> T_scale = calibrate_temperature(model, X_valid, y_valid, mu, sd, device='cuda')
>>> print(f"Temperature scale: {T_scale:.3f}")
>>>
>>> # Evaluate
>>> metrics = evaluate_classifier(model, data_test, mu, sd, device='cuda')
>>> print(f"Test AUC: {metrics['auc']:.3f}")
"""

# ======================== Survival classifier (latest) ========================
import copy
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import tqdm

__all__ = [
    "Standardizer",
    "SurvivalNet",
    "FocalWithLogits",
    "hard_negative_indices",
    "youden_threshold",
    "train_classifier",
    "calibrate_temperature",
    "evaluate_classifier",
]


# ---------- Normalizer ----------
class Standardizer:
    """
    Feature standardizer for neural network input normalization.

    Computes mean and standard deviation from training data and stores them
    for consistent normalization across train/validation/test sets. Ensures
    zero mean and unit variance, which improves neural network training
    convergence and stability.

    Attributes
    ----------
    mu : ndarray of shape (n_features,), dtype=float32
        Mean of each feature computed from training data
    sd : ndarray of shape (n_features,), dtype=float32
        Standard deviation of each feature. Features with zero variance
        are assigned sd=1.0 to avoid division by zero.

    Notes
    -----
    - Always fit on training data only, then apply to all sets
    - Handles constant features (sd=0) gracefully by setting sd=1
    - Stores parameters as float32 for PyTorch compatibility

    Examples
    --------
    >>> # Fit on training data
    >>> X_train = np.random.randn(1000, 12).astype(np.float32)
    >>> std = Standardizer(X_train)
    >>>
    >>> # Transform training data
    >>> X_train_norm = (X_train - std.mu) / std.sd
    >>> assert np.allclose(X_train_norm.mean(axis=0), 0, atol=1e-5)
    >>> assert np.allclose(X_train_norm.std(axis=0), 1, atol=1e-5)
    >>>
    >>> # Apply to test data (using training statistics)
    >>> X_test = np.random.randn(200, 12).astype(np.float32)
    >>> X_test_norm = (X_test - std.mu) / std.sd
    >>>
    >>> # Convert to PyTorch tensors
    >>> mu_t, sd_t = std.torch(device='cuda')
    >>> X_t = torch.from_numpy(X_train).cuda()
    >>> X_norm_t = (X_t - mu_t) / sd_t
    """

    def __init__(self, X: npt.NDArray[np.float32]):
        """
        Compute standardization parameters from training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix. Should be the full training set
            to get accurate statistics.
        """
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0.0] = 1.0  # Avoid division by zero for constant features
        self.mu = mu.astype(np.float32)
        self.sd = sd.astype(np.float32)

    def torch(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert standardization parameters to PyTorch tensors.

        Parameters
        ----------
        device : str
            Target device: 'cpu', 'cuda', 'cuda:0', etc.

        Returns
        -------
        mu_tensor : torch.Tensor
            Mean values as PyTorch tensor on specified device
        sd_tensor : torch.Tensor
            Standard deviation values as PyTorch tensor on specified device
        """
        return torch.from_numpy(self.mu).float().to(device), torch.from_numpy(
            self.sd
        ).float().to(device)


# ---------- Model ----------
class SurvivalNet(nn.Module):
    """
    Multi-layer perceptron for binary trajectory survival classification.

    This is the original version using ReLU activations and dropout, suitable
    for both raw and physics-augmented feature inputs. The network predicts the
    probability that a trajectory survives passage through the lens without
    colliding with apertures.

    Architecture
    ------------
    - Backbone: Linear → ReLU → Dropout × len(hidden)
    - Head: Linear → 1 logit
    - Output: Single sigmoid probability (apply after forward pass)
    - Temperature parameter T: Learned post-training for calibration

    Parameters
    ----------
    d_in : int, default=6
        Input feature dimension. Use 6 for raw or 12 for augmented inputs.
    hidden : tuple of int, default=(256, 128, 64)
        Sizes of hidden layers. Each element creates one Linear-ReLU-Dropout
        block. Typical: (256, 128, 64) for ~100k samples.
    pdrop : float, default=0.05
        Dropout probability applied after each hidden layer.

    Attributes
    ----------
    backbone : nn.Sequential
        Feature extraction layers (Linear → ReLU → Dropout)
    head : nn.Linear
        Final layer mapping to a single logit
    T : nn.Parameter
        Log-temperature parameter for post-training calibration

    Notes
    -----
    - During training: Returns raw logits (apply_temp=False)
    - During evaluation: Returns temperature-scaled logits (apply_temp=True)
    - Temperature scaling: z_scaled = z_raw / exp(T)
    - exp(T) > 1 softens probabilities (less confident)
    - exp(T) < 1 sharpens probabilities (more confident)

    Examples
    --------
    >>> model = SurvivalNet(d_in=12, hidden=(256, 128, 64), pdrop=0.05)
    >>> x = torch.randn(32, 12)
    >>> logits = model(x, apply_temp=False)
    >>> probs = torch.sigmoid(logits)
    >>> model.eval()
    >>> with torch.no_grad():
    >>>     logits_cal = model(x, apply_temp=True)
    >>>     probs_cal = torch.sigmoid(logits_cal)
    >>> print(f"Temperature scale: {torch.exp(model.T).item():.3f}")
    """

    def __init__(
        self,
        d_in: int = 6,
        hidden: tuple[int, ...] = (256, 128, 64),
        pdrop: float = 0.05,
    ):
        super().__init__()
        layers = []
        d = d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(inplace=True), nn.Dropout(pdrop)]
            d = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)
        self.T = nn.Parameter(torch.zeros(()))  # log-temperature for calibration

    def forward(self, x: torch.Tensor, apply_temp: bool = True) -> torch.Tensor:
        """
        Forward pass through the classifier.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, d_in)
            Standardized input features.
        apply_temp : bool, default=True
            Whether to apply the learned temperature scaling.

        Returns
        -------
        logits : torch.Tensor of shape (batch_size,)
            Raw or temperature-scaled logits. Apply sigmoid to obtain probabilities.
        """
        z = self.head(self.backbone(x)).squeeze(-1)
        return z / torch.exp(self.T) if apply_temp else z


class InputDropout(nn.Module):
    """
    Feature dropout layer for input regularization.

    Randomly zeros out a fraction of input features during training. This technique
    improves generalization for tabular neural networks by preventing the model from
    over-relying on specific correlated features (common in physics-augmented datasets).

    Each forward pass applies a Bernoulli mask to the input tensor `x`, where each feature
    has probability `p` of being zeroed. The remaining active features are scaled by
    `1 / (1 - p)` to preserve the expected feature magnitude.

    Parameters
    ----------
    p : float, default=0.05
        Dropout probability for input features. A good range is 0.03–0.10 for
        well-behaved features, or up to 0.15–0.20 if features are highly correlated.

    Notes
    -----
    - Only active during training (`model.train()`); no effect during evaluation.
    - Can be combined with hidden-layer dropout and small Gaussian input noise.
    - Useful for physics-augmented or engineered features that encode redundant
      information (e.g., `E_x`, `E_y`, `|E|`).

    Examples
    --------
    >>> x = torch.randn(4, 12)
    >>> drop = InputDropout(p=0.05)
    >>> drop.train()
    >>> y = drop(x)
    >>> assert y.shape == x.shape
    >>> drop.eval()
    >>> y_eval = drop(x)  # identical to input (no dropout)
    """

    def __init__(self, p: float = 0.05):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random feature dropout during training.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, n_features)
            Input tensor to apply dropout to.

        Returns
        -------
        torch.Tensor
            Tensor with randomly dropped features during training, or the original
            input during evaluation.
        """
        if not self.training or self.p <= 0.0:
            return x
        mask = torch.empty_like(x).bernoulli_(1 - self.p) / (1 - self.p)
        return x * mask


class SurvivalNetLN(nn.Module):
    """
    Multi-layer perceptron for binary trajectory survival classification
    with Layer Normalization, SiLU activations, and input dropout.

    This variant of the survival classifier introduces LayerNorm and smooth
    SiLU activations, which improve convergence and stability for tabular
    physics-based features. It also supports optional InputDropout on the
    raw input features to regularize against correlated or redundant inputs.

    The model predicts the probability that a molecular trajectory survives
    passage through the electrostatic lens system (i.e., does not collide
    with apertures). It outputs a single logit per sample, which is scaled
    post-training by a learned temperature parameter for probability
    calibration.

    Architecture
    ------------
    Input → (InputDropout) → [Linear → LayerNorm → SiLU → Dropout] × N → Linear(1)

    Parameters
    ----------
    d_in : int, default=6
        Number of input features. Use 6 for raw features, or 12 for
        physics-augmented inputs.
    hidden : tuple of int, default=(256, 128, 64)
        Sizes of the hidden layers. Each element creates one Linear–LayerNorm–
        SiLU–Dropout block. A typical configuration for augmented inputs is
        (256, 128, 64).
    pdrop : float, default=0.05
        Dropout probability applied after each hidden layer.
    input_drop : float, default=0.05
        Dropout probability applied directly to the input features before
        the first layer. Acts as feature dropout and regularization against
        correlated inputs.

    Attributes
    ----------
    input_dropout : InputDropout
        Optional input feature dropout layer active only during training.
    backbone : nn.Sequential
        Sequential feature extraction network composed of Linear → LayerNorm
        → SiLU → Dropout blocks.
    head : nn.Linear
        Final linear projection to a single output logit.
    T : nn.Parameter
        Log-temperature parameter used for post-training probability
        calibration. The effective temperature scale is exp(T).

    Notes
    -----
    - During training, the model returns raw logits (apply_temp=False).
    - During evaluation, the logits are divided by exp(T) to apply the learned
      temperature scaling.
    - Temperature scaling >1 softens probabilities (less confident);
      <1 sharpens probabilities (more confident).
    - Works well with focal loss, hard-negative mining, and Brier-based
      calibration for imbalanced survival prediction tasks.

    Examples
    --------
    >>> model = SurvivalNetLN(d_in=12, hidden=(256, 128, 64),
    ...                       pdrop=0.1, input_drop=0.05).to('cuda')
    >>> x = torch.randn(32, 12, device='cuda')
    >>> logits = model(x, apply_temp=False)         # raw logits
    >>> probs = torch.sigmoid(logits)               # uncalibrated probabilities
    >>> model.eval()
    >>> logits_cal = model(x, apply_temp=True)      # temperature-scaled logits
    >>> probs_cal = torch.sigmoid(logits_cal)       # calibrated probabilities
    >>> print(f"Temperature scale: {torch.exp(model.T).item():.3f}")
    """

    def __init__(
        self,
        d_in: int = 6,
        hidden: tuple[int, ...] = (256, 128, 64),
        pdrop: float = 0.05,
        input_drop: float = 0.05,
    ):
        super().__init__()
        self.input_dropout = InputDropout(input_drop)
        layers: list[nn.Module] = []
        d = d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(pdrop)]
            d = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)
        self.T = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, apply_temp: bool = True) -> torch.Tensor:
        """
        Forward pass through the survival classifier.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, d_in)
            Standardized input features.
        apply_temp : bool, default=True
            Whether to apply the learned temperature scaling.

        Returns
        -------
        logits : torch.Tensor of shape (batch_size,)
            Raw or temperature-scaled logits. Apply `torch.sigmoid` to obtain
            probabilities.
        """
        x = self.input_dropout(x)
        z = self.head(self.backbone(x)).squeeze(-1)
        return z / torch.exp(self.T) if apply_temp else z


# ---------- Losses & mining ----------
class FocalWithLogits(nn.Module):
    """
    Focal loss for binary classification (logit input, numerically stable).

    This implementation uses BCE-with-logits for stability and applies the
    focal modulation in probability space without extra passes or redundant
    sigmoids. It is suited for severe class imbalance and pairs well with
    hard-negative mining.

    Mathematical Form
    -----------------
    FL(p_t) = - α_t (1 - p_t)^γ log(p_t)

    where:
      - p_t = p if y = 1, else (1 - p)
      - p   = σ(z), z are the logits
      - γ   : focusing parameter (gamma)
      - α_t : class weight for positives (alpha) and 1 - alpha for negatives

    Parameters
    ----------
    gamma : float, default=2.0
        Focusing strength. Larger values emphasize hard samples.
    alpha : float or None, default=None
        Positive-class weight. If None, no class weighting is applied.
        A good default is the positive fraction n_pos / (n_pos + n_neg).
    reduction : {'mean', 'sum', 'none'}, default='mean'
        Reduction to apply to the final loss.

    Returns
    -------
    loss : torch.Tensor
        Scalar if reduction='mean' or 'sum', else per-sample losses.

    Notes
    -----
    - Use with large batches when possible to expose more hard examples.
    - Combine with logit-space hard-negative mining for best results.

    Examples
    --------
    >>> focal = FocalWithLogits(gamma=2.0, alpha=0.15)
    >>> logits = torch.randn(1024, device='cuda')
    >>> targets = (torch.rand(1024, device='cuda') < 0.15).float()
    >>> loss = focal(logits, targets)
    >>> float(loss)
    """

    def __init__(
        self, gamma: float = 2.0, alpha: float | None = None, reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE with logits in a numerically stable form
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, p, 1.0 - p)
        mod = (1.0 - pt).clamp_min(1e-6).pow(self.gamma)

        loss = bce * mod
        if self.alpha is not None:
            w_pos = torch.as_tensor(
                self.alpha, device=logits.device, dtype=logits.dtype
            )
            w_neg = 1.0 - w_pos
            w = torch.where(targets > 0.5, w_pos, w_neg)
            loss = loss * w

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def hard_negative_indices_from_logits(
    neg_logits: npt.NDArray[np.float32],
    low: float = -0.6,
    high: float = 0.6,
    max_neg: int | None = None,
) -> npt.NDArray[np.int_]:
    """
    Select hard-negative examples in logit space.

    Mining in logit space is more stable than probability space because the
    sigmoid saturates at the extremes. The window [-0.6, 0.6] corresponds
    approximately to probabilities in [0.35, 0.65].

    Parameters
    ----------
    neg_logits : ndarray of shape (n_negatives,)
        Model logits for negative examples only.
    low : float, default=-0.6
        Lower bound for the hard window in logit space.
    high : float, default=0.6
        Upper bound for the hard window in logit space.
    max_neg : int or None, default=None
        Maximum number of negatives to return. If None, return all.

    Returns
    -------
    indices : ndarray of shape (n_hard,)
        Indices into the negatives array selecting hard examples.

    Notes
    -----
    - If no examples fall in [low, high], consider widening to [-1.0, 1.0].
    - Pair with focal loss and balanced batches for best effect.

    Examples
    --------
    >>> idx = hard_negative_indices_from_logits(logits_neg, low=-0.6, high=0.6, max_neg=2*npos)
    >>> X_neg_hard = X_neg[idx]
    """
    idx = np.where((neg_logits >= low) & (neg_logits <= high))[0]
    if idx.size == 0:
        return np.array([], dtype=int)
    if (max_neg is not None) and (idx.size > max_neg):
        idx = np.random.default_rng(12345).choice(idx, size=max_neg, replace=False)
    return idx


def fbeta_threshold(
    y: npt.NDArray[np.int_],
    p: npt.NDArray[np.float64],
    beta: float = 2.0,
    n: int = 1024,
) -> float:
    """
    Choose a probability threshold that maximizes F-beta.

    This threshold is useful when recall is more important than precision (beta > 1),
    or vice versa (beta < 1). It scans a grid of thresholds and returns the best one.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Binary labels in {0, 1}.
    p : ndarray of shape (n_samples,)
        Predicted probabilities in [0, 1].
    beta : float, default=2.0
        Trade-off factor. beta > 1 favors recall, beta < 1 favors precision.
    n : int, default=1024
        Number of candidate thresholds (quantiles) to scan.

    Returns
    -------
    thr : float
        Threshold that maximizes F-beta on the supplied data.

    Notes
    -----
    - Use on a validation set to avoid biasing test metrics.
    - Combine with youden_threshold for a secondary operating point.

    Examples
    --------
    >>> thr_f2 = fbeta_threshold(y_valid, p_valid, beta=2.0)
    >>> yhat = (p_valid >= thr_f2).astype(int)
    """
    thr = np.quantile(p, np.linspace(0.0, 1.0, n))
    best, bestT = -1.0, 0.5
    for t in thr:
        yhat = (p >= t).astype(int)
        tp = int(np.sum((y == 1) & (yhat == 1)))
        fp = int(np.sum((y == 0) & (yhat == 1)))
        fn = int(np.sum((y == 1) & (yhat == 0)))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f = (1 + beta * beta) * prec * rec / (beta * beta * prec + rec + 1e-12)
        if f > best:
            best, bestT = f, float(t)
    return bestT


# ---------- Metrics ----------
def _confusion(y, yhat):
    y = y.astype(int)
    yhat = yhat.astype(int)
    tp = int(np.sum((y == 1) & (yhat == 1)))
    fp = int(np.sum((y == 0) & (yhat == 1)))
    fn = int(np.sum((y == 1) & (yhat == 0)))
    tn = int(np.sum((y == 0) & (yhat == 0)))
    return tp, fp, fn, tn


def _roc_auc_from_scores(y, s, n=1024):
    y = y.astype(int)
    thr = np.quantile(s, np.linspace(0, 1, n))
    P = (y == 1).sum()
    N = (y == 0).sum()
    if P == 0 or N == 0:
        return np.nan
    tpr = []
    fpr = []
    for t in thr:
        yhat = (s >= t).astype(int)
        tp, fp, fn, tn = _confusion(y, yhat)
        tpr.append(tp / (tp + fn + 1e-12))
        fpr.append(fp / (fp + tn + 1e-12))
    order = np.argsort(fpr)
    return float(np.trapezoid(np.array(tpr)[order], np.array(fpr)[order]))


def _pr_auc_from_scores(y, s, n=1024):
    y = y.astype(int)
    thr = np.quantile(s, np.linspace(0, 1, n))
    P = (y == 1).sum()
    if P == 0:
        return np.nan
    prec = []
    rec = []
    for t in thr:
        yhat = (s >= t).astype(int)
        tp, fp, fn, tn = _confusion(y, yhat)
        prec.append(tp / (tp + fp + 1e-12))
        rec.append(tp / (tp + fn + 1e-12))
    order = np.argsort(rec)
    rec = np.array(rec)[order]
    prec = np.array(prec)[order]
    return float(np.trapezoid(prec, rec))


def youden_threshold(y, s, n=1024):
    y = y.astype(int)
    thr = np.quantile(s, np.linspace(0, 1, n))
    bestJ, bestT = -1.0, 0.5
    for t in thr:
        yhat = (s >= t).astype(int)
        tp, fp, fn, tn = _confusion(y, yhat)
        tpr = tp / (tp + fn + 1e-12)
        fpr = fp / (fp + tn + 1e-12)
        J = tpr - fpr
        if J > bestJ:
            bestJ, bestT = J, float(t)
    return bestT


# ---------- Training (focal + mining) ----------
def train_classifier(
    data_tr_full,  # RawDataFull
    data_va_full,  # RawDataFull
    mu: npt.NDArray[np.float32],
    sd: npt.NDArray[np.float32],
    device: str,
    epochs: int = 30,
    batch: int = 8192,
    lr: float = 3e-4,
    wd: float = 1e-3,
    gamma: float = 2.0,
    alpha: float | None = None,
    patience: int = 5,
    pdrop: float = 0.05,
    input_drop: float = 0.05,
    hidden: tuple[int, ...] = (256, 128, 64),
) -> nn.Module:
    """
    Train binary survival classifier with focal loss and hard-negative mining.

    This implements a sophisticated training pipeline designed for severe class
    imbalance (typical: 70-95% collisions). Combines three key techniques:

    1. **Focal Loss**: Down-weights easy examples, focuses on hard cases
    2. **Hard-Negative Mining**: Each epoch, select only difficult negatives near
       the decision boundary rather than training on millions of easy negatives
    3. **Early Stopping**: Monitors validation loss to prevent overfitting

    Training Algorithm
    ------------------
    For each epoch:
    1. Get model predictions on all training data
    2. Identify "hard" negatives: predictions in [mine_low, mine_high]
    3. Create epoch dataset: ALL positives + selected hard negatives
    4. Train on this balanced subset with focal loss
    5. Validate and check for early stopping
    6. As model improves, decision boundary shifts → different hard negatives

    Parameters
    ----------
    data_tr_full : RawDataFull
        Full training dataset with X (features) and y (labels).
        X should already be augmented with physics features if desired.
    data_va_full : RawDataFull
        Validation dataset for early stopping and hyperparameter tuning.
        Same feature space as training data.
    mu : ndarray of shape (n_features,)
        Feature means from Standardizer(data_tr_full.X).mu
    sd : ndarray of shape (n_features,)
        Feature standard deviations from Standardizer(data_tr_full.X).sd
    device : str
        PyTorch device: 'cpu', 'cuda', 'cuda:0', etc.
        Use 'cuda' if available for 10-50x speedup.
    epochs : int, default=30
        Maximum training epochs. Actual may be less due to early stopping.
        Typical: 20-50 epochs for convergence.
    batch : int, default=8192
        Batch size for training. Larger = more stable gradients, faster training.
        Typical: 4096-16384 depending on GPU memory. CPU: use 512-2048.
    lr : float, default=3e-4
        Learning rate for AdamW optimizer.
        Typical range: 1e-4 to 1e-3. Lower if training unstable.
    wd : float, default=1e-4
        Weight decay (L2 regularization) for AdamW.
        Prevents overfitting. Range: 1e-5 to 1e-3.
    gamma : float, default=2.0
        Focal loss focusing parameter. Higher = more focus on hard examples.
        gamma=0: standard BCE, gamma=2: moderate focusing, gamma=5: aggressive.
    alpha : float or None, default=None
        Focal loss class weight for positive class.
        Set to minority class frequency for balanced contributions.
        Example: If 15% survive, use alpha=0.15. None = no weighting.
    patience : int, default=5
        Early stopping patience. Stop if validation loss doesn't improve
        for this many epochs. Prevents overfitting.

    Returns
    -------
    model : SurvivalNet (nn.Module)
        Trained classifier with best validation loss weights restored.
        Ready for evaluation or calibration. Temperature parameter T
        initialized to 0 (will be optimized in calibrate_temperature).

    Notes
    -----
    - Model architecture auto-detected from data_tr_full.X.shape[1]
    - Training uses standardized features: (X - mu) / sd
    - Validation loss uses standard BCE (not focal) for fair comparison
    - Best model (lowest val loss) automatically restored at end
    - Progress bar via tqdm shows epoch progress
    - Gradient clipping (norm=1.0) for training stability

    Performance Expectations
    ------------------------
    Training time (100k samples, 12 features):
    - CPU: ~5-10 minutes per epoch
    - GPU (CUDA): ~10-30 seconds per epoch

    Typical convergence:
    - Loss stabilizes after 10-20 epochs
    - Early stopping triggers around epoch 15-25
    - Validation AUC: 0.90-0.995 depending on data separability

    Memory requirements:
    - ~200 MB per 100k samples (data + model + gradients)
    - GPU: Needs ~1-2 GB VRAM for batch=8192

    Hyperparameter Tuning Tips
    ---------------------------
    **If underfitting (low train/val accuracy):**
    - Increase model capacity: hidden=(512, 256, 128)
    - Decrease weight decay: wd=1e-5
    - Increase learning rate: lr=1e-3
    - Train longer: epochs=50

    **If overfitting (high train, low val accuracy):**
    - Increase weight decay: wd=1e-3
    - Increase dropout: pdrop=0.1
    - Reduce model capacity: hidden=(128, 64)
    - More hard-negative mining: mine_low=0.3, mine_high=0.7

    **If training unstable (NaN loss):**
    - Decrease learning rate: lr=1e-4
    - Reduce batch size: batch=2048
    - Check for extreme feature values (use standardization!)
    - Increase gradient clipping: clip_grad_norm_(model.parameters(), 0.5)

    Examples
    --------
    >>> from classifier_code import train_classifier, Standardizer
    >>> from data_creation import build_raw_dataset_full
    >>> from physics_augment import augment_with_physics
    >>>
    >>> # Prepare data
    >>> data_train = build_raw_dataset_full(...)
    >>> data_valid = build_raw_dataset_full(...)
    >>>
    >>> # Augment with physics
    >>> X_train_aug = augment_with_physics(data_train.X, R=0.022, L=0.6, ...)
    >>> X_valid_aug = augment_with_physics(data_valid.X, R=0.022, L=0.6, ...)
    >>> data_train.X = X_train_aug
    >>> data_valid.X = X_valid_aug
    >>>
    >>> # Standardize
    >>> std = Standardizer(data_train.X)
    >>>
    >>> # Train classifier
    >>> model = train_classifier(
    ...     data_tr_full=data_train,
    ...     data_va_full=data_valid,
    ...     mu=std.mu,
    ...     sd=std.sd,
    ...     device='cuda',
    ...     epochs=30,
    ...     batch=8192,
    ...     lr=3e-4,
    ...     gamma=2.0,  # Focal loss strength
    ...     alpha=0.15,  # If 15% survival rate
    ...     patience=5
    ... )
    >>>
    >>> # Model ready for calibration
    >>> T_scale = calibrate_temperature(model, X_valid_aug, data_valid.y,
    ...                                   std.mu, std.sd, device='cuda')
    >>> print(f"Temperature scale: {T_scale:.3f}")
    >>>
    >>> # Save model
    >>> torch.save({
    ...     'model_state': model.state_dict(),
    ...     'mu': std.mu,
    ...     'sd': std.sd,
    ...     'temperature': model.T.item()
    ... }, 'classifier.pt')

    See Also
    --------
    SurvivalNet : Model architecture
    FocalWithLogits : Loss function
    hard_negative_indices : Mining strategy
    calibrate_temperature : Post-training probability calibration
    evaluate_classifier : Model evaluation metrics
    """
    model = SurvivalNetLN(
        d_in=data_tr_full.X.shape[1], pdrop=pdrop, input_drop=input_drop, hidden=hidden
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     opt, mode="min", factor=0.5, patience=3
    # )

    focal = FocalWithLogits(gamma=gamma, alpha=alpha)

    Xtr = torch.from_numpy(data_tr_full.X).float()
    ytr = torch.from_numpy(data_tr_full.y.astype(np.float32)).float()
    Xva = torch.from_numpy(data_va_full.X).float().to(device)
    yva = torch.from_numpy(data_va_full.y.astype(np.float32)).float().to(device)

    mu_t = torch.from_numpy(mu).float().to(device)
    sd_t = torch.from_numpy(sd).float().to(device)

    # add near the top of train_classifier, after Xtr/ytr are defined:
    n_train = Xtr.shape[0]
    seen = np.zeros(n_train, dtype=np.int32)  # how often each index has been used

    best = 1e9
    best_state = None
    wait = 0
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0

        # ---- dynamic hard-negative mining per epoch (logit space) ----
        model.eval()  # eval mode for mining
        with torch.inference_mode():
            Xt_all = ((Xtr - mu_t.cpu()) / sd_t.cpu()).to(device)
            logits_all = model(Xt_all, apply_temp=False).cpu().numpy()

        y_np = ytr.numpy()
        neg_idx = np.where(y_np == 0)[0]
        pos_idx = np.where(y_np == 1)[0]

        if pos_idx.size == 0:
            # degenerate: keep loop alive with a random small negative subset
            rng = np.random.default_rng(42 + ep)
            sel = rng.choice(neg_idx, size=min(batch * 4, neg_idx.size), replace=False)
        else:
            cand_local = hard_negative_indices_from_logits(
                logits_all[neg_idx],
                low=-0.6,
                high=0.6,
                max_neg=int(max(1, 4 * pos_idx.size)),
            )
            if cand_local.size == 0:
                cand_local = hard_negative_indices_from_logits(
                    logits_all[neg_idx],
                    low=-1.0,
                    high=1.0,
                    max_neg=int(max(1, 4 * pos_idx.size)),
                )
            mined_pool = neg_idx[cand_local]
            # prefer negatives we have used less often (top-k by smallest 'seen')
            k = int(max(1, 2 * pos_idx.size))
            if mined_pool.size > k:
                pool_seen = seen[mined_pool]
                keep = np.argsort(pool_seen, kind="stable")[:k]
                mined_neg = mined_pool[keep]
            else:
                mined_neg = mined_pool

            sel = np.concatenate([pos_idx, mined_neg])

        seen[sel] += 1

        rng = np.random.default_rng(42 + ep)
        rng.shuffle(sel)

        X_epoch = Xtr[sel]
        y_epoch = ytr[sel]
        ds = torch.utils.data.TensorDataset(X_epoch, y_epoch)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch, shuffle=True, drop_last=False
        )
        model.train()  # back to train mode

        for xb, yb in tqdm.tqdm(dl, desc=f"[CLS] Epoch {ep}/{epochs}"):
            xb = ((xb - mu_t.cpu()) / sd_t.cpu()).to(device)
            yb = yb.to(device)
            logits = model(xb, apply_temp=False)
            loss = focal(logits, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()

        # validation BCE (for early stopping)
        # model.eval()
        # with torch.no_grad():
        #     v_logits = model((Xva - mu_t) / sd_t, apply_temp=False)
        #     v_loss = nn.functional.binary_cross_entropy_with_logits(
        #         v_logits, yva
        #     ).item()
        # print(
        #     f"[CLS] Epoch {ep}: train_loss={tot / max(1, len(dl)):.4f}  val_loss={v_loss:.4f}"
        # )
        # validation (Brier)
        model.eval()
        with torch.no_grad():
            v_logits = model((Xva - mu_t) / sd_t, apply_temp=False)
            p_va = torch.sigmoid(v_logits)
            v_loss = torch.mean((p_va - yva) ** 2).item()  # <-- Brier
        print(
            f"[CLS] Epoch {ep}: train_loss={tot / max(1, len(dl)):.4f}  val_loss={v_loss:.4f}"
        )

        # scheduler.step(v_loss)

        if v_loss < best - 1e-4:
            best = v_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"[CLS] Early stop @ {ep} (best val={best:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ---------- Temperature calibration on VALID ----------
def calibrate_temperature(
    model: nn.Module,
    Xva_np: npt.NDArray[np.float32],
    yva_np: npt.NDArray[np.int64],
    mu: npt.NDArray[np.float32],
    sd: npt.NDArray[np.float32],
    device: str,
    iters: int = 200,
    lr: float = 5e-2,
    loss: str = "brier",
    clamp_logits: float | None = 15.0,
) -> float:
    """
    Calibrate probability estimates via temperature scaling on validation data.

    This function optimizes the model's single temperature parameter to improve
    probability calibration while preserving ranking. It minimizes either the
    Brier score or the negative log-likelihood.

    Parameters
    ----------
    model : nn.Module
        Trained classifier with attribute T (log-temperature).
    Xva_np : ndarray of shape (n_valid, n_features)
        Validation features in the same space used for training.
    yva_np : ndarray of shape (n_valid,)
        Validation labels in {0, 1}.
    mu : ndarray of shape (n_features,)
        Feature means from training standardization.
    sd : ndarray of shape (n_features,)
        Feature standard deviations from training standardization.
    device : str
        Target device for calibration ('cpu', 'cuda', etc.).
    iters : int, default=200
        Optimization iterations for the temperature parameter.
    lr : float, default=5e-2
        Learning rate for the temperature optimizer (Adam).
    loss : {'brier', 'nll'}, default='brier'
        Calibration objective. 'brier' often works well. 'nll' is also common.
    clamp_logits : float or None, default=15.0
        Optional clamp on logits for numerical stability. Set None to disable.

    Returns
    -------
    temperature_scale : float
        Learned temperature scale exp(T). Values > 1.0 soften probabilities.

    Notes
    -----
    - Use only on validation data to avoid optimistic bias.
    - This updates model.T in-place.

    Examples
    --------
    >>> T_scale = calibrate_temperature(model, X_valid, y_valid, mu, sd, device='cuda', loss='nll')
    >>> float(T_scale)
    """
    model.eval()
    Xva = torch.from_numpy(Xva_np).float().to(device)
    yva = torch.from_numpy(yva_np.astype(np.float32)).float().to(device)
    mu_t = torch.from_numpy(mu).float().to(device)
    sd_t = torch.from_numpy(sd).float().to(device)

    model.T.requires_grad_(True)
    opt = torch.optim.Adam([model.T], lr=lr)

    for _ in range(iters):
        z = model((Xva - mu_t) / sd_t, apply_temp=False) / torch.exp(model.T)
        if clamp_logits is not None:
            z = z.clamp(-float(clamp_logits), float(clamp_logits))

        if loss == "brier":
            p = torch.sigmoid(z)
            obj = torch.mean((p - yva) ** 2)
        elif loss == "nll":
            obj = nn.functional.binary_cross_entropy_with_logits(z, yva)
        else:
            raise ValueError("loss must be 'brier' or 'nll'")

        opt.zero_grad(set_to_none=True)
        obj.backward()
        opt.step()

        with torch.no_grad():
            model.T.clamp_(-5.0, 5.0)

    return float(torch.exp(model.T).detach().cpu().item())


# ---------- Evaluation helper ----------
@torch.no_grad()
def evaluate_classifier(model, data, mu, sd, device, thr=None, tag="TEST"):
    X = data.X
    y = data.y.astype(int)
    mu_t = torch.from_numpy(mu).float().to(device)
    sd_t = torch.from_numpy(sd).float().to(device)
    Xt = ((torch.from_numpy(X).float() - mu_t) / sd_t).to(device)
    logits = model(Xt, apply_temp=True).cpu().numpy()
    p = 1.0 / (1.0 + np.exp(-logits))
    if thr is None:
        thr = youden_threshold(y, p)
    yhat = (p >= thr).astype(int)

    tp, fp, fn, tn = _confusion(y, yhat)
    acc = float(np.mean(y == yhat))
    tpr = tp / (tp + fn + 1e-12)
    tnr = tn / (tn + fp + 1e-12)
    balc = 0.5 * (tpr + tnr)
    auc = _roc_auc_from_scores(y, p)
    prauc = _pr_auc_from_scores(y, p)
    brier = float(np.mean((p - y) ** 2))
    print(f"=== Survival ({tag}) ===")
    print(f"Accuracy          : {acc:.3f}")
    print(f"Balanced accuracy : {balc:.3f}")
    print(f"ROC-AUC           : {auc:.3f}")
    print(f"PR-AUC            : {prauc:.3f}")
    print(f"Brier score       : {brier:.5f}")
    print(f"Threshold         : {thr:.3f}")
    print(f"Confusion (tp, fp, fn, tn): {tp}, {fp}, {fn}, {tn}")
    return dict(acc=acc, balc=balc, auc=auc, prauc=prauc, brier=brier, thr=thr)
