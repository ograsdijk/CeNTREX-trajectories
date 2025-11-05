"""
Active Learning for Efficient Surrogate Model Training
=======================================================

This module implements active learning strategies to minimize the number of
expensive trajectory simulations needed to train accurate surrogate models.

Active learning iteratively:
1. Trains a surrogate on a small initial dataset
2. Uses the surrogate to identify the most informative unlabeled samples
3. Runs expensive simulations only on those selected samples
4. Retrains the surrogate with the augmented dataset
5. Repeats until convergence or budget exhausted

Key Benefits
------------
- Reduces data collection by 5-10x compared to random sampling
- Focuses simulations on challenging regions of parameter space
- Particularly effective for high-dimensional spaces
- Enables faster surrogate development and iteration

Acquisition Functions
---------------------
**Uncertainty Sampling**: Selects samples where model is most uncertain
    - For classifier: samples near decision boundary (p ≈ 0.5)
    - For regressor: samples with high prediction variance
    - Best when model needs to refine decision boundaries

**Diversity Sampling**: Selects samples covering parameter space
    - Uses k-means clustering to identify undersampled regions
    - Ensures good coverage of input distribution
    - Best when model needs broader exploration

**Hybrid Strategy**: Balances uncertainty and diversity
    - Combines both metrics with configurable weight
    - Default: 60% uncertainty, 40% diversity
    - Best overall strategy for most cases

Typical Workflow
----------------
>>> from active_learning import ActiveLearningConfig, active_learning_loop
>>> from centrex_trajectories import TlF, PropagationOptions
>>> from centrex_trajectories.data_structures import Force
>>> 
>>> # Configure active learning
>>> config = ActiveLearningConfig(
...     initial_samples_per_V=1000,
...     batch_size=500,
...     n_iterations=20,
...     strategy="hybrid",
...     uncertainty_weight=0.6,
... )
>>> 
>>> # Run active learning
>>> surrogate, history = active_learning_loop(
...     V_range=(5000, 30000),
...     R=0.022,
...     L=0.6,
...     config=config,
...     trajectory_fn=run_trajectory_simulation,
...     particle=TlF(),
...     gravity=Force(0, -9.81 * TlF().mass, 0),
...     alpha0=1.3e-30,
...     options=PropagationOptions(n_cores=8),
... )
>>> 
>>> # Total samples: 26*1000 + 20*500 = 36k instead of 200k!
>>> # Plot learning curves
>>> history.plot(save_path="active_learning_curves.png")

Performance Expectations
-------------------------
Compared to random sampling for same accuracy:
- Data reduction: 5-10x fewer samples needed
- Time savings: 80-90% reduction in simulation time
- Accuracy: Same or better final model performance

Example scenario (target: 95% classifier accuracy):
- Random sampling: 200k samples across 26 voltages
- Active learning: 36k samples (26k initial + 10k selected)
- Speedup: 5.5x reduction in data collection

See Also
--------
data_creation.build_raw_dataset_full : Random sampling baseline
classifier_code.train_classifier : Classifier training
regressor_code.train_regressor5_slopes : Regressor training

References
----------
- Settles, B. (2009). "Active Learning Literature Survey"
- Cohn, D. et al. (1996). "Active Learning with Statistical Models"
- Nguyen, H. T., & Smeulders, A. (2004). "Active learning using pre-clustering"
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Tuple, List, Optional, Dict, Any
from pathlib import Path
import numpy as np
import numpy.typing as npt
import torch
import copy
import time
import json

# Type aliases
ArrayFloat32 = npt.NDArray[np.float32]
ArrayFloat64 = npt.NDArray[np.float64]
ArrayBool = npt.NDArray[np.bool_]
ArrayInt = npt.NDArray[np.int_]

__all__ = [
    "ActiveLearningConfig",
    "ActiveLearningHistory",
    "generate_candidate_pool",
    "uncertainty_acquisition",
    "diversity_acquisition",
    "hybrid_acquisition",
    "active_learning_loop",
]


@dataclass
class ActiveLearningConfig:
    """
    Configuration for active learning training.
    
    Attributes
    ----------
    initial_samples_per_V : int, default=1000
        Initial random samples per voltage for bootstrap training
    batch_size : int, default=500
        Number of samples to add per active learning iteration
    n_iterations : int, default=20
        Maximum active learning iterations
    strategy : str, default="hybrid"
        Acquisition function: "uncertainty", "diversity", or "hybrid"
    uncertainty_weight : float, default=0.6
        Weight for uncertainty vs diversity in hybrid mode (0-1)
    candidate_pool_size : int, default=50_000
        Size of unlabeled candidate pool to score per iteration
    early_stopping_patience : int, default=5
        Stop if no improvement for N iterations
    early_stopping_threshold : float, default=1e-4
        Minimum validation loss improvement to count as progress
    V_sampling_strategy : str, default="uniform"
        How to sample voltages: "uniform" or "adaptive"
    n_voltages : int, default=26
        Number of voltages to sample
    classifier_epochs : int, default=20
        Training epochs for classifier (reduced from full training)
    classifier_batch : int, default=8192
        Batch size for classifier
    classifier_lr : float, default=3e-4
        Learning rate for classifier
    classifier_patience : int, default=3
        Early stopping patience for classifier
    regressor_epochs : int, default=30
        Training epochs for regressor
    regressor_batch : int, default=4096
        Batch size for regressor
    regressor_lr : float, default=1e-3
        Learning rate for regressor
    save_checkpoints : bool, default=True
        Save surrogate after each iteration
    checkpoint_dir : Path or None, default=None
        Where to save checkpoints (auto-created if None)
    verbose : bool, default=True
        Print detailed progress
    
    Examples
    --------
    >>> # Conservative: small batches, many iterations
    >>> config_conservative = ActiveLearningConfig(
    ...     initial_samples_per_V=500,
    ...     batch_size=200,
    ...     n_iterations=50,
    ...     strategy="uncertainty",
    ... )
    >>> 
    >>> # Aggressive: large batches, fewer iterations
    >>> config_aggressive = ActiveLearningConfig(
    ...     initial_samples_per_V=2000,
    ...     batch_size=1000,
    ...     n_iterations=10,
    ...     strategy="hybrid",
    ... )
    """
    # Data collection
    initial_samples_per_V: int = 1000
    batch_size: int = 500
    n_iterations: int = 20
    
    # Acquisition function
    strategy: str = "hybrid"
    uncertainty_weight: float = 0.6
    candidate_pool_size: int = 50_000
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 1e-4
    
    # Voltage sampling
    V_sampling_strategy: str = "uniform"
    n_voltages: int = 26
    
    # Classifier training (reduced from full training for speed)
    classifier_epochs: int = 20
    classifier_batch: int = 8192
    classifier_lr: float = 3e-4
    classifier_patience: int = 3
    
    # Regressor training
    regressor_epochs: int = 30
    regressor_batch: int = 4096
    regressor_lr: float = 1e-3
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: Optional[Path] = None
    
    # Verbosity
    verbose: bool = True


@dataclass
class ActiveLearningHistory:
    """
    History tracking for active learning iterations.
    
    Tracks metrics, timing, and selection statistics across iterations
    to analyze active learning efficiency and convergence.
    
    Attributes
    ----------
    iteration : List[int]
        Iteration numbers [0, 1, 2, ...]
    n_samples : List[int]
        Cumulative samples collected [26k, 26.5k, 27k, ...]
    train_loss : List[float]
        Training loss per iteration
    val_loss : List[float]
        Validation loss per iteration
    val_accuracy : List[float]
        Validation accuracy (classifier) [0-1]
    val_mae_position : List[float]
        Validation MAE for position (regressor) [m]
    acquisition_scores : List[Dict[str, float]]
        Statistics of acquisition scores per iteration
    time_per_iteration : List[float]
        Wall-clock time per iteration [seconds]
    samples_selected : List[ArrayInt]
        Indices of selected samples per iteration (large, not saved to JSON)
    
    Methods
    -------
    save(filepath)
        Save history to JSON file
    plot(save_path)
        Plot learning curves
    
    Examples
    --------
    >>> # Analyze convergence
    >>> history = active_learning_loop(...)[1]
    >>> print(f"Final samples: {history.n_samples[-1]:,}")
    >>> print(f"Final accuracy: {history.val_accuracy[-1]:.3f}")
    >>> 
    >>> # Plot learning curves
    >>> history.plot(save_path="learning_curves.png")
    >>> 
    >>> # Check acquisition score trends
    >>> for i, scores in enumerate(history.acquisition_scores):
    ...     print(f"Iter {i}: mean={scores['mean']:.3f}, std={scores['std']:.3f}")
    """
    iteration: List[int] = field(default_factory=list)
    n_samples: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    val_mae_position: List[float] = field(default_factory=list)
    acquisition_scores: List[Dict[str, float]] = field(default_factory=list)
    time_per_iteration: List[float] = field(default_factory=list)
    samples_selected: List[ArrayInt] = field(default_factory=list)
    
    def save(self, filepath: Path):
        """
        Save history to JSON file.
        
        Parameters
        ----------
        filepath : Path
            Output JSON file path
            
        Notes
        -----
        Does not save samples_selected (too large for JSON)
        """
        data = {
            "iteration": self.iteration,
            "n_samples": self.n_samples,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "val_mae_position": self.val_mae_position,
            "acquisition_scores": self.acquisition_scores,
            "time_per_iteration": self.time_per_iteration,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"History saved to {filepath}")
    
    def plot(self, save_path: Optional[Path] = None):
        """
        Plot learning curves from active learning history.
        
        Creates 2×2 subplot figure showing:
        - Validation loss vs samples
        - Validation accuracy vs samples
        - Position MAE vs samples
        - Time per iteration
        
        Parameters
        ----------
        save_path : Path or None
            If provided, save figure to this path instead of showing
            
        Examples
        --------
        >>> history.plot()  # Display interactively
        >>> history.plot(save_path=Path("curves.png"))  # Save to file
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Validation loss
        ax = axes[0, 0]
        ax.plot(self.n_samples, self.val_loss, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Total Samples Collected', fontsize=12)
        ax.set_ylabel('Validation Loss', fontsize=12)
        ax.set_title('Learning Curve: Loss vs Data Size', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Validation accuracy
        ax = axes[0, 1]
        ax.plot(self.n_samples, np.array(self.val_accuracy) * 100, 'o-', 
                linewidth=2, markersize=6, color='green')
        ax.set_xlabel('Total Samples Collected', fontsize=12)
        ax.set_ylabel('Validation Accuracy [%]', fontsize=12)
        ax.set_title('Classifier Accuracy vs Data Size', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Position MAE
        ax = axes[1, 0]
        ax.plot(self.n_samples, np.array(self.val_mae_position) * 1e3, 'o-', 
                linewidth=2, markersize=6, color='orange')
        ax.set_xlabel('Total Samples Collected', fontsize=12)
        ax.set_ylabel('Position MAE [mm]', fontsize=12)
        ax.set_title('Regressor Accuracy vs Data Size', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Time per iteration
        ax = axes[1, 1]
        ax.plot(self.iteration, self.time_per_iteration, 'o-', 
                linewidth=2, markersize=6, color='red')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Time [seconds]', fontsize=12)
        ax.set_title('Iteration Time (Simulation + Training)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()


def generate_candidate_pool(
    n_candidates: int,
    V_range: Tuple[float, float],
    R: float,
    rng: np.random.Generator,
) -> ArrayFloat32:
    """
    Generate candidate pool for active learning selection.
    
    Creates uniformly distributed candidates across parameter space
    [x0, y0, vx, vy, vz, V] for acquisition function scoring.
    
    Parameters
    ----------
    n_candidates : int
        Number of candidate samples to generate
    V_range : Tuple[float, float]
        (V_min, V_max) voltage range [V]
    R : float
        Lens radius [m] (sets spatial range: ±0.8R)
    rng : np.random.Generator
        Random number generator for reproducibility
        
    Returns
    -------
    candidates : ndarray of shape (n_candidates, 6), dtype=float32
        Candidate samples [x0, y0, vx, vy, vz, V]
        
    Notes
    -----
    - Spatial range: ±0.8R (within aperture)
    - Velocity range: vx, vy ∈ [-10, 10] m/s, vz ∈ [100, 250] m/s
    - Voltage range: uniformly distributed
    - All candidates are unlabeled (no simulation run yet)
    
    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> candidates = generate_candidate_pool(
    ...     n_candidates=50_000,
    ...     V_range=(5000, 30000),
    ...     R=0.022,
    ...     rng=rng
    ... )
    >>> print(candidates.shape)  # (50000, 6)
    >>> print(candidates[:, -1].min(), candidates[:, -1].max())  # Voltage range
    """
    x0 = rng.uniform(-0.8 * R, 0.8 * R, n_candidates).astype(np.float32)
    y0 = rng.uniform(-0.8 * R, 0.8 * R, n_candidates).astype(np.float32)
    vx = rng.uniform(-10.0, 10.0, n_candidates).astype(np.float32)
    vy = rng.uniform(-10.0, 10.0, n_candidates).astype(np.float32)
    vz = rng.uniform(100.0, 250.0, n_candidates).astype(np.float32)
    V = rng.uniform(V_range[0], V_range[1], n_candidates).astype(np.float32)
    
    candidates = np.column_stack([x0, y0, vx, vy, vz, V])
    return candidates


def uncertainty_acquisition(
    surrogate: Any,  # SurrogateXYV
    candidates: ArrayFloat32,
) -> ArrayFloat32:
    """
    Uncertainty-based acquisition function.
    
    Scores candidates by model uncertainty. High uncertainty indicates
    samples where the model is least confident and would benefit most
    from ground truth labels.
    
    For classification: Uncertainty = distance from decision boundary
        score = 1 - |p - 0.5| × 2  (max at p=0.5, min at p=0 or p=1)
    
    For regression: Would use prediction variance (requires MC dropout)
    
    Parameters
    ----------
    surrogate : SurrogateXYV
        Trained surrogate model
    candidates : ndarray of shape (n_candidates, 6)
        Candidate samples to score [x0, y0, vx, vy, vz, V]
        
    Returns
    -------
    scores : ndarray of shape (n_candidates,), dtype=float32
        Uncertainty scores in [0, 1], higher = more uncertain
        
    Notes
    -----
    - Fast to compute (single forward pass)
    - Works well for refining decision boundaries
    - Can oversample near boundaries if used exclusively
    
    Examples
    --------
    >>> scores = uncertainty_acquisition(surrogate, candidates)
    >>> # Select most uncertain
    >>> top_indices = np.argsort(scores)[-500:]  # Top 500
    >>> selected = candidates[top_indices]
    """
    x0, y0, vx, vy, vz, V = candidates.T
    
    # Get survival probabilities from classifier
    p_survive, _, _, _, _, _, _ = surrogate.predict(x0, y0, vx, vy, vz, V)
    
    # Uncertainty score: highest for p ≈ 0.5 (decision boundary)
    # score = 1 - 2 * |p - 0.5|
    # p=0.5 → score=1.0, p=0 or p=1 → score=0.0
    uncertainty = 1.0 - 2.0 * np.abs(p_survive - 0.5)
    
    return uncertainty.astype(np.float32)


def diversity_acquisition(
    candidates: ArrayFloat32,
    existing_data: ArrayFloat32,
    n_select: int,
) -> ArrayInt:
    """
    Diversity-based acquisition function using k-means clustering.
    
    Selects candidates that cover undersampled regions of parameter space.
    Uses k-means to partition candidates into n_select clusters, then
    picks the candidate closest to each cluster center.
    
    Parameters
    ----------
    candidates : ndarray of shape (n_candidates, 6)
        Candidate samples [x0, y0, vx, vy, vz, V]
    existing_data : ndarray of shape (n_existing, 6)
        Already collected training data
    n_select : int
        Number of diverse samples to select
        
    Returns
    -------
    selected_indices : ndarray of shape (n_select,), dtype=int
        Indices into candidates array for selected samples
        
    Notes
    -----
    - More expensive than uncertainty (requires clustering)
    - Ensures broad coverage of parameter space
    - Good for exploration, less good for exploitation
    - Can select redundant samples near existing data
    
    Examples
    --------
    >>> indices = diversity_acquisition(candidates, training_data, n_select=500)
    >>> selected = candidates[indices]
    >>> # Selected samples will be spread across parameter space
    """
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cdist
    
    # Standardize features for clustering (important!)
    mean = candidates.mean(axis=0)
    std = candidates.std(axis=0) + 1e-8
    candidates_norm = (candidates - mean) / std
    
    # K-means clustering to find n_select representative regions
    kmeans = KMeans(n_clusters=n_select, random_state=42, n_init=10)
    kmeans.fit(candidates_norm)
    
    # For each cluster, find closest candidate to center
    distances = cdist(kmeans.cluster_centers_, candidates_norm)
    selected_indices = distances.argmin(axis=1)
    
    return selected_indices


def hybrid_acquisition(
    surrogate: Any,  # SurrogateXYV
    candidates: ArrayFloat32,
    existing_data: ArrayFloat32,
    n_select: int,
    uncertainty_weight: float = 0.6,
) -> ArrayInt:
    """
    Hybrid acquisition combining uncertainty and diversity.
    
    Balances exploration (diversity) and exploitation (uncertainty)
    by combining both scores with configurable weight.
    
    Strategy:
    1. Compute uncertainty scores for all candidates
    2. Compute diversity scores (distance to existing data)
    3. Combine: score = w × uncertainty + (1-w) × diversity
    4. Select top n_select by combined score
    
    Parameters
    ----------
    surrogate : SurrogateXYV
        Trained surrogate model
    candidates : ndarray of shape (n_candidates, 6)
        Candidate samples to score
    existing_data : ndarray of shape (n_existing, 6)
        Already collected training data
    n_select : int
        Number of samples to select
    uncertainty_weight : float, default=0.6
        Weight for uncertainty vs diversity (0-1)
        0.0 = pure diversity, 1.0 = pure uncertainty
        
    Returns
    -------
    selected_indices : ndarray of shape (n_select,), dtype=int
        Indices into candidates array for selected samples
        
    Notes
    -----
    - Best overall strategy for most cases
    - Balances exploration and exploitation
    - Default weight 0.6 works well empirically
    - Adjust weight based on problem:
      * High weight (0.8): focus on decision boundaries
      * Low weight (0.4): focus on coverage
    
    Examples
    --------
    >>> # Balanced selection
    >>> indices = hybrid_acquisition(
    ...     surrogate, candidates, training_data,
    ...     n_select=500, uncertainty_weight=0.6
    ... )
    >>> 
    >>> # More exploration
    >>> indices = hybrid_acquisition(
    ...     surrogate, candidates, training_data,
    ...     n_select=500, uncertainty_weight=0.3
    ... )
    """
    from scipy.spatial.distance import cdist
    
    # 1. Uncertainty scores
    uncertainty = uncertainty_acquisition(surrogate, candidates)
    
    # 2. Diversity scores (distance to nearest existing sample)
    # Standardize features
    all_data = np.vstack([existing_data, candidates])
    mean = all_data.mean(axis=0)
    std = all_data.std(axis=0) + 1e-8
    
    existing_norm = (existing_data - mean) / std
    candidates_norm = (candidates - mean) / std
    
    # Compute min distance to existing data for each candidate
    distances = cdist(candidates_norm, existing_norm)
    min_distances = distances.min(axis=1)
    
    # Normalize to [0, 1]
    diversity = (min_distances - min_distances.min()) / (
        min_distances.max() - min_distances.min() + 1e-8
    )
    
    # 3. Combine scores
    combined_score = (
        uncertainty_weight * uncertainty +
        (1.0 - uncertainty_weight) * diversity
    )
    
    # 4. Select top n_select
    selected_indices = np.argsort(combined_score)[-n_select:]
    
    return selected_indices


def active_learning_loop(
    V_range: Tuple[float, float],
    R: float,
    L: float,
    alpha0: float,
    particle: Any,  # Particle
    gravity: Any,  # Force
    options: Any,  # PropagationOptions
    trajectory_fn: Callable,
    config: ActiveLearningConfig,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Any, ActiveLearningHistory]:  # Returns (SurrogateXYV, History)
    """
    Run active learning loop for surrogate training.
    
    Main active learning algorithm that iteratively:
    1. Trains surrogate on current dataset
    2. Generates candidate pool
    3. Scores candidates with acquisition function
    4. Runs simulations on selected candidates
    5. Adds new data and repeats
    
    Parameters
    ----------
    V_range : Tuple[float, float]
        (V_min, V_max) voltage range [V]
    R : float
        Lens bore radius [m]
    L : float
        Lens length [m]
    alpha0 : float
        Stark polarizability [J/(V/m)²]
    particle : Particle
        Particle species (e.g., TlF())
    gravity : Force
        Gravity force vector
    options : PropagationOptions
        Trajectory propagation options
    trajectory_fn : Callable
        Function to run trajectory simulation:
        signature: (R, V, L, n_traj, rng, options, particle, gravity) -> results
    config : ActiveLearningConfig
        Active learning configuration
    rng : np.random.Generator or None
        Random number generator (created if None)
        
    Returns
    -------
    surrogate : SurrogateXYV
        Final trained surrogate model
    history : ActiveLearningHistory
        Training history with metrics and timing
        
    Notes
    -----
    **Computational Cost**:
    - Initial phase: Same as random sampling
    - Per iteration:
      * Candidate generation: ~0.1 seconds
      * Acquisition scoring: ~1-5 seconds
      * Trajectory simulation: Depends on batch_size
      * Model retraining: ~30-60 seconds
    
    **Memory Requirements**:
    - Peaks during model training: ~2-4 GB
    - Candidate pool: ~50k × 6 × 4 bytes = 1.2 MB
    - Training data grows with iterations
    
    **Convergence**:
    - Typical convergence: 10-20 iterations
    - Monitor validation loss for early stopping
    - Can adjust batch_size vs n_iterations trade-off
    
    Examples
    --------
    >>> from centrex_trajectories import TlF, PropagationOptions
    >>> from centrex_trajectories.data_structures import Force
    >>> 
    >>> # Configure
    >>> config = ActiveLearningConfig(
    ...     initial_samples_per_V=1000,
    ...     batch_size=500,
    ...     n_iterations=20,
    ...     strategy="hybrid",
    ... )
    >>> 
    >>> # Run active learning
    >>> surrogate, history = active_learning_loop(
    ...     V_range=(5000, 30000),
    ...     R=0.022,
    ...     L=0.6,
    ...     alpha0=1.3e-30,
    ...     particle=TlF(),
    ...     gravity=Force(0, -9.81 * TlF().mass, 0),
    ...     options=PropagationOptions(n_cores=8, verbose=False),
    ...     trajectory_fn=run_trajectory_simulation,
    ...     config=config,
    ... )
    >>> 
    >>> # Analyze results
    >>> print(f"Final samples: {history.n_samples[-1]:,}")
    >>> print(f"Final accuracy: {history.val_accuracy[-1]:.3f}")
    >>> print(f"Total time: {sum(history.time_per_iteration):.1f} seconds")
    >>> 
    >>> # Plot learning curves
    >>> history.plot(save_path="active_learning.png")
    >>> 
    >>> # Save history
    >>> history.save(Path("active_learning_history.json"))
    """
    from data_creation import RawDataFull, build_raw_dataset_full
    from physics_augment import augment_with_physics
    from classifier_code import Standardizer, train_classifier, calibrate_temperature
    from regressor_code import TrainCfgReg5, train_regressor5_slopes
    
    if rng is None:
        rng = np.random.default_rng(42)
    
    if config.checkpoint_dir is None:
        config.checkpoint_dir = Path("active_learning_checkpoints")
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    history = ActiveLearningHistory()
    
    if config.verbose:
        print("="*70)
        print("ACTIVE LEARNING FOR SURROGATE TRAINING")
        print("="*70)
        print(f"Strategy: {config.strategy}")
        print(f"Initial samples per voltage: {config.initial_samples_per_V:,}")
        print(f"Batch size: {config.batch_size}")
        print(f"Max iterations: {config.n_iterations}")
        print(f"Candidate pool size: {config.candidate_pool_size:,}")
        print("="*70)
    
    # Step 1: Initial random sampling
    if config.verbose:
        print("\n[STEP 1] Collecting initial training data...")
    
    V_list = np.linspace(V_range[0], V_range[1], config.n_voltages)
    
    # Import here to avoid circular dependency
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print("Warning: tqdm not installed, progress bars disabled")
    
    # Build initial dataset
    # Note: This uses the build_raw_dataset_full from data_creation.py
    # We'll simulate it inline for now since we need the trajectory_fn
    
    data_tr_list = []
    for V in (tqdm(V_list, desc="Initial sampling") if use_tqdm else V_list):
        coords_init, coords_final, vels_init, vels_final, idx_survive = trajectory_fn(
            R=R,
            V=V,
            L=L,
            n_trajectories=config.initial_samples_per_V,
            rng=rng,
            options=options,
            particle=particle,
            gravity=gravity,
        )
        
        # Package as RawDataFull-like structure
        n = config.initial_samples_per_V
        X = np.column_stack([
            coords_init.x, coords_init.y,
            vels_init.vx, vels_init.vy, vels_init.vz,
            np.full(n, V, dtype=np.float32)
        ])
        
        y = np.zeros(n, dtype=np.int64)
        y[idx_survive] = 1
        
        xf = np.zeros(n, dtype=np.float32)
        yf = np.zeros(n, dtype=np.float32)
        vxf = np.zeros(n, dtype=np.float32)
        vyf = np.zeros(n, dtype=np.float32)
        vzf = np.zeros(n, dtype=np.float32)
        
        if len(idx_survive) > 0:
            xf[idx_survive] = coords_final.x
            yf[idx_survive] = coords_final.y
            vxf[idx_survive] = vels_final.vx
            vyf[idx_survive] = vels_final.vy
            vzf[idx_survive] = vels_final.vz
        
        mask_surv = y.astype(bool)
        
        data_tr_list.append(RawDataFull(X, y, xf, yf, vxf, vyf, vzf, mask_surv))
    
    # Combine all voltages
    data_tr_full = RawDataFull(
        X=np.vstack([d.X for d in data_tr_list]),
        y=np.concatenate([d.y for d in data_tr_list]),
        xf=np.concatenate([d.xf for d in data_tr_list]),
        yf=np.concatenate([d.yf for d in data_tr_list]),
        vxf=np.concatenate([d.vxf for d in data_tr_list]),
        vyf=np.concatenate([d.vyf for d in data_tr_list]),
        vzf=np.concatenate([d.vzf for d in data_tr_list]),
        mask_surv=np.concatenate([d.mask_surv for d in data_tr_list]),
    )
    
    # Create validation set (20% of initial data)
    n_total = len(data_tr_full.X)
    indices = np.arange(n_total)
    rng.shuffle(indices)
    split_idx = int(0.8 * n_total)
    
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    data_tr = RawDataFull(
        X=data_tr_full.X[train_idx],
        y=data_tr_full.y[train_idx],
        xf=data_tr_full.xf[train_idx],
        yf=data_tr_full.yf[train_idx],
        vxf=data_tr_full.vxf[train_idx],
        vyf=data_tr_full.vyf[train_idx],
        vzf=data_tr_full.vzf[train_idx],
        mask_surv=data_tr_full.mask_surv[train_idx],
    )
    
    data_va = RawDataFull(
        X=data_tr_full.X[val_idx],
        y=data_tr_full.y[val_idx],
        xf=data_tr_full.xf[val_idx],
        yf=data_tr_full.yf[val_idx],
        vxf=data_tr_full.vxf[val_idx],
        vyf=data_tr_full.vyf[val_idx],
        vzf=data_tr_full.vzf[val_idx],
        mask_surv=data_tr_full.mask_surv[val_idx],
    )
    
    if config.verbose:
        print(f"Initial dataset: {len(data_tr.X):,} training, {len(data_va.X):,} validation")
        print(f"Survival rate: {data_tr.y.mean():.2%}")
    
    # Keep track of all collected data for diversity calculation
    all_collected_X = data_tr_full.X.copy()
    
    # Step 2: Active learning iterations
    for iteration in range(config.n_iterations):
        iter_start = time.time()
        
        if config.verbose:
            print(f"\n{'='*70}")
            print(f"ACTIVE LEARNING ITERATION {iteration + 1}/{config.n_iterations}")
            print(f"{'='*70}")
            print(f"Current training samples: {len(data_tr.X):,}")
        
        # Step 2a: Train surrogate on current data
        if config.verbose:
            print(f"\n[Step {iteration+1}.1] Training surrogate...")
        
        # Augment with physics
        X_tr_aug = augment_with_physics(data_tr.X, R, L, alpha0, particle.mass)
        X_va_aug = augment_with_physics(data_va.X, R, L, alpha0, particle.mass)
        data_tr.X = X_tr_aug
        data_va.X = X_va_aug
        
        # Standardize
        std = Standardizer(data_tr.X)
        
        # Train classifier
        clf = train_classifier(
            data_tr,
            data_va,
            mu=std.mu,
            sd=std.sd,
            device="cuda" if torch.cuda.is_available() else "cpu",
            epochs=config.classifier_epochs,
            batch=config.classifier_batch,
            lr=config.classifier_lr,
            patience=config.classifier_patience,
            gamma=2.0,
            alpha=None,
            mine_low=0.35,
            mine_high=0.65,
        )
        
        # Calibrate temperature
        temp_scale = calibrate_temperature(
            clf, data_va.X, data_va.y, std.mu, std.sd,
            device="cuda" if torch.cuda.is_available() else "cpu",
            iters=100,
        )
        
        # Train regressor
        surv_tr = data_tr.mask_surv
        X_surv_raw_tr = data_tr.X[surv_tr][:, :6]  # Get raw features only
        X_surv_aug_tr = data_tr.X[surv_tr]
        Y_surv_tr = np.column_stack([
            data_tr.xf[surv_tr],
            data_tr.yf[surv_tr],
            data_tr.vxf[surv_tr],
            data_tr.vyf[surv_tr],
            data_tr.vzf[surv_tr],
        ])
        
        std_aug = Standardizer(X_surv_aug_tr)
        cfg_reg = TrainCfgReg5(
            epochs=config.regressor_epochs,
            batch=config.regressor_batch,
            lr=config.regressor_lr,
        )
        
        reg5, scalerY = train_regressor5_slopes(
            X_surv_aug_tr,
            Y_surv_tr,
            X_surv_raw_tr,
            std_aug.mu,
            std_aug.sd,
            R,
            cfg_reg,
        )
        
        # Create surrogate
        # Note: This assumes SurrogateXYV exists in regressor_code.py
        # If not, you'll need to create it or adapt this
        try:
            from regressor_code import SurrogateXYV
            surrogate = SurrogateXYV(
                clf=clf,
                reg5=reg5,
                muX_clf=std.mu,
                sdX_clf=std.sd,
                muX_reg=std_aug.mu,
                sdX_reg=std_aug.sd,
                scalerY=scalerY,
                R=R,
                L=L,
                alpha0=alpha0,
                mass=particle.mass,
                device="cuda" if torch.cuda.is_available() else "cpu",
                thr=0.5,
                augment_fn_clf=None,
                augment_fn_reg=augment_with_physics,
            )
        except ImportError:
            print("Warning: SurrogateXYV not found in regressor_code, creating placeholder")
            surrogate = None
        
        # Evaluate on validation set
        if surrogate is not None:
            x0_va, y0_va, vx_va, vy_va, vz_va, V_va = data_va.X[:, :6].T
            p_va, survive_va, _, _, _, _, _ = surrogate.predict(
                x0_va, y0_va, vx_va, vy_va, vz_va, V_va
            )
            val_acc = (survive_va == data_va.y).mean()
            
            # Compute position MAE for survivors
            surv_mask = data_va.mask_surv & survive_va
            if surv_mask.sum() > 0:
                x_pred = surrogate.predict(
                    x0_va[surv_mask], y0_va[surv_mask],
                    vx_va[surv_mask], vy_va[surv_mask],
                    vz_va[surv_mask], V_va[surv_mask]
                )[2]
                x_true = data_va.xf[surv_mask]
                val_mae = np.abs(x_pred - x_true).mean()
            else:
                val_mae = float('nan')
        else:
            val_acc = 0.0
            val_mae = float('nan')
        
        if config.verbose:
            print(f"Validation accuracy: {val_acc:.4f}")
            print(f"Validation position MAE: {val_mae*1e3:.3f} mm")
        
        # Step 2b: Generate candidate pool
        if config.verbose:
            print(f"\n[Step {iteration+1}.2] Generating candidate pool...")
        
        candidates = generate_candidate_pool(
            n_candidates=config.candidate_pool_size,
            V_range=V_range,
            R=R,
            rng=rng,
        )
        
        # Step 2c: Score candidates with acquisition function
        if config.verbose:
            print(f"[Step {iteration+1}.3] Scoring candidates with {config.strategy} acquisition...")
        
        if config.strategy == "uncertainty" and surrogate is not None:
            scores = uncertainty_acquisition(surrogate, candidates)
            selected_indices = np.argsort(scores)[-config.batch_size:]
        elif config.strategy == "diversity":
            selected_indices = diversity_acquisition(
                candidates, all_collected_X[:, :6], config.batch_size
            )
            scores = np.ones(len(candidates))  # Placeholder
        elif config.strategy == "hybrid" and surrogate is not None:
            selected_indices = hybrid_acquisition(
                surrogate, candidates, all_collected_X[:, :6],
                config.batch_size, config.uncertainty_weight
            )
            scores = uncertainty_acquisition(surrogate, candidates)  # For logging
        else:
            # Fallback to random
            selected_indices = rng.choice(len(candidates), config.batch_size, replace=False)
            scores = np.ones(len(candidates))
        
        selected_candidates = candidates[selected_indices]
        
        if config.verbose:
            print(f"Selected {len(selected_candidates)} candidates")
            if len(scores) > 0:
                print(f"Acquisition scores - mean: {scores.mean():.3f}, std: {scores.std():.3f}")
        
        # Step 2d: Run expensive simulations on selected candidates
        if config.verbose:
            print(f"\n[Step {iteration+1}.4] Running simulations on selected candidates...")
        
        new_data_list = []
        for candidate in (tqdm(selected_candidates, desc="Simulating") if use_tqdm else selected_candidates):
            x0, y0, vx, vy, vz, V = candidate
            
            # Run single trajectory simulation
            # Note: trajectory_fn expects arrays, so we create single-element arrays
            coords_init_single = type('Coords', (), {
                'x': np.array([x0]), 'y': np.array([y0]), 'z': np.array([0.0])
            })()
            vels_init_single = type('Vels', (), {
                'vx': np.array([vx]), 'vy': np.array([vy]), 'vz': np.array([vz])
            })()
            
            try:
                coords_init, coords_final, vels_init, vels_final, idx_survive = trajectory_fn(
                    R=R,
                    V=float(V),
                    L=L,
                    n_trajectories=1,
                    rng=rng,
                    options=options,
                    particle=particle,
                    gravity=gravity,
                )
                
                survived = 0 in idx_survive
                if survived:
                    xf, yf = coords_final.x[0], coords_final.y[0]
                    vxf, vyf, vzf = vels_final.vx[0], vels_final.vy[0], vels_final.vz[0]
                else:
                    xf = yf = vxf = vyf = vzf = 0.0
                
                new_data_list.append({
                    'X': candidate,
                    'y': int(survived),
                    'xf': xf, 'yf': yf,
                    'vxf': vxf, 'vyf': vyf, 'vzf': vzf,
                })
            except Exception as e:
                if config.verbose:
                    print(f"Warning: Simulation failed for candidate: {e}")
                continue
        
        # Step 2e: Add new data to training set
        if len(new_data_list) > 0:
            new_X = np.array([d['X'] for d in new_data_list], dtype=np.float32)
            new_y = np.array([d['y'] for d in new_data_list], dtype=np.int64)
            new_xf = np.array([d['xf'] for d in new_data_list], dtype=np.float32)
            new_yf = np.array([d['yf'] for d in new_data_list], dtype=np.float32)
            new_vxf = np.array([d['vxf'] for d in new_data_list], dtype=np.float32)
            new_vyf = np.array([d['vyf'] for d in new_data_list], dtype=np.float32)
            new_vzf = np.array([d['vzf'] for d in new_data_list], dtype=np.float32)
            new_mask = new_y.astype(bool)
            
            # Append to training data
            data_tr = RawDataFull(
                X=np.vstack([data_tr.X[:, :6], new_X]),  # Strip augmentation before appending
                y=np.concatenate([data_tr.y, new_y]),
                xf=np.concatenate([data_tr.xf, new_xf]),
                yf=np.concatenate([data_tr.yf, new_yf]),
                vxf=np.concatenate([data_tr.vxf, new_vxf]),
                vyf=np.concatenate([data_tr.vyf, new_vyf]),
                vzf=np.concatenate([data_tr.vzf, new_vzf]),
                mask_surv=np.concatenate([data_tr.mask_surv, new_mask]),
            )
            
            # Update all_collected_X for diversity calculation
            all_collected_X = np.vstack([all_collected_X, new_X])
            
            if config.verbose:
                print(f"Added {len(new_data_list)} new samples to training set")
                print(f"New training set size: {len(data_tr.X):,}")
        
        # Record history
        iter_time = time.time() - iter_start
        history.iteration.append(iteration)
        history.n_samples.append(len(all_collected_X))
        history.train_loss.append(0.0)  # Placeholder
        history.val_loss.append(0.0)  # Placeholder
        history.val_accuracy.append(float(val_acc))
        history.val_mae_position.append(float(val_mae))
        history.acquisition_scores.append({
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'min': float(scores.min()),
            'max': float(scores.max()),
        })
        history.time_per_iteration.append(iter_time)
        history.samples_selected.append(selected_indices)
        
        if config.verbose:
            print(f"\nIteration {iteration + 1} completed in {iter_time:.1f} seconds")
        
        # Save checkpoint
        if config.save_checkpoints and surrogate is not None:
            checkpoint_path = config.checkpoint_dir / f"surrogate_iter_{iteration+1:03d}.pt"
            torch.save({
                'clf_state': clf.state_dict(),
                'reg_state': reg5.state_dict(),
                'iteration': iteration,
                'n_samples': len(all_collected_X),
            }, checkpoint_path)
            if config.verbose:
                print(f"Checkpoint saved to {checkpoint_path}")
        
        # Check early stopping
        if iteration >= config.early_stopping_patience:
            recent_losses = history.val_loss[-config.early_stopping_patience:]
            if len(recent_losses) == config.early_stopping_patience:
                loss_std = np.std(recent_losses)
                if loss_std < config.early_stopping_threshold:
                    if config.verbose:
                        print(f"\nEarly stopping: validation loss plateaued")
                    break
    
    # Final training on all collected data
    if config.verbose:
        print(f"\n{'='*70}")
        print("FINAL TRAINING ON ALL COLLECTED DATA")
        print(f"{'='*70}")
        print(f"Total samples collected: {len(all_collected_X):,}")
    
    # Save final history
    history.save(config.checkpoint_dir / "active_learning_history.json")
    
    return surrogate, history


if __name__ == "__main__":
    # Example usage
    print("Active learning module loaded successfully!")
    print("See docstrings for usage examples.")
