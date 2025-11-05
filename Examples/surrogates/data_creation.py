"""
Training Data Creation for Surrogate Models
============================================

This module generates training datasets for neural network surrogates that replace
expensive trajectory simulations. It runs full physics-based trajectory propagation
across a range of voltages and initial conditions, then packages the results into
structured datasets suitable for machine learning.

The workflow:
1. Sample voltages from a specified range
2. Generate random initial conditions for each voltage
3. Run trajectory simulations through the beamline
4. Record initial conditions, final states, and survival outcomes
5. Package into RawDataFull containers for training

Usage Example
-------------
>>> from centrex_trajectories import TlF, PropagationOptions
>>> from centrex_trajectories.data_structures import Force
>>>
>>> # Define simulation parameters
>>> V_list = np.linspace(5000, 30000, 26)  # 26 voltage points
>>> n_traj_per_V = 20000  # 20k trajectories per voltage
>>> R = 0.022  # Lens bore radius [m]
>>> L = 0.6    # Lens length [m]
>>>
>>> # Generate training data
>>> data = build_raw_dataset_full(
>>>     V_list=V_list,
>>>     per_V=n_traj_per_V,
>>>     R=R, L=L,
>>>     rng=np.random.default_rng(42),
>>>     options=PropagationOptions(n_cores=8),
>>>     particle=TlF(),
>>>     gravity=Force(0, -9.81 * TlF().mass, 0),
>>>     trajectory_simulation_fn=my_simulation_function
>>> )
>>>
>>> # Data now ready for neural network training
>>> print(f"Total samples: {len(data.X)}")
>>> print(f"Survival rate: {data.y.mean():.2%}")
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable, Optional, Tuple

import numpy as np
import numpy.typing as npt
import tqdm

from centrex_trajectories.data_structures import Coordinates, Gravity, Velocities
from centrex_trajectories.particles import Particle
from centrex_trajectories.propagation_options import PropagationOptions

__all__ = ["RawDataFull", "build_raw_dataset_full"]


# ---------- (1) Build dataset with final velocities ----------
@dataclass
class RawDataFull:
    """
    Container for raw trajectory simulation data used to train surrogate models.

    This stores initial conditions, survival outcomes, and final states for
    all trajectories across all sampled voltages. Non-surviving trajectories
    have their final states set to zero.

    Attributes
    ----------
    X : ndarray of shape (N, 6)
        Input features: [x0, y0, vx, vy, vz, V]
        - x0, y0: Initial transverse positions [m]
        - vx, vy: Initial transverse velocities [m/s]
        - vz: Initial forward velocity [m/s]
        - V: Applied voltage [V]
    y : ndarray of shape (N,)
        Binary survival labels: 1 = survived, 0 = collided
    xf : ndarray of shape (N,)
        Final x position [m] (0 for non-survivors)
    yf : ndarray of shape (N,)
        Final y position [m] (0 for non-survivors)
    vxf : ndarray of shape (N,)
        Final x velocity [m/s] (0 for non-survivors)
    vyf : ndarray of shape (N,)
        Final y velocity [m/s] (0 for non-survivors)
    vzf : ndarray of shape (N,)
        Final z velocity [m/s] (0 for non-survivors)
    mask_surv : ndarray of shape (N,), dtype=bool
        Boolean mask: True for survivors, False for collided trajectories

    Notes
    -----
    - Total size N = n_voltages × n_trajectories_per_voltage
    - All arrays are float32 for memory efficiency with neural networks
    - Survivors and non-survivors are mixed in the dataset (use mask_surv to filter)

    Examples
    --------
    >>> data = RawDataFull(X, y, xf, yf, vxf, vyf, vzf, mask_surv)
    >>>
    >>> # Extract only survivors for regression training
    >>> X_survivors = data.X[data.mask_surv]
    >>> Y_survivors = np.column_stack([
    ...     data.xf[data.mask_surv],
    ...     data.yf[data.mask_surv],
    ...     data.vxf[data.mask_surv],
    ...     data.vyf[data.mask_surv],
    ...     data.vzf[data.mask_surv]
    ... ])
    >>>
    >>> # Use full dataset for classification
    >>> X_all = data.X
    >>> y_all = data.y  # 0 or 1
    """

    X: npt.NDArray[np.float32]  # [N,6] = [x0,y0,vx,vy,vz,V]
    y: npt.NDArray[np.int64]  # [N]   0/1 survival
    xf: npt.NDArray[np.float32]  # [N]   final x (0 for non-survivors)
    yf: npt.NDArray[np.float32]  # [N]   final y
    vxf: npt.NDArray[np.float32]  # [N]   final vx (0 for non-survivors)
    vyf: npt.NDArray[np.float32]  # [N]   final vy
    vzf: npt.NDArray[np.float32]  # [N]   final vz
    mask_surv: npt.NDArray[np.bool_]  # [N]   bool mask


def save_raw_data_full(
    data: RawDataFull,
    path: str,
    *,
    meta: Optional[dict] = None,
) -> None:
    """
    Save a RawDataFull instance to a compressed .npz file, including metadata.

    This function stores all numerical arrays and an optional metadata
    dictionary (e.g. voltages, geometry, dataset version) inside a single
    compressed file. The metadata is serialized as a UTF-8 encoded JSON
    string under the key 'metadata'.

    Parameters
    ----------
    data : RawDataFull
        The dataset instance to save.
    path : str
        Output file path (should end with `.npz`).
    meta : dict, optional
        Additional metadata to include, such as simulation parameters:
        {'V_list': [...], 'R': R, 'L': L, 'per_V': {...}}.

    Notes
    -----
    - Arrays are stored with `numpy.savez_compressed` for efficient storage.
    - Metadata is embedded as JSON, ensuring full self-containment.
    - `version` and `timestamp` are always included in the metadata.

    Examples
    --------
    >>> meta = {
    ...     "V_list": [10e3, 15e3, 20e3],
    ...     "R": 0.025,
    ...     "L": 0.10,
    ...     "per_V": {"train": 50000, "val": 20000, "test": 50000},
    ...     "split": "train"
    ... }
    >>> save_raw_data_full(data_tr_full, "data_tr_full.npz", version="1.1", meta=meta)
    """
    metadata = {
        "timestamp": datetime.utcnow().isoformat(),
        **(meta or {}),
    }

    np.savez_compressed(
        path,
        **asdict(data),
        metadata=json.dumps(metadata).encode("utf-8"),
    )


def load_raw_data_full(path: str) -> tuple[RawDataFull, dict]:
    """
    Load a RawDataFull instance and metadata from a compressed .npz file.

    Parameters
    ----------
    path : str
        Path to the `.npz` file created by `save_raw_data_full`.

    Returns
    -------
    (RawDataFull, dict)
        A tuple containing:
        - The reconstructed RawDataFull dataset.
        - The metadata dictionary (including version, timestamp, etc.).

    Notes
    -----
    - The metadata field may include keys such as 'V_list', 'R', 'L',
      and 'per_V' depending on what was provided during saving.
    - Automatically decodes and parses the JSON metadata entry.

    Examples
    --------
    >>> data_loaded, meta = load_raw_data_full("data_tr_full.npz")
    >>> meta["version"]
    '1.1'
    >>> data_loaded.X.shape
    (250000, 6)
    """
    npz = np.load(path, allow_pickle=False)
    metadata = json.loads(npz["metadata"].tobytes().decode("utf-8"))
    data = RawDataFull(
        X=npz["X"],
        y=npz["y"],
        xf=npz["xf"],
        yf=npz["yf"],
        vxf=npz["vxf"],
        vyf=npz["vyf"],
        vzf=npz["vzf"],
        mask_surv=npz["mask_surv"],
    )
    return data, metadata


def build_raw_dataset_full(
    V_list: npt.NDArray[np.float64],
    per_V: int,
    R: float,
    L: float,
    rng: np.random.Generator,
    options: PropagationOptions,  # PropagationOptions - avoid circular import
    particle: Particle,  # Particle - avoid circular import
    gravity: Gravity,  # Gravity - avoid circular import
    trajectory_simulation_fn: Callable,  # Callable - simulation function
) -> RawDataFull:
    """
    Build training dataset by running trajectory simulations across voltages.

    For each voltage in V_list, this function:
    1. Generates random initial conditions (per_V trajectories)
    2. Runs full trajectory propagation through the beamline
    3. Records which trajectories survive and their final states
    4. Accumulates all data into a single RawDataFull container

    This is the primary data generation function for training ML surrogates.

    Parameters
    ----------
    V_list : ndarray of shape (n_voltages,)
        Array of voltages to sample [V]. Typically spans the operating range,
        e.g., np.linspace(5000, 30000, 26) for 5-30 kV in 26 steps.
    per_V : int
        Number of trajectories to simulate per voltage. Recommended: 10,000-50,000
        for good statistics. More trajectories = better ML model accuracy.
    R : float
        Lens bore radius [m]. Used by the simulation function to set apertures.
        Typical: 0.015-0.030 m for molecular beam lenses.
    L : float
        Lens length [m]. Distance over which electric field acts.
        Typical: 0.3-0.8 m for electrostatic quadrupole lenses.
    rng : np.random.Generator
        NumPy random number generator for reproducible sampling.
        Create with: np.random.default_rng(seed)
    options : PropagationOptions
        Configuration for trajectory propagation (cores, verbosity, etc.).
        Example: PropagationOptions(n_cores=8, verbose=False)
    particle : Particle
        Particle species (mass, Stark properties). Example: TlF()
    gravity : Force
        Gravity force vector. Example: Force(0, -9.81*particle.mass, 0)
    trajectory_simulation_fn : Callable
        Function that runs trajectory simulation. Must have signature:

        fn(R, V, L, n_trajectories, rng, options, particle, gravity)
            -> Tuple[Coordinates, Coordinates, Velocities, Velocities, NDArray]

        Returns:
        - coords_init: Initial positions
        - coords_final: Final positions (survivors only)
        - vels_init: Initial velocities
        - vels_final: Final velocities (survivors only)
        - idx_survive: Indices of surviving trajectories

    Returns
    -------
    RawDataFull
        Complete dataset ready for ML training. Contains:
        - X: Input features [x0, y0, vx, vy, vz, V]
        - y: Binary labels (survival)
        - xf, yf, vxf, vyf, vzf: Final states
        - mask_surv: Boolean survival mask

        Total size: len(V_list) × per_V samples

    Notes
    -----
    - Progress bar displayed via tqdm
    - All outputs cast to float32 for ML efficiency
    - Non-survivors have final states set to 0.0
    - Survivors sorted by index for consistency

    Performance
    -----------
    - Time per voltage: ~10-60 seconds (depends on per_V and n_cores)
    - Memory: ~200 MB per 100k trajectories (input + output data)
    - Recommended: Run with n_cores=8 for good CPU utilization

    Examples
    --------
    >>> import numpy as np
    >>> from centrex_trajectories import TlF, PropagationOptions
    >>> from centrex_trajectories.data_structures import Force
    >>>
    >>> # Define voltage range
    >>> V_list = np.linspace(5000, 30000, 26)
    >>>
    >>> # Generate dataset
    >>> data = build_raw_dataset_full(
    ...     V_list=V_list,
    ...     per_V=20000,
    ...     R=0.022,
    ...     L=0.6,
    ...     rng=np.random.default_rng(42),
    ...     options=PropagationOptions(n_cores=8),
    ...     particle=TlF(),
    ...     gravity=Force(0, -9.81 * TlF().mass, 0),
    ...     trajectory_simulation_fn=my_sim_function
    ... )
    >>>
    >>> print(f"Dataset size: {len(data.X):,}")
    >>> print(f"Survival rate: {data.y.mean():.2%}")
    >>> print(f"Survivors: {data.mask_surv.sum():,}")

    See Also
    --------
    RawDataFull : Output container for this function
    physics_augment.augment_with_physics : Add physics-derived features
    classifier_code.train_classifier : Train survival classifier
    regressor_code.train_regressor5_slopes : Train final state regressor
    """
    feats, labels = [], []
    xf_all, yf_all = [], []
    vxf_all, vyf_all, vzf_all = [], [], []  # <-- fixed: three lists
    masks = []
    for V in tqdm.tqdm(V_list, desc="Simulating (raw XY+V dataset)"):
        ci, cf, vi, vf, idx_survive = trajectory_simulation_fn(
            R=R,
            V=V,
            L=L,
            n_trajectories=per_V,
            rng=rng,
            options=options,
            particle=particle,
            gravity=gravity,
        )
        idx_survive = np.array(sorted(idx_survive), dtype=int)
        mask = np.zeros(per_V, dtype=bool)
        mask[idx_survive] = True

        X = np.column_stack([ci.x, ci.y, vi.vx, vi.vy, vi.vz, np.full(per_V, V, float)])
        y = mask.astype(np.int64)

        xf = np.zeros(per_V, float)
        yf = np.zeros(per_V, float)
        vxf = np.zeros(per_V, float)
        vyf = np.zeros(per_V, float)
        vzf = np.zeros(per_V, float)
        if idx_survive.size:
            xf[idx_survive] = cf.x
            yf[idx_survive] = cf.y
            vxf[idx_survive] = vf.vx
            vyf[idx_survive] = vf.vy
            vzf[idx_survive] = vf.vz

        feats.append(X)
        labels.append(y)
        masks.append(mask)
        xf_all.append(xf)
        yf_all.append(yf)
        vxf_all.append(vxf)
        vyf_all.append(vyf)
        vzf_all.append(vzf)

    X = np.vstack(feats).astype(np.float32)
    y = np.concatenate(labels)
    xf = np.concatenate(xf_all).astype(np.float32)
    yf = np.concatenate(yf_all).astype(np.float32)
    vxf = np.concatenate(vxf_all).astype(np.float32)
    vyf = np.concatenate(vyf_all).astype(np.float32)
    vzf = np.concatenate(vzf_all).astype(np.float32)
    m = np.concatenate(masks).astype(bool)
    return RawDataFull(X, y, xf, yf, vxf, vyf, vzf, m)
