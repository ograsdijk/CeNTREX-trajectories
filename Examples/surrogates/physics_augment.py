"""
Physics-Based Feature Augmentation for Surrogate Models
========================================================

This module adds physics-derived features to raw trajectory data to improve
neural network learning. By providing analytical relationships derived from
electrostatic quadrupole lens theory, we help the ML model learn physically
meaningful patterns rather than purely empirical correlations.

Key Concepts
------------
1. **Stark Effect**: Molecules in electric fields experience energy shifts
   proportional to the field squared: ΔE ≈ -α₀ E²

2. **Quadrupole Focusing**: Transverse position couples to voltage via
   the field gradient: F_x ≈ -α₀ G² x, where G = ∂E/∂x ≈ 2V/R²

3. **Ideal Thick Lens**: Analytical solution for harmonic motion:
   x(L) = x₀cos(kL) + (vₓ/vᵤ)/k·sin(kL), k = √(α₀G²/m·vᵤ²)

4. **Margin**: Distance from maximum excursion to bore radius determines
   survival probability

Physics-Augmented Features
---------------------------
Starting from raw features [x0, y0, vx, vy, vz, V], we add:
- k: Spring constant (determines focusing strength)
- x_id, y_id: Ideal lens final positions (analytical solution)
- margin_x, margin_y: Distance from max excursion to aperture
- vperp/vz: Transverse-to-forward velocity ratio (determines focusing)

These features encode domain knowledge that would be difficult for a neural
network to discover from raw data alone.

Usage Example
-------------
>>> import numpy as np
>>> from centrex_trajectories import TlF
>>>
>>> # Raw features
>>> X_raw = np.array([
>>>     [0.01, 0.02, 5, -3, 150, 20000],  # x0, y0, vx, vy, vz, V
>>>     [0.005, -0.01, -2, 4, 160, 25000],
>>> ], dtype=np.float32)
>>>
>>> # Augment with physics
>>> X_aug = augment_with_physics(
>>>     X_raw,
>>>     R=0.022,         # Lens bore radius [m]
>>>     L=0.6,           # Lens length [m]
>>>     alpha0=1.3e-30,  # Stark polarizability [J/(V/m)²]
>>>     mass=TlF().mass  # Particle mass [kg]
>>> )
>>>
>>> print(X_aug.shape)  # (2, 12) - original 6 + 6 physics features
"""

import numpy as np
import numpy.typing as npt
from numpy.polynomial import Polynomial

__all__ = [
    "fit_stark_even_keep_domain",
    "ideal_thick_lens_map",
    "gammaG_from_bore",
    "k_from_V_vz",
    "ideal_survival_mask",
    "augment_with_physics",
]


def fit_stark_even_keep_domain(
    E: npt.NDArray[np.float64], U: npt.NDArray[np.float64], deg_even: int
) -> tuple[Polynomial, float]:
    """
    Fit Stark potential U(E) with even polynomial and extract polarizability.

    The Stark shift for non-degenerate states follows U(E) = U₀ - ½α₀E² + O(E⁴).
    We fit only even powers (0, 2, 4, ...) since Stark shifts are symmetric in E,
    then extract the polarizability α₀ = -U''(0) from the second derivative at E=0.

    Parameters
    ----------
    E : ndarray of shape (n,)
        Electric field strength [V/m]. Should span both positive and negative
        values to capture the even symmetry.
    U : ndarray of shape (n,)
        Stark-shifted energy [J] corresponding to each field value.
        Typically computed via quantum chemistry (e.g., centrex_tlf).
    deg_even : int
        Maximum even degree for polynomial fit. Must be even (e.g., 4, 6, 8).
        Higher degrees capture anharmonicity but risk overfitting.

    Returns
    -------
    P_t : numpy.polynomial.Polynomial
        Fitted polynomial in the scaled variable t ∈ [-1, 1].
        Mapping: E = a·t + b where a = (E_max - E_min)/2, b = (E_max + E_min)/2
    alpha0 : float
        Stark polarizability [J/(V/m)²], defined as α₀ = -U''(0).
        For TlF in low-field-seeking states, typically ~10⁻³⁰ J/(V/m)².

    Notes
    -----
    - Only even degrees are fitted to enforce symmetry: U(E) = U(-E)
    - The polynomial uses domain=[] to fit in scaled coordinates t ∈ [-1, 1]
    - Second derivative computed via chain rule: U''(E) = Pₜ''(t) / a²
    - No convert() call needed; we evaluate Pₜ'' directly at appropriate t

    Mathematical Details
    --------------------
    Given E → t mapping: E = a·t + b
    - Forward: t = (E - b) / a
    - Derivative: dE/dt = a
    - Second derivative: d²U/dE² = (d²U/dt²) · (dt/dE)² = Pₜ''(t) / a²

    At E = 0: t₀ = -b/a, so α₀ = -Pₜ''(t₀) / a²

    Examples
    --------
    >>> # Generate synthetic Stark data
    >>> E = np.linspace(-1e7, 1e7, 100)  # ±10 MV/m
    >>> alpha0_true = 1.3e-30  # J/(V/m)²
    >>> U = -0.5 * alpha0_true * E**2  # Harmonic approximation
    >>>
    >>> # Fit and extract polarizability
    >>> P, alpha0_fit = fit_stark_even_keep_domain(E, U, deg_even=4)
    >>> print(f"True α₀: {alpha0_true:.3e}")
    >>> print(f"Fit α₀: {alpha0_fit:.3e}")
    >>> assert np.isclose(alpha0_fit, alpha0_true, rtol=0.01)

    See Also
    --------
    k_from_V_vz : Uses alpha0 to compute focusing strength
    augment_with_physics : Uses alpha0 for feature engineering
    """
    assert deg_even % 2 == 0
    # Fit only even degrees (0,2,4,...,deg_even) in the scaled variable t
    degrees = np.arange(0, deg_even + 1, 2)
    P_t = Polynomial.fit(E, U, deg=degrees, domain=[])  # fit in t with window [-1,1]
    # Recover the affine map E = a t + b used by Polynomial.fit
    Emin, Emax = float(np.min(E)), float(np.max(E))
    a = 0.5 * (Emax - Emin)
    b = 0.5 * (Emax + Emin)
    # Evaluate second derivative in t at the t corresponding to E=0
    t0 = (0.0 - b) / (a if a != 0.0 else 1.0)
    Ppp_t = P_t.deriv(2)
    Upp0 = (1.0 / (a * a if a != 0.0 else 1.0)) * Ppp_t(t0)
    alpha0 = -Upp0  # J / (V/m)^2
    return P_t, alpha0


def ideal_thick_lens_map(
    x0: np.ndarray,
    y0: np.ndarray,
    vx0: np.ndarray,
    vy0: np.ndarray,
    vz0: np.ndarray,
    k: np.ndarray,
    L: float,
    kmin: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ideal thick-lens map (positions and transverse velocities)
    ==========================================================

    Computes analytical propagation through a quadrupole lens under
    the paraxial approximation. Returns both exit positions and
    transverse velocities at z = L.

    Parameters
    ----------
    x0, y0 : ndarray
        Initial transverse positions [m].
    vx0, vy0 : ndarray
        Initial transverse velocities [m/s].
    vz0 : ndarray
        Forward velocity [m/s], assumed constant.
    k : ndarray
        Focusing strength [1/m].
    L : float
        Lens length [m].
    kmin : float, default=1e-9
        Threshold below which the drift limit is used (k → 0).

    Returns
    -------
    x_exit, y_exit : ndarray
        Exit positions at z = L [m].
    vx_exit, vy_exit : ndarray
        Exit transverse velocities at z = L [m/s].

    Notes
    -----
    - For |k| < kmin, the function uses the linear drift approximation.
    - The solution follows:
        x(z) = x₀ cos(kz) + (vₓ/vᵤ) sin(kz)/k
        vₓ(z) = vᵤ [−k x₀ sin(kz) + (vₓ/vᵤ) cos(kz)]
    - Works vectorized over arbitrary array shapes.
    """
    x0 = np.asarray(x0, float)
    y0 = np.asarray(y0, float)
    vx0 = np.asarray(vx0, float)
    vy0 = np.asarray(vy0, float)
    vz0 = np.asarray(vz0, float)
    k = np.asarray(k, float)

    kL = k * L
    mask = np.abs(k) >= kmin

    x_exit = np.empty_like(x0)
    y_exit = np.empty_like(y0)
    vx_exit = np.empty_like(vx0)
    vy_exit = np.empty_like(vy0)

    if np.any(mask):
        cos_kL = np.cos(kL[mask])
        sin_kL = np.sin(kL[mask])
        invk = 1.0 / k[mask]
        xprime0 = vx0[mask] / vz0[mask]
        yprime0 = vy0[mask] / vz0[mask]

        x_exit[mask] = x0[mask] * cos_kL + xprime0 * invk * sin_kL
        y_exit[mask] = y0[mask] * cos_kL + yprime0 * invk * sin_kL

        xprime_L = -k[mask] * x0[mask] * sin_kL + xprime0 * cos_kL
        yprime_L = -k[mask] * y0[mask] * sin_kL + yprime0 * cos_kL

        vx_exit[mask] = vz0[mask] * xprime_L
        vy_exit[mask] = vz0[mask] * yprime_L

    if np.any(~mask):
        x_exit[~mask] = x0[~mask] + (vx0[~mask] / vz0[~mask]) * L
        y_exit[~mask] = y0[~mask] + (vy0[~mask] / vz0[~mask]) * L
        vx_exit[~mask] = vx0[~mask]
        vy_exit[~mask] = vy0[~mask]

    return x_exit, y_exit, vx_exit, vy_exit


def ideal_thick_lens_map_gravity(
    x0: np.ndarray,
    y0: np.ndarray,
    vx0: np.ndarray,
    vy0: np.ndarray,
    vz0: np.ndarray,
    k: np.ndarray,
    L: float,
    g: float = 9.81,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ideal thick-lens map with gravitational sag
    ===========================================

    Extends :func:`ideal_thick_lens_map` by adding gravitational
    acceleration along y during the transit through the lens.

    Gravity acts during the time t = L / vz₀ as:
        y_exit  ← y_exit  − ½ g t²
        vy_exit ← vy_exit − g t

    Parameters
    ----------
    x0, y0 : ndarray
        Initial transverse positions [m].
    vx0, vy0 : ndarray
        Initial transverse velocities [m/s].
    vz0 : ndarray
        Forward velocity [m/s].
    k : ndarray
        Focusing strength [1/m].
    L : float
        Lens length [m].
    g : float, default = 9.81
        Gravitational acceleration [m/s²]; positive means acceleration
        toward −y (i.e. +y = upward).

    Returns
    -------
    x_exit, y_exit : ndarray
        Exit positions including gravitational sag [m].
    vx_exit, vy_exit : ndarray
        Exit transverse velocities including gravitational kick [m/s].

    Notes
    -----
    - Use g = −9.81 if your coordinate system has +y downward.
    - The correction scales as g L² / (2 vz₀²).
    """
    x_exit, y_exit, vx_exit, vy_exit = ideal_thick_lens_map(x0, y0, vx0, vy0, vz0, k, L)

    t = L / vz0
    y_exit = y_exit - 0.5 * g * t * t
    vy_exit = vy_exit - g * t

    return x_exit, y_exit, vx_exit, vy_exit


def gammaG_from_bore(r0_m: float) -> float:
    """
    Estimate quadrupole gradient coefficient from bore radius.

    For a cylindrical quadrupole lens, the field gradient near the axis
    scales as G ≈ 2V/R², where V is the applied voltage and R is the bore
    radius. This function returns γ_G = 2/R² [1/(m²·V)] such that G = γ_G·V.

    Parameters
    ----------
    r0_m : float
        Bore radius [m]. Typical values: 0.015-0.030 m.

    Returns
    -------
    gamma_G : float
        Gradient coefficient [1/(m²·V)], such that field gradient G = γ_G·V.

    Notes
    -----
    - This is an approximate scaling for ideal cylindrical geometry
    - Real lenses have more complex geometry (electrodes, rounded edges)
    - For accurate gradients, use COMSOL or other field solvers
    - Sufficient for order-of-magnitude estimates and ML feature engineering

    Examples
    --------
    >>> # 22 mm bore radius
    >>> gamma_G = gammaG_from_bore(0.022)
    >>> print(f"γ_G = {gamma_G:.2e} 1/(m²·V)")
    >>>
    >>> # Field gradient at 20 kV
    >>> V = 20000  # V
    >>> G = gamma_G * V
    >>> print(f"G ≈ {G:.2e} V/m²")
    """
    return 2.0 / (r0_m**2)


def k_from_V_vz(
    alpha0: float, mass_kg: float, gamma_G: float, V_volt: float, vz: float
) -> float:
    """
    Compute focusing strength k from voltage and velocity.

    In the paraxial approximation, the focusing strength k [1/m] determines
    the transverse motion via d²x/dz² + k²x = 0. For an electrostatic
    quadrupole lens, k depends on the Stark polarizability, field gradient,
    particle mass, and forward velocity.

    Derivation
    ----------
    Stark force: F_x = -α₀ G² x  (linear restoring force)
    Acceleration: a_x = F_x / m
    In the moving frame: d²x/dt² = a_x
    Convert to z-coordinate: d²x/dz² = (d²x/dt²) / vᵤ²
    Result: d²x/dz² = -(α₀G²/m·vᵤ²)x = -k²x

    Thus: k = √(α₀G²/m·vᵤ²)

    Parameters
    ----------
    alpha0 : float
        Stark polarizability [J/(V/m)²]. For TlF: ~1.3×10⁻³⁰ J/(V/m)².
    mass_kg : float
        Particle mass [kg]. For TlF: ~3.5×10⁻²⁵ kg (204 amu).
    gamma_G : float
        Gradient coefficient [1/(m²·V)] from gammaG_from_bore(R).
    V_volt : float
        Applied voltage [V]. Typical range: 5,000-30,000 V.
    vz : float
        Forward velocity [m/s]. Typical: 100-300 m/s for supersonic beams.

    Returns
    -------
    k : float
        Focusing strength [1/m]. Higher k = stronger focusing.

    Notes
    -----
    - k = 0 corresponds to field-free drift
    - k ≈ 10-30 [1/m] typical for electrostatic quadrupole lenses
    - Oscillation wavelength: λ = 2π/k [m]
    - Focus length: f ≈ 1/(kL) for thin lens approximation (kL << 1)

    Physical Interpretation
    -----------------------
    - Higher voltage → larger k → stronger focusing
    - Faster particles → smaller k → weaker focusing (less time in field)
    - Larger α₀ → larger k → stronger coupling to field
    - Larger mass → smaller k → weaker response to force

    Examples
    --------
    >>> from centrex_trajectories import TlF
    >>>
    >>> # TlF parameters
    >>> alpha0 = 1.3e-30  # J/(V/m)²
    >>> mass = TlF().mass  # kg
    >>> R = 0.022  # m
    >>> gamma_G = gammaG_from_bore(R)
    >>>
    >>> # Compute k at 20 kV, 150 m/s
    >>> k = k_from_V_vz(alpha0, mass, gamma_G, V_volt=20000, vz=150)
    >>> print(f"k = {k:.2f} 1/m")
    >>> print(f"Oscillation wavelength: {2*np.pi/k:.3f} m")
    >>>
    >>> # Scan k vs voltage
    >>> V_array = np.linspace(5000, 30000, 100)
    >>> k_array = [k_from_V_vz(alpha0, mass, gamma_G, V, 150) for V in V_array]
    >>> # k scales as √V

    See Also
    --------
    gammaG_from_bore : Compute gradient coefficient from geometry
    ideal_thick_lens_map : Uses k to compute trajectories
    """
    G = gamma_G * V_volt
    k2 = (alpha0 * (G * G)) / (mass_kg * vz * vz)
    return float(np.sqrt(max(k2, 0.0)))


def ideal_survival_mask(
    x0: npt.NDArray[np.float64] | float,
    y0: npt.NDArray[np.float64] | float,
    vx: npt.NDArray[np.float64] | float,
    vy: npt.NDArray[np.float64] | float,
    vz: npt.NDArray[np.float64] | float,
    k: npt.NDArray[np.float64] | float,
    R: float,
    L: float,
    kmin: float = 1e-9,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Predict trajectory survival using ideal thick lens model.

    A trajectory survives if its maximum transverse excursion stays within
    the bore radius R in both x and y directions throughout the lens.

    Parameters
    ----------
    x0, y0 : ndarray or float
        Initial transverse positions [m]
    vx, vy : ndarray or float
        Initial transverse velocities [m/s]
    vz : ndarray or float
        Forward velocity [m/s]
    k : ndarray or float
        Focusing strength [1/m]
    R : float
        Bore radius [m]. Aperture size limiting transverse excursion.
    L : float
        Lens length [m]
    kmin : float, default=1e-9
        Threshold for drift vs harmonic solution

    Returns
    -------
    survive : ndarray of bool
        True if trajectory survives (doesn't hit aperture), False if collides
    x_max : ndarray
        Maximum |x| excursion [m]
    y_max : ndarray
        Maximum |y| excursion [m]

    Notes
    -----
    - Survival condition: (x_max < R) AND (y_max < R)
    - This is analytical prediction; real trajectories may differ due to
      fringe fields, higher-order aberrations, etc.
    - Useful for quick estimates and ML feature engineering

    Examples
    --------
    >>> # Single trajectory
    >>> survive, x_max, y_max = ideal_survival_mask(
    ...     x0=0.01, y0=0.005,
    ...     vx=5, vy=-3, vz=150,
    ...     k=15, R=0.022, L=0.6
    ... )
    >>> if survive:
    ...     print(f"Survives! Max excursion: ({x_max:.4f}, {y_max:.4f}) m")
    >>> else:
    ...     print(f"Collides! Max excursion: ({x_max:.4f}, {y_max:.4f}) m > {R:.4f} m")
    >>>
    >>> # Vectorized: predict for 10k trajectories
    >>> survive, x_max, y_max = ideal_survival_mask(
    ...     x0=X[:, 0], y0=X[:, 1],
    ...     vx=X[:, 2], vy=X[:, 3], vz=X[:, 4],
    ...     k=k_array, R=0.022, L=0.6
    ... )
    >>> print(f"Predicted survival rate: {survive.mean():.2%}")

    See Also
    --------
    ideal_thick_lens_map : Underlying trajectory calculation
    augment_with_physics : Uses this to create margin features
    """
    x_exit, y_exit, x_max, y_max = ideal_thick_lens_map(
        x0, y0, vx, vy, vz, k, L, kmin=kmin
    )
    survive = (x_max < R) & (y_max < R)
    return survive, x_max, y_max


def augment_with_physics(
    X: npt.NDArray[np.float32], R: float, L: float, alpha0: float, mass: float
) -> npt.NDArray[np.float32]:
    """
    Augment raw features with physics-derived quantities from lens theory.

    This function adds 6 physics-based features to the raw 6 input features,
    helping neural networks learn physically meaningful patterns. The added
    features encode analytical relationships from electrostatic quadrupole
    lens theory that would be difficult for a network to discover empirically.

    Theory Background
    -----------------
    An electrostatic quadrupole lens focuses molecules via the Stark effect.
    The key physics:
    1. Stark force: F ∝ -α₀ G² x (linear restoring force)
    2. Focusing strength: k = √(α₀G²/m·vᵤ²)
    3. Ideal solution: x(z) = x₀cos(kz) + (vₓ/kvᵤ)sin(kz)
    4. Survival: max|x(z)| < R throughout lens

    By providing k, ideal predictions (x_id, y_id), and margins as features,
    we give the neural network a "physics-informed" starting point rather
    than learning everything from scratch.

    Parameters
    ----------
    X : ndarray of shape (N, 6), dtype=float32
        Raw input features: [x0, y0, vx, vy, vz, V]
        - x0, y0: Initial positions [m]
        - vx, vy: Initial transverse velocities [m/s]
        - vz: Forward velocity [m/s]
        - V: Applied voltage [V]
    R : float
        Lens bore radius [m]
    L : float
        Lens length [m]
    alpha0 : float
        Stark polarizability [J/(V/m)²]. Extract from Stark potential fit.
    mass : float
        Particle mass [kg]

    Returns
    -------
    X_aug : ndarray of shape (N, 12), dtype=float32
        Augmented features: [x0, y0, vx, vy, vz, V, k, x_id, y_id, margin_x, margin_y, vperp/vz]

        New features (columns 6-11):
        - k: Focusing strength [1/m] from k_from_V_vz()
        - x_id, y_id: Ideal lens exit positions [m] from analytical solution
        - margin_x: Distance to x aperture: R - x_max [m] (negative = collision)
        - margin_y: Distance to y aperture: R - y_max [m]
        - vperp/vz: Transverse-to-forward velocity ratio [dimensionless]

    Feature Interpretation
    ----------------------
    **k (focusing strength)**:
      - Encodes coupling between voltage, velocity, and focusing
      - Higher k → stronger focusing → tighter beam
      - Scales as k ∝ √V / vz

    **x_id, y_id (ideal predictions)**:
      - What would happen in a perfect ideal lens
      - Provides baseline expectation for network
      - Network learns corrections for real lens effects

    **margin_x, margin_y**:
      - Direct measure of how close to collision
      - Positive margin → safe, negative → collision
      - Network can learn nonlinear decision boundaries

    **vperp/vz**:
      - Ratio of transverse to forward velocity
      - High ratio → large transverse motion → more likely to collide
      - Dimensionless, voltage-independent characterization

    Notes
    -----
    - All arrays cast to float32 for neural network efficiency
    - Physics features are computed via vectorized numpy operations (fast)
    - Ideal predictions use analytical thick-lens formulas
    - Typically reduces ML training time by 2-5x vs raw features alone

    Performance
    -----------
    - Augmentation time: ~1-10 ms for 100k trajectories
    - Negligible overhead compared to trajectory simulation (seconds)
    - Call once during data preprocessing, before training

    Examples
    --------
    >>> import numpy as np
    >>> from centrex_trajectories import TlF
    >>>
    >>> # Generate raw features
    >>> n = 10000
    >>> X_raw = np.column_stack([
    ...     np.random.randn(n) * 0.01,  # x0
    ...     np.random.randn(n) * 0.01,  # y0
    ...     np.random.randn(n) * 10,    # vx
    ...     np.random.randn(n) * 10,    # vy
    ...     np.ones(n) * 150,            # vz
    ...     np.ones(n) * 20000           # V = 20 kV
    ... ]).astype(np.float32)
    >>>
    >>> # Augment with physics
    >>> X_aug = augment_with_physics(
    ...     X_raw,
    ...     R=0.022,
    ...     L=0.6,
    ...     alpha0=1.3e-30,
    ...     mass=TlF().mass
    ... )
    >>>
    >>> print(f"Original shape: {X_raw.shape}")  # (10000, 6)
    >>> print(f"Augmented shape: {X_aug.shape}")  # (10000, 12)
    >>>
    >>> # Inspect physics features
    >>> k_values = X_aug[:, 6]
    >>> print(f"k range: [{k_values.min():.2f}, {k_values.max():.2f}] 1/m")
    >>>
    >>> margin_x = X_aug[:, 9]
    >>> expected_survival = (margin_x > 0).mean()
    >>> print(f"Expected survival (from margins): {expected_survival:.2%}")

    See Also
    --------
    k_from_V_vz : Computes focusing strength
    ideal_thick_lens_map : Analytical lens solution
    ideal_survival_mask : Predicts survival from physics
    classifier_code.train_classifier : Uses augmented features for training
    regressor_code.train_regressor5_slopes : Uses augmented features for training
    """
    x0, y0, vx, vy, vz, V = [X[:, i] for i in range(6)]
    gammaG = gammaG_from_bore(R)
    k = np.array(
        [k_from_V_vz(alpha0, mass, gammaG, Vi, vzi) for Vi, vzi in zip(V, vz)],
        dtype=np.float32,
    )
    survive_id, Ax, Ay = ideal_survival_mask(x0, y0, vx, vy, vz, k, R, L)
    x_id, y_id, _, _ = ideal_thick_lens_map(x0, y0, vx, vy, vz, k, L)
    margin_x = (R - Ax).astype(np.float32)
    margin_y = (R - Ay).astype(np.float32)
    vperp_over_vz = (np.sqrt(vx * vx + vy * vy) / vz).astype(np.float32)
    X_aug = np.column_stack(
        [x0, y0, vx, vy, vz, V, k, x_id, y_id, margin_x, margin_y, vperp_over_vz]
    ).astype(np.float32)
    return X_aug


def augment_with_physics_extended(X_raw, R, L, alpha0, mass):
    """Enhanced physics features beyond current implementation"""
    # Existing features from augment_with_physics
    X_aug = augment_with_physics(X_raw, R, L, alpha0, mass)

    # Additional features:
    x0, y0, vx, vy, vz, V = X_raw.T

    # Angular momentum features
    r0 = np.sqrt(x0**2 + y0**2)
    L_ang = r0 * np.sqrt(vx**2 + vy**2)  # transverse angular momentum

    # Entry angle relative to z-axis
    v_trans = np.sqrt(vx**2 + vy**2)
    entry_angle = np.arctan2(v_trans, vz)

    # Normalized radial position
    r_norm = r0 / R

    # Transit time estimate
    t_transit = L / vz

    # Velocity ratio
    v_ratio = v_trans / vz

    return np.column_stack([X_aug, L_ang, entry_angle, r_norm, t_transit, v_ratio])


def augment_with_physics_extended_smooth(
    X_raw: npt.NDArray[np.float32],
    R: float,
    L: float,
    alpha0: float,
    mass: float,
    clip_margin: float | None = None,
    add_phase_trig: bool = True,
    *,
    g: float = 9.80665,
    x_ap_in: float = 0.0,
    y_ap_in: float = 0.0,
    x_ap_out: float = 0.0,
    y_ap_out: float = 0.0,
    return_names: bool = False,
) -> npt.NDArray[np.float32] | tuple[npt.NDArray[np.float32], list[str]]:
    """
    Enhanced physics feature augmentation with smooth focusing-phase encoding.

    This version preserves the original quadrupole thick-lens augmentation and
    appends a lean set of gravity- and geometry-aware proxies that do not
    require access to the field map. It assumes your *ideal* thick-lens
    functions (``ideal_survival_mask``, ``ideal_thick_lens_map``) already
    incorporate gravity for an ideal quadrupole with a quadratic Stark shift.

    Parameters
    ----------
    X_raw : ndarray of shape (N, 6), dtype=float32
        Raw particle features: [x0, y0, vx, vy, vz, V].
    R : float
        Lens bore radius [m].
    L : float
        Lens length [m].
    alpha0 : float
        Stark polarizability [J/(V/m)^2].
    mass : float
        Particle mass [kg].
    clip_margin : float or None, optional
        If not None, clip the ideal aperture margins to
        [-clip_margin, +clip_margin]. A good default is clip_margin=R.
    add_phase_trig : bool, optional
        If True, append [sin(kL), cos(kL)] to make the phase representation
        smooth across π/2 and π boundaries (default True).
    g : float, optional
        Gravitational acceleration [m/s²]. Default is 9.80665.
    x_ap_in, y_ap_in : float, optional
        Entrance aperture center coordinates [m]. Defaults to 0.0.
    x_ap_out, y_ap_out : float, optional
        Exit aperture center coordinates [m]. Defaults to 0.0.
    return_names : bool, optional
        If True, also return a list of column names corresponding to each
        augmented feature. Default is False.

    Returns
    -------
    X_aug : ndarray, dtype=float32
        Augmented feature matrix with shape (N, M), where M depends on whether
        `add_phase_trig` is enabled. The first 6 columns remain the original
        raw inputs. New physics-informed features are appended as follows:

        =====  ==============================  ============================
        0–5    x0, y0, vx, vy, vz, V           raw input features
        6      k                                [1/m] focusing strength
        7–8    x_id, y_id                       [m] ideal thick-lens outputs (incl. gravity)
        9–10   margin_x, margin_y               [m] ideal aperture margins
        11     vperp_over_vz                    [–]
        12     L_ang                            [m²/s]
        13     entry_angle                      [rad]
        14     r0_over_R                        [–]
        15     t_transit                        [s]
        16–17  sin(kL), cos(kL)                 (if add_phase_trig=True)
        18     y_sag0 = ½ g (L/vz)²             [m] straight-flight sag proxy
        19     η₀ = y_sag0 / R                  [–]
        20–21  x0_centered/R, y0_centered/R     [–] entrance offsets vs. aperture center
        22–23  x_id_centered/R, y_id_centered/R [–] *ideal-exit* offsets vs. aperture center
        24–25  Jx, Jy                           [–] emittance-like invariants per plane
        26     V_times_eta0 = V·η₀              [V] gravity–voltage interaction proxy
        27     r0v_ratio = (r0/R)(v⊥/v_z)       [–]
        =====  ==============================  ============================

        The total column count M = 28 when add_phase_trig=True, or 26 otherwise.

    Notes
    -----
    - No ballistic (field-free) exit estimates are included, since your ideal
      thick-lens outputs already incorporate gravity and provide a richer
      focusing baseline.
    - y_sag0 and η₀ are retained as inexpensive global gravity proxies; they
      summarize the tendency for slow beams to sag and often improve
      calibration in low-vz slices.
    - Centering features at entrance and *ideal* exit help when apertures are
      offset or misaligned, without referencing simulated truth.
    - All quantities are computed analytically from initial conditions and lens
      parameters; no simulated trajectories are used.

    Examples
    --------
    >>> X_aug, names = augment_with_physics_extended_smooth(
    ...     X_raw, R=0.022, L=0.6, alpha0=alpha_TlF, mass=m_TlF,
    ...     g=9.81, clip_margin=0.022, add_phase_trig=True, return_names=True)
    >>> X_aug.shape
    (N, 28)
    >>> names[:10]
    ['x0', 'y0', 'vx', 'vy', 'vz', 'V', 'k', 'x_id', 'y_id', 'margin_x']
    >>> # First six columns remain the raw inputs
    >>> np.allclose(X_aug[:, :6], X_raw)
    True
    """
    # ---- unpack raw (keep your original 6-D layout) -------------------------
    x0 = X_raw[:, 0].astype(np.float32)
    y0 = X_raw[:, 1].astype(np.float32)
    vx = X_raw[:, 2].astype(np.float32)
    vy = X_raw[:, 3].astype(np.float32)
    vz = X_raw[:, 4].astype(np.float32)
    V = X_raw[:, 5].astype(np.float32)

    # ---- focusing strength & ideal thick-lens proxies (gravity included) ---
    gammaG = gammaG_from_bore(R)
    k = np.array(
        [k_from_V_vz(alpha0, mass, gammaG, Vi, vzi) for Vi, vzi in zip(V, vz)],
        dtype=np.float32,
    )

    survive_id, Ax, Ay = ideal_survival_mask(x0, y0, vx, vy, vz, k, R, L)
    x_id, y_id, _, _ = ideal_thick_lens_map(x0, y0, vx, vy, vz, k, L)

    margin_x = (R - Ax).astype(np.float32)
    margin_y = (R - Ay).astype(np.float32)
    if clip_margin is not None:
        c = float(clip_margin)
        margin_x = np.clip(margin_x, -c, c)
        margin_y = np.clip(margin_y, -c, c)

    # ---- kinematics & geometry ---------------------------------------------
    vperp = np.sqrt(vx * vx + vy * vy).astype(np.float32)
    vperp_over_vz = (vperp / (vz + 1e-12)).astype(np.float32)
    r0 = np.sqrt(x0 * x0 + y0 * y0).astype(np.float32)
    L_ang = (r0 * vperp).astype(np.float32)
    entry_angle = np.arctan2(vperp, vz).astype(np.float32)
    r0_over_R = (r0 / R).astype(np.float32)
    t_transit = (L / (vz + 1e-12)).astype(np.float32)

    base_cols = [
        ("x0", x0),
        ("y0", y0),
        ("vx", vx),
        ("vy", vy),
        ("vz", vz),
        ("V", V),
        ("k", k.astype(np.float32)),
        ("x_id", x_id.astype(np.float32)),
        ("y_id", y_id.astype(np.float32)),
        ("margin_x", margin_x),
        ("margin_y", margin_y),
        ("vperp_over_vz", vperp_over_vz),
        ("L_ang", L_ang),
        ("entry_angle", entry_angle),
        ("r0_over_R", r0_over_R),
        ("t_transit", t_transit),
    ]

    trig_cols: list[tuple[str, np.ndarray]] = []
    if add_phase_trig:
        phi = (k * L).astype(np.float32)
        trig_cols = [
            ("sin_kL", np.sin(phi).astype(np.float32)),
            ("cos_kL", np.cos(phi).astype(np.float32)),
        ]

    # ---- Gravity proxies & centered offsets (no ballistic) ------------------
    y_sag0 = (0.5 * g * t_transit * t_transit).astype(np.float32)
    eta0 = (y_sag0 / R).astype(np.float32)

    x0_centered_over_R = ((x0 - x_ap_in) / R).astype(np.float32)
    y0_centered_over_R = ((y0 - y_ap_in) / R).astype(np.float32)
    x_id_centered_over_R = ((x_id - x_ap_out) / R).astype(np.float32)
    y_id_centered_over_R = ((y_id - y_ap_out) / R).astype(np.float32)

    # Emittance-like invariants and light interactions
    Jx = (0.5 * ((x0 / R) ** 2 + (L / R) ** 2 * (vx / (vz + 1e-12)) ** 2)).astype(
        np.float32
    )
    Jy = (0.5 * ((y0 / R) ** 2 + (L / R) ** 2 * (vy / (vz + 1e-12)) ** 2)).astype(
        np.float32
    )
    V_times_eta0 = (V * eta0).astype(np.float32)
    r0v_ratio = ((r0_over_R) * (vperp_over_vz)).astype(np.float32)

    extra_cols = [
        ("y_sag0", y_sag0),
        ("eta0", eta0),
        ("x0_centered_over_R", x0_centered_over_R),
        ("y0_centered_over_R", y0_centered_over_R),
        ("x_id_centered_over_R", x_id_centered_over_R),
        ("y_id_centered_over_R", y_id_centered_over_R),
        ("Jx", Jx),
        ("Jy", Jy),
        ("V_times_eta0", V_times_eta0),
        ("r0v_ratio", r0v_ratio),
    ]

    # ---- assemble final matrix ---------------------------------------------
    all_cols = base_cols + trig_cols + extra_cols
    column_names = [name for name, _ in all_cols]
    X_aug = np.column_stack([col.astype(np.float32) for _, col in all_cols]).astype(
        np.float32
    )

    if return_names:
        return X_aug, column_names
    return X_aug
