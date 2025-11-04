"""
Random Coordinate and Velocity Generation
==========================================

This module provides functions for generating random initial conditions for
particle trajectories in molecular beam simulations. It implements statistically
correct sampling methods for common distributions.

Sampling Methods
----------------
**Spatial Distributions:**
1. **Normal/Gaussian circle**: Models thermal sources or diffuse beams
2. **Uniform circle**: Models uniform illumination or well-collimated sources

**Velocity Distributions:**
1. **Normal/Gaussian**: Models thermal velocity distributions (Maxwell-Boltzmann)

Mathematical Background
-----------------------
**Uniform Circle Sampling:**
Simply sampling r ~ U(0, R) and θ ~ U(0, 2π) produces incorrect distribution
(higher density at center). Correct method:
    r = √(U(0, R²))    [U = uniform random variable]
    θ = U(0, 2π)
This ensures uniform density per unit area.

**Normal Circle Sampling:**
Independent normal distributions in x and y:
    x ~ N(μₓ, σ)
    y ~ N(μᵧ, σ)
Produces 2D Gaussian with circular symmetry (isotropy).

Random Number Generation
-------------------------
All functions accept an optional `rng` parameter (np.random.Generator) for:
- Reproducibility: Pass np.random.default_rng(seed) for fixed seed
- Performance: Reuse same generator for multiple calls
- Parallel safety: Use separate generators per thread

If `rng=None`, creates a new generator with system entropy.

Examples
--------
>>> import numpy as np
>>> from centrex_trajectories.random_generation import *
>>>
>>> # Reproducible generation with seed
>>> rng = np.random.default_rng(42)
>>> coords = generate_random_coordinates_uniform_circle(
...     radius=1e-3, number=10000, x=0, y=0, z=0, rng=rng
... )
>>>
>>> # Thermal velocity distribution
>>> vels = generate_random_velocities_normal(
...     vx=0, vy=0, vz=200,  # 200 m/s forward
...     sigma_vx=50, sigma_vy=50, sigma_vz=10,  # Velocity spreads
...     number=10000, rng=rng
... )
"""

from typing import List, Optional

import numpy as np

from .data_structures import Coordinates, Velocities

__all__: List[str] = [
    "generate_random_coordinates_normal_circle",
    "generate_random_coordinates_uniform_circle",
    "generate_random_velocities_normal",
]


def generate_random_coordinates_normal_circle(
    sigma: float,
    number: int,
    *,
    x: float = 0,
    y: float = 0,
    z: float = 0,
    rng: Optional[np.random.Generator] = None,
) -> Coordinates:
    """
    Generate normally distributed (Gaussian) samples in a circular pattern.

    Creates 2D Gaussian distribution with circular symmetry (isotropic) centered
    at (x, y) with standard deviation σ in both directions. This models thermal
    sources, diffuse beams, or Gaussian beam profiles.

    Mathematical Model
    ------------------
    Each coordinate is sampled independently:
        X ~ N(μₓ, σ²)
        Y ~ N(μᵧ, σ²)
        Z = z₀ (constant)

    The radial distribution follows Rayleigh distribution:
        R = √(X² + Y²) ~ Rayleigh(σ)
        P(r) = (r/σ²) exp(-r²/2σ²)

    Mean radial distance: <r> ≈ 1.25σ
    RMS radial distance: √<r²> = √2·σ ≈ 1.41σ

    Args:
        sigma (float):
            Standard deviation in both x and y directions (meters).
            Determines the width of the distribution. 68% of points
            fall within √2·σ ≈ 1.41σ of the center.
        number (int):
            Number of coordinate samples to generate.
        x (float, optional):
            Center x coordinate in meters. Defaults to 0.
        y (float, optional):
            Center y coordinate in meters. Defaults to 0.
        z (float, optional):
            Z coordinate (constant for all points) in meters. Defaults to 0.
        rng (Optional[np.random.Generator], optional):
            NumPy random number generator for reproducibility. If None, creates
            a new generator with system entropy. Defaults to None.

    Returns:
        Coordinates:
            Container with x, y, z arrays of shape (number,) containing the
            randomly generated coordinates following 2D Gaussian distribution.

    Examples
    --------
    >>> # Thermal beam source with 1mm width
    >>> coords = generate_random_coordinates_normal_circle(
    ...     sigma=1e-3, number=10000, x=0, y=0, z=0
    ... )
    >>>
    >>> # Reproducible generation
    >>> rng = np.random.default_rng(42)
    >>> coords = generate_random_coordinates_normal_circle(
    ...     sigma=0.5e-3, number=1000, z=0.1, rng=rng
    ... )
    >>>
    >>> # Check distribution (should be ≈ 1.41*sigma)
    >>> r = np.sqrt(coords.x**2 + coords.y**2)
    >>> print(f"RMS radius: {np.sqrt(np.mean(r**2)):.4f}")

    Notes
    -----
    - Independent x and y sampling ensures rotational invariance
    - Approximately 68% of points within circle of radius √2·σ
    - Approximately 95% of points within circle of radius 2√2·σ ≈ 2.83σ
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample independent Gaussian distributions in x and y
    # This produces circular symmetry (isotropic 2D Gaussian)
    return Coordinates(
        x=rng.normal(x, sigma, number),
        y=rng.normal(y, sigma, number),
        z=z * np.ones(number),  # Constant z coordinate
    )


def generate_random_coordinates_uniform_circle(
    radius: float,
    number: int,
    *,
    x: float = 0,
    y: float = 0,
    z: float = 0,
    rng: Optional[np.random.Generator] = None,
) -> Coordinates:
    """
    Generate coordinates uniformly distributed over a circular disk.

    Produces points with uniform density per unit area within a circle of given
    radius. This models well-collimated beams, uniform aperture illumination, or
    particles uniformly distributed at a circular cross-section.

    Mathematical Method
    -------------------
    Naive approach (r ~ U(0,R), θ ~ U(0,2π)) produces incorrect distribution
    with higher density near center because equal increments in r correspond to
    different area increments (dA = r·dr·dθ).

    Correct method for uniform area density:
        θ ~ U(0, 2π)
        r = √(U(0, R²))

    This comes from inverse transform sampling where cumulative distribution
    F(r) = r²/R² for uniform density, so r = R·√U where U ~ U(0,1).

    The resulting distribution has:
        P(r) = 2r/R² for 0 ≤ r ≤ R  (linearly increasing with radius)

    Mean radial distance: <r> = (2/3)·R ≈ 0.67R
    RMS radial distance: √<r²> = R/√2 ≈ 0.71R

    Args:
        radius (float):
            Radius of the circular disk in meters. All generated points will
            have distance from center ≤ radius.
        number (int):
            Number of coordinate samples to generate.
        x (float, optional):
            Center x coordinate in meters. Defaults to 0.
        y (float, optional):
            Center y coordinate in meters. Defaults to 0.
        z (float, optional):
            Z coordinate (constant for all points) in meters. Defaults to 0.
        rng (Optional[np.random.Generator], optional):
            NumPy random number generator for reproducibility. If None, creates
            a new generator with system entropy. Defaults to None.

    Returns:
        Coordinates:
            Container with x, y, z arrays of shape (number,) containing the
            randomly generated coordinates uniformly distributed over the disk.

    Examples
    --------
    >>> # Uniform beam through 1mm radius aperture
    >>> coords = generate_random_coordinates_uniform_circle(
    ...     radius=1e-3, number=10000, x=0, y=0, z=0
    ... )
    >>>
    >>> # Reproducible generation
    >>> rng = np.random.default_rng(42)
    >>> coords = generate_random_coordinates_uniform_circle(
    ...     radius=0.5e-3, number=1000, z=0.1, rng=rng
    ... )
    >>>
    >>> # Verify uniform distribution
    >>> r = np.sqrt(coords.x**2 + coords.y**2)
    >>> print(f"Mean radius: {np.mean(r):.4f} (expect {2/3 * radius:.4f})")
    >>> print(f"Max radius: {np.max(r):.4f} (expect {radius:.4f})")

    Notes
    -----
    - All points guaranteed to be within circle: r ≤ radius
    - Uniform density: equal probability per unit area
    - Angular distribution is uniform: θ ~ U(0, 2π)
    - Radial probability density: P(r) = 2r/R² (not uniform in r!)
    - For Gaussian beams, use generate_random_coordinates_normal_circle
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample angle uniformly (rotational symmetry)
    theta = rng.uniform(0, 2 * np.pi, number)

    # Sample radius using inverse transform method for uniform area density
    # r² ~ U(0, R²) ensures P(r) ∝ r, giving uniform density per unit area
    r = np.sqrt(rng.uniform(0, radius**2, number))

    # Convert polar (r, θ) to Cartesian (x, y) and translate to center
    return Coordinates(
        x=r * np.cos(theta) + x,
        y=r * np.sin(theta) + y,
        z=z * np.ones(number),  # Constant z coordinate
    )


def generate_random_velocities_normal(
    vx: float,
    vy: float,
    vz: float,
    sigma_vx: float,
    sigma_vy: float,
    sigma_vz: float,
    number: int,
    *,
    rng: Optional[np.random.Generator] = None,
) -> Velocities:
    """
    Generate normally distributed (Gaussian) velocity samples.

    Creates velocity distributions following independent Gaussian distributions
    in each component. This models thermal velocity distributions (Maxwell-Boltzmann),
    beam velocity spreads, or Doppler broadening effects.

    Mathematical Model
    ------------------
    Each velocity component is sampled independently:
        vₓ ~ N(μₓ, σₓ²)
        vᵧ ~ N(μᵧ, σᵧ²)
        vᵤ ~ N(μᵤ, σᵤ²)

    For thermal equilibrium at temperature T:
        σᵥ = √(kᵦT/m)
    where kᵦ is Boltzmann constant and m is particle mass.

    Speed distribution (for isotropic case σₓ = σᵧ = σᵤ = σ, μ = 0):
        v = |v⃗| follows Maxwell-Boltzmann distribution
        P(v) = √(2/π) (v²/σ³) exp(-v²/2σ²)

    Most probable speed: vₚ = √2·σ
    Mean speed: <v> = √(8/π)·σ ≈ 1.60σ
    RMS speed: √<v²> = √3·σ ≈ 1.73σ

    Args:
        vx (float):
            Mean velocity in x direction (m/s). Positive for motion in +x.
        vy (float):
            Mean velocity in y direction (m/s). Positive for motion in +y.
        vz (float):
            Mean velocity in z direction (m/s). Typically the beam velocity
            (e.g., 200 m/s for molecular beams).
        sigma_vx (float):
            Standard deviation in x velocity (m/s). Determines transverse
            velocity spread. For thermal: σᵥ = √(kᵦT/m).
        sigma_vy (float):
            Standard deviation in y velocity (m/s). Often equals sigma_vx
            for isotropic transverse motion.
        sigma_vz (float):
            Standard deviation in z velocity (m/s). Longitudinal velocity spread,
            typically smaller than transverse for supersonic beams.
        number (int):
            Number of velocity samples to generate.
        rng (Optional[np.random.Generator], optional):
            NumPy random number generator for reproducibility. If None, creates
            a new generator with system entropy. Defaults to None.

    Returns:
        Velocities:
            Container with vx, vy, vz arrays of shape (number,) containing the
            randomly generated velocities following Gaussian distributions.

    Examples
    --------
    >>> # Thermal beam at room temperature (σ ≈ 50 m/s for TlF)
    >>> vels = generate_random_velocities_normal(
    ...     vx=0, vy=0, vz=200,  # 200 m/s forward
    ...     sigma_vx=50, sigma_vy=50, sigma_vz=10,  # Spreads
    ...     number=10000
    ... )
    >>>
    >>> # Calculate temperature from velocity spread
    >>> # For TlF (m ≈ 204 amu): T = m·σᵥ²/kᵦ
    >>> import scipy.constants as const
    >>> m_TlF = 204 * const.atomic_mass  # kg
    >>> T = m_TlF * 50**2 / const.k  # Kelvin
    >>>
    >>> # Supersonic beam (cold longitudinal, warm transverse)
    >>> vels = generate_random_velocities_normal(
    ...     vx=0, vy=0, vz=300,
    ...     sigma_vx=30, sigma_vy=30, sigma_vz=5,  # Δvᵤ/vᵤ ≈ 1.7%
    ...     number=5000
    ... )
    >>>
    >>> # Reproducible generation
    >>> rng = np.random.default_rng(42)
    >>> vels = generate_random_velocities_normal(
    ...     vx=0, vy=0, vz=200,
    ...     sigma_vx=50, sigma_vy=50, sigma_vz=10,
    ...     number=1000, rng=rng
    ... )

    Notes
    -----
    - Independent sampling in each direction
    - 68% of samples within ±σᵥ of mean in each component
    - 95% of samples within ±2σᵥ of mean in each component
    - For isotropic distribution: set σₓ = σᵧ = σᵤ and μₓ = μᵧ = μᵤ = 0
    - Speed |v⃗| follows Maxwell-Boltzmann distribution for isotropic case
    - Velocity spread relates to temperature: σᵥ = √(kᵦT/m)

    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample independent Gaussian distributions for each velocity component
    return Velocities(
        vx=rng.normal(vx, sigma_vx, number),
        vy=rng.normal(vy, sigma_vy, number),
        vz=rng.normal(vz, sigma_vz, number),
    )
