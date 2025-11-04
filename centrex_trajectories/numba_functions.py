import numba as nb
import numpy as np
import numpy.typing as npt

from .common_types import NDArray_or_Float


@nb.njit
def _translation(
    x: NDArray_or_Float,
    y: NDArray_or_Float,
    z: NDArray_or_Float,
    x0: float,
    y0: float,
) -> tuple[NDArray_or_Float, NDArray_or_Float, NDArray_or_Float]:
    """
    Translate coordinates relative to lens center.

    Args:
        x, y, z: Particle coordinates
        x0, y0: Lens center coordinates

    Returns:
        Translated coordinates (x-x0, y-y0, z)
    """
    return (x - x0, y - y0, z)


@nb.njit
def _rotation(
    x: NDArray_or_Float,
    y: NDArray_or_Float,
    z: NDArray_or_Float,
    cos_tilt: float,
    sin_tilt: float,
) -> tuple[NDArray_or_Float, NDArray_or_Float, NDArray_or_Float]:
    """
    Rotate coordinates around x-axis by tilt angle.

    Args:
        x, y, z: Coordinates to rotate
        cos_tilt: Cosine of tilt angle
        sin_tilt: Sine of tilt angle

    Returns:
        Rotated coordinates (x, y', z') where:
        y' = y*cos(tilt) - z*sin(tilt)
        z' = y*sin(tilt) + z*cos(tilt)
    """
    return (
        x,
        y * cos_tilt - z * sin_tilt,
        y * sin_tilt + z * cos_tilt,
    )


@nb.njit
def _coordinate_transformation(
    x: NDArray_or_Float,
    y: NDArray_or_Float,
    z: NDArray_or_Float,
    x0: float,
    y0: float,
    cos_tilt: float,
    sin_tilt: float,
) -> tuple[NDArray_or_Float, NDArray_or_Float, NDArray_or_Float]:
    """
    Apply translation and rotation to transform coordinates to lens frame.

    Args:
        x, y, z: Lab frame coordinates
        x0, y0: Lens center position
        cos_tilt: Cosine of tilt angle
        sin_tilt: Sine of tilt angle

    Returns:
        Coordinates in lens frame
    """
    # Translate
    x_t = x - x0
    y_t = y - y0
    z_t = z

    # Rotate around x-axis
    return (
        x_t,
        y_t * cos_tilt - z_t * sin_tilt,
        y_t * sin_tilt + z_t * cos_tilt,
    )


@nb.njit(inline="always")
def _polyval_scalar(x: float, coeffs: npt.NDArray[np.float64]) -> float:
    """Numba‑accelerated Horner‑scheme evaluation for scalar x."""
    acc: float = 0.0
    for c in coeffs[::-1]:  # Horner: highest power first
        acc = acc * x + c
    return acc


@nb.njit(inline="always")
def _polyval_1d(
    x: npt.NDArray[np.float64], coeffs: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Vectorised Horner‑scheme evaluation for a 1‑D array x."""
    out = np.zeros_like(x)
    for c in coeffs[::-1]:
        out *= x
        out += c
    return out


@nb.njit(inline="always")
def _polyval2d_scalar(x: float, y: float, coeffs: npt.NDArray[np.float64]) -> float:
    acc = 0.0
    # coeffs shape: (nx+1, ny+1), where coeffs[i, j] multiplies x^i y^j
    for i in range(coeffs.shape[0] - 1, -1, -1):
        row_acc = 0.0
        for j in range(coeffs.shape[1] - 1, -1, -1):
            row_acc = row_acc * y + float(coeffs[i, j])
        acc = acc * x + row_acc
    return acc


@nb.njit
def _force_eql_poly_scalar(
    x: float,
    y: float,
    x0: float,
    y0: float,
    pot_z: float,
    ex_coeff: npt.NDArray[np.float64],
    ey_coeff: npt.NDArray[np.float64],
    ex_x_coeff: npt.NDArray[np.float64],
    ex_y_coeff: npt.NDArray[np.float64],
    ey_x_coeff: npt.NDArray[np.float64],
    ey_y_coeff: npt.NDArray[np.float64],
    stark_deriv_coeff: npt.NDArray[np.float64],
) -> tuple[float, float, float]:
    # coordinate transform
    _x = x - x0
    _y = y - y0

    # polynomial electric field components
    Ex = pot_z * _polyval2d_scalar(_x, _y, ex_coeff)
    Ey = pot_z * _polyval2d_scalar(_x, _y, ey_coeff)

    # derivatives
    Ex_x = pot_z * _polyval2d_scalar(_x, _y, ex_x_coeff)
    Ex_y = pot_z * _polyval2d_scalar(_x, _y, ex_y_coeff)
    Ey_x = pot_z * _polyval2d_scalar(_x, _y, ey_x_coeff)
    Ey_y = pot_z * _polyval2d_scalar(_x, _y, ey_y_coeff)

    # |E| and its gradient
    E_mag = np.hypot(Ex, Ey)
    if E_mag < 1e-15:
        return 0.0, 0.0, 0.0  # centre: no force

    inv_E = 1.0 / E_mag
    dEx = (Ex * Ex_x + Ey * Ey_x) * inv_E
    dEy = (Ex * Ex_y + Ey * Ey_y) * inv_E

    # d(Stark)/dE
    dVdE = _polyval_scalar(E_mag, stark_deriv_coeff)

    # force = –dVdE ∇|E|
    return -dVdE * dEx, -dVdE * dEy, 0.0


@nb.njit
def _force_eql_scalar(
    x: float,
    y: float,
    x0: float,
    y0: float,
    k: float,
    stark_deriv_coeff: npt.NDArray[np.float64],
) -> tuple[float, float, float]:
    """
    Fast per‑particle force for an electrostatic quadrupole lens.
    Uses the user‑provided _polyval_scalar to evaluate dV/dE(E).
    """
    # shift to lens centre
    _x = x - x0
    _y = y - y0
    r2 = _x * _x + _y * _y
    if r2 < 1e-20:  # on axis → no transverse force
        return 0.0, 0.0, 0.0

    r = np.sqrt(r2)
    dx = _x / r
    dy = _y / r

    # |E| and dE/dr  (linear in r)
    v2_over_r2 = k
    E_mag = v2_over_r2 * r
    dEdr = v2_over_r2  # constant

    # d(Stark)/dE via your Horner helper
    dVdE = _polyval_scalar(E_mag, stark_deriv_coeff)

    coeff = -dVdE * dEdr
    return coeff * dx, coeff * dy, 0.0


@nb.njit
def _force_eql_scalar_tilt(
    x: float,
    y: float,
    z: float,
    x0: float,
    y0: float,
    k: float,
    cos_tilt: float,
    sin_tilt: float,
    stark_deriv_coeff: npt.NDArray[
        np.float64
    ],  # dV/dE polynomial coeffs (C-contiguous, float64)
) -> tuple[float, float, float]:
    """
    Fast per-particle force for an electrostatic quadrupole lens with tilt.
    Precomputed parameters: x0, y0, cos_tilt, sin_tilt, k=2*V/R^2.
    """
    # 1) lab → lens coords: translate, then rotate about x by +tilt
    xp = x - x0
    yp0 = y - y0
    yp = yp0 * cos_tilt - z * sin_tilt
    # zp = yp0 * sin_tilt + z * cos_tilt  # not needed for force; included for clarity

    # 2) lens-frame radial unit vector
    r2 = xp * xp + yp * yp
    if r2 < 1e-20:
        return 0.0, 0.0, 0.0

    r = np.sqrt(r2)
    dxp = xp / r
    dyp = yp / r

    # 3) |E| and dE/dr in lens frame (quadrupole: E = k * r, dE/dr = k)
    E_mag = k * r
    dVdE = _polyval_scalar(E_mag, stark_deriv_coeff)
    coeff = -dVdE * k  # (-dV/dE) * dE/dr

    # 4) force in lens frame: purely transverse
    fxp = coeff * dxp
    fyp = coeff * dyp
    # fzp = 0

    # 5) lens → lab: rotate vector by R_x(-tilt)
    # R_x(-θ): (fx = fxp, fy =  fyp*cosθ + 0*sinθ, fz = -fyp*sinθ + 0*cosθ)
    fx = fxp
    fy = fyp * cos_tilt
    fz = -fyp * sin_tilt

    return fx, fy, fz


@nb.njit
def _force_eql_poly_scalar_tilt(
    x: float,
    y: float,
    z: float,
    x0: float,
    y0: float,
    cos_tilt: float,
    sin_tilt: float,
    pot_z: float,
    ex_coeff: npt.NDArray[np.float64],
    ey_coeff: npt.NDArray[np.float64],
    ex_x_coeff: npt.NDArray[np.float64],
    ex_y_coeff: npt.NDArray[np.float64],
    ey_x_coeff: npt.NDArray[np.float64],
    ey_y_coeff: npt.NDArray[np.float64],
    stark_deriv_coeff: npt.NDArray[np.float64],
) -> tuple[float, float, float]:
    """
    Fast per-particle force for a polynomial electrostatic lens with tilt.
    Combines coordinate transformation, polynomial field evaluation, and rotation back to lab frame.
    """
    # 1) lab → lens coords: translate, then rotate about x by +tilt
    xp = x - x0
    yp0 = y - y0
    yp = yp0 * cos_tilt - z * sin_tilt
    # zp = yp0 * sin_tilt + z * cos_tilt  # not needed if pot_z is pre-evaluated

    # 2) polynomial electric field components in lens frame
    Ex = pot_z * _polyval2d_scalar(xp, yp, ex_coeff)
    Ey = pot_z * _polyval2d_scalar(xp, yp, ey_coeff)

    # 3) derivatives in lens frame
    Ex_x = pot_z * _polyval2d_scalar(xp, yp, ex_x_coeff)
    Ex_y = pot_z * _polyval2d_scalar(xp, yp, ex_y_coeff)
    Ey_x = pot_z * _polyval2d_scalar(xp, yp, ey_x_coeff)
    Ey_y = pot_z * _polyval2d_scalar(xp, yp, ey_y_coeff)

    # 4) |E| and its gradient in lens frame
    E_mag = np.hypot(Ex, Ey)
    if E_mag < 1e-15:
        return 0.0, 0.0, 0.0  # centre: no force

    inv_E = 1.0 / E_mag
    dEx = (Ex * Ex_x + Ey * Ey_x) * inv_E
    dEy = (Ex * Ex_y + Ey * Ey_y) * inv_E

    # 5) d(Stark)/dE
    dVdE = _polyval_scalar(E_mag, stark_deriv_coeff)

    # 6) force in lens frame = –dVdE ∇|E|
    fxp = -dVdE * dEx
    fyp = -dVdE * dEy
    # fzp = 0  (no z-field component in simplified model)

    # 7) lens → lab: rotate vector by R_x(-tilt)
    fx = fxp
    fy = fyp * cos_tilt  # + 0 * sin_tilt
    fz = -fyp * sin_tilt  # + 0 * cos_tilt

    return fx, fy, fz
