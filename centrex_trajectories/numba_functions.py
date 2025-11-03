import numba as nb
import numpy as np
import numpy.typing as npt


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
def _polyval2d_scalar(x: float, y: float, coeffs: npt.NDArray[np.floating]) -> float:
    result = 0.0
    for i in range(coeffs.shape[0]):
        x_pow = x**i
        for j in range(coeffs.shape[1]):
            result += coeffs[i, j] * x_pow * y**j
    return result


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
    z: float,
    x0: float,
    y0: float,
    V: float,
    R: float,
    stark_deriv_coeff: npt.NDArray[np.float64],
    cos_tilt: float=0,
    sin_tilt: float=0,
) -> tuple[float, float, float]:
    """
    Fast per‑particle force for an electrostatic quadrupole lens.
    Uses the user‑provided _polyval_scalar to evaluate dV/dE(E).
    """
    # shift to lens centre
    _x = x - x0
    _y = y - y0
    _y = _y * cos_tilt - z * sin_tilt  # rotate y,z according to tilt
    _z = y * sin_tilt + z * cos_tilt

    r2 = _x * _x + _y * _y
    if r2 < 1e-20:  # on axis → no transverse force
        return 0.0, 0.0, 0.0

    r = np.sqrt(r2)
    dx = _x / r
    _dy = _y / r
    dy = _dy * cos_tilt
    dz = -_dy * sin_tilt
    # |E| and dE/dr  (linear in r)
    v2_over_r2 = 2 * V / R**2
    E_mag = v2_over_r2 * r
    dEdr = v2_over_r2  # constant

    # d(Stark)/dE via your Horner helper
    dVdE = _polyval_scalar(E_mag, stark_deriv_coeff)

    coeff = -dVdE * dEdr
    return coeff * dx, coeff * dy, coeff * dz
