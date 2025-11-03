from __future__ import annotations

from typing import Optional, Self, Tuple, overload

import numpy as np
import numpy.typing as npt
from numpy.polynomial import Polynomial

from .common_types import NDArray_or_Float
from .numba_functions import _polyval2d_scalar, _polyval_1d, _polyval_scalar


class FastPolynomial(Polynomial):
    """
    NumPy‑compatible polynomial that accelerates evaluation with Numba *and*
    always returns its own subclass from algebraic and helper methods.
    """

    @overload
    def __call__(self, x: float) -> float: ...

    @overload
    def __call__(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

    def __call__(self, x):
        if (
            self.domain.tolist() == [-1.0, 1.0]
            and self.coef.dtype == np.float64
            and self.coef.ndim == 1
        ):
            coef = np.ascontiguousarray(self.coef, dtype=np.float64)
            # scalar fast‑path
            if np.isscalar(x):
                return _polyval_scalar(x, coef)
            # 1‑D array fast‑path
            x_arr = np.asarray(x, dtype=np.float64)
            if x_arr.ndim == 1 and x_arr.flags.c_contiguous:
                return _polyval_1d(x_arr, coef)
        # fallback → NumPy implementation
        return super().__call__(x)

    # ---- ONE central override that fixes return‑types everywhere -----------
    def _wrap(
        self,
        coef: npt.NDArray[np.float64],
        domain: Optional[Tuple[float, float]] = None,
        window: Optional[Tuple[float, float]] = None,
    ) -> Self:
        return self.__class__(
            coef, domain=domain or self.domain, window=window or self.window
        )

    # ---- Re‑export constructors so they return FastPolynomial --------------
    # (These are @classmethods on the base class; we simply inherit them
    # and rely on cls = FastPolynomial inside.)
    fit = Polynomial.fit.__func__  # type: ignore
    fromroots = Polynomial.fromroots.__func__  # type: ignore
    basis = Polynomial.basis.__func__  # type: ignore
    identity = Polynomial.identity.__func__  # type: ignore


class Polynomial2D:
    """2D polynomial with fitting, evaluation, differentiation, and pretty-printing.

    Attributes:
        kx (int): Maximum exponent in x-direction.
        ky (int): Maximum exponent in y-direction.
        coeffs (np.npt.NDArray): Coefficient matrix of shape (kx+1, ky+1).
        order (int): Optional total-degree cap for the polynomial.
    """

    _subscript_mapping = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    _superscript_mapping = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

    def __init__(
        self,
        kx: int,
        ky: int,
        coeffs: npt.NDArray[np.float64] | None = None,
        order: int | None = None,
    ) -> None:
        """Initialize a 2D polynomial.

        Args:
            kx (int): Maximum exponent in x-direction.
            ky (int): Maximum exponent in y-direction.
            coeffs (np.ndarray, optional): Coefficient matrix of shape (kx+1, ky+1).
                Defaults to None.
            order (int, optional): Total-degree cap for the polynomial. Defaults to None.
        """
        self.kx = kx
        self.ky = ky
        self.coeffs = coeffs
        self.order = order

    def __call__(self, x: NDArray_or_Float, y: NDArray_or_Float) -> NDArray_or_Float:
        """Evaluate the polynomial at coordinates (x,y).

        This is a shorthand for the evaluate method.

        Args:
            x (NDArray_or_Float): X coordinates.
            y (NDArray_or_Float): Y coordinates.

        Returns:
            NDArray_or_Float: Evaluated polynomial values.
        """
        return self.evaluate(x, y)

    # ─────────────────────────────── helpers ────────────────────────────────
    @classmethod
    def _str_term_unicode(cls, power: str, arg: str) -> str:
        """Return a Unicode term like ·x² or ·y for pretty printing.

        Args:
            power (str): Power of the term.
            arg (str): Variable name.

        Returns:
            str: Unicode formatted term.
        """
        return (
            f"·{arg}{power.translate(cls._superscript_mapping)}"
            if power != "1"
            else f"·{arg}"
        )

    def _generate_string(self, term_method) -> str:
        """Generate a formatted string of the polynomial terms.

        Args:
            term_method (callable): Function to format term power and variable.
                Should accept (power: str, arg: str) and return formatted string.

        Returns:
            str: Formatted polynomial string with all non-zero terms.

        Notes:
            Handles special cases for zero and constant terms.
            Formats terms with appropriate signs (+ or -).
        """
        if self.coeffs is None:
            return "Polynomial2D(coeffs=None)"

        out = f"{self.coeffs[0, 0]}" if self.coeffs[0, 0] != 0 else ""

        for i, j in np.ndindex((self.kx + 1, self.ky + 1)):
            coef = self.coeffs[i, j]
            if coef == 0 or (i == 0 and j == 0):
                continue

            sign, coef_abs = ("+", coef) if coef >= 0 else ("-", -coef)
            term = f" {sign} {coef_abs}"

            if i:
                term += term_method(str(i), "x")
            if j:
                term += term_method(str(j), "y")

            out += term

        return out.lstrip(" +").lstrip(" -")

    # ───────────────────────────── dunder/magic ─────────────────────────────
    def __repr__(self) -> str:
        """Return string representation of the polynomial.

        Returns:
            str: Unicode formatted representation of the polynomial expression.
        """
        return f"Polynomial2D({self._generate_string(Polynomial2D._str_term_unicode)})"

    def __mul__(self, scalar: float | int) -> Polynomial2D:
        """Multiply polynomial by a scalar.

        Args:
            scalar (float | int): Value to multiply coefficients by.

        Returns:
            Polynomial2D: New polynomial with scaled coefficients.

        Raises:
            TypeError: If scalar is not a float or int.
            ValueError: If coeffs are not set.
        """
        if not isinstance(scalar, (float, int)):
            raise TypeError(
                f"Can only multiply by scalars, got {type(scalar).__name__}"
            )

        if self.coeffs is None:
            raise ValueError("Cannot multiply: polynomial coefficients not set")

        return Polynomial2D(
            self.kx,
            self.ky,
            self.coeffs * scalar,
            order=self.order,
        )

    def __rmul__(self, scalar: float | int) -> Polynomial2D:
        """Right multiplication by a scalar.

        Args:
            scalar (float | int): Value to multiply coefficients by.

        Returns:
            Polynomial2D: New polynomial with scaled coefficients.

        See Also:
            __mul__: For detailed behavior.
        """
        return self.__mul__(scalar)

    def __imul__(self, scalar: float | int) -> Self:
        """In-place multiplication by a scalar.

        Args:
            scalar (float | int): Value to multiply coefficients by.

        Returns:
            Polynomial2D: Self with updated coefficients.

        Raises:
            TypeError: If scalar is not a float or int.
            ValueError: If coeffs are not set.
        """
        if not isinstance(scalar, (float, int)):
            raise TypeError(
                f"Can only multiply by scalars, got {type(scalar).__name__}"
            )

        if self.coeffs is None:
            raise ValueError("Cannot multiply: polynomial coefficients not set")

        self.coeffs *= scalar
        return self

    def __truediv__(self, scalar: float | int) -> Polynomial2D:
        """Divide polynomial by a scalar.

        Args:
            scalar (float | int): Value to divide coefficients by.

        Returns:
            Polynomial2D: New polynomial with scaled coefficients.

        Raises:
            TypeError: If scalar is not a float or int.
            ValueError: If coeffs are not set or scalar is zero.
        """
        if not isinstance(scalar, (float, int)):
            raise TypeError(f"Can only divide by scalars, got {type(scalar).__name__}")

        if scalar == 0:
            raise ValueError("Cannot divide by zero")

        if self.coeffs is None:
            raise ValueError("Cannot divide: polynomial coefficients not set")

        return Polynomial2D(
            self.kx,
            self.ky,
            self.coeffs / scalar,
            order=self.order,
        )

    def __rtruediv__(self, scalar: float | int) -> Self:
        """Not supported - cannot divide a scalar by a polynomial.

        Args:
            scalar (float | int): Scalar value.

        Raises:
            TypeError: Always raised as operation is not meaningful.
        """
        raise TypeError("Cannot divide a scalar by a polynomial")

    def __itruediv__(self, scalar: float | int) -> Self:
        """In-place division by a scalar.

        Args:
            scalar (float | int): Value to divide coefficients by.

        Returns:
            Polynomial2D: Self with updated coefficients.

        Raises:
            TypeError: If scalar is not a float or int.
            ValueError: If coeffs are not set or scalar is zero.
        """
        if not isinstance(scalar, (float, int)):
            raise TypeError(f"Can only divide by scalars, got {type(scalar).__name__}")

        if scalar == 0:
            raise ValueError("Cannot divide by zero")

        if self.coeffs is None:
            raise ValueError("Cannot divide: polynomial coefficients not set")

        self.coeffs /= scalar
        return self

    # ─────────────────────────── public interface ───────────────────────────
    def fit(
        self: Self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        order: int | None = None,
        zero_terms: list[tuple[int, int]] | None = None,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        int,
        npt.NDArray[np.float64],
    ]:
        """Perform least‐squares fit of z = f(x,y) to a 2‑D polynomial.

        Args:
            x (np.ndarray): X coordinates.
            y (np.ndarray): Y coordinates.
            z (np.ndarray): Data values at (x, y).
            order (int, optional): Total‐degree cap for the polynomial.
                Defaults to self.order.
            zero_terms (list[tuple[int, int]], optional): Coefficient indices to force to zero.
                Defaults to None.

        Returns:
            tuple:
                coeffs (np.ndarray): Solution coefficients (flattened).
                residuals (np.ndarray): Sum of squared residuals.
                rank (int): Effective rank of the Vandermonde matrix.
                singular_values (np.ndarray): Singular values from the fit.
        """
        order = self.order if order is None else order
        zero_terms = [] if zero_terms is None else zero_terms

        vand = np.polynomial.polynomial.polyvander2d(x, y, [self.kx, self.ky])

        for i, j in np.ndindex((self.kx + 1, self.ky + 1)):
            if (order is not None and i + j > order) or (i, j) in zero_terms:
                vand[:, i * (self.ky + 1) + j] = 0.0

        mask = ~np.isnan(z)
        sol = np.linalg.lstsq(vand[mask], z[mask], rcond=None)

        self.coeffs = sol[0].reshape((self.kx + 1, self.ky + 1))
        for i, j in np.ndindex(self.coeffs.shape):
            if (order is not None and i + j > order) or (i, j) in zero_terms:
                self.coeffs[i, j] = 0.0

        return sol  # type: ignore[return-value]  # NumPy stubs: OK

    def derivative(self, axis: str = "x") -> Polynomial2D:
        """Compute ∂/∂x or ∂/∂y, returning a new Polynomial2D.

        Args:
            axis (str): Axis to differentiate along, 'x' or 'y'. Defaults to 'x'.

        Returns:
            Polynomial2D: The derivative polynomial.

        Raises:
            ValueError: If coeffs are unset or axis is invalid.
        """
        if self.coeffs is None:
            raise ValueError("Cannot differentiate before fitting/setting coeffs.")

        if axis == "x":
            return Polynomial2D(
                self.kx - 1,
                self.ky,
                self.coeffs[1:, :] * (np.arange(1, self.kx + 1)[:, None]),
            )
        if axis == "y":
            return Polynomial2D(
                self.kx,
                self.ky - 1,
                self.coeffs[:, 1:] * (np.arange(1, self.ky + 1)[None, :]),
            )
        raise ValueError(f"Axis must be 'x' or 'y', got {axis!r}.")

    def evaluate(
        self,
        x: NDArray_or_Float,
        y: NDArray_or_Float,
    ) -> NDArray_or_Float:
        """Evaluate the polynomial at given coordinates.

        Args:
            x (NDArray_or_Float): X coordinates.
            y (NDArray_or_Float): Y coordinates.

        Returns:
            NDArray_or_Float: Evaluated polynomial values.

        Raises:
            ValueError: If coefficients are unset.
        """
        if self.coeffs is None:
            raise ValueError("Polynomial has no coefficients set.")
        if isinstance(x, float) and isinstance(y, float):
            return _polyval2d_scalar(x, y, self.coeffs)
        else:
            return np.polynomial.polynomial.polyval2d(x, y, self.coeffs)

    def fast_scalar_evaluate(self, x: float, y: float) -> float:
        """Evaluate the polynomial at given coordinates.

        Args:
            x (float): X coordinates.
            y (float): Y coordinates.

        Returns:
            float: Evaluated polynomial values.
        """
        if self.coeffs is None:
            raise ValueError("Polynomial has no coefficients set.")
        return _polyval2d_scalar(x, y, self.coeffs)
