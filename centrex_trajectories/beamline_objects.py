from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, cast, overload

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq

from .common_types import NDArray_or_Float
from .data_structures import Acceleration, Coordinates, Force, Velocities
from .numba_functions import _force_eql_poly_scalar, _force_eql_scalar
from .particles import Particle
from .polynomials import FastPolynomial, Polynomial2D
from .propagation_ballistic import calculate_time_ballistic, propagate_ballistic
from .propagation_options import PropagationType
from .utils import bounds_check_tolerance

__all__ = [
    "Section",
    "LinearSection",
    "ElectrostaticQuadrupoleLens",
    "ElectrostaticLensPolynomial",
    "MagnetostaticHexapoleLens",
    "CircularAperture",
    "RectangularAperture",
    "RectangularApertureFinite",
    "PlateElectrodes",
    "Bore",
]

path = Path(__file__).resolve().parent
with open(path / "saved_data" / "stark_poly.pkl", "rb") as f:
    stark_poly: npt.NDArray[np.float64] = pickle.load(f)
    stark_potential_default = FastPolynomial(stark_poly)
    stark_potential_default_derivative = stark_potential_default.deriv()


@dataclass
class Section:
    """
    Represents a generic section of the beamline.

    Attributes:
        name (str): Name of the section.
        objects (List[BeamlineObject]): List of objects within the section that particles can interact with.
        start (float): Start position of the section along the z-axis [m].
        stop (float): End position of the section along the z-axis [m].
        save_collisions (bool): Whether to save collision data (coordinates and velocities) for this section.
        propagation_type (PropagationType): Type of propagation used for particle motion in this section.
        force (Optional[Force]): Additional constant force acting on particles in this section (e.g., gravity or electric field).
    """

    name: str
    objects: List[BeamlineObject]
    start: float
    stop: float
    save_collisions: bool
    propagation_type: PropagationType = PropagationType.ballistic
    force: Optional[Force] = None

    def __post_init__(self):
        """
        Check if all objects reside fully inside the section, runs upon initialization.
        """
        for o in self.objects:
            assert o.check_in_bounds(self.start, self.stop), (
                f"{o} not inside {self.name}"
            )


@dataclass
class LinearSection(Section):
    """
    Linear Section dataclass.

    Attributes:
        x (float): x-coordinate of the section [m].
        y (float): y-coordinate of the section [m].
        spring_constant (tuple[float, ...]): Spring constants in x, y, and z directions [N/m].
        propagation_type (PropagationType): Propagation type for linear motion.
        force (Optional[Force]): A constant force acting on the section.
    """

    x: float = 0.0
    y: float = 0.0
    spring_constant: tuple[float, ...] = (0.0, 0.0, 0.0)
    propagation_type: PropagationType = PropagationType.linear
    force: Optional[Force] = None


class ODESection:
    """
    Base class for sections using ODE-based propagation.

    Attributes:
        propagation_type (PropagationType): Type of propagation for ODE-based motion.
        name (str): Name of the section.
        objects (List[BeamlineObject]): List of objects within the section that particles can interact with.
        start (float): Start position of the section along the z-axis [m].
        stop (float): End position of the section along the z-axis [m].
        save_collisions (bool): Whether to save collision data (coordinates and velocities) for this section.
    """

    propagation_type: PropagationType = PropagationType.ode

    def __init__(
        self,
        name: str,
        objects: List[BeamlineObject],
        start: float,
        stop: float,
        save_collisions: bool,
    ):
        self.name = name
        self.objects = objects
        self.start = start
        self.stop = stop
        self.save_collisions = save_collisions

    def _check_objects(self):
        """
        Check if all objects reside fully inside the section, runs upon initializatin.
        """
        for o in self.objects:
            assert o.check_in_bounds(self.start, self.stop), (
                f"{o} not inside {self.name}"
            )

    @overload
    def force(
        self, t: float, x: float, y: float, z: float
    ) -> Tuple[float, float, float]: ...

    @overload
    def force(
        self,
        t: float,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> Tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]: ...

    def force(self, t, x, y, z):
        raise NotImplementedError


class ElectrostaticQuadrupoleLens(ODESection):
    """
    Represents an electrostatic quadrupole lens.

    Attributes:
        name (str): Name of the lens.
        objects (List[Any]): List of objects within the lens that particles can interact with.
        start (float): Start position of the lens along the z-axis [m].
        stop (float): End position of the lens along the z-axis [m].
        V (float): Voltage applied to the electrodes [Volts].
        R (float): Radius of the lens bore [m].
        x (float): x-coordinate of the lens center [m].
        y (float): y-coordinate of the lens center [m].
        save_collisions (bool): Whether to save collision data (coordinates and velocities) for this lens.
        stark_potential (Optional[np.ndarray]): Polynomial coefficients of the Stark potential as a function of the electric field.
    """

    def __init__(
        self,
        name: str,
        objects: List[Any],
        start: float,
        stop: float,
        V: float,
        R: float,
        x: float = 0,
        y: float = 0,
        tilt: float = 0,
        save_collisions: bool = False,
        stark_potential: None | npt.NDArray[np.float64] = None,
    ) -> None:
        """
        Electrostatic Quadrupole Lens Section

        Args:
            name (str): name of electrostatic quadrupole lens
            objects (List): objects inside the section which particles can collide with
            start (float): start of section in z [m]
            stop (float): stop of section in z [m]
            V (float): Voltage on electrodes [Volts]
            R (float): radius of lens bore [m]
            save_collisions (Optional[bool], optional): Save the coordinates and
                                                    velocities of collisions in this
                                                    section. Defaults to False.
            stark_potential (Optional[np.ndarray]): polynomial coefficients of the stark
                                                    potential as a function of electric
                                                    field
        """
        super().__init__(name, objects, start, stop, save_collisions)
        self.V = V
        self.R = R
        self.x0 = x
        self.y0 = y
        self.tilt = tilt
        self._sin_tilt = np.sin(tilt)
        self._cos_tilt = np.cos(tilt)
        self._check_objects()
        self._initialize_potentials(stark_potential)

    def _initialize_potentials(
        self, stark_potential: None | npt.NDArray[np.float64] = None
    ) -> None:
        """
        Generate the radial derivative of the Stark potential for the force calculation
        """
        if stark_potential is None:
            self._stark_potential = stark_potential_default
        else:
            self._stark_potential = FastPolynomial(stark_potential)

        self._stark_potential_derivative = self._stark_potential.deriv()
  
    def translation(self,
        x: NDArray_or_Float,
        y: NDArray_or_Float,
        z: NDArray_or_Float,
    ) -> Tuple[
        npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]
    ]:
        """
        Given the position x, y, z, generate the relative coordinates w.r.t. the lens
        
        Args:
            x (Union[NDArray[np.float64], float]): x coordinate(s) [m]
            y (Union[NDArray[np.float64], float]): y coordinate(s) [m]
            z (Union[NDArray[np.float64], float]): z coordinate(s) [m]

        Returns:
            Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]] transformed coordinates
        """
        return (x - self.x0, y - self.y0, z)
    
    def rotation(self,
        x: NDArray_or_Float,
        y: NDArray_or_Float,
        z: NDArray_or_Float,
    ) -> Tuple[
        npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]
    ]:
        """
        Given the position x, y, z, generate the relative coordinates w.r.t. the lens coordinate system
        
        Args:
            x (Union[NDArray[np.float64], float]): x coordinate(s) [m]
            y (Union[NDArray[np.float64], float]): y coordinate(s) [m]
            z (Union[NDArray[np.float64], float]): z coordinate(s) [m]

        Returns:
            Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]] transformed coordinates
        """
        return (x, y*np.cos(self.tilt)-z*np.sin(self.tilt), y*np.sin(self.tilt)+z*np.cos(self.tilt))
    
    def coordinate_transformation(self,
        x: NDArray_or_Float,
        y: NDArray_or_Float,
        z: NDArray_or_Float,
    ) -> Tuple[
        npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]
    ]:
        _x, _y, _z = self.translation(x,y,z)
        return self.rotation(_x, _y, _z)

    def electric_field(
        self,
        x: NDArray_or_Float,
        y: NDArray_or_Float,
        z: NDArray_or_Float,
    ) -> NDArray_or_Float:
        """
        Calculate the electric field of the lens at x,y,z

        Args:
            x (NDArray_or_Float): x coordinate(s) [m]
            y (NDArray_or_Float): y coordinate(s) [m]
            z (NDArray_or_Float): z coordinate(s) [m]

        Returns:
            Union[NDArray_or_Float, float]: electric field at x,y,z in V/m
        """
        _x, _y, _z = self.coordinate_transformation(x,y,z)

        return cast(
            NDArray_or_Float, 2 * self.V * np.sqrt(_x**2 + _y**2) / (self.R) ** 2
        )

    def electric_field_derivative_r(
        self,
        x: NDArray_or_Float,
        y: NDArray_or_Float,
        z: NDArray_or_Float,
    ) -> NDArray_or_Float:
        """
        Derivative of the electric field in r: dE/dr (r = sqrt(x^2+y^2))

        Args:
            x (Union[npt.NDArray[np.float64], float]): x coordinate(s) [m]
            y (Union[npt.NDArray[np.float64], float]): y coordinate(s) [m]
            z (Union[npt.NDArray[np.float64], float]): z coordinates(s) [m]

        Returns:
            Union[npt.NDArray[np.float64], float]: derivative of electric field in r V/m^2
        """
        if isinstance(x, np.ndarray):
            return 2 * self.V / self.R**2 * np.ones(x.shape)
        else:
            return 2 * self.V / self.R**2

    @overload
    def force(
        self, t: float, x: float, y: float, z: float
    ) -> Tuple[float, float, float]: ...

    @overload
    def force(
        self,
        t: float,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> Tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]: ...

    def force(self, t, x, y, z):
        """
        Calculate the force at x,y,z

        Args:
            t (float): time [s]
            x (Union[NDArray[np.float64], float]): x coordinate(s) [m]
            y (Union[NDArray[np.float64], float]): y coordinate(s) [m]
            z (Union[NDArray[np.float64], float]): z coordinate(s) [m]

        Returns:
            List: force in x, y and z
        """
        _x, _y, _z = self.coordinate_transformation(x,y,z)
        r = np.sqrt(_x**2 + _y**2)
        if r == 0:
            dx = 0.0
            dy = 0.0
        else:
            dx = _x / r
            dy = _y / r
        stark = -self.stark_potential_derivative(
            x, y, z
        ) * self.electric_field_derivative_r(x, y, z)
        Fy = stark * dy
        if isinstance(x, np.ndarray):
            return (stark * dx, Fy * np.cos(self.tilt), - Fy * np.sin(self.tilt), np.zeros_like(x))
        else:
            return (stark * dx, Fy * np.cos(self.tilt), - Fy * np.sin(self.tilt), 0.0)

    def force_fast_scalar(
        self, t: float, x: float, y: float, z: float
    ) -> tuple[float, float, float]:
        return _force_eql_scalar(
            x,
            y,
            z,
            self.x0,
            self.y0,
            self.V,
            self.R,
            self._stark_potential_derivative.coef,
            self._cos_tilt,
            self._sin_tilt
        )

    def stark_potential_derivative(
        self,
        x: NDArray_or_Float,
        y: NDArray_or_Float,
        z: NDArray_or_Float,
    ) -> NDArray_or_Float:
        """
        Calculate the derivative (dStark/dE) of the stark potential at x,y,z

        Args:
            x (Union[NDArray[np.float64], float]): x coordinate(s) [m]
            y (Union[NDArray[np.float64], float]): y coordinate(s) [m]
            z (Union[NDArray[np.float64], float]): z coordinate(s) [m]

        Returns:
            Union[np.ndarray, float]: derivative of stark potential dV/dE
        """
        E = self.electric_field(x, y, z)
        return self._stark_potential_derivative(E)

    def stark_potential(
        self,
        x: NDArray_or_Float,
        y: NDArray_or_Float,
        z: NDArray_or_Float,
    ) -> NDArray_or_Float:
        """
        Calculate the the stark potential at x,y,z

        Args:
            x (Union[NDArray[np.float64], float]): x coordinate(s) [m]
            y (Union[NDArray[np.float64], float]): y coordinate(s) [m]
            z (Union[NDArray[np.float64], float]): z coordinate(s) [m]

        Returns:
            Union[NDArray[np.float64], float]: stark potential
        """
        electric_field = self.electric_field(x, y, z)
        return self._stark_potential(electric_field)

    def stark_potential_E(self, electric_field: NDArray_or_Float) -> NDArray_or_Float:
        """
        Calculate the stark potential as a function of electric field in V/m

        Args:
            electric_field (Union[np.ndarray, float]): electric field in V/m

        Returns:
            Union[np.ndarray, float]: stark potential
        """
        return self._stark_potential(electric_field)

    def stark_potential_E_derivative(
        self,
        electric_field: NDArray_or_Float,
    ) -> NDArray_or_Float:
        """
        Calculate the derivative (dStark/dE) of the stark potential as a function of E

        Args:
            electric_field (Union[np.ndarray, float]): electric field in V/m

        Returns:
            Union[np.ndarray, float]: stark potential derivative dV/dE
        """
        return self._stark_potential_derivative(electric_field)



lens_coeffs_default = np.array(
    [
        [
            0.00000000e00,
            0.00000000e00,
            -1.85807316e-03,
            -3.68838654e-01,
            9.05179603e01,
            5.02802652e02,
            -2.33774653e05,
        ],
        [
            0.00000000e00,
            -4.01516882e03,
            -4.57787118e-02,
            -8.53726662e03,
            -9.44499189e03,
            5.34213172e08,
            2.04894221e07,
        ],
        [
            1.55556469e-02,
            -1.08269983e01,
            4.98195718e02,
            1.21401040e05,
            -6.62042908e06,
            -2.00043668e08,
            1.63384388e10,
        ],
        [
            4.95047018e-01,
            -8.94339785e03,
            1.44796476e04,
            -1.69855041e09,
            -6.86084354e07,
            -6.47064425e09,
            -2.57901217e04,
        ],
        [
            -1.83341406e02,
            4.73566410e04,
            2.21617329e06,
            -3.02270553e08,
            -7.46259863e09,
            -1.14022371e05,
            -3.74750620e05,
        ],
        [
            -1.83757741e03,
            5.34389645e08,
            -1.65987718e07,
            6.46968544e09,
            -2.13239592e04,
            -1.31548209e02,
            -7.07835673e00,
        ],
        [
            2.78503072e05,
            -4.15374780e07,
            -1.08606156e09,
            -1.22530620e05,
            -2.91966193e06,
            -3.60532824e01,
            -5.51716726e02,
        ],
    ]
)


class ElectrostaticLensPolynomial(ElectrostaticQuadrupoleLens):
    """
    Electrostatic quadrupole lens using a 2D polynomial potential.

    This class extends ElectrostaticQuadrupoleLens by representing the lens
    potential as a 2D polynomial in x and y, scaled by the applied voltage V,
    with an optional z-dependence for more realistic field modeling.

    Attributes:
        _potential_xy_at_unit_voltage (Polynomial2D): Base polynomial potential at unit voltage.
        potential_xy (Polynomial2D): Scaled polynomial by current voltage.
        potential_z (Callable[[NDArray_or_Float], NDArray_or_Float]): Z-dependence of the potential.
        Ex, Ey (Polynomial2D): Electric field components -∂Φ/∂x, -∂Φ/∂y.
        Ex_x, Ex_y, Ey_x, Ey_y (Polynomial2D): Second derivatives of potential.
    """

    def __init__(
        self,
        name: str,
        objects: list[Any],
        start: float,
        stop: float,
        V: float,
        R: float,
        x: float = 0,
        y: float = 0,
        save_collisions: bool = False,
        stark_potential: None | npt.NDArray[np.float64] = None,
        potential_xy: Polynomial2D = Polynomial2D(
            kx=6, ky=6, coeffs=lens_coeffs_default
        ),
        potential_z: Optional[Callable[[NDArray_or_Float], NDArray_or_Float]] = None,
    ) -> None:
        """
        Initialize a polynomial-based electrostatic lens.

        Args:
            name: Identifier for this lens section.
            objects: Beamline objects contained in this section.
            start: Z-coordinate where the lens begins [m].
            stop: Z-coordinate where the lens ends [m].
            V: Initial voltage applied to electrodes [V].
            R: Lens bore radius [m] (for parent class consistency).
            x: Lens center x-coordinate [m].
            y: Lens center y-coordinate [m].
            save_collisions: Whether to record collisions inside this lens.
            stark_potential: Optional custom Stark potential coefficients.
            potential_xy: 2D polynomial defining unit-voltage potential in xy-plane.
            potential_z: Function defining z-dependence of potential. If None,
                         a constant function returning 1.0 will be used.
        """
        self._potential_xy = potential_xy
        self._V = V
        self._potential_xy_at_unit_voltage = potential_xy
        self.potential_xy = self._potential_xy_at_unit_voltage * self._V
        if potential_z is None:

            @overload
            def _potential_z(z: float) -> float: ...
            @overload
            def _potential_z(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

            def _potential_z(z):
                if isinstance(z, np.ndarray):
                    return np.ones(z.shape)
                else:
                    return 1.0

            self.potential_z = _potential_z
        else:
            self.potential_z = potential_z
        super().__init__(
            name, objects, start, stop, V, R, x, y, save_collisions, stark_potential
        )
        self._initialize_fields()

    @property
    def V(self) -> float:
        """Get the voltage applied to the lens.

        Returns:
            float: Current voltage in Volts.
        """
        return self._V

    @V.setter
    def V(self, voltage: float) -> None:
        """
        Set lens voltage and rescale the polynomial potential.

        Args:
            voltage: New voltage [V].
        """
        self._V = voltage
        self.potential_xy = self._potential_xy_at_unit_voltage * voltage
        self._initialize_fields()

    def _initialize_fields(self) -> None:
        """Initialize polynomial field components and their derivatives."""
        self.Ex = self.potential_xy.derivative("x")
        self.Ey = self.potential_xy.derivative("y")
        self.Ex_x = self.Ex.derivative("x")
        self.Ex_y = self.Ex.derivative("y")
        self.Ey_x = self.Ey.derivative("x")
        self.Ey_y = self.Ey.derivative("y")

    def stark_potential(
        self, x: NDArray_or_Float, y: NDArray_or_Float, z: NDArray_or_Float
    ) -> NDArray_or_Float:
        """
        Evaluate Stark potential at (x,y,z) based on polynomial electric field magnitude.

        Args:
            x: X-coordinates where potential is evaluated [m].
            y: Y-coordinates where potential is evaluated [m].
            z: Z-coordinates where potential is evaluated [m].

        Returns:
            Stark potential (energy) as function of |E| at the given points [J].
        """
        electric_field = np.linalg.norm(self.electric_field(x, y, z), axis=0)
        return self._stark_potential(electric_field)

    def electric_field(
        self, x: NDArray_or_Float, y: NDArray_or_Float, z: NDArray_or_Float
    ) -> NDArray_or_Float:
        """
        Compute electric field vector from the 2D polynomial potential.

        The field includes z-dependence through potential_z(z).

        Args:
            x: X-coordinates [m].
            y: Y-coordinates [m].
            z: Z-coordinates [m].

        Returns:
            Array of electric field components [Ex, Ey, Ez=0] in V/m.
        """
        _x = self.x_transformed(x)
        _y = self.y_transformed(y)
        pot_z = self.potential_z(z)
        if isinstance(_x, np.ndarray):
            return np.asarray(
                [
                    pot_z * self.Ex(_x, _y),
                    pot_z * self.Ey(_x, _y),
                    np.zeros(_x.shape),
                ]
            )
        else:
            return pot_z * np.asarray([self.Ex(_x, _y), self.Ey(_x, _y), 0.0])

    def stark_potential_derivative(
        self, x: NDArray_or_Float, y: NDArray_or_Float, z: NDArray_or_Float
    ) -> NDArray_or_Float:
        """
        Compute derivative d(Stark)/dE at (x,y,z) using the polynomial field.

        Args:
            x: X-coordinates where derivative is evaluated [m].
            y: Y-coordinates where derivative is evaluated [m].
            z: Z-coordinates where derivative is evaluated [m].

        Returns:
            d(Stark potential)/dE evaluated at |E|(x,y,z) [J/(V/m)].
        """
        E = np.linalg.norm(self.electric_field(x, y, z), axis=0)
        return self._stark_potential_derivative(E)

    def force(
        self, t: float, x: NDArray_or_Float, y: NDArray_or_Float, z: NDArray_or_Float
    ) -> tuple[NDArray_or_Float, NDArray_or_Float, NDArray_or_Float]:
        """
        Calculate force on a particle due to Stark interaction in polynomial lens.

        Uses F = –(dStark/dE) ∇|E|, where |E| is from the 2D polynomial with z-dependence.
        Optimized for performance with reduced redundant calculations.

        Args:
            t: Time [s] (unused; included for interface compatibility).
            x: X-position(s) [m].
            y: Y-position(s) [m].
            z: Z-position(s) [m].

        Returns:
            Tuple of force components (Fx, Fy, Fz=0) in Newtons.
        """
        # Transform coordinates once
        _x = self.x_transformed(x)
        _y = self.y_transformed(y)
        pot_z = self.potential_z(z)

        # Get electric field components directly
        Ex = pot_z * self.Ex(_x, _y)
        Ey = pot_z * self.Ey(_x, _y)

        # Get field derivatives
        Ex_x = pot_z * self.Ex_x(_x, _y)
        Ex_y = pot_z * self.Ex_y(_x, _y)
        Ey_x = pot_z * self.Ey_x(_x, _y)
        Ey_y = pot_z * self.Ey_y(_x, _y)

        # Calculate field magnitude with safe handling of small values
        electric_field = np.sqrt(Ex**2 + Ey**2)

        # Handle division by zero safely
        mask = electric_field > 1e-15
        if isinstance(electric_field, np.ndarray):
            electric_field_inverse = np.zeros_like(electric_field)
            electric_field_inverse[mask] = 1.0 / electric_field[mask]
        else:
            electric_field_inverse = 1.0 / electric_field if mask else 0.0

        # Calculate gradient of the field magnitude
        dEx = (Ex * Ex_x + Ey * Ey_x) * electric_field_inverse
        dEy = (Ex * Ex_y + Ey * Ey_y) * electric_field_inverse

        # Get Stark potential derivative at the field magnitude
        stark_potential_derivative = self._stark_potential_derivative(electric_field)

        # Calculate forces
        Fx = -stark_potential_derivative * dEx
        Fy = -stark_potential_derivative * dEy

        # Return optimized result
        if isinstance(x, np.ndarray):
            return Fx, Fy, np.zeros_like(x)
        else:
            return Fx, Fy, 0.0

    def force_fast_scalar(
        self, t: float, x: float, y: float, z: float
    ) -> tuple[float, float, float]:
        pot_z = self.potential_z(z)

        return _force_eql_poly_scalar(
            x,
            y,
            self.x0,
            self.y0,
            pot_z,
            self.Ex.coeffs,
            self.Ey.coeffs,
            self.Ex_x.coeffs,
            self.Ex_y.coeffs,
            self.Ey_x.coeffs,
            self.Ey_y.coeffs,
            self._stark_potential_derivative.coef,
        )


class MagnetostaticHexapoleLens(ODESection):
    """
    Magnetic Hexapole Lens class.

    Attributes:
        name (str): Name of the magnetic hexapole lens.
        objects (List[Any]): Objects inside the section that particles can collide with.
        start (float): Start of the section in z [m].
        stop (float): End of the section in z [m].
        particle (Particle): Particle interacting with the lens.
        Rin (float): Inner radius of the lens [m].
        Rout (float): Outer radius of the lens [m].
        M (int): Number of magnet sections.
        n (int): Number of poles (e.g., n=3 for hexapole).
        save_collisions (Optional[bool]): Whether to save the coordinates and velocities of collisions in this section.

    The magnetic field can be expressed as:
        Br = Bar * cos(3ϕ) * (r/Rin)**2
        Bϕ = -Bar * sin(3ϕ) * (r/Rin)**2
        Bz = 0
    From Manipulating beams of paramagnetic atoms and molecules using
    inhomogeneous magnetic fields (Progress in Nuclear Magnetic Resonance Spectroscopy,
    Volumes 120-121, https://doi.org/10.1016/j.pnmrs.2020.08.002)
    where Bar is given by:
        B0 * (n/(n-1)) * cos(π/M)**n * sin(2π/M)/(nπ/M) * (1-(Rin/Rout)**(n-1))
        if n > 1 or
        B0 * sin(2π/M)/(2π/M) * ln(Rout/Rin)
        if n = 1
    Here n is a measure for how many poles (Hexapole has n=3), M is the number of
    magnet sections, Rin is the inner radius, Rout the outer radius and B0 is a constant
    scaling the magnetic field.
    Expression from Eq 5 a/b from Application of permanent magnets in accelerators
    and electron storage rings (Journal of Applied Physics 57, 3605 (1985)).
    https://doi.org/10.1063/1.335021

    Converting to cartesian coordinates:
        r = sqrt(x**2 + y**2)
        ϕ = arctan2(y,x)
        sin(ϕ) = y/r
        cos(ϕ) = x/r
        sin(3ϕ) = 3*sin(ϕ)*cos(ϕ)**2 - sin(ϕ)**3
        cos(3ϕ) = cos(ϕ)**3 - 3*sin(ϕ)**2 * cos(ϕ)
        Bx = cos(ϕ)*Br - sin(ϕ)*Bϕ
        By = sin(ϕ)*Br + cos(ϕ)*Bϕ
        Bz = Bz

    Using sympy:
    ```Python
    import sympy as smp
    r, ϕ, Bar, Rin, x, y, μx, μy = smp.symbols("r ϕ Bar Rin x y μx μy")

    # substitutions
    subsine = [(smp.sin(ϕ), y/r), (smp.cos(ϕ), x/r)]
    subr = [(r, smp.sqrt(x**2 + y**2))]
    subsine3 = [
        (smp.sin(3*ϕ), 3*smp.sin(ϕ)*smp.cos(ϕ)**2 - smp.sin(ϕ)**3),
        (smp.cos(3*ϕ), smp.cos(ϕ)**3 - 3*smp.sin(ϕ)**2*smp.cos(ϕ))
    ]

    Br = Bar * smp.cos(3*ϕ) * (r/Rin)**2
    Bϕ = -Bar * smp.sin(3*ϕ) * (r/Rin)**2
    Bx = smp.cos(ϕ) * Br - smp.sin(ϕ)*Bϕ
    By = smp.sin(ϕ) * Br + smp.cos(ϕ)*Bϕ

    Bx = Bx.subs(subsine3 + subsine + subr).simplify()
    By = By.subs(subsine3 + subsine + subr).simplify()

    dBx = smp.sqrt(smp.diff(Bx,x)**2 + smp.diff(By,x)**2).simplify()
    dBy = smp.sqrt(smp.diff(Bx,y)**2 + smp.diff(By,y)**2).simplify()

    fx = (x/r * dBx).subs(subr).simplify() * -μx
    fy = (y/r * dBy).subs(subr).simplify() * -μy

    ```

    """

    def __init__(
        self,
        name: str,
        objects: List[Any],
        start: float,
        stop: float,
        particle: Particle,
        Rin: float,
        Rout: float,
        M: int,  # nr of blocks in the magnet
        n: int = 3,  # n = 2 -> quadrupole; n = 3 -> hexapole
        save_collisions: Optional[bool] = False,
    ) -> None:
        self.name = name
        self.objects = objects
        self.start = start
        self.stop = stop
        self.particle = particle

        self.Rin = Rin
        self.Rout = Rout
        self.n = n
        self.M = M
        self._magnetic_field_aperture_radius()

    @overload
    def force(
        self, t: float, x: float, y: float, z: float
    ) -> Tuple[float, float, float]: ...

    @overload
    def force(
        self,
        t: float,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> Tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]: ...

    def force(
        self,
        t,
        x,
        y,
        z,
    ):
        """
        Calculate the force at x,y,z

        The magnetic force on the dipole is given by F = ∇(μ⋅B).

        Args:
            t (float): time [s]
            x (Union[NDArray[np.float64], float]): x coordinate(s) [m]
            y (Union[NDArray[np.float64], float]): y coordinate(s) [m]
            z (Union[NDArray[np.float64], float]): z coordinate(s) [m]

        Returns:
            List: force in x, y and z
        """
        # μ points in the direction of B for a sufficiently large field (e.g. in the
        # hexapole), so μ points along x/r, y/r, 0

        fx = -2 * x * self.Bar / self.Rin**2 * self.particle.magnetic_moment
        fy = -2 * y * self.Bar / self.Rin**2 * self.particle.magnetic_moment
        fz = fx * 0.0
        return (fx, fy, fz)

    def _magnetic_field_aperture_radius(self) -> None:
        """
        Expression from Eq 5 a/b from Application of permanent magnets in accelerators
        and electron storage rings (Journal of Applied Physics 57, 3605 (1985)).
        https://doi.org/10.1063/1.335021
        """
        Br = 1
        n, M = self.n, self.M
        Rin, Rout = self.Rin, self.Rout
        if self.n > 1:
            self.Bar: float = (
                Br
                * (n / (n - 1))
                * np.cos(np.pi / M) ** n
                * (np.sin(n * np.pi / M) / (n * np.pi / M))
                * (1 - (Rin / Rout) ** (n - 1))
            )
        elif self.n == 1:
            self.Bar = Br * np.sin(np.pi / M) / (2 * np.pi / M) * np.log(Rout / Rin)

    def magnetic_field_cylindrical(
        self,
        r: NDArray_or_Float,
        ϕ: NDArray_or_Float,
        z: NDArray_or_Float,
    ) -> Tuple[NDArray_or_Float, NDArray_or_Float, NDArray_or_Float]:
        Br = self.Bar * np.cos(3 * ϕ) * (r / self.Rin) ** 2
        Bϕ = -self.Bar * np.sin(3 * ϕ) * (r / self.Rin) ** 2
        Bz = Br * 0.0
        return (Br, Bϕ, Bz)

    def magnetic_field_cartesian(
        self,
        x: NDArray_or_Float,
        y: NDArray_or_Float,
        z: NDArray_or_Float,
    ) -> Tuple[NDArray_or_Float, NDArray_or_Float, NDArray_or_Float]:
        r = np.sqrt(x**2 + y**2)
        ϕ = np.arctan2(y, x)
        Br, Bϕ, Bz = self.magnetic_field_cylindrical(r, ϕ, z)
        Bx = np.cos(ϕ) * Br - np.sin(ϕ) * Bϕ
        By = np.sin(ϕ) * Br + np.cos(ϕ) * Bϕ
        Bz = Bx * 0.0
        return (Bx, By, Bz)


@dataclass
class BeamlineObject:
    """
    Base class for objects within the beamline.

    Attributes:
        x (float): x-coordinate of the object [m].
        y (float): y-coordinate of the object [m].
        z (float): z-coordinate of the object [m].
    """

    x: float
    y: float
    z: float

    def check_in_bounds(self, start: float, stop: float) -> bool:
        """
        Check if the aperture is within the specified bounds

        Args:
            start (float): start coordinate in z [m]
            stop (float): stop coordinate in z [m]

        Returns:
            bool: True if within bounds, else False
        """
        return (self.z >= start) & (self.z <= stop)

    def get_acceptance(
        self,
        start: Coordinates,
        stop: Coordinates,
        vels: Velocities,
        acceleration: Acceleration,
    ) -> npt.NDArray[np.bool_]:
        raise NotImplementedError

    def collision_event_function(self, x: float, y: float, z: float) -> float:
        raise NotImplementedError


@dataclass
class CircularAperture(BeamlineObject):
    """
    Represents a circular aperture.

    Attributes:
        x (float): x-coordinate of the aperture center [m].
        y (float): y-coordinate of the aperture center [m].
        z (float): z-coordinate of the aperture center [m].
        r (float): Radius of the aperture [m].
    """

    r: float

    @property
    def z_stop(self):
        return self.z

    def get_acceptance(
        self,
        start: Coordinates,
        stop: Coordinates,
        vels: Velocities,
        acceleration: Acceleration,
    ) -> npt.NDArray[np.bool_]:
        """
        check if the supplied coordinates are within the aperture

        Args:
            coords (Coordinates): coordinates to check

        Returns:
            np.ndarray: boolean array where True indicates coordinates are within the
            aperture
        """
        assert np.allclose(stop.z, self.z), (
            "supplied coordinates not at location of aperture"
        )
        return (stop.x - self.x) ** 2 + (stop.y - self.y) ** 2 <= self.r**2

    def collision_event_function(self, x: float, y: float, z: float) -> float:
        return (x - self.x) + (y - self.y) + (z - self.z)


@dataclass
class RectangularAperture(BeamlineObject):
    """
    Represents a rectangular aperture.

    Attributes:
        x (float): x-coordinate of the aperture center [m].
        y (float): y-coordinate of the aperture center [m].
        z (float): z-coordinate of the aperture center [m].
        wx (float): Width of the aperture along the x-axis [m].
        wy (float): Height of the aperture along the y-axis [m].
    """

    wx: float
    wy: float

    @property
    def z_stop(self):
        return self.z

    def get_acceptance(
        self,
        start: Coordinates,
        stop: Coordinates,
        vels: Velocities,
        acceleration: Acceleration,
    ) -> npt.NDArray[np.bool_]:
        """
        check if the supplied coordinates are within the aperture

        Args:
            coords (Coordinates): coordinates to check

        Returns:
            np.ndarray: boolean array where True indicates coordinates are within the
            aperture
        """
        assert np.allclose(stop.z, self.z), (
            "supplied coordinates not at location of aperture"
        )
        return (np.abs((stop.x - self.x)) <= self.wx / 2) & (
            np.abs((stop.y - self.y)) <= self.wy / 2
        )

    def collision_event_function(self, x: float, y: float, z: float) -> float:
        x_factor = 0 if (x >= self.x - self.wx / 2 and x <= self.x + self.wx / 2) else 1
        y_factor = 0 if (y >= self.y - self.wy / 2 and y <= self.y + self.wy / 2) else 1
        return z - self.z + x_factor + y_factor


@dataclass
class RectangularApertureFinite(BeamlineObject):
    """
    Represents a rectangular aperture with finite plate dimensions.

    Attributes:
        x (float): x-coordinate of the aperture center [m].
        y (float): y-coordinate of the aperture center [m].
        z (float): z-coordinate of the aperture center [m].
        wx (float): Width of the aperture along the x-axis [m].
        wy (float): Height of the aperture along the y-axis [m].
        wxp (float): Width of the plate containing the aperture [m].
        wyp (float): Height of the plate containing the aperture [m].
    """

    wx: float
    wy: float
    wxp: float
    wyp: float

    @property
    def z_stop(self):
        return self.z

    def get_acceptance(
        self,
        start: Coordinates,
        stop: Coordinates,
        vels: Velocities,
        acceleration: Acceleration,
    ) -> npt.NDArray[np.bool_]:
        """
        check if the supplied coordinates are within the aperture

        Args:
            start (Coordinates): start coordinates
            stop (Coordinates): stop coordinates
            vels (Velocities): velocities of the particles
            force (Force): force acting on the particles

        Returns:
            np.ndarray: boolean array where True indicates coordinates are within the
            aperture
        """
        assert np.allclose(stop.z, self.z), (
            "supplied coordinates not at location of aperture"
        )
        inside_aperture = (np.abs((stop.x - self.x)) <= self.wx / 2) & (
            np.abs((stop.y - self.y)) <= self.wy / 2
        )
        outside_plate = (np.abs((stop.x - self.x)) >= self.wxp / 2) & (
            np.abs((stop.y - self.y)) >= self.wyp / 2
        )
        return inside_aperture | outside_plate

    def collision_event_function(self, x: float, y: float, z: float) -> float:
        inside_aperture_x = (x >= self.x - self.wx / 2) and (x <= self.x + self.wx / 2)
        outside_plate_x = (x >= self.x + self.wxp / 2) or (x <= self.x - self.wxp / 2)
        inside_aperture_y = (y >= self.y - self.wy / 2) and (y <= self.y + self.wy / 2)
        outside_plate_y = (y >= self.y + self.wyp / 2) or (y <= self.y - self.wyp / 2)
        x_factor = 0 if inside_aperture_x or outside_plate_x else 1
        y_factor = 0 if inside_aperture_y or outside_plate_y else 1
        return z - self.z + x_factor + y_factor


@dataclass
class PlateElectrodes(BeamlineObject):
    """
    Represents a pair of plate electrodes.

    Attributes:
        x (float): x-coordinate of the electrode center [m].
        y (float): y-coordinate of the electrode center [m].
        z (float): z-coordinate of the electrode center [m].
        length (float): Length of the electrodes along the z-axis [m].
        width (float): Width of the electrodes along the y-axis [m].
        separation (float): Separation distance between the electrodes [m].
    """

    x: float
    y: float
    z: float
    length: float
    width: float
    separation: float

    @property
    def z_stop(self) -> float:
        return self.z + self.length

    def check_in_bounds(self, start: float, stop: float) -> bool:
        return self.z >= start and self.z + self.length <= stop

    def get_acceptance(
        self,
        start: Coordinates,
        stop: Coordinates,
        vels: Velocities,
        acceleration: Acceleration,
    ) -> npt.NDArray[np.bool_]:
        m, _, _ = self.get_collisions(start, stop, vels, acceleration)
        return ~m

    def get_collisions(
        self,
        start: Coordinates,
        stop: Coordinates,
        vels: Velocities,
        acceleration: Acceleration,
    ) -> Tuple[npt.NDArray[np.bool_], Coordinates, Velocities]:
        """
        Calculate collisions with the plate electrodes.

        Args:
            start (Coordinates): Start coordinates of the particles.
            stop (Coordinates): Stop coordinates of the particles.
            vels (Velocities): Velocities of the particles.
            acceleration (Acceleration): Acceleration acting on the particles.

        Returns:
            Tuple[npt.NDArray[np.bool_], Coordinates, Velocities]:
                - Boolean array indicating collisions.
                - Coordinates of collisions.
                - Velocities at collisions.
        """
        # Calculate time to reach the plates
        dx_upper = (self.x + self.separation / 2) - start.x
        dx_lower = (self.x - self.separation / 2) - start.x

        # Determine collision times
        t_upper = calculate_time_ballistic(dx_upper, vels.vx, acceleration.ax)
        t_lower = calculate_time_ballistic(dx_lower, vels.vx, acceleration.ax)

        # Propagate positions and velocities
        t = np.where(vels.vx > 0, t_upper, t_lower)
        x, v = propagate_ballistic(t, start, vels, acceleration)

        # Check if collisions occur within the plate bounds
        mask = (self.z <= x.z) & (x.z <= self.z + self.length)
        mask &= abs(x.y - self.y) <= self.width / 2

        return mask, x.get_masked(mask), v.get_masked(mask)

    def collision_event_function(self, x: float, y: float, z: float) -> float:
        """
        Calculate the collision event function for a particle.

        Args:
            x (float): x-coordinate of the particle [m].
            y (float): y-coordinate of the particle [m].
            z (float): z-coordinate of the particle [m].

        Returns:
            float: Value of the collision event function.
        """
        # Check if the particle is within the z bounds of the electrodes
        z_factor = 0 if self.z <= z <= self.z + self.length else 1

        # Check if the particle is within the x bounds of the electrodes
        x_factor = (
            0
            if self.x - self.separation / 2 <= x <= self.x + self.separation / 2
            else 1
        )

        # Check if the particle is within the y bounds of the electrodes
        y_factor = 0 if abs(y - self.y) <= self.width / 2 else 1

        return z_factor + x_factor + y_factor


@dataclass
class Bore(BeamlineObject):
    """
    Cylindrical bore.

    Attributes:
        x (float): x-coordinate of the bore center [m].
        y (float): y-coordinate of the bore center [m].
        z (float): z-coordinate of the bore start [m].
        length (float): Length of the bore along the z-axis [m].
        radius (float): Radius of the bore [m].
    """

    x: float
    y: float
    z: float
    length: float
    radius: float

    @property
    def z_stop(self) -> float:
        """
        Calculate the z-coordinate of the bore's end.

        Returns:
            float: z-coordinate of the bore's end [m].
        """
        return self.z + self.length

    def check_in_bounds(
        self, start: float, stop: float, rel_tol: float = 1e-9, abs_tol: float = 0.0
    ) -> bool:
        """
        Check if the bore is within the specified z-coordinate bounds.

        Args:
            start (float): Start coordinate in z [m].
            stop (float): Stop coordinate in z [m].
            rel_tol (float): Relative tolerance for boundary checks.
            abs_tol (float): Absolute tolerance for boundary checks.

        Returns:
            bool: True if the bore is within bounds, False otherwise.
        """
        return (
            math.isclose(self.z, start, rel_tol=rel_tol, abs_tol=abs_tol)
            or self.z > start
        ) and (
            math.isclose(self.z + self.length, stop, rel_tol=rel_tol, abs_tol=abs_tol)
            or self.z + self.length < stop
        )

    def get_acceptance(
        self,
        start: Coordinates,
        stop: Coordinates,
        vels: Velocities,
        acceleration: Acceleration,
    ) -> npt.NDArray[np.bool_]:
        """
        Determine which particles pass through the bore without collision.

        Args:
            start (Coordinates): Start coordinates of the particles.
            stop (Coordinates): Stop coordinates of the particles.
            vels (Velocities): Velocities of the particles.
            acceleration (Acceleration): Acceleration acting on the particles.

        Returns:
            npt.NDArray[np.bool_]: Boolean array indicating particle acceptance.
        """
        mask_z = (stop.z >= self.z) & (stop.z <= self.z + self.length)
        mask_r = ((stop.x - self.x) ** 2 + (stop.y - self.y) ** 2) > self.radius**2
        accept_array = np.ones(stop.x.shape, dtype=np.bool_)
        accept_array[mask_z & mask_r] = False
        return accept_array

    def collision_event_function(self, x: float, y: float, z: float) -> float:
        """
        Calculate the collision event function for a particle.

        Args:
            x (float): x-coordinate of the particle [m].
            y (float): y-coordinate of the particle [m].
            z (float): z-coordinate of the particle [m].

        Returns:
            float: Value of the collision event function.
        """
        r = np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)

        # z_factor for checking if z coordinates are within the bore
        z_factor = int(not bounds_check_tolerance(z, self.z, self.z + self.length))

        return (r - self.radius) + z_factor

    def get_collisions_linear(
        self,
        start: Coordinates,
        stop: Coordinates,
        vels: Velocities,
        acc: Acceleration,
        w: Tuple[float, float, float],  # (ω,ω,0)
        trap_center: Tuple[float, float],  # (sx,sy)
        *,
        scan_fraction: int = 8,  # bracket step = T/scan_fraction
    ) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64], Coordinates, Velocities]:
        """
        Calculate collisions for particles moving linearly through the bore.

        Args:
            start (Coordinates): Start coordinates of the particles.
            stop (Coordinates): Stop coordinates of the particles.
            vels (Velocities): Velocities of the particles.
            acc (Acceleration): Acceleration acting on the particles.
            w (Tuple[float, float, float]): Angular frequencies (ω, ω, 0).
            trap_center (Tuple[float, float]): Center of the trap (sx, sy).
            scan_fraction (int, optional): Fraction of the period for bracketing steps. Defaults to 8.

        Returns:
            Tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64], Coordinates, Velocities]:
                - Boolean array indicating collisions.
                - Array of collision times.
                - Coordinates of collisions.
                - Velocities at collisions.
        """
        # ---------- unpack arrays ------------------------------------------
        x0, y0, z0 = map(np.asarray, (start.x, start.y, start.z))
        vx0, vy0, vz0 = map(np.asarray, (vels.vx, vels.vy, vels.vz))
        z_goal = np.asarray(stop.z) if np.ndim(stop.z) else np.full_like(z0, stop.z)
        N = x0.size
        ω = w[0]
        if not (np.allclose(w[0], w[1]) and w[2] == 0 and ω > 0):
            raise ValueError(f"need isotropic trap with w=(ω,ω,0) and ω>0, w = {w}")

        # cylinder axis (broadcast scalars)
        cx = np.asarray(self.x) if np.ndim(self.x) else np.full(N, self.x)
        cy = np.asarray(self.y) if np.ndim(self.y) else np.full(N, self.y)

        # spring centre
        sx, sy = trap_center
        sx = np.asarray(sx) if np.ndim(sx) else np.full(N, sx)
        sy = np.asarray(sy) if np.ndim(sy) else np.full(N, sy)

        # static shift = trap centre + const-force offset
        ox = sx + acc.ax / ω**2
        oy = sy + acc.ay / ω**2

        # coefficients in cylinder frame   R(t)=C+ A cos + B sin
        Cx, Cy = ox - cx, oy - cy
        Ax, Ay = x0 - ox, y0 - oy
        Bx, By = vx0 / ω, vy0 / ω

        R = float(self.radius)
        R2 = R * R
        az = acc.az
        T = 2 * math.pi / ω  # full period
        dt = T / scan_fraction  # safe bracket step  (π/4ω default)

        # ---------- flight time to stop.z  ---------------------------------
        dz = z_goal - z0
        if abs(az) < 1e-14:
            with np.errstate(divide="ignore", invalid="ignore"):
                t_end = dz / vz0
        else:
            a, b, c = 0.5 * az, vz0, -dz
            disc = b * b - 4 * a * c
            disc[disc < 0] = 0
            rt = np.sqrt(disc)
            t1, t2 = (-b - rt) / (2 * a), (-b + rt) / (2 * a)
            t_end = np.where(dz >= 0, np.maximum(t1, t2), np.minimum(t1, t2))
        valid = (t_end > 0) & np.isfinite(t_end)

        # ---------- outputs -------------------------------------------------
        hit = np.zeros(N, bool)
        t_hit = np.full(N, np.nan)
        xh = np.full(N, np.nan)
        yh = np.full(N, np.nan)
        zh = np.full(N, np.nan)
        vxh = np.full(N, np.nan)
        vyh = np.full(N, np.nan)
        vzh = np.full(N, np.nan)

        if not valid.any():  # nothing moves forward
            return hit, t_hit, Coordinates(xh, yh, zh), Velocities(vxh, vyh, vzh)

        survivors = np.where(valid)[0]

        # ===== loop over survivors =========================================
        for i in survivors:
            # ----- exact peak radius (isotropic) ---------------------------
            AA, BB, AB = (
                Ax[i] ** 2 + Ay[i] ** 2,
                Bx[i] ** 2 + By[i] ** 2,
                Ax[i] * Bx[i] + Ay[i] * By[i],
            )
            amp = math.sqrt(0.5 * (AA + BB) + 0.5 * math.hypot(AA - BB, 2 * AB))
            if math.hypot(Cx[i], Cy[i]) + amp < R - 1e-9:
                continue  # geometrically impossible

            # analytic x,y in cylinder frame
            def xy(t):
                Cθ, Sθ = math.cos(ω * t), math.sin(ω * t)
                return (
                    Cx[i] + Ax[i] * Cθ + Bx[i] * Sθ,
                    Cy[i] + Ay[i] * Cθ + By[i] * Sθ,
                )

            def f(t: float) -> float:
                x_rel, y_rel = xy(t)
                return x_rel * x_rel + y_rel * y_rel - R2

            # starts outside?
            if f(0.0) > 0:
                hit[i] = True
                t_hit[i] = 0.0
                xh[i] = x0[i]
                yh[i] = y0[i]
                zh[i] = z0[i]
                vxh[i] = vx0[i]
                vyh[i] = vy0[i]
                vzh[i] = vz0[i]
                continue

            # ----- bracket first root with step dt -------------------------
            a, fa = 0.0, 0.0
            b = min(float(t_end[i]), dt)
            fb = f(b)
            while fb < 0 and b < t_end[i]:
                a, fa = b, fb
                b = min(b + dt, t_end[i])
                fb = f(b)
            if fb < 0:  # never crosses
                continue

            # ----- Brent root ----------------------------------------------
            MAX_ITER = 60
            t0 = brentq(f, a, b, xtol=1e-12, rtol=1e-10, maxiter=MAX_ITER)

            # Position & velocity at impact
            xr, yr = xy(t0)
            if abs(math.hypot(xr, yr) - R) > 1e-8:  # 10 nm tolerance
                continue

            hit[i] = True
            t_hit[i] = t0
            xh[i] = cx[i] + xr
            yh[i] = cy[i] + yr
            zh[i] = z0[i] + vz0[i] * t0 + 0.5 * az * t0 * t0
            vxh[i] = -ω * Ax[i] * math.sin(ω * t0) + ω * Bx[i] * math.cos(ω * t0)
            vyh[i] = -ω * Ay[i] * math.sin(ω * t0) + ω * By[i] * math.cos(ω * t0)
            vzh[i] = vz0[i] + az * t0

        return hit, t_hit, Coordinates(xh, yh, zh), Velocities(vxh, vyh, vzh)
