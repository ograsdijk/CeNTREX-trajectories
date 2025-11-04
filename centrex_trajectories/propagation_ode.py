"""
ODE-Based Trajectory Propagation
=================================

This module implements trajectory propagation using adaptive-step ODE integration
for sections with position-dependent forces. It's designed for scenarios where
analytical solutions are not available, such as:

- Electrostatic lenses with complex field geometries
- Magnetic field gradients
- Stark-effect-based focusing elements
- Any spatially-varying force fields

Integration Strategy
--------------------
Uses scipy.integrate.solve_ivp with:
- Adaptive step-size control (RK45 method)
- Event detection for collision/boundary conditions
- Parallel processing across multiple trajectories (joblib)
- Configurable tolerance (default: rtol=1e-7, atol=1e-7)

The equations of motion are:
    dx/dt = vx,    dvx/dt = Fx/m
    dy/dt = vy,    dvy/dt = Fy/m
    dz/dt = vz,    dvz/dt = Fz/m

where F = F_position_dependent(t,x,y,z) + F_constant

Event System
------------
The module supports two types of events:
1. **z_stop_event**: Terminal event when particle reaches target z position
2. **collision_events**: Terminal events for hitting physical objects

All events stop integration when triggered (terminal=True). 
"""

from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult

from .data_structures import Coordinates, Force, Velocities
from .particles import TlF
from .propagation_options import PropagationOptions

__all__: List[str] = []


def z_stop_event_generator(
    z_stop: float,
) -> Callable[[float, npt.NDArray[np.float64]], float]:
    """
    Generate a terminal event function for solve_ivp that triggers when z reaches z_stop.

    This creates a closure that captures the target z position and returns an event
    function compatible with scipy.integrate.solve_ivp's event system. The event
    function returns zero when the particle reaches z_stop, causing integration to halt.

    Args:
        z_stop (float):
            Target z coordinate at which to stop integration. Should be greater than
            the initial z position for forward propagation.

    Returns:
        Callable[[float, npt.NDArray[np.float64]], float]:
            Event function that takes time and state vector [x,y,z,vx,vy,vz] and
            returns (z - z_stop). The function has terminal=True attribute set.

    Notes:
        The event function is marked as terminal, meaning integration stops when
        the event is triggered (when the return value crosses zero).
    """

    def event(
        t: float,
        y: npt.NDArray[np.float64],
    ) -> float:
        # Return (z - z_stop); zero crossing indicates arrival at z_stop
        return y[2] - z_stop

    # Mark as terminal event: stop integration when triggered
    event.terminal = True  # type: ignore[attr-defined]
    return event


def collision_event_generator(
    collision_events: Sequence[Callable[[float, float, float], float]],
) -> Sequence[Callable[[float, npt.NDArray[np.float64]], float]]:
    """
    Convert collision event functions to solve_ivp-compatible event functions.

    Beamline objects (apertures, walls, etc.) provide collision event functions that
    take (x, y, z) coordinates and return a signed distance (negative=inside object,
    positive=outside object, zero=on surface). This function wraps them to work with
    solve_ivp's event system which expects functions of (t, state_vector).

    Args:
        collision_events (Sequence[Callable[[float, float, float], float]]):
            List of collision event functions from beamline objects. Each function
            takes (x, y, z) and returns signed distance to object boundary.

    Returns:
        Sequence[Callable[[float, npt.NDArray[np.float64]], float]]:
            List of wrapped event functions compatible with solve_ivp. Each function
            takes (t, [x,y,z,vx,vy,vz]) and returns signed distance. All are marked
            as terminal events.

    Notes:
        Each wrapped event function is marked as terminal, so integration stops
        immediately when any collision is detected (signed distance crosses zero).
    """

    def create_event(
        collision_event: Callable[[float, float, float], float],
    ) -> Callable[[float, npt.NDArray[np.float64]], float]:
        """Create a single wrapped event function."""

        def event(t: float, y: npt.NDArray[np.float64]) -> float:
            # Extract position from state vector and call original collision function
            return collision_event(y[0], y[1], y[2])

        # Mark as terminal: stop integration on collision
        event.terminal = True  # type: ignore[attr-defined]
        return event

    # Wrap all collision events
    return [create_event(collision_event) for collision_event in collision_events]


def solve_ode(
    t: float,
    x: Coordinates,
    v: Velocities,
    z_stop: float,
    mass: float,
    force: Callable[[float, float, float, float], Tuple[float, float, float]],
    force_cst: Force,
    events: List[Callable[[float, npt.NDArray[np.float64]], float]],
) -> OptimizeResult:
    """
    Solve the trajectory propagation ODE for a single particle trajectory.

    Integrates the equations of motion for a particle moving through a region with
    position-dependent forces. Uses adaptive-step RK45 integration with event detection
    for collision and boundary conditions.

    Args:
        t (float):
            Initial time at the start of propagation.
        x (Coordinates):
            Initial position of the particle (x, y, z).
        v (Velocities):
            Initial velocity of the particle (vx, vy, vz).
        z_stop (float):
            Target z coordinate where integration should end. Must be greater than x.z
            for forward propagation.
        mass (float):
            Particle mass in kg.
        force (Callable[[float, float, float, float], Tuple[float, float, float]]):
            Position-dependent force function: (t, x, y, z) -> (Fx, Fy, Fz) in Newtons.
        force_cst (Force):
            Constant force component (e.g., gravity) added to position-dependent force.
        events (List[Callable[[float, npt.NDArray[np.float64]], float]]):
            List of event functions for collision detection. Each function takes
            (t, state_vector) and returns signed distance to boundary.

    Returns:
        OptimizeResult:
            Solution object from solve_ivp containing:
            - t: Array of time points
            - y: Array of state vectors [x, y, z, vx, vy, vz] at each time point
            - success: Boolean indicating if integration succeeded
            - message: Description of termination reason
            - t_events: Times when events were triggered

    Raises:
        ValueError:
            If z_stop <= initial z position (invalid integration direction).

    Notes:
        - Integration time span is estimated as 2 * Δz / vz (conservative upper bound)
        - Uses rtol=1e-7, atol=1e-7 for high accuracy
        - Integration stops early if any event is triggered (collision or z_stop reached)
    """
    # Estimate integration time span (factor of 2 for safety margin)
    t_span = [t, t + 2 * (z_stop - x.z) / v.vz]

    # Validate that we're propagating in the positive z direction
    if t_span[1] <= t_span[0]:
        raise ValueError(
            f"Invalid integration direction: z_stop ({z_stop}) must be greater than "
            f"initial z position ({x.z}). Check that particle velocity is positive."
        )

    # Create z_stop event and combine with collision events
    z_stop_event = z_stop_event_generator(z_stop)
    all_events = events + [z_stop_event]

    # Pack initial state into vector [x, y, z, vx, vy, vz]
    initial_state = [x.x, x.y, x.z, v.vx, v.vy, v.vz]

    # Create partial function with fixed parameters for cleaner solve_ivp call
    ode_rhs = partial(ode_fun, mass=mass, force_fn=force, force_cst=force_cst)

    # Integrate equations of motion with adaptive step-size control
    sol = solve_ivp(
        ode_rhs,
        t_span,
        initial_state,
        events=all_events,
        rtol=1e-7,  # Relative tolerance
        atol=1e-7,  # Absolute tolerance
        method="RK45",  # Explicit Runge-Kutta method of order 5(4)
    )

    return sol


def ode_fun(
    t: float,
    d: List[float],
    mass: float,
    force_fn: Callable[[float, float, float, float], Tuple[float, float, float]],
    force_cst: Force,
) -> Tuple[float, float, float, float, float, float]:
    """
    Right-hand side (RHS) of the trajectory propagation ODE system.

    This function defines the system of first-order ODEs that govern particle motion:
        dx/dt = vx
        dy/dt = vy
        dz/dt = vz
        dvx/dt = (Fx_position_dependent + Fx_constant) / mass
        dvy/dt = (Fy_position_dependent + Fy_constant) / mass
        dvz/dt = (Fz_position_dependent + Fz_constant) / mass

    The position-dependent force is evaluated at the current position, and the constant
    force (e.g., gravity) is added to get the total acceleration.

    Args:
        t (float):
            Current time in the integration (seconds).
        d (List[float]):
            State vector [x, y, z, vx, vy, vz] where:
            - x, y, z: Position in meters
            - vx, vy, vz: Velocity in m/s
        mass (float):
            Particle mass in kg.
        force_fn (Callable[[float, float, float, float], Tuple[float, float, float]]):
            Position-dependent force function: (t, x, y, z) -> (Fx, Fy, Fz) in Newtons.
            This typically comes from electrostatic lenses, magnetic gradients, etc.
        force_cst (Force):
            Constant force component (e.g., gravity) in Newtons.

    Returns:
        Tuple[float, float, float, float, float, float]:
            Time derivatives [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt] of the
            state vector. This is the RHS of the ODE system d(state)/dt = f(t, state).

    Notes:
        This function is called repeatedly by the ODE solver at different time points
        during adaptive step-size integration. It should be as efficient as possible.
    """
    # Unpack state vector into position and velocity components
    x, y, z, vx, vy, vz = d

    # Evaluate position-dependent force at current location
    fx, fy, fz = force_fn(t, x, y, z)

    # Calculate acceleration from position-dependent force (F = ma -> a = F/m)
    ax = fx / mass
    ay = fy / mass
    az = fz / mass

    # Return derivatives: [velocity components, total acceleration components]
    # Total acceleration = position-dependent + constant (gravity, etc.)
    return (
        vx,  # dx/dt
        vy,  # dy/dt
        vz,  # dz/dt
        ax + force_cst.fx / mass,  # dvx/dt
        ay + force_cst.fy / mass,  # dvy/dt
        az + force_cst.fz / mass,  # dvz/dt
    )


def propagate_ODE_trajectories(
    t_start: npt.NDArray[np.float64],
    origin: Coordinates,
    velocities: Velocities,
    z_stop: float,
    mass: float,
    force_fun: Callable[[float, float, float, float], Tuple[float, float, float]],
    force_cst: Force = Force(0.0, -9.81 * TlF().mass, 0.0),
    events: Sequence[Callable[[float, float, float], float]] = [],
    z_save: Optional[
        Union[List[Union[float, int]], npt.NDArray[Union[np.float64, np.int_]]]
    ] = None,
    options: PropagationOptions = PropagationOptions(),
) -> List[OptimizeResult]:
    """
    Propagate multiple particle trajectories through an ODE section using parallel processing.

    This is the main entry point for ODE-based trajectory propagation. It handles multiple
    trajectories in parallel using joblib, with each trajectory integrated independently.
    Use this for sections with position-dependent forces where analytical solutions are
    not available.

    Args:
        t_start (npt.NDArray[np.float64]):
            Initial time for each trajectory (seconds). Array length = number of trajectories.
        origin (Coordinates):
            Initial positions for all trajectories. Contains arrays x, y, z in meters.
        velocities (Velocities):
            Initial velocities for all trajectories. Contains arrays vx, vy, vz in m/s.
        z_stop (float):
            Target z coordinate where integration should end for all trajectories (meters).
            Must be greater than all initial z positions.
        mass (float):
            Particle mass in kg (same for all trajectories).
        force_fun (Callable[[float, float, float, float], Tuple[float, float, float]]):
            Position-dependent force function: (t, x, y, z) -> (Fx, Fy, Fz) in Newtons.
            This is typically the `force_fast_scalar` method from an ODESection object.
        force_cst (Force, optional):
            Constant force component (e.g., gravity) in Newtons.
            Defaults to TlF gravity: Force(0.0, -9.81 * TlF().mass, 0.0).
        events (Sequence[Callable[[float, float, float], float]], optional):
            List of collision event functions from beamline objects. Each takes (x, y, z)
            and returns signed distance to boundary. Defaults to [] (no collisions).
        z_save (Optional[Union[List[Union[float, int]], npt.NDArray]], optional):
            Z positions where intermediate states should be saved. Currently unused but
            reserved for future dense output functionality. Defaults to None.
        options (PropagationOptions, optional):
            Configuration for propagation behavior including:
            - n_cores: Number of parallel workers (-1 for all cores)
            - verbose: Verbosity level for joblib progress
            Defaults to PropagationOptions() (single core, no verbosity).

    Returns:
        List[OptimizeResult]:
            List of ODE solution objects, one per trajectory. Each OptimizeResult contains:
            - t: Array of time points where solution was evaluated
            - y: State array [x, y, z, vx, vy, vz] at each time point
            - success: Boolean indicating if integration succeeded
            - message: Description of termination reason
            - t_events: List of times when events were triggered
            
            The order matches the input order. Trajectories that hit objects will
            terminate early (check y[2,-1] < z_stop to detect collisions).

    Raises:
        ValueError:
            - If input arrays have mismatched lengths
            - If Parallel returns None (should not happen in normal operation)

    Performance Notes:
        - Parallel processing provides ~linear speedup with core count up to physical cores
        - Each trajectory is independent, no inter-trajectory communication
        - Memory usage scales linearly with number of trajectories and integration steps
        - For N trajectories on K cores: wall time ≈ (N/K) * single_trajectory_time
        - Overhead from joblib is typically <1% of total computation time

    Example:
        >>> # Propagate 1000 TlF molecules through an electrostatic lens
        >>> coords = Coordinates(x=x_array, y=y_array, z=z_start)
        >>> vels = Velocities(vx=vx_array, vy=vy_array, vz=vz_array)
        >>> t_init = np.zeros(1000)
        >>> 
        >>> solutions = propagate_ODE_trajectories(
        ...     t_start=t_init,
        ...     origin=coords,
        ...     velocities=vels,
        ...     z_stop=0.5,  # Stop at 0.5 m
        ...     mass=TlF().mass,
        ...     force_fun=lens.force_fast_scalar,
        ...     events=[aperture.collision_event_function],
        ...     options=PropagationOptions(n_cores=-1)  # Use all cores
        ... )
        >>> 
        >>> # Check how many made it through
        >>> survivors = sum(sol.y[2, -1] >= 0.5 for sol in solutions)
    """
    # Validate input array lengths
    if len(t_start) != len(origin) or len(origin) != len(velocities):
        raise ValueError(
            f"Input array length mismatch: t_start has {len(t_start)} elements, "
            f"origin has {len(origin)}, velocities has {len(velocities)}. "
            f"All must have the same length (number of trajectories)."
        )

    # Convert collision events to solve_ivp-compatible format
    collision_events = collision_event_generator(events)

    # Parallel integration of all trajectories
    # joblib.Parallel distributes work across n_cores worker processes
    # delayed(solve_ode) creates a callable for each trajectory
    solutions = Parallel(n_jobs=options.n_cores, verbose=int(options.verbose))(
        delayed(solve_ode)(
            t, x, v, z_stop, mass, force_fun, force_cst, collision_events
        )
        for t, x, v in zip(t_start, origin, velocities)
    )
    
    # Safety check (Parallel should never return None with this usage pattern)
    if solutions is None:
        raise ValueError(
            "Parallel processing returned None. This indicates a critical error "
            "in the joblib backend."
        )

    # Convert to list and assert type for type checker
    # joblib.Parallel returns the same type as the generator would produce
    return list(solutions)  # type: ignore[return-value]
