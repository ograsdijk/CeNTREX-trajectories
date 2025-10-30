from copy import deepcopy
from typing import List, Optional, Tuple, Union
import cupy as cp
import numpy as np
import numpy.typing as npt

from .data_structures import Acceleration, Coordinates, Velocities
from .propagation_options import PropagationOptions

__all__: List[str] = []


def propagate_ballistic(
    t: npt.NDArray[np.float64],
    origin: Coordinates,
    velocities: Velocities,
    acceleration: Acceleration,
) -> Tuple[Coordinates, Velocities]:
    """
    Propagate trajectories starting at `origin` with `velocities` for a time `t`.

    Args:
        t (npt.NDArray[np.float64]): 1D array of times to propagate per particle.
        origin (Coordinates): Initial coordinates of each particle.
        velocities (Velocities): Initial velocities of each particle.
        acceleration (Acceleration): Acceleration acting on each particle.

    Returns:
        Tuple[Coordinates, Velocities]: Coordinates and velocities after propagating for time `t`.

    Notes:
        - `t` must be a 1D array with the same length as the number of particles.
        - Negative or infinite values in `t` are not allowed.
        - If `t` is zero, the function returns the initial `origin` and `velocities`.
    """
    # Validate inputs
    if not isinstance(t, cp.ndarray):
        raise TypeError("`t` must be a CuPy array.")
    if t.ndim != 1:
        raise ValueError("`t` must be a 1-dimensional array.")
    if np.any(t < 0):
        raise ValueError("`t` contains negative values, which are not allowed.")
    if np.any(np.isinf(t)):
        raise ValueError("`t` contains infinite values, which are not allowed.")
    if not isinstance(origin, Coordinates):
        raise TypeError("`origin` must be an instance of Coordinates.")
    if not isinstance(velocities, Velocities):
        raise TypeError("`velocities` must be an instance of Velocities.")
    if not isinstance(acceleration, Acceleration):
        raise TypeError("`acceleration` must be an instance of Acceleration.")

    # Precompute terms
    t_squared = t**2
    half_t_squared = 0.5 * t_squared

    # Use precomputed terms
    return (
        Coordinates(
            origin.x + velocities.vx * t + acceleration.ax * half_t_squared,
            origin.y + velocities.vy * t + acceleration.ay * half_t_squared,
            origin.z + velocities.vz * t + acceleration.az * half_t_squared,
        ),
        Velocities(
            velocities.vx + acceleration.ax * t,
            velocities.vy + acceleration.ay * t,
            velocities.vz + acceleration.az * t,
        ),
    )


def calculate_time_ballistic(
    dx: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    a: Union[float, npt.NDArray[np.float64]] = 0.0,
) -> npt.NDArray[np.float64]:
    """
    Calculate the time it takes for a ballistic trajectory to travel a distance `dx`.

    Args:
        dx (npt.NDArray[np.float64]): Distance to travel.
        v (npt.NDArray[np.float64]): Initial velocity.
        a (Union[float, npt.NDArray[np.float64]], optional): Acceleration. Defaults to 0.0.

    Returns:
        npt.NDArray[np.float64]: Time required to travel the distance `dx`.
                                    Returns NaN for invalid or negative times.
    """
    dx = cp.asarray(dx, dtype=cp.float64)
    v = cp.asarray(v, dtype=cp.float64)
    a = cp.asarray(a, dtype=cp.float64) if not isinstance(a, float) else a

    # Case 1: No acceleration
    if cp.all(cp.asarray(a == 0)):
        t = dx / v
        t[t < 0] = cp.nan  # Replace negative times with NaN
        return t

    # Case 2: With acceleration
    discriminant = 2 * a * dx + v**2
    valid_discriminant = discriminant >= 0

    # Initialize time array with NaN
    t = cp.full_like(dx, cp.nan, dtype=cp.float64)

    # Compute roots only for valid discriminant
    if isinstance(a, (float, int)):
        sqrt_discriminant = np.sqrt(discriminant[valid_discriminant])
        t1 = (sqrt_discriminant - v[valid_discriminant]) / a
        t2 = (-sqrt_discriminant - v[valid_discriminant]) / a
    else:
        sqrt_discriminant = np.sqrt(discriminant[valid_discriminant])
        t1 = (sqrt_discriminant - v[valid_discriminant]) / a[valid_discriminant]
        t2 = (-sqrt_discriminant - v[valid_discriminant]) / a[valid_discriminant]

    # Select positive roots
    positive_t1 = t1 > 0
    positive_t2 = t2 > 0

    t[valid_discriminant] = np.where(positive_t1, t1, np.where(positive_t2, t2, np.nan))
    return t


def propagate_ballistic_trajectories(
    t_start: npt.NDArray[np.float64],
    origin: Coordinates,
    velocities: Velocities,
    objects: List,
    z_stop: float,
    acceleration: Acceleration = Acceleration(0.0, -9.81, 0.0),
    z_save: Optional[Union[List[float], npt.NDArray[np.float64]]] = None,
    save_collisions: bool = False,
    options: PropagationOptions = PropagationOptions(),
) -> Tuple[
    npt.NDArray[np.bool_],
    npt.NDArray[np.float64],
    Coordinates,
    Velocities,
    int,
    List[Tuple[Coordinates, Velocities]],
]:
    """
    Propagate ballistic trajectories, storing timestamps, coordinates, and velocities.

    Args:
        t_start (npt.NDArray[np.float64]): Start times for each trajectory.
        origin (Coordinates): Initial coordinates of each particle.
        velocities (Velocities): Initial velocities of each particle.
        objects (List): List of objects (e.g., apertures) to interact with.
        z_stop (float): Z-coordinate at which to stop trajectory propagation.
        acceleration (Acceleration, optional): Acceleration acting on particles.
                                                Defaults to Acceleration(0.0, -9.81, 0.0).
        z_save (Optional[Union[List[float], npt.NDArray[np.float64]]], optional):
                Z-coordinates at which to save intermediate states. Defaults to None.
        save_collisions (bool, optional): Whether to save collision data. Defaults to False.
        options (PropagationOptions, optional): Options for trajectory propagation.
                                                Defaults to PropagationOptions().

    Returns:
        Tuple[...]: See docstring for details.
    """
    # Initialize variables for collisions, indices, and accepted trajectories
    collisions = []
    indices = cp.arange(origin.x.size)
    accepted_coords = deepcopy(origin)
    accepted_velocities = deepcopy(velocities)
    t_accepted = deepcopy(t_start)
    # Iterate through objects (e.g., apertures) to propagate trajectories
    for obj in objects:
        # Calculate the distance to the next object along the z-axis
        dz = obj.z_stop - accepted_coords.z

        if not cp.allclose(dz, 0):  # If there is a non-zero distance to propagate
            vz = accepted_velocities.vz  # Extract z-velocity
            az = acceleration.az  # Extract z-acceleration
            t = calculate_time_ballistic(
                dz, vz, az
            )  # Calculate time to reach the object

            # Propagate coordinates and velocities to the object's z-position
            x, v = propagate_ballistic(
                t, accepted_coords, accepted_velocities, acceleration
            )
        else:  # If already at the object's z-position
            t = deepcopy(t_accepted)
            x, v = deepcopy(accepted_coords), deepcopy(accepted_velocities)

        # Check if the trajectories are accepted by the object
        try:
            acceptance = obj.get_acceptance(
                accepted_coords, Coordinates(cp.asnumpy(x.x), cp.asnumpy(x.y), cp.asnumpy(x.z)), accepted_velocities, acceleration
            )
            acceptance = cp.asarray(acceptance, dtype=bool)  # Ensure acceptance is a boolean array
        except AssertionError as error:
            raise AssertionError(f"{obj} -> {error.args[0]}")

        # Exclude trajectories with invalid times
        acceptance &= ~cp.isnan(t)

        # Save collision data if required
        if save_collisions:
            if hasattr(obj, "get_collisions"):
                collisions.append(
                    obj.get_collisions(
                        accepted_coords, x, accepted_velocities, acceleration
                    )[1:]
                )
            else:
                collisions.append(
                    (x.get_masked(~acceptance), v.get_masked(~acceptance))
                )

        # Update accepted trajectories based on the object's acceptance criteria
        accepted_coords = accepted_coords.get_masked(acceptance)
        accepted_velocities = accepted_velocities.get_masked(acceptance)
        t_accepted = t_accepted[acceptance]
        indices = indices[acceptance]

    # Determine which trajectories survive
    survive = cp.zeros(origin.x.size, dtype=bool)
    survive[indices] = True
    nr_collisions = survive.size - survive.sum()

    # Save intermediate states at specified z-positions if provided
    if z_save is not None:
        for idz, z in enumerate(z_save):
            coords = accepted_coords.get_last()
            vels = accepted_velocities.get_last()
            dz = z - coords.z
            dt = calculate_time_ballistic(dz, vels.vz, acceleration.az)
            x, v = propagate_ballistic(dt, coords, vels, acceleration)
            if idz != 0:
                t_accepted = cp.column_stack([t_accepted, t_accepted[:, -1] + dt])
            else:
                t_accepted = cp.column_stack([t_accepted, t_accepted + dt])
            accepted_coords.column_stack(x)
            accepted_velocities.column_stack(v)

    # Propagate to the final z-stop position
    coords = accepted_coords.get_last()
    vels = accepted_velocities.get_last()
    dt = calculate_time_ballistic(z_stop - coords.z, vels.vz, acceleration.az)
    x, v = propagate_ballistic(dt, coords, vels, acceleration)

    # Save final timestamps, coordinates, and velocities
    if z_save is not None and len(z_save) > 0:
        t_accepted = cp.column_stack([t_accepted, t_accepted[:, -1] + dt])
    else:
        t_accepted = cp.column_stack([t_accepted, t_accepted + dt])
    accepted_coords.column_stack(x)
    accepted_velocities.column_stack(v)

    # Return results
    return (
        survive,
        t_accepted,
        accepted_coords,
        accepted_velocities,
        nr_collisions,
        collisions,
    )
