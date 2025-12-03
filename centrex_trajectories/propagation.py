"""
Trajectory Propagation Framework
=================================

This module provides the high-level interface for propagating molecular beam trajectories
through complex beamline configurations. It implements an adaptive multi-method approach
that switches between different propagation strategies based on the physics of each section.

Propagation Methods
-------------------
1. **Ballistic**: Constant acceleration (uniform fields, drift regions)
   - Uses analytical solutions: x(t) = x₀ + v₀t + ½at²
   - Efficient columnar array storage (all trajectories synchronized)
   - Collision detection via geometric intersection tests

2. **Linear (Harmonic)**: Linear restoring forces (ion guides, traps)
   - Uses analytical solutions: x(t) = A·sin(ωt) + B·cos(ωt)
   - Angular frequency ω = √(k/m) from spring constant
   - Efficient for transverse confinement with axial drift

3. **ODE**: Position-dependent forces (electrostatic lenses, gradients)
   - Uses adaptive step-size ODE integration (scipy.integrate.solve_ivp)
   - Switches to per-trajectory storage (different step counts)
   - Event detection for collisions with objects

Architecture
------------
The propagation engine uses a hybrid storage strategy for optimal performance:
- **2D Arrays**: Used for ballistic/linear sections (same steps for all trajectories)
- **Trajectories Dict**: Used during/after ODE sections (variable steps per trajectory)

Automatic conversion between storage formats occurs at section boundaries, allowing
efficient propagation through mixed beamline configurations.

Key Functions
-------------
- `propagate_trajectories`: Main entry point for trajectory propagation
- `do_ballistic`: Handle ballistic propagation through a section
- `do_linear`: Handle linear (harmonic) propagation through a section

Performance Notes
-----------------
- Columnar (2D array) storage is ~10x faster than per-trajectory storage for ballistic
- ODE integration is ~100x slower than ballistic, use sparingly
- Pre-filtering trajectories by aperture acceptance reduces unnecessary ODE integrations
- Numba-accelerated collision detection provides significant speedup

See Also
--------
- propagation_ballistic: Low-level ballistic propagation implementation
- propagation_linear: Low-level linear/harmonic propagation implementation
- propagation_ode: Low-level ODE integration implementation
- propagation_options: Configuration options for propagation behavior
"""

import copy
import math
from typing import List, Optional, Tuple, Union, cast

import numpy as np
import numpy.typing as npt

from .beamline_objects import Bore, LinearSection, ODESection, Section
from .common_types import ForceType
from .data_structures import (
    Acceleration,
    Coordinates,
    Force,
    SectionData,
    Trajectories,
    Trajectory,
    Velocities,
)
from .particles import Particle, TlF
from .propagation_ballistic import propagate_ballistic_trajectories
from .propagation_linear import propagate_linear_trajectories
from .propagation_ode import propagate_ODE_trajectories
from .propagation_ode_vectorized import ode_fun_batch, rk4_batch_fixed_steps
from .propagation_options import PropagationOptions, PropagationType

__all__: List[str] = ["PropagationType", "propagate_trajectories", "PropagationOptions"]


def do_ballistic(
    indices: npt.NDArray[np.int_],
    timestamps_tracked: npt.NDArray[np.float64],
    coordinates_tracked: Coordinates,
    velocities_tracked: Velocities,
    trajectories: Trajectories,
    section: Union[Section, ODESection],
    particle: Particle,
    force: Force,
    z_save_section: Optional[Union[List[float], npt.NDArray[np.float64]]],
    options: PropagationOptions,
) -> Tuple[
    npt.NDArray[np.float64],
    Coordinates,
    Velocities,
    npt.NDArray[np.int_],
    Trajectories,
    SectionData,
]:
    """
    Propagate trajectories ballistically through a section.

    This function handles ballistic (constant acceleration) propagation where particles
    move under the influence of constant forces. It's more performant than ODE-based
    propagation and is used for drift sections or sections with uniform fields.

    Args:
        indices (npt.NDArray[np.int_]):
            Array of trajectory indices for tracking which particles survive.
        timestamps_tracked (npt.NDArray[np.float64]):
            Array of tracked timestamps. Can be 1D (initial) or 2D (accumulated).
        coordinates_tracked (Coordinates):
            Current positions of all tracked trajectories.
        velocities_tracked (Velocities):
            Current velocities of all tracked trajectories.
        trajectories (Trajectories):
            Container for detailed trajectory data (used after ODE sections).
        section (Union[Section, ODESection]):
            The section through which to propagate, containing geometry and objects.
        particle (Particle):
            The particle species being propagated (contains mass, etc.).
        force (Force):
            External force applied during propagation (e.g., gravity).
        z_save_section (Optional[Union[List[float], npt.NDArray[np.float64]]]):
            Z positions within this section where intermediate states should be saved.
        options (PropagationOptions):
            Configuration options for propagation behavior.

    Returns:
        tuple:
            A tuple containing:
            - Updated timestamps_tracked (npt.NDArray[np.float64]): With new timestep appended.
            - Updated coordinates_tracked (Coordinates): With final positions appended.
            - Updated velocities_tracked (Velocities): With final velocities appended.
            - Updated indices (npt.NDArray[np.int_]): Only surviving trajectory indices.
            - Updated trajectories (Trajectories): Updated with new trajectory data.
            - SectionData: Statistics about collisions and survival in this section.
    """
    # Calculate total constant force and resulting acceleration
    # Section force is added to external force (e.g., gravity + static fields)
    force_cst = force + section.force
    acceleration = Acceleration(
        force_cst.fx / particle.mass,
        force_cst.fy / particle.mass,
        force_cst.fz / particle.mass,
    )
    # Propagate trajectories ballistically through the section
    # Extract last timestamp (handling both 1D initial and 2D accumulated arrays)
    (
        mask,
        timestamp_list,
        coord_list,
        velocities_list,
        nr_collisions,
        collisions,
    ) = propagate_ballistic_trajectories(
        timestamps_tracked
        if timestamps_tracked.ndim == 1
        else timestamps_tracked[:, -1],
        coordinates_tracked.get_last(),
        velocities_tracked.get_last(),
        section.objects,
        section.stop,
        acceleration,
        z_save=z_save_section,
        save_collisions=section.save_collisions,
        options=options,
    )

    # Filter out trajectories that collided with objects
    # mask is True for trajectories that survived
    timestamps_tracked = timestamps_tracked[mask]
    coordinates_tracked = coordinates_tracked.get_masked(mask)
    velocities_tracked = velocities_tracked.get_masked(mask)
    indices = indices[mask]

    # Append final state to tracked arrays (columnar storage for efficiency)
    timestamps_tracked = np.column_stack([timestamps_tracked, timestamp_list])
    coordinates_tracked.column_stack(coord_list)
    velocities_tracked.column_stack(velocities_list)

    # Update trajectories container if it's being used (after first ODE section)
    if len(trajectories) != 0:
        # Remove trajectories that collided
        remove = [k for k in trajectories.keys() if k not in indices]
        trajectories.delete_trajectories(remove)

        # Add latest data for surviving trajectories
        for index, t, c, v in zip(indices, timestamp_list, coord_list, velocities_list):
            trajectories.add_data(index, t, cast(Coordinates, c), cast(Velocities, v))

    # Create section statistics
    section_data = SectionData(section.name, collisions, nr_collisions, len(mask))

    return (
        timestamps_tracked,
        coordinates_tracked,
        velocities_tracked,
        indices,
        trajectories,
        section_data,
    )


def do_linear(
    indices: npt.NDArray[np.int_],
    timestamps_tracked: npt.NDArray[np.float64],
    coordinates_tracked: Coordinates,
    velocities_tracked: Velocities,
    trajectories: Trajectories,
    section: LinearSection,
    particle: Particle,
    force: Force,
    z_save_section: Optional[Union[List[float], npt.NDArray[np.float64]]],
    options: PropagationOptions,
) -> Tuple[
    npt.NDArray[np.float64],
    Coordinates,
    Velocities,
    npt.NDArray[np.int_],
    Trajectories,
    SectionData,
]:
    """
    Propagate trajectories through a linear (harmonic) section.

    This function handles propagation through sections with linear restoring forces,
    such as harmonic traps or ion guides. The motion is analytically solvable with
    sinusoidal trajectories in the transverse directions and ballistic in z.

    Args:
        indices (npt.NDArray[np.int_]):
            Array of trajectory indices for tracking which particles survive.
        timestamps_tracked (npt.NDArray[np.float64]):
            Array of tracked timestamps. Can be 1D (initial) or 2D (accumulated).
        coordinates_tracked (Coordinates):
            Current positions of all tracked trajectories.
        velocities_tracked (Velocities):
            Current velocities of all tracked trajectories.
        trajectories (Trajectories):
            Container for detailed trajectory data (used after ODE sections).
        section (LinearSection):
            The linear section with spring constants defining the restoring forces.
        particle (Particle):
            The particle species being propagated (contains mass, etc.).
        force (Force):
            External force applied during propagation (e.g., gravity).
        z_save_section (Optional[Union[List[float], npt.NDArray[np.float64]]]):
            Z positions within this section where intermediate states should be saved.
        options (PropagationOptions):
            Configuration options for propagation behavior.

    Returns:
        tuple:
            A tuple containing:
            - Updated timestamps_tracked (npt.NDArray[np.float64]): With new timestep appended.
            - Updated coordinates_tracked (Coordinates): With final positions appended.
            - Updated velocities_tracked (Velocities): With final velocities appended.
            - Updated indices (npt.NDArray[np.int_]): Only surviving trajectory indices.
            - Updated trajectories (Trajectories): Updated with new trajectory data.
            - SectionData: Statistics about collisions and survival in this section.
    """
    # Calculate total constant force and acceleration (non-harmonic component)
    force_cst = force + section.force
    acceleration = Acceleration(
        force_cst.fx / particle.mass,
        force_cst.fy / particle.mass,
        force_cst.fz / particle.mass,
    )

    # Calculate angular frequencies for harmonic motion in each direction
    # ω = √(k/m) where k is spring constant and m is particle mass
    w = (
        math.sqrt(section.spring_constant[0] / particle.mass),
        math.sqrt(section.spring_constant[1] / particle.mass),
        math.sqrt(section.spring_constant[2] / particle.mass),
    )
    # Propagate trajectories through linear (harmonic) section
    # Extract last timestamp (handling both 1D initial and 2D accumulated arrays)
    (
        mask,
        timestamp_list,
        coord_list,
        velocities_list,
        nr_collisions,
        collisions,
    ) = propagate_linear_trajectories(
        t_start=timestamps_tracked
        if timestamps_tracked.ndim == 1
        else timestamps_tracked[:, -1],
        origin=coordinates_tracked.get_last(),
        velocities=velocities_tracked.get_last(),
        objects=section.objects,
        z_stop=section.stop,
        acceleration=acceleration,
        w=w,
        trap_center=(section.x, section.y),
        z_save=z_save_section,
        save_collisions=section.save_collisions,
        options=options,
    )

    # Filter out trajectories that collided with objects
    timestamps_tracked = timestamps_tracked[mask]
    coordinates_tracked = coordinates_tracked.get_masked(mask)
    velocities_tracked = velocities_tracked.get_masked(mask)
    indices = indices[mask]

    # Append final state to tracked arrays
    timestamps_tracked = np.column_stack([timestamps_tracked, timestamp_list])
    coordinates_tracked.column_stack(coord_list)
    velocities_tracked.column_stack(velocities_list)

    # Update trajectories container if it's being used
    if len(trajectories) != 0:
        # Remove trajectories that collided
        remove = [k for k in trajectories.keys() if k not in indices]
        trajectories.delete_trajectories(remove)

        # Add latest data for surviving trajectories
        for index, t, c, v in zip(indices, timestamp_list, coord_list, velocities_list):
            trajectories.add_data(index, t, cast(Coordinates, c), cast(Velocities, v))

    # Create section statistics
    section_data = SectionData(section.name, collisions, nr_collisions, len(mask))

    return (
        timestamps_tracked,
        coordinates_tracked,
        velocities_tracked,
        indices,
        trajectories,
        section_data,
    )


def propagate_trajectories(
    sections: List[Union[Section, ODESection, LinearSection]],
    coordinates_init: Coordinates,
    velocities_init: Velocities,
    particle: Particle,
    t_start: Optional[npt.NDArray[np.float64]] = None,
    force: Force = Force(0.0, -9.81 * TlF().mass, 0.0),
    z_save: Optional[List[float]] = None,
    options: PropagationOptions = PropagationOptions(),
) -> Tuple[List[SectionData], Trajectories]:
    """
    Propagate trajectories through a series of sections starting from initial
    coordinates and velocities.

    Args:
        sections (List[Union[Section, ODESection, LinearSection]]):
            List of sections to propagate through.
        coordinates_init (Coordinates):
            Initial positions of the particles.
        velocities_init (Velocities):
            Initial velocities of the particles.
        particle (Particle):
            The particle to propagate.
        t_start (Optional[npt.NDArray[np.float64]], optional):
            Initial timestamps. Defaults to None.
        force (Force, optional):
            External force applied during propagation. Defaults to gravity
            (Force(0.0, -9.81 * TlF().mass, 0.0)).
        z_save (Optional[List[float]], optional):
            Z positions at which to save timestamps, coordinates, and velocities.
            Defaults to None.
        options (PropagationOptions, optional):
            Options for propagation. Defaults to PropagationOptions().

    Returns:
        Tuple[List[SectionData], Trajectories]:
            A tuple containing:
            - A list of SectionData objects with data for each section.
            - The surviving trajectories as a Trajectories object.
    """
    # Initialize tracking arrays
    # Indices track which trajectories survive (used for filtering after collisions)
    indices = np.arange(len(coordinates_init))

    # Initialize state arrays for all trajectories
    # These use efficient columnar storage for ballistic/linear propagation
    timestamps_tracked = (
        t_start.copy() if t_start is not None else np.zeros(len(indices))
    )
    coordinates_tracked = copy.deepcopy(coordinates_init)
    velocities_tracked = copy.deepcopy(velocities_init)

    # Container for per-section statistics (collisions, survival rates, etc.)
    section_data = []

    # Container for detailed trajectory data (initially empty, populated after ODE sections)
    # Trajectories store variable-length data more efficiently than 2D arrays
    trajectories = Trajectories()

    # Main propagation loop: process each section sequentially
    for section in sections:
        # Determine which z_save positions fall within this section
        if z_save is not None:
            z_save_section: Optional[List[float]] = [
                zs for zs in z_save if zs >= section.start and zs <= section.stop
            ]
        else:
            z_save_section = None

        # PERFORMANCE NOTE: Storage strategy optimization
        # ================================================
        # Ballistic/linear propagation uses 2D arrays (columnar storage) because all
        # trajectories take the same number of integration steps. This is fast and
        # memory-efficient.
        #
        # ODE propagation uses adaptive step sizes, so trajectories have different
        # numbers of time points. We switch to Trajectories (dict of variable-length
        # arrays) for ODE sections.
        #
        # After ODE sections, we convert back to 2D arrays starting from the ODE
        # endpoint, allowing efficient ballistic propagation to resume.

        # Route to appropriate propagation method based on section type
        if section.propagation_type == PropagationType.ballistic:
            (
                timestamps_tracked,
                coordinates_tracked,
                velocities_tracked,
                indices,
                trajectories,
                sec_dat,
            ) = do_ballistic(
                indices=indices,
                timestamps_tracked=timestamps_tracked,
                coordinates_tracked=coordinates_tracked,
                velocities_tracked=velocities_tracked,
                trajectories=trajectories,
                section=section,
                particle=particle,
                force=force,
                z_save_section=z_save_section,
                options=options,
            )
            section_data.append(sec_dat)
        # ODE propagation for sections with position-dependent forces
        elif section.propagation_type == PropagationType.ode:
            # Check if particles need to be propagated ballistically to reach ODE section start
            # This can happen if there's a gap between sections or after initialization
            if np.any(
                coordinates_tracked.get_last().z < section.start
            ) and not np.allclose(coordinates_tracked.get_last().z, section.start):
                # Bridge the gap with ballistic propagation
                (
                    timestamps_tracked,
                    coordinates_tracked,
                    velocities_tracked,
                    indices,
                    trajectories,
                    sec_dat,
                ) = do_ballistic(
                    indices=indices,
                    timestamps_tracked=timestamps_tracked,
                    coordinates_tracked=coordinates_tracked,
                    velocities_tracked=velocities_tracked,
                    trajectories=trajectories,
                    section=Section(
                        name="_",
                        objects=[],
                        start=coordinates_tracked.get_last().z[
                            0
                        ],  # just give it one start coord, start is not used here
                        stop=section.start,
                        save_collisions=False,
                    ),
                    particle=particle,
                    force=force,
                    z_save_section=z_save_section,
                    options=options,
                )

            # Convert 2D arrays to Trajectories container if not already done
            # (this happens before the first ODE section)
            if len(trajectories) == 0:
                for index, t, c, v in zip(
                    indices, timestamps_tracked, coordinates_tracked, velocities_tracked
                ):
                    trajectories.add_data(
                        index, t, cast(Coordinates, c), cast(Velocities, v)
                    )

            # Set up force function for ODE integration
            if isinstance(section, ODESection):
                # Use optimized scalar force function if available (e.g., from electrostatic lens)
                force_fun = cast(
                    ForceType, getattr(section, "force_fast_scalar", section.force)
                )  # type: ignore[assignment]
                force_cst = force
            else:
                # No section-specific force, only external force
                def force_fun(
                    t: float,
                    x: float,
                    y: float,
                    z: float,
                ) -> Tuple[float, float, float]:
                    return (0.0, 0.0, 0.0)

                force_cst = force + section.force

            nr_trajectories = len(trajectories)

            # PRE-FILTER: Check transverse acceptance before ODE integration
            # ============================================================
            # For performance, check if trajectories are within object apertures using
            # fast ballistic calculation. This only works if objects start exactly at
            # the ODE section start.
            #
            # TODO: Implement additional ODE stop event for out-of-aperture trajectories
            # during integration (would catch trajectories that start inside but drift out)
            acceleration = Acceleration(
                force_cst.fx / particle.mass,
                force_cst.fy / particle.mass,
                force_cst.fz / particle.mass,
            )
            masks = [
                obj.get_acceptance(
                    coordinates_tracked.get_last(),
                    coordinates_tracked.get_last(),
                    velocities_tracked.get_last(),
                    acceleration,
                )
                for obj in section.objects
            ]

            nr_collisions = 0
            collisions = []

            # Apply acceptance filters from all objects in this section
            if len(masks) > 0:
                # Combine all acceptance masks (trajectory must pass all apertures)
                mask = np.bitwise_and.reduce(masks)

                # Count and optionally save collision data
                nr_collisions += (~mask).sum()
                if section.save_collisions:
                    collisions.append(
                        (
                            coordinates_tracked.get_last()[~mask],
                            velocities_tracked.get_last()[~mask],
                        )
                    )

                # Filter out trajectories that failed acceptance check
                timestamps_tracked = timestamps_tracked[mask]
                coordinates_tracked = coordinates_tracked.get_masked(mask)
                velocities_tracked = velocities_tracked.get_masked(mask)
                indices = indices[mask]

                # Remove from trajectories container
                if len(trajectories) != 0:
                    remove = [k for k in trajectories.keys() if k not in indices]
                    trajectories.delete_trajectories(remove)

            # Integrate equations of motion using ODE solver
            # This handles position-dependent forces (e.g., electrostatic lenses)
            solutions = propagate_ODE_trajectories(
                t_start=timestamps_tracked[:, -1]
                if timestamps_tracked.ndim > 1
                else timestamps_tracked[-1],
                origin=coordinates_tracked.get_last(),
                velocities=velocities_tracked.get_last(),
                z_stop=section.stop,
                mass=particle.mass,
                force_fun=force_fun,
                force_cst=force_cst,
                events=[obj.collision_event_function for obj in section.objects],
                options=options,
            )

            # Extract final states from ODE solutions and update tracking structures
            timestamps = []
            coords = []
            velocities = []
            for sol, index in zip(solutions, indices):
                # Get final time and state from each solution
                timestamps.append(sol.t[-1])
                # Store full ODE solution in trajectories container
                trajectories.add_data_ode(index, sol)
                # Extract final position and velocity
                coords.append([sol.y[0, -1], sol.y[1, -1], sol.y[2, -1]])
                velocities.append([sol.y[3, -1], sol.y[4, -1], sol.y[5, -1]])

            # Update tracked arrays with ODE endpoint data
            # Converting back to 2D arrays for efficient subsequent ballistic propagation
            timestamps_tracked = np.column_stack([timestamps_tracked, timestamps])
            coordinates_tracked.column_stack(Coordinates(*np.array(coords).T))
            velocities_tracked.column_stack(Velocities(*np.array(velocities).T))

            # POST-FILTER: Check for trajectories that terminated early (collisions during ODE)
            # ODE solver stops integration when collision events are triggered
            mask = np.ones(len(solutions), dtype=bool)
            for idx, (sol, index) in enumerate(zip(solutions, indices)):
                # If final z position doesn't match section stop, trajectory hit an object
                if not math.isclose(sol.y[2, -1], section.stop):
                    mask[idx] = False

            # Handle trajectories that collided during ODE integration
            if (~mask).sum() > 0:
                nr_collisions += (~mask).sum()
                if section.save_collisions:
                    collisions.append(
                        (
                            coordinates_tracked.get_last()[~mask],  # type: ignore[index]
                            velocities_tracked.get_last()[~mask],  # type: ignore[index]
                        )
                    )

                # Remove collided trajectories from tracking
                coordinates_tracked = coordinates_tracked.get_masked(mask)
                velocities_tracked = velocities_tracked.get_masked(mask)
                indices = indices[mask]
                timestamps_tracked = timestamps_tracked[mask]

                # Remove from trajectories container
                if len(trajectories) != 0:
                    remove = [k for k in trajectories.keys() if k not in indices]
                    trajectories.delete_trajectories(remove)

            # Record section statistics
            section_data.append(
                SectionData(section.name, collisions, nr_collisions, nr_trajectories)
            )

        # Vectorized ODE propagation for sections with position and time dependent forces
        elif section.propagation_type == PropagationType.ode_vectorized:
            # Check if particles need to be propagated ballistically to reach ODE section start
            # This can happen if there's a gap between sections or after initialization
            if np.any(
                coordinates_tracked.get_last().z < section.start
            ) and not np.allclose(coordinates_tracked.get_last().z, section.start):
                # Bridge the gap with ballistic propagation
                (
                    timestamps_tracked,
                    coordinates_tracked,
                    velocities_tracked,
                    indices,
                    trajectories,
                    sec_dat,
                ) = do_ballistic(
                    indices=indices,
                    timestamps_tracked=timestamps_tracked,
                    coordinates_tracked=coordinates_tracked,
                    velocities_tracked=velocities_tracked,
                    trajectories=trajectories,
                    section=Section(
                        name="_",
                        objects=[],
                        start=coordinates_tracked.get_last().z[
                            0
                        ],  # just give it one start coord, start is not used here
                        stop=section.start,
                        save_collisions=False,
                    ),
                    particle=particle,
                    force=force,
                    z_save_section=z_save_section,
                    options=options,
                )
            coord_last = coordinates_tracked.get_last()
            vel_last = velocities_tracked.get_last()
            tstart = (
                timestamps_tracked[:, -1]
                if timestamps_tracked.ndim > 1
                else timestamps_tracked[-1]
            )
            if options.section_options.get(section.name) is None:
                raise ValueError("Options for vectorized ODE solver required")
            t, d, survived = rk4_batch_fixed_steps(
                f=ode_fun_batch,
                t0=tstart,
                z_stop=section.stop,
                d0=np.column_stack(
                    (
                        coord_last.x,
                        coord_last.y,
                        coord_last.z,
                        vel_last.vx,
                        vel_last.vy,
                        vel_last.vz,
                    )
                ),
                n_steps=options.section_options[section.name].n_steps,
                n_save=options.section_options[section.name].n_save,
                mass=particle.mass,
                force_fn=section.force,
                force_cst=force,
                check_bounds_functions=[
                    obj.check_in_bounds_vectorized for obj in section.objects
                ],
            )

            timestamps_tracked = timestamps_tracked[survived]
            coordinates_tracked = coordinates_tracked.get_masked(survived)
            velocities_tracked = velocities_tracked.get_masked(survived)
            indices = indices[survived]

            for idx in range(t.shape[0]):
                timestamps_tracked = np.column_stack(
                    [timestamps_tracked, t[idx, survived]]
                )
                coordinates_tracked.column_stack(
                    Coordinates(
                        d[idx, survived, 0], d[idx, survived, 1], z=d[idx, survived, 2]
                    )
                )
                velocities_tracked.column_stack(
                    Velocities(
                        d[idx, survived, 3], d[idx, survived, 4], d[idx, survived, 5]
                    )
                )

            # Update trajectories container if it's being used
            if len(trajectories) != 0:
                raise NotImplementedError(
                    "Vectorized ODE propagation not implemented for existing Trajectories"
                )

            # Create section statistics
            section_data = SectionData(section.name, [], ~survived.sum(), len(t))

        # Linear (harmonic) propagation for sections with linear restoring forces
        elif section.propagation_type == PropagationType.linear:
            assert type(section) is LinearSection
            (
                timestamps_tracked,
                coordinates_tracked,
                velocities_tracked,
                indices,
                trajectories,
                sec_dat,
            ) = do_linear(
                indices=indices,
                timestamps_tracked=timestamps_tracked,
                coordinates_tracked=coordinates_tracked,
                velocities_tracked=velocities_tracked,
                trajectories=trajectories,
                section=section,
                particle=particle,
                force=force,
                z_save_section=z_save_section,
                options=options,
            )
            section_data.append(sec_dat)

    # Final conversion: If we never hit an ODE section, convert 2D arrays to Trajectories
    # This ensures consistent output format regardless of propagation method
    if len(trajectories) == 0:
        for index, t, c, v in zip(
            indices, timestamps_tracked, coordinates_tracked, velocities_tracked
        ):
            trajectories[index] = Trajectory(
                t, cast(Coordinates, c), cast(Velocities, v), index
            )

    # Clean up: Remove duplicate time entries that can occur at section boundaries
    for trajectory in trajectories.values():
        trajectory.remove_duplicate_entries()

    return section_data, trajectories
