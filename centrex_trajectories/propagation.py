import copy
import math
from typing import List, Optional, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
import cupy as cp
import gc
from .beamline_objects import LinearSection, ODESection, Section
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

    Args:
        indices (npt.NDArray[np.int_]):
            Array of trajectory indices.
        timestamps_tracked (npt.NDArray[np.float64]):
            Array of tracked timestamps.
        coordinates_tracked (Coordinates):
            Tracked coordinates of the trajectories.
        velocities_tracked (Velocities):
            Tracked velocities of the trajectories.
        trajectories (Trajectories):
            Object holding trajectory data.
        section (Union[Section, ODESection]):
            The section through which to propagate.
        particle (Particle):
            The particle being propagated.
        force (Force):
            External force applied during propagation.
        z_save_section (Optional[Union[List[float], npt.NDArray[np.float64]]]):
            Z positions to save data for this section.
        options (PropagationOptions):
            Options for propagation.

    Returns:
        tuple:
            A tuple containing:
            - Updated timestamps_tracked (npt.NDArray[np.float64]).
            - Updated coordinates_tracked (Coordinates).
            - Updated velocities_tracked (Velocities).
            - Updated indices (npt.NDArray[np.int_]).
            - Updated trajectories (Trajectories).
            - SectionData for the section.
    """
    force_cst = force + section.force
    acceleration = Acceleration(
        force_cst.fx / particle.mass,
        force_cst.fy / particle.mass,
        force_cst.fz / particle.mass,
    )
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

    # only keep trajectories that made it through
    timestamps_tracked = timestamps_tracked[mask]
    coordinates_tracked = coordinates_tracked.get_masked(mask)
    velocities_tracked = velocities_tracked.get_masked(mask)
    indices = indices[mask]

    # append latest timestamps, coordinates and velocities to the 2D arrays
    timestamps_tracked = cp.column_stack([timestamps_tracked, timestamp_list])
    coordinates_tracked.column_stack(coord_list)
    velocities_tracked.column_stack(velocities_list)

    # remove trajectories that didn't make it through
    if len(trajectories) != 0:
        if isinstance(indices, cp.ndarray):
            indices = cp.asnumpy(indices)
            timestamp_list = cp.asnumpy(timestamp_list)
            coord_list = Coordinates(cp.asnumpy(coord_list.x), cp.asnumpy(coord_list.y), cp.asnumpy(coord_list.z))
            velocities_list = Velocities(cp.asnumpy(velocities_list.vx), cp.asnumpy(velocities_list.vy), cp.asnumpy(velocities_list.vz))
        remove = [k for k in trajectories.keys() if k not in indices]
        trajectories.delete_trajectories(remove)

        # update trajectories that did make it through
        for index, t, c, v in zip(indices, timestamp_list, coord_list, velocities_list):
            index = int(index)
            trajectories.add_data(index, t, cast(Coordinates, c), cast(Velocities, v))
    if not isinstance(indices, cp.ndarray):
        indices = cp.array(indices)
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
    Propagate trajectories through a linear section.

    Args:
        indices (npt.NDArray[np.int_]):
            Array of trajectory indices.
        timestamps_tracked (npt.NDArray[np.float64]):
            Array of tracked timestamps.
        coordinates_tracked (Coordinates):
            Tracked coordinates of the trajectories.
        velocities_tracked (Velocities):
            Tracked velocities of the trajectories.
        trajectories (Trajectories):
            Object holding trajectory data.
        section (LinearSection):
            The linear section through which to propagate.
        particle (Particle):
            The particle being propagated.
        force (Force):
            External force applied during propagation.
        z_save_section (Optional[Union[List[float], npt.NDArray[np.float64]]]):
            Z positions to save data for this section.
        options (PropagationOptions):
            Options for propagation.

    Returns:
        tuple:
            A tuple containing:
            - Updated timestamps_tracked (npt.NDArray[np.float64]).
            - Updated coordinates_tracked (Coordinates).
            - Updated velocities_tracked (Velocities).
            - Updated indices (npt.NDArray[np.int_]).
            - Updated trajectories (Trajectories).
            - SectionData for the section.
    """
    force_cst = force + section.force
    acceleration = Acceleration(
        force_cst.fx / particle.mass,
        force_cst.fy / particle.mass,
        force_cst.fz / particle.mass,
    )
    w = (
        math.sqrt(section.spring_constant[0] / particle.mass),
        math.sqrt(section.spring_constant[1] / particle.mass),
        math.sqrt(section.spring_constant[2] / particle.mass),
    )
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

    # only keep trajectories that made it through
    timestamps_tracked = timestamps_tracked[mask]
    coordinates_tracked = coordinates_tracked.get_masked(mask)
    velocities_tracked = velocities_tracked.get_masked(mask)
    indices = indices[mask]

    # append latest timestamps, coordinates and velocities to the 2D arrays
    timestamps_tracked = cp.column_stack([timestamps_tracked, timestamp_list])
    coordinates_tracked.column_stack(coord_list)
    velocities_tracked.column_stack(velocities_list)

    # remove trajectories that didn't make it through
    if len(trajectories) != 0:
        remove = [k for k in trajectories.keys() if k not in indices]
        trajectories.delete_trajectories(remove)

        # update trajectories that did make it through
        for index, t, c, v in zip(indices, timestamp_list, coord_list, velocities_list):
            trajectories.add_data(index, t, cast(Coordinates, c), cast(Velocities, v))

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
    # initialize index array to keeps track of trajectory indices that make it through
    indices = cp.arange(len(coordinates_init))

    # initialize 2D arrays for keeping track of the ballistic coordinates
    timestamps_tracked = (
        cp.asarray(t_start) if t_start is not None else cp.zeros(len(indices))
    )
    coordinates_tracked = Coordinates(
        cp.asarray(coordinates_init.x),
        cp.asarray(coordinates_init.y),
        cp.asarray(coordinates_init.z),
    )
    velocities_tracked = Velocities(
        cp.asarray(velocities_init.vx),
        cp.asarray(velocities_init.vy),
        cp.asarray(velocities_init.vz),
    )

    # list to store SectionData for each section
    section_data = []

    # class to hold trajectories
    trajectories = Trajectories()

    # propagate through sections
    for section in sections:
        if z_save is not None:
            # select z positions from z_save that are within the section
            z_save_section: Optional[List[float]] = [
                zs for zs in z_save if zs >= section.start and zs <= section.stop
            ]
        else:
            z_save_section = None
        # Initially when trajectories are propagated ballistically they are stored in 2D
        # arrays, because particles take the same number of steps when propagating
        # ballistically. This is no longer true when propagating with an ODE solver, and
        # then storage is switched to Trajectories containing a single Trajectory for
        # each trajectory.
        # For performance this is only done after the first ODE section since the 2D
        # array storage and propagation method is much more performant.
        # After the ODE section the coordinates and velocities are transformed into
        # 2D arrays again, starting and the end of the ODE section. This allows for
        # use of the performant ballistic propagation method again after the ODE section

        # propagate ballistic if section is ballistic
        if section.propagation_type == PropagationType.ballistic:
            if not isinstance(timestamps_tracked, cp.ndarray):
            # initialize 2D arrays for keeping track of the ballistic coordinates
                timestamps_tracked = (
                    cp.asarray(t_start) if t_start is not None else cp.zeros(len(indices))
                )
                coordinates_tracked = Coordinates(
                    cp.asarray(coordinates_tracked.x),
                    cp.asarray(coordinates_tracked.y),
                    cp.asarray(coordinates_tracked.z),
                )
                velocities_tracked = Velocities(
                    cp.asarray(velocities_tracked.vx),
                    cp.asarray(velocities_tracked.vy),
                    cp.asarray(velocities_tracked.vz),
                )
                indices = cp.asarray(indices)
                
            
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
        # propagate ODE if section is ODE
        elif section.propagation_type == PropagationType.ode:
            if np.any(
                coordinates_tracked.get_last().z < section.start
            ) and not np.allclose(coordinates_tracked.get_last().z, section.start):
                if not isinstance(timestamps_tracked, cp.ndarray):
                    # initialize 2D arrays for keeping track of the ballistic coordinates
                    timestamps_tracked = (
                        cp.asarray(t_start) if t_start is not None else cp.zeros(len(indices))
                    )
                    coordinates_tracked = Coordinates(
                        cp.asarray(coordinates_tracked.x),
                        cp.asarray(coordinates_tracked.y),
                        cp.asarray(coordinates_tracked.z),
                    )
                    velocities_tracked = Velocities(
                        cp.asarray(velocities_tracked.vx),
                        cp.asarray(velocities_tracked.vy),
                        cp.asarray(velocities_tracked.vz),
                    )
                    indices = cp.asarray(indices)
                    # do ballistic until ode section
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
            coordinates_tracked = Coordinates(
                cp.asnumpy(coordinates_tracked.x),
                cp.asnumpy(coordinates_tracked.y),
                cp.asnumpy(coordinates_tracked.z),
            )
            velocities_tracked = Velocities(
                cp.asnumpy(velocities_tracked.vx),
                cp.asnumpy(velocities_tracked.vy),
                cp.asnumpy(velocities_tracked.vz),
            )
            timestamps_tracked = cp.asnumpy(timestamps_tracked)
            indices = cp.asnumpy(indices)

            if len(trajectories) == 0:
                for index, t, c, v in zip(
                    indices, timestamps_tracked, coordinates_tracked, velocities_tracked
                ):
                    trajectories.add_data(
                        index, t, cast(Coordinates, c), cast(Velocities, v)
                    )
            if isinstance(section, ODESection):
                force_fun = cast(
                    ForceType, getattr(section, "force_fast_scalar", section.force)
                )
                force_cst = force
            else:

                def force_fun(
                    t: float,
                    x: float,
                    y: float,
                    z: float,
                ) -> Tuple[float, float, float]:
                    return (0.0, 0.0, 0.0)

                force_cst = force + section.force

            nr_trajectories = len(trajectories)

            # Checking if any trajectories are outside the acceptance of the ODESection
            # object before starting the ODE trajectory solver.
            # Only works if objects start at the same exact z coordinate as the
            # ODEsection, since this uses ballistic trajectories to get the acceptance.
            # Should change to using and additional stop event that if the trajectory is
            # outside a certain range within the object z range it will stop the ODE
            # solver
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

            if len(masks) > 0:
                mask = np.bitwise_and.reduce(masks)

                nr_collisions += (~mask).sum()

                if section.save_collisions:
                    collisions.append(
                        (
                            coordinates_tracked.get_last()[~mask],
                            velocities_tracked.get_last()[~mask],
                        )
                    )

                timestamps_tracked = timestamps_tracked[mask]
                coordinates_tracked = coordinates_tracked.get_masked(mask)
                velocities_tracked = velocities_tracked.get_masked(mask)
                indices = indices[mask]

                if len(trajectories) != 0:
                    remove = [k for k in trajectories.keys() if k not in indices]
                    trajectories.delete_trajectories(remove)

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

            timestamps = []
            coords = []
            velocities = []
            for sol, index in zip(solutions, indices):
                timestamps.append(sol.t[-1])
                trajectories.add_data_ode(index, sol)
                coords.append([sol.y[0, -1], sol.y[1, -1], sol.y[2, -1]])
                velocities.append([sol.y[3, -1], sol.y[4, -1], sol.y[5, -1]])

            timestamps_tracked = np.column_stack([timestamps_tracked, timestamps])

            coordinates_tracked.column_stack(Coordinates(*np.array(coords).T))
            velocities_tracked.column_stack(Velocities(*np.array(velocities).T))
            # check for trajectories that didn't make it, e.g. hit objects during the
            # ode solver and terminated early
            mask = np.ones(len(solutions), dtype=bool)
            for idx, (sol, index) in enumerate(zip(solutions, indices)):
                if not math.isclose(sol.y[2, -1], section.stop):
                    mask[idx] = False

            if (~mask).sum() > 0:
                nr_collisions += (~mask).sum()
                if section.save_collisions:
                    collisions.append(
                        (
                            coordinates_tracked.get_last()[~mask],
                            velocities_tracked.get_last()[~mask],
                        )
                    )

                coordinates_tracked = coordinates_tracked.get_masked(mask)
                velocities_tracked = velocities_tracked.get_masked(mask)
                indices = indices[mask]
                timestamps_tracked = timestamps_tracked[mask]

                if len(trajectories) != 0:
                    remove = [k for k in trajectories.keys() if k not in indices]
                    trajectories.delete_trajectories(remove)

            section_data.append(
                SectionData(section.name, collisions, nr_collisions, nr_trajectories)
            )

        # propagate linear if section is linear
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
        gc.collect()  # Collect garbage to free memory after each section
    coordinates_tracked = Coordinates(
        cp.asnumpy(coordinates_tracked.x),
        cp.asnumpy(coordinates_tracked.y),
        cp.asnumpy(coordinates_tracked.z),
    )
    velocities_tracked = Velocities(
        cp.asnumpy(velocities_tracked.vx),
        cp.asnumpy(velocities_tracked.vy),
        cp.asnumpy(velocities_tracked.vz),
    )
    timestamps_tracked = cp.asnumpy(timestamps_tracked)
    indices = cp.asnumpy(indices)
    if len(trajectories) == 0:
        for index, t, c, v in zip(
            indices, timestamps_tracked, coordinates_tracked, velocities_tracked
        ):
            trajectories[index] = Trajectory(
                t, cast(Coordinates, c), cast(Velocities, v), index
            )

    # remove coordinate entries in a trajectory
    for trajectory in trajectories.values():
        trajectory.remove_duplicate_entries()
    return section_data, trajectories
