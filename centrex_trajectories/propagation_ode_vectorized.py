from typing import Callable, Sequence

import numpy as np
import numpy.typing as npt

from .data_structures import Force

Array = npt.NDArray[np.float64]


def ode_fun_batch(
    t: Array,  # (n_traj,)
    d: Array,  # (n_traj, 6) = [x, y, z, vx, vy, vz]
    mass: float,
    force_fn: Callable[
        [Array, Array, Array, Array],  # (t, x, y, z)
        tuple[Array, Array, Array],  # (fx, fy, fz), each (n_traj,)
    ],
    force_cst: Force,
) -> Array:
    """
    Batched RHS of the trajectory propagation ODE system.

    t : (n_traj,)
    d : (n_traj, 6) with columns [x, y, z, vx, vy, vz]
    Returns: (n_traj, 6) with derivatives in same order.
    """
    # unpack along columns
    x = d[:, 0]
    y = d[:, 1]
    z = d[:, 2]
    vx = d[:, 3]
    vy = d[:, 4]
    vz = d[:, 5]

    # position-dependent force for all trajectories
    fx, fy, fz = force_fn(t, x, y, z)  # each (n_traj,)

    # accelerations from position-dependent force
    ax = fx / mass
    ay = fy / mass
    az = fz / mass

    inv_m = 1.0 / mass
    # allocate derivative array
    dd = np.empty_like(d)

    # positions evolve with velocities
    dd[:, 0] = vx  # dx/dt
    dd[:, 1] = vy  # dy/dt
    dd[:, 2] = vz  # dz/dt

    # velocities evolve with total acceleration (pos-dependent + constant)
    dd[:, 3] = ax + force_cst.fx * inv_m
    dd[:, 4] = ay + force_cst.fy * inv_m
    dd[:, 5] = az + force_cst.fz * inv_m

    return dd


def rk4_batch_fixed_steps(
    f: Callable[
        [
            Array,
            Array,
            float,
            Callable[[Array, Array, Array, Array], tuple[Array, Array, Array]],
            "Force",
        ],
        Array,
    ],
    t0: Array,  # (n_traj,)
    z_stop: float,
    d0: Array,  # (n_traj, 6)  [x, y, z, vx, vy, vz]
    n_steps: int,
    n_save: int,  # = number of points to save
    mass: float,
    force_fn: Callable[[Array, Array, Array, Array], tuple[Array, Array, Array]],
    force_cst: Force,
    check_bounds_functions: Sequence[Callable[[Array, Array, Array], Array]],
) -> tuple[Array, Array, Array]:
    """
    Batched RK4 with a fixed number of steps per trajectory.

    - n_steps: total RK4 steps
    - n_steps_save: number of snapshots to save along the trajectory

    Behavior:
      - step 0 does not have to be saved (may be, depending on spacing)
      - the final step (n_steps-1) is ALWAYS saved
      - exactly n_steps_save snapshots are saved, at approximately even step indices
    """
    if n_save < 1:
        raise ValueError("n_steps_save must be at least 1")
    if n_save > n_steps:
        raise ValueError("n_steps_save cannot exceed n_steps")

    # --- initial conditions ---
    t0 = np.asarray(t0, dtype=float)  # (n_traj,)
    d = d0.astype(np.float64, copy=True)  # (n_traj, 6)

    z0 = d[:, 2]  # initial z for each trajectory
    vz0 = d[:, 5]

    t_end = t0 + (z_stop - z0) / vz0
    t = t0.copy()
    dt = (t_end - t0) / n_steps
    dt_col = dt[:, None]

    mask_survive = np.ones(t0.shape, dtype=bool)

    # --- choose save step indices in [0, n_steps-1] ---
    # guarantee last index is n_steps-1, no duplicates
    if n_save == 1:
        save_steps = np.array([n_steps - 1], dtype=int)
    else:
        # evenly spaced in 1..n_steps (1-based), then convert to 0-based
        # this always gives last index = n_steps-1
        save_steps = np.round(np.linspace(1, n_steps, n_save)).astype(int) - 1
    # sanity: sorted, unique
    save_steps = np.unique(np.sort(save_steps))
    # if rounding collapsed some points, pad from the end to get exactly n_steps_save
    if save_steps.size < n_save:
        # fill missing with closest-from-end indices
        extra = n_steps - 1 - np.arange(0, n_save - save_steps.size)
        save_steps = np.unique(np.sort(np.concatenate([save_steps, extra])))
    # ensure size and last index
    save_steps = save_steps[-n_save:]
    assert save_steps[-1] == n_steps - 1

    # --- allocate output ---
    n_traj = t0.size
    t_save = np.empty((n_save, n_traj), dtype=float)
    d_save = np.empty((n_save, n_traj, d.shape[1]), dtype=float)

    save_idx = 0
    next_save_step = save_steps[save_idx]

    # --- main RK4 loop ---
    for step in range(n_steps):
        k1 = f(t, d, mass, force_fn, force_cst)
        k2 = f(t + 0.5 * dt, d + 0.5 * dt_col * k1, mass, force_fn, force_cst)
        k3 = f(t + 0.5 * dt, d + 0.5 * dt_col * k2, mass, force_fn, force_cst)
        k4 = f(t + dt, d + dt_col * k3, mass, force_fn, force_cst)

        d += dt_col * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        for bound_fn in check_bounds_functions:
            mask_survive &= bound_fn(d[:, 0], d[:, 1], d[:, 2])
        t += dt

        if step == next_save_step:
            t_save[save_idx] = t
            d_save[save_idx] = d
            save_idx += 1
            if save_idx < n_save:
                next_save_step = save_steps[save_idx]

    # by construction, save_steps[-1] == n_steps-1, so t_save[-1], d_save[-1]
    # are the final time/state

    return t_save, d_save, mask_survive
