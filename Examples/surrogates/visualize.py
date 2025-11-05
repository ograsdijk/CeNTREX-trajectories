import matplotlib.pyplot as plt
import numpy as np


def compute_error_maps_full(
    x_true,
    y_true,
    vx_true,
    vy_true,
    vz_true,
    x_pred,
    y_pred,
    vx_pred,
    vy_pred,
    vz_pred,
    x0,
    y0,
    vz0,
    R=None,
):
    """
    Returns a dict with inputs (r0, vz0) and absolute errors:
    rerr (radial pos), ex, ey (pos), evx, evy, evz (vel).
    If R is given, also returns r0_over_R for convenience.
    All positions in meters, velocities in m/s.
    """
    r0 = np.sqrt(x0**2 + y0**2)
    rerr = np.sqrt((x_pred - x_true) ** 2 + (y_pred - y_true) ** 2)
    emap = {
        "r0": r0,
        "vz0": vz0,
        "rerr": rerr,
        "ex": np.abs(x_pred - x_true),
        "ey": np.abs(y_pred - y_true),
        "evx": np.abs(vx_pred - vx_true),
        "evy": np.abs(vy_pred - vy_true),
        "evz": np.abs(vz_pred - vz_true),
    }
    if R is not None and R > 0:
        emap["r0_over_R"] = r0 / R
    return emap


def _binstats_xy(x, y, xbins, q=(50, 90, 99)):
    x = np.asarray(x)
    y = np.asarray(y)
    xbins = np.asarray(xbins)
    idx = np.digitize(x, xbins) - 1
    centers = 0.5 * (xbins[:-1] + xbins[1:])
    percs = {f"p{qq}": np.full(centers.size, np.nan) for qq in q}
    for b in range(centers.size):
        m = idx == b
        if np.any(m):
            yy = y[m]
            for qq in q:
                percs[f"p{qq}"][b] = np.percentile(yy, qq)
    return centers, percs


def _binstats_2d(x, y, z, xbins, ybins, q=50):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    xbins = np.asarray(xbins)
    ybins = np.asarray(ybins)
    Z = np.full((ybins.size - 1, xbins.size - 1), np.nan)
    xi = np.digitize(x, xbins) - 1
    yi = np.digitize(y, ybins) - 1
    for i in range(xbins.size - 1):
        for j in range(ybins.size - 1):
            m = (xi == i) & (yi == j)
            if m.any():
                Z[j, i] = np.percentile(z[m], q)
    Xg = 0.5 * (xbins[:-1] + xbins[1:])
    Yg = 0.5 * (ybins[:-1] + ybins[1:])
    return Xg, Yg, Z


def show_error_plots_full(emap, R=None):
    """
    Plots (each in its own figure):
      POS:
        1) Histogram of radial error |Δr| (mm)
        2) |Δr| vs r0 (P50/P90/P99) [mm]
        3) |Δr| vs vz0 (P50/P90/P99) [mm]
        4) 2D heatmap of median |Δr| over (r0, vz0) [mm]
      VEL:
        5) Histograms of |Δvx|, |Δvy|, |Δvz| (m/s)
        6) |Δv*| vs r0 (P50/P90/P99) for each component (m/s)
        7) |Δv*| vs vz0 (P50/P90/P99) for each component (m/s)
        8) 2D heatmaps of median |Δvx|, |Δvy|, |Δvz| over (r0, vz0) (m/s)
    """
    r0 = np.asarray(emap["r0"])
    vz0 = np.asarray(emap["vz0"])
    mm = 1e3

    # ----- Position errors -----
    rerr_mm = np.asarray(emap["rerr"]) * mm
    ex_mm = np.asarray(emap["ex"]) * mm
    ey_mm = np.asarray(emap["ey"]) * mm

    # 1) Histogram of radial error
    fig1, ax1 = plt.subplots()
    ax1.hist(rerr_mm, bins=np.linspace(0.0, np.percentile(rerr_mm, 99.5), 100))
    ax1.set_xlabel("radial error |Δr| (mm)")
    ax1.set_ylabel("count")
    ax1.set_title("Position error histogram")

    # Bins
    r0_max = np.percentile(r0, 99.5)
    r0_bins = np.linspace(0.0, max(r0_max, 1e-9), 60)
    vz_bins = np.linspace(np.percentile(vz0, 0.5), np.percentile(vz0, 99.5), 60)

    # 2) |Δr| vs r0 percentiles
    r0_c, per_r0 = _binstats_xy(r0, rerr_mm, r0_bins, q=(50, 90, 99))
    fig2, ax2 = plt.subplots()
    ax2.plot(r0_c * mm, per_r0["p50"], label="P50")
    ax2.plot(r0_c * mm, per_r0["p90"], label="P90")
    ax2.plot(r0_c * mm, per_r0["p99"], label="P99")
    ax2.set_xlabel("initial radius r0 (mm)")
    ax2.set_ylabel("radial error percentile (mm)")
    ax2.set_title("|Δr| vs r0")
    ax2.legend()
    ax2.grid(True)
    if R is not None and R > 0:

        def r_mm_to_r_over_R(x_mm):
            return (x_mm / 1e3) / R

        def r_over_R_to_r_mm(x_rel):
            return (x_rel * R) * 1e3

        sec = ax2.secondary_xaxis("top", functions=(r_mm_to_r_over_R, r_over_R_to_r_mm))
        sec.set_xlabel("r0 / R")

    # 3) |Δr| vs vz0 percentiles
    vz_c, per_vz = _binstats_xy(vz0, rerr_mm, vz_bins, q=(50, 90, 99))
    fig3, ax3 = plt.subplots()
    ax3.plot(vz_c, per_vz["p50"], label="P50")
    ax3.plot(vz_c, per_vz["p90"], label="P90")
    ax3.plot(vz_c, per_vz["p99"], label="P99")
    ax3.set_xlabel("v_z0 (m/s)")
    ax3.set_ylabel("radial error percentile (mm)")
    ax3.set_title("|Δr| vs v_z0")
    ax3.legend()
    ax3.grid(True)

    # 4) 2D heatmap median |Δr|
    Xg, Yg, Zmed = _binstats_2d(r0, vz0, rerr_mm, r0_bins, vz_bins, q=50)
    fig4, ax4 = plt.subplots()
    im = ax4.imshow(
        Zmed,
        origin="lower",
        extent=[(r0_bins[0] * mm), (r0_bins[-1] * mm), vz_bins[0], vz_bins[-1]],
        aspect="auto",
        interpolation="nearest",
    )
    ax4.set_xlabel("r0 (mm)")
    ax4.set_ylabel("v_z0 (m/s)")
    ax4.set_title("Median |Δr| (mm) over (r0, v_z0)")
    fig4.colorbar(im, ax=ax4, label="median |Δr| (mm)")

    # ----- Velocity errors -----
    evx = np.asarray(emap["evx"])
    evy = np.asarray(emap["evy"])
    evz = np.asarray(emap["evz"])

    # 5) Velocity error histograms
    fig5, axs5 = plt.subplots(1, 3, figsize=(12, 3))
    for ax, e, name in zip(axs5, [evx, evy, evz], ["|Δv_x|", "|Δv_y|", "|Δv_z|"]):
        ax.hist(e, bins=np.linspace(0.0, np.percentile(e, 99.5), 100))
        ax.set_xlabel(f"{name} (m/s)")
        ax.set_ylabel("count")
        ax.set_title(f"{name} histogram")
    fig5.tight_layout()

    # 6) |Δv*| vs r0 percentiles
    for e, comp in [(evx, "vx"), (evy, "vy"), (evz, "vz")]:
        c, per = _binstats_xy(r0, e, r0_bins, q=(50, 90, 99))
        fig, ax = plt.subplots()
        ax.plot(c * mm, per["p50"], label="P50")
        ax.plot(c * mm, per["p90"], label="P90")
        ax.plot(c * mm, per["p99"], label="P99")
        ax.set_xlabel("initial radius r0 (mm)")
        ax.set_ylabel(f"|Δ{comp}| percentile (m/s)")
        ax.set_title(f"|Δ{comp}| vs r0")
        ax.legend()
        ax.grid(True)
        if R is not None and R > 0:

            def r_mm_to_r_over_R(x_mm):
                return (x_mm / 1e3) / R

            def r_over_R_to_r_mm(x_rel):
                return (x_rel * R) * 1e3

            sec = ax.secondary_xaxis(
                "top", functions=(r_mm_to_r_over_R, r_over_R_to_r_mm)
            )
            sec.set_xlabel("r0 / R")

    # 7) |Δv*| vs vz0 percentiles
    for e, comp in [(evx, "vx"), (evy, "vy"), (evz, "vz")]:
        c, per = _binstats_xy(vz0, e, vz_bins, q=(50, 90, 99))
        fig, ax = plt.subplots()
        ax.plot(c, per["p50"], label="P50")
        ax.plot(c, per["p90"], label="P90")
        ax.plot(c, per["p99"], label="P99")
        ax.set_xlabel("v_z0 (m/s)")
        ax.set_ylabel(f"|Δ{comp}| percentile (m/s)")
        ax.set_title(f"|Δ{comp}| vs v_z0")
        ax.legend()
        ax.grid(True)

    # 8) 2D heatmaps for |Δvx|, |Δvy|, |Δvz|
    for e, title in [
        (evx, "Median |Δv_x| (m/s)"),
        (evy, "Median |Δv_y| (m/s)"),
        (evz, "Median |Δv_z| (m/s)"),
    ]:
        Xg2, Yg2, Zmed2 = _binstats_2d(r0, vz0, e, r0_bins, vz_bins, q=50)
        fig, ax = plt.subplots()
        im2 = ax.imshow(
            Zmed2,
            origin="lower",
            extent=[(r0_bins[0] * mm), (r0_bins[-1] * mm), vz_bins[0], vz_bins[-1]],
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_xlabel("r0 (mm)")
        ax.set_ylabel("v_z0 (m/s)")
        ax.set_title(title)
        fig.colorbar(im2, ax=ax, label=title)
    plt.show()
