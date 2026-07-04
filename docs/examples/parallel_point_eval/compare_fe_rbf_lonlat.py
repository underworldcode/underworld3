"""FE-vs-RBF SLCN solid-body rotation on a SphericalManifold, compared to
the analytic rotated Gaussian, rendered in an equirectangular (lat/lon)
projection.

Both modes solve the same advection-diffusion PDE for a Gaussian carried by
solid-body rotation V = (-y, x, 0) (omega = 1 about z) over one full 2*pi
rotation. They differ only in the semi-Lagrangian trace-back evaluation:

  - FE  : adv.solve(..., _evalf=False) -> DMInterpolation / FE shape functions
          (the cell-location bypass in this worktree makes this correct on a
          2-manifold; previously it mis-routed ~16% of trace-back points).
  - RBF : adv.solve(..., _evalf=True)  -> Shepard inverse-distance (the old
          workaround; linear, O(h) regardless of element order).

monotone_mode="clamp" is used for BOTH so the only difference is the
trace-back interpolation method, not the limiter.

Because the flow is solid-body rotation, the analytic solution at time t is
the initial Gaussian re-centred at (cos t, sin t, 0) -- so we have an exact
reference at every step.

Outputs (in ./_fe_vs_rbf_lonlat/):
  - fe_vs_rbf_lonlat.mp4 : 3-panel equirectangular movie  FE | RBF | analytic
  - l2_error_vs_angle.png: relative-L2 error vs rotation angle, FE and RBF
  - data_{fe,rbf}.npz     : per-frame lat/lon grids + L2 series (re-renderable)

Run:
    pixi run -e amr-dev python compare_fe_rbf_lonlat.py
"""

import os
import gc

import numpy as np
import sympy

import underworld3 as uw

# --- configuration (matches animate_manifold_advection.py) -----------------
CELL_SIZE = 0.075
N_STEPS = 72
DT = 2 * np.pi / N_STEPS
DIFFUSIVITY = 1.0e-4
GAUSSIAN_SIGMA = 0.3
T_DEGREE = 3
QDEGREE = T_DEGREE
MONOTONE_MODE = "clamp"  # same for both modes; isolates the _evalf difference

# equirectangular sampling grid
NLON, NLAT = 360, 180

OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_fe_vs_rbf_lonlat"
)


def make_lonlat_grid():
    """Regular (lon, lat) grid -> (N,3) cartesian points on the unit sphere."""
    lon = np.linspace(-180.0, 180.0, NLON)
    lat = np.linspace(-90.0, 90.0, NLAT)
    lon_g, lat_g = np.meshgrid(lon, lat)
    theta = np.radians(90.0 - lat_g)  # colatitude
    phi = np.radians(lon_g)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    xyz = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    return lon, lat, xyz, (NLAT, NLON)


def analytic_on_points(pts, t):
    """Gaussian (sigma) centred at the rotated point (cos t, sin t, 0)."""
    c = np.array([np.cos(t), np.sin(t), 0.0])
    d2 = ((pts - c) ** 2).sum(axis=1)
    return np.exp(-d2 / (2.0 * GAUSSIAN_SIGMA**2))


def run_mode(mode, grid_xyz, grid_shape):
    """Build a fresh mesh/solver, run one full rotation, return per-frame
    lat/lon grids, relative-L2 series, and rotation angles (degrees)."""
    evalf = mode == "rbf"

    mesh = uw.meshing.SphericalManifold(
        radius=1.0, cellSize=CELL_SIZE, qdegree=QDEGREE
    )
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=T_DEGREE)
    node = np.asarray(T.coords)

    # initial condition: analytic field at t = 0
    T.data[:, 0] = analytic_on_points(node, 0.0)

    x_sym, y_sym, z_sym = mesh.X
    V_sym = sympy.Matrix([[-y_sym, x_sym, sympy.sympify(0)]])

    adv = uw.systems.AdvDiffusionSLCN(
        mesh, u_Field=T, V_fn=V_sym, order=1, monotone_mode=MONOTONE_MODE
    )
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = DIFFUSIVITY
    adv.f = sympy.Matrix.zeros(1, 1)

    n_frames = N_STEPS + 1
    grids = np.empty((n_frames, *grid_shape))
    l2 = np.empty(n_frames)
    angles = np.empty(n_frames)

    def sample_grid():
        # grid_xyz is constant across frames -> DMInterpolation structure is
        # built once and cached; subsequent calls are cheap evaluates.
        vals = uw.function.evaluate(T.sym[0], grid_xyz)
        return np.asarray(vals).reshape(grid_shape)

    def rel_l2(t):
        ana = analytic_on_points(node, t)
        num = np.asarray(T.data[:, 0])
        return float(
            np.sqrt(((num - ana) ** 2).mean()) / np.sqrt((ana**2).mean())
        )

    grids[0] = sample_grid()
    l2[0] = rel_l2(0.0)
    angles[0] = 0.0

    print(f"[{mode}] stepping {N_STEPS} x dt={DT:.4f} "
          f"({np.degrees(DT):.1f} deg/step), monotone={MONOTONE_MODE}")
    for k in range(N_STEPS):
        adv.solve(timestep=DT, _evalf=evalf)
        t = (k + 1) * DT
        grids[k + 1] = sample_grid()
        l2[k + 1] = rel_l2(t)
        angles[k + 1] = np.degrees(t)
        if (k + 1) % 8 == 0:
            tmax = float(np.asarray(T.data[:, 0]).max())
            tmin = float(np.asarray(T.data[:, 0]).min())
            print(f"[{mode}] step {k + 1:3d}/{N_STEPS} t={t:5.2f} "
                  f"relL2={l2[k + 1]:.3e}  T:[{tmin:+.3f},{tmax:+.3f}]")

    # free for the next mode
    del adv, T, mesh
    gc.collect()
    return grids, l2, angles


def _field_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    # signed_warm: violet for undershoot (<0), white at 0, warm for the blob.
    return LinearSegmentedColormap.from_list(
        "signed_warm",
        [
            (0.00, "#7e2e9d"),
            (0.15, "#bb91dd"),
            (0.231, "#ffffff"),
            (0.31, "#cfe5ff"),
            (0.46, "#4f9aff"),
            (0.65, "#3ec76b"),
            (0.85, "#ffa040"),
            (1.00, "#c41e1e"),
        ],
    )


def render_movie(results, ana_grids, angles_deg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

    cmap = _field_cmap()
    clim = (-0.3, 1.0)
    extent = [-180, 180, -90, 90]

    fe = results["fe"]["grids"]
    rbf = results["rbf"]["grids"]
    n_frames = fe.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 3.4), constrained_layout=True)
    panels = [("FE  (_evalf=False)", fe), ("RBF (_evalf=True)", rbf),
              ("analytic", ana_grids)]
    ims = []
    for ax, (title, data) in zip(axes, panels):
        im = ax.imshow(data[0], extent=extent, origin="lower", aspect="auto",
                       cmap=cmap, vmin=clim[0], vmax=clim[1])
        ax.set_title(title)
        ax.set_xlabel("longitude")
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks([-90, -45, 0, 45, 90])
        ims.append(im)
    axes[0].set_ylabel("latitude")
    fig.colorbar(ims[-1], ax=axes, shrink=0.85, label="T")
    suptitle = fig.suptitle("rotation 0.0 deg")

    def update(k):
        ims[0].set_data(fe[k])
        ims[1].set_data(rbf[k])
        ims[2].set_data(ana_grids[k])
        suptitle.set_text(f"rotation {angles_deg[k]:.0f} deg  (step {k}/{N_STEPS})")
        return (*ims, suptitle)

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)

    mp4 = os.path.join(OUT_DIR, "fe_vs_rbf_lonlat.mp4")
    try:
        anim.save(mp4, writer=FFMpegWriter(fps=12, bitrate=4000))
        print(f"  movie: {mp4}")
    except Exception as exc:  # noqa: BLE001
        gif = os.path.join(OUT_DIR, "fe_vs_rbf_lonlat.gif")
        print(f"  ffmpeg failed ({exc}); writing GIF instead")
        anim.save(gif, writer=PillowWriter(fps=12))
        print(f"  movie: {gif}")
    plt.close(fig)


def render_l2(results, angles_deg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(angles_deg, results["fe"]["l2"] * 100, "-o", ms=3,
            label="FE  (_evalf=False)", color="#c41e1e")
    ax.plot(angles_deg, results["rbf"]["l2"] * 100, "-s", ms=3,
            label="RBF (_evalf=True)", color="#4f9aff")
    ax.set_xlabel("rotation angle (deg)")
    ax.set_ylabel("relative L2 error vs analytic (%)")
    ax.set_title(f"SLCN rotation on SphericalManifold "
                 f"(cs={CELL_SIZE}, P{T_DEGREE}, monotone={MONOTONE_MODE})")
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.grid(True, alpha=0.3)
    ax.legend()
    png = os.path.join(OUT_DIR, "l2_error_vs_angle.png")
    fig.savefig(png, dpi=140)
    print(f"  L2 plot: {png}")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    lon, lat, grid_xyz, grid_shape = make_lonlat_grid()

    # analytic lat/lon grids (mode-independent)
    ana_grids = np.empty((N_STEPS + 1, *grid_shape))
    for k in range(N_STEPS + 1):
        ana_grids[k] = analytic_on_points(grid_xyz, k * DT).reshape(grid_shape)
    angles_deg = np.degrees(np.arange(N_STEPS + 1) * DT)

    results = {}
    for mode in ("fe", "rbf"):
        print(f"\n=== {mode.upper()} ===")
        grids, l2, angles = run_mode(mode, grid_xyz, grid_shape)
        results[mode] = dict(grids=grids, l2=l2, angles=angles)
        np.savez_compressed(
            os.path.join(OUT_DIR, f"data_{mode}.npz"),
            grids=grids, l2=l2, angles=angles, ana_grids=ana_grids,
            angles_deg=angles_deg,
        )
        print(f"[{mode}] final relL2 = {l2[-1] * 100:.2f}%   "
              f"amplitude retained = {grids[-1].max() / grids[0].max() * 100:.1f}%")

    print("\nRendering outputs...")
    render_l2(results, angles_deg)
    render_movie(results, ana_grids, angles_deg)

    print("\nSummary:")
    print(f"  {'mode':>5} {'final relL2 %':>14} {'amp retained %':>15}")
    for mode in ("fe", "rbf"):
        g = results[mode]["grids"]
        print(f"  {mode:>5} {results[mode]['l2'][-1] * 100:>14.2f} "
              f"{g[-1].max() / g[0].max() * 100:>15.1f}")


if __name__ == "__main__":
    main()
