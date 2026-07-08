"""Time-dependent scalar diffusion on a SphericalManifold.

The canonical Laplace–Beltrami demonstrator: a pole-localised
Gaussian on a unit sphere relaxes toward its spatial mean. Each
spherical-harmonic component decays at its own analytic rate

    T_lm(t) = T_lm(0) · exp( -l(l+1) κ t / R^2 )

so high-l (small-scale) features die first and the low-l components
linger longest. Numerically the field flattens monotonically and
the L2 deviation from the mean decays exponentially.

Three pieces:
  (A) Time-step the Eulerian ``Diffusion`` solver forward, snapshot
      the field at several times, write a pyvista frame for each.
  (B) Track ``L2(T - <T>)`` against the dominant Y_10 decay rate
      ``exp(-2 κ t)`` (for the centered Gaussian, l=1 dominates).
  (C) Assemble a 2×3 grid of snapshots into a single figure.

Run:
    pixi run -e amr-dev python -u \\
        docs/examples/manifold_pdes/example_manifold_diffusion.py

Notes
-----
* ``Diffusion(...)`` is a linear time-step solve. Configuring it with
  ``snes_type = "ksponly"`` skips Newton's line-search machinery so
  the per-step iteration is a single linear KSP — clean output, no
  spurious ``DIVERGED_LINE_SEARCH`` warnings from a residual that's
  already at quadrature noise level.
* ``petsc_use_constant_nullspace = True`` handles the constant
  Laplace–Beltrami kernel on the closed manifold (conservative
  diffusion preserves the spatial mean exactly in the continuum;
  PETSc's nullspace projection enforces it discretely).
"""

from __future__ import annotations

import os

import numpy as np

import underworld3 as uw
from underworld3.systems import Diffusion


# ---------------------------------------------------------------------------
# Set up
# ---------------------------------------------------------------------------

CELL_SIZE = 0.15
DIFFUSIVITY = 0.05
DT = 0.1
N_STEPS = 40
SNAPSHOT_STEPS = (0, 4, 10, 20, 30, 40)
SIGMA = 0.4


def build_initial_gaussian(T, sphere):
    """Plant a Gaussian centred at (1, 0, 0) with std-dev ``SIGMA``
    (geodesic units on the unit sphere). Subtracts the discrete mean
    so the field is mean-zero — the conservative diffusion preserves
    that exactly."""
    coords = np.asarray(T.coords)
    arc = np.arccos(np.clip(coords[:, 0], -1.0, 1.0))
    blob = np.exp(-arc ** 2 / (2.0 * SIGMA ** 2))
    new_T = np.zeros_like(T.data)
    new_T[:, 0] = blob - blob.mean()
    T.pack_raw_data_to_petsc(new_T, sync=True)


def build_solver(sphere, T):
    diff = Diffusion(sphere, u_Field=T, order=1, theta=1.0)
    diff.constitutive_model = uw.constitutive_models.DiffusionModel
    diff.constitutive_model.Parameters.diffusivity = DIFFUSIVITY
    diff.petsc_use_constant_nullspace = True
    # Linear time-step solve — skip Newton's line search entirely.
    diff.petsc_options["snes_type"] = "ksponly"
    return diff


# ---------------------------------------------------------------------------
# pyvista — one snapshot per step + composite grid
# ---------------------------------------------------------------------------

def write_snapshot(sphere, T, step_idx, t, out_dir, lims):
    try:
        import pyvista as pv
        import underworld3.visualisation as vis
    except ImportError:
        return None

    pvmesh = vis.mesh_to_pv_mesh(sphere)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, T.sym)

    plotter = pv.Plotter(window_size=(700, 600), off_screen=True)
    plotter.set_background("white")
    plotter.add_mesh(
        pvmesh,
        scalars="T",
        cmap="RdBu_r",
        clim=lims,
        show_edges=False,
        smooth_shading=False,
        scalar_bar_args={"title": "T", "color": "black"},
    )
    plotter.add_text(f"t = {t:.2f}", position="upper_left", color="black", font_size=12)
    plotter.camera_position = "yz"
    plotter.camera.azimuth = 30
    plotter.camera.elevation = 20

    out_path = os.path.join(out_dir, f"_diffusion_step{step_idx:03d}.png")
    plotter.screenshot(out_path)
    plotter.close()
    return out_path


def make_composite(snapshot_paths, out_path):
    """Stitch the snapshots into a single 2 × 3 grid via PIL."""
    try:
        from PIL import Image
    except ImportError:
        print("  PIL not available — skipping composite.")
        return
    imgs = [Image.open(p) for p in snapshot_paths]
    w, h = imgs[0].size
    rows, cols = 2, 3
    grid = Image.new("RGB", (cols * w, rows * h), "white")
    for i, img in enumerate(imgs):
        r, c = divmod(i, cols)
        grid.paste(img, (c * w, r * h))
    grid.save(out_path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("Pole-localised Gaussian → relaxation toward zero mean on a")
    print(f"unit SphericalManifold (cellSize={CELL_SIZE}, κ={DIFFUSIVITY}, dt={DT})")
    print("=" * 72)

    out_dir = os.path.dirname(__file__)
    sphere = uw.meshing.SphericalManifold(radius=1.0, cellSize=CELL_SIZE)

    T = uw.discretisation.MeshVariable("T", sphere, 1, degree=2)
    build_initial_gaussian(T, sphere)
    diff = build_solver(sphere, T)

    # Color limits fixed by the initial field so the colormap is
    # comparable across snapshots.
    lims = (float(T.data.min()), float(T.data.max()))
    print(f"\nIC: T range = [{lims[0]:.4f}, {lims[1]:.4f}], "
          f"mean = {T.data[:, 0].mean():.6f}")

    # Decay diagnostic: ||T - <T>||_2 over time.
    snapshot_paths = []
    times = []
    rms_history = []

    if 0 in SNAPSHOT_STEPS:
        p = write_snapshot(sphere, T, 0, 0.0, out_dir, lims)
        if p is not None:
            snapshot_paths.append(p)
    times.append(0.0)
    rms_history.append(float(np.sqrt((T.data[:, 0] ** 2).mean())))

    print(f"\n{'step':>5} {'t':>8} {'||T||_2':>14} {'max':>14} "
          f"{'Y_10 decay':>14}")
    print("-" * 65)
    print(f"{0:>5} {0.0:>8.2f} {rms_history[0]:>14.4e} "
          f"{T.data.max():>14.4e} {1.0:>14.4e}")

    for step in range(1, N_STEPS + 1):
        diff.solve(timestep=DT)
        t = step * DT
        rms = float(np.sqrt((T.data[:, 0] ** 2).mean()))
        # Y_10 has eigenvalue 2; ||Y_10(t)|| ~ exp(-2 κ t)
        y10_factor = float(np.exp(-2.0 * DIFFUSIVITY * t))
        rel = rms / rms_history[0]
        times.append(t)
        rms_history.append(rms)

        if step % 4 == 0 or step in SNAPSHOT_STEPS:
            print(f"{step:>5} {t:>8.2f} {rms:>14.4e} "
                  f"{T.data.max():>14.4e} {y10_factor:>14.4e}")

        if step in SNAPSHOT_STEPS:
            p = write_snapshot(sphere, T, step, t, out_dir, lims)
            if p is not None:
                snapshot_paths.append(p)

    print("-" * 65)
    print("Y_10 column is the analytic decay envelope of the l=1 mode;")
    print("the numerical ||T||_2 / ||T||_0 should approach it as high-l")
    print("modes die out and the l=1 component dominates the residual.")

    # Composite figure
    composite_path = os.path.join(out_dir, "_diffusion_composite.png")
    if snapshot_paths:
        make_composite(snapshot_paths, composite_path)
        print(f"\nComposite snapshot grid: {composite_path}")


if __name__ == "__main__":
    main()
