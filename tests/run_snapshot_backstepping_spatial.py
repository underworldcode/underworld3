"""Snapshot toolkit demonstration: spatial view of back-stepping.

Companion to ``run_snapshot_backstepping_demo.py``. That script answers
"when" via a CFL-ratio time series; this one answers "what" via 2×2
spatial panels at the four moments that matter:

    [initial state (snapshot taken here)]      [after speculative bad step]
    [after model.load_state(snap)]                [after substep recovery to same t]

Each panel shows the swarm particles coloured by their carried
material value (initial radial position), with the domain boundary
drawn as context. The diagonal pairs tell two stories:

  - top-left vs. bottom-left should be **visually identical**. That's
    the proof that model.load_state(snap) put the captured state back
    exactly. If the figure ever stops showing two identical panels
    in that diagonal, the snapshot mechanism has broken.

  - top-right vs. bottom-right are the same simulation time reached
    by two different paths: a single too-large Δt step (corner-clumping,
    over-stretched, CFL violated) vs. ten substeps at sub-CFL Δt.

Run:
    pixi run -e amr-dev python tests/run_snapshot_backstepping_spatial.py

Output:
    snapshot_backstepping_spatial.png in the current working directory.
"""

import numpy as np
import sympy
import matplotlib.pyplot as plt

import underworld3 as uw


def _capture(swarm, material):
    """Snapshot the swarm spatial state for plotting (positions + material)."""
    coords = swarm._particle_coordinates.data.copy()
    mat = np.asarray(material.data).copy()
    return coords, mat


def main(out_path: str = "snapshot_backstepping_spatial.png"):
    uw.reset_default_model()
    model = uw.get_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8.0
    )

    x_sym, y_sym = mesh.X
    V_fn = sympy.Matrix([[x_sym - 0.5, y_sym - 0.5]]).T

    swarm = uw.swarm.Swarm(mesh)
    material = swarm.add_variable("material", 1, dtype=float)
    swarm.populate(fill_param=2)

    cfl_threshold = mesh.get_min_radius()
    small_dt = 0.05
    candidate_dt = 0.5
    n_substeps = int(round(candidate_dt / small_dt))

    # Colour each particle by its initial radial distance from centre.
    coords_initial = swarm._particle_coordinates.data.copy()
    material.data[:, 0] = np.linalg.norm(coords_initial - 0.5, axis=1)

    # Initial state.
    state_initial = _capture(swarm, material)

    # Take the snapshot — this is the state that bottom-left will
    # have to match after restore.
    snap = model.save_state()

    # --- Speculative big step ---
    swarm.advection(V_fn, delta_t=candidate_dt, step_limit=False)
    state_after_bad = _capture(swarm, material)
    max_disp_bad = np.max(
        np.linalg.norm(state_after_bad[0] - state_initial[0], axis=1)
    )
    cfl_bad = max_disp_bad / cfl_threshold

    # --- model.load_state(snap) ---
    model.load_state(snap)
    state_after_restore = _capture(swarm, material)

    # --- Substep recovery to the same target time ---
    for _ in range(n_substeps):
        swarm.advection(V_fn, delta_t=small_dt, step_limit=False)
    state_after_recovery = _capture(swarm, material)
    max_disp_recovery = np.max(
        np.linalg.norm(state_after_recovery[0] - state_initial[0], axis=1)
    )
    cfl_recovery_per_step = (
        np.max(
            np.linalg.norm(
                state_after_recovery[0] - state_initial[0], axis=1
            )
        )
        / n_substeps
        / cfl_threshold
    )

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 11), constrained_layout=True)

    panels = [
        (
            axes[0, 0],
            state_initial,
            "Initial state",
            "t = 0.00  •  snapshot taken here",
        ),
        (
            axes[0, 1],
            state_after_bad,
            "After speculative Δt = {:.2f}".format(candidate_dt),
            "t = {:.2f}  •  CFL = {:.1f} × threshold".format(
                candidate_dt, cfl_bad
            ),
        ),
        (
            axes[1, 0],
            state_after_restore,
            "After model.load_state(snap)",
            "t = 0.00  •  visually identical to top-left",
        ),
        (
            axes[1, 1],
            state_after_recovery,
            "After {} substeps at Δt = {:.3f}".format(n_substeps, small_dt),
            "t = {:.2f}  •  same time as top-right, CFL safe".format(
                candidate_dt
            ),
        ),
    ]

    # Common colour scale across all four panels so colours mean the
    # same thing everywhere.
    all_mat = np.concatenate(
        [s[1][:, 0] for s in (state_initial, state_after_bad,
                              state_after_restore, state_after_recovery)]
    )
    vmin, vmax = float(all_mat.min()), float(all_mat.max())

    last_sc = None
    for ax, (coords, mat), title, subtitle in panels:
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=mat[:, 0], s=8, cmap="viridis",
            vmin=vmin, vmax=vmax,
        )
        last_sc = sc
        # Domain boundary.
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="0.5", linewidth=0.9)
        # Generous limits so the bad-step overshoot is visible if any
        # particles strayed past the boundary.
        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.15, 1.15)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Single colourbar on the right.
    cbar = fig.colorbar(
        last_sc, ax=axes.ravel().tolist(),
        shrink=0.55, pad=0.02, aspect=30,
    )
    cbar.set_label("material  =  initial radial distance from centre",
                   fontsize=9)

    fig.suptitle(
        "Adaptive-Δt back-stepping  •  spatial view\n"
        "Top-left ↔ Bottom-left identical (snap-back).  Top-right ↔ Bottom-right same simulation time, different path.",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Wrote {out_path}")
    print(f"  CFL ratio (bad single step):   {cfl_bad:.2f}")
    print(f"  CFL ratio per substep (mean):  {cfl_recovery_per_step:.3f}")
    print(f"  max disp bad path:             {max_disp_bad:.4f}")
    print(f"  max disp recovery path (cumul): {max_disp_recovery:.4f}")


if __name__ == "__main__":
    main()
