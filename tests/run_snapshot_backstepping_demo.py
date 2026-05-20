"""Snapshot toolkit demonstration: time-series view of back-stepping.

A small adaptive-Δt drama on one axis. The y-axis is the canonical
adaptive-Δt diagnostic: max per-step particle displacement compared
to the mesh cell radius (CFL ratio). The story:

  - timestep forward at small Δt for a while (CFL well under 1),
  - take a snapshot,
  - try one too-large Δt (CFL spikes far above 1),
  - detect the bad step, call ``model.load_state(snap)``,
  - replay the same time interval with many small steps (CFL stays small),
  - continue past the speculative end-time.

The plot shows two overlapping segments in the snap-back zone:

  - the abandoned big step (dashed red X — single tall spike, CFL ≫ 1),
  - the kept substep trajectory (solid blue dots — each well under 1).

At ``t = t_speculative_end`` both an abandoned and a recovered value
exist. The time axis is genuinely multi-valued there — that's the
visual point of the figure.

Run:
    pixi run -e amr-dev python tests/run_snapshot_backstepping_demo.py

Output:
    snapshot_backstepping_demo.png in the current working directory.

Companion to ``tests/test_0007_snapshot_inmemory.py``'s
``test_backstepping_cfl_recovery_end_to_end``.
"""

import numpy as np
import sympy
import matplotlib.pyplot as plt

import underworld3 as uw


def _max_step_displacement(coords_now: np.ndarray, coords_before: np.ndarray) -> float:
    """Largest distance any local particle moved during the last step."""
    return float(np.max(np.linalg.norm(coords_now - coords_before, axis=1)))


def main(out_path: str = "snapshot_backstepping_demo.png"):
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

    # Time series we'll plot. (t_end_of_step, max_step_displacement)
    # for every step we keep.
    times_kept = []
    cfl_kept = []

    def take_step(dt: float):
        before = swarm._particle_coordinates.data.copy()
        swarm.advection(V_fn, delta_t=dt, step_limit=False)
        after = swarm._particle_coordinates.data
        return _max_step_displacement(after, before)

    # --- Phase 1: normal stepping ---
    n_phase1 = 5
    t = 0.0
    for _ in range(n_phase1):
        disp = take_step(small_dt)
        t += small_dt
        times_kept.append(t)
        cfl_kept.append(disp / cfl_threshold)

    t_snap = t
    snap = model.save_state()

    # --- Phase 2: speculative big step ---
    disp_bad = take_step(candidate_dt)
    t_bad_end = t_snap + candidate_dt
    cfl_bad = disp_bad / cfl_threshold

    # --- CFL violated → restore ---
    model.load_state(snap)

    # --- Phase 3: substep replay ---
    times_recovered = []
    cfl_recovered = []
    for _ in range(n_substeps):
        disp = take_step(small_dt)
        t += small_dt
        times_recovered.append(t)
        cfl_recovered.append(disp / cfl_threshold)

    # --- Phase 4: continue past the speculative endpoint ---
    n_phase4 = 5
    times_post = []
    cfl_post = []
    for _ in range(n_phase4):
        disp = take_step(small_dt)
        t += small_dt
        times_post.append(t)
        cfl_post.append(disp / cfl_threshold)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Shaded snap-back zone.
    ax.axvspan(t_snap, t_bad_end, color="0.94", zorder=0)

    # CFL = 1 reference.
    ax.axhline(1.0, color="0.6", linestyle="-", linewidth=0.8)
    ax.text(
        0.005, 1.04,
        "CFL = 1 (one cell radius per step)",
        fontsize=9, color="0.4", transform=ax.get_yaxis_transform(),
        va="bottom",
    )

    # Phase 1: pre-snapshot trajectory.
    ax.plot(
        times_kept, cfl_kept,
        marker="o", markersize=5, linewidth=1.5, color="C0",
        label="Time-stepping at Δt = {:.2f}".format(small_dt),
    )

    # Snapshot marker.
    ax.axvline(t_snap, color="0.6", linestyle=":", linewidth=1)
    ax.annotate(
        "snapshot taken",
        xy=(t_snap, 0.02), xycoords=("data", "axes fraction"),
        xytext=(-4, 4), textcoords="offset points",
        ha="right", va="bottom", fontsize=9, color="0.3",
    )

    # Speculative bad step: dashed from snapshot horizontal-ish to (t_bad_end, cfl_bad).
    ax.plot(
        [t_snap, t_bad_end], [cfl_kept[-1], cfl_bad],
        linestyle="--", linewidth=1.5, color="C3", alpha=0.7,
        label="Speculative Δt = {:.2f}".format(candidate_dt),
    )
    ax.scatter(
        [t_bad_end], [cfl_bad], marker="X", s=130, color="C3", zorder=5,
    )
    ax.annotate(
        "abandoned: CFL = {:.1f}".format(cfl_bad),
        xy=(t_bad_end, cfl_bad),
        xytext=(8, -2), textcoords="offset points",
        ha="left", va="center", fontsize=10, color="C3", fontweight="bold",
    )

    # Snap-back arrow.
    ax.annotate(
        "",
        xy=(t_snap + 0.003, 0.18),
        xytext=(t_bad_end - 0.003, max(cfl_bad - 0.5, 1.5)),
        arrowprops=dict(
            arrowstyle="->", color="0.45",
            connectionstyle="arc3,rad=-0.35", linewidth=1.4,
        ),
    )
    ax.text(
        0.5 * (t_snap + t_bad_end),
        0.4 * cfl_bad,
        "model.load_state(snap)",
        ha="center", va="center", fontsize=10, color="0.35",
        style="italic",
        bbox=dict(facecolor="white", edgecolor="0.7", boxstyle="round,pad=0.25"),
    )

    # Phase 3: recovered substeps.
    ax.plot(
        times_recovered, cfl_recovered,
        marker="o", markersize=5, linewidth=1.5, color="C0",
    )

    # Phase 4: continuation.
    ax.plot(
        times_post, cfl_post,
        marker="o", markersize=5, linewidth=1.5, color="C0",
    )

    # Snap-back zone label (above the axes).
    ax.text(
        0.5 * (t_snap + t_bad_end),
        1.015, "snap-back zone — t is multi-valued",
        ha="center", va="bottom", fontsize=10, color="0.4",
        transform=ax.get_xaxis_transform(),
    )

    ax.set_xlabel("simulation time  t")
    ax.set_ylabel("CFL ratio  =  max per-step displacement / cell radius")
    ax.set_title(
        "Adaptive-Δt back-stepping  •  model.save_state() / model.load_state()",
        pad=22,
    )
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, axis="y", color="0.92", linewidth=0.6)
    ax.set_xlim(-0.02, t + 0.02)
    ax.set_ylim(-0.3, cfl_bad * 1.12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Wrote {out_path}")
    print(f"  t_snap                 = {t_snap:.3f}")
    print(f"  speculative Δt         = {candidate_dt:.3f}")
    print(f"  CFL ratio (bad step)   = {cfl_bad:.2f}")
    print(f"  substeps to recover:    {n_substeps} × Δt = {small_dt:.3f}")
    print(f"  max CFL on substeps:    {max(cfl_recovered):.3f}")


if __name__ == "__main__":
    main()
