"""Phase B field plots — velocity / strain-rate / stress for yield-active cases.

Runs the bench_ti_vep_harmonic geometry with ETD-2 for one period at one
or more yield-active (θ, τ_y) combinations, captures the full mesh-variable
fields at peak forcing, and plots velocity vectors + strain-rate magnitude
+ stress magnitude with the spatial τ_y(x) yield zone overlaid.

Run::

    pixi run -e amr-dev python -u docs/developer/design/experiments/exp-integrator/_plot_phase_b_fields.py
"""

import os
import time
import sys

import numpy as np
import sympy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator

import underworld3 as uw
from underworld3.function import expression


# Add the design dir to path so we can reuse the killer-test build helpers
_DESIGN_DIR = os.path.dirname(os.path.abspath(__file__))
if _DESIGN_DIR not in sys.path:
    sys.path.insert(0, _DESIGN_DIR)
from _exp_integrator_phase_b_killer import build_ti_exp_stokes  # noqa: E402


V0 = 0.5
OMEGA = np.pi / 2.0
DT = 0.05
H = 1.0
W = 1.0
FAULT_WIDTH = 0.06


def run_capture_at_yield_peak(theta_deg, tau_y_at_fault, n_periods=2):
    """Run ``n_periods`` of the harmonic forcing and capture fields at
    the step where the yield-zone σ_II reaches its peak — i.e. when yield
    is most active in the fault zone.

    Strategy: run forward, after each step record σ_II_max in the
    fault-zone mask AND a snapshot of the full state; at the end pick
    the step with the largest in-fault σ_II to plot.
    """
    label = f"fields_th{theta_deg:+.0f}_ty{tau_y_at_fault:.2f}".replace(".", "p")
    mesh, stokes, V_top, n_vec = build_ti_exp_stokes(label, theta_deg, tau_y_at_fault)
    cm = stokes.constitutive_model
    DFDt = stokes.Unknowns.DFDt

    E_sym = stokes.Unknowns.E
    n_x, n_y = n_vec
    cx, cy = 0.5 * W, 0.5 * H
    sigma_coords = DFDt.psi_star[0].coords
    # fault-zone mask: signed distance to fault line ≤ 1.5·FAULT_WIDTH
    sd = np.abs((sigma_coords[:, 0] - cx) * n_x + (sigma_coords[:, 1] - cy) * n_y)
    fault_mask = sd < 1.5 * FAULT_WIDTH

    # τ_y(x) field — constant in time, evaluate once
    ty_field = np.asarray(
        uw.function.evaluate(cm.Parameters.yield_stress.sym, sigma_coords)
    ).flatten()

    T_END = n_periods * 2.0 * np.pi / OMEGA
    snapshots = []          # list of dicts captured each step
    snapshot_metrics = []   # [(t, sigma_II_in_fault_max), ...]
    t_cur = 0.0
    while t_cur < T_END - 1e-9:
        dt = min(DT, T_END - t_cur)
        t_end_step = t_cur + dt
        v_now = V0 * float(np.cos(OMEGA * t_end_step))
        V_top.sym = sympy.Float(v_now)
        cm.Parameters.dt_elastic = dt
        stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=2)

        sigma_arr = np.asarray(DFDt.psi_star[0].array)
        sigma_II = np.sqrt(0.5 * (sigma_arr ** 2).sum(axis=(1, 2)))
        # Track σ_II in fault zone, and discount the initial transient
        # (first half-period where σ is still ramping from zero) by
        # only considering steps after t > T_period.
        t_recordable = t_end_step > 2.0 * np.pi / OMEGA
        in_fault_max = float(sigma_II[fault_mask].max()) if fault_mask.any() else 0.0
        snapshot_metrics.append((t_end_step, in_fault_max, t_recordable))

        # Capture every step (cheap) — snapshot for the chosen step at end
        # Strain rate (eval is the expensive bit, do it inline)
        edot_xx = np.asarray(uw.function.evaluate(E_sym[0, 0], sigma_coords)).flatten()
        edot_xy = np.asarray(uw.function.evaluate(E_sym[0, 1], sigma_coords)).flatten()
        edot_yy = np.asarray(uw.function.evaluate(E_sym[1, 1], sigma_coords)).flatten()
        edot_II = np.sqrt(0.5 * (edot_xx ** 2 + edot_yy ** 2 + 2 * edot_xy ** 2))
        u_arr = np.asarray(stokes.u.array)
        snapshots.append(dict(
            t=t_end_step,
            v_top=v_now,
            u_coords=stokes.u.coords.copy(),
            u=u_arr.reshape(-1, 2).copy(),
            sigma_arr=sigma_arr.copy(),
            sigma_II=sigma_II.copy(),
            edot_II=edot_II,
            edot_xy=edot_xy,
        ))
        t_cur = t_end_step

    # Pick the step (after the first period) with the biggest in-fault σ_II.
    candidates = [(t, m) for (t, m, ok) in snapshot_metrics if ok]
    if not candidates:
        # Fallback: take the last step
        chosen = snapshots[-1]
    else:
        idx = int(np.argmax([m for (_, m) in candidates]))
        # candidates index → snapshot index (count of recordable + initial transient)
        first_recordable = next(
            i for i, sm in enumerate(snapshot_metrics) if sm[2]
        )
        chosen = snapshots[first_recordable + idx]
    chosen.update(
        theta_deg=theta_deg,
        tau_y_at_fault=tau_y_at_fault,
        n_vec=n_vec,
        sigma_coords=sigma_coords,
        ty_field=ty_field,
        fault_mask=fault_mask,
        T_END=T_END,
    )
    chosen["yield_ratio"] = chosen["sigma_II"] / np.maximum(ty_field, 1e-30)
    chosen["sigma_xy"] = chosen["sigma_arr"][:, 0, 1]
    chosen["sigma_xx"] = chosen["sigma_arr"][:, 0, 0]
    chosen["sigma_yy"] = chosen["sigma_arr"][:, 1, 1]

    print(
        f"  picked step at t={chosen['t']:.3f} (V_top={chosen['v_top']:+.4f}); "
        f"max in-fault σ_II = {float(chosen['sigma_II'][fault_mask].max()):.4f} "
        f"(τ_y_centre={tau_y_at_fault}, ratio "
        f"{float(chosen['sigma_II'][fault_mask].max())/tau_y_at_fault:.3f}·τ_y)",
        flush=True,
    )
    return chosen


def plot_one(snapshot, out_path):
    th = snapshot["theta_deg"]
    ty_fault = snapshot["tau_y_at_fault"]
    n_x, n_y = snapshot["n_vec"]
    cx, cy = 0.5 * W, 0.5 * H

    sx, sy = snapshot["sigma_coords"][:, 0], snapshot["sigma_coords"][:, 1]
    tri = Triangulation(sx, sy)

    ux, uy = snapshot["u_coords"][:, 0], snapshot["u_coords"][:, 1]
    u_x, u_y = snapshot["u"][:, 0], snapshot["u"][:, 1]

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), sharex=True, sharey=True)

    # ---- Top-left: velocity field with fault overlay ------------------
    ax = axes[0, 0]
    speed = np.sqrt(u_x ** 2 + u_y ** 2)
    ax.tricontourf(
        Triangulation(ux, uy), speed, levels=24, cmap="Blues", alpha=0.7,
    )
    # Subsample for arrows (~every 2nd node)
    sub = slice(None, None, 4)
    ax.quiver(
        ux[sub], uy[sub], u_x[sub], u_y[sub],
        scale=8.0, width=0.0035, color="0.2", alpha=0.85,
    )
    _overlay_fault(ax, snapshot)
    ax.set_title("velocity field (arrows + |u| heatmap)")
    ax.set_aspect("equal")

    # ---- Top-right: |ε̇|_II ---------------------------------------------
    ax = axes[0, 1]
    cax = ax.tricontourf(tri, snapshot["edot_II"], levels=20, cmap="viridis")
    fig.colorbar(cax, ax=ax, fraction=0.040, pad=0.02)
    _overlay_fault(ax, snapshot)
    ax.set_title(r"$|\dot\varepsilon|_{II}$ (strain-rate 2nd invariant)")
    ax.set_aspect("equal")

    # ---- Bottom-left: |σ|_II with τ_y(x) contour ------------------------
    ax = axes[1, 0]
    cax = ax.tricontourf(tri, snapshot["sigma_II"], levels=20, cmap="magma")
    fig.colorbar(cax, ax=ax, fraction=0.040, pad=0.02)
    # Contour the τ_y(x) field at a few values to show the fault
    tri_full = Triangulation(sx, sy)
    ax.tricontour(
        tri_full, snapshot["ty_field"],
        levels=[0.5, 1.0, 5.0, 50.0], colors="cyan", linewidths=0.7, alpha=0.7,
    )
    _overlay_fault(ax, snapshot, color="white")
    ax.set_title(r"$|\sigma|_{II}$ — cyan: $\tau_y(x)$ contours (0.5, 1, 5, 50)")
    ax.set_aspect("equal")

    # ---- Bottom-right: yield ratio σ_II / τ_y(x) ------------------------
    ax = axes[1, 1]
    ratio = snapshot["yield_ratio"]
    levels = np.linspace(0, 1.2, 25)
    cax = ax.tricontourf(tri, np.clip(ratio, 0, 1.2), levels=levels, cmap="RdYlGn_r")
    fig.colorbar(cax, ax=ax, fraction=0.040, pad=0.02, label=r"$|\sigma|_{II}/\tau_y(x)$")
    ax.tricontour(tri, ratio, levels=[1.0], colors="black", linewidths=1.2)
    _overlay_fault(ax, snapshot)
    ax.set_title(r"yield activation: $|\sigma|_{II}/\tau_y(x)$  (black contour at 1)")
    ax.set_aspect("equal")

    fig.suptitle(
        f"ETD-2 fields at yield-active step "
        f"(t={snapshot['t']:.2f}, V_top={snapshot['v_top']:+.3f}) — "
        f"θ={th:+.0f}°, fault τ_y={ty_fault}, "
        f"max |σ_II/τ_y| = {ratio.max():.3f}",
        fontsize=12, y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def _overlay_fault(ax, snap, color="red"):
    """Draw the fault line and the 1·FAULT_WIDTH band."""
    n_x, n_y = snap["n_vec"]
    cx, cy = 0.5 * W, 0.5 * H
    # Fault segment endpoints (length 0.6 like FAULT_LENGTH in the bench)
    L = 0.6
    t_x, t_y = n_y, -n_x  # tangent
    p1 = (cx - 0.5 * L * t_x, cy - 0.5 * L * t_y)
    p2 = (cx + 0.5 * L * t_x, cy + 0.5 * L * t_y)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=1.5, alpha=0.85)


def main():
    os.makedirs("output", exist_ok=True)
    cases = [(0.0, 0.15), (15.0, 0.15)]
    for theta, ty in cases:
        print(f"\n=== θ={theta:+.0f}°, τ_y={ty:.2f} ===", flush=True)
        t0 = time.time()
        snap = run_capture_at_yield_peak(theta, ty, n_periods=2)
        print(f"  ran in {time.time()-t0:.1f}s, max per-node yield ratio = {snap['yield_ratio'].max():.3f}")
        out = f"output/exp_integrator_phase_b_fields_th{theta:+.0f}_ty{ty:.2f}".replace(".", "p") + ".png"
        plot_one(snap, out)


if __name__ == "__main__":
    main()
