"""Phase G spatial-field comparison: velocity, strain rate, stress, viscosity.

Loads ``output/phase_g_<label>_spatial.npz`` (saved at end of each run by
the corresponding runner) and renders side-by-side spatial plots for
visual physics comparison between baseline and v5 variants.
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

OUT_DIR = "output"


def _load_spatial(label, step=90):
    """Load spatial snapshot at a given step (default 90 = peak of 2nd cycle)."""
    path = os.path.join(OUT_DIR, f"phase_g_{label}_spatial_step{step:03d}.npz")
    if not os.path.exists(path):
        # Fallback: any spatial file for this label
        candidates = sorted([
            f for f in os.listdir(OUT_DIR)
            if f.startswith(f"phase_g_{label}_spatial") and f.endswith(".npz")
        ])
        if not candidates:
            return None
        path = os.path.join(OUT_DIR, candidates[-1])
    return dict(np.load(path))


def _scatter_field(ax, coords, values, cmap, vmin=None, vmax=None,
                   log_scale=False, title=""):
    norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else None
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=values,
                    cmap=cmap, s=8, marker='s',
                    vmin=None if log_scale else vmin,
                    vmax=None if log_scale else vmax,
                    norm=norm)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=9)
    # Overlay fault location (theta=15°, length 0.6, centred at 0.5,0.5)
    fault_length = 0.6
    theta = np.radians(15.0)
    cx, cy = 0.5, 0.5
    dx = 0.5 * fault_length * np.cos(theta)
    dy = 0.5 * fault_length * np.sin(theta)
    ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy],
            'w-', lw=1.5, alpha=0.7)
    ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy],
            'k--', lw=0.7, alpha=0.7)


def main():
    SNAPSHOT_STEP = 90  # mid-second-loading-cycle (peak loading, steady state)
    cases = [
        ("v3_baseline_const_eta", "BDF-1 baseline\n(in-residual elasticity)"),
        ("v5b_etd2_blend75",      "v5b ETD-2\n(etd_blend=0.75)"),
    ]
    snaps = [(label, name, _load_spatial(label, SNAPSHOT_STEP))
             for label, name in cases]
    snaps = [(l, n, s) for (l, n, s) in snaps if s is not None]
    if not snaps:
        print("No spatial snapshots found — run main runner first.")
        return
    print(f"Plotting spatial fields at step {SNAPSHOT_STEP} for "
          f"{len(snaps)} variant(s)")

    # Determine common color scales for fair comparison
    sigma_eq_max = max(s["sigma_eq"].max() for _, _, s in snaps)
    edot_max     = max(s["edot_inv_II"].max() for _, _, s in snaps)
    edot_min     = max(min(s["edot_inv_II"].min() for _, _, s in snaps), 1e-6)
    visc_max     = max(s["viscosity"].max() for _, _, s in snaps)
    visc_min     = max(min(s["viscosity"].min() for _, _, s in snaps), 1e-4)

    nrows, ncols = 4, len(snaps)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
    if ncols == 1:
        axes = axes[:, None]

    for col, (label, name, snap) in enumerate(snaps):
        # Row 0: velocity quiver (subsampled) + |u|
        ax = axes[0, col]
        u_coords = snap["u_coords"]
        u_array = snap["u_array"]
        u_mag = np.sqrt(u_array[:, 0]**2 + u_array[:, 1]**2)
        sc = ax.scatter(u_coords[:, 0], u_coords[:, 1], c=u_mag,
                        cmap="viridis", s=4, marker='o')
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        # Quiver subsample (every nth point)
        n = 8
        ax.quiver(u_coords[::n, 0], u_coords[::n, 1],
                  u_array[::n, 0], u_array[::n, 1],
                  scale=5.0, color="white", width=0.003)
        ax.set_aspect("equal")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        # Fault overlay
        fault_length = 0.6
        theta = np.radians(15.0)
        cx, cy = 0.5, 0.5
        dx = 0.5 * fault_length * np.cos(theta)
        dy = 0.5 * fault_length * np.sin(theta)
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy],
                'k--', lw=0.7, alpha=0.7)
        ax.set_title(f"{name}\n|u| (max={u_mag.max():.3e})", fontsize=9)

        # Row 1: |ε̇|_II (log)
        ax = axes[1, col]
        _scatter_field(ax, snap["sigma_coords"], snap["edot_inv_II"],
                       "magma", vmin=edot_min, vmax=edot_max,
                       log_scale=True,
                       title=rf"$|\dot{{\varepsilon}}|_{{II}}$ "
                             f"(range {snap['edot_inv_II'].min():.2e}–"
                             f"{snap['edot_inv_II'].max():.2e})")

        # Row 2: σ_eq
        ax = axes[2, col]
        _scatter_field(ax, snap["sigma_coords"], snap["sigma_eq"],
                       "plasma", vmin=0, vmax=sigma_eq_max,
                       log_scale=False,
                       title=rf"$|\sigma|_{{eq}}$ "
                             f"(max={snap['sigma_eq'].max():.3f})")

        # Row 3: viscosity (log)
        ax = axes[3, col]
        _scatter_field(ax, snap["sigma_coords"], snap["viscosity"],
                       "viridis", vmin=visc_min, vmax=visc_max,
                       log_scale=True,
                       title=rf"$\eta_{{eff}}$ "
                             f"(range {snap['viscosity'].min():.2e}–"
                             f"{snap['viscosity'].max():.2e})")

    fig.suptitle(
        "Phase G spatial fields — last-step snapshot. Fault dashed; "
        "common colour scales per row.\n"
        "Different σ-amplitude ⇒ qualitatively different stress fields.",
        fontsize=11, y=1.0
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_png = os.path.join(OUT_DIR, "phase_g_spatial_comparison.png")
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
