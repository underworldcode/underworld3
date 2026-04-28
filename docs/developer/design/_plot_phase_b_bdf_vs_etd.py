"""Plot BDF-1 vs ETD-2 (lumped) vs split-ETD-2 trajectories at
τ_y=0.05, θ=+15°.

Loads time series saved by ``_phase_b_bdf_vs_etd_at_tight_yield.py``
(BDF-1, lumped ETD-2) and ``_phase_d_killer_split.py`` (split ETD-2);
produces a 3-panel figure on shared time axes:

  1. centre σ_xy(t)
  2. global max |σ|_II(t)
  3. global max |u_y|(t)

τ_y=0.05 reference lines are drawn on panels 1 and 2.  The split
trace is expected to sit inside the lumped runaway and ideally close
to the BDF-1 baseline.

Run::

    pixi run -e amr-dev python -u docs/developer/design/_plot_phase_b_bdf_vs_etd.py
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OMEGA = np.pi / 2.0
DT = 0.05
THETA = 15.0
TAU_Y = 0.05
OUT_DIR = "output"


def _path(integrator):
    return os.path.join(
        OUT_DIR,
        f"phase_b_{integrator}_th{THETA:+.0f}_ty{TAU_Y:.2f}".replace(".", "p") + ".npz",
    )


def _load(integrator):
    path = _path(integrator)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"missing trajectory cache {path} — run "
            f"_phase_b_bdf_vs_etd_at_tight_yield.py first"
        )
    return np.load(path)


def main():
    bdf = _load("bdf")
    etd = _load("etd")
    try:
        split = _load("etd-split")
    except FileNotFoundError:
        split = None
        print("  (no split-ETD trajectory cached — skipping)", flush=True)

    n_bdf = int(bdf["n_steps"])
    n_etd = int(etd["n_steps"])
    t_bdf = (np.arange(n_bdf) + 1) * DT
    t_etd = (np.arange(n_etd) + 1) * DT
    period = 2.0 * np.pi / OMEGA
    if split is not None:
        n_split = int(split["n_steps"])
        t_split = (np.arange(n_split) + 1) * DT

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.5), sharex=True)

    # Panel 1 — centre σ_xy(t)
    ax = axes[0]
    ax.plot(t_bdf / period, bdf["sigma_xy_centre"], "-", color="#1f77b4",
            label=f"BDF-1 (peak |σ_xy|={np.abs(bdf['sigma_xy_centre']).max():.3f})")
    ax.plot(t_etd / period, etd["sigma_xy_centre"], "-", color="#d62728",
            label=f"ETD-2 lumped (peak |σ_xy|={np.abs(etd['sigma_xy_centre']).max():.3f})")
    if split is not None:
        ax.plot(t_split / period, split["sigma_xy_centre"], "-", color="#2ca02c",
                label=f"ETD-2 split (peak |σ_xy|={np.abs(split['sigma_xy_centre']).max():.3f})")
    ax.axhline(+TAU_Y, color="#888888", lw=0.8, linestyle="--")
    ax.axhline(-TAU_Y, color="#888888", lw=0.8, linestyle="--",
               label=rf"$\pm\tau_y={TAU_Y}$")
    ax.set_ylabel(r"centre $\sigma_{xy}$")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2 — global max |σ|_II(t)
    ax = axes[1]
    ax.semilogy(t_bdf / period, np.abs(bdf["sigma_II_max_per_step"]),
                "-", color="#1f77b4",
                label=f"BDF-1 (peak={bdf['sigma_II_max_per_step'].max():.3f})")
    ax.semilogy(t_etd / period, np.abs(etd["sigma_II_max_per_step"]),
                "-", color="#d62728",
                label=f"ETD-2 lumped (peak={etd['sigma_II_max_per_step'].max():.3f})")
    if split is not None:
        ax.semilogy(t_split / period, np.abs(split["sigma_II_max_per_step"]),
                    "-", color="#2ca02c",
                    label=f"ETD-2 split (peak={split['sigma_II_max_per_step'].max():.3f})")
    ax.axhline(TAU_Y, color="#888888", lw=0.8, linestyle="--",
               label=rf"$\tau_y={TAU_Y}$")
    ax.set_ylabel(r"max $|\sigma|_{II}$ (log)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3, which="both")

    # Panel 3 — global max |u_y|(t)
    ax = axes[2]
    ax.semilogy(t_bdf / period, bdf["u_y_max_per_step"],
                "-", color="#1f77b4",
                label=f"BDF-1 (peak={bdf['u_y_max_per_step'].max():.3f})")
    ax.semilogy(t_etd / period, etd["u_y_max_per_step"],
                "-", color="#d62728",
                label=f"ETD-2 lumped (peak={etd['u_y_max_per_step'].max():.3f})")
    if split is not None:
        ax.semilogy(t_split / period, split["u_y_max_per_step"],
                    "-", color="#2ca02c",
                    label=f"ETD-2 split (peak={split['u_y_max_per_step'].max():.3f})")
    ax.set_ylabel(r"max $|u_y|$ (log)")
    ax.set_xlabel(r"time $t / T$ (periods)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        rf"BDF-1 vs ETD-2 lumped vs ETD-2 split at $\theta=+15^\circ$, $\tau_y={TAU_Y}$, RES=32",
        y=0.995,
    )
    fig.tight_layout()

    out_png = os.path.join(OUT_DIR, "exp_integrator_phase_b_bdf_vs_etd.png")
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
