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


def _load_bdf2_from_log(log_path=None):
    """Parse the per-step BDF-2 trace from the runner's stdout log.
    Returns ``(t, sigma_II, u_y, sigma_par)`` numpy arrays of the
    sparse sample points (every 5 steps after the first 5) up to the
    runaway. None if the log doesn't exist.
    """
    if log_path is None:
        # Tracked trace lives next to this script (git won't track *.log).
        candidates = [
            os.path.join(
                os.path.dirname(__file__),
                "_phase_b_bdf2_th+15_ty0p05.trace.txt",
            ),
            os.path.join(OUT_DIR, "phase_b_bdf2_th+15_ty0p05.log"),
        ]
        log_path = next((p for p in candidates if os.path.exists(p)), None)
        if log_path is None:
            return None
    elif not os.path.exists(log_path):
        return None
    import re
    pat = re.compile(
        r"step\s+(\d+)/\d+\s+t=([\d.+\-eE]+)\s+V=[\d.+\-eE]+\s+iters=\s*\d+\s+"
        r"\|σ\|_II=([\d.+\-eE]+)\s+\|u_y\|=([\d.+\-eE]+)\s+\|σ_∥\|=([\d.+\-eE]+)"
    )
    rows = []
    with open(log_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                rows.append((int(m.group(1)), float(m.group(2)),
                             float(m.group(3)), float(m.group(4)),
                             float(m.group(5))))
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])
    return (
        np.array([r[1] for r in rows]),
        np.array([r[2] for r in rows]),
        np.array([r[3] for r in rows]),
        np.array([r[4] for r in rows]),
    )


def main():
    bdf = _load("bdf")
    etd = _load("etd")
    try:
        split = _load("etd-split")
    except FileNotFoundError:
        split = None
        print("  (no split-ETD trajectory cached — skipping)", flush=True)
    try:
        hybrid = _load("hybrid")
    except FileNotFoundError:
        hybrid = None
        print("  (no hybrid trajectory cached — skipping)", flush=True)

    bdf2 = _load_bdf2_from_log()
    if bdf2 is None:
        print("  (no BDF-2 log found — skipping)", flush=True)

    n_bdf = int(bdf["n_steps"])
    n_etd = int(etd["n_steps"])
    t_bdf = (np.arange(n_bdf) + 1) * DT
    t_etd = (np.arange(n_etd) + 1) * DT
    period = 2.0 * np.pi / OMEGA
    if split is not None:
        n_split = int(split["n_steps"])
        t_split = (np.arange(n_split) + 1) * DT
    if hybrid is not None:
        n_hybrid = int(hybrid["n_steps"])
        t_hybrid = (np.arange(n_hybrid) + 1) * DT

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.5), sharex=True)

    # Panel 1 — fault-resolved |σ_∥|(t) at fault centre.
    ax = axes[0]
    if "sigma_par_centre" in bdf.files:
        ax.plot(t_bdf / period, bdf["sigma_par_centre"], "-", color="#1f77b4",
                label=f"BDF-1 (peak |σ_∥|={bdf['sigma_par_centre'].max():.3f})")
    if "sigma_par_centre" in etd.files:
        ax.plot(t_etd / period, etd["sigma_par_centre"], "-", color="#d62728",
                label=f"ETD-2 lumped (peak |σ_∥|={etd['sigma_par_centre'].max():.3f})")
    if split is not None and "sigma_par_centre" in split.files:
        ax.plot(t_split / period, split["sigma_par_centre"], "-", color="#2ca02c",
                label=f"ETD-2 split (peak |σ_∥|={split['sigma_par_centre'].max():.3f})")
    if hybrid is not None and "sigma_par_centre" in hybrid.files:
        ax.plot(t_hybrid / period, hybrid["sigma_par_centre"], "-", color="#9467bd",
                label=f"hybrid (peak |σ_∥|={hybrid['sigma_par_centre'].max():.3f})")
    if bdf2 is not None:
        t_b2, sII_b2, uy_b2, spar_b2 = bdf2
        ax.plot(t_b2 / period, spar_b2, "x-", color="#ff7f0e",
                label=f"BDF-2 → blow-up (peak |σ_∥|={spar_b2.max():.3f})")
    ax.axhline(TAU_Y, color="#888888", lw=0.8, linestyle="--",
               label=rf"$\tau_y={TAU_Y}$")
    ax.set_ylabel(r"centre $|\sigma_\parallel|$  (resolved fault shear)")
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
    if hybrid is not None:
        ax.semilogy(t_hybrid / period, np.abs(hybrid["sigma_II_max_per_step"]),
                    "-", color="#9467bd",
                    label=f"hybrid (peak={hybrid['sigma_II_max_per_step'].max():.3f})")
    if bdf2 is not None:
        t_b2, sII_b2, uy_b2, spar_b2 = bdf2
        ax.semilogy(t_b2 / period, sII_b2, "x-", color="#ff7f0e",
                    label=f"BDF-2 → blow-up (last sample={sII_b2[-1]:.2e})")
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
    if hybrid is not None:
        ax.semilogy(t_hybrid / period, hybrid["u_y_max_per_step"],
                    "-", color="#9467bd",
                    label=f"hybrid (peak={hybrid['u_y_max_per_step'].max():.3f})")
    if bdf2 is not None:
        t_b2, sII_b2, uy_b2, spar_b2 = bdf2
        ax.semilogy(t_b2 / period, uy_b2, "x-", color="#ff7f0e",
                    label=f"BDF-2 → blow-up (last sample={uy_b2[-1]:.2e})")
    ax.set_ylabel(r"max $|u_y|$ (log)")
    ax.set_xlabel(r"time $t / T$ (periods)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        rf"BDF-1 vs ETD-2 lumped/split/hybrid at $\theta=+15^\circ$, $\tau_y={TAU_Y}$, RES=32",
        y=0.995,
    )
    fig.tight_layout()

    out_png = os.path.join(OUT_DIR, "exp_integrator_phase_b_bdf_vs_etd.png")
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
