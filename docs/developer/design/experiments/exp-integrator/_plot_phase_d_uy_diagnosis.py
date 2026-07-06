"""Diagnose: is split-ETD's |u_y| growing without bound, or settling?

Plots BDF-1 / lumped-ETD / split-ETD u_y(t) and σ_∥(t) on shared time
axes — answers whether the 16× higher |u_y| peak in split-ETD is a
stable accumulation matching the elastic-loading/plastic-yielding
cycle, or unbounded drift.
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DT = 0.05
OMEGA = np.pi / 2.0
PERIOD = 2.0 * np.pi / OMEGA
TAU_Y = 0.05
OUT_DIR = "output"


def main():
    bdf = np.load(os.path.join(OUT_DIR, "phase_b_bdf_th+15_ty0p05.npz"))
    etd = np.load(os.path.join(OUT_DIR, "phase_b_etd_th+15_ty0p05.npz"))
    split = np.load(os.path.join(OUT_DIR, "phase_b_etd-split_th+15_ty0p05.npz"))

    t_b = (np.arange(int(bdf["n_steps"])) + 1) * DT / PERIOD
    t_e = (np.arange(int(etd["n_steps"])) + 1) * DT / PERIOD
    t_s = (np.arange(int(split["n_steps"])) + 1) * DT / PERIOD

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax = axes[0]
    ax.plot(t_b, bdf["u_y_max_per_step"], "-", color="#1f77b4", label="BDF-1")
    ax.plot(t_e, etd["u_y_max_per_step"], "-", color="#d62728",
            label="ETD lumped", alpha=0.5)
    ax.plot(t_s, split["u_y_max_per_step"], "-", color="#2ca02c",
            label="split (explicit-parallel)")
    ax.set_yscale("log")
    ax.set_ylabel(r"max $|u_y|$  (log)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, which="both")
    ax.set_title(
        r"split-ETD vs BDF-1 / lumped-ETD: $|u_y|$ and $|\sigma_\parallel|$  "
        r"($\tau_y=0.05$, $\theta=+15^\circ$)"
    )

    ax = axes[1]
    ax.plot(t_b, bdf["sigma_par_centre"], "-", color="#1f77b4", label="BDF-1")
    ax.plot(t_e, etd["sigma_par_centre"], "-", color="#d62728",
            label="ETD lumped", alpha=0.5)
    ax.plot(t_s, split["sigma_par_centre"], "-", color="#2ca02c",
            label="split (explicit-parallel)")
    ax.axhline(TAU_Y, color="black", lw=0.7, linestyle="--",
               label=rf"$\tau_y={TAU_Y}$")
    ax.set_xlabel(r"time $t/T$ (periods)")
    ax.set_ylabel(r"centre $|\sigma_\parallel|$")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "exp_integrator_phase_d_uy_diagnosis.png")
    fig.savefig(out_png, dpi=140)
    print(f"  wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
