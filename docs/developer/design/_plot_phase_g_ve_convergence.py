"""Phase G v5b convergence plot: error vs Δt for each integrator config.

Loads ``output/phase_g_ve_convergence.npz`` and produces:
  1. Log-log error (RMS) vs Δt for all configs
  2. Slope estimates (effective convergence order) per config
  3. Reference slope-1 and slope-2 lines for visual comparison

Expectations:
  - BDF-1, ETD-1: slope 1 (first-order)
  - ETD-2 (full): slope 2 (second-order)
  - ETD-2 + blend < 1: somewhere between 1 and 2
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "output"


def main():
    npz_path = os.path.join(OUT_DIR, "phase_g_ve_convergence.npz")
    if not os.path.exists(npz_path):
        print(f"  not found: {npz_path}  (run _phase_g_ve_convergence.py first)")
        return
    data = np.load(npz_path)

    cases = [
        ("bdf1",     "BDF-1",                  "#1f77b4", "-",  "o"),
        ("etd1",     "ETD-1",                  "#ff7f0e", "-",  "s"),
        ("etd2_b25", "ETD-2 (etd_blend=0.25)", "#fb9a99", ":",  "v"),
        ("etd2_b50", "ETD-2 (etd_blend=0.50)", "#e41a1c", "--", "^"),
        ("etd2_b75", "ETD-2 (etd_blend=0.75)", "#cb181d", "-.", "<"),
        ("etd2",     "ETD-2 (full, β=1.0)",    "#67000d", "-",  "D"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel 1: log-log RMS error vs Δt
    for label, name, color, ls, marker in cases:
        dt_key = f"{label}_dt"
        rms_key = f"{label}_rms_err"
        if dt_key not in data:
            continue
        dt_arr = data[dt_key]
        rms_arr = data[rms_key]
        if len(dt_arr) == 0:
            continue
        ax1.loglog(dt_arr, rms_arr, marker=marker, color=color, ls=ls,
                    label=name, lw=1.5, ms=8)
    # Reference slopes
    ax1_dt = np.array([0.005, 0.2])
    ax1_e1 = 0.02 * (ax1_dt / ax1_dt[1])
    ax1_e2 = 0.001 * (ax1_dt / ax1_dt[1])**2
    ax1.loglog(ax1_dt, ax1_e1, 'k:', lw=0.7, alpha=0.5,
                label=r'slope 1 (1st order)')
    ax1.loglog(ax1_dt, ax1_e2, 'k--', lw=0.7, alpha=0.5,
                label=r'slope 2 (2nd order)')
    ax1.set_xlabel(r"$\Delta t$")
    ax1.set_ylabel(r"RMS error vs analytical")
    ax1.set_title(r"Phase G v5b VE-only convergence "
                   r"(σ(0)=0, γ̇(t)=γ̇₀sin(ωt), 4 periods)")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    # Panel 2: empirical convergence order vs blend value
    ax2_blend = []
    ax2_order = []
    ax2_label = []
    ax2_color = []
    for label, name, color, ls, marker in cases:
        dt_key = f"{label}_dt"
        rms_key = f"{label}_rms_err"
        if dt_key not in data:
            continue
        dt_arr = data[dt_key]
        rms_arr = data[rms_key]
        if len(dt_arr) < 2:
            continue
        # Fit slope on log-log: log(rms) = order * log(dt) + const
        log_dt = np.log(dt_arr)
        log_rms = np.log(rms_arr)
        slope = float(np.polyfit(log_dt, log_rms, 1)[0])
        # Determine blend value for plotting
        if "b25" in label:
            beta = 0.25
        elif "b50" in label:
            beta = 0.50
        elif "b75" in label:
            beta = 0.75
        elif "etd2" in label and "b" not in label:
            beta = 1.0
        elif "etd1" in label:
            beta = 0.0
        elif "bdf1" in label:
            beta = -0.05  # plot offset to distinguish from ETD-1
        else:
            beta = None
        if beta is None:
            continue
        ax2_blend.append(beta)
        ax2_order.append(slope)
        ax2_label.append(name)
        ax2_color.append(color)

    if ax2_blend:
        ax2.scatter(ax2_blend, ax2_order, c=ax2_color, s=120,
                     edgecolor='black', zorder=3)
        for b, o, n in zip(ax2_blend, ax2_order, ax2_label):
            ax2.annotate(n, (b, o), xytext=(8, 0),
                          textcoords='offset points', fontsize=8, va='center')
        # Reference lines: order 1 and 2
        ax2.axhline(1.0, color='gray', ls=':', alpha=0.5, label='1st order')
        ax2.axhline(2.0, color='gray', ls='--', alpha=0.5, label='2nd order')
    ax2.set_xlabel(r"etd_blend (β) — fraction of ETD-2 retained")
    ax2.set_ylabel(r"empirical convergence order (slope on log-log)")
    ax2.set_title(r"Convergence order vs blend factor")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(-0.15, 1.15)
    ax2.set_ylim(0.5, 2.5)

    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "phase_g_ve_convergence.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}", flush=True)
    print("  Empirical convergence orders:")
    for b, o, n in zip(ax2_blend, ax2_order, ax2_label):
        print(f"    {n:30s}: {o:.3f}")


if __name__ == "__main__":
    main()
