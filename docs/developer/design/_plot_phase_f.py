"""Plot Phase F results — predictor-corrector experiments on isotropic VEP.

Reads the per-step trace files (text format, written each step) so the
plot reproduces from a fresh clone if the npz files are absent.
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "output"
TRACE_DIR = "docs/developer/design"
DT = 0.05
OMEGA = np.pi / 2.0
PERIOD = 2.0 * np.pi / OMEGA
TAU_Y_FAULT = 0.05


def _load_trace(label):
    path = os.path.join(TRACE_DIR, f"_phase_f_{label}.trace.txt")
    if not os.path.exists(path):
        return None
    rows = np.loadtxt(path, comments="#")
    if rows.size == 0:
        return None
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    # Columns: step, t, V_top, snes_iters, picard_iters, sigma_eq_max,
    # sigma_eq_max_after_correction, u_y_max, yielded_fraction
    return dict(
        step=rows[:, 0],
        t=rows[:, 1],
        V=rows[:, 2],
        snes_iters=rows[:, 3],
        picard_iters=rows[:, 4],
        sigma_eq_max=rows[:, 5],
        sigma_eq_max_corrected=rows[:, 6],
        u_y_max=rows[:, 7],
        yielded_fraction=rows[:, 8],
    )


def main():
    cases = [
        ("bdf1_iso",       "BDF-1 (yield-in-residual)", "#1f77b4", "-"),
        ("etd1_pc1",       "ETD-1 + RR single-shot",   "#2ca02c", "-"),
        ("etd1_pc_picard", "ETD-1 + RR + Picard",       "#17becf", "--"),
        ("etd2_pc1",       "ETD-2 + RR single-shot",   "#ff7f0e", "-"),
        ("etd2_pc_picard", "ETD-2 + RR + Picard",       "#d62728", ":"),
    ]
    traces = {label: _load_trace(label) for label, _, _, _ in cases}

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Panel 1 — σ_eq_max (log)
    ax = axes[0]
    for label, name, color, ls in cases:
        tr = traces[label]
        if tr is None:
            continue
        ax.semilogy(tr["t"] / PERIOD, tr["sigma_eq_max"], ls, color=color,
                    label=f"{name} (peak={tr['sigma_eq_max'].max():.3f})")
    ax.axhline(TAU_Y_FAULT, color="#888888", lw=0.7, linestyle="--",
               label=rf"$\tau_y^{{fault}}={TAU_Y_FAULT}$")
    ax.set_ylabel(r"max $|\sigma|_{eq}$ (log)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax.set_title(
        rf"Phase F: predictor-corrector on isotropic VEP, "
        rf"localised weak zone (τ_y_fault={TAU_Y_FAULT}, τ_y_bulk=200)"
    )

    # Panel 2 — |u_y|_max (log)
    ax = axes[1]
    for label, name, color, ls in cases:
        tr = traces[label]
        if tr is None:
            continue
        ax.semilogy(tr["t"] / PERIOD, tr["u_y_max"], ls, color=color,
                    label=f"{name} (peak={tr['u_y_max'].max():.3e})")
    ax.set_ylabel(r"max $|u_y|$ (log)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # Panel 3 — yielded fraction
    ax = axes[2]
    for label, name, color, ls in cases:
        tr = traces[label]
        if tr is None:
            continue
        ax.plot(tr["t"] / PERIOD, tr["yielded_fraction"] * 100, ls, color=color,
                label=name)
    ax.set_ylabel(r"yielded fraction (%)")
    ax.set_xlabel(r"time $t / T$ (periods)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "exp_integrator_phase_f_predictor_corrector.png")
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
