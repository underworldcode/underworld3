"""Plot Phase G v3 comparison: baseline vs lag (vs predictor when available).

Reads per-step trace files (text format, written each step) so the
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


def _load_trace(label):
    path = os.path.join(TRACE_DIR, f"_phase_g_{label}.trace.txt")
    if not os.path.exists(path):
        return None
    rows = np.loadtxt(path, comments="#")
    if rows.size == 0:
        return None
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    # columns: step, t, V_top, snes, sigma_eq_max, u_y_max,
    #          eta_lag_min, eta_lag_max, yielded_fraction
    return dict(
        step=rows[:, 0],
        t=rows[:, 1],
        V=rows[:, 2],
        snes=rows[:, 3],
        sigma_eq=rows[:, 4],
        u_y_max=rows[:, 5],
        eta_min=rows[:, 6],
        eta_max=rows[:, 7],
        yielded=rows[:, 8],
    )


def main():
    cases = [
        ("v3_baseline_const_eta", "BDF-1 baseline (in-residual elasticity)",  "#1f77b4", "-"),
        ("v5b_bdf1",              "v5b BDF-1 (yield-on-total)",              "#2ca02c", "-"),
        ("v5b_etd1",              "v5b ETD-1",                                "#ff7f0e", ":"),
        ("v5b_bdf2",              "v5b BDF-2 (no damping)",                  "#9467bd", "--"),
        ("v5b_bdf2_blend25",      "v5b BDF-2 (bdf_blend=0.25)",              "#9467bd", "-"),
        ("v5b_etd2",              "v5b ETD-2 (no damping)",                  "#d62728", "--"),
        ("v5b_etd2_blend25",      "v5b ETD-2 (etd_blend=0.25)",              "#d62728", "-"),
    ]
    traces = {label: _load_trace(label) for label, _, _, _ in cases}

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    # Panel 1 — σ_eq_max (drop runaway tail: any step where σ > 5·baseline_peak)
    ax = axes[0]
    baseline_tr = traces.get("v3_baseline_const_eta")
    sigma_clip = 5 * float(baseline_tr["sigma_eq"].max()) if baseline_tr is not None else 50.0
    for label, name, color, ls in cases:
        tr = traces[label]
        if tr is None:
            continue
        # Clip data after the first runaway step for readability
        runaway_idx = np.argmax(tr["sigma_eq"] > sigma_clip)
        if tr["sigma_eq"][runaway_idx] > sigma_clip and runaway_idx > 0:
            t_plot = tr["t"][:runaway_idx] / PERIOD
            s_plot = tr["sigma_eq"][:runaway_idx]
            tail_note = f", runaway @ step {int(tr['step'][runaway_idx])}"
        else:
            t_plot = tr["t"] / PERIOD
            s_plot = tr["sigma_eq"]
            tail_note = f", {int(tr['step'][-1])} steps"
        ax.plot(t_plot, s_plot, ls, color=color,
                label=f"{name} (peak={s_plot.max():.3f}{tail_note})")
    ax.set_ylabel(r"max $|\sigma|_{eq}$")
    ax.set_ylim(0, sigma_clip * 1.05)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title(
        "Phase G v3 isotropic VEP harmonic, RES=32, "
        "fault τ_y=0.05, bulk τ_y=200, θ=15°"
    )

    # Panel 2 — |u_y|_max (log, clipped to 5x baseline for readability)
    ax = axes[1]
    baseline_uy = float(baseline_tr["u_y_max"].max()) if baseline_tr is not None else 1.0
    uy_clip = 5 * baseline_uy
    for label, name, color, ls in cases:
        tr = traces[label]
        if tr is None:
            continue
        runaway_idx = np.argmax(tr["u_y_max"] > uy_clip)
        if tr["u_y_max"][runaway_idx] > uy_clip and runaway_idx > 0:
            t_plot = tr["t"][:runaway_idx] / PERIOD
            u_plot = tr["u_y_max"][:runaway_idx]
        else:
            t_plot = tr["t"] / PERIOD
            u_plot = tr["u_y_max"]
        ax.semilogy(t_plot, u_plot, ls, color=color, label=name)
    ax.set_ylabel(r"max $|u_y|$ (log, clipped)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # Panel 3 — SNES iters
    ax = axes[2]
    for label, name, color, ls in cases:
        tr = traces[label]
        if tr is None:
            continue
        ax.plot(tr["t"] / PERIOD, tr["snes"], ls, color=color,
                label=f"{name} (mean={tr['snes'].mean():.1f})")
    ax.set_ylabel("SNES iters")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 4 — η_lag range and yielded fraction
    ax = axes[3]
    ax_y = ax.twinx()
    for label, name, color, ls in cases:
        tr = traces[label]
        if tr is None:
            continue
        ax.semilogy(tr["t"] / PERIOD, tr["eta_min"], ls, color=color,
                    label=f"{name} η_min")
        ax.semilogy(tr["t"] / PERIOD, tr["eta_max"], ls, color=color,
                    alpha=0.4, label=f"{name} η_max")
        ax_y.plot(tr["t"] / PERIOD, tr["yielded"] * 100,
                  ":", color=color, alpha=0.7,
                  label=f"{name} yielded%")
    ax.set_ylabel(r"η_lag range (log)")
    ax_y.set_ylabel(r"yielded %", color="grey")
    ax.set_xlabel(r"time $t / T$ (periods)")
    ax.legend(loc="lower left", fontsize=7)
    ax_y.legend(loc="lower right", fontsize=7)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "phase_g_v3_comparison.png")
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
