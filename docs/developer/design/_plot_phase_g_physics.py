"""Phase G physics comparison plots — comparing baseline vs v5 variants.

Reads per-step trace files (text format) and shows:
  1. σ_eq_max trajectory vs time (overlaid with V_top driving)
  2. σ_eq_max vs V_top — hysteresis-style phase plot (steady-state limit cycle)
  3. yielded fraction vs time
  4. σ_eq vs |u_y|_max — response signature

Per-step trace columns: step, t, V_top, snes, sigma_eq_max, u_y_max,
                        eta_lag_min, eta_lag_max, yielded_fraction
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
    return dict(
        step=rows[:, 0],
        t=rows[:, 1],
        V=rows[:, 2],
        snes=rows[:, 3],
        sigma_eq=rows[:, 4],
        u_y_max=rows[:, 5],
        yielded=rows[:, 8],
    )


def main():
    cases = [
        ("v3_baseline_const_eta", "BDF-1 baseline (in-residual elasticity)",  "#1f77b4", "-"),
        ("v5b_etd1",              "v5b ETD-1",                                "#ff7f0e", "--"),
        ("v5b_etd2_blend75",      "v5b ETD-2 (etd_blend=0.75)",              "#d62728", "-"),
    ]
    traces = {label: _load_trace(label) for label, _, _, _ in cases}

    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

    # -- Panel 1: σ_eq trajectory + V_top driving --
    ax = fig.add_subplot(gs[0, :])
    ax2 = ax.twinx()
    # Plot V_top from baseline trace (same for all)
    bl = traces.get("v3_baseline_const_eta")
    if bl is not None:
        ax2.fill_between(bl["t"] / PERIOD, 0, bl["V"], color="lightblue",
                         alpha=0.4, label=r"$V_{top}$ driving")
        ax2.set_ylabel(r"$V_{top}$ (driving)", color="steelblue")
        ax2.tick_params(axis='y', labelcolor='steelblue')
    for label, name, color, ls in cases:
        tr = traces[label]
        if tr is None:
            continue
        ax.plot(tr["t"] / PERIOD, tr["sigma_eq"], ls, color=color,
                label=f"{name}", lw=1.5)
    ax.set_ylabel(r"max $|\sigma|_{eq}$ (over domain)")
    ax.set_xlabel(r"time $t/T$ (forcing periods)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    if ax2:
        ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_title("Phase G: σ_eq response to harmonic shear loading "
                 "(RES=32, fault τ_y=0.05, bulk τ_y=200)")

    # -- Panel 2: σ vs V_top (hysteresis-style, steady-state cycle only) --
    ax = fig.add_subplot(gs[1, 0])
    for label, name, color, ls in cases:
        tr = traces[label]
        if tr is None:
            continue
        # Use only steady-state cycles (skip first cycle to remove transient)
        mask = tr["t"] / PERIOD > 1.0
        ax.plot(tr["V"][mask], tr["sigma_eq"][mask], ls, color=color,
                label=name, lw=1.0, alpha=0.7)
        # Mark start of steady state
        if mask.any():
            i0 = np.argmax(mask)
            ax.plot(tr["V"][i0], tr["sigma_eq"][i0], 'o', color=color,
                    markersize=5)
    ax.set_xlabel(r"$V_{top}$ (driving)")
    ax.set_ylabel(r"max $|\sigma|_{eq}$")
    ax.set_title("Hysteresis: σ vs V (steady-state, t/T > 1)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # -- Panel 3: σ vs |u_y|_max --
    ax = fig.add_subplot(gs[1, 1])
    for label, name, color, ls in cases:
        tr = traces[label]
        if tr is None:
            continue
        mask = tr["t"] / PERIOD > 1.0
        ax.plot(tr["u_y_max"][mask], tr["sigma_eq"][mask], ls, color=color,
                label=name, lw=1.0, alpha=0.7)
    ax.set_xlabel(r"max $|u_y|$ (response)")
    ax.set_ylabel(r"max $|\sigma|_{eq}$")
    ax.set_title("Stress vs vertical-velocity response (steady-state)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    # -- Panel 4: yielded fraction --
    ax = fig.add_subplot(gs[2, :])
    ax2 = ax.twinx()
    if bl is not None:
        ax2.fill_between(bl["t"] / PERIOD, 0, bl["V"], color="lightblue",
                         alpha=0.3)
        ax2.set_ylabel(r"$V_{top}$", color="steelblue")
        ax2.tick_params(axis='y', labelcolor='steelblue')
    for label, name, color, ls in cases:
        tr = traces[label]
        if tr is None:
            continue
        ax.plot(tr["t"] / PERIOD, tr["yielded"] * 100, ls, color=color,
                label=name, lw=1.5)
    ax.set_xlabel(r"time $t/T$")
    ax.set_ylabel(r"yielded fraction (%)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("Yield activity over the loading history")

    # -- Panel 5: SNES iterations (efficiency) --
    ax = fig.add_subplot(gs[3, :])
    for label, name, color, ls in cases:
        tr = traces[label]
        if tr is None:
            continue
        ax.plot(tr["t"] / PERIOD, tr["snes"], ls, color=color,
                label=f"{name} (mean={tr['snes'].mean():.1f})",
                lw=1.0, alpha=0.7)
    ax.set_xlabel(r"time $t/T$")
    ax.set_ylabel(r"SNES iterations / step")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("SNES iteration count — solver efficiency")

    out_png = os.path.join(OUT_DIR, "phase_g_physics_comparison.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
