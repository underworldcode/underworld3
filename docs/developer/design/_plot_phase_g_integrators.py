"""Phase G v5b integrator comparison: time-to-blowup + physics validation.

Two-panel plot:
  1. σ_eq(t) trajectories overlaid — physics validation. Each variant
     should track baseline-BDF-1 if the architecture is integrator-
     agnostic.
  2. Cumulative SNES iterations + step-of-blowup markers — time-to-
     blowup signature. v5b variants should run further than their
     standard-yield-in-residual counterparts.

Reads per-step trace files. Drops trajectories at runaway/divergence
for cleanliness.
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


def _truncate_at_blowup(tr, sigma_thresh=15.0, uy_thresh=5.0):
    """Find first step where σ or |u_y| exceeds threshold; clip there."""
    if tr is None:
        return None
    bad = (tr["sigma_eq"] > sigma_thresh) | (tr["u_y_max"] > uy_thresh)
    if bad.any():
        i_blow = int(np.argmax(bad))
        return {k: v[:i_blow] if hasattr(v, '__len__') else v for k, v in tr.items()}, int(tr["step"][i_blow])
    return tr, None


def main():
    cases = [
        ("v3_baseline_const_eta", "BDF-1 baseline",                "#1f77b4", "-",  3.0),
        ("v5b_bdf1",              "v5b BDF-1",                     "#2ca02c", "-",  2.5),
        ("v5b_etd1",              "v5b ETD-1",                     "#ff7f0e", ":",  2.0),
        ("v5b_bdf2",              "v5b BDF-2 (no damping)",        "#9467bd", "--", 2.0),
        ("v5b_bdf2_blend25",      "v5b BDF-2 (bdf_blend=0.25)",    "#9467bd", "-",  1.5),
        ("v5b_etd2",              "v5b ETD-2 (no damping)",        "#d62728", "--", 2.0),
        ("v5b_etd2_blend25",      "v5b ETD-2 (etd_blend=0.25)",    "#d62728", "-",  1.5),
    ]
    traces_raw = {label: _load_trace(label) for label, _, _, _, _ in cases}

    # Determine sigma threshold for "blowup" — 10x baseline peak.
    # 10x is permissive: damped variants oscillate higher than baseline
    # but stay bounded; we want to flag genuine runaway, not transient
    # peaks of damped runs.
    baseline_tr = traces_raw.get("v3_baseline_const_eta")
    sigma_thresh = 10.0 * float(baseline_tr["sigma_eq"].max()) if baseline_tr is not None else 15.0

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 1, hspace=0.30)

    # Panel 1: σ trajectory + V_top driving
    ax = fig.add_subplot(gs[0])
    ax2 = ax.twinx()
    if baseline_tr is not None:
        ax2.fill_between(baseline_tr["t"] / PERIOD, 0, baseline_tr["V"],
                         color="lightblue", alpha=0.4, label=r"$V_{top}$ driving")
        ax2.set_ylabel(r"$V_{top}$", color="steelblue", fontsize=10)
        ax2.tick_params(axis='y', labelcolor='steelblue')

    blowup_steps = {}
    for label, name, color, ls, _ in cases:
        tr_raw = traces_raw[label]
        if tr_raw is None:
            continue
        tr, blowup_step = _truncate_at_blowup(tr_raw,
                                                sigma_thresh=sigma_thresh,
                                                uy_thresh=2.0)
        if blowup_step is not None:
            blowup_steps[label] = blowup_step
            label_str = f"{name} (blew up @ step {blowup_step})"
        else:
            label_str = f"{name} ({int(tr_raw['step'][-1])} steps)"
        ax.plot(tr["t"] / PERIOD, tr["sigma_eq"], ls, color=color,
                label=label_str, lw=1.5)
    ax.set_ylabel(r"max $|\sigma|_{eq}$")
    ax.set_xlabel(r"time $t/T$ (forcing periods)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    if ax2:
        ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_title("Phase G v5b: σ trajectory across integrators "
                 "(yield-on-total; truncated at blowup)")

    # Panel 2: SNES iterations / step
    ax = fig.add_subplot(gs[1])
    for label, name, color, ls, _ in cases:
        tr_raw = traces_raw[label]
        if tr_raw is None:
            continue
        tr, _ = _truncate_at_blowup(tr_raw, sigma_thresh=sigma_thresh, uy_thresh=2.0)
        ax.plot(tr["t"] / PERIOD, tr["snes"], ls, color=color,
                label=f"{name} (mean={tr_raw['snes'].mean():.1f})",
                lw=1.0, alpha=0.8)
    ax.set_ylabel(r"SNES iterations / step")
    ax.set_xlabel(r"time $t/T$")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("Solver efficiency — per-step SNES iteration count")

    # Panel 3: cumulative SNES iterations vs steps completed (time-to-blowup)
    ax = fig.add_subplot(gs[2])
    for label, name, color, ls, _ in cases:
        tr_raw = traces_raw[label]
        if tr_raw is None:
            continue
        tr, blowup_step = _truncate_at_blowup(tr_raw,
                                                sigma_thresh=sigma_thresh,
                                                uy_thresh=2.0)
        cumsum = np.cumsum(tr["snes"])
        ax.plot(tr["step"], cumsum, ls, color=color,
                label=f"{name}", lw=1.5)
        # Mark blowup point
        if blowup_step is not None:
            ax.plot(tr["step"][-1], cumsum[-1], 'X', color=color, markersize=10,
                    markeredgecolor='black', markeredgewidth=1)
    ax.set_xlabel(r"step number")
    ax.set_ylabel(r"cumulative SNES iterations")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("Time-to-blowup: cumulative SNES vs step count "
                 "(× = blowup; lower curve = more efficient)")

    out_png = os.path.join(OUT_DIR, "phase_g_integrators_comparison.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}", flush=True)
    if blowup_steps:
        print("  blowup summary:")
        for label, step in blowup_steps.items():
            print(f"    {label:30s}: step {step}", flush=True)


if __name__ == "__main__":
    main()
