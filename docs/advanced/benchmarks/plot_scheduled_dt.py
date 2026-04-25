"""Plot the scheduled-dt experiment results.

Two figures (one for VE, one for VEP min), each with three panels:
  - top:    sigma_xy(t) for all 4 configs vs analytical (VE only)
  - middle: BDF order used per step (1 vs 2)
  - bottom: dt(t)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def step_square_wave_stress(t, eta, mu, gamma_dot_0, half_period):
    t_r = eta / mu
    sigma_ss = eta * gamma_dot_0
    out = np.zeros_like(t)
    for i, ti in enumerate(t):
        n = int(ti / half_period)
        t_local = ti - n * half_period
        sigma_start = 0.0
        for j in range(n):
            sign = 1.0 if j % 2 == 0 else -1.0
            sigma_target = sign * sigma_ss
            sigma_start = sigma_target + (sigma_start - sigma_target) * np.exp(-half_period / t_r)
        sign = 1.0 if n % 2 == 0 else -1.0
        sigma_target = sign * sigma_ss
        out[i] = sigma_target + (sigma_start - sigma_target) * np.exp(-t_local / t_r)
    return out


C_FIX_MAX = "#9467BD"   # purple    fixed coarse
C_FIX_MIN = "#1F77B4"   # blue      fixed fine
C_CURR    = "#D62728"   # red       scheduled, current code
C_EXPT    = "#2CA02C"   # green     scheduled + BDF-1 restart experiment


def plot_panel(prefix, ana=None, tau_y=None, title="", outpath=""):
    fixed_max  = np.load(f"output/scheduled_dt/{prefix}_fixed_max.npz")
    fixed_min  = np.load(f"output/scheduled_dt/{prefix}_fixed_min.npz")
    sched_curr = np.load(f"output/scheduled_dt/{prefix}_sched_current.npz")
    sched_expt = np.load(f"output/scheduled_dt/{prefix}_sched_experiment.npz")

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1.5]})

    # === Top: stress trace
    ax = axes[0]
    if ana is not None:
        t_ana = np.linspace(0, max(sched_expt["times"][-1], 8.0), 4000)
        ax.plot(t_ana, ana(t_ana), "-", color="black", lw=1.0, alpha=0.5,
                label="VE analytical")
    if tau_y is not None:
        ax.axhline(tau_y, color="grey", ls="--", lw=0.8, alpha=0.6,
                   label=fr"$\pm\tau_y$")
        ax.axhline(-tau_y, color="grey", ls="--", lw=0.8, alpha=0.6)

    for tr, col, mk, lab in [
        (fixed_max,  C_FIX_MAX, "o",
         f"fixed dt_max=0.20 (peak={float(tr_peak(fixed_max)):.4f}{tr_extra(fixed_max)})"),
        (fixed_min,  C_FIX_MIN, "s",
         f"fixed dt_min=0.10 (peak={float(tr_peak(fixed_min)):.4f}{tr_extra(fixed_min)})"),
        (sched_curr, C_CURR,    "^",
         f"scheduled, current code     (peak={float(tr_peak(sched_curr)):.4f}{tr_extra(sched_curr)})"),
        (sched_expt, C_EXPT,    "d",
         f"scheduled, BDF-1 restart    (peak={float(tr_peak(sched_expt)):.4f}{tr_extra(sched_expt)})"),
    ]:
        ax.plot(tr["times"], tr["stress"], "-", color=col, lw=0.8,
                marker=mk, ms=4, alpha=0.85, label=lab)
    ax.axhline(0, color="grey", lw=0.5, alpha=0.4)
    ax.set_ylabel(r"$\sigma_{xy}$")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
    ax.grid(True, alpha=0.3)

    # === Middle: BDF order
    ax = axes[1]
    for tr, col, lab, yshift in [
        (fixed_max,  C_FIX_MAX, "fix max",   0.30),
        (fixed_min,  C_FIX_MIN, "fix min",   0.10),
        (sched_curr, C_CURR,    "current",  -0.10),
        (sched_expt, C_EXPT,    "BDF-1 rst", -0.30),
    ]:
        y = tr["orders"].astype(float) + yshift
        ax.plot(tr["times"], y, "o", color=col, ms=3.5, alpha=0.85, label=lab)
    ax.set_yticks([1, 2])
    ax.set_ylim(0.5, 2.5)
    ax.set_ylabel("BDF order")
    ax.legend(loc="center right", fontsize=8, ncol=4, framealpha=0.92)
    ax.grid(True, alpha=0.3)

    # === Bottom: dt
    ax = axes[2]
    for tr, col, lab in [
        (fixed_max,  C_FIX_MAX, "fix max"),
        (fixed_min,  C_FIX_MIN, "fix min"),
        (sched_curr, C_CURR,    "scheduled, current code"),
        (sched_expt, C_EXPT,    "scheduled, BDF-1 restart"),
    ]:
        ax.step(tr["times"], tr["dts"], where="post", color=col, lw=1.2,
                label=lab, alpha=0.9)
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"$dt$")
    ax.set_yscale("log")
    ax.set_ylim(0.05, 0.4)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}", flush=True)


def tr_peak(tr):
    return np.abs(tr["stress"]).max()


def tr_extra(tr):
    """Extra summary string per trace."""
    if "max_err" in tr.files and not np.isnan(float(tr["max_err"])):
        return f", err={float(tr['max_err']):.4f}"
    if "overshoots" in tr.files:
        return f", over={int(tr['overshoots'])}"
    return ""


if __name__ == "__main__":
    os.makedirs("docs/advanced/figures", exist_ok=True)

    # Pure VE
    def ana(t):
        return step_square_wave_stress(np.asarray(t), 1.0, 1.0, 1.0, 2.0)
    plot_panel(
        "ve", ana=ana, tau_y=None,
        title="Pure VE under step-change BC: scheduled BDF-1 restart at dt-change",
        outpath="docs/advanced/figures/scheduled_dt_handoff_VE.png",
    )

    # VEP min
    plot_panel(
        "vep", ana=None, tau_y=0.5,
        title=r"VEP min ($\tau_y = 0.5$) under step-change BC: scheduled BDF-1 restart at dt-change",
        outpath="docs/advanced/figures/scheduled_dt_handoff_VEP.png",
    )
