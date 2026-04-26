"""Plot the VE/VEP benchmark results from on-disk ``.npz`` files.

Reads ``output/benchmarks/{ve_harmonic,ve_square,vep_square}.npz`` and
produces three figures in ``docs/advanced/figures/``.  Style is shared
across the three so the plots can be compared directly.

Each figure has the same layout:

  Top panel:    σ_xy(t) — simulation markers, analytical solid line,
                ±τ_y guide for the VEP case, light-blue filled driving
                term γ̇(t) for context (rescaled to fit beside σ).
  Middle panel: |error| log-scale.
  Bottom panel: dt(t) (relevant once we add variable-dt benchmarks).

Run after one or more of ``bench_*.py`` have produced their npz:

    pixi run -e amr-dev python docs/advanced/benchmarks/plot_benchmarks.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _bench_helpers import load_run, FIG_DIR


# Shared style ---------------------------------------------------------
plt.rcParams.update({
    "figure.figsize": (11, 8),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "legend.framealpha": 0.92,
})

C_BDF1 = "#1F77B4"         # blue circles — BDF-1
C_BDF2 = "#D62728"         # red squares — BDF-2
C_ANA = "black"            # solid black — analytical
C_DRIVE = "#1F77B4"        # light blue — driving (filled)
C_ERR_BDF1 = "#1F77B4"
C_ERR_BDF2 = "#D62728"
C_DT = "#2CA02C"           # green — dt
C_YIELD = "grey"


def _plot_three_panel(name, t_ana_grid, sigma_ana_grid, info, *, tau_y=None):
    """Common three-panel layout used by all three benchmarks.

    Reads BDF-1 + BDF-2 traces from one npz and overlays both.  The
    npz's per-step arrays are: ``sigma_bdf1``, ``sigma_bdf2``,
    ``sigma_ana``.
    """
    arrays, params, extra = info
    times = arrays["times"]
    sigma_ana = arrays["sigma_ana"]
    dts = arrays["dts"]
    gamma_dot = arrays["gamma_dot"]
    sigma_bdf1 = arrays["sigma_bdf1"]
    sigma_bdf2 = arrays["sigma_bdf2"]

    fig, (ax_top, ax_err, ax_dt) = plt.subplots(
        3, 1, sharex=True,
        gridspec_kw={"height_ratios": [3.5, 1.5, 1.0]},
    )

    # --- Top: σ(t)
    sigma_max = float(np.max(np.abs(sigma_ana_grid))) or 1.0
    gamma_max = float(np.max(np.abs(gamma_dot))) or 1.0
    drive_scale = 0.5 * sigma_max / gamma_max
    ax_top.fill_between(
        times, 0.0, drive_scale * gamma_dot,
        color=C_DRIVE, alpha=0.18, linewidth=0,
        label=fr"driving $\dot\gamma(t)$ (×{drive_scale:.2f})",
    )
    ax_top.plot(t_ana_grid, sigma_ana_grid, "-", color=C_ANA, lw=1.4,
                label="analytical")
    ax_top.plot(times, sigma_bdf1, "o", color=C_BDF1, ms=4.2, alpha=0.78,
                mec=C_BDF1, mfc="white", mew=1.3,
                label="BDF-1")
    ax_top.plot(times, sigma_bdf2, "s", color=C_BDF2, ms=3.8, alpha=0.85,
                label="BDF-2")
    if tau_y is not None:
        ax_top.axhline(+tau_y, color=C_YIELD, ls="--", lw=0.9, alpha=0.7,
                       label=fr"$\pm\tau_y$ = $\pm${tau_y:g}")
        ax_top.axhline(-tau_y, color=C_YIELD, ls="--", lw=0.9, alpha=0.7)
    ax_top.axhline(0, color="grey", lw=0.4, alpha=0.4)
    ax_top.set_ylabel(r"$\sigma_{xy}$")

    # Title with the headline numbers per order
    bits = [name]
    if "err_max_bdf1" in extra:
        bits.append(fr"BDF-1 max|err|={extra['err_max_bdf1']:.2e}")
    if "err_max_bdf2" in extra:
        bits.append(fr"BDF-2 max|err|={extra['err_max_bdf2']:.2e}")
    if "De" in extra:
        bits.append(fr"De={extra['De']:.3f}")
    if "tau_y" in extra:
        bits.append(fr"$\tau_y={extra['tau_y']:g}$")
    ax_top.set_title("    ".join(bits))
    ax_top.legend(loc="lower right", ncol=2)

    # --- Middle: |sigma − sigma_ana| for both orders
    err1 = np.abs(sigma_bdf1 - sigma_ana)
    err2 = np.abs(sigma_bdf2 - sigma_ana)
    eps = 1e-9
    ax_err.semilogy(times, np.maximum(err1, eps), "-", color=C_ERR_BDF1,
                    lw=0.8, marker="o", ms=2.8, mec=C_ERR_BDF1, mfc="white",
                    label="BDF-1")
    ax_err.semilogy(times, np.maximum(err2, eps), "-", color=C_ERR_BDF2,
                    lw=0.8, marker="s", ms=2.8, label="BDF-2")
    ax_err.set_ylabel(r"$|\sigma_{\mathrm{sim}} - \sigma_{\mathrm{ana}}|$")
    ax_err.legend(loc="upper right", ncol=2, fontsize=8)
    ax_err.set_ylim(bottom=eps * 0.9)

    # --- Bottom: dt
    ax_dt.step(times, dts, where="post", color=C_DT, lw=1.1)
    ax_dt.set_xlabel(r"Time $t / t_r$")
    ax_dt.set_ylabel(r"$\Delta t$")
    ax_dt.set_ylim(0.0, max(dts) * 1.1)

    plt.tight_layout()
    return fig


def plot_ve_harmonic():
    arrays, params, extra = load_run("ve_harmonic")
    eta, mu = params["eta"], params["mu"]
    omega = extra["omega"]
    gd0 = extra["gamma_dot_0"]
    # Fine analytical grid for the smooth curve
    from _bench_helpers import maxwell_oscillatory
    t_grid = np.linspace(0, arrays["times"][-1], 2000)
    sigma_grid = maxwell_oscillatory(t_grid, eta, mu, gd0, omega)
    fig = _plot_three_panel("VE harmonic", t_grid, sigma_grid,
                            (arrays, params, extra))
    ax_top = fig.axes[0]
    info = (
        f"Amplitude  ana={extra['A_ana']:.4f}\n"
        f"           BDF-1={extra['A_bdf1']:.4f}  BDF-2={extra['A_bdf2']:.4f}\n"
        f"Phase lag  ana={extra['phi_ana']:.4f}\n"
        f"           BDF-1={extra['phi_bdf1']:.4f}  BDF-2={extra['phi_bdf2']:.4f}"
    )
    ax_top.text(0.02, 0.97, info, transform=ax_top.transAxes,
                ha="left", va="top",
                fontsize=8.5, family="monospace",
                bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.92,
                          boxstyle="round,pad=0.4"))
    out = f"{FIG_DIR}/bench_ve_harmonic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_ve_square():
    arrays, params, extra = load_run("ve_square")
    eta, mu = params["eta"], params["mu"]
    half_period = extra["half_period"]
    gd0 = extra["gamma_dot_0"]
    from _bench_helpers import maxwell_square_wave
    t_grid = np.linspace(0, arrays["times"][-1], 2000)
    sigma_grid = maxwell_square_wave(t_grid, eta, mu, gd0, half_period)
    fig = _plot_three_panel("VE square wave", t_grid, sigma_grid,
                            (arrays, params, extra))
    out = f"{FIG_DIR}/bench_ve_square.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_vep_square():
    arrays, params, extra = load_run("vep_square")
    eta, mu = params["eta"], params["mu"]
    half_period = extra["half_period"]
    tau_y = extra["tau_y"]
    gd0 = extra["gamma_dot_0"]
    from _bench_helpers import vep_square_wave
    t_grid = np.linspace(0, arrays["times"][-1], 2000)
    sigma_grid = vep_square_wave(t_grid, eta, mu, gd0, tau_y, half_period)
    fig = _plot_three_panel("VEP square wave (Min mode)", t_grid, sigma_grid,
                            (arrays, params, extra), tau_y=tau_y)
    out = f"{FIG_DIR}/bench_vep_square.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_convergence():
    """Three-panel log-log convergence plot, one panel per case."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    cases = [
        ("convergence_ve_harmonic", "VE harmonic"),
        ("convergence_ve_square",   "VE square wave"),
        ("convergence_vep_square",  "VEP square wave"),
    ]
    for ax, (name, title) in zip(axes, cases):
        try:
            arrays, params, extra = load_run(name)
        except FileNotFoundError:
            ax.set_title(f"{title} — no data")
            continue
        order = arrays["order"]; dt = arrays["dt"]
        max_abs = arrays["max_abs"]; rms = arrays["rms"]

        for o, marker, lbl_color in [(1, "o", C_BDF1), (2, "s", C_BDF2)]:
            mask = order == o
            if not mask.any():
                continue
            d = dt[mask]; e = max_abs[mask]; r = rms[mask]
            ax.loglog(d, e, marker=marker, color=lbl_color, ms=7,
                      lw=1.5, label=fr"BDF-{o} max$|\,\mathrm{{err}}\,|$")
            ax.loglog(d, r, marker=marker, color=lbl_color, ms=5,
                      lw=1.0, ls=":", alpha=0.7,
                      label=fr"BDF-{o} rms")

        # Reference slopes — a guide line through the smallest-dt BDF-2 max-abs
        if (order == 2).any():
            mask2 = order == 2
            d_ref = float(dt[mask2].min())
            e_ref = float(max_abs[mask2][np.argmin(dt[mask2])])
            d_grid = np.array([dt.min() * 0.7, dt.max() * 1.3])
            ax.loglog(d_grid, e_ref * (d_grid / d_ref) ** 2,
                      "k--", lw=0.8, alpha=0.5, label=r"slope 2")
            ax.loglog(d_grid, e_ref * (d_grid / d_ref) ** 1,
                      "k:", lw=0.8, alpha=0.5, label=r"slope 1")
        ax.set_title(title)
        ax.set_xlabel(r"$\Delta t$")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    axes[0].set_ylabel("error")
    plt.tight_layout()
    out = f"{FIG_DIR}/bench_convergence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    for plotter, name in [
        (plot_ve_harmonic, "ve_harmonic"),
        (plot_ve_square, "ve_square"),
        (plot_vep_square, "vep_square"),
        (plot_convergence, "convergence"),
    ]:
        try:
            plotter()
        except FileNotFoundError:
            print(f"  skipping {name} — no .npz on disk")
