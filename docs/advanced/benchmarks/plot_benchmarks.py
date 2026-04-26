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

C_SIM = "#D62728"          # red — simulation markers
C_ANA = "black"            # solid black — analytical
C_DRIVE = "#1F77B4"        # light blue — driving (filled)
C_ERR = "#9467BD"          # purple — error trace
C_DT = "#2CA02C"           # green — dt
C_YIELD = "grey"


def _plot_three_panel(name, t_ana_grid, sigma_ana_grid, info, *, tau_y=None):
    """Common three-panel layout used by all three benchmarks.

    ``info`` is the loaded payload (arrays + metadata).
    ``t_ana_grid``, ``sigma_ana_grid`` are a fine-grained analytical curve
    for the smooth black line (the per-step ``sigma_ana`` is only
    sampled at solve points).
    """
    arrays, params, extra = info
    times = arrays["times"]
    sigma = arrays["sigma"]
    sigma_ana = arrays["sigma_ana"]
    dts = arrays["dts"]
    gamma_dot = arrays["gamma_dot"]

    fig, (ax_top, ax_err, ax_dt) = plt.subplots(
        3, 1, sharex=True,
        gridspec_kw={"height_ratios": [3.5, 1.5, 1.0]},
    )

    # --- Top: σ(t)
    # Background: driving γ̇(t), rescaled to fit alongside σ
    sigma_max = float(np.max(np.abs(sigma_ana_grid))) or 1.0
    gamma_max = float(np.max(np.abs(gamma_dot))) or 1.0
    drive_scale = 0.5 * sigma_max / gamma_max  # fits inside the σ-range
    ax_top.fill_between(
        times, 0.0, drive_scale * gamma_dot,
        color=C_DRIVE, alpha=0.20, linewidth=0,
        label=fr"driving $\dot\gamma(t)$ (scaled ×{drive_scale:.2f})",
    )
    ax_top.plot(t_ana_grid, sigma_ana_grid, "-", color=C_ANA, lw=1.4,
                label="analytical")
    ax_top.plot(times, sigma, "o", color=C_SIM, ms=3.6, alpha=0.85,
                label="simulation")
    if tau_y is not None:
        ax_top.axhline(+tau_y, color=C_YIELD, ls="--", lw=0.9, alpha=0.7,
                       label=fr"$\pm\tau_y$ = $\pm${tau_y:g}")
        ax_top.axhline(-tau_y, color=C_YIELD, ls="--", lw=0.9, alpha=0.7)
    ax_top.axhline(0, color="grey", lw=0.4, alpha=0.4)
    ax_top.set_ylabel(r"$\sigma_{xy}$")

    # Title with the headline numbers
    title_bits = [name]
    if "err_max" in extra:
        title_bits.append(fr"max$|\,\mathrm{{err}}\,| = {extra['err_max']:.2e}$")
    if "err_rms" in extra:
        title_bits.append(fr"rms = {extra['err_rms']:.2e}")
    if "De" in extra:
        title_bits.append(fr"$\mathrm{{De}} = {extra['De']:.3f}$")
    if "tau_y" in extra:
        title_bits.append(fr"$\tau_y = {extra['tau_y']:g}$")
    ax_top.set_title("    ".join(title_bits))
    ax_top.legend(loc="lower right")

    # --- Middle: |sigma − sigma_ana|
    err = np.abs(sigma - sigma_ana)
    eps = max(1e-12, err.min() * 0.1) if err.min() > 0 else 1e-12
    ax_err.semilogy(times, np.maximum(err, eps), "-", color=C_ERR, lw=0.9,
                    marker="o", ms=2.8)
    ax_err.set_ylabel(r"$|\sigma_{\mathrm{sim}} - \sigma_{\mathrm{ana}}|$")
    ax_err.set_ylim(bottom=eps)

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
    # Add an extra info box for amplitude / phase
    ax_top = fig.axes[0]
    info = (
        f"Amplitude: sim {extra['A_sim']:.4f}  /  ana {extra['A_ana']:.4f}\n"
        f"Phase lag: sim {extra['phi_sim']:.4f}  /  ana {extra['phi_ana']:.4f}  rad"
    )
    ax_top.text(0.02, 0.97, info, transform=ax_top.transAxes,
                ha="left", va="top",
                fontsize=9,
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


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    for plotter, name in [
        (plot_ve_harmonic, "ve_harmonic"),
        (plot_ve_square, "ve_square"),
        (plot_vep_square, "vep_square"),
    ]:
        try:
            plotter()
        except FileNotFoundError:
            print(f"  skipping {name} — no .npz on disk")
