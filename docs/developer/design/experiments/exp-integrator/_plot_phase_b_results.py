"""Phase B comparison plots — ETD-2 vs BDF-1 vs BDF-2.

Produces side-by-side panels reading the saved npz traces:
- ``output/benchmarks/ve_harmonic.npz``           (BDF-1, BDF-2 + analytical)
- ``output/exp_integrator_phase_b_ve_harmonic.npz`` (ETD-2 + analytical)

Outputs PNGs in ``output/exp_integrator_phase_b_*.png``.

Run::

    pixi run -e amr-dev python -u docs/developer/design/experiments/exp-integrator/_plot_phase_b_results.py
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive — write files only
import matplotlib.pyplot as plt


C_ANA = "#222222"
C_BDF1 = "#1f77b4"
C_BDF2 = "#d62728"
C_ETD = "#2ca02c"


def plot_ve_harmonic():
    bdf = np.load("output/benchmarks/ve_harmonic.npz", allow_pickle=True)
    etd = np.load("output/exp_integrator_phase_b_ve_harmonic.npz", allow_pickle=True)

    # Both runs share the same time grid and analytical reference; sanity-check.
    t_bdf = bdf["arr_times"]
    t_etd = etd["times"]
    sigma_ana_bdf = bdf["arr_sigma_ana"]
    sigma_ana_etd = etd["sigma_ana"]
    sigma_bdf1 = bdf["arr_sigma_bdf1"]
    sigma_bdf2 = bdf["arr_sigma_bdf2"]
    sigma_etd2 = etd["sigma_exp"]

    assert np.allclose(t_bdf, t_etd), "Time grids differ between runs"

    err_bdf1 = np.abs(sigma_bdf1 - sigma_ana_bdf)
    err_bdf2 = np.abs(sigma_bdf2 - sigma_ana_bdf)
    err_etd2 = np.abs(sigma_etd2 - sigma_ana_etd)

    A_inf = float(etd["A_inf"])

    fig, (ax_s, ax_e) = plt.subplots(
        2, 1, figsize=(10.5, 7.0), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.2]},
    )

    # σ trace panel
    ax_s.plot(t_bdf, sigma_ana_bdf, "-", color=C_ANA, lw=1.5, label="analytical")
    ax_s.plot(t_bdf, sigma_bdf1, ".", color=C_BDF1, ms=4, alpha=0.7,
              label=f"BDF-1 (max|err|={err_bdf1.max():.2e})")
    ax_s.plot(t_bdf, sigma_bdf2, ".", color=C_BDF2, ms=4, alpha=0.7,
              label=f"BDF-2 (max|err|={err_bdf2.max():.2e})")
    ax_s.plot(t_etd, sigma_etd2, ".", color=C_ETD, ms=4, alpha=0.85,
              label=f"ETD-2 (max|err|={err_etd2.max():.2e})")
    ax_s.set_ylabel(r"$\sigma_{xy}$ at centre")
    ax_s.axhline(0, color="0.7", lw=0.6, zorder=0)
    ax_s.axhline(+A_inf, color="0.6", lw=0.6, ls="--", zorder=0)
    ax_s.axhline(-A_inf, color="0.6", lw=0.6, ls="--", zorder=0)
    ax_s.set_title(
        "bench_ve_harmonic (peak-start IC, ω=π/2, dt=0.05) — ETD-2 vs BDF-1, BDF-2"
    )
    ax_s.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax_s.grid(True, alpha=0.3)

    # |error| panel (semilog)
    ax_e.semilogy(t_bdf, err_bdf1 + 1e-16, "-", color=C_BDF1, lw=0.9, alpha=0.85, label="BDF-1")
    ax_e.semilogy(t_bdf, err_bdf2 + 1e-16, "-", color=C_BDF2, lw=0.9, alpha=0.85, label="BDF-2")
    ax_e.semilogy(t_etd, err_etd2 + 1e-16, "-", color=C_ETD, lw=1.1, alpha=0.95, label="ETD-2")
    ax_e.set_xlabel("t")
    ax_e.set_ylabel(r"$|\sigma - \sigma_\mathrm{ana}|$")
    ax_e.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax_e.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    out = "output/exp_integrator_phase_b_ve_harmonic.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    print(f"  BDF-1 max|err|={err_bdf1.max():.4e}  rms={np.sqrt((err_bdf1**2).mean()):.4e}")
    print(f"  BDF-2 max|err|={err_bdf2.max():.4e}  rms={np.sqrt((err_bdf2**2).mean()):.4e}")
    print(f"  ETD-2 max|err|={err_etd2.max():.4e}  rms={np.sqrt((err_etd2**2).mean()):.4e}")


def plot_killer_summary():
    """Killer-test summary: bar chart of |τ_resolved|/τ_y per (θ, τ_y) for ETD-2 and BDF-1."""
    # Hard-coded from the BDF-1 production npz files (already validated centre probes
    # earlier in the session) and the latest ETD-2 sweep.
    cases = [
        # (theta_deg, tau_y, etd_tau_res_ratio, bdf1_tau_res_ratio, bdf2_tau_res_ratio_log10)
        (0,    0.15, 1.103, 1.122, np.log10(5.689)),
        (15,   0.15, 1.118, 1.143, np.log10(2.157e9)),
        (-15,  0.15, 1.120, 1.127, np.log10(6.889e7)),
        (0,    0.30, 0.922, 1.150, np.log10(9.620)),
        (15,   0.30, 0.804, 1.139, np.log10(9.091e9)),
        (-15,  0.30, 0.803, 1.138, np.log10(1.859e8)),
    ]
    labels = [f"θ={c[0]:+}°,\nτ_y={c[1]}" for c in cases]
    etd_ratios = [c[2] for c in cases]
    bdf1_ratios = [c[3] for c in cases]
    bdf2_log10 = [c[4] for c in cases]
    x = np.arange(len(cases))

    fig, (ax_main, ax_bdf2) = plt.subplots(
        2, 1, figsize=(10.5, 7.0), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )

    width = 0.36
    ax_main.bar(x - width / 2, bdf1_ratios, width, color=C_BDF1, alpha=0.85, label="BDF-1 (production)")
    ax_main.bar(x + width / 2, etd_ratios, width, color=C_ETD, alpha=0.9, label="ETD-2 (this work)")
    ax_main.axhline(1.0, color="0.4", lw=0.8, ls="--", zorder=0, label=r"$\tau_y$")
    ax_main.axhline(1.20, color="0.7", lw=0.8, ls=":", zorder=0, label=r"gate (1.20·$\tau_y$)")
    ax_main.set_ylabel(r"$|\tau_\mathrm{resolved}|$ at fault centre / $\tau_y$")
    ax_main.set_title(
        "bench_ti_vep_harmonic killer test — ETD-2 vs BDF-1 (centre probe, 6/6 PASS)"
    )
    ax_main.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax_main.grid(True, alpha=0.3, axis="y")
    ax_main.set_ylim(0.0, 1.4)

    # BDF-2 |σ_xy| log-blow-up panel — BDF-2 is the integrator ETD-2 *replaces*
    ax_bdf2.bar(x, bdf2_log10, color=C_BDF2, alpha=0.85, label=r"BDF-2 $\log_{10}|\sigma_{xy}|$ (centre)")
    ax_bdf2.axhline(np.log10(1.5), color="0.4", lw=0.8, ls="--", zorder=0,
                    label=r"$\log_{10}(1.5\cdot\tau_y\sim O(1))$")
    ax_bdf2.set_xticks(x)
    ax_bdf2.set_xticklabels(labels, fontsize=9)
    ax_bdf2.set_ylabel(r"$\log_{10}|\sigma_{xy}|$ at fault centre")
    ax_bdf2.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax_bdf2.grid(True, alpha=0.3, axis="y")
    ax_bdf2.set_title("BDF-2: blows up to 10⁵–10⁹ on every yield-active combo")

    fig.tight_layout()
    out = "output/exp_integrator_phase_b_killer_summary.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main():
    os.makedirs("output", exist_ok=True)
    plot_ve_harmonic()
    plot_killer_summary()


if __name__ == "__main__":
    main()
