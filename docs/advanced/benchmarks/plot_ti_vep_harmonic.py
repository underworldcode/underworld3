"""Plot TI-VEP harmonic angled-fault benchmark traces from saved npz.

Reads the σ=0-IC results saved by ``bench_ti_vep_harmonic_zeroIC.py``
and produces one combined figure showing global σ_xy and resolved
fault-plane shear over time, for the three fault angles
(θ ∈ {0°, +15°, -15°}) and two yield stresses (τ_y ∈ {0.15, 0.30}).

Output: ``docs/advanced/figures/bench_ti_vep_harmonic.png``
"""

import os
import numpy as np
import matplotlib
if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from _bench_helpers import OUTPUT_DIR, FIG_DIR


ANGLES = (0.0, 15.0, -15.0)
TAU_YS = (0.15, 0.30)


def _load(theta_deg, tau_y):
    tag = f"ti_vep_harmonic_zIC_th{theta_deg:+.0f}_ty{tau_y:.2f}".replace(".", "p")
    return np.load(os.path.join(OUTPUT_DIR, f"{tag}.npz"))


def main():
    fig, axes = plt.subplots(
        len(TAU_YS), len(ANGLES),
        figsize=(15, 8), sharex=True, sharey='row',
    )

    for row, ty in enumerate(TAU_YS):
        for col, theta in enumerate(ANGLES):
            ax = axes[row, col]
            d = _load(theta, ty)
            t = d['times']
            sxy_1 = d['sigma_xy_bdf1']
            sxy_2 = d['sigma_xy_bdf2']
            tres_1 = d['tau_resolved_bdf1']
            tres_2 = d['tau_resolved_bdf2']

            # Reconstruct V_top(t) and the VE-no-yield envelope from
            # saved scalars.  Note: the bench's saved ``sigma_ve`` was
            # computed with γ̇_0 = 2·V_0/H — wrong for these BCs (Top
            # moves, Bottom fixed → γ̇_0 = V_0/H).  Recompute here.
            V0 = float(d['V0'])
            omega = float(d['OMEGA'])
            eta_1 = float(d['ETA_1']); mu = float(d['MU'])
            t_r = eta_1 / mu
            De = omega * t_r
            phi = float(np.arctan(De))
            H = 1.0  # domain height in the bench
            gamma_dot_0 = V0 / H
            A_inf = 2.0 * eta_1 * (gamma_dot_0 / 2.0) / np.sqrt(1.0 + De**2)
            #         ↑  σ = 2η ε̇    ↑  ε̇ = γ̇/2 (tensor strain rate)
            sigma_ve = A_inf * np.cos(omega * t)
            v_top = V0 * np.cos(omega * t + phi)

            # Light-blue filled driving overlay, rescaled to half-peak σ
            sig_max = max(float(np.abs(sxy_1).max()),
                          float(np.abs(sigma_ve).max())) or 1.0
            drive_scale = 0.5 * sig_max / V0
            ax.fill_between(
                t, 0.0, drive_scale * v_top,
                color="#1F77B4", alpha=0.18, linewidth=0,
                label=fr"driving $V_{{\rm top}}(t)$ (×{drive_scale:.2f})",
            )

            # VE no-yield envelope (light grey, dashed)
            ax.plot(t, sigma_ve, ':', color='0.4', linewidth=1,
                    label=r'VE (no yield)')

            # τ_y guidelines
            ax.axhline(+ty, color='gray', linestyle=':', alpha=0.6,
                       linewidth=1, label=rf'$\pm\tau_y={ty}$')
            ax.axhline(-ty, color='gray', linestyle=':', alpha=0.6,
                       linewidth=1)

            # Global σ_xy (BDF-1 line, BDF-2 dots)
            ax.plot(t, sxy_1, '-', color='steelblue', linewidth=1.4,
                    alpha=0.8, label=r'$\sigma_{xy}$ (BDF-1)')
            ax.plot(t, sxy_2, 'o', color='steelblue', markersize=2,
                    markerfacecolor='none', markeredgewidth=0.6,
                    label=r'$\sigma_{xy}$ (BDF-2)')

            # Resolved fault-plane shear
            ax.plot(t, tres_1, '-', color='crimson', linewidth=1.4,
                    alpha=0.9, label=r'$\tau_{\rm resolved}$ (BDF-1)')
            ax.plot(t, tres_2, 's', color='crimson', markersize=2,
                    markerfacecolor='none', markeredgewidth=0.6,
                    label=r'$\tau_{\rm resolved}$ (BDF-2)')

            ax.set_title(rf'$\theta = {theta:+.0f}°,\;\tau_y = {ty}$',
                         fontsize=11)
            ax.grid(True, alpha=0.3)
            if row == len(TAU_YS) - 1:
                ax.set_xlabel(r'Time $t/t_r$')
            if col == 0:
                ax.set_ylabel(r'Stress')
            if row == 0 and col == len(ANGLES) - 1:
                ax.legend(fontsize=8, loc='upper right', framealpha=0.9)

    fig.suptitle(
        "TI-VEP harmonic shear with embedded fault — "
        r"$V_{\rm top}(t) = V_0\cos(\omega t + \varphi)$,  "
        r"$V_0 = 0.5$,  $\omega = \pi/2$,  $\eta = \mu = 1$,  "
        r"$\Delta t = 0.05$",
        fontsize=12, y=0.995,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(FIG_DIR, "bench_ti_vep_harmonic.png")
    fig.savefig(out_path, dpi=150)
    print(f"  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
