"""Compare midpoint-corrected kinematic ETD against curvS-FSSA across
two dt regimes:

  Left column (dt-factor=1.0, γΔt ≈ 0.09): all schemes are in FE's
  stable regime — curvS, midpoint, FE+FSSA all decay monotonically.
  Midpoint provides no measurable improvement over curvS.

  Right column (dt-factor=20, γΔt ≈ 1.8): drunken-sailor regime.
  FE-noFSSA blows up. FE+FSSA over-damps to the noise floor in
  one step. curvS-FSSA decays cleanly. Midpoint-FSSA also decays
  cleanly but no better than curvS.

Inputs (npz files in output/):
  - phase_i2d_fs_etd_dtf0.05_n200_internal_res20.npz   small-dt reference
  - phase_i2d_fs_etd_dtf1.00_n32_internal_res20.npz    large-dt comparison
  - phase_i2d_fs_etd_dtf20.00_n16_internal_res20.npz   drunken-sailor regime
"""

import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = "output"


STYLE = {
    'FSSA=1_UPD=curvS':    ('#1f77b4', '-',  'o', 'curvS-FSSA (kinematic ETD)'),
    'FSSA=0_UPD=curvS':    ('#9bcae1', '--', 'o', 'curvS-noFSSA'),
    'FSSA=1_UPD=midpoint': ('#d62728', '-',  's', 'midpoint-FSSA (RK2 ETD)'),
    'FSSA=1_UPD=fe':       ('#ff7f0e', '--', 'v', 'FE+FSSA'),
    'FSSA=0_UPD=fe':       ('#bcbd22', ':',  '^', 'FE-noFSSA'),
}


def load(path):
    if not os.path.isfile(path):
        return None
    z = np.load(path, allow_pickle=True)
    keys = list(z.keys())
    schemes = sorted(set(k.rsplit('_', 1)[0] for k in keys))
    out = {}
    for s in schemes:
        prefix = s.rsplit('_dtf', 1)[0]
        out[prefix] = {
            't': z[f"{s}_t"],
            'A': z[f"{s}_A"],
            'Amax': z[f"{s}_Amax"],
        }
    return out


def fit_reference_gamma(ref):
    """Fit γ_eff from the small-dt reference."""
    t = ref['t']
    A = ref['A']
    mask = (t > 0.5) & (np.abs(A) > 1e-12)
    if mask.sum() < 5:
        return None, None
    slope, intercept = np.polyfit(t[mask], np.log(np.abs(A[mask])), 1)
    return -slope, intercept


def panel(ax_a, ax_l, store, title, ref=None, gamma_fit=None,
          intercept=None, t_extrap=None):
    if ref is not None:
        ax_a.plot(ref['t'], ref['A'], color='black', lw=2.0, alpha=0.6,
                  label="small-dt ref (curvS, dt·0.05)")
        ax_l.semilogy(ref['t'], np.maximum(np.abs(ref['A']), 1e-18),
                      color='black', lw=2.0, alpha=0.6,
                      label="small-dt ref")
    if gamma_fit is not None and t_extrap is not None:
        t_dense = np.linspace(0, t_extrap, 400)
        A_dense = np.exp(intercept) * np.exp(-gamma_fit * t_dense)
        ax_a.plot(t_dense, A_dense, color='black', lw=1.0, ls=':',
                  alpha=0.5,
                  label=fr"extrap $\gamma$={gamma_fit:.4f}")
        ax_l.semilogy(t_dense, A_dense, color='black', lw=1.0, ls=':',
                      alpha=0.5, label=fr"extrap $\gamma$={gamma_fit:.4f}")

    for prefix, d in store.items():
        if prefix not in STYLE:
            continue
        color, ls, mk, lab = STYLE[prefix]
        ax_a.plot(d['t'], d['A'], ls, color=color, marker=mk, ms=5,
                  lw=1.4, label=lab)
        ax_l.semilogy(d['t'],
                      np.maximum(np.abs(d['A']), 1e-18), ls,
                      color=color, marker=mk, ms=5, lw=1.4,
                      label=lab)
    ax_a.axhline(0.0, color='grey', lw=0.5, alpha=0.4)
    ax_a.set_title(title, fontsize=11)
    ax_a.set_ylabel("mode-10 amplitude (signed)")
    ax_a.grid(alpha=0.3)
    ax_a.legend(fontsize=7, loc='best')

    ax_l.set_xlabel("t")
    ax_l.set_ylabel("|mode-10 amplitude|  (log)")
    ax_l.grid(alpha=0.3, which='both')
    ax_l.legend(fontsize=7, loc='best')


def main():
    smalldt = load(os.path.join(
        OUT_DIR, "phase_i2d_fs_etd_dtf0.05_n200_internal_res20.npz"))
    largedt = load(os.path.join(
        OUT_DIR, "phase_i2d_fs_etd_dtf1.00_n32_internal_res20.npz"))
    dtf20 = load(os.path.join(
        OUT_DIR, "phase_i2d_fs_etd_dtf20.00_n16_internal_res20.npz"))
    if smalldt is None or largedt is None:
        print(f"  missing required input npz", flush=True)
        sys.exit(1)

    ref = smalldt.get('FSSA=1_UPD=curvS')
    gamma_fit, intercept = fit_reference_gamma(ref)
    print(f"  small-dt fit: γ_eff = {gamma_fit:.4f}", flush=True)

    if dtf20 is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex='col')
        panel(axes[0, 0], axes[1, 0], largedt,
              "γΔt ≈ 0.09 (FE-stable regime, dt·estimate_dt)",
              ref=ref, gamma_fit=gamma_fit, intercept=intercept,
              t_extrap=60)
        panel(axes[0, 1], axes[1, 1], dtf20,
              "γΔt ≈ 1.8 (drunken-sailor regime, dt·20·estimate_dt)",
              ref=None, gamma_fit=gamma_fit, intercept=intercept,
              t_extrap=600)
        # Cap log y-axis lower bound for readability
        for ax in axes[1, :]:
            ax.set_ylim(1e-5, 1.0)
        # Cap signed y-axis upper for the dtf20 panel — FE-noFSSA blows up
        axes[0, 1].set_ylim(-0.1, 0.1)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        panel(axes[0], axes[1], largedt,
              "γΔt ≈ 0.09 (FE-stable regime)",
              ref=ref, gamma_fit=gamma_fit, intercept=intercept,
              t_extrap=60)

    fig.suptitle("Midpoint-corrected kinematic ETD vs curvS-FSSA — "
                 "internal-boundary single-mode test",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "phase_i2d_fs_midpoint.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}", flush=True)

    # Print summary table
    print("\n  Summary:")
    print(f"  {'regime':>16s} {'scheme':30s} {'A_final':>12s} "
          f"{'A_max':>12s} {'t_final':>10s}")
    for tag, store in [("[small-dt ref]", smalldt),
                       ("[dtf=1, γΔt~0.09]", largedt),
                       ("[dtf=20, γΔt~1.8]", dtf20 or {})]:
        if not store:
            continue
        for prefix, d in store.items():
            print(f"  {tag:>16s} {prefix:30s}  "
                  f"{d['A'][-1]:+12.4e} {d['Amax'][-1]:12.4e} "
                  f"{d['t'][-1]:10.2f}")


if __name__ == "__main__":
    main()
