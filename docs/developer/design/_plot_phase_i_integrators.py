"""Cost-vs-accuracy comparison: FE / RK2 / RK4 / curvS-FSSA / midpoint-FSSA
across dt-factor in {1, 2, 5, 10, 20}.

Each scheme can take its own number of steps; the question is which scheme
reaches a given (final t, A_mode) at lowest total Stokes-solve cost.

Reference: FE-noFSSA at small dt (no corrections, stable by step size alone).

Inputs (npz in output/):
  - phase_i2d_fs_etd_dtf0.05_n200_internal_res20.npz   ref (FE-noFSSA)
  - phase_i2d_fs_etd_dtf{1.00,2.00,5.00,10.00,20.00}_n*_internal_res20.npz

Outputs:
  - output/phase_i2d_fs_integrators_trajectories.png  one panel per scheme,
        with dt-factors overlaid; reference as black solid line
  - output/phase_i2d_fs_integrators_cost.png         A_final vs total
        Stokes solves; one curve per scheme
"""

import os
import sys
import glob
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = "output"


# Cost per step (Stokes solves)
COST_PER_STEP = {
    'fe':       1,
    'rk2':      2,
    'rk4':      4,
    'curvS':    1,
    'midpoint': 2,
}

SCHEME_LABEL = {
    'fe':       'FE-noFSSA',
    'rk2':      'RK2 (no γ)',
    'rk4':      'RK4 (no γ)',
    'curvS':    'curvS-FSSA',
    'midpoint': 'midpoint-FSSA (RK2 + ETD)',
}

SCHEME_COLOR = {
    'fe':       '#7f7f7f',
    'rk2':      '#1f77b4',
    'rk4':      '#2ca02c',
    'curvS':    '#d62728',
    'midpoint': '#ff7f0e',
}

DTF_LIST = [1.0, 2.0, 5.0, 10.0, 20.0]
DTF_MARKER = {1.0: 'o', 2.0: 's', 5.0: 'v', 10.0: '^', 20.0: 'D'}
DTF_ALPHA = {1.0: 1.0, 2.0: 0.85, 5.0: 0.7, 10.0: 0.55, 20.0: 0.4}


def load_runs():
    """Find npz files for all dtf values; group by scheme/dtf."""
    runs = {}  # runs[scheme_short][dtf] = {'t', 'A', 'Amax', 'n_steps'}
    for path in sorted(glob.glob(os.path.join(
            OUT_DIR, "phase_i2d_fs_etd_dtf*_internal_res20.npz"))):
        z = np.load(path, allow_pickle=True)
        keys = list(z.keys())
        for k in keys:
            if not k.endswith('_t'):
                continue
            schemekey = k[:-2]      # drop '_t'
            t = z[f"{schemekey}_t"]
            A = z[f"{schemekey}_A"]
            Amax = z[f"{schemekey}_Amax"]
            # Parse "FSSA={0,1}_UPD={fe,rk2,rk4,curvS,midpoint}_dtf{x.xx}_internal"
            parts = schemekey.split('_')
            update = parts[1].split('=')[1]   # fe/rk2/rk4/curvS/midpoint
            dtf_str = parts[2].lstrip('dtf')
            dtf = float(dtf_str)
            n_steps = len(t) - 1
            runs.setdefault(update, {})[dtf] = {
                't': t, 'A': A, 'Amax': Amax, 'n_steps': n_steps,
                'fssa': bool(int(parts[0].split('=')[1])),
            }
    return runs


def get_reference():
    """Find the FE-noFSSA small-dt reference (preferred), or fall back
    to whatever long small-dt run is present."""
    candidates = [
        ("FE-noFSSA",
         "phase_i2d_fs_etd_dtf0.05_n200_internal_res20.npz",
         'FSSA=0_UPD=fe_dtf0.05_internal'),
        ("curvS-FSSA",
         "phase_i2d_fs_etd_dtf0.05_n200_internal_res20.npz",
         'FSSA=1_UPD=curvS_dtf0.05_internal'),
    ]
    for label, fn, key in candidates:
        path = os.path.join(OUT_DIR, fn)
        if not os.path.isfile(path):
            continue
        z = np.load(path, allow_pickle=True)
        if f"{key}_t" not in z:
            continue
        return label, {
            't': z[f"{key}_t"], 'A': z[f"{key}_A"],
            'Amax': z[f"{key}_Amax"],
        }
    return None, None


def fit_gamma(ref):
    t = ref['t']
    A = ref['A']
    mask = (t > 0.5) & (np.abs(A) > 1e-12)
    if mask.sum() < 5:
        return None, None
    slope, intercept = np.polyfit(t[mask], np.log(np.abs(A[mask])), 1)
    return -slope, intercept


def main():
    runs = load_runs()
    ref_label, ref = get_reference()

    # Decide which scheme blocks to render (those with at least one dtf)
    schemes_in_data = [s for s in
                       ('fe', 'rk2', 'rk4', 'curvS', 'midpoint')
                       if s in runs and len(runs[s]) > 0]
    if not schemes_in_data:
        print("No scheme data found", flush=True)
        sys.exit(1)

    print(f"  reference: {ref_label}", flush=True)
    if ref is not None:
        gamma_fit, intercept = fit_gamma(ref)
        if gamma_fit is not None:
            print(f"  reference fit γ_eff = {gamma_fit:.4f}", flush=True)

    # === Trajectory panel: one panel per scheme ===
    n = len(schemes_in_data)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 8), sharey='row')
    if n == 1:
        axes = axes[:, None]

    t_extrap = 80.0  # axis max for visual consistency
    for col, scheme in enumerate(schemes_in_data):
        ax_a = axes[0, col]
        ax_l = axes[1, col]
        # Reference curve
        if ref is not None:
            ax_a.plot(ref['t'], ref['A'], color='black', lw=1.8,
                      alpha=0.7,
                      label=f"{ref_label} (small-dt ref)")
            ax_l.semilogy(ref['t'],
                          np.maximum(np.abs(ref['A']), 1e-18),
                          color='black', lw=1.8, alpha=0.7,
                          label=f"{ref_label}")
            if gamma_fit is not None:
                t_dense = np.linspace(0, t_extrap, 400)
                A_dense = np.exp(intercept) * np.exp(-gamma_fit * t_dense)
                ax_a.plot(t_dense, A_dense, color='black', ls=':',
                          lw=0.9, alpha=0.5,
                          label=fr"extrap $\gamma$={gamma_fit:.4f}")
                ax_l.semilogy(t_dense, A_dense, color='black', ls=':',
                              lw=0.9, alpha=0.5)

        for dtf in DTF_LIST:
            d = runs[scheme].get(dtf)
            if d is None:
                continue
            color = SCHEME_COLOR[scheme]
            mk = DTF_MARKER[dtf]
            alpha = DTF_ALPHA[dtf]
            label = (fr"dt·{dtf:g}, n={d['n_steps']}, "
                     f"{COST_PER_STEP[scheme] * d['n_steps']} solves")
            ax_a.plot(d['t'], d['A'], '-', color=color, alpha=alpha,
                      marker=mk, ms=5, lw=1.2, label=label)
            ax_l.semilogy(d['t'],
                          np.maximum(np.abs(d['A']), 1e-18), '-',
                          color=color, alpha=alpha, marker=mk, ms=5,
                          lw=1.2, label=label)

        title_label = SCHEME_LABEL[scheme]
        ax_a.set_title(f"{title_label}\n({COST_PER_STEP[scheme]}× Stokes "
                       f"solve / step)", fontsize=10)
        ax_a.axhline(0.0, color='grey', lw=0.4, alpha=0.5)
        ax_a.grid(alpha=0.3)
        ax_a.set_xlim(0, t_extrap)
        ax_a.legend(fontsize=6.5, loc='best', ncol=1)
        if col == 0:
            ax_a.set_ylabel("mode-10 amplitude (signed)")

        ax_l.grid(alpha=0.3, which='both')
        ax_l.set_xlim(0, t_extrap)
        ax_l.set_ylim(1e-5, 1.0)
        ax_l.set_xlabel("t")
        if col == 0:
            ax_l.set_ylabel("|mode-10 amplitude|  (log)")

    fig.suptitle("Free-surface relaxation: integrator comparison "
                 "across dt-factors (single-mode IC, internal "
                 "boundary, res=20)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "phase_i2d_fs_integrators_trajectories.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}", flush=True)

    # === Cost-vs-accuracy panel ===
    if ref is not None:
        fig2, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(13, 5))
        for scheme in schemes_in_data:
            xs, ys, ys_max = [], [], []
            for dtf in DTF_LIST:
                d = runs[scheme].get(dtf)
                if d is None:
                    continue
                # Total Stokes solves to reach final t
                cost = COST_PER_STEP[scheme] * d['n_steps']
                # Reference value at the run's final t (linear interp in
                # ref data; if t exceeds ref range, extrapolate via fit)
                t_final = d['t'][-1]
                if gamma_fit is not None:
                    A_ref_final = (np.exp(intercept)
                                   * np.exp(-gamma_fit * t_final))
                else:
                    A_ref_final = np.interp(
                        t_final, ref['t'], ref['A'])
                # Error vs reference
                err = abs(d['A'][-1] - A_ref_final)
                xs.append(cost)
                ys.append(err)
                ys_max.append(d['Amax'][-1])
            if not xs:
                continue
            color = SCHEME_COLOR[scheme]
            ax_c1.plot(xs, ys, '-o', color=color, lw=1.5, ms=6,
                       label=SCHEME_LABEL[scheme])
            ax_c2.plot(xs, ys_max, '-s', color=color, lw=1.5, ms=6,
                       label=SCHEME_LABEL[scheme])

        ax_c1.set_xlabel("Total Stokes solves to reach t≈60")
        ax_c1.set_ylabel("|A_final − A_ref(t_final)|")
        ax_c1.set_yscale('log')
        ax_c1.set_xscale('log')
        ax_c1.set_title("Error vs reference at run's final t")
        ax_c1.grid(alpha=0.3, which='both')
        ax_c1.legend(fontsize=8)

        ax_c2.set_xlabel("Total Stokes solves")
        ax_c2.set_ylabel("A_max at final step")
        ax_c2.set_yscale('log')
        ax_c2.set_xscale('log')
        ax_c2.set_title("Final A_max (blow-up if ≫ initial)")
        ax_c2.grid(alpha=0.3, which='both')
        ax_c2.legend(fontsize=8)

        fig2.suptitle("Cost vs accuracy across schemes",
                      fontsize=12)
        fig2.tight_layout()
        out2 = os.path.join(OUT_DIR, "phase_i2d_fs_integrators_cost.png")
        fig2.savefig(out2, dpi=140, bbox_inches="tight")
        plt.close(fig2)
        print(f"  wrote {out2}", flush=True)

    # === Print table ===
    print(f"\n  {'scheme':>20s} {'dtf':>6s} {'n':>4s} "
          f"{'solves':>7s} {'t_final':>9s} {'A_final':>11s} "
          f"{'A_max':>11s}")
    for scheme in schemes_in_data:
        for dtf in DTF_LIST:
            d = runs[scheme].get(dtf)
            if d is None:
                continue
            cost = COST_PER_STEP[scheme] * d['n_steps']
            print(f"  {SCHEME_LABEL[scheme]:>20s} {dtf:>6.2f} "
                  f"{d['n_steps']:>4d} {cost:>7d} "
                  f"{d['t'][-1]:>9.2f} {d['A'][-1]:>+11.3e} "
                  f"{d['Amax'][-1]:>11.3e}")


if __name__ == "__main__":
    main()
