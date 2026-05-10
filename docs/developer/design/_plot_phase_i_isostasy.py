"""Isostatic-relaxation comparison: bulge-amplitude trajectory and
final-surface profile across schemes.

Reference: FE-noFSSA at small dt (no corrections).

Inputs (npz in output/):
  - phase_i2d_fs_etd_dtf0.10_n80_isostasy_res20.npz   small-dt ref
  - phase_i2d_fs_etd_dtf{1.00,5.00}_n*_isostasy_res20.npz   sweeps

Outputs:
  - output/phase_i2d_fs_isostasy_trajectories.png  h_pole(t) per scheme
  - output/phase_i2d_fs_isostasy_profile.png       final-step δr(θ)
"""

import os
import sys
import glob
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = "output"


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
    'midpoint': 'midpoint-FSSA',
}

SCHEME_COLOR = {
    'fe':       '#7f7f7f',
    'rk2':      '#1f77b4',
    'rk4':      '#2ca02c',
    'curvS':    '#d62728',
    'midpoint': '#ff7f0e',
}


def load_runs(adaptive=False):
    """Load runs. If adaptive=True, only use the *_adt.npz files;
    if adaptive=False, only use the non-adaptive files."""
    runs = {}
    pattern = ("phase_i2d_fs_etd_dtf*_isostasy_res20_adt.npz"
               if adaptive
               else "phase_i2d_fs_etd_dtf*_isostasy_res20.npz")
    for path in sorted(glob.glob(os.path.join(OUT_DIR, pattern))):
        # Skip adt files when loading non-adaptive
        if not adaptive and path.endswith("_adt.npz"):
            continue
        z = np.load(path, allow_pickle=True)
        keys = list(z.keys())
        for k in keys:
            if not k.endswith('_t'):
                continue
            schemekey = k[:-2]
            t = z[f"{schemekey}_t"]
            hmax = z[f"{schemekey}_hmax"]
            hpole = z[f"{schemekey}_hpole"]
            try:
                final_dr = z[f"{schemekey}_finalDr"]
                final_th = z[f"{schemekey}_finalTh"]
            except KeyError:
                final_dr = None
                final_th = None
            try:
                dt_history = z[f"{schemekey}_dthistory"]
            except KeyError:
                dt_history = None
            parts = schemekey.split('_')
            update = parts[1].split('=')[1]
            dtf_str = parts[2].lstrip('dtf')
            dtf = float(dtf_str)
            n_steps = len(t) - 1
            runs.setdefault(update, {})[dtf] = {
                't': t, 'hmax': hmax, 'hpole': hpole,
                'final_dr': final_dr, 'final_th': final_th,
                'dt_history': dt_history,
                'n_steps': n_steps,
                'fssa': bool(int(parts[0].split('=')[1])),
            }
    return runs


def get_reference():
    candidates = [
        ("FE-noFSSA",
         "phase_i2d_fs_etd_dtf0.10_n80_isostasy_res20.npz",
         'FSSA=0_UPD=fe_dtf0.10_isostasy'),
    ]
    for label, fn, key in candidates:
        path = os.path.join(OUT_DIR, fn)
        if not os.path.isfile(path):
            continue
        z = np.load(path, allow_pickle=True)
        if f"{key}_t" not in z:
            continue
        return label, {
            't': z[f"{key}_t"], 'hmax': z[f"{key}_hmax"],
            'hpole': z[f"{key}_hpole"],
            'final_dr': z[f"{key}_finalDr"],
            'final_th': z[f"{key}_finalTh"],
        }
    return None, None


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--adaptive', action='store_true',
                   help="Plot the adaptive-dt sweep instead of fixed-dt")
    args = p.parse_args()
    runs = load_runs(adaptive=args.adaptive)
    ref_label, ref = get_reference()
    suffix = "_adt" if args.adaptive else ""

    schemes_in_data = [s for s in
                       ('fe', 'rk2', 'rk4', 'curvS', 'midpoint')
                       if s in runs and len(runs[s]) > 0]
    if not schemes_in_data:
        print("No scheme data found", flush=True)
        sys.exit(1)

    # === Trajectory panel ===
    n = len(schemes_in_data)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 4),
                             sharey=True)
    if n == 1:
        axes = [axes]

    DTF_LIST = [1.0, 2.0, 5.0]
    DTF_MARKER = {1.0: 'o', 2.0: 'v', 5.0: 's'}
    DTF_ALPHA = {1.0: 1.0, 2.0: 0.85, 5.0: 0.7}

    t_max_seen = 0.0
    for col, scheme in enumerate(schemes_in_data):
        ax = axes[col]
        if ref is not None:
            ax.plot(ref['t'], ref['hpole'], color='black', lw=1.8,
                    alpha=0.7,
                    label=f"{ref_label} (small-dt ref)")
            t_max_seen = max(t_max_seen, float(ref['t'][-1]))
        for dtf in DTF_LIST:
            d = runs[scheme].get(dtf)
            if d is None:
                continue
            color = SCHEME_COLOR[scheme]
            mk = DTF_MARKER[dtf]
            alpha = DTF_ALPHA[dtf]
            cost = COST_PER_STEP[scheme] * d['n_steps']
            label = f"dt·{dtf:g}, n={d['n_steps']}, {cost} solves"
            ax.plot(d['t'], d['hpole'], '-', color=color, alpha=alpha,
                    marker=mk, ms=5, lw=1.3, label=label)
            t_max_seen = max(t_max_seen, float(d['t'][-1]))

        ax.set_title(f"{SCHEME_LABEL[scheme]} "
                     f"({COST_PER_STEP[scheme]}× / step)",
                     fontsize=10)
        ax.set_xlabel("t")
        if col == 0:
            ax.set_ylabel("h_pole (radial rise above blob)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc='lower right')

    title_extra = " — adaptive Δt" if args.adaptive else " — fixed Δt"
    fig.suptitle("Isostatic relaxation: surface bulge over a buoyant "
                 "blob (flat IC, internal boundary, res=20)"
                 + title_extra, fontsize=11)
    fig.tight_layout()
    out = os.path.join(
        OUT_DIR, f"phase_i2d_fs_isostasy_trajectories{suffix}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}", flush=True)

    # === Final profile panel ===
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    ax_ref = axes2[0]
    ax_dtf1 = axes2[1]

    if ref is not None and ref['final_th'] is not None:
        ax_ref.plot(ref['final_th'], ref['final_dr'],
                    color='black', lw=1.8,
                    label=f"{ref_label} (small-dt ref)")
    for dtf in (1.0, 5.0):
        for scheme in schemes_in_data:
            d = runs[scheme].get(dtf)
            if d is None or d['final_th'] is None:
                continue
            ax_target = ax_dtf1 if dtf == 1.0 else ax_ref
            color = SCHEME_COLOR[scheme]
            ls = '-' if dtf == 1.0 else '--'
            ax_target.plot(d['final_th'], d['final_dr'],
                           ls, color=color, lw=1.4,
                           label=f"{SCHEME_LABEL[scheme]} dt·{dtf:g}")

    ax_ref.set_xlabel("θ (rad)")
    ax_ref.set_ylabel("δr at final step")
    ax_ref.set_title("Final surface profile (ref + dt-factor=5 overlaid)")
    ax_ref.grid(alpha=0.3)
    ax_ref.legend(fontsize=7, loc='best')

    ax_dtf1.set_xlabel("θ (rad)")
    ax_dtf1.set_title("Final surface profile (dt-factor=1)")
    ax_dtf1.grid(alpha=0.3)
    ax_dtf1.legend(fontsize=7, loc='best')

    fig2.suptitle("Isostatic relaxation: final boundary δr(θ)"
                  + title_extra, fontsize=11)
    fig2.tight_layout()
    out2 = os.path.join(
        OUT_DIR, f"phase_i2d_fs_isostasy_profile{suffix}.png")

    # === Δt-history panel (only meaningful for adaptive) ===
    if args.adaptive:
        fig3, ax_dt = plt.subplots(1, 1, figsize=(8, 5))
        for scheme in schemes_in_data:
            for dtf in DTF_LIST:
                d = runs[scheme].get(dtf)
                if d is None or d['dt_history'] is None:
                    continue
                color = SCHEME_COLOR[scheme]
                mk = DTF_MARKER.get(dtf, 'o')
                ls = '-' if dtf == 1.0 else ('--' if dtf == 2.0 else ':')
                steps = np.arange(1, len(d['dt_history']) + 1)
                ax_dt.plot(steps, d['dt_history'], ls,
                           color=color, marker=mk, ms=5, lw=1.3,
                           label=f"{SCHEME_LABEL[scheme]} dt·{dtf:g}")
        ax_dt.set_xlabel("step")
        ax_dt.set_ylabel("Δt (re-evaluated each step)")
        ax_dt.set_yscale('log')
        ax_dt.set_title("Time-step growth as v decays "
                        "(velocity-CFL-bounded estimate_dt)")
        ax_dt.grid(alpha=0.3, which='both')
        ax_dt.legend(fontsize=7, loc='best', ncol=2)
        fig3.tight_layout()
        out3 = os.path.join(
            OUT_DIR, f"phase_i2d_fs_isostasy_dthistory{suffix}.png")
        fig3.savefig(out3, dpi=140, bbox_inches="tight")
        plt.close(fig3)
        print(f"  wrote {out3}", flush=True)
    fig2.savefig(out2, dpi=140, bbox_inches="tight")
    plt.close(fig2)
    print(f"  wrote {out2}", flush=True)

    # === Summary table ===
    print(f"\n  {'scheme':>16s} {'dtf':>5s} {'n':>4s} {'solves':>7s} "
          f"{'t_final':>9s} {'h_max':>11s} {'h_pole':>11s}")
    if ref is not None:
        print(f"  {'--ref--':>16s} {'0.10':>5s} {'80':>4s} "
              f"{'80':>7s} {ref['t'][-1]:>9.2f} "
              f"{ref['hmax'][-1]:>11.4e} {ref['hpole'][-1]:>+11.4e}")
    for scheme in schemes_in_data:
        for dtf in DTF_LIST:
            d = runs[scheme].get(dtf)
            if d is None:
                continue
            cost = COST_PER_STEP[scheme] * d['n_steps']
            print(f"  {SCHEME_LABEL[scheme]:>16s} {dtf:>5.2f} "
                  f"{d['n_steps']:>4d} {cost:>7d} "
                  f"{d['t'][-1]:>9.2f} "
                  f"{d['hmax'][-1]:>11.4e} {d['hpole'][-1]:>+11.4e}")


if __name__ == "__main__":
    main()
