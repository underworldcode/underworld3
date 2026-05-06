"""Summary figure for the FSSA × ETD investigation: 3 test cases ×
both signed-amplitude and log-|amplitude| in a 2×3 grid."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = "output"

# For each case: (file tag, title, A0 reference, τ_analytical or None)
# τ for half-space mode-m: τ = 2η|k|/ρg = 2·m/r_o·(η/ρg). With ρg=1,
# η=1, r_o=1: τ_m = 2m. Mode 10 → τ = 20.
CASES = [
    ("dtf1.00_n8_icsingle",
     "Single-mode IC (mode 10), Δt = estimate_dt()", 0.05, 20.0),
    ("dtf1.00_n8_icmulti",
     "Multi-mode IC (modes 10+25), Δt = estimate_dt()", 0.0125, 20.0),
    ("dtf10.00_n8_icsingle_V",
     "Single mode + visc contrast, Δt = 10·estimate_dt()", 0.05, None),
]

# Curv (closed-form αh) is dropped — it's the analytic solution
# packaged as an integrator and not really a method to compare against
# in the general case where we want to integrate u_n directly.
# ETD-scalar (mode-projection) is also dropped — it was a stepping
# stone that doesn't generalise. The interesting comparison is
# kinematic ETD (curvS / empE) vs FE / FE+FSSA.
STYLE = {
    'FSSA=0_UPD=fe':    ('#d62728', '--', 'o', 'FE'),
    'FSSA=1_UPD=fe':    ('#ff7f0e', '-',  's', 'FE+FSSA'),
    'FSSA=1_UPD=curvS': ('#000000', '-',  '<', 'kinematic ETD (curv γ) + FSSA'),
    'FSSA=0_UPD=curvS': ('#888888', '--', '<', 'kinematic ETD (curv γ)'),
    'FSSA=1_UPD=empE':  ('#8c564b', '-',  'h', 'kinematic ETD (empirical γ) + FSSA'),
    'FSSA=0_UPD=empE':  ('#c49c94', '--', 'h', 'kinematic ETD (empirical γ)'),
}


def main():
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    for col, (tag, title, A0, tau_ref) in enumerate(CASES):
        npz_path = os.path.join(OUT_DIR, f"phase_i2d_fs_etd_{tag}.npz")
        if not os.path.isfile(npz_path):
            print(f"  missing {npz_path}, skipping", flush=True)
            continue
        z = np.load(npz_path, allow_pickle=True)
        keys = list(z.keys())
        schemes = sorted(set(k.rsplit('_', 1)[0] for k in keys))

        ax_a = axes[0, col]
        ax_l = axes[1, col]

        # Analytical mode-10 decay reference
        if tau_ref is not None:
            t_max_for_ref = 0.0
            for s in schemes:
                t = z[f"{s}_t"]
                t_max_for_ref = max(t_max_for_ref, float(t[-1]))
            t_dense = np.linspace(0, t_max_for_ref, 200)
            A_dense = A0 * np.exp(-t_dense / tau_ref)
            ax_a.plot(t_dense, A_dense, color='black', lw=2.5,
                      alpha=0.35,
                      label=fr"analytical $A_0 e^{{-t/{tau_ref:.0f}}}$")
            ax_l.semilogy(t_dense, A_dense, color='black', lw=2.5,
                          alpha=0.35,
                          label="analytical")

        for s in schemes:
            prefix = s.rsplit('_dtf', 1)[0]
            if prefix not in STYLE:
                continue
            color, ls, mk, lab = STYLE[prefix]
            t = z[f"{s}_t"]
            A = z[f"{s}_A"]
            ax_a.plot(t, A, ls, color=color, marker=mk, ms=5, lw=1.3,
                      label=lab)
            ax_l.semilogy(t, np.maximum(np.abs(A), 1e-18), ls,
                          color=color, marker=mk, ms=5, lw=1.3,
                          label=lab)

        ax_a.axhline(0.0, color='grey', lw=0.5, alpha=0.6)
        ax_a.set_title(title, fontsize=10)
        ax_a.set_ylabel("mode-10 amplitude (signed)")
        ax_a.grid(alpha=0.3)
        if col == 0:
            ax_a.legend(fontsize=8, loc='lower left', ncol=2)

        ax_l.set_xlabel("t")
        ax_l.set_ylabel("|amplitude|  (log)")
        ax_l.grid(alpha=0.3, which='both')

    fig.suptitle("Free-surface relaxation on annulus: kinematic ETD"
                 " vs FE / FE+FSSA",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "phase_i2d_fs_summary.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
