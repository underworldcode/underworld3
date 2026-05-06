"""Plot the buoyancy (forced free surface) case. Diagnostic is A_max
(max |δr| on upper boundary) since there's no mode-10 IC."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = "output"
INPUT = os.path.join(OUT_DIR,
                     "phase_i2d_fs_etd_dtf1.00_n8_buoyancy.npz")

STYLE = {
    'FSSA=1_UPD=fe':    ('#ff7f0e', '-',  's', 'FE + FSSA'),
    'FSSA=0_UPD=fe':    ('#d62728', '--', 'o', 'FE'),
    'FSSA=1_UPD=curvS': ('#000000', '-',  '<', 'kinematic ETD (curv γ) + FSSA'),
    'FSSA=1_UPD=empE':  ('#8c564b', '-',  'h', 'kinematic ETD (empirical γ)'),
}


def main():
    if not os.path.isfile(INPUT):
        print(f"  missing {INPUT}", flush=True)
        return
    z = np.load(INPUT, allow_pickle=True)
    keys = list(z.keys())
    schemes = sorted(set(k.rsplit('_', 1)[0] for k in keys))

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(13, 5))

    for s in schemes:
        prefix = s.rsplit('_dtf', 1)[0]
        if prefix not in STYLE:
            continue
        color, ls, mk, lab = STYLE[prefix]
        t = z[f"{s}_t"]
        Amax = z[f"{s}_Amax"]
        ax_lin.plot(t, Amax, ls, color=color, marker=mk, ms=6, lw=1.4,
                    label=lab)
        ax_log.semilogy(t, np.maximum(Amax, 1e-18), ls, color=color,
                        marker=mk, ms=6, lw=1.4, label=lab)

    for ax in (ax_lin, ax_log):
        ax.set_xlabel("t")
        ax.grid(alpha=0.3)
    ax_lin.set_ylabel(r"max $|\delta r|$  on upper boundary")
    ax_log.set_ylabel(r"max $|\delta r|$  (log)")
    ax_lin.set_title("Linear scale")
    ax_log.set_title("Log scale")
    ax_lin.legend(fontsize=9, loc='upper left')

    fig.suptitle(
        "Forced free surface: buoyant blob in interior, flat IC.\n"
        "FE schemes grow linearly without bound. The kinematic ETD\n"
        "saturates at the driven equilibrium "
        r"$h_{\mathrm{eq}}=u_n/\gamma$.",
        fontsize=11, y=1.04)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "phase_i2d_fs_buoyancy.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
