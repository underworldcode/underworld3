"""Plot Phase I-2D FSSA × ETD comparison from saved npz."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = "output"


def plot_one(npz_path, out_png, title_suffix=""):
    z = np.load(npz_path, allow_pickle=True)
    keys = list(z.keys())

    # Group keys by scheme label
    schemes = sorted(set(k.rsplit('_', 1)[0] for k in keys))

    fig, (ax_amp, ax_abs) = plt.subplots(1, 2, figsize=(13, 5))

    style = {
        'FSSA=0_UPD=fe':   ('#d62728', '--', 'o'),  # red dashed
        'FSSA=1_UPD=fe':   ('#ff7f0e', '-',  's'),  # orange solid
        'FSSA=0_UPD=etd':  ('#1f77b4', '--', '^'),  # blue dashed
        'FSSA=1_UPD=etd':  ('#2ca02c', '-',  'D'),  # green solid
        'FSSA=0_UPD=curv': ('#9467bd', '--', 'v'),  # purple dashed
        'FSSA=1_UPD=curv': ('#17becf', '-',  'P'),  # cyan solid
    }
    label_map = {
        'FSSA=0_UPD=fe':   'FE only',
        'FSSA=1_UPD=fe':   'FE + FSSA',
        'FSSA=0_UPD=etd':  'ETD scalar mode',
        'FSSA=1_UPD=etd':  'ETD scalar + FSSA',
        'FSSA=0_UPD=curv': 'ETD curvature-τ',
        'FSSA=1_UPD=curv': 'ETD curvature + FSSA',
    }

    for s in schemes:
        # label includes _ic{s,m}_{v,V}; strip the suffix to match style
        # First strip _ic..._{v,V}, then strip _dtf...
        prefix_full = s.rsplit('_dtf', 1)[0]
        prefix = prefix_full
        if prefix not in style:
            continue
        color, ls, mk = style[prefix]
        t = z[f"{s}_t"]
        A = z[f"{s}_A"]
        Amax = z[f"{s}_Amax"]
        ax_amp.plot(t, A, ls, color=color, marker=mk, ms=6, lw=1.4,
                    label=label_map[prefix])
        ax_abs.semilogy(t, np.maximum(np.abs(A), 1e-18), ls,
                        color=color, marker=mk, ms=6, lw=1.4,
                        label=label_map[prefix])

    ax_amp.axhline(0.0, color='grey', lw=0.5, alpha=0.6)
    ax_amp.set_xlabel("t")
    ax_amp.set_ylabel("mode-10 amplitude (signed)")
    ax_amp.set_title(f"Surface mode amplitude vs time{title_suffix}")
    ax_amp.legend(fontsize=9, loc='upper right')
    ax_amp.grid(alpha=0.3)

    ax_abs.set_xlabel("t")
    ax_abs.set_ylabel("|mode-10 amplitude| (log)")
    ax_abs.set_title(f"|amplitude| (log scale){title_suffix}")
    ax_abs.legend(fontsize=9, loc='lower right')
    ax_abs.grid(alpha=0.3, which='both')

    fig.suptitle("Free-surface relaxation on annulus: FSSA × ETD",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}", flush=True)


def main():
    for f in os.listdir(OUT_DIR):
        if f.startswith("phase_i2d_fs_etd_") and f.endswith(".npz"):
            path = os.path.join(OUT_DIR, f)
            tag = f.replace("phase_i2d_fs_etd_", "").replace(".npz", "")
            out_png = os.path.join(OUT_DIR, f"phase_i2d_fs_etd_{tag}.png")
            plot_one(path, out_png, title_suffix=f"  ({tag})")


if __name__ == "__main__":
    main()
