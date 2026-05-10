"""Compare capped-Δt vs uncapped-Δt for rk2 and rk4 on the
free-surface continent test.

Reads ONLY checkpoints — pyvista VTU + profile npz from:
  output/continent_fs_snapshots_struct/        (uncapped, all 4 schemes)
  output/continent_fs_snapshots_struct_capped/ (capped, rk2 & rk4)

Outputs:
  output/phase_i2d_fs_continent_fs_capped_vs_uncapped.png
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyvista as pv


OUT_DIR = "output"
DIR_UNCAP = os.path.join(OUT_DIR, "continent_fs_snapshots_struct")
DIR_CAP   = os.path.join(OUT_DIR, "continent_fs_snapshots_struct_capped")


def _load_profile(snap_dir, scheme, label):
    p = os.path.join(snap_dir, f"profile_{scheme}_{label}.npz")
    if not os.path.isfile(p):
        return None
    d = np.load(p)
    return {'theta': d['theta'], 'dr': d['dr'],
            'h_pole': float(d['h_pole'])}


def _vtu_area(snap_dir, scheme, label):
    p = os.path.join(snap_dir, f"pv_{scheme}_{label}.vtu")
    if not os.path.isfile(p):
        return None
    m = pv.read(p)
    sized = m.compute_cell_sizes(length=False, area=True, volume=False)
    return float(np.asarray(sized.cell_data["Area"]).sum())


def main():
    A_0 = np.pi * (1.0 ** 2 - 0.5 ** 2)
    schemes = ['rk2', 'rk4']
    labels = ['halfway', 'final']

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    for row, scheme in enumerate(schemes):
        # Left: signed dr(θ), halfway (dotted) and final (solid),
        # capped vs uncapped
        ax_p = axes[row, 0]
        for source, color, snap in [
                ('uncapped', '#1f77b4', DIR_UNCAP),
                ('capped',   '#d62728', DIR_CAP)]:
            d_h = _load_profile(snap, scheme, 'halfway')
            d_f = _load_profile(snap, scheme, 'final')
            if d_h is not None:
                ax_p.plot(d_h['theta'], d_h['dr'], ':',
                          color=color, lw=1.2, alpha=0.8,
                          label=f"{source} halfway (h={d_h['h_pole']:.3e})")
            if d_f is not None:
                ax_p.plot(d_f['theta'], d_f['dr'], '-',
                          color=color, lw=1.5,
                          label=f"{source} final   (h={d_f['h_pole']:.3e})")
        # Reference circle (dr = 0)
        ax_p.axhline(0.0, color='grey', lw=0.6, alpha=0.5)
        # Mark continent block extent
        ax_p.axvspan(-0.4, 0.4, alpha=0.10, color='gold',
                     label='block extent')
        ax_p.set_xlabel("θ (rad)")
        ax_p.set_ylabel("δr = r − r_o")
        ax_p.set_title(f"{scheme}: surface profile (dotted=halfway, "
                       f"solid=final)")
        ax_p.grid(alpha=0.3)
        ax_p.legend(fontsize=8, loc='best')

        # Right: bar chart of volume change at halfway and final
        ax_v = axes[row, 1]
        x_labels, vals_un, vals_cap = [], [], []
        for label in labels:
            x_labels.append(label)
            A_u = _vtu_area(DIR_UNCAP, scheme, label)
            A_c = _vtu_area(DIR_CAP, scheme, label)
            vals_un.append(100.0 * (A_u - A_0) / A_0
                           if A_u is not None else 0.0)
            vals_cap.append(100.0 * (A_c - A_0) / A_0
                            if A_c is not None else 0.0)
        x = np.arange(len(x_labels))
        w = 0.35
        ax_v.bar(x - w/2, vals_un, w, color='#1f77b4',
                 label='uncapped', edgecolor='black')
        ax_v.bar(x + w/2, vals_cap, w, color='#d62728',
                 label='capped', edgecolor='black')
        # Annotations
        for xi, vu, vc in zip(x, vals_un, vals_cap):
            ax_v.text(xi - w/2, vu, f"{vu:+.2f}%",
                      ha='center', va='top' if vu < 0 else 'bottom',
                      fontsize=8)
            ax_v.text(xi + w/2, vc, f"{vc:+.2f}%",
                      ha='center', va='top' if vc < 0 else 'bottom',
                      fontsize=8)
        ax_v.axhline(0.0, color='black', lw=0.6)
        ax_v.set_xticks(x)
        ax_v.set_xticklabels(x_labels)
        ax_v.set_ylabel("ΔA / A₀  (%)")
        ax_v.set_title(f"{scheme}: volume change")
        ax_v.grid(alpha=0.3, axis='y')
        ax_v.legend(fontsize=9)

    fig.suptitle("Free-surface continent (structured mesh): "
                 "capped-Δt vs uncapped, rk2 & rk4",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(
        OUT_DIR, "phase_i2d_fs_continent_fs_capped_vs_uncapped.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
