"""Compare continent integrator sweep on structured vs unstructured
meshes (dt-factor=1, n=24, adaptive Δt).

Inputs:
  output/phase_i2d_fs_etd_dtf1.00_n24_continent_res20_adt.npz
  output/phase_i2d_fs_etd_dtf1.00_n24_continent_res20_adt_struct.npz

Outputs:
  output/phase_i2d_fs_continent_struct_vs_unstruct.png
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = "output"

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


def load(path):
    if not os.path.isfile(path):
        return None
    z = np.load(path, allow_pickle=True)
    out = {}
    for k in z.files:
        if not k.endswith('_t'):
            continue
        scheme_key = k[:-2]
        upd = scheme_key.split('_')[1].split('=')[1]
        out[upd] = {
            't': z[f"{scheme_key}_t"],
            'hpole': z[f"{scheme_key}_hpole"],
            'hmax': z[f"{scheme_key}_hmax"],
        }
    return out


def main():
    unstr = load(os.path.join(
        OUT_DIR,
        "phase_i2d_fs_etd_dtf1.00_n24_continent_res20_adt.npz"))
    struc = load(os.path.join(
        OUT_DIR,
        "phase_i2d_fs_etd_dtf1.00_n24_continent_res20_adt_struct.npz"))

    if unstr is None:
        print("missing unstructured npz")
        return
    if struc is None:
        print("structured npz not yet written; "
              "showing unstructured only")

    # Reference equilibrium peak (≈ 0.041 from earlier curvS)
    h_eq_peak = 0.041

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Top row: trajectories. Left = unstructured, right = structured.
    for col, (label, store) in enumerate(
            [("unstructured", unstr), ("structured", struc)]):
        ax = axes[0, col]
        if store is None:
            ax.text(0.5, 0.5, "(no data)", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f"{label} — no data")
            continue
        ax.axhline(h_eq_peak, color='black', lw=1.0, alpha=0.4,
                   label=f"approx. peak h_eq={h_eq_peak}")
        ax.axhline(0.051, color='black', lw=0.8, alpha=0.3, ls=':',
                   label="analytic uniform-bulge h_b=0.051")
        for upd, d in store.items():
            color = SCHEME_COLOR.get(upd, 'gray')
            ax.plot(d['t'], d['hpole'], '-', color=color,
                    marker='o', ms=4, lw=1.4,
                    label=SCHEME_LABEL.get(upd, upd))
        ax.set_xlabel("t")
        ax.set_ylabel("h_pole")
        ax.set_title(f"{label} mesh — h_pole(t)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')

    # Bottom row: final-state bar chart
    schemes = ['fe', 'rk2', 'rk4', 'curvS', 'midpoint']
    x = np.arange(len(schemes))
    w = 0.4

    ax_bar = axes[1, 0]
    if unstr is not None:
        u_vals = [unstr.get(s, {}).get('hpole', [0])[-1] if s in unstr else 0
                  for s in schemes]
        ax_bar.bar(x - w/2, u_vals, w, label='unstructured',
                   color='#cccccc', edgecolor='black')
    if struc is not None:
        s_vals = [struc.get(s, {}).get('hpole', [0])[-1] if s in struc else 0
                  for s in schemes]
        ax_bar.bar(x + w/2, s_vals, w, label='structured',
                   color='#ffd47f', edgecolor='black')
    ax_bar.axhline(h_eq_peak, color='black', lw=1.0, alpha=0.4,
                   label=f"~peak {h_eq_peak}")
    ax_bar.axhline(0.051, color='black', lw=0.8, alpha=0.3, ls=':',
                   label="analytic 0.051")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([SCHEME_LABEL[s] for s in schemes],
                           rotation=30, ha='right', fontsize=8)
    ax_bar.set_ylabel("final h_pole")
    ax_bar.set_title("Final h_pole by scheme")
    ax_bar.grid(alpha=0.3, axis='y')
    ax_bar.legend(fontsize=8)

    # Bottom-right: h_max - h_pole asymmetry indicator
    ax_asym = axes[1, 1]
    if unstr is not None:
        u_a = [(unstr.get(s, {}).get('hmax', [0])[-1]
                - unstr.get(s, {}).get('hpole', [0])[-1])
               if s in unstr else 0 for s in schemes]
        ax_asym.bar(x - w/2, u_a, w, label='unstructured',
                    color='#cccccc', edgecolor='black')
    if struc is not None:
        s_a = [(struc.get(s, {}).get('hmax', [0])[-1]
                - struc.get(s, {}).get('hpole', [0])[-1])
               if s in struc else 0 for s in schemes]
        ax_asym.bar(x + w/2, s_a, w, label='structured',
                    color='#ffd47f', edgecolor='black')
    ax_asym.set_xticks(x)
    ax_asym.set_xticklabels([SCHEME_LABEL[s] for s in schemes],
                            rotation=30, ha='right', fontsize=8)
    ax_asym.set_ylabel("h_max − h_pole  (asymmetry)")
    ax_asym.set_title("Bulge asymmetry (0 = symmetric)")
    ax_asym.grid(alpha=0.3, axis='y')
    ax_asym.legend(fontsize=8)

    fig.suptitle("Continent isostasy: structured vs unstructured "
                 "mesh, dt-factor=1, n=24, adaptive Δt",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(
        OUT_DIR, "phase_i2d_fs_continent_struct_vs_unstruct.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")

    # Print summary
    print(f"\n  {'scheme':12s} {'unstruct h_pole':>16s} "
          f"{'struct h_pole':>16s} {'unstruct asym':>16s} "
          f"{'struct asym':>14s}")
    for s in schemes:
        u_hp = unstr.get(s, {}).get('hpole', [0])[-1] if unstr else None
        u_hm = unstr.get(s, {}).get('hmax', [0])[-1] if unstr else None
        s_hp = struc.get(s, {}).get('hpole', [0])[-1] if struc else None
        s_hm = struc.get(s, {}).get('hmax', [0])[-1] if struc else None
        u_a = (u_hm - u_hp) if (u_hp is not None and u_hm is not None) else None
        s_a = (s_hm - s_hp) if (s_hp is not None and s_hm is not None) else None
        u_s = f"{u_hp:+.3e}" if u_hp is not None else "—"
        s_s = f"{s_hp:+.3e}" if s_hp is not None else "—"
        ua_s = f"{u_a:+.3e}" if u_a is not None else "—"
        sa_s = f"{s_a:+.3e}" if s_a is not None else "—"
        print(f"  {SCHEME_LABEL[s]:12s} {u_s:>16s} {s_s:>16s} "
              f"{ua_s:>16s} {sa_s:>14s}")


if __name__ == "__main__":
    main()
