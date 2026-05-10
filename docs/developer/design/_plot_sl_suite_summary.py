"""SL-suite summary plots.

Two figures, ready for visual inspection in Preview:

1. `phase_i2d_fs_continent_sl_suite_topo_vs_t.png`
   h_pole(t) for each variant, parsed from the per-step run log.

2. `phase_i2d_fs_continent_sl_suite_profiles.png`
   Surface profile dr(θ) at halfway and final, read from the per-run
   profile npz captured by `_phase_i_fs_continent_fs_snapshots.py`.
"""

import os
import re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = "output"


# (label, log path, snapshot dir, color, linestyle)
VARIANTS = [
    ("rk4_sl  (cap=18)",
     "/tmp/v2p1_rk4sl_run.log",
     os.path.join(OUT_DIR, "continent_fs_snapshots_struct_v2p1_rk4sl"),
     "#d62728", "-"),
    ("rk4_sl  (relax c=0.5 → Δt≈8)",
     "/tmp/v2p1_rk4sl_relaxc05.log",
     os.path.join(OUT_DIR,
                  "continent_fs_snapshots_struct_rk4sl_relaxc05"),
     "#1f77b4", "--"),
    ("rk4_sl  (relax c=1.0 → Δt≈16)",
     "/tmp/v2p1_rk4sl_relaxc1.log",
     os.path.join(OUT_DIR,
                  "continent_fs_snapshots_struct_rk4sl_relaxc1"),
     "#2ca02c", "-"),
]


_RE_STEP = re.compile(
    r"step\s+(\d+):\s+h_pole=([+\-0-9.eE]+)\s+Δt=([+\-0-9.eE]+)")
_RE_AREA = re.compile(
    r"\[(\w+)/(halfway|final)\]\s+curved\s+area_uw=([+\-0-9.eE]+)\s+"
    r"ΔA=([+\-0-9.eE]+)\s+\(([+\-0-9.eE]+)%\)")


def parse_log(path):
    """Return dict with arrays h, dt, t (cumulative)."""
    if not os.path.isfile(path):
        return None
    h, dt = [], []
    delta_a = {}
    with open(path) as f:
        for line in f:
            m = _RE_STEP.search(line)
            if m:
                h.append(float(m.group(2)))
                dt.append(float(m.group(3)))
                continue
            m = _RE_AREA.search(line)
            if m:
                delta_a[m.group(2)] = float(m.group(5))
    if not h:
        return None
    h = np.asarray(h)
    dt = np.asarray(dt)
    t = np.cumsum(dt)
    return dict(h=h, dt=dt, t=t, delta_a=delta_a)


def topo_vs_t():
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, log, _, color, ls in VARIANTS:
        d = parse_log(log)
        if d is None:
            print(f"  (skip — no log) {label}")
            continue
        da_final = d['delta_a'].get('final', np.nan)
        ax.plot(d['t'], d['h'], ls, color=color, lw=1.6, marker='o',
                ms=2.5,
                label=f"{label}   ΔA_final={da_final:+.2f}%")
    ax.axhline(0.0, color='grey', lw=0.4, alpha=0.5)
    ax.axhline(0.040, color='black', lw=0.6, alpha=0.4, ls=':',
               label="≈ analytic h_eq (uniform-bulge estimate)")
    ax.set_xlabel("simulated time t")
    ax.set_ylabel("h_pole at θ=0 (= δr at the surface above the block)")
    ax.set_title("Continent isostasy — surface uplift at the pole over time\n"
                 "(structured polar-quad mesh, V2/P1, free surface)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    out = os.path.join(
        OUT_DIR, "phase_i2d_fs_continent_sl_suite_topo_vs_t.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    return out


def profiles():
    """Surface dr(θ) for each variant at halfway and final."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, label_short in zip(axes, ('halfway', 'final')):
        for label, log, snap, color, ls in VARIANTS:
            prof = os.path.join(snap, f"profile_*_{label_short}.npz")
            import glob
            cands = glob.glob(prof)
            if not cands:
                continue
            d = np.load(cands[0])
            h_pole = float(d['h_pole']) if 'h_pole' in d.files else np.nan
            ax.plot(d['theta'], d['dr'], ls, color=color, lw=1.5,
                    label=f"{label}  h_p={h_pole:+.3f}")
        ax.axhline(0.0, color='grey', lw=0.4, alpha=0.5)
        ax.axvline(0.0, color='grey', lw=0.4, alpha=0.5, ls=':')
        ax.set_xlabel("θ (rad)")
        ax.set_title(f"surface profile — {label_short}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    axes[0].set_ylabel("δr = r − r_o")
    fig.suptitle("Continent isostasy — surface profile by scheme\n"
                 "(V2/P1, structured polar-quad mesh, capped/dtf labels)",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(
        OUT_DIR, "phase_i2d_fs_continent_sl_suite_profiles.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    return out


def main():
    p1 = topo_vs_t()
    p2 = profiles()
    print("\nGenerated:")
    for p in (p1, p2):
        print(f"  {p}")


if __name__ == "__main__":
    main()
