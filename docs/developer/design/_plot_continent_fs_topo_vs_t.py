"""Time evolution of h_pole (topography at the centre of the block,
θ=0) for various integration schemes on the free-surface continent.

Reads ONLY the per-step log lines (no re-run). Each per-step line
in the run logs has the form:
    step N: h_pole=±X.XXXe-XX Δt=X.XXXe+XX
We parse these and reconstruct t = cumsum(Δt).

Inputs (run logs):
  output/run_continent_fs_struct.log
      (uncapped, schemes rk2, rk4, curvS, midpoint)
  output/run_continent_fs_struct_rk2_capped.log
      (rk2 with Δt cap = 18.0)
  output/run_continent_fs_struct_rk4_capped.log
      (rk4 with Δt cap = 20.0)

Output:
  output/phase_i2d_fs_continent_fs_topo_vs_t.png
"""

import os
import re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = "output"

SCHEME_COLOR = {
    'rk2':      '#1f77b4',
    'rk4':      '#2ca02c',
    'curvS':    '#d62728',
    'midpoint': '#ff7f0e',
}


_RE_SCHEME = re.compile(r"^=== (\w+) ===")
_RE_STEP = re.compile(
    r"step (\d+): h_pole=([+\-0-9.eE]+)\s+Δt=([+\-0-9.eE]+)")


def parse_log(path):
    """Return {scheme: {'t': array, 'h_pole': array, 'dt': array}}."""
    if not os.path.isfile(path):
        return {}
    out = {}
    cur = None
    with open(path) as f:
        for line in f:
            m = _RE_SCHEME.search(line)
            if m:
                cur = m.group(1)
                out.setdefault(cur, {'h': [], 'dt': []})
                continue
            m = _RE_STEP.search(line)
            if m and cur is not None:
                out[cur]['h'].append(float(m.group(2)))
                out[cur]['dt'].append(float(m.group(3)))
    # Convert to arrays + cumulative time
    for s, d in out.items():
        h = np.asarray(d['h'])
        dt = np.asarray(d['dt'])
        t = np.concatenate([[0.0], np.cumsum(dt)])
        # h_pole was recorded AFTER each step; pair t[1:] with h[:]
        d['h'] = h
        d['dt'] = dt
        d['t'] = t[1:]   # time at end of each step
    return out


def main():
    uncap = parse_log(
        os.path.join(OUT_DIR, "run_continent_fs_struct.log"))
    rk2_cap = parse_log(os.path.join(
        OUT_DIR, "run_continent_fs_struct_rk2_capped.log"))
    rk4_cap = parse_log(os.path.join(
        OUT_DIR, "run_continent_fs_struct_rk4_capped.log"))
    rk2_half = parse_log(os.path.join(
        OUT_DIR, "run_continent_fs_struct_rk2_capped_half.log"))
    rk4_half = parse_log(os.path.join(
        OUT_DIR, "run_continent_fs_struct_rk4_capped_half.log"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: all four uncapped schemes
    ax_un = axes[0]
    for scheme, d in uncap.items():
        if scheme not in SCHEME_COLOR or len(d['h']) == 0:
            continue
        ax_un.plot(d['t'], d['h'], '-o',
                   color=SCHEME_COLOR[scheme], ms=3.5, lw=1.4,
                   label=f"{scheme}")
    ax_un.axhline(0.0, color='grey', lw=0.4, alpha=0.5)
    ax_un.set_xlabel("t")
    ax_un.set_ylabel("h_pole = δr at centre of block (θ=0)")
    ax_un.set_title("Uncapped Δt — all four schemes")
    ax_un.grid(alpha=0.3)
    ax_un.legend(fontsize=10, loc='best')

    # Right panel: rk2 / rk4 uncapped vs capped vs half-capped
    ax_cap = axes[1]
    for scheme, color in [('rk2', '#1f77b4'),
                          ('rk4', '#2ca02c')]:
        d_u = uncap.get(scheme)
        if d_u is not None and len(d_u['h']) > 0:
            ax_cap.plot(d_u['t'], d_u['h'], '-o',
                        color=color, ms=3, lw=1.2,
                        label=f"{scheme} uncapped")
        cap_log = rk2_cap if scheme == 'rk2' else rk4_cap
        d_c = cap_log.get(scheme)
        if d_c is not None and len(d_c['h']) > 0:
            ax_cap.plot(d_c['t'], d_c['h'], '--s',
                        color=color, ms=3, lw=1.2, alpha=0.85,
                        label=f"{scheme} capped (Δt~halfway)")
        half_log = rk2_half if scheme == 'rk2' else rk4_half
        d_h = half_log.get(scheme)
        if d_h is not None and len(d_h['h']) > 0:
            ax_cap.plot(d_h['t'], d_h['h'], ':',
                        color=color, lw=1.6,
                        label=f"{scheme} half-cap (Δt~halfway/2)")
    ax_cap.axhline(0.0, color='grey', lw=0.4, alpha=0.5)
    ax_cap.set_xlabel("t")
    ax_cap.set_ylabel("h_pole at θ=0")
    ax_cap.set_title("rk2 / rk4: uncapped vs capped vs half-cap")
    ax_cap.grid(alpha=0.3)
    ax_cap.legend(fontsize=8, loc='best')

    fig.suptitle("Free-surface continent (structured): topography "
                 "above the block over time",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(
        OUT_DIR, "phase_i2d_fs_continent_fs_topo_vs_t.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
