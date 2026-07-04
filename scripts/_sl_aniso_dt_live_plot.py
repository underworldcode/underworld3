"""Live plot of the aniso-dt validation runs.
Reads the step lines from one or more log files and plots
Nu, vrms, dt vs step.

Usage:  python _sl_aniso_dt_live_plot.py <log1> [log2 ...]
"""
import sys
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_log(path):
    """Pull step-line metrics from a validation log."""
    step_re = re.compile(
        r"^\s*(\d+)\s+([\d\.\+\-eE]+)\s+([\d\.\+\-eE]+)\s+"
        r"([\d\.]+)s\s+([\d\.\+\-eE]+)\s+([\+\-]?[\d\.]+)\s+"
        r"\[([\+\-]?[\d\.]+),([\+\-]?[\d\.]+)\]\s+ok")
    rows = []
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            m = step_re.match(line)
            if m:
                rows.append([
                    int(m.group(1)),      # step
                    float(m.group(2)),    # t
                    float(m.group(3)),    # dt
                    float(m.group(4)),    # wall
                    float(m.group(5)),    # vrms
                    float(m.group(6)),    # Nu
                    float(m.group(7)),    # Tmin
                    float(m.group(8)),    # Tmax
                ])
    if not rows:
        return None
    a = np.asarray(rows)
    return dict(step=a[:, 0], t=a[:, 1], dt=a[:, 2], wall=a[:, 3],
                vrms=a[:, 4], Nu=a[:, 5],
                Tmin=a[:, 6], Tmax=a[:, 7])


def load_npz(path):
    """Read the continuously-written history.npz."""
    if not os.path.exists(path):
        return None
    z = np.load(path)
    if len(z['step']) == 0:
        return None
    return dict(step=z['step'], t=z['t'], dt=z['dt'],
                wall=z['wall'], vrms=z['vrms'], Nu=z['Nu'],
                Tmin=z['Tmin'], Tmax=z['Tmax'])


def load_source(path):
    """Heuristic: .npz → npz loader; otherwise log parser."""
    if path.endswith('.npz'):
        return load_npz(path)
    return parse_log(path)


if len(sys.argv) < 2:
    sys.exit("usage: python _sl_aniso_dt_live_plot.py <log1> [log2 ...]")

datasets = []
for path in sys.argv[1:]:
    d = load_source(path)
    if d is None:
        print(f"  no step data in {path}")
        continue
    label = os.path.basename(path).replace('aniso_dt_', '').replace(
        '.log', '').replace('history_', '').replace('.npz', '')
    # If the path is a history.npz, label by its directory
    if path.endswith('history.npz'):
        label = os.path.basename(os.path.dirname(path))
    datasets.append((label, d))
    print(f"  {label}: {len(d['step'])} steps "
          f"({d['step'].min()}..{d['step'].max()}), "
          f"t={d['t'].max():.5f}")

if not datasets:
    sys.exit("no datasets parsed")

fig, ax = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

for (label, d) in datasets:
    ax[0].plot(d['t'], d['Nu'], '-o', ms=3, label=label)
    ax[1].semilogy(d['t'], d['vrms'], '-o', ms=3, label=label)
    ax[2].semilogy(d['t'], d['dt'], '-o', ms=3, label=label)
    ax[3].plot(d['t'], d['Tmax'], '-^', ms=3, label=f"{label} Tmax")
    ax[3].plot(d['t'], d['Tmin'], '-v', ms=3, label=f"{label} Tmin")

ax[0].set_ylabel("Nu (mid-shell)")
ax[0].grid(alpha=0.3); ax[0].legend(fontsize=9)
ax[1].set_ylabel(r"$v_\mathrm{rms}$")
ax[1].grid(alpha=0.3, which='both')
ax[2].set_ylabel(r"$\Delta t$")
ax[2].grid(alpha=0.3, which='both')
ax[3].set_ylabel("T extents")
ax[3].axhline(0.0, color='gray', ls=':', lw=0.7)
ax[3].axhline(1.0, color='gray', ls=':', lw=0.7)
ax[3].grid(alpha=0.3)
ax[3].set_xlabel("simulated time t")
ax[3].legend(fontsize=8, ncol=2)

fig.suptitle("aniso-dt validation: Nu / vrms / dt / T extents",
             fontsize=11)
fig.tight_layout()

out = os.path.expanduser(
    "~/+Simulations/StagnantLid/aniso_dt_validate/live.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches='tight')
print(f"wrote {out}")
