"""Stress amplitude and phase of the tidally flexed shell against the Maxwell
time: reads records.json of each run, fits the trace projections on the two
forcing patterns to a sinusoid over the last period."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sim = sys.argv[1]
runs = sys.argv[2:]
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
summary = []
for run in runs:
    rs = json.load(open(os.path.join(sim, run, "records.json")))
    t = np.array([r["t"] for r in rs]); inv = np.array([r["stress_inv_rms"] for r in rs])
    tc = np.array([r["tr_on_cos"] for r in rs]); ts = np.array([r["tr_on_sin"] for r in rs])
    vt = np.array([r["vt_rms"] for r in rs])
    axes[0].plot(t, inv, label=run); axes[1].plot(t, tc, label=f"{run} cos"); axes[1].plot(t, ts, "--", label=f"{run} sin")
    axes[2].plot(t, vt, label=run)
    last = t > t.max() - 1.0 + 1e-9
    # the forcing's cos-pattern coefficient is A cos(2 pi t): fit tc = a cos + b sin
    M = np.column_stack([np.cos(2 * np.pi * t[last]), np.sin(2 * np.pi * t[last])])
    a, b = np.linalg.lstsq(M, tc[last], rcond=None)[0]
    amp, phase = np.hypot(a, b), np.degrees(np.arctan2(b, a))
    summary.append((run, inv[last].max(), amp, phase))
axes[0].set_title("rms in-plane stress invariant"); axes[1].set_title("stress trace projected on the forcing patterns")
axes[2].set_title("rms tangential speed"); 
for ax in axes: ax.set_xlabel("t (periods)"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(sim, "sweep.png"), dpi=120)
print("run  max|inv|  trace amplitude on cos pattern  phase (deg, lag of the stress behind the radial displacement rate)")
for s in summary: print(f"{s[0]:8s} {s[1]:.4f} {s[2]:.4f} {s[3]:7.1f}")
