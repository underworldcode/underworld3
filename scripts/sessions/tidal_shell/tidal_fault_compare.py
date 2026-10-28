"""Faulted against unfaulted shell: the global velocity and stress diagnostics
over the cycle, and the slip on each stripe. Usage: <sim> <faulted run> <reference run>"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sim, fr, rr = sys.argv[1:4]
F = json.load(open(os.path.join(sim, fr, "records.json")))
R = json.load(open(os.path.join(sim, rr, "records.json")))
n = min(len(F), len(R)); F, R = F[:n], R[:n]
t = np.array([r["t"] for r in F])
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
for key, ax, ttl in (("vt_rms", axes[0], "rms tangential speed"), ("stress_inv_rms", axes[1], "rms in-plane stress invariant")):
    ax.plot(t, [r[key] for r in R], "k", label="unfaulted")
    ax.plot(t, [r[key] for r in F], "r", label="4 stripes")
    ax.set_title(ttl); ax.set_xlabel("t (periods)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
names = sorted(F[0]["slip"].keys())
for nm in names:
    axes[2].plot(t, [r["slip"][nm]["max"] for r in F], label=f"{nm} max")
axes[2].set_title("slip on the stripes (max over the trace)"); axes[2].set_xlabel("t (periods)"); axes[2].grid(alpha=0.3); axes[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(sim, f"{fr}_vs_{rr}.png"), dpi=120)
last = t > t.max() - 1.0 + 1e-9
def rms(key, rs): return float(np.sqrt(np.mean([r[key] ** 2 for r, m in zip(rs, last) if m])))
print(f"{fr} vs {rr} over the last period:")
print(f"  rms tangential speed  faulted {rms('vt_rms', F):.5f}  unfaulted {rms('vt_rms', R):.5f}  ratio {rms('vt_rms', F)/rms('vt_rms', R):.4f}")
print(f"  rms stress invariant  faulted {rms('stress_inv_rms', F):.5f}  unfaulted {rms('stress_inv_rms', R):.5f}  ratio {rms('stress_inv_rms', F)/rms('stress_inv_rms', R):.4f}")
print(f"  membrane rel err      faulted {rms('rel_err_membrane', F):.4f}  unfaulted {rms('rel_err_membrane', R):.4f}")
for nm in names:
    v = [r["slip"][nm]["max"] for r, m in zip(F, last) if m]
    print(f"  {nm}: max slip over the cycle {max(v):.4f}, rms {np.sqrt(np.mean(np.square(v))):.4f}")
