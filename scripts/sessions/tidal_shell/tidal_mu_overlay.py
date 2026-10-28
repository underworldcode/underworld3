"""Accumulated lateral slip (mean over the stripes) through the last cycle for
several friction coefficients, and the net slip per orbit against mu.
Usage: <sim> <run:mu> [<run:mu> ...]"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sim = sys.argv[1]
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
mus, nets, peaks = [], [], []
for arg in sys.argv[2:]:
    run, mu = arg.split(":")
    try:
        mu = float(mu)
    except ValueError:
        pass
    rs = json.load(open(os.path.join(sim, run, "records.json")))
    t = np.array([r["t"] for r in rs]); last = t > t.max() - 1.0 + 1e-9
    names = sorted(rs[0]["slip"].keys())
    cum = np.mean([[r["slip"][nm]["cumulative"] for r in rs] for nm in names], axis=0)[last]
    rate = np.mean([[r["slip"][nm]["signed_mean"] for r in rs] for nm in names], axis=0)[last]
    ph = (t[last] - np.floor(t[last] - 1e-9)) * 360.0
    axes[0].plot(ph, cum - cum[0], label=f"{mu}" if isinstance(mu, str) else f"mu = {mu}")
    axes[1].plot(ph, rate, label=f"{mu}" if isinstance(mu, str) else f"mu = {mu}")
    mus.append(mu); nets.append(cum[-1] - cum[0]); peaks.append(np.abs(rate).max())
axes[0].set_title("accumulated lateral slip through the last cycle (mean over the stripes)")
axes[1].set_title("signed slip rate (mean over the stripes)")
for ax in axes[:2]:
    ax.axhline(0, color="k", lw=0.6); ax.grid(alpha=0.3); ax.set_xlim(0, 360); ax.set_xlabel("mean anomaly (deg)"); ax.legend(fontsize=8)
xs = mus if all(not isinstance(m, str) for m in mus) else np.arange(len(mus))
axes[2].plot(xs, nets, "o-", label="net slip per orbit"); axes[2].plot(xs, peaks, "s--", label="peak slip rate")
if xs is not mus: axes[2].set_xticks(xs); axes[2].set_xticklabels(mus, rotation=20, fontsize=8)
axes[2].set_xlabel("friction coefficient mu"); axes[2].grid(alpha=0.3); axes[2].legend(fontsize=8); axes[2].set_title("asymmetry (net slip per orbit) against friction")
out = sys.argv[1 + len(sys.argv[2:]) + 1] if False else os.environ.get("OVERLAY_OUT", "berne_mu_overlay.png")
fig.tight_layout(); fig.savefig(os.path.join(sim, out), dpi=120)
print({m: (round(n, 5), round(p, 4)) for m, n, p in zip(mus, nets, peaks)})
