"""Overlay vrms(t) and Nu(t) for an arbitrary set of run tags. Used to compare
the resolved 'truth' against adaptation variants in the gentle (resolvable)
regime — see whether each variant tracks the decay or injects spurious energy.

Usage:
  python multi_history_plot.py --sim-dir ~/+Simulations/StagnantLid \
      --tags cmp2_ref48 cmp2_uniform cmp2_adapt cmp2_adapt_natural cmp2_adapt_oldframe \
      --labels "ref48 (truth)" "uniform res24" "adapt forced" "adapt natural" "adapt old-frame" \
      --out gentle_regime_compare.png
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--sim-dir", default="~/+Simulations/StagnantLid")
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--labels", nargs="+", default=None)
ap.add_argument("--out", default="gentle_regime_compare.png")
ap.add_argument("--tmax", type=float, default=0.02)
args = ap.parse_args()
SIM = os.path.expanduser(args.sim_dir)
labels = args.labels or args.tags

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
colors = plt.cm.tab10(np.linspace(0, 1, len(args.tags)))
for tag, lab, c in zip(args.tags, labels, colors):
    hp = os.path.join(SIM, tag, "history.npz")
    if not os.path.exists(hp):
        print(f"  [skip] {tag}")
        continue
    h = np.load(hp)
    m = h["t"] <= args.tmax
    lw = 2.5 if "truth" in lab or "ref" in lab else 1.5
    ax[0].plot(h["t"][m], h["vrms"][m], "-", lw=lw, color=c, label=lab)
    ax[1].plot(h["t"][m], h["Nu"][m], "-", lw=lw, color=c, label=lab)
ax[0].set_title("vrms vs t (gentle regime: truth DECAYS)")
ax[0].set_xlabel("t"); ax[0].set_ylabel("vrms"); ax[0].legend(fontsize=8)
ax[1].set_title("Nu vs t (1.0 = pure conduction)")
ax[1].set_xlabel("t"); ax[1].set_ylabel("Nu"); ax[1].legend(fontsize=8)
ax[1].axhline(1.0, ls=":", color="grey", lw=1)
fig.tight_layout()
outp = os.path.join(SIM, args.out)
fig.savefig(outp, dpi=130)
print("wrote", outp)

# numeric: vrms at the last common time
print(f"\n{'run':24s} {'vrms@tmax':>11} {'Nu@tmax':>9}")
for tag, lab in zip(args.tags, labels):
    hp = os.path.join(SIM, tag, "history.npz")
    if not os.path.exists(hp):
        continue
    h = np.load(hp)
    m = h["t"] <= args.tmax
    if not m.any():
        continue
    print(f"{lab:24s} {h['vrms'][m][-1]:>11.3f} {h['Nu'][m][-1]:>+9.3f}")
