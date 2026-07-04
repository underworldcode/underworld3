"""Plot achieved-refinement time series from fault_convection_adapt_loop runs.

Reads refine_quality.npz + history.npz from one or more <sim-dir>/<tag> run
directories and overlays the fault/bulk and thermal-BL/bulk NN-spacing ratios
(the RIGHT measure of achieved refinement — see project memory) alongside vrms.

Usage:
  python fault_refine_quality_plot.py --sim-dir ~/+Simulations/StagnantLid+Fault \
      --tags rq_nofault rq_passive_uniform rq_passive_gmsh \
      --labels "no-fault" "passive uniform" "passive gmsh"
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--sim-dir", default="~/+Simulations/StagnantLid+Fault")
p.add_argument("--tags", nargs="+", required=True)
p.add_argument("--labels", nargs="+", default=None)
p.add_argument("--out", default="refine_quality_compare.png")
args = p.parse_args()

SIM = os.path.expanduser(args.sim_dir)
labels = args.labels or args.tags

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
colors = plt.cm.tab10(np.linspace(0, 1, len(args.tags)))

for tag, lab, c in zip(args.tags, labels, colors):
    rqp = os.path.join(SIM, tag, "refine_quality.npz")
    if not os.path.exists(rqp):
        print(f"  [skip] no refine_quality.npz for {tag}")
        continue
    z = np.load(rqp)
    step = z["step"]
    fr, bl = z["fault_ratio"], z["bl_ratio"]
    nfault = z["n_fault"]
    # fault/bulk ratio (lower = finer at the fault; ~1 = no fault refinement)
    if np.isfinite(fr).any():
        ax[0, 0].plot(step, fr, "-o", ms=3, color=c, label=lab)
    # thermal-BL/bulk ratio
    ax[0, 1].plot(step, bl, "-o", ms=3, color=c, label=lab)
    # n nodes near the fault
    ax[1, 0].plot(step, nfault, "-o", ms=3, color=c, label=lab)
    # vrms from history
    hp = os.path.join(SIM, tag, "history.npz")
    if os.path.exists(hp):
        h = np.load(hp)
        ax[1, 1].semilogy(h["step"], h["vrms"], "-", color=c, label=lab)

# Reference lines for the documented creation cap (~1/1.8 to ~1/1.5 = 0.56-0.67).
for thr, txt in [(1.0, "no refinement"), (0.6, "creation cap ~1.6x"),
                 (0.2, "gmsh+maintain ~5x")]:
    ax[0, 0].axhline(thr, ls=":", color="grey", lw=1)
    ax[0, 0].text(ax[0, 0].get_xlim()[1], thr, f" {txt}", va="center",
                  fontsize=7, color="grey")

ax[0, 0].set_title("fault/bulk NN-spacing ratio (lower = finer at fault)")
ax[0, 0].set_xlabel("step"); ax[0, 0].set_ylabel("ratio"); ax[0, 0].legend(fontsize=8)
ax[0, 0].invert_yaxis()
ax[0, 1].set_title("thermal-BL/bulk NN-spacing ratio")
ax[0, 1].set_xlabel("step"); ax[0, 1].set_ylabel("ratio"); ax[0, 1].legend(fontsize=8)
ax[0, 1].invert_yaxis()
ax[1, 0].set_title("# nodes within 1.5*width of the fault")
ax[1, 0].set_xlabel("step"); ax[1, 0].set_ylabel("n_fault"); ax[1, 0].legend(fontsize=8)
ax[1, 1].set_title("vrms (convective vigour)")
ax[1, 1].set_xlabel("step"); ax[1, 1].set_ylabel("vrms"); ax[1, 1].legend(fontsize=8)

fig.tight_layout()
outp = os.path.join(SIM, args.out)
fig.savefig(outp, dpi=130)
print(f"wrote {outp}")

# Print a compact numeric summary (median over the back half of each run).
print("\n=== achieved-refinement summary (median over back half of run) ===")
for tag, lab in zip(args.tags, labels):
    rqp = os.path.join(SIM, tag, "refine_quality.npz")
    if not os.path.exists(rqp):
        continue
    z = np.load(rqp)
    n = len(z["step"]); half = n // 2
    fr = z["fault_ratio"][half:]; bl = z["bl_ratio"][half:]
    fr = fr[np.isfinite(fr)]; bl = bl[np.isfinite(bl)]
    frm = np.median(fr) if fr.size else float("nan")
    blm = np.median(bl) if bl.size else float("nan")
    print(f"  {lab:20s}: fault/bulk={frm:.3f} ({1/frm:.2f}x finer)  "
          f"BL/bulk={blm:.3f} ({1/blm:.2f}x finer)  "
          f"n_fault~{int(np.median(z['n_fault'][half:]))}")
