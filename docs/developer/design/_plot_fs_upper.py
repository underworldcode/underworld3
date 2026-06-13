"""Plot the stress-equilibrium free-surface integrator on the upper-surface
isostasy problem: topography evolution + scheme comparison.

Reads output/phase_i2d_fs_upper_*.npz (from _phase_i_fs_isostasy_upper.py),
writes figures to ~/+Simulations/freesurface_stress_equilibrium/.
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.expanduser("~/+Simulations/freesurface_stress_equilibrium")
os.makedirs(OUT, exist_ok=True)
IN = "output"
R_O = 1.0
EQ = 0.0227   # curvS kinematic equilibrium reference


def load(tag):
    fns = glob.glob(os.path.join(IN, f"phase_i2d_fs_upper_dtf{tag}*.npz"))
    fns = [f for f in fns if "_vel" not in f]   # scheme-comparison files only
    if not fns:
        return None
    d = dict(np.load(sorted(fns)[-1], allow_pickle=True))
    labels = sorted({k.rsplit("_", 1)[0] for k in d})
    return d, labels


def scheme_name(label):
    for part in label.split("_"):
        if part.startswith("UPD="):
            base = part[4:]
    fssa = "FSSA=1" in label
    return base + ("+FSSA" if fssa else "")


COLORS = {"rk4": "C3", "curvS": "C1", "etd_topo": "C0", "fe": "C7",
          "rk2": "C4"}


def color(label):
    for k, c in COLORS.items():
        if f"UPD={k}_" in label or label.endswith(f"UPD={k}"):
            return c
    return "C2"


def style(label):
    """(color, linestyle, marker): FSSA on = dashed + square, off = solid + o."""
    fssa = "FSSA=1" in label
    return color(label), ("--" if fssa else "-"), ("s" if fssa else "o")


fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# --- Panels A,B: h_pole(t) at dtf1 and dtf4 ---
for ax, tag, title in ((axes[0, 0], "1.00", "Δt = 1·Δt_est  (moderate)"),
                       (axes[0, 1], "4.00", "Δt = 4·Δt_est  (large — rk4 limit)")):
    res = load(tag)
    if res is None:
        ax.set_title(f"{title}\n(no data)")
        continue
    d, labels = res
    for lab in labels:
        t = d[f"{lab}_t"]
        hp = d[f"{lab}_hpole"]
        c, ls, mk = style(lab)
        ax.plot(t, hp, color=c, ls=ls, lw=2, label=scheme_name(lab),
                marker=mk, ms=4, markerfacecolor="none" if ls == "--" else c)
    ax.axhline(EQ, color="k", ls=":", lw=1.2, label="kinematic eq (0.0227)")
    ax.set_xlabel("time"); ax.set_ylabel("h at pole (topography above blob)")
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# --- Panel C: etd_topo evolving topography h(θ) at snapshots (dtf1) ---
axC = axes[1, 0]
res = load("1.00")
if res is not None:
    d, labels = res
    etd = [l for l in labels if "UPD=etd_topo" in l and "FSSA=0" in l
           and f"{l}_drsnaps" in d]
    if etd:
        lab = etd[0]
        snaps = d[f"{lab}_drsnaps"]            # (n_step+1, n_upper)
        th = d[f"{lab}_finalTh"]
        order = np.argsort(th)
        n = snaps.shape[0]
        idxs = np.linspace(0, n - 1, min(7, n)).astype(int)
        cmap = plt.cm.viridis(np.linspace(0, 1, len(idxs)))
        for c, i in zip(cmap, idxs):
            axC.plot(th[order], snaps[i][order], color=c, lw=1.5,
                     label=f"step {i}")
        axC.axhline(EQ, color="k", ls="--", lw=1)
        axC.set_xlabel("θ (rad)"); axC.set_ylabel("surface deflection h(θ)")
        axC.set_title("etd_topo: topography forming (dtf1)")
        axC.legend(fontsize=7, ncol=2); axC.grid(alpha=0.3)

# --- Panel D: final surface in polar (the annulus bulge) ---
axD = axes[1, 1]; axD.remove()
axD = fig.add_subplot(2, 2, 4, projection="polar")
res = load("1.00")
if res is not None:
    d, labels = res
    th_circ = np.linspace(0, 2 * np.pi, 200)
    axD.plot(th_circ, np.full_like(th_circ, R_O), "k:", lw=0.8,
             label="undeformed r_o")
    for lab in labels:
        if "UPD=etd_topo" in lab and "FSSA=0" in lab:
            th = d[f"{lab}_finalTh"]; dr = d[f"{lab}_finalDr"]
            o = np.argsort(th)
            # exaggerate deflection ×5 for visibility
            axD.plot(th[o], R_O + 5 * dr[o], color="C0", lw=2,
                     label="etd_topo surface (×5)")
    axD.plot([0], [0.7], "r*", ms=14, label="buoyant blob")
    axD.set_rmax(1.25); axD.set_title("final surface (deflection ×5)",
                                      va="bottom")
    axD.legend(fontsize=7, loc="lower left", bbox_to_anchor=(-0.1, -0.1))

fig.suptitle("Stress-equilibrium free-surface integrator (etd_topo) — "
             "upper-surface isostasy", fontsize=13)
fig.tight_layout()
p = os.path.join(OUT, "etd_topo_isostasy_upper.png")
fig.savefig(p, dpi=130)
print(f"Wrote {p}")
