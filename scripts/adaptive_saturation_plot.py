"""Progress plotter for the 3-way saturation experiment. Reads
whatever the parallel runs have written so far (partial OK):

  1. Nu(t) and vrms(t) overlay (matplotlib — scalar histories).
  2. Latest-checkpoint T + mesh per model via the UW/pyvista path
     (P3 T on its own DOF cloud + deformed-mesh edges, white bg,
     lighting off — the required high-order render).

Run any time while the runs progress.
"""
from __future__ import annotations
import os
import glob
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import underworld3 as uw
import underworld3.visualisation as vis

DIR = "/tmp/metric_mesh/sat"
MODELS = [("ref24", 24, "k", "ref uniform res-24"),
          ("u16", 16, "#c0392b", "uniform res-16"),
          ("a16", 16, "#e07b00",
           "res-16 + adapt (cumulative, over-adapted)"),
          ("a16p", 16, "#1f4e8c",
           "res-16 + adapt (pristine, conservative)"),
          ("a16s", 16, "#2ca02c",
           "res-16 + adapt (pristine, AGGRESSIVE)"),
          ("a16x", 16, "#9467bd",
           "res-16 + adapt (pristine, amp=24 ≡ a16s [amp no-op])"),
          ("a16y", 16, "#8c564b",
           "res-16 + adapt (pristine, cap=5 β=300 — true strong)"),
          ("a16z", 16, "#17becf",
           "res-16 + adapt (pristine, pct 85/99 — budget conc.)"),
          ("a16c", 16, "#d62728",
           "res-16 + adapt (pristine, coarsen_cap=4 — over-coarse)"),
          ("a16c2", 16, "#e377c2",
           "res-16 + adapt (pristine, coarsen_cap=2)"),
          ("a16c15", 16, "#7f7f7f",
           "res-16 + adapt (pristine, coarsen_cap=1.5)"),
          ("a16e", 16, "#2ca02c",
           "res-16 + adapt (equidistribution, resolution_ratio=2)"),
          ("a16ed", 16, "#ff7f0e",
           "res-16 + adapt (equidist R=2 + EMA-G damping)"),
          ("a16r15", 16, "#1f77b4",
           "res-16 + adapt (equidist R=1.5 — CORRECTED prod)"),
          ("a16r15e", 16, "#9467bd",
           "res-16 + adapt (equidist R=1.5 + EMA-G damping)")]


def latest_ckpt(tag):
    fs = glob.glob(f"{DIR}/sat_{tag}.mesh.T.*.h5")
    idx = []
    for f in fs:
        m = re.search(r"\.mesh\.T\.(\d+)\.h5$", f)
        if m:
            idx.append(int(m.group(1)))
    return max(idx) if idx else None


# ---- 1. Nu(t) / vrms(t) -------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
print(f"{'model':>26} | {'steps':>5} {'t_end':>8} "
      f"{'Nu_last':>8} {'vrms_last':>10}")
print("-" * 66)
for tag, res, col, lab in MODELS:
    hp = f"{DIR}/sat_{tag}_hist.npz"
    if not os.path.exists(hp):
        print(f"{lab:>26} |  (no history yet)")
        continue
    try:
        z = np.load(hp)
    except Exception:
        print(f"{lab:>26} |  (history mid-write, skip)")
        continue
    t, Nu, vr = z["t"], z["Nu"], z["vrms"]
    ax[0].plot(t, Nu, "-", color=col, lw=1.7, label=lab)
    ax[1].plot(t, vr, "-", color=col, lw=1.7, label=lab)
    print(f"{lab:>26} | {int(z['step'][-1]):5d} {t[-1]:8.4f} "
          f"{Nu[-1]:+8.3f} {vr[-1]:10.3e}")
ax[0].set_xlabel("dimensionless time")
ax[0].set_ylabel("Nu")
ax[0].set_title("Nusselt(t) — overshoot then settle")
ax[1].set_xlabel("dimensionless time")
ax[1].set_ylabel("vrms")
ax[1].set_title("vrms(t)")
for a in ax:
    a.legend(fontsize=9)
    a.grid(alpha=0.3)
fig.suptitle("3-way saturation: res-24 vs res-16 vs res-16+adapt "
             "(does adapt land BETWEEN once settled?)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{DIR}/sat_timeseries.png", dpi=130)
print(f"\nsaved {DIR}/sat_timeseries.png")

# ---- 2. latest T + mesh per model (UW/pyvista path) ---------------
import pyvista as pv

pv.OFF_SCREEN = True
present = [(t, r, c, l) for (t, r, c, l) in MODELS
           if latest_ckpt(t) is not None]
if present:
    import math
    n = len(present)
    ncol = math.ceil(math.sqrt(n))
    nrow = math.ceil(n / ncol)
    pl = pv.Plotter(shape=(nrow, ncol), off_screen=True,
                    window_size=(950 * ncol, 950 * nrow))
    pl.set_background("white")
    from underworld3.meshing.smoothing import (
        _tri_cells, _signed_areas)
    for k, (tag, res, _, lab) in enumerate(present):
        rr, cc = divmod(k, ncol)
        idx = latest_ckpt(tag)
        m = uw.discretisation.Mesh(
            f"{DIR}/sat_{tag}.mesh.{idx:05}.h5")
        Tv = uw.discretisation.MeshVariable(
            "T", m, vtype=uw.VarType.SCALAR, degree=3,
            continuous=True)
        Tv.read_timestep(f"sat_{tag}", "T", idx, outputPath=DIR)
        pv_T = vis.meshVariable_to_pv_mesh_object(Tv)
        pv_T.point_data["T"] = np.asarray(Tv.data[:, 0])
        edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
        # quantify mesh compression (the over-adaptation check)
        tris = _tri_cells(m.dm)
        A = np.abs(_signed_areas(np.asarray(m.X.coords), tris))
        qmin = A.min() / A.mean()
        pl.subplot(rr, cc)
        pl.add_text(f"{lab}\nckpt {idx}   minA/meanA={qmin:.3f}",
                    font_size=16, color="black")
        # RdBu_r: T=0 (cold) blue, T=1 (hot) red, 0.5 white — the
        # free-surface viz convention (white bg + lighting=False so
        # mid-T isn't dirty grey; crisp dark edges for cell quality)
        pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
                    clim=(0.0, 1.0), show_edges=False,
                    lighting=False,
                    show_scalar_bar=(k == n - 1),
                    scalar_bar_args=dict(title="T", color="black"))
        pl.add_mesh(edges, color="#202020", line_width=0.8,
                    lighting=False)
        pl.view_xy()
        pl.camera.zoom(1.3)
    out = f"{DIR}/sat_fields.png"
    pl.screenshot(out)
    print(f"saved {out}")
else:
    print("no checkpoints yet for the field render")
