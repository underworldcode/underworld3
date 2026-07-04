"""Is the velocity genuinely blotchy (adaptation-loop problem) or a sampling
artifact? For one or more steps: print raw V.data stats and sample |v| on a
regular grid (clean, mesh-topology-independent) — a smooth grid means the data
is fine; a blotchy grid means the field itself is wrong.
"""
from __future__ import annotations
import os, glob, re, argparse
import numpy as np, sympy, underworld3 as uw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--tag', type=str, default='fault_ti_Ra1e6_fmg')
ap.add_argument('--steps', type=str, default='')   # comma list, '' = first+mid+last
args = ap.parse_args()
DIR = os.path.expanduser(f'~/+Simulations/StagnantLid/{args.tag}')
allc = sorted(glob.glob(os.path.join(DIR, "step*.mesh.00000.h5")),
              key=lambda c: int(re.search(r"step(\d+)\.mesh", c).group(1)))
labels = ([f"step{int(s):04d}" for s in args.steps.split(",")] if args.steps else
          [re.search(r"(step\d+)\.mesh", os.path.basename(c)).group(1)
           for c in (allc[0], allc[len(allc)//2], allc[-1])])

g = np.linspace(-1, 1, 360); GX, GY = np.meshgrid(g, g)
pts = np.column_stack([GX.ravel(), GY.ravel()])
rr = np.linalg.norm(pts, axis=1); inside = (rr > 0.505) & (rr < 0.995)

fig, ax = plt.subplots(1, len(labels), figsize=(6.2 * len(labels), 6.4))
if len(labels) == 1: ax = [ax]
for j, label in enumerate(labels):
    mesh = uw.discretisation.Mesh(os.path.join(DIR, f"{label}.mesh.00000.h5"))
    V = uw.discretisation.MeshVariable("V_v2p1", mesh, mesh.dim, degree=2)
    V.read_timestep(label, "V_v2p1", 0, outputPath=DIR)
    vmag_dof = np.sqrt((V.data ** 2).sum(1))
    print(f"[{label}] V.data: comp range x[{V.data[:,0].min():.2f},{V.data[:,0].max():.2f}] "
          f"y[{V.data[:,1].min():.2f},{V.data[:,1].max():.2f}]  "
          f"|v| max={vmag_dof.max():.2f} mean={vmag_dof.mean():.2f} n_dof={V.data.shape[0]}",
          flush=True)
    Vg = np.asarray(uw.function.evaluate(V.sym, pts[inside])).reshape(-1, mesh.dim)
    Vmg = np.full(pts.shape[0], np.nan)
    Vmg[inside] = np.sqrt((Vg ** 2).sum(1))
    # smoothness probe: how rough is the grid field? (local |Laplacian| proxy)
    field = Vmg.reshape(GX.shape)
    print(f"           grid |v|: max={np.nanmax(Vmg):.2f}  "
          f"NaN-inside={np.isnan(Vmg[inside]).sum()}", flush=True)
    im = ax[j].imshow(field, origin="lower", extent=[-1, 1, -1, 1], cmap="magma",
                      vmin=0, vmax=np.nanpercentile(Vmg, 99))
    ax[j].set_title(f"|v| on grid — {label}")
    ax[j].set_aspect("equal"); ax[j].axis("off")
    plt.colorbar(im, ax=ax[j], shrink=0.6)
fig.suptitle(f"velocity smoothness diagnosis (regular-grid sample) — {args.tag}", fontsize=13)
fig.tight_layout()
out = os.path.join(DIR, "V_griddiag.png")
fig.savefig(out, dpi=150); print("→", out)
