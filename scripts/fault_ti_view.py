"""Look at the TI fault convection: T + mesh + fault, |v| + streamlines, and a
fault-zoom with velocity vectors (to see whether shear localises along the
anisotropic weak fault). Loads the latest snapshot of a run tag.
"""
from __future__ import annotations
import os, glob, re, argparse
import numpy as np, sympy, underworld3 as uw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

ap = argparse.ArgumentParser()
ap.add_argument('--tag', type=str, default='fault_ti_Ra1e6_penalty')
ap.add_argument('--fault-dip-deg', type=float, default=30.0)
ap.add_argument('--fault-depth', type=float, default=0.225)
ap.add_argument('--step', type=str, default='')   # '' = latest
args = ap.parse_args()
DIR = os.path.expanduser(f'~/+Simulations/StagnantLid/{args.tag}')

cands = sorted(glob.glob(os.path.join(DIR, "step*.mesh.00000.h5")),
               key=lambda c: int(re.search(r"step(\d+)\.mesh", c).group(1)))
label = args.step or re.search(r"(step\d+)\.mesh", os.path.basename(cands[-1])).group(1)
print(f"rendering {label} of {args.tag}", flush=True)
mesh = uw.discretisation.Mesh(os.path.join(DIR, f"{label}.mesh.00000.h5"))
T = uw.discretisation.MeshVariable("T_v2p1", mesh, 1, degree=3, varsymbol="T")
V = uw.discretisation.MeshVariable("V_v2p1", mesh, mesh.dim, degree=2)
T.read_timestep(label, "T_v2p1", 0, outputPath=DIR)
V.read_timestep(label, "V_v2p1", 0, outputPath=DIR)

dm = mesh.dm
pS, pE = dm.getDepthStratum(0); cS, cE = dm.getHeightStratum(0)
C = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, mesh.dim)
tris = np.asarray([[p - pS for p in dm.getTransitiveClosure(c)[0] if pS <= p < pE]
                   for c in range(cS, cE)])
tri = Triangulation(C[:, 0], C[:, 1], tris)

# fault trace
dl = np.deg2rad(args.fault_dip_deg); P0 = np.array([0., 1.])
dh = np.cos(dl) * np.array([-1., 0.]) - np.sin(dl) * np.array([0., 1.])
L = args.fault_depth / np.sin(dl)
xy = P0[None, :] + np.linspace(0, L, 25)[:, None] * dh[None, :]
zx = (xy[:, 0].min() - 0.22, xy[:, 0].max() + 0.30)
zy = (xy[:, 1].min() - 0.28, xy[:, 1].max() + 0.10)

# fields at vertices
Tv = np.asarray(uw.function.evaluate(T.sym[0], C)).reshape(-1)
Vv = np.asarray(uw.function.evaluate(V.sym, C)).reshape(-1, mesh.dim)
spd = np.linalg.norm(Vv, axis=1)
# check free-slip quality: radial velocity on the outer ring
r = np.linalg.norm(C, axis=1)
outer = np.abs(r - 1.0) < 1e-3
vr_out = (Vv[outer] * (C[outer] / r[outer, None])).sum(1)
print(f"|v|max={spd.max():.3e}  outer-ring |v.n| max={np.abs(vr_out).max():.2e} "
      f"(should be << |v|max if free-slip clean)", flush=True)

fig, ax = plt.subplots(1, 3, figsize=(19, 6.6))
# T + mesh + fault
ax[0].tripcolor(tri, Tv, shading="gouraud", cmap="RdBu_r")
ax[0].triplot(tri, color="k", lw=0.18, alpha=0.35)
ax[0].plot(xy[:, 0], xy[:, 1], "k-", lw=2.0)
ax[0].set_title(f"T + adapted mesh ({label})")
# |v| + streamlines
ax[1].tripcolor(tri, spd, shading="gouraud", cmap="magma")
# streamlines need a grid: sample velocity on a regular grid
gx = np.linspace(-1, 1, 220); gy = np.linspace(-1, 1, 220)
GX, GY = np.meshgrid(gx, gy)
pts = np.column_stack([GX.ravel(), GY.ravel()])
rr = np.linalg.norm(pts, axis=1)
inside = (rr > 0.5) & (rr < 1.0)
UU = np.full(pts.shape[0], np.nan); VV = np.full(pts.shape[0], np.nan)
gv = np.asarray(uw.function.evaluate(V.sym, pts[inside])).reshape(-1, mesh.dim)
UU[inside] = gv[:, 0]; VV[inside] = gv[:, 1]
ax[1].streamplot(GX, GY, UU.reshape(GX.shape), VV.reshape(GX.shape),
                 color="cyan", density=1.4, linewidth=0.6, arrowsize=0.7)
ax[1].plot(xy[:, 0], xy[:, 1], "w-", lw=2.0)
ax[1].set_title("|v| + streamlines")
# fault zoom: mesh + velocity vectors
ax[2].triplot(tri, color="0.3", lw=0.6)
near = (C[:, 0] > zx[0]) & (C[:, 0] < zx[1]) & (C[:, 1] > zy[0]) & (C[:, 1] < zy[1])
sc = max(1, near.sum() // 250)
ax[2].quiver(C[near][::sc, 0], C[near][::sc, 1], Vv[near][::sc, 0], Vv[near][::sc, 1],
             color="b", scale_units="xy", angles="xy", width=0.004)
ax[2].plot(xy[:, 0], xy[:, 1], "r-", lw=2.4)
ax[2].set_xlim(*zx); ax[2].set_ylim(*zy)
ax[2].set_title("fault zoom: mesh + v vectors")
for a in ax:
    a.set_aspect("equal"); a.axis("off")
fig.suptitle(f"TI weak fault in convection ({args.tag}, {label})  "
             f"|v|max={spd.max():.2f}", fontsize=14)
fig.tight_layout()
out = os.path.join(DIR, f"view_{label}.png")
fig.savefig(out, dpi=150)
print("→", out)
