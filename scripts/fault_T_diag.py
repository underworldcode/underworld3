"""Is the blotchy T a real field artifact or a render artifact? Compare:
  (a) T sampled on a regular grid (clean, mesh-topology-independent),
  (b) T at mesh vertices + gouraud triplot (what fault_ti_view used),
  (c) the per-cell T range (max-min within each cell) — large values flag
      genuine sub-cell oscillation (P3 overshoot), small flag a render issue.
Writes <tag>/T_diag.png and prints T stats.
"""
from __future__ import annotations
import os, glob, re, argparse
import numpy as np, underworld3 as uw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

ap = argparse.ArgumentParser()
ap.add_argument('--tag', type=str, default='fault_ti_Ra1e6_fmg')
ap.add_argument('--step', type=str, default='')
args = ap.parse_args()
DIR = os.path.expanduser(f'~/+Simulations/StagnantLid/{args.tag}')
cands = sorted(glob.glob(os.path.join(DIR, "step*.mesh.00000.h5")),
               key=lambda c: int(re.search(r"step(\d+)\.mesh", c).group(1)))
label = args.step or re.search(r"(step\d+)\.mesh", os.path.basename(cands[-1])).group(1)
print(f"diagnosing T of {label}", flush=True)
mesh = uw.discretisation.Mesh(os.path.join(DIR, f"{label}.mesh.00000.h5"))
T = uw.discretisation.MeshVariable("T_v2p1", mesh, 1, degree=3, varsymbol="T")
T.read_timestep(label, "T_v2p1", 0, outputPath=DIR)

print(f"  T.data raw: min={T.data.min():.4f} max={T.data.max():.4f} "
      f"mean={T.data.mean():.4f}  n_dof={T.data.size}", flush=True)
print(f"  T.data outside [0,1]: {(T.data < -1e-9).sum()} below, "
      f"{(T.data > 1+1e-9).sum()} above", flush=True)

# (a) regular grid sample
g = np.linspace(-1, 1, 400); GX, GY = np.meshgrid(g, g)
pts = np.column_stack([GX.ravel(), GY.ravel()])
rr = np.linalg.norm(pts, axis=1); inside = (rr > 0.5) & (rr < 1.0)
Tg = np.full(pts.shape[0], np.nan)
Tg[inside] = np.asarray(uw.function.evaluate(T.sym[0], pts[inside])).reshape(-1)
print(f"  T on grid: min={np.nanmin(Tg):.4f} max={np.nanmax(Tg):.4f}", flush=True)

# (b) vertex values + triangulation
dm = mesh.dm
pS, pE = dm.getDepthStratum(0); cS, cE = dm.getHeightStratum(0)
C = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, mesh.dim)
tris = np.asarray([[p - pS for p in dm.getTransitiveClosure(c)[0] if pS <= p < pE]
                   for c in range(cS, cE)])
tri = Triangulation(C[:, 0], C[:, 1], tris)
Tv = np.asarray(uw.function.evaluate(T.sym[0], C)).reshape(-1)

# (c) per-cell T range from the 3 vertex values (proxy for sub-cell wiggle)
cell_rng = np.array([Tv[t].max() - Tv[t].min() for t in tris])
print(f"  per-cell vertex-T range: median={np.median(cell_rng):.4f} "
      f"p95={np.percentile(cell_rng,95):.4f} max={cell_rng.max():.4f}", flush=True)

fig, ax = plt.subplots(1, 3, figsize=(18, 6.2))
ax[0].imshow(Tg.reshape(GX.shape), origin="lower", extent=[-1, 1, -1, 1],
             cmap="RdBu_r", vmin=0, vmax=1)
ax[0].set_title("(a) T on regular grid (clean sample)")
ax[1].tripcolor(tri, Tv, shading="gouraud", cmap="RdBu_r", vmin=0, vmax=1)
ax[1].set_title("(b) T at vertices + gouraud (what looked blotchy)")
sc = ax[2].tripcolor(tri, cell_rng, cmap="hot_r")
ax[2].set_title("(c) per-cell vertex-T range (sub-cell wiggle)")
plt.colorbar(sc, ax=ax[2], shrink=0.7)
for a in ax:
    a.set_aspect("equal"); a.axis("off")
fig.suptitle(f"T diagnosis — {args.tag}/{label}", fontsize=13)
fig.tight_layout()
out = os.path.join(DIR, "T_diag.png")
fig.savefig(out, dpi=150); print("→", out)
