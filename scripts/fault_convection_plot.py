"""Render a fault-convection snapshot in standard convection style:
temperature on a red-blue colormap, and the actual adapted MESH (triangle
edges). Loads the latest snapshot (moved mesh + T) and writes
<tag>/final_state.png next to the run outputs.
"""
from __future__ import annotations
import os, glob, re, argparse
import numpy as np
import underworld3 as uw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

ap = argparse.ArgumentParser()
ap.add_argument('--tag', type=str, default='fault_dip30_gamg')
ap.add_argument('--fault-dip-deg', type=float, default=30.0)
ap.add_argument('--fault-theta-deg', type=float, default=90.0)
ap.add_argument('--fault-depth', type=float, default=0.225)
ap.add_argument('--fault-dip-dir', type=str, default='east')
args = ap.parse_args()
DIR = os.path.expanduser(f'~/+Simulations/StagnantLid/{args.tag}')


def mesh_triangulation(mesh):
    """matplotlib Triangulation from the DMPlex (serial): vertex coords +
    cell→vertex closure, using local vertex indices (p - pStart)."""
    dm = mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)        # vertices
    cStart, cEnd = dm.getHeightStratum(0)       # cells
    coords = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, mesh.dim)
    tris = []
    for c in range(cStart, cEnd):
        cl = dm.getTransitiveClosure(c)[0]
        tris.append([p - pStart for p in cl if pStart <= p < pEnd])
    return coords, Triangulation(coords[:, 0], coords[:, 1], np.asarray(tris))


def fault_trace():
    delta = np.deg2rad(args.fault_dip_deg)
    th0 = np.deg2rad(args.fault_theta_deg)
    P0 = np.array([np.cos(th0), np.sin(th0)])
    e_hat = np.array([np.cos(th0), np.sin(th0)])
    t_hat = np.array([-np.sin(th0), np.cos(th0)])
    side = 1.0 if args.fault_dip_dir == 'east' else -1.0
    dhat = side * np.cos(delta) * t_hat - np.sin(delta) * e_hat
    L = args.fault_depth / np.sin(delta)
    s = np.linspace(0.0, L, 25)[:, None]
    return P0[None, :] + s * dhat[None, :]


cands = glob.glob(os.path.join(DIR, "step*.mesh.00000.h5"))
steps = sorted(int(re.search(r"step(\d+)\.mesh", os.path.basename(c)).group(1)) for c in cands)
if not steps:
    raise SystemExit(f"no step snapshots in {DIR}")
label = f"step{steps[-1]:04d}"
print(f"loading {label} from {DIR}")

mesh = uw.discretisation.Mesh(os.path.join(DIR, f"{label}.mesh.00000.h5"))
T = uw.discretisation.MeshVariable("T_v2p1", mesh, 1, degree=3, varsymbol="T")
T.read_timestep(label, "T_v2p1", 0, outputPath=DIR)

coords, triang = mesh_triangulation(mesh)
Tv = np.asarray(uw.function.evaluate(T.sym[0], coords)).reshape(-1)
xy = fault_trace()

fig, ax = plt.subplots(1, 2, figsize=(12.5, 6.4))
for a in ax:
    a.set_aspect("equal"); a.axis("off")

# left: temperature, standard red-blue (hot=red, cold=blue)
tpc = ax[0].tripcolor(triang, Tv, shading="gouraud", cmap="RdBu_r",
                      vmin=0.0, vmax=1.0)
ax[0].plot(xy[:, 0], xy[:, 1], color="k", lw=1.5)
ax[0].set_title(f"Temperature ({label})")
plt.colorbar(tpc, ax=ax[0], shrink=0.7, label="T")

# right: the adapted mesh (triangle edges) + fault trace
ax[1].triplot(triang, color="0.35", lw=0.3)
ax[1].plot(xy[:, 0], xy[:, 1], color="r", lw=1.8)
ax[1].set_title("Adapted mesh (fault in red)")

fig.tight_layout()
out = os.path.join(DIR, "final_state.png")
fig.savefig(out, dpi=150)
print("→", out)
print(f"T range [{Tv.min():.3f}, {Tv.max():.3f}], n_vertices={len(coords)}, "
      f"n_cells={triang.triangles.shape[0]}")
