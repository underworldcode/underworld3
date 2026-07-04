"""Zoom on the fault region of the latest adapted mesh, with the |∇T|×fault
metric overlaid, to see whether the fault is actually being refined or just
swamped by the convection (|∇T|) refinement. Writes <tag>/mesh_zoom.png.
"""
from __future__ import annotations
import os, glob, re, argparse
import numpy as np, sympy, underworld3 as uw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

ap = argparse.ArgumentParser()
ap.add_argument('--tag', type=str, default='fault_iso_Ra1e6')
ap.add_argument('--fault-dip-deg', type=float, default=30.0)
ap.add_argument('--fault-depth', type=float, default=0.225)
ap.add_argument('--fault-width', type=float, default=0.05)
ap.add_argument('--fault-refine-amp', type=float, default=18.0)
ap.add_argument('--strategy', type=str, default='med')
args = ap.parse_args()
DIR = os.path.expanduser(f'~/+Simulations/StagnantLid/{args.tag}')

cands = sorted(glob.glob(os.path.join(DIR, "step*.mesh.00000.h5")),
               key=lambda c: int(re.search(r"step(\d+)\.mesh", c).group(1)))
label = re.search(r"(step\d+)\.mesh", os.path.basename(cands[-1])).group(1)
mesh = uw.discretisation.Mesh(cands[-1])
T = uw.discretisation.MeshVariable("T_v2p1", mesh, 1, degree=3, varsymbol="T")
T.read_timestep(label, "T_v2p1", 0, outputPath=DIR)

dm = mesh.dm
pS, pE = dm.getDepthStratum(0); cS, cE = dm.getHeightStratum(0)
C = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, mesh.dim)
tris = np.asarray([[p - pS for p in dm.getTransitiveClosure(c)[0] if pS <= p < pE]
                   for c in range(cS, cE)])
tri = Triangulation(C[:, 0], C[:, 1], tris)

# fault trace + a fault Surface to recover the distance/metric on this mesh
delta = np.deg2rad(args.fault_dip_deg)
P0 = np.array([0., 1.]); t_hat = np.array([-1., 0.]); e_hat = np.array([0., 1.])
dh = np.cos(delta)*t_hat - np.sin(delta)*e_hat
s = np.linspace(0, args.fault_depth/np.sin(delta), 25)[:, None]
xy = P0[None, :] + s*dh[None, :]
fault = uw.meshing.Surface("f", mesh, np.column_stack([xy, np.zeros(25)]), symbol="F")
fault.discretize(); _ = fault.distance

# the combined metric the mover sees: rho_T(|grad T|) * fault_rho
rho_T = uw.meshing.metric_density_from_gradient(mesh, T, strategy=args.strategy, name="r")
d = fault.distance.sym[0]
refine_w = 1.5*args.fault_width
fault_rho = 1.0 + args.fault_refine_amp*sympy.exp(-(d/refine_w)**2)
rhoT_v = np.asarray(uw.function.evaluate(rho_T, C)).reshape(-1)
frho_v = np.asarray(uw.function.evaluate(fault_rho, C)).reshape(-1)
rho_v = rhoT_v * frho_v
print(f"metric: rho_T [{rhoT_v.min():.2f},{rhoT_v.max():.2f}]  "
      f"fault_rho [{frho_v.min():.2f},{frho_v.max():.2f}]  "
      f"combined [{rho_v.min():.2f},{rho_v.max():.2f}]", flush=True)

zx = (xy[:, 0].min()-0.22, xy[:, 0].max()+0.30)
zy = (xy[:, 1].min()-0.28, xy[:, 1].max()+0.10)

fig, ax = plt.subplots(1, 3, figsize=(18, 6.2))
# full mesh
ax[0].triplot(tri, color="0.35", lw=0.25); ax[0].plot(xy[:, 0], xy[:, 1], "r-", lw=1.6)
ax[0].set_title(f"full adapted mesh ({label})")
# zoom on fault
ax[1].triplot(tri, color="0.25", lw=0.6); ax[1].plot(xy[:, 0], xy[:, 1], "r-", lw=2.2)
ax[1].set_xlim(*zx); ax[1].set_ylim(*zy)
ax[1].set_title("ZOOM on fault — is it refined vs surroundings?")
# combined metric (what the mover targets), zoomed
sc = ax[2].tripcolor(tri, np.log10(rho_v), shading="gouraud", cmap="hot_r")
ax[2].plot(xy[:, 0], xy[:, 1], "c-", lw=2); ax[2].set_xlim(*zx); ax[2].set_ylim(*zy)
ax[2].set_title("log10 combined metric rho_T*fault_rho (zoom)")
plt.colorbar(sc, ax=ax[2], shrink=0.7)
for a in ax:
    a.set_aspect("equal"); a.axis("off")
fig.tight_layout()
out = os.path.join(DIR, "mesh_zoom.png")
fig.savefig(out, dpi=150)
print("→", out)
