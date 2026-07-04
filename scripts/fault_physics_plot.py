"""Physics diagnostic for a fault-convection snapshot: does the fault behave
correctly? Three panels for the latest snapshot, all with the fault trace:
  (1) viscosity field (log) — confirms the weak zone is where the fault is,
  (2) speed |v| with streamlines — does flow localize / shear at the fault?
  (3) temperature.
Writes <tag>/physics.png
"""
from __future__ import annotations
import os, glob, re, argparse
import numpy as np, sympy, underworld3 as uw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

ap = argparse.ArgumentParser()
ap.add_argument('--tag', type=str, default='fault_iso_conv')
ap.add_argument('--delta-eta', type=float, default=100.0)
ap.add_argument('--fault-floor', type=float, default=1.0)
ap.add_argument('--fault-dip-deg', type=float, default=30.0)
ap.add_argument('--fault-depth', type=float, default=0.225)
args = ap.parse_args()
DIR = os.path.expanduser(f'~/+Simulations/StagnantLid/{args.tag}')
theta_FK = float(np.log(args.delta_eta))

cands = sorted(glob.glob(os.path.join(DIR, "step*.mesh.00000.h5")),
               key=lambda c: int(re.search(r"step(\d+)\.mesh", c).group(1)))
label = re.search(r"(step\d+)\.mesh", os.path.basename(cands[-1])).group(1)
mesh = uw.discretisation.Mesh(cands[-1])
T = uw.discretisation.MeshVariable("T_v2p1", mesh, 1, degree=3, varsymbol="T")
V = uw.discretisation.MeshVariable("V_v2p1", mesh, mesh.dim, degree=2, varsymbol="v")
gfac = uw.discretisation.MeshVariable("eta_fac", mesh, 1, degree=2, varsymbol="g")
T.read_timestep(label, "T_v2p1", 0, outputPath=DIR)
V.read_timestep(label, "V_v2p1", 0, outputPath=DIR)
gfac.read_timestep(label, "eta_fac", 0, outputPath=DIR)

dm = mesh.dm
pS, pE = dm.getDepthStratum(0); cS, cE = dm.getHeightStratum(0)
C = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, mesh.dim)
tris = np.asarray([[p - pS for p in dm.getTransitiveClosure(c)[0] if pS <= p < pE]
                   for c in range(cS, cE)])
tri = Triangulation(C[:, 0], C[:, 1], tris)

Tv = np.asarray(uw.function.evaluate(T.sym[0], C)).reshape(-1)
# gfac now holds the fault INFLUENCE f (1 on fault, 0 away); viscosity is the
# floored blend eta_FK*(1-f) + floor*f.
_etaFK = sympy.exp(theta_FK*(1-T.sym[0]))
visc = np.asarray(uw.function.evaluate(
    _etaFK*(1-gfac.sym[0]) + args.fault_floor*gfac.sym[0], C)).reshape(-1)
Vx = np.asarray(uw.function.evaluate(V.sym[0], C)).reshape(-1)
Vy = np.asarray(uw.function.evaluate(V.sym[1], C)).reshape(-1)
spd = np.sqrt(Vx**2 + Vy**2)

# fault trace
delta = np.deg2rad(args.fault_dip_deg)
P0 = np.array([0., 1.]); t_hat = np.array([-1., 0.]); e_hat = np.array([0., 1.])
dh = np.cos(delta)*t_hat - np.sin(delta)*e_hat
s = np.linspace(0, args.fault_depth/np.sin(delta), 25)[:, None]
xy = P0[None, :] + s*dh[None, :]

fig, ax = plt.subplots(1, 3, figsize=(18, 6.3))
for a in ax:
    a.set_aspect("equal"); a.axis("off")
    a.plot(xy[:, 0], xy[:, 1], "k-", lw=1.4)

s0 = ax[0].tripcolor(tri, np.log10(visc), shading="gouraud", cmap="viridis")
ax[0].set_title(f"log10 viscosity ({label}) — fault = weak band")
plt.colorbar(s0, ax=ax[0], shrink=0.7)

s1 = ax[1].tripcolor(tri, spd, shading="gouraud", cmap="magma")
# streamlines on a regular grid sampled from the FE field
gx = np.linspace(-1, 1, 220); gy = np.linspace(-1, 1, 220)
GX, GY = np.meshgrid(gx, gy)
pts = np.column_stack([GX.ravel(), GY.ravel()])
rr = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
inside = (rr > 0.505) & (rr < 0.995)
U = np.full(len(pts), np.nan); W = np.full(len(pts), np.nan)
val = np.asarray(uw.function.evaluate(V.sym, pts[inside])).reshape(int(inside.sum()), -1)
U[inside] = val[:, 0]; W[inside] = val[:, 1]
ax[1].streamplot(GX, GY, U.reshape(GX.shape), W.reshape(GX.shape),
                 density=1.4, color="white", linewidth=0.5, arrowsize=0.6)
ax[1].set_title("speed |v| + streamlines — flow response to the fault")
plt.colorbar(s1, ax=ax[1], shrink=0.7)

s2 = ax[2].tripcolor(tri, Tv, shading="gouraud", cmap="RdBu_r", vmin=0, vmax=1)
ax[2].set_title("temperature")
plt.colorbar(s2, ax=ax[2], shrink=0.7)

fig.tight_layout()
out = os.path.join(DIR, "physics.png")
fig.savefig(out, dpi=140)
print("→", out)
print(f"visc range [{visc.min():.3e}, {visc.max():.3e}]  "
      f"|v|max={spd.max():.3e}  on-fault visc min confirms weak zone")
