"""Single OT_adapt move with the analytic fault band as extra_density, on a
REAL convection T (loaded from a prior run snapshot). Confirms (a) the tanh
window + perpendicular gaussian JIT-compile in the OT mover (no `re` error),
(b) the mesh is sliver-free with nice |grad T| grading (Wednesday quality),
(c) the fault corridor is refined. Renders full mesh + fault zoom + T overlay
into ~/+Simulations/StagnantLid/<tag>/ot_preview.png.
"""
from __future__ import annotations
import os, glob, re, argparse
import numpy as np, sympy, underworld3 as uw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

ap = argparse.ArgumentParser()
ap.add_argument('--src-tag', type=str, default='fault_iso_Ra1e6')  # run whose T we use
ap.add_argument('--res', type=int, default=32)
ap.add_argument('--refinement', type=float, default=5.0)           # OT refinement
ap.add_argument('--fault-refine-amp', type=float, default=18.0)
ap.add_argument('--fault-width', type=float, default=0.05)
ap.add_argument('--fault-dip-deg', type=float, default=30.0)
ap.add_argument('--fault-depth', type=float, default=0.225)
ap.add_argument('--n-moves', type=int, default=3)                  # repeat OT_adapt
ap.add_argument('--tag', type=str, default='fault_ot_preview')
args = ap.parse_args()
SRC = os.path.expanduser(f'~/+Simulations/StagnantLid/{args.src_tag}')
OUT = os.path.expanduser(f'~/+Simulations/StagnantLid/{args.tag}')
os.makedirs(OUT, exist_ok=True)
refine_w = 1.5 * args.fault_width


def tris_of(mesh):
    dm = mesh.dm
    pS, pE = dm.getDepthStratum(0); cS, cE = dm.getHeightStratum(0)
    return np.asarray([[p - pS for p in dm.getTransitiveClosure(c)[0]
                        if pS <= p < pE] for c in range(cS, cE)])


mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5,
                          cellSize=1.0 / args.res, qdegree=3)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)

# load a real convection T from the source run
cands = sorted(glob.glob(os.path.join(SRC, "step*.mesh.00000.h5")),
               key=lambda c: int(re.search(r"step(\d+)\.mesh", c).group(1)))
label = re.search(r"(step\d+)\.mesh", os.path.basename(cands[-1])).group(1)
print(f"using T from {label} of {args.src_tag}", flush=True)
msnap = uw.discretisation.Mesh(cands[-1])
Tsnap = uw.discretisation.MeshVariable("Ts", msnap, 1, degree=3, varsymbol="T")
Tsnap.read_timestep(label, "T_v2p1", 0, outputPath=SRC)
T.data[:, 0] = np.asarray(uw.function.evaluate(Tsnap.sym[0], T.coords)).reshape(-1)

# fault geometry (12 o'clock, dipping)
delta = np.deg2rad(args.fault_dip_deg)
P0 = np.array([0.0, 1.0]); t_hat = np.array([-1.0, 0.0]); e_hat = np.array([0.0, 1.0])
dhat = np.cos(delta) * t_hat - np.sin(delta) * e_hat
n_vec = np.array([-dhat[1], dhat[0]]); n_vec /= np.linalg.norm(n_vec)
L = args.fault_depth / np.sin(delta)
xy = P0[None, :] + np.linspace(0, L, 25)[:, None] * dhat[None, :]

# analytic fault band in mesh coords (same as the driver)
Xc = mesh.CoordinateSystem.X
perp = (Xc[0] - float(P0[0])) * float(n_vec[0]) + (Xc[1] - float(P0[1])) * float(n_vec[1])
sal = (Xc[0] - float(P0[0])) * float(dhat[0]) + (Xc[1] - float(P0[1])) * float(dhat[1])
window = (sympy.Rational(1, 4) * (1 + sympy.tanh(sal / refine_w))
          * (1 - sympy.tanh((sal - float(L)) / refine_w)))
fault_band = 1.0 + args.fault_refine_amp * sympy.exp(-(perp / refine_w) ** 2) * window

C0 = np.asarray(mesh.X.coords).copy()
tris0 = tris_of(mesh)
for k in range(args.n_moves):
    moved = mesh.OT_adapt(T, refinement=args.refinement, coarsening="auto",
                          metric_choice="front-following",
                          grad_smoothing_length="auto",
                          extra_density=fault_band,
                          fields_to_remap=[T], skip_threshold=None, verbose=True)
    print(f"move {k+1}/{args.n_moves}: moved={moved}", flush=True)
C1 = np.asarray(mesh.X.coords).copy()
tris1 = tris_of(mesh)

# mesh quality
try:
    q = np.asarray(mesh.quality())
    qstr = f"quality min/mean={q.min():.3f}/{q.mean():.3f}"
except Exception as e:
    qstr = f"(quality n/a: {str(e)[:40]})"
disp = np.linalg.norm(C1 - C0, axis=1)
dist = np.array([np.min(np.linalg.norm(xy - c, axis=1)) for c in C0])
near = dist < 0.12
print(f"n={len(C0)}  max|disp|={disp.max():.3f}  near-fault|disp|={disp[near].mean():.3f}"
      f"  far={disp[~near].mean():.4f}  {qstr}", flush=True)

zx = (xy[:, 0].min() - 0.22, xy[:, 0].max() + 0.30)
zy = (xy[:, 1].min() - 0.28, xy[:, 1].max() + 0.10)
Tv = np.asarray(uw.function.evaluate(T.sym[0], C1)).reshape(-1)

fig, ax = plt.subplots(1, 3, figsize=(18, 6.4))
tri1 = Triangulation(C1[:, 0], C1[:, 1], tris1)
ax[0].triplot(tri1, color="0.3", lw=0.3); ax[0].plot(xy[:, 0], xy[:, 1], "r-", lw=1.6)
ax[0].set_title(f"OT_adapt x{args.n_moves} (refine={args.refinement}) — full mesh")
ax[1].triplot(tri1, color="0.2", lw=0.8); ax[1].plot(xy[:, 0], xy[:, 1], "r-", lw=2.4)
ax[1].set_xlim(*zx); ax[1].set_ylim(*zy)
ax[1].set_title(f"fault zoom — {qstr}")
ax[2].tripcolor(tri1, Tv, shading="gouraud", cmap="RdBu_r")
ax[2].triplot(tri1, color="k", lw=0.15, alpha=0.4)
ax[2].plot(xy[:, 0], xy[:, 1], "k-", lw=1.4)
ax[2].set_title("T + mesh")
for a in ax:
    a.set_aspect("equal"); a.axis("off")
fig.suptitle(f"OT reset mover + analytic fault band  (T from {args.src_tag}/{label}, "
             f"uniform res{args.res}, amp={args.fault_refine_amp})", fontsize=14)
fig.tight_layout()
out = os.path.join(OUT, "ot_preview.png")
fig.savefig(out, dpi=160)
print("→", out)
