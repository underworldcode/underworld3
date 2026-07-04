"""High-res before/after view of ONE fault-driven mesh adapt, to tune the
adaptation strength (resolution_ratio, fault amp) before a time-loop run.
Uniform annulus + dipping fault + uniform T (so only the fault drives the
move). Renders the mesh before vs after, full annulus and zoomed on the
fault, into ~/+Simulations/StagnantLid/<tag>/adapt_preview.png.
"""
from __future__ import annotations
import os, argparse
import numpy as np, sympy, underworld3 as uw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

ap = argparse.ArgumentParser()
ap.add_argument('--res', type=int, default=24)
ap.add_argument('--cell-outer', type=float, default=0.0)
ap.add_argument('--cell-inner', type=float, default=0.0)
ap.add_argument('--resolution-ratio', type=float, default=2.5)
ap.add_argument('--fault-refine-amp', type=float, default=12.0)
ap.add_argument('--fault-width', type=float, default=0.05)
ap.add_argument('--fault-dip-deg', type=float, default=30.0)
ap.add_argument('--fault-depth', type=float, default=0.225)
ap.add_argument('--n-outer', type=int, default=12)
ap.add_argument('--method', type=str, default='anisotropic')   # or 'ot'
ap.add_argument('--tag', type=str, default='fault_preview_uniform24')
args = ap.parse_args()
OUT = os.path.expanduser(f'~/+Simulations/StagnantLid/{args.tag}')
os.makedirs(OUT, exist_ok=True)


def tris_of(mesh):
    dm = mesh.dm
    pS, pE = dm.getDepthStratum(0); cS, cE = dm.getHeightStratum(0)
    return np.asarray([[p - pS for p in dm.getTransitiveClosure(c)[0]
                        if pS <= p < pE] for c in range(cS, cE)])


if getattr(args, "cell_outer", 0) > 0 and getattr(args, "cell_inner", 0) > 0:
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5,
                              cellSizeOuter=args.cell_outer,
                              cellSizeInner=args.cell_inner, qdegree=3)
else:
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5,
                              cellSize=1.0 / args.res, qdegree=3)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)
T.data[:] = 0.5

delta = np.deg2rad(args.fault_dip_deg)
P0 = np.array([0.0, 1.0]); t_hat = np.array([-1.0, 0.0]); e_hat = np.array([0.0, 1.0])
dhat = np.cos(delta) * t_hat - np.sin(delta) * e_hat
L = args.fault_depth / np.sin(delta)
s = np.linspace(0, L, 25)[:, None]
xy = P0[None, :] + s * dhat[None, :]
fault = uw.meshing.Surface("fault", mesh, np.column_stack([xy, np.zeros(25)]), symbol="F")
fault.discretize()
_ = fault.distance

tris = tris_of(mesh)
coords0 = np.asarray(mesh.X.coords).copy()

rho_T = uw.meshing.metric_density_from_gradient(mesh, T, strategy="med", name="r")
d = fault.distance.sym[0]
fault_rho = 1.0 + args.fault_refine_amp * sympy.exp(-(d / (1.5 * args.fault_width)) ** 2)
_mk = dict(relax=0.2, n_outer=args.n_outer)
if args.method in ("anisotropic", "aniso", "tensor"):
    _mk["resolution_ratio"] = args.resolution_ratio
uw.meshing.smooth_mesh_interior(
    mesh, metric=rho_T * fault_rho, method=args.method,
    skip_threshold=None, slip_surfaces=True,
    method_kwargs=_mk, verbose=False)
coords1 = np.asarray(mesh.X.coords).copy()

# fault bounding box for the zoom
zx = (xy[:, 0].min() - 0.18, xy[:, 0].max() + 0.18)
zy = (xy[:, 1].min() - 0.18, xy[:, 1].max() + 0.08)

fig, ax = plt.subplots(2, 2, figsize=(12, 12))
panels = [(coords0, "before"), (coords1, "after (slip ON)")]
for col, (C, lbl) in enumerate(panels):
    tri = Triangulation(C[:, 0], C[:, 1], tris)
    # full annulus
    ax[0, col].triplot(tri, color="0.3", lw=0.4)
    ax[0, col].plot(xy[:, 0], xy[:, 1], "r-", lw=2)
    ax[0, col].set_aspect("equal"); ax[0, col].axis("off")
    ax[0, col].set_title(f"mesh {lbl}")
    # zoom on the fault
    ax[1, col].triplot(tri, color="0.3", lw=0.6)
    ax[1, col].plot(xy[:, 0], xy[:, 1], "r-", lw=2.2)
    ax[1, col].set_xlim(*zx); ax[1, col].set_ylim(*zy)
    ax[1, col].set_aspect("equal"); ax[1, col].axis("off")
    ax[1, col].set_title(f"fault zoom {lbl}")

fig.suptitle(f"uniform res{args.res}, R={args.resolution_ratio}, "
             f"fault_amp={args.fault_refine_amp}, n_outer={args.n_outer}",
             fontsize=13)
fig.tight_layout()
out = os.path.join(OUT, "adapt_preview.png")
fig.savefig(out, dpi=150)
print("→", out)

# how much did interior nodes migrate toward the fault?
disp = np.linalg.norm(coords1 - coords0, axis=1)
dist_to_fault = np.array([np.min(np.linalg.norm(xy - c, axis=1)) for c in coords0])
near = dist_to_fault < 0.15
try:
    q = mesh.quality()
    qmin = float(np.min(q)) if hasattr(q, "__len__") else float(q)
    qstr = f"  mesh.quality min/mean={qmin:.3f}"
except Exception as e:
    qstr = f"  (quality n/a: {str(e)[:40]})"
print(f"n_nodes={len(coords0)}  max|disp|={disp.max():.3f}  "
      f"mean|disp| near-fault={disp[near].mean():.3f}  far={disp[~near].mean():.4f}"
      f"{qstr}")
