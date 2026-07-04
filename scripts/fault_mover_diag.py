"""Diagnose why the fault metric isn't bunching nodes on the fault.
Renders, on a uniform annulus with a dipping fault and uniform T:
  (1) the metric field rho the mover is given (should be HIGH on the fault),
  (2) the mesh after the ANISOTROPIC mover,
  (3) the mesh after the OT/equidistribution mover,
each with a displacement quiver, zoomed on the fault.
Writes ~/+Simulations/StagnantLid/fault_mover_diag/diag.png
"""
from __future__ import annotations
import os, argparse, warnings
import numpy as np, sympy, underworld3 as uw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument('--res', type=int, default=24)
ap.add_argument('--amp', type=float, default=18.0)
ap.add_argument('--width', type=float, default=0.075)
ap.add_argument('--R', type=float, default=4.0)
ap.add_argument('--include-gradient', action='store_true',
                help='multiply by metric_density_from_gradient (default off, '
                     'isolate the fault term)')
args = ap.parse_args()
OUT = os.path.expanduser('~/+Simulations/StagnantLid/fault_mover_diag')
os.makedirs(OUT, exist_ok=True)


def tris_of(mesh):
    dm = mesh.dm
    pS, pE = dm.getDepthStratum(0); cS, cE = dm.getHeightStratum(0)
    return np.asarray([[p - pS for p in dm.getTransitiveClosure(c)[0]
                        if pS <= p < pE] for c in range(cS, cE)])


def fault_of(mesh):
    delta = np.deg2rad(30.0)
    P0 = np.array([0.0, 1.0]); t = np.array([-1.0, 0.0]); e = np.array([0.0, 1.0])
    dhat = np.cos(delta) * t - np.sin(delta) * e
    s = np.linspace(0, 0.3 / np.sin(delta), 25)[:, None]
    xy = P0[None, :] + s * dhat[None, :]
    f = uw.meshing.Surface("fault", mesh, np.column_stack([xy, np.zeros(25)]), symbol="F")
    f.discretize(); _ = f.distance
    return f, xy


def metric_expr(mesh, fault, T):
    d = fault.distance.sym[0]
    frho = 1.0 + args.amp * sympy.exp(-(d / args.width) ** 2)
    if args.include_gradient:
        rho_T = uw.meshing.metric_density_from_gradient(mesh, T, strategy="med", name="r")
        return rho_T * frho
    return frho


def run_mover(method):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5,
                              cellSize=1.0 / args.res, qdegree=3)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3); T.data[:] = 0.5
    fault, xy = fault_of(mesh)
    tris = tris_of(mesh)
    C0 = np.asarray(mesh.X.coords).copy()
    rho = metric_expr(mesh, fault, T)
    mk = dict(relax=0.2, n_outer=12)
    if method == "anisotropic":
        mk["resolution_ratio"] = args.R
    uw.meshing.smooth_mesh_interior(
        mesh, metric=rho, method=method, skip_threshold=None,
        slip_surfaces=True, method_kwargs=mk, verbose=False)
    C1 = np.asarray(mesh.X.coords).copy()
    return C0, C1, tris, xy


# metric field (on the undeformed mesh)
mesh0 = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5,
                           cellSize=1.0 / args.res, qdegree=3)
T0 = uw.discretisation.MeshVariable("T", mesh0, 1, degree=3); T0.data[:] = 0.5
fault0, xy0 = fault_of(mesh0)
tris0 = tris_of(mesh0)
C0 = np.asarray(mesh0.X.coords)
rho_vals = np.asarray(uw.function.evaluate(metric_expr(mesh0, fault0, T0), C0)).reshape(-1)
print(f"metric rho: min={rho_vals.min():.2f} max={rho_vals.max():.2f} "
      f"(peak should be ~{1+args.amp:.0f} on the fault)")

Ca, Ca1, trisa, xya = run_mover("anisotropic")
Co, Co1, triso, xyo = run_mover("ot")

zx = (xy0[:, 0].min() - 0.2, xy0[:, 0].max() + 0.25)
zy = (xy0[:, 1].min() - 0.22, xy0[:, 1].max() + 0.1)

fig, ax = plt.subplots(1, 3, figsize=(18, 6.4))
# (1) metric field
sc = ax[0].tripcolor(Triangulation(C0[:, 0], C0[:, 1], tris0), rho_vals,
                     shading="gouraud", cmap="hot_r")
ax[0].plot(xy0[:, 0], xy0[:, 1], "c-", lw=2)
ax[0].set_title("metric rho given to the mover (peak = fault)")
plt.colorbar(sc, ax=ax[0], shrink=0.7)
# (2) anisotropic after, with quiver
ax[1].triplot(Triangulation(Ca1[:, 0], Ca1[:, 1], trisa), color="0.4", lw=0.5)
d = Ca1 - Ca
ax[1].quiver(Ca[:, 0], Ca[:, 1], d[:, 0], d[:, 1], color="b",
             angles="xy", scale_units="xy", scale=1, width=0.003)
ax[1].plot(xya[:, 0], xya[:, 1], "r-", lw=2)
ax[1].set_title("anisotropic mover (mesh + node displacement)")
# (3) OT after, with quiver
ax[2].triplot(Triangulation(Co1[:, 0], Co1[:, 1], triso), color="0.4", lw=0.5)
d = Co1 - Co
ax[2].quiver(Co[:, 0], Co[:, 1], d[:, 0], d[:, 1], color="b",
             angles="xy", scale_units="xy", scale=1, width=0.003)
ax[2].plot(xyo[:, 0], xyo[:, 1], "r-", lw=2)
ax[2].set_title("OT / equidistribute mover")
for a in ax:
    a.set_xlim(*zx); a.set_ylim(*zy); a.set_aspect("equal"); a.axis("off")
fig.tight_layout()
out = os.path.join(OUT, "diag.png")
fig.savefig(out, dpi=150)
print("→", out)
for nm, C, C1 in [("anisotropic", Ca, Ca1), ("ot", Co, Co1)]:
    disp = np.linalg.norm(C1 - C, axis=1)
    dist = np.array([np.min(np.linalg.norm(xy0 - c, axis=1)) for c in C])
    near = dist < 0.12
    print(f"{nm:12s} max|disp|={disp.max():.3f}  near-fault mean={disp[near].mean():.3f}  "
          f"far={disp[~near].mean():.4f}")
