"""Fair head-to-head: take the developing T from a real convection run to set
the |∇T| metric, then on a fresh UNIFORM res32 mesh apply the ANISOTROPIC
mover at several R (resolution_ratio) values plus the equidistribution mmpde
(no R), with the SAME combined metric rho = rho_T(|∇T|) * fault_rho. Render
each adapted mesh (full + fault zoom) so we can see how R captures the fault.

Writes ~/+Simulations/StagnantLid/<src-tag>/R_sweep.png
"""
from __future__ import annotations
import os, glob, re, argparse
import numpy as np, sympy, underworld3 as uw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

ap = argparse.ArgumentParser()
ap.add_argument('--src-tag', type=str, default='fault_iso_Ra1e6')   # run whose T we use
ap.add_argument('--res', type=int, default=32)                      # fresh uniform mesh
ap.add_argument('--fault-refine-amp', type=float, default=18.0)
ap.add_argument('--fault-width', type=float, default=0.05)
ap.add_argument('--fault-dip-deg', type=float, default=30.0)
ap.add_argument('--fault-depth', type=float, default=0.225)
ap.add_argument('--strategy', type=str, default='med')
args = ap.parse_args()
SRC = os.path.expanduser(f'~/+Simulations/StagnantLid/{args.src_tag}')

# --- load the latest snapshot T from the run ---
cands = sorted(glob.glob(os.path.join(SRC, "step*.mesh.00000.h5")),
               key=lambda c: int(re.search(r"step(\d+)\.mesh", c).group(1)))
label = re.search(r"(step\d+)\.mesh", os.path.basename(cands[-1])).group(1)
print(f"using T from {label} of {args.src_tag}", flush=True)
msnap = uw.discretisation.Mesh(cands[-1])
Tsnap = uw.discretisation.MeshVariable("T_v2p1", msnap, 1, degree=3, varsymbol="T")
Tsnap.read_timestep(label, "T_v2p1", 0, outputPath=SRC)

# fault geometry (shared)
delta = np.deg2rad(args.fault_dip_deg)
P0 = np.array([0., 1.]); t_hat = np.array([-1., 0.]); e_hat = np.array([0., 1.])
dh = np.cos(delta)*t_hat - np.sin(delta)*e_hat
xyf = P0[None, :] + np.linspace(0, args.fault_depth/np.sin(delta), 25)[:, None]*dh[None, :]
zx = (xyf[:, 0].min()-0.22, xyf[:, 0].max()+0.30)
zy = (xyf[:, 1].min()-0.28, xyf[:, 1].max()+0.10)


def tris_of(mesh):
    dm = mesh.dm
    pS, pE = dm.getDepthStratum(0); cS, cE = dm.getHeightStratum(0)
    return np.asarray([[p - pS for p in dm.getTransitiveClosure(c)[0] if pS <= p < pE]
                       for c in range(cS, cE)])


def run_one(method, R):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1.0/args.res, qdegree=3)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)
    # interpolate the run's T onto the fresh uniform mesh
    T.data[:, 0] = np.asarray(uw.function.evaluate(Tsnap.sym[0], T.coords)).reshape(-1)
    fault = uw.meshing.Surface("f", mesh, np.column_stack([xyf, np.zeros(25)]), symbol="F")
    fault.discretize(); _ = fault.distance
    rho_T = uw.meshing.metric_density_from_gradient(mesh, T, strategy=args.strategy, name="r")
    d = fault.distance.sym[0]
    rho = rho_T * (1.0 + args.fault_refine_amp*sympy.exp(-(d/(1.5*args.fault_width))**2))
    mk = dict(relax=0.2, n_outer=12)
    if method == "anisotropic":
        mk["resolution_ratio"] = R
    C0 = np.asarray(mesh.X.coords).copy()
    uw.meshing.smooth_mesh_interior(mesh, metric=rho, method=method, skip_threshold=None,
                                    slip_surfaces=True, method_kwargs=mk, verbose=False)
    C1 = np.asarray(mesh.X.coords).copy()
    disp = np.linalg.norm(C1-C0, axis=1)
    dist = np.array([np.min(np.linalg.norm(xyf-c, axis=1)) for c in C0])
    near = dist < 0.12
    lab = f"{method} R{R}" if method == "anisotropic" else "mmpde (no R)"
    print(f"{lab:18s} near-fault disp={disp[near].mean():.4f} far={disp[~near].mean():.4f}", flush=True)
    return C1, tris_of(mesh), lab


CONFIGS = [("anisotropic", 2), ("anisotropic", 3), ("anisotropic", 5),
           ("anisotropic", 8), ("mmpde", None)]
results = [run_one(m, R) for m, R in CONFIGS]

# big fault-zoom-focused figure: top row small full mesh, bottom row LARGE zoom
fig = plt.figure(figsize=(5.2*len(results), 12.5))
gs = fig.add_gridspec(2, len(results), height_ratios=[1.0, 2.3])
for j, (C, tris, lab) in enumerate(results):
    tri = Triangulation(C[:, 0], C[:, 1], tris)
    a0 = fig.add_subplot(gs[0, j]); a1 = fig.add_subplot(gs[1, j])
    a0.triplot(tri, color="0.3", lw=0.25); a0.plot(xyf[:, 0], xyf[:, 1], "r-", lw=1.3)
    a0.set_title(lab, fontsize=14)
    a1.triplot(tri, color="0.2", lw=0.9); a1.plot(xyf[:, 0], xyf[:, 1], "r-", lw=2.4)
    a1.set_xlim(*zx); a1.set_ylim(*zy)
    for a in (a0, a1):
        a.set_aspect("equal"); a.axis("off")
fig.suptitle(f"anisotropic-mover R sweep vs mmpde — T from {args.src_tag}/{label}, "
             f"uniform res{args.res}, fault_amp={args.fault_refine_amp}  (bottom = fault zoom)",
             fontsize=15)
fig.tight_layout()
out = os.path.join(SRC, "R_sweep.png")
fig.savefig(out, dpi=180)
print("→", out)
