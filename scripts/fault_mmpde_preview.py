"""Verify the CORRECTED MMPDE recipe (matches stagnant_lid Wednesday config):
  metric = metric_density_from_gradient(T, refinement=R, front-following)
           * (1 + amp*exp(-(d/w)^2))      # fault band, signed Surface distance
  smooth_mesh_interior(method="mmpde", method_kwargs=step_frac/accel=cg/momentum)
on a REAL convection T. Renders full mesh + fault zoom + T overlay so we can
judge grading quality AND fault refinement together. Sweeps a couple of R.
Writes ~/+Simulations/StagnantLid/<tag>/mmpde_preview.png.
"""
from __future__ import annotations
import os, glob, re, argparse
import numpy as np, sympy, underworld3 as uw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

ap = argparse.ArgumentParser()
ap.add_argument('--src-tag', type=str, default='fault_iso_Ra1e6')
ap.add_argument('--res', type=int, default=32)
ap.add_argument('--fault-refine-amp', type=float, default=18.0)
ap.add_argument('--fault-width', type=float, default=0.05)
ap.add_argument('--fault-dip-deg', type=float, default=30.0)
ap.add_argument('--fault-depth', type=float, default=0.225)
ap.add_argument('--combine', type=str, default='max', choices=['max', 'product'])
ap.add_argument('--tag', type=str, default='fault_mmpde_preview')
args = ap.parse_args()
SRC = os.path.expanduser(f'~/+Simulations/StagnantLid/{args.src_tag}')
OUT = os.path.expanduser(f'~/+Simulations/StagnantLid/{args.tag}')
os.makedirs(OUT, exist_ok=True)
refine_w = 1.5 * args.fault_width

cands = sorted(glob.glob(os.path.join(SRC, "step*.mesh.00000.h5")),
               key=lambda c: int(re.search(r"step(\d+)\.mesh", c).group(1)))
label = re.search(r"(step\d+)\.mesh", os.path.basename(cands[-1])).group(1)
print(f"using T from {label} of {args.src_tag}", flush=True)
msnap = uw.discretisation.Mesh(cands[-1])
Tsnap = uw.discretisation.MeshVariable("Ts", msnap, 1, degree=3, varsymbol="T")
Tsnap.read_timestep(label, "T_v2p1", 0, outputPath=SRC)

delta = np.deg2rad(args.fault_dip_deg)
P0 = np.array([0.0, 1.0]); t_hat = np.array([-1.0, 0.0]); e_hat = np.array([0.0, 1.0])
dhat = np.cos(delta) * t_hat - np.sin(delta) * e_hat
L = args.fault_depth / np.sin(delta)
xy = P0[None, :] + np.linspace(0, L, 25)[:, None] * dhat[None, :]
zx = (xy[:, 0].min() - 0.22, xy[:, 0].max() + 0.30)
zy = (xy[:, 1].min() - 0.28, xy[:, 1].max() + 0.10)


def tris_of(mesh):
    dm = mesh.dm
    pS, pE = dm.getDepthStratum(0); cS, cE = dm.getHeightStratum(0)
    return np.asarray([[p - pS for p in dm.getTransitiveClosure(c)[0]
                        if pS <= p < pE] for c in range(cS, cE)])


def run_R(R):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5,
                              cellSize=1.0 / args.res, qdegree=3)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)
    T.data[:, 0] = np.asarray(uw.function.evaluate(Tsnap.sym[0], T.coords)).reshape(-1)
    fault = uw.meshing.Surface("f", mesh, np.column_stack([xy, np.zeros(25)]), symbol="F")
    fault.discretize(); _ = fault.distance
    # CORRECTED metric: effective-R front-following |grad T| density x fault band
    rho_T = uw.meshing.metric_density_from_gradient(
        mesh, T, refinement=R, coarsening="auto",
        metric_choice="front-following", name="loop")
    d = fault.distance.sym[0]
    fault_rho = 1.0 + args.fault_refine_amp * sympy.exp(-(d / refine_w) ** 2)
    # MAX (not product): the fault must refine INDEPENDENTLY of the local |grad T|.
    # The product lets the cold-lid low-gradient zone (rho_T~0.25 at the fault)
    # cancel the fault band; max() makes the fault density ~= amp regardless.
    rho = sympy.Max(rho_T, fault_rho) if args.combine == "max" else rho_T * fault_rho
    C0 = np.asarray(mesh.X.coords).copy()
    # CORRECTED mover kwargs: mmpde's own CG iteration (NOT relax/n_outer)
    uw.meshing.smooth_mesh_interior(
        mesh, metric=rho, method="mmpde", skip_threshold=None,
        slip_surfaces=True,
        method_kwargs=dict(step_frac=0.2, accel="cg", momentum=0.0),
        verbose=False)
    C1 = np.asarray(mesh.X.coords).copy()
    q = mesh.quality()
    disp = np.linalg.norm(C1 - C0, axis=1)
    dist = np.array([np.min(np.linalg.norm(xy - c, axis=1)) for c in C0])
    near = dist < 0.12
    print(f"R={R}: q_min={q['q_min']:.3f} slivers(<0.3)={q['n_q_lt_0p3']} "
          f"near-fault|disp|={disp[near].mean():.4f} far={disp[~near].mean():.4f}",
          flush=True)
    Tv = np.asarray(uw.function.evaluate(T.sym[0], C1)).reshape(-1)
    return C1, tris_of(mesh), Tv, q


Rs = [3.0, 5.0, 8.0]
results = [run_R(R) for R in Rs]

fig = plt.figure(figsize=(6.0 * len(Rs), 11.5))
gs = fig.add_gridspec(2, len(Rs), height_ratios=[1.0, 1.6])
for j, (R, (C, tris, Tv, q)) in enumerate(zip(Rs, results)):
    tri = Triangulation(C[:, 0], C[:, 1], tris)
    a0 = fig.add_subplot(gs[0, j]); a1 = fig.add_subplot(gs[1, j])
    a0.tripcolor(tri, Tv, shading="gouraud", cmap="RdBu_r")
    a0.triplot(tri, color="k", lw=0.18, alpha=0.45)
    a0.plot(xy[:, 0], xy[:, 1], "k-", lw=1.4)
    a0.set_title(f"R={R}  q_min={q['q_min']:.2f}  slivers={q['n_q_lt_0p3']}")
    a1.triplot(tri, color="0.2", lw=0.8); a1.plot(xy[:, 0], xy[:, 1], "r-", lw=2.4)
    a1.set_xlim(*zx); a1.set_ylim(*zy)
    a1.set_title(f"fault zoom (R={R})")
    for a in (a0, a1):
        a.set_aspect("equal"); a.axis("off")
fig.suptitle(f"CORRECTED MMPDE (own CG iter) + effective-R front-following metric "
             f"x fault band — T from {args.src_tag}/{label}, uniform res{args.res}, "
             f"amp={args.fault_refine_amp}", fontsize=14)
fig.tight_layout()
out = os.path.join(OUT, "mmpde_preview.png")
fig.savefig(out, dpi=160)
print("→", out)
