"""Scalar (isotropic) vs ANISOTROPIC TENSOR fault metric, MMPDE mover.

Per _winslow_mmpde docstring: the metric is an SPD dxd TENSOR; build it small
ACROSS the fault (along its normal n) to align thin cell rows onto the fault:
    M = rho_T * I  +  (Rf^2 - 1) * exp(-(d/W)^2) * n n^T
(rho_T = front-following |grad T| density; the fault term refines ONLY the
normal direction, localized to the fault, so cells go thin-across/long-along
and a node row lands ON the centre-line).

Compares isotropic scalar (what we had) vs tensor at two anisotropy ratios.
Shows fault-zoom mesh + perpendicular node-distance histogram for each.
Writes ~/+Simulations/StagnantLid/fault_aniso_metric/aniso.png
"""
from __future__ import annotations
import os, glob, re
import numpy as np, sympy, underworld3 as uw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

SRC = os.path.expanduser('~/+Simulations/StagnantLid/fault_iso_Ra1e6')
OUT = os.path.expanduser('~/+Simulations/StagnantLid/fault_aniso_metric')
os.makedirs(OUT, exist_ok=True)
RES, R, AMP = 32, 5.0, 30.0
W = 1.5 * 0.05

c = sorted(glob.glob(SRC + "/step*.mesh.00000.h5"),
           key=lambda c: int(re.search(r"step(\d+)\.mesh", c).group(1)))[-1]
lab = re.search(r"(step\d+)\.mesh", os.path.basename(c)).group(1)
ms = uw.discretisation.Mesh(c)
Ts = uw.discretisation.MeshVariable("Ts", ms, 1, degree=3, varsymbol="T")
Ts.read_timestep(lab, "T_v2p1", 0, outputPath=SRC)

dlt = np.deg2rad(30.0); P0 = np.array([0., 1.])
dh = np.cos(dlt)*np.array([-1., 0.])-np.sin(dlt)*np.array([0., 1.])
n_vec = np.array([-dh[1], dh[0]]); n_vec /= np.linalg.norm(n_vec)
L = 0.225/np.sin(dlt); xy = P0[None, :]+np.linspace(0, L, 25)[:, None]*dh[None, :]
zx = (xy[:, 0].min()-0.16, xy[:, 0].max()+0.20); zy = (xy[:, 1].min()-0.20, xy[:, 1].max()+0.07)


def tris_of(mesh):
    dm = mesh.dm
    pS, pE = dm.getDepthStratum(0); cS, cE = dm.getHeightStratum(0)
    return np.asarray([[p - pS for p in dm.getTransitiveClosure(c)[0]
                        if pS <= p < pE] for c in range(cS, cE)])


def perp_dist(C):
    rel = C - P0[None, :]
    return rel @ n_vec, (rel @ dh > 0) & (rel @ dh < L)


def run(kind, Rf):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1.0/RES, qdegree=3)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)
    T.data[:, 0] = np.asarray(uw.function.evaluate(Ts.sym[0], T.coords)).reshape(-1)
    fault = uw.meshing.Surface("f", mesh, np.column_stack([xy, np.zeros(25)]), symbol="F")
    fault.discretize(); _ = fault.distance
    rho_T = uw.meshing.metric_density_from_gradient(
        mesh, T, refinement=R, coarsening="auto", metric_choice="front-following", name="loop")
    d = fault.distance.sym[0]
    gauss = sympy.exp(-(d / W) ** 2)
    if kind == "scalar":                       # isotropic: what we had
        metric = sympy.Max(rho_T, 1.0 + AMP * gauss)
    else:                                      # anisotropic tensor (small across n)
        n = sympy.Matrix([float(n_vec[0]), float(n_vec[1])])
        nnT = n * n.T
        I2 = sympy.eye(2)
        metric = rho_T * I2 + (Rf ** 2 - 1.0) * gauss * nnT
    C0 = np.asarray(mesh.X.coords).copy()
    uw.meshing.smooth_mesh_interior(
        mesh, metric=metric, method="mmpde", skip_threshold=None, slip_surfaces=True,
        method_kwargs=dict(step_frac=0.2, accel="cg", momentum=0.0), verbose=False)
    C1 = np.asarray(mesh.X.coords).copy()
    q = mesh.quality()
    return C1, tris_of(mesh), kind, Rf, q


CONFIGS = [("scalar", None), ("tensor", 5.0), ("tensor", 8.0), ("tensor", 12.0)]
results = [run(k, Rf) for k, Rf in CONFIGS]

fig = plt.figure(figsize=(5.0*len(results), 9.5))
gs = fig.add_gridspec(2, len(results), height_ratios=[1.6, 1.0])
for j, (C, tris, kind, Rf, q) in enumerate(results):
    tri = Triangulation(C[:, 0], C[:, 1], tris)
    a0 = fig.add_subplot(gs[0, j]); a1 = fig.add_subplot(gs[1, j])
    a0.triplot(tri, color="0.2", lw=0.7); a0.plot(xy[:, 0], xy[:, 1], "r-", lw=2.2)
    a0.set_xlim(*zx); a0.set_ylim(*zy); a0.set_aspect("equal"); a0.axis("off")
    ttl = "scalar (isotropic)" if kind == "scalar" else f"tensor Rf={Rf:g}"
    a0.set_title(f"{ttl}\nq_min={q['q_min']:.2f} slivers={q['n_q_lt_0p3']}")
    perp, inside = perp_dist(C)
    pv = perp[inside]; pv = pv[np.abs(pv) < 0.16]
    a1.hist(pv, bins=41, range=(-0.16, 0.16), color="0.4")
    a1.axvline(0, color="r", lw=1.5)
    a1.set_xlabel("perp dist to fault"); a1.set_ylabel("node count")
    mind = np.abs(pv).min()
    a1.set_title(f"min|d|={mind:.4f}  |d|<0.02: {np.sum(np.abs(pv)<0.02)}")
fig.suptitle("Isotropic SCALAR vs ANISOTROPIC TENSOR fault metric (MMPDE). Tensor = "
             "rho_T·I + (Rf²-1)·exp(-(d/W)²)·nnᵀ\nWant: thin cells ACROSS the fault + "
             "node row ON the centre-line (histogram spike at d=0)", fontsize=12)
fig.tight_layout()
out = os.path.join(OUT, "aniso.png")
fig.savefig(out, dpi=150)
print("→", out)
for C, tris, kind, Rf, q in results:
    perp, inside = perp_dist(C)
    pv = perp[inside]; pv = pv[np.abs(pv) < 0.16]
    ttl = "scalar" if kind == "scalar" else f"tensor Rf={Rf:g}"
    print(f"{ttl:12s}: q_min={q['q_min']:.3f} slivers={q['n_q_lt_0p3']} "
          f"min|d|={np.abs(pv).min():.4f} nodes|d|<0.02={np.sum(np.abs(pv)<0.02)} "
          f"aspect_max={q['aspect_max']:.1f}", flush=True)
