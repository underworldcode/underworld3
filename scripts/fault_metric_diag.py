"""Diagnose WHERE nodes end up relative to the fault centre-line. The user
observes the mesh refining to the EDGES of the fault band, not bringing nodes
to the centre. For several band widths / profiles, adapt with MMPDE then plot:
  (top)  fault-zoom mesh, fault line in red
  (bottom) histogram of node PERPENDICULAR distance to the fault, with the
           metric profile rho(d) overlaid — a centre-seeking metric should give
           a histogram PEAKED at d=0; edge-piling shows a dip at 0 with humps.
Writes ~/+Simulations/StagnantLid/fault_metric_diag/diag.png
"""
from __future__ import annotations
import os, glob, re
import numpy as np, sympy, underworld3 as uw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

SRC = os.path.expanduser('~/+Simulations/StagnantLid/fault_iso_Ra1e6')
OUT = os.path.expanduser('~/+Simulations/StagnantLid/fault_metric_diag')
os.makedirs(OUT, exist_ok=True)
RES, R, AMP = 32, 5.0, 30.0

c = sorted(glob.glob(SRC + "/step*.mesh.00000.h5"),
           key=lambda c: int(re.search(r"step(\d+)\.mesh", c).group(1)))[-1]
lab = re.search(r"(step\d+)\.mesh", os.path.basename(c)).group(1)
ms = uw.discretisation.Mesh(c)
Ts = uw.discretisation.MeshVariable("Ts", ms, 1, degree=3, varsymbol="T")
Ts.read_timestep(lab, "T_v2p1", 0, outputPath=SRC)

d = np.deg2rad(30.0); P0 = np.array([0., 1.]); dh = np.cos(d)*np.array([-1., 0.])-np.sin(d)*np.array([0., 1.])
n_vec = np.array([-dh[1], dh[0]]); n_vec /= np.linalg.norm(n_vec)
L = 0.225/np.sin(d); xy = P0[None, :]+np.linspace(0, L, 25)[:, None]*dh[None, :]
zx = (xy[:, 0].min()-0.18, xy[:, 0].max()+0.22); zy = (xy[:, 1].min()-0.22, xy[:, 1].max()+0.08)


def tris_of(mesh):
    dm = mesh.dm
    pS, pE = dm.getDepthStratum(0); cS, cE = dm.getHeightStratum(0)
    return np.asarray([[p - pS for p in dm.getTransitiveClosure(c)[0]
                        if pS <= p < pE] for c in range(cS, cE)])


def perp_dist(C):
    # signed perpendicular distance to the infinite fault line, masked to the
    # along-fault extent so we only histogram nodes beside the actual segment.
    rel = C - P0[None, :]
    perp = rel @ n_vec
    sal = rel @ dh
    inside = (sal > 0.0) & (sal < L)
    return perp, inside


def run(profile, w):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1.0/RES, qdegree=3)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)
    T.data[:, 0] = np.asarray(uw.function.evaluate(Ts.sym[0], T.coords)).reshape(-1)
    fault = uw.meshing.Surface("f", mesh, np.column_stack([xy, np.zeros(25)]), symbol="F")
    fault.discretize(); _ = fault.distance
    rho_T = uw.meshing.metric_density_from_gradient(
        mesh, T, refinement=R, coarsening="auto", metric_choice="front-following", name="loop")
    dd = fault.distance.sym[0]
    if profile == "gauss":
        band = 1.0 + AMP*sympy.exp(-(dd/w)**2)
    elif profile == "lorentz":          # sharper peak, heavier tail
        band = 1.0 + AMP/(1.0 + (dd/w)**2)
    elif profile == "quartic":          # flatter centre, steeper walls
        band = 1.0 + AMP*sympy.exp(-(dd/w)**4)
    rho = sympy.Max(rho_T, band)
    uw.meshing.smooth_mesh_interior(
        mesh, metric=rho, method="mmpde", skip_threshold=None, slip_surfaces=True,
        method_kwargs=dict(step_frac=0.2, accel="cg", momentum=0.0), verbose=False)
    C = np.asarray(mesh.X.coords).copy()
    return C, tris_of(mesh), profile, w


CONFIGS = [("gauss", 0.075), ("gauss", 0.035), ("gauss", 0.020), ("lorentz", 0.025)]
results = [run(p, w) for p, w in CONFIGS]

fig = plt.figure(figsize=(5.0*len(results), 9.5))
gs = fig.add_gridspec(2, len(results), height_ratios=[1.5, 1.0])
for j, (C, tris, prof, w) in enumerate(results):
    tri = Triangulation(C[:, 0], C[:, 1], tris)
    a0 = fig.add_subplot(gs[0, j]); a1 = fig.add_subplot(gs[1, j])
    a0.triplot(tri, color="0.2", lw=0.7); a0.plot(xy[:, 0], xy[:, 1], "r-", lw=2.2)
    a0.set_xlim(*zx); a0.set_ylim(*zy); a0.set_aspect("equal"); a0.axis("off")
    a0.set_title(f"{prof} w={w}")
    perp, inside = perp_dist(C)
    pv = perp[inside]
    pv = pv[np.abs(pv) < 0.18]
    a1.hist(pv, bins=41, range=(-0.18, 0.18), color="0.4")
    a1.axvline(0, color="r", lw=1.5)
    # metric profile (scaled) overlaid
    dgrid = np.linspace(-0.18, 0.18, 200)
    if prof == "gauss":
        prof_v = AMP*np.exp(-(dgrid/w)**2)
    elif prof == "lorentz":
        prof_v = AMP/(1.0+(dgrid/w)**2)
    else:
        prof_v = AMP*np.exp(-(dgrid/w)**4)
    a1b = a1.twinx()
    a1b.plot(dgrid, prof_v, "b-", lw=1.4, alpha=0.7)
    a1b.set_ylabel("rho_fault(d)", color="b")
    a1.set_xlabel("perp dist to fault")
    a1.set_ylabel("node count")
fig.suptitle("Node perpendicular-distance distribution after MMPDE (max combine, R=5, amp=30)\n"
             "centre-seeking metric -> histogram PEAKS at d=0; edge-piling -> dip at 0", fontsize=13)
fig.tight_layout()
out = os.path.join(OUT, "diag.png")
fig.savefig(out, dpi=150)
print("→", out)
for C, tris, prof, w in results:
    perp, inside = perp_dist(C)
    pv = perp[inside]; pv = pv[np.abs(pv) < 0.18]
    core = np.sum(np.abs(pv) < w/2); flank = np.sum((np.abs(pv) >= w/2) & (np.abs(pv) < 1.5*w))
    print(f"{prof:8s} w={w:.3f}: nodes |d|<w/2 (core)={core}  flank[w/2,1.5w]={flank}  "
          f"min|d|={np.abs(pv).min():.4f}", flush=True)
