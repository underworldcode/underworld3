"""Visual of the frozen-snapshot sweep (step 20 a16e overshoot
field): one panel per parameter combo, triangles shaded by shape
quality q (RdYlGn, red=poor). Same combos as _snap_sweep.py.
Writes one PNG for Preview so the poor-cell clusters are visible,
not just tabulated."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import underworld3 as uw
from underworld3.meshing import (
    smooth_mesh_interior, metric_density_from_gradient)
from underworld3.meshing.smoothing import _tri_cells

D = "/tmp/metric_mesh/sat"
STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SRC = "a16e"
MOV = dict(relax=0.05, n_outer=25, beta=200.0,
           geom_mean_smoothing=1.0)
COMBOS = [
    ("refine-only  R1", dict(resolution_ratio=1.0)),
    ("cc2-ref (clean bar)", dict(coarsen_cap=2.0, aniso_cap=4.0)),
    ("equidist iso  R2  (=a16e)", dict(resolution_ratio=2.0,
                                       aniso_to_base=False)),
    ("equidist base R2", dict(resolution_ratio=2.0,
                              aniso_to_base=True)),
    ("equidist iso  R1.5", dict(resolution_ratio=1.5,
                                aniso_to_base=False)),
    ("equidist base R1.5", dict(resolution_ratio=1.5,
                                aniso_to_base=True)),
    ("equidist base R2.5", dict(resolution_ratio=2.5,
                                aniso_to_base=True)),
]


def qual(m):
    tri = _tri_cells(m.dm)
    X = np.asarray(m.X.coords)[:, :2]
    v0, v1, v2 = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
    a = np.linalg.norm(v1 - v2, axis=1)
    b = np.linalg.norm(v2 - v0, axis=1)
    c = np.linalg.norm(v0 - v1, axis=1)
    A = np.maximum(0.5 * np.abs(np.cross(v1 - v0, v2 - v0)),
                   1e-300)
    q = 4.0 * np.sqrt(3.0) * A / (a * a + b * b + c * c)
    Lmax = np.maximum.reduce([a, b, c])
    aspect = Lmax * Lmax / (2.0 * A)
    rel = A / A.mean()
    bt = int(((rel > 2) & (aspect > 4)).sum())
    return q, X, tri, bt


fig, ax = plt.subplots(2, 4, figsize=(22, 11))
ax = ax.ravel()
for n, (label, kw) in enumerate(COMBOS):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5,
                              cellSize=1.0 / 16, qdegree=3)
    T = uw.discretisation.MeshVariable(
        "T", mesh, vtype=uw.VarType.SCALAR, degree=3,
        continuous=True)
    T.read_timestep(f"sat_{SRC}", "T", STEP, outputPath=D)
    rho = metric_density_from_gradient(
        mesh, T, amp=16.0, name=f"snv_{n}",
        lo_percentile=50.0, hi_percentile=97.0)
    smooth_mesh_interior(mesh, metric=rho, method="anisotropic",
                         method_kwargs={**MOV, **kw})
    q, X, tri, bt = qual(mesh)
    a = ax[n]
    Tr = mtri.Triangulation(X[:, 0], X[:, 1], tri)
    tpc = a.tripcolor(Tr, facecolors=q, cmap="RdYlGn",
                      vmin=0.15, vmax=0.7, edgecolors="k",
                      linewidth=0.15)
    a.set_aspect("equal")
    a.set_axis_off()
    a.set_title(f"{label}\nqmin={q.min():.3f}  "
                f"q<0.3:{int((q < 0.3).sum())}  "
                f"BIG&THIN:{bt}", fontsize=12)
    print(f"  {label}: qmin={q.min():.3f} "
          f"n<.3={int((q < 0.3).sum())} bt={bt}", flush=True)
ax[7].set_axis_off()
fig.colorbar(tpc, ax=ax[7], shrink=0.7,
             label="cell shape quality q (red = poor)")
fig.suptitle(f"Frozen a16e step-{STEP} overshoot field — one "
             f"pristine adaptation per combo. R is the lever: "
             f"R=1.5 clean, R=2 marginal, R=2.5 sliver mess.",
             fontsize=14)
out = f"{D}/snap_sweep_step{STEP}.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved {out}")
