"""Controlled snapshot experiment: FREEZE the T field from an
a16e checkpoint (default the step-20 overshoot, where equidist
makes its clustered poor cells), put it on a fresh *pristine
undeformed* res-16 Annulus (exactly what adapt_pristine feeds the
mover), then run ONE mover adaptation per parameter combo and
measure cell quality. No time loop, no EMA (single event), no
Stokes — isolates the mesh *construction* effect alone.

  python scripts/_snap_sweep.py [step=20] [src_tag=a16e]

Combos (EMA excluded; relax/n_outer/beta = a16e mover settings):
  refine-only          R=1                         (context, no de-res)
  cc2-ref              coarsen_cap=2 aniso_cap=4    (known solver-clean bar)
  equidist  iso-bump   R=2 aniso_to_base=False      (= a16e, the problem)
  equidist  base-bump  R=2 aniso_to_base=True       (the proposed fix)
  base-bump R=1.5/2.5  clamp-width lever on the fix
  iso-bump  R=1.5      smaller clamp alone (no base-keying)
"""
import sys
import numpy as np
import underworld3 as uw
from underworld3.meshing import (
    smooth_mesh_interior, metric_density_from_gradient)
from underworld3.meshing.smoothing import _tri_cells

D = "/tmp/metric_mesh/sat"
STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SRC = sys.argv[2] if len(sys.argv) > 2 else "a16e"
MOV = dict(relax=0.05, n_outer=25, beta=200.0,
           geom_mean_smoothing=1.0)            # EMA off (1 event)

COMBOS = [
    ("refine-only      ", dict(resolution_ratio=1.0)),
    ("cc2-ref          ", dict(coarsen_cap=2.0, aniso_cap=4.0)),
    ("equidist iso  R2  ", dict(resolution_ratio=2.0,
                                aniso_to_base=False)),
    ("equidist base R2  ", dict(resolution_ratio=2.0,
                                aniso_to_base=True)),
    ("equidist base R1.5", dict(resolution_ratio=1.5,
                                aniso_to_base=True)),
    ("equidist base R2.5", dict(resolution_ratio=2.5,
                                aniso_to_base=True)),
    ("equidist iso  R1.5", dict(resolution_ratio=1.5,
                                aniso_to_base=False)),
]


def quality(m):
    tri = _tri_cells(m.dm)
    X = np.asarray(m.X.coords)[:, :2]
    v0, v1, v2 = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
    a = np.linalg.norm(v1 - v2, axis=1)
    b = np.linalg.norm(v2 - v0, axis=1)
    c = np.linalg.norm(v0 - v1, axis=1)
    A = np.maximum(0.5 * np.abs(np.cross(v1 - v0, v2 - v0)),
                   1e-300)
    q = 4.0 * np.sqrt(3.0) * A / (a * a + b * b + c * c)

    def ang(o, p, r):
        return np.degrees(np.arccos(np.clip(
            (p * p + r * r - o * o) / (2 * p * r), -1, 1)))
    am = np.maximum.reduce([ang(a, b, c), ang(b, c, a),
                            ang(c, a, b)])
    Lmax = np.maximum.reduce([a, b, c])
    aspect = Lmax * Lmax / (2.0 * A)
    rel = A / A.mean()
    et = {}
    for ti, (i, j, k) in enumerate(tri):
        for u, w in ((i, j), (j, k), (k, i)):
            et.setdefault((min(u, w), max(u, w)), []).append(ti)
    jr = np.array([max(A[t]) / min(A[t])
                   for t in et.values() if len(t) == 2] or [1.0])
    return dict(
        mn=A.min() / A.mean(), qmin=q.min(),
        n03=int((q < 0.3).sum()), n02=int((q < 0.2).sum()),
        angx=am.max(), n165=int((am > 165).sum()),
        aspx=aspect.max(), jmx=jr.max(),
        bt=int(((rel > 2) & (aspect > 4)).sum()))


print(f"\nFROZEN snapshot: {SRC} step {STEP}  "
      f"(one pristine adaptation per combo)\n")
hdr = ("combo               | minA/mn qmin  n<.3 n<.2 | angMx "
       ">165 aspMx jumpMx BIG&THIN")
print(hdr)
print("-" * len(hdr))
for label, kw in COMBOS:
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5,
                              cellSize=1.0 / 16, qdegree=3)
    T = uw.discretisation.MeshVariable(
        "T", mesh, vtype=uw.VarType.SCALAR, degree=3,
        continuous=True)
    T.read_timestep(f"sat_{SRC}", "T", STEP, outputPath=D)
    rho = metric_density_from_gradient(
        mesh, T, amp=16.0, name=f"snap_{label.strip()}",
        lo_percentile=50.0, hi_percentile=97.0)
    smooth_mesh_interior(mesh, metric=rho, method="anisotropic",
                         method_kwargs={**MOV, **kw})
    r = quality(mesh)
    print(f"{label} | {r['mn']:6.3f} {r['qmin']:.3f} "
          f"{r['n03']:4d} {r['n02']:4d} | {r['angx']:5.1f} "
          f"{r['n165']:4d} {r['aspx']:5.1f} {r['jmx']:5.1f} "
          f"   {r['bt']:4d}")
print("\nTarget: equidist variant matching cc2-ref's qmin / n<.3 "
      "/ BIG&THIN — parameter-free (R only).")
