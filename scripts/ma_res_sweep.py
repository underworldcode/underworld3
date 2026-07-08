"""Is the BFO convex-branch undershoot a single-shot under-resolution
limit? Sweep mesh resolution for AMP=8 and compare the converged FE
deep/near edge ratio to the (resolution-independent) exact ~10.5.

If the FE ratio climbs strongly toward ~10 as RES increases, a
single MA solve is resolution-limited in the ~WIDTH-wide metric
band ⇒ the path forward is an outer map iteration (mesh adapts
toward the feature, re-solve with better local resolution). If it
stays ~1.1 at all RES, the formulation is wrong.
"""
from __future__ import annotations
import numpy as np
import sympy
import underworld3 as uw
from underworld3.meshing.smoothing import (
    _winslow_elliptic, _auto_pinned_labels)

R_I, R_O = 0.5, 1.0
WIDTH = 0.12
AMP = 8.0


def mesh_triangles(m):
    dm = m.dm
    cS, cE = dm.getHeightStratum(0)
    pS, pE = dm.getDepthStratum(0)
    tris = []
    for c in range(cS, cE):
        cl = dm.getTransitiveClosure(c)[0]
        vs = [p - pS for p in cl if pS <= p < pE]
        if len(vs) == 3:
            tris.append(vs)
    return np.asarray(tris, dtype=np.int64)


def edge_ratio(coords, TRI):
    a, b, c = coords[TRI[:, 0]], coords[TRI[:, 1]], coords[TRI[:, 2]]
    el = np.concatenate([
        np.linalg.norm(a - b, axis=1),
        np.linalg.norm(b - c, axis=1),
        np.linalg.norm(c - a, axis=1)])
    cent = (a + b + c) / 3.0
    cr = np.repeat(np.sqrt((cent ** 2).sum(axis=1)), 3)
    near = cr > (R_O - WIDTH)
    deep = cr < (R_O - 0.30)
    return el[near].mean(), el[deep].mean()


print(f"AMP={AMP}  exact deep/near ≈ 10.5 (resolution-independent)")
for RES in (16, 32, 48):
    mesh = uw.meshing.Annulus(radiusOuter=R_O, radiusInner=R_I,
                              cellSize=1.0 / RES, qdegree=3)
    TRI = mesh_triangles(mesh)
    c0 = np.asarray(mesh.X.coords).copy()
    r0 = uw.discretisation.MeshVariable(
        f"r0rs{RES}", mesh, vtype=uw.VarType.SCALAR, degree=1,
        continuous=True)
    X0 = np.asarray(mesh.X.coords)
    r0.data[:, 0] = np.sqrt((X0 ** 2).sum(axis=1))
    f = 1.0 + AMP * sympy.exp(-(((r0.sym[0]) - R_O) / WIDTH) ** 2)
    pinned = _auto_pinned_labels(mesh)
    _winslow_elliptic(mesh, f, pinned, False, n_picard=120,
                      relax=1.0, step_frac=None, picard_relax=0.25)
    c1 = np.asarray(mesh.X.coords).copy()
    en0, ed0 = edge_ratio(c0, TRI)
    en1, ed1 = edge_ratio(c1, TRI)
    print(f"  RES={RES:>2}  cells/band≈{WIDTH*RES:.1f}  "
          f"deep/near {ed0/en0:.3f}->{ed1/en1:.3f}  "
          f"near {en0:.4f}->{en1:.4f}")
