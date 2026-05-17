"""Tune the Monge-Ampere equidistribution move on an undeformed
Annulus: sweep (n_picard, relax, step_frac) and report the
deep/near edge-length ratio AND the minimum signed triangle area
(tangling check). Calls _winslow_elliptic directly so no rebuild is
needed between parameter trials.
"""
from __future__ import annotations
import numpy as np
import sympy
import underworld3 as uw
from underworld3.meshing.smoothing import (
    _winslow_elliptic, _auto_pinned_labels)

R_O, R_I = 1.0, 0.5
RES = 16
WIDTH = 0.12


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


def signed_areas(coords, TRI):
    a, b, c = coords[TRI[:, 0]], coords[TRI[:, 1]], coords[TRI[:, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                  - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


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


TRIALS = [
    dict(n_outer=1,  n_picard=40, relax=1.0, step_frac=None,
         picard_relax=0.4),
    dict(n_outer=4,  n_picard=40, relax=1.0, step_frac=None,
         picard_relax=0.4),
    dict(n_outer=8,  n_picard=40, relax=1.0, step_frac=None,
         picard_relax=0.4),
    dict(n_outer=15, n_picard=40, relax=1.0, step_frac=None,
         picard_relax=0.4),
]

for amp in (0.0, 8.0, 20.0):
    print(f"\n================  AMP = {amp:g}  ================")
    for t in TRIALS:
        mesh = uw.meshing.Annulus(
            radiusOuter=R_O, radiusInner=R_I,
            cellSize=1.0 / RES, qdegree=3)
        TRI = mesh_triangles(mesh)
        c0 = np.asarray(mesh.X.coords).copy()
        a0 = signed_areas(c0, TRI)
        orient = np.sign(np.median(a0))  # consistent CCW/CW sign

        r0 = uw.discretisation.MeshVariable(
            f"r0_{int(amp)}_{t['n_picard']}_{int(t['relax']*100)}",
            mesh, vtype=uw.VarType.SCALAR, degree=1, continuous=True)
        X0 = np.asarray(mesh.X.coords)
        r0.data[:, 0] = np.sqrt(X0[:, 0] ** 2 + X0[:, 1] ** 2)
        f = 1.0 + amp * sympy.exp(
            -(((r0.sym[0]) - R_O) / WIDTH) ** 2)

        pinned = _auto_pinned_labels(mesh)
        _winslow_elliptic(mesh, f, pinned, False,
                          n_outer=t["n_outer"],
                          n_picard=t["n_picard"],
                          relax=t["relax"],
                          step_frac=t["step_frac"],
                          picard_relax=t["picard_relax"])

        c1 = np.asarray(mesh.X.coords).copy()
        a1 = signed_areas(c1, TRI) * orient   # positive = good
        en0, ed0 = edge_ratio(c0, TRI)
        en1, ed1 = edge_ratio(c1, TRI)
        n_inv = int((a1 <= 0.0).sum())
        print(
            f"  outer={t['n_outer']:>2}  "
            f"ratio {ed0/en0:.3f}->{ed1/en1:.3f}  "
            f"near {en0:.4f}->{en1:.4f}  deep {ed0:.4f}->{ed1:.4f}  "
            f"min_area {a1.min():.2e}  inverted={n_inv}")
