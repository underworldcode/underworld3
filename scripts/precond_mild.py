"""Mild-spring preconditioner for MA. Idea (user): spring at a weak
metric (or few sweeps) sets the node-motion DIRECTION while keeping
the mesh VALID (it degenerates only when pushed hard); then MA does
the strong part stably & — from a pre-aligned start — hopefully
cheaper.

For target AMP ∈ {8, 20}, identical fresh Annulus setups:
  A   MA only  (full AMP)                         [baseline]
  Bm  spring only, MILD (AMP=2)                    [stays valid?]
  E   spring(AMP=2) -> MA(full, n_outer=1)
  F   spring(AMP=2) -> MA(full, n_outer=3 resid.)
  G   spring(full AMP, 30 sweeps) -> MA(full,n1)   ["few iters"]
Report wall time, honest deep/near, mesh quality minA/meanA
(0 ⇒ degenerate sliver; healthy ~0.02–0.3).
"""
from __future__ import annotations
import time
import numpy as np
import sympy
import underworld3 as uw
from underworld3.meshing.smoothing import (
    _winslow_spring, _winslow_elliptic, _edge_pairs,
    _auto_pinned_labels, _tri_cells, _signed_areas)

R_O, R_I, WIDTH, RES = 1.0, 0.5, 0.12, 16
SPRING_MILD_AMP = 2.0


def honest_ratio(coords, edges):
    v0, v1 = edges[:, 0], edges[:, 1]
    Le = np.linalg.norm(coords[v1] - coords[v0], axis=1)
    nv = coords.shape[0]
    s = np.zeros(nv); c = np.zeros(nv)
    for a in (v0, v1):
        np.add.at(s, a, Le); np.add.at(c, a, 1.0)
    nl = s / np.maximum(c, 1.0)
    r = np.sqrt((coords ** 2).sum(axis=1))
    deep = (r >= R_I) & (r < R_I + 0.20)
    near = (r > R_O - 0.05)
    return float(nl[deep].mean() / nl[near].mean())


def quality(mesh):
    tris = _tri_cells(mesh.dm)
    if tris is None:
        return float("nan")
    a = np.abs(_signed_areas(np.asarray(mesh.X.coords), tris))
    return float(a.min() / a.mean())


def case(amp, tag):
    m = uw.meshing.Annulus(radiusOuter=R_O, radiusInner=R_I,
                           cellSize=1.0 / RES, qdegree=3)
    r0 = uw.discretisation.MeshVariable(
        f"r0_{tag}", m, vtype=uw.VarType.SCALAR,
        degree=1, continuous=True)
    X0 = np.asarray(m.X.coords)
    r0.data[:, 0] = np.sqrt((X0 ** 2).sum(axis=1))
    full = 1.0 + amp * sympy.exp(
        -(((r0.sym[0]) - R_O) / WIDTH) ** 2)
    mild = 1.0 + SPRING_MILD_AMP * sympy.exp(
        -(((r0.sym[0]) - R_O) / WIDTH) ** 2)
    return m, full, mild


def show(amp, name, m, e, dt):
    print(f"{amp:4.0f} {name:>26} {dt:8.2f} "
          f"{honest_ratio(np.asarray(m.X.coords), e):10.3f} "
          f"{quality(m):11.4f}", flush=True)


print(f"{'AMP':>4} {'variant':>26} {'time/s':>8} "
      f"{'deep/near':>10} {'minA/meanA':>11}")
print("-" * 64)
for amp in (8.0, 20.0):
    m, full, mild = case(amp, f"A{int(amp)}")
    e = _edge_pairs(m.dm); p = _auto_pinned_labels(m)
    t = time.perf_counter(); _winslow_elliptic(m, full, p, False)
    show(amp, "A MA only", m, e, time.perf_counter() - t)

    m, full, mild = case(amp, f"Bm{int(amp)}")
    e = _edge_pairs(m.dm); p = _auto_pinned_labels(m)
    t = time.perf_counter(); _winslow_spring(m, mild, p, False)
    show(amp, "Bm spring MILD only", m, e, time.perf_counter() - t)

    m, full, mild = case(amp, f"E{int(amp)}")
    e = _edge_pairs(m.dm); p = _auto_pinned_labels(m)
    t = time.perf_counter()
    _winslow_spring(m, mild, p, False)
    _winslow_elliptic(m, full, p, False, n_outer=1)
    show(amp, "E springMILD->MA n1", m, e, time.perf_counter() - t)

    m, full, mild = case(amp, f"F{int(amp)}")
    e = _edge_pairs(m.dm); p = _auto_pinned_labels(m)
    t = time.perf_counter()
    _winslow_spring(m, mild, p, False)
    _winslow_elliptic(m, full, p, False, n_outer=3)
    show(amp, "F springMILD->MA n3", m, e, time.perf_counter() - t)

    m, full, mild = case(amp, f"G{int(amp)}")
    e = _edge_pairs(m.dm); p = _auto_pinned_labels(m)
    t = time.perf_counter()
    _winslow_spring(m, full, p, False, n_sweeps=30)
    _winslow_elliptic(m, full, p, False, n_outer=1)
    show(amp, "G springFULL30->MA n1", m, e,
         time.perf_counter() - t)
