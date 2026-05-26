"""MA efficiency: is the cheap elastic spring a good PRECONDITIONER
for the (stable but expensive) Monge–Ampère solve?

For AMP ∈ {8, 20}, on identical fresh Annulus setups, compare:
  A  MA only            (_winslow_elliptic from uniform; baseline)
  B  spring only        (reference — cheap but weaker)
  C  spring → MA n_out=1 (MA polishes; treats source as from-uniform)
  D  spring → MA n_out=3 (MA polishes the patch-volume RESIDUAL)

Report: wall time, honest deep/near grading, and mesh quality
(min triangle area / mean — lower = nastier cells).
Per-line flush so it is killable early.
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
    dm = mesh.dm
    tris = _tri_cells(dm)
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
    f = 1.0 + amp * sympy.exp(-(((r0.sym[0]) - R_O) / WIDTH) ** 2)
    return m, f


print(f"{'AMP':>4} {'variant':>22} {'time/s':>8} "
      f"{'deep/near':>10} {'minA/meanA':>11}")
print("-" * 60)
for amp in (8.0, 20.0):
    # A: MA only
    m, f = case(amp, f"A{int(amp)}")
    e = _edge_pairs(m.dm); p = _auto_pinned_labels(m)
    t = time.perf_counter()
    _winslow_elliptic(m, f, p, False)
    dt = time.perf_counter() - t
    print(f"{amp:4.0f} {'A MA only':>22} {dt:8.2f} "
          f"{honest_ratio(np.asarray(m.X.coords), e):10.3f} "
          f"{quality(m):11.4f}", flush=True)

    # B: spring only
    m, f = case(amp, f"B{int(amp)}")
    e = _edge_pairs(m.dm); p = _auto_pinned_labels(m)
    t = time.perf_counter()
    _winslow_spring(m, f, p, False)
    dt = time.perf_counter() - t
    print(f"{amp:4.0f} {'B spring only':>22} {dt:8.2f} "
          f"{honest_ratio(np.asarray(m.X.coords), e):10.3f} "
          f"{quality(m):11.4f}", flush=True)

    # C: spring -> MA (n_outer=1, source as-from-uniform)
    m, f = case(amp, f"C{int(amp)}")
    e = _edge_pairs(m.dm); p = _auto_pinned_labels(m)
    t = time.perf_counter()
    _winslow_spring(m, f, p, False)
    _winslow_elliptic(m, f, p, False, n_outer=1)
    dt = time.perf_counter() - t
    print(f"{amp:4.0f} {'C spring->MA n1':>22} {dt:8.2f} "
          f"{honest_ratio(np.asarray(m.X.coords), e):10.3f} "
          f"{quality(m):11.4f}", flush=True)

    # D: spring -> MA (n_outer=3, MA drives the patch-vol residual)
    m, f = case(amp, f"D{int(amp)}")
    e = _edge_pairs(m.dm); p = _auto_pinned_labels(m)
    t = time.perf_counter()
    _winslow_spring(m, f, p, False)
    _winslow_elliptic(m, f, p, False, n_outer=3)
    dt = time.perf_counter() - t
    print(f"{amp:4.0f} {'D spring->MA n3':>22} {dt:8.2f} "
          f"{honest_ratio(np.asarray(m.X.coords), e):10.3f} "
          f"{quality(m):11.4f}", flush=True)
