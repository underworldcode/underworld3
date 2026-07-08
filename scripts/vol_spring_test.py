"""Volumetric spring: equal edge springs (shape) + per-cell area
constraint (size). Compare to the known MA-only baseline
(AMP8: d/n 1.71, minA/meanA 0.026, ~11s ; AMP20: 1.54, 0.281, ~18s).

Sweep the size/shape weight ratio. Want: strong deep/near AND
healthy minA/meanA (no slivers — the whole point of equal edge
springs) at low cost.
"""
from __future__ import annotations
import time
import numpy as np
import sympy
import underworld3 as uw
from underworld3.meshing.smoothing import (
    _winslow_spring, _edge_pairs, _auto_pinned_labels,
    _tri_cells, _signed_areas)

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
    tris = _tri_cells(mesh.dm)
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


print("MA-only baseline: AMP8 d/n=1.71 q=0.026 ~11s | "
      "AMP20 d/n=1.54 q=0.281 ~18s")
print(f"{'AMP':>4} {'shape_w':>7} {'size_w':>6} {'time/s':>7} "
      f"{'deep/near':>10} {'minA/meanA':>11}")
print("-" * 56)
for amp in (2.0, 8.0, 20.0):
    for sw, zw in ((1.0, 4.0), (1.0, 8.0), (1.0, 20.0),
                   (0.3, 8.0)):
        m, f = case(amp, f"v{int(amp)}_{int(sw*10)}_{int(zw)}")
        e = _edge_pairs(m.dm)
        p = _auto_pinned_labels(m)
        t = time.perf_counter()
        _winslow_spring(m, f, p, False, shape_w=sw, size_w=zw)
        dt = time.perf_counter() - t
        print(f"{amp:4.0f} {sw:7.1f} {zw:6.1f} {dt:7.2f} "
              f"{honest_ratio(np.asarray(m.X.coords), e):10.3f} "
              f"{quality(m):11.4f}", flush=True)
