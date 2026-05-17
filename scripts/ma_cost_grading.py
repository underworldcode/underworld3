"""MA-only cost + HONEST grading + mesh-validity check, cold & warm,
across AMP. Confirms the direct-solver speedup preserves the
grading/quality and the AMP=0 exact-no-op invariant.

Recorded BFO baselines (project memory, GAMG path, honest metric):
  AMP 0 → d/n ≈ 1.02 (no-op)   AMP 8 → 1.71   AMP 20 → 1.54
Grading must match these to within noise; minA>0 (no tangle).
Spring is NOT re-run here (reference its recorded ~0.3 s / 1.65–1.79).
"""
from __future__ import annotations
import time
import numpy as np
import sympy
import underworld3 as uw
from underworld3.meshing.smoothing import (
    _winslow_elliptic, _edge_pairs, _auto_pinned_labels,
    _tri_cells, _signed_areas)

R_O, R_I, WIDTH, RES = 1.0, 0.5, 0.12, 16
AMPS = [0.0, 2.0, 8.0, 20.0]


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


def min_area(m):
    tris = _tri_cells(m.dm)
    if tris is None:
        return float("nan")
    a = np.abs(_signed_areas(np.asarray(m.X.coords), tris))
    return float(a.min() / a.mean())


print(f"{'AMP':>5} {'cold/s':>7} {'warm/s':>7} {'d/n':>6} "
      f"{'minA/meanA':>10}   baseline d/n")
print("-" * 56)
BASE = {0.0: 1.02, 2.0: 1.43, 8.0: 1.71, 20.0: 1.54}
for k, amp in enumerate(AMPS):
    m, f = case(amp, f"c{k}")
    e = _edge_pairs(m.dm)
    pin = _auto_pinned_labels(m)
    t = time.perf_counter(); _winslow_elliptic(m, f, pin, False)
    cold = time.perf_counter() - t
    dn = honest_ratio(np.asarray(m.X.coords), e)
    ma = min_area(m)
    t = time.perf_counter(); _winslow_elliptic(m, f, pin, False)
    warm = time.perf_counter() - t
    print(f"{amp:5.1f} {cold:7.2f} {warm:7.2f} {dn:6.3f} "
          f"{ma:10.4f}   (was ~{BASE[amp]:.2f})", flush=True)
