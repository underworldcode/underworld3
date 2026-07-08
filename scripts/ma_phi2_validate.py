"""Validate φ=P2 vs the shipped φ=P3 on the DIRECT path across the
full AMP sweep: grading must match the recorded baseline
(1.02/1.43/1.71/1.54), AMP=0 exact no-op, no tangle (minA>0), and
P2 should be ~2× cheaper. If it holds, phi_degree default 3→2 is a
clean solver-independent win on top of the shipped ~10×.
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
BASE = {0.0: 1.02, 2.0: 1.43, 8.0: 1.71, 20.0: 1.54}


def honest_ratio(coords, edges):
    v0, v1 = edges[:, 0], edges[:, 1]
    Le = np.linalg.norm(coords[v1] - coords[v0], axis=1)
    nv = coords.shape[0]
    s = np.zeros(nv); c = np.zeros(nv)
    for a in (v0, v1):
        np.add.at(s, a, Le); np.add.at(c, a, 1.0)
    nl = s / np.maximum(c, 1.0)
    r = np.sqrt((coords ** 2).sum(axis=1))
    return float(nl[(r >= R_I) & (r < R_I + 0.20)].mean()
                 / nl[r > R_O - 0.05].mean())


def min_area(m):
    tris = _tri_cells(m.dm)
    a = np.abs(_signed_areas(np.asarray(m.X.coords), tris))
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


print(f"{'AMP':>5} | {'P3 d/n':>7} {'P2 d/n':>7} {'base':>5} | "
      f"{'P3 minA':>8} {'P2 minA':>8} | {'P3 t':>6} {'P2 t':>6} (cold)")
print("-" * 72)
for k, amp in enumerate(AMPS):
    m3, f = case(amp, f"p3_{k}")
    e = _edge_pairs(m3.dm); pin = _auto_pinned_labels(m3)
    t = time.perf_counter()
    _winslow_elliptic(m3, f, pin, False, phi_degree=3)
    t3 = time.perf_counter() - t
    dn3 = honest_ratio(np.asarray(m3.X.coords), e)
    ma3 = min_area(m3)

    m2, f = case(amp, f"p2_{k}")
    e = _edge_pairs(m2.dm); pin = _auto_pinned_labels(m2)
    t = time.perf_counter()
    _winslow_elliptic(m2, f, pin, False, phi_degree=2)
    t2 = time.perf_counter() - t
    dn2 = honest_ratio(np.asarray(m2.X.coords), e)
    ma2 = min_area(m2)

    flag = "" if abs(dn3 - dn2) < 8e-3 else "  <-MISMATCH"
    print(f"{amp:5.1f} | {dn3:7.3f} {dn2:7.3f} ~{BASE[amp]:.2f} | "
          f"{ma3:8.4f} {ma2:8.4f} | {t3:6.2f} {t2:6.2f}{flag}",
          flush=True)
