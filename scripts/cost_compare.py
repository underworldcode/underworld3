"""Cost vs grading: time the elastic-spring and the Monge–Ampère
(BFO) metric paths on identical undeformed Annulus setups, with the
HONEST deep/near metric. Per-step printed so it is killable early.

Wall time is the cold cost a caller sees on the FIRST call (it
includes one-time PETSc DM / JIT solver build for MA and the edge
cache for spring; subsequent calls on the same-topology mesh reuse
the cache — a warm re-time is also reported).
"""
from __future__ import annotations
import time
import numpy as np
import sympy
import underworld3 as uw
from underworld3.meshing.smoothing import (
    _winslow_elliptic, _edge_pairs, _auto_pinned_labels)

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


print(f"{'AMP':>5} {'nodes':>6} {'edges':>6} | "
      f"{'spring t/s':>10} {'warm':>6} {'d/n':>5} | "
      f"{'MA t/s':>8} {'warm':>6} {'d/n':>5}")
print("-" * 78)
for k, amp in enumerate(AMPS):
    # spring (public API default path)
    m, f = case(amp, f"sp{k}")
    e = _edge_pairs(m.dm)
    nv, ne = np.asarray(m.X.coords).shape[0], e.shape[0]
    t = time.perf_counter()
    uw.meshing.smooth_mesh_interior(m, metric=f, verbose=False)
    ts_cold = time.perf_counter() - t
    rs = honest_ratio(np.asarray(m.X.coords), e)
    t = time.perf_counter()                       # warm re-call
    uw.meshing.smooth_mesh_interior(m, metric=f, verbose=False)
    ts_warm = time.perf_counter() - t

    # MA (preserved, called directly)
    m, f = case(amp, f"ma{k}")
    e = _edge_pairs(m.dm)
    pin = _auto_pinned_labels(m)
    t = time.perf_counter()
    _winslow_elliptic(m, f, pin, False)
    tm_cold = time.perf_counter() - t
    rm = honest_ratio(np.asarray(m.X.coords), e)
    t = time.perf_counter()                       # warm re-call
    _winslow_elliptic(m, f, pin, False)
    tm_warm = time.perf_counter() - t

    print(f"{amp:5.1f} {nv:6d} {ne:6d} | "
          f"{ts_cold:10.2f} {ts_warm:6.2f} {rs:5.2f} | "
          f"{tm_cold:8.2f} {tm_warm:6.2f} {rm:5.2f}", flush=True)
