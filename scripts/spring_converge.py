"""Did the spring PCG actually converge — and does converging it
properly grade more? Run AMP=8 with increasing iteration budgets,
report final |g| (→0 ⇒ true equilibrium reached), rms(L-L0)/L0,
the HONEST deep/near grading, and wall time.

If |g|→~1e-6 and grading plateaus ⇒ the converged spring
equilibrium IS that weak (fixed-topology frustration). If grading
keeps rising as |g| falls ⇒ 300 iters was just too few (cheap to
fix — more PCG iters).
"""
from __future__ import annotations
import time
import numpy as np
import sympy
import underworld3 as uw
from underworld3.meshing.smoothing import (
    _winslow_spring, _edge_pairs, _auto_pinned_labels)

R_O, R_I, WIDTH, RES, AMP = 1.0, 0.5, 0.12, 16, 8.0


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


for n in (300, 1000, 3000, 10000):
    m = uw.meshing.Annulus(radiusOuter=R_O, radiusInner=R_I,
                           cellSize=1.0 / RES, qdegree=3)
    r0 = uw.discretisation.MeshVariable(
        f"r0_{n}", m, vtype=uw.VarType.SCALAR,
        degree=1, continuous=True)
    X0 = np.asarray(m.X.coords)
    r0.data[:, 0] = np.sqrt((X0 ** 2).sum(axis=1))
    f = 1.0 + AMP * sympy.exp(-(((r0.sym[0]) - R_O) / WIDTH) ** 2)
    e = _edge_pairs(m.dm)
    pin = _auto_pinned_labels(m)
    t = time.perf_counter()
    # verbose prints the final "spring PCG iter N/N: ... |g|=..."
    _winslow_spring(m, f, pin, True, n_sweeps=n)
    dt = time.perf_counter() - t
    rr = honest_ratio(np.asarray(m.X.coords), e)
    print(f"==> n_sweeps={n:>6}  time={dt:7.2f}s  "
          f"honest deep/near={rr:.3f}", flush=True)
