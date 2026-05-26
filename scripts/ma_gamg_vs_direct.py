"""BFO MA: parallel-scalable GAMG-reuse path vs the serial MUMPS
direct path. Same _winslow_elliptic, only linear_solver= differs.

Validates (a) grading bit-for-bit unchanged (d/n must match the
recorded 1.02/1.43/1.71/1.54), (b) cost cold+warm, (c) that the
factor/setup-once-reuse + Krylov warm-start actually fire — reported
as the φ-Poisson KSP iteration count per Picard iter (should be a
few once warm, NOT a fresh GAMG setup each time).
"""
from __future__ import annotations
import time
import numpy as np
import sympy
import underworld3 as uw
from underworld3.meshing.smoothing import (
    _winslow_elliptic, _edge_pairs, _auto_pinned_labels,
    _tri_cells, _signed_areas, _WINSLOW_CACHE)

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
    deep = (r >= R_I) & (r < R_I + 0.20)
    near = (r > R_O - 0.05)
    return float(nl[deep].mean() / nl[near].mean())


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


def ksp_its(m, amp):
    """Re-run AMP=8 with a per-Picard φ-KSP-iteration probe."""
    m2, f = case(amp, "kp")
    e = _edge_pairs(m2.dm)
    pin = _auto_pinned_labels(m2)
    _winslow_elliptic(m2, f, pin, False, linear_solver="gamg")
    # warm call, instrument the cached φ solver's KSP
    k = [kk for kk in _WINSLOW_CACHE
         if kk[0] == id(m2) and kk[-1] == "gamg"][0]
    phi, ps, gradphi, gproj, hsolver, vol = _WINSLOW_CACHE[k]
    raw = ps.solve
    its = []
    def w(*a, **kw):
        r = raw(*a, **kw)
        try:
            its.append(ps.snes.getKSP().getIterationNumber())
        except Exception:
            its.append(-1)
        return r
    ps.solve = w
    _winslow_elliptic(m2, f, pin, False, linear_solver="gamg")
    return its


print(f"{'AMP':>5} | {'direct':>16} | {'gamg':>16} | grading")
print(f"{'':>5} | {'cold':>7}{'warm':>9} | {'cold':>7}{'warm':>9} | "
      f"{'dir d/n':>8}{'gmg d/n':>9}  base")
print("-" * 74)
for k, amp in enumerate(AMPS):
    md, f = case(amp, f"d{k}")
    e = _edge_pairs(md.dm); pin = _auto_pinned_labels(md)
    t = time.perf_counter()
    _winslow_elliptic(md, f, pin, False, linear_solver="direct")
    dc = time.perf_counter() - t
    dn_d = honest_ratio(np.asarray(md.X.coords), e)
    t = time.perf_counter()
    _winslow_elliptic(md, f, pin, False, linear_solver="direct")
    dw = time.perf_counter() - t

    mg, f = case(amp, f"g{k}")
    e = _edge_pairs(mg.dm); pin = _auto_pinned_labels(mg)
    t = time.perf_counter()
    _winslow_elliptic(mg, f, pin, False, linear_solver="gamg")
    gc = time.perf_counter() - t
    dn_g = honest_ratio(np.asarray(mg.X.coords), e)
    t = time.perf_counter()
    _winslow_elliptic(mg, f, pin, False, linear_solver="gamg")
    gw = time.perf_counter() - t

    print(f"{amp:5.1f} | {dc:7.2f}{dw:9.2f} | {gc:7.2f}{gw:9.2f} | "
          f"{dn_d:8.3f}{dn_g:9.3f}  ~{BASE[amp]:.2f}", flush=True)

its = ksp_its(None, 8.0)
print(f"\nφ-Poisson KSP iters / Picard (gamg, AMP=8, warm call): "
      f"first={its[0]} rest={its[1:6]}... max={max(its)} "
      f"mean={np.mean(its):.1f}")
print("(low & flat ⇒ hierarchy built once + Krylov warm-start firing)")
