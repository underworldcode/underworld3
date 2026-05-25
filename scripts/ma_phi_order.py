"""Does the φ-potential order drive the GAMG fragility?

Memory records P2/P3 give identical grading (the det term is inert;
the cap is structural). P3 is GAMG-hostile (high-order stiffness
defeats aggregation AMG). Test phi_degree ∈ {1,2,3} × {direct,gamg}
at the resolution where gamg failed (24) and one above (32).
Report grading (must stay ~1.71), cost, and the φ-KSP converged
reason / iters (gamg) — the robustness signal.
"""
from __future__ import annotations
import time
import numpy as np
import sympy
import underworld3 as uw
from underworld3.meshing.smoothing import (
    _winslow_elliptic, _edge_pairs, _auto_pinned_labels, _WINSLOW_CACHE)

R_O, R_I, WIDTH, AMP = 1.0, 0.5, 0.12, 8.0


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


def case(res, tag):
    m = uw.meshing.Annulus(radiusOuter=R_O, radiusInner=R_I,
                           cellSize=1.0 / res, qdegree=3)
    r0 = uw.discretisation.MeshVariable(
        f"r0_{tag}", m, vtype=uw.VarType.SCALAR,
        degree=1, continuous=True)
    X0 = np.asarray(m.X.coords)
    r0.data[:, 0] = np.sqrt((X0 ** 2).sum(axis=1))
    f = 1.0 + AMP * sympy.exp(-(((r0.sym[0]) - R_O) / WIDTH) ** 2)
    return m, f


print(f"{'RES':>4} {'P':>2} {'solver':>7} | {'cold':>6} {'warm':>6} "
      f"| {'d/n':>6} | φ-KSP reason/its (gamg)")
print("-" * 70)
for res in (24, 32):
    for pdeg in (1, 2, 3):
        for mode in ("direct", "gamg"):
            m, f = case(res, f"{mode}{res}p{pdeg}")
            e = _edge_pairs(m.dm)
            pin = _auto_pinned_labels(m)
            t = time.perf_counter()
            _winslow_elliptic(m, f, pin, False,
                              linear_solver=mode, phi_degree=pdeg)
            cold = time.perf_counter() - t
            dn = honest_ratio(np.asarray(m.X.coords), e)
            t = time.perf_counter()
            _winslow_elliptic(m, f, pin, False,
                              linear_solver=mode, phi_degree=pdeg)
            warm = time.perf_counter() - t
            ks = ""
            if mode == "gamg":
                k = [kk for kk in _WINSLOW_CACHE
                     if kk[0] == id(m) and kk[-2] == "gamg"
                     and kk[-1] == pdeg][0]
                ps = _WINSLOW_CACHE[k][1]
                ksp = ps.snes.getKSP()
                ks = (f"reason={ksp.getConvergedReason()} "
                      f"its={ksp.getIterationNumber()}")
            print(f"{res:4d} {pdeg:2d} {mode:>7} | {cold:6.2f} "
                  f"{warm:6.2f} | {dn:6.3f} | {ks}", flush=True)
    print("-" * 70)
