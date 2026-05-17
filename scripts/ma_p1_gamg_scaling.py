"""P1 vs P2, GAMG vs direct, scaling with #triangles. P1 gives the
smallest / most AMG-friendly matrices — does P1+GAMG converge
robustly and scale where P2/P3+GAMG were erratic?  Grading expected
P1≈1.40, P2≈1.71 (P1 is ~18% weaker — this is a robustness/scaling
check, not a grading proposal). AMP=8. Per-row print (killable).
"""
from __future__ import annotations
import time
import numpy as np
import sympy
import underworld3 as uw
from underworld3.meshing.smoothing import (
    _winslow_elliptic, _edge_pairs, _auto_pinned_labels,
    _tri_cells, _WINSLOW_CACHE)

R_O, R_I, WIDTH, AMP = 1.0, 0.5, 0.12, 8.0
RESS = [16, 24, 32, 48, 64]


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


print(f"{'RES':>4} {'tris':>6} {'P':>2} {'solver':>7} | "
      f"{'cold':>6} {'warm':>6} | {'d/n':>6} | gamg KSP | ok?")
print("-" * 74)
for res in RESS:
    for pdeg in (1, 2):
        for mode in ("direct", "gamg"):
            m, f = case(res, f"{mode}{res}p{pdeg}")
            e = _edge_pairs(m.dm)
            ntri = _tri_cells(m.dm).shape[0]
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
            ks, ok = "", "y"
            if mode == "gamg":
                k = [kk for kk in _WINSLOW_CACHE
                     if kk[0] == id(m) and kk[-2] == "gamg"
                     and kk[-1] == pdeg][0]
                ksp = _WINSLOW_CACHE[k][1].snes.getKSP()
                rsn, nit = ksp.getConvergedReason(), ksp.getIterationNumber()
                ks = f"r={rsn} it={nit}"
                ok = "y" if (rsn > 0 and nit < 9999) else "FAIL"
            exp = 1.40 if pdeg == 1 else 1.71
            if abs(dn - exp) > 0.12:
                ok = "FAIL"
            print(f"{res:4d} {ntri:6d} {pdeg:2d} {mode:>7} | "
                  f"{cold:6.2f} {warm:6.2f} | {dn:6.3f} | "
                  f"{ks:>10} | {ok}", flush=True)
    print("-" * 74)
