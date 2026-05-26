"""Cost scaling: serial MUMPS direct vs GAMG-reuse, AMP=8, as the
Annulus is refined. Direct is optimal for tiny 2D problems; the
question is whether the GAMG-reuse path's cost grows *slower* (the
parallel/3D-scalable argument). Grading must stay bit-for-bit at
every resolution. Per-RES print (killable).
"""
from __future__ import annotations
import time
import numpy as np
import sympy
import underworld3 as uw
from underworld3.meshing.smoothing import (
    _winslow_elliptic, _edge_pairs, _auto_pinned_labels)

R_O, R_I, WIDTH, AMP = 1.0, 0.5, 0.12, 8.0
RESS = [16, 24, 32, 48]


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


def timed(m, f, mode):
    e = _edge_pairs(m.dm); pin = _auto_pinned_labels(m)
    t = time.perf_counter()
    _winslow_elliptic(m, f, pin, False, linear_solver=mode)
    cold = time.perf_counter() - t
    dn = honest_ratio(np.asarray(m.X.coords), e)
    t = time.perf_counter()
    _winslow_elliptic(m, f, pin, False, linear_solver=mode)
    warm = time.perf_counter() - t
    return cold, warm, dn


print(f"{'RES':>4} {'nodes':>7} | {'direct cold/warm':>18} | "
      f"{'gamg cold/warm':>18} | {'warm ratio':>10} | d/n")
print("-" * 78)
for res in RESS:
    md, f = case(res, f"d{res}")
    nv = np.asarray(md.X.coords).shape[0]
    dc, dw, dnd = timed(md, f, "direct")
    mg, f = case(res, f"g{res}")
    gc, gw, dng = timed(mg, f, "gamg")
    ratio = gw / dw if dw > 0 else float("nan")
    flag = "" if abs(dnd - dng) < 5e-3 else "  <-GRADING MISMATCH"
    print(f"{res:4d} {nv:7d} | {dc:8.2f}{dw:9.2f}   | "
          f"{gc:8.2f}{gw:9.2f}   | {ratio:9.2f}x | "
          f"{dnd:.3f}/{dng:.3f}{flag}", flush=True)
print("\n(direct optimal for tiny 2D; watch whether the gamg/direct "
      "warm ratio shrinks with N — the 3D/parallel-scalable signal)")
