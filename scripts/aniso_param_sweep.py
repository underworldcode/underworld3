"""Characterise the anisotropic mover: sweep beta / aniso_cap /
relax on the interior radial feature (PEAK=0.70) and report
minA/meanA + rim-radial. Is there a stable regime, or is the
decoupled direct-Winslow form structurally folding-prone here?
"""
from __future__ import annotations
import numpy as np
import sympy
import underworld3 as uw
from underworld3.meshing.smoothing import (
    _winslow_anisotropic, _edge_pairs, _auto_pinned_labels,
    _tri_cells, _signed_areas)

R_O, R_I, WIDTH, RES, AMP, PEAK = 1.0, 0.5, 0.12, 16, 8.0, 0.70


def case(tag):
    m = uw.meshing.Annulus(radiusOuter=R_O, radiusInner=R_I,
                           cellSize=1.0 / RES, qdegree=3)
    r0 = uw.discretisation.MeshVariable(
        f"r0_{tag}", m, vtype=uw.VarType.SCALAR, degree=1,
        continuous=True)
    X0 = np.asarray(m.X.coords)
    r0.data[:, 0] = np.sqrt((X0 ** 2).sum(axis=1))
    f = 1.0 + AMP * sympy.exp(-(((r0.sym[0]) - PEAK) / WIDTH) ** 2)
    return m, f


m0, _ = case("u")
edges = _edge_pairs(m0.dm)
tris = _tri_cells(m0.dm)
X0 = np.asarray(m0.X.coords).copy()
A0 = np.abs(_signed_areas(X0, tris))
print(f"undeformed minA/meanA = {A0.min()/A0.mean():.4f}")


def split(coords):
    p0, p1 = coords[edges[:, 0]], coords[edges[:, 1]]
    mid = 0.5 * (p0 + p1)
    rm = np.linalg.norm(mid, axis=1)
    rh = mid / np.maximum(rm, 1e-30)[:, None]
    ev = p1 - p0
    L = np.linalg.norm(ev, axis=1)
    fr = np.abs((ev * rh).sum(axis=1)) / np.maximum(L, 1e-30)
    return rm, L, fr > np.cos(np.pi / 4)


print(f"\n{'beta':>6} {'cap':>4} {'relax':>5} {'nout':>4} | "
      f"{'minA/meanA':>10} {'rim-rad/dr0':>11} {'max|dx|':>9}")
print("-" * 60)
i = 0
for beta in (5.0, 20.0, 50.0, 200.0):
    for cap in (2.0, 4.0, 8.0):
        for relax in (0.2, 0.4):
            i += 1
            m, f = case(f"s{i}")
            pin = _auto_pinned_labels(m)
            try:
                _winslow_anisotropic(
                    m, f, pin, False, beta=beta, aniso_cap=cap,
                    relax=relax, n_outer=12)
            except Exception as e:
                print(f"{beta:6.0f} {cap:4.0f} {relax:5.2f}  ERR {e}")
                continue
            X = np.asarray(m.X.coords).copy()
            A = np.abs(_signed_areas(X, tris))
            mA = A.min() / A.mean()
            rm, L, isr = split(X)
            rim = rm > (R_O - 0.06)
            rr = L[rim & isr].mean() if (rim & isr).any() else np.nan
            dr0 = (R_O - R_I) / RES
            dx = float(np.linalg.norm(X - X0, axis=1).max())
            print(f"{beta:6.0f} {cap:4.0f} {relax:5.2f} {12:4d} | "
                  f"{mA:10.4f} {rr/dr0:11.3f} {dx:9.4f}")
