"""Smoke test for the (3) anisotropic tensor mover.

Radial Gaussian feature on a res-16 Annulus. Confirms: runs, the
mesh stays valid (no inverted cell), AMP=0 is an exact no-op, and
the move is non-trivial for AMP>0. Anisotropy-aware numbers + a
render come in the proper validation script; this is plumbing only.
"""
from __future__ import annotations
import time
import numpy as np
import sympy
import underworld3 as uw
from underworld3.meshing.smoothing import (
    _winslow_anisotropic, _auto_pinned_labels, _tri_cells,
    _signed_areas)

R_O, R_I, WIDTH, RES = 1.0, 0.5, 0.12, 16


def case(amp, tag, **kw):
    m = uw.meshing.Annulus(radiusOuter=R_O, radiusInner=R_I,
                           cellSize=1.0 / RES, qdegree=3)
    r0 = uw.discretisation.MeshVariable(
        f"r0_{tag}", m, vtype=uw.VarType.SCALAR, degree=1,
        continuous=True)
    X0 = np.asarray(m.X.coords).copy()
    r0.data[:, 0] = np.sqrt((X0 ** 2).sum(axis=1))
    f = 1.0 + amp * sympy.exp(-(((r0.sym[0]) - R_O) / WIDTH) ** 2)
    pin = _auto_pinned_labels(m)
    tris = _tri_cells(m.dm)
    a0 = _signed_areas(X0, tris)
    orient = np.sign(np.median(a0)) or 1.0
    t = time.perf_counter()
    _winslow_anisotropic(m, f, pin, True, **kw)
    dt = time.perf_counter() - t
    X1 = np.asarray(m.X.coords).copy()
    a1 = _signed_areas(X1, tris) * orient
    moved = float(np.linalg.norm(X1 - X0, axis=1).max())
    valid = bool(a1.min() > 0.0)
    print(f"[{tag}] amp={amp:5.1f} kw={kw}  time={dt:5.2f}s  "
          f"max|Δx|={moved:.4e}  minA*orient={a1.min():.3e}  "
          f"valid={valid}")
    return moved, valid


print("=== anisotropic mover smoke test (res-16 Annulus) ===")
m0, v0 = case(0.0, "amp0", n_outer=3)
assert v0, "AMP=0 produced an invalid mesh"
assert m0 < 1e-9, f"AMP=0 must be an exact no-op, got max|Δx|={m0:.2e}"
print("  -> AMP=0 exact no-op OK")

m8, v8 = case(8.0, "amp8", n_outer=5)
assert v8, "AMP=8 produced an invalid (tangled) mesh"
assert m8 > 1e-4, f"AMP=8 should move nodes, got {m8:.2e}"
print("  -> AMP=8 moves nodes, mesh valid OK")

print("smoke test PASSED")
