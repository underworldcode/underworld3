"""Smoke test for the locked-in adaptation API:
  uw.meshing.metric_density_from_gradient  + smooth_mesh_interior(
  method="anisotropic", method_kwargs=...).
Checks: public import, the helper's cache (callable per-step with
no duplicate-MeshVariable error), method_kwargs pass-through, a
valid moved mesh, and the uniform-field ~no-op.
"""
from __future__ import annotations
import numpy as np
import sympy
import underworld3 as uw
from underworld3.meshing import (
    smooth_mesh_interior, metric_density_from_gradient)
from underworld3.meshing.smoothing import _tri_cells, _signed_areas

R_O, R_I, RES = 1.0, 0.5, 16


def fresh():
    m = uw.meshing.Annulus(radiusOuter=R_O, radiusInner=R_I,
                           cellSize=1.0 / RES, qdegree=3)
    T = uw.discretisation.MeshVariable(
        "T", m, vtype=uw.VarType.SCALAR, degree=3, continuous=True)
    return m, T


# 1. gradient-driven density on a feature field, then move
m, T = fresh()
X0 = np.asarray(m.X.coords).copy()
tris = _tri_cells(m.dm)
r = np.sqrt((np.asarray(T.coords) ** 2).sum(axis=1))
T.data[:, 0] = np.exp(-((r - 0.7) / 0.12) ** 2)      # a "front"
rho = metric_density_from_gradient(m, T, amp=8.0)
# cache test: a second call must NOT raise (no dup MeshVariable)
rho2 = metric_density_from_gradient(m, T, amp=8.0)
print("metric_density_from_gradient cache OK (2 calls)")
smooth_mesh_interior(m, metric=rho, method="anisotropic",
                     method_kwargs=dict(aniso_cap=2.0, relax=0.2,
                                        n_outer=8), verbose=True)
X1 = np.asarray(m.X.coords).copy()
a = _signed_areas(X1, tris)
orient = np.sign(np.median(_signed_areas(X0, tris))) or 1.0
moved = float(np.linalg.norm(X1 - X0, axis=1).max())
mA = np.abs(a).min() / np.abs(a).mean()
print(f"moved max|Δx|={moved:.4e}  minA/meanA={mA:.4f}  "
      f"valid={(a*orient).min() > 0}")
assert (a * orient).min() > 0.0, "tangled mesh"
assert moved > 1e-4, "expected node movement on a gradient metric"

# 2. method_kwargs really reaches the mover (bad kwarg → TypeError)
try:
    smooth_mesh_interior(m, metric=rho, method="anisotropic",
                         method_kwargs=dict(not_a_real_kwarg=1))
    raise AssertionError("bad method_kwargs silently accepted")
except TypeError:
    print("method_kwargs pass-through OK (unknown kwarg → TypeError)")

# 3. uniform field ⇒ ρ≈1 ⇒ ~no-op (gradient ~0 everywhere)
m3, T3 = fresh()
T3.data[:, 0] = 1.0
X0 = np.asarray(m3.X.coords).copy()
rho3 = metric_density_from_gradient(m3, T3, amp=8.0)
smooth_mesh_interior(m3, metric=rho3, method="anisotropic",
                     method_kwargs=dict(n_outer=4))
d = float(np.linalg.norm(
    np.asarray(m3.X.coords) - X0, axis=1).max())
print(f"uniform-field move max|Δx|={d:.2e} (≈ no-op)")
assert d < 1e-3, f"uniform field should be ~no-op, got {d:.2e}"

print("API smoke PASSED")
