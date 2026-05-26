"""Decisive diagnostic: compare the FE Monge-Ampere realized radial
node displacement to the EXACT equidistribution displacement.

Exact: node initially at fractional radial position xi (uniform
start) must move to r_eq(xi) = interp(xi, cumulative-mass, s). The
required radial displacement profile dr_exact(r0) is what grad(phi)
must reproduce. If the FE dr is an order of magnitude smaller / a
near-uniform shift rather than the sharply varying exact profile,
the FE MA solve is converging to a spurious (non-Brenier) branch,
not just an inaccurate Hessian.
"""
from __future__ import annotations
import numpy as np
import sympy
import underworld3 as uw
from underworld3.meshing.smoothing import (
    _winslow_elliptic, _auto_pinned_labels)

R_I, R_O = 0.5, 1.0
WIDTH = 0.12
AMP = 8.0
RES = 16

# ---- exact radial equidistribution map ----
N = 200_000
s = np.linspace(R_I, R_O, N)
ds = s[1] - s[0]
rho = 1.0 + AMP * np.exp(-(((s - R_O) / WIDTH) ** 2))
dens = rho * s
m = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * ds)])
m /= m[-1]


def r_exact(r0):
    xi = (r0 - R_I) / (R_O - R_I)
    return np.interp(xi, m, s)


# ---- FE Monge-Ampere solve ----
mesh = uw.meshing.Annulus(radiusOuter=R_O, radiusInner=R_I,
                          cellSize=1.0 / RES, qdegree=3)
r0v = uw.discretisation.MeshVariable(
    "r0diag", mesh, vtype=uw.VarType.SCALAR, degree=1, continuous=True)
X0 = np.asarray(mesh.X.coords)
r0v.data[:, 0] = np.sqrt((X0 ** 2).sum(axis=1))
f = 1.0 + AMP * sympy.exp(-(((r0v.sym[0]) - R_O) / WIDTH) ** 2)

c0 = np.asarray(mesh.X.coords).copy()
rad0 = np.sqrt((c0 ** 2).sum(axis=1))
pinned = _auto_pinned_labels(mesh)
_winslow_elliptic(mesh, f, pinned, True, n_picard=120, relax=1.0,
                  step_frac=None, picard_relax=0.25)
c1 = np.asarray(mesh.X.coords).copy()
rad1 = np.sqrt((c1 ** 2).sum(axis=1))

dr_fe = rad1 - rad0
dr_ex = r_exact(rad0) - rad0

# bin by initial radius
edges = np.linspace(R_I, R_O, 11)
print(f"\nAMP={AMP}  radial displacement: FE vs EXACT (by r0 bin)")
print(f"{'r0 bin':>14}  {'mean dr_FE':>11}  {'mean dr_EXACT':>13}  "
      f"{'ratio':>7}")
for k in range(len(edges) - 1):
    lo, hi = edges[k], edges[k + 1]
    sel = (rad0 >= lo) & (rad0 < hi)
    if sel.sum() == 0:
        continue
    fe = dr_fe[sel].mean()
    ex = dr_ex[sel].mean()
    rr = fe / ex if abs(ex) > 1e-9 else float("nan")
    print(f"  [{lo:.3f},{hi:.3f})  {fe:>11.4f}  {ex:>13.4f}  "
          f"{rr:>7.3f}")
print(f"\n max|dr_FE|={np.abs(dr_fe).max():.4f}   "
      f"max|dr_EXACT|={np.abs(dr_ex).max():.4f}")
