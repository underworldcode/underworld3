"""(3) anisotropic mover — NON-SEPARABLE feature: the case it
earns its keep (kickoff brief).

Compact Cartesian Gaussian blob at an interior point P=(0.78,0):
ρ = 1 + AMP·exp(-|X-P|²/W²). Neither pure-r nor pure-θ, so the
explicit 1-D OT (exact + cheap for separable features) does NOT
apply — this is the regime the general tensor mover is for.

Compare (3) vs the isotropic scalar paths (MA, spring) on:
  * minA/meanA               — sliver / quality (higher = better)
  * far/near edge ratio       — concentration toward the blob
  * fraction of nodes within W of P  — did nodes migrate in
and SHOW the grids (zoomed on the blob). Success per the brief:
(3) gives cleaner, blob-aligned cells (fewer slivers) at
comparable concentration — it does NOT beat the node-count cap.
"""
from __future__ import annotations
import numpy as np
import sympy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import underworld3 as uw
from underworld3.meshing.smoothing import (
    _winslow_elliptic, _winslow_anisotropic, _winslow_spring,
    _edge_pairs, _auto_pinned_labels, _tri_cells, _signed_areas)

R_O, R_I, RES, AMP = 1.0, 0.5, 24, 8.0
PX, PY, W = 0.78, 0.0, 0.10


def mk(tag):
    m = uw.meshing.Annulus(radiusOuter=R_O, radiusInner=R_I,
                           cellSize=1.0 / RES, qdegree=3)
    Xv = uw.discretisation.MeshVariable(
        f"X0_{tag}", m, vtype=uw.VarType.VECTOR, degree=1,
        continuous=True)
    X0 = np.asarray(m.X.coords)
    Xv.data[:, 0] = X0[:, 0]
    Xv.data[:, 1] = X0[:, 1]
    x, y = Xv.sym[0], Xv.sym[1]
    f = 1.0 + AMP * sympy.exp(
        -(((x - PX) ** 2 + (y - PY) ** 2) / W ** 2))
    return m, f, X0.copy()


m0, _, X0u = mk("u")
edges = _edge_pairs(m0.dm)
tris = _tri_cells(m0.dm)


def report(name, X, X0):
    d = np.hypot(X[:, 0] - PX, X[:, 1] - PY)
    nv = X.shape[0]
    v0, v1 = edges[:, 0], edges[:, 1]
    Le = np.linalg.norm(X[v1] - X[v0], axis=1)
    s = np.zeros(nv)
    cc = np.zeros(nv)
    for a in (v0, v1):
        np.add.at(s, a, Le)
        np.add.at(cc, a, 1.0)
    nl = s / np.maximum(cc, 1.0)
    near, far = d < 1.5 * W, d > 5 * W
    fn = float(nl[far].mean() / nl[near].mean())
    frac = float((d < 1.5 * W).mean())
    A = np.abs(_signed_areas(X, tris))
    minA = float(A.min() / A.mean())
    drift = float(np.abs(np.hypot(X[:, 0], X[:, 1])
                         - np.hypot(X0[:, 0], X0[:, 1])).max())
    print(f"  {name:<18} far/near={fn:5.2f}  frac@P={frac:6.4f}  "
          f"minA/meanA={minA:6.4f}  rdrift={drift:.2e}")
    return fn, frac, minA


print(f"Non-separable blob P=({PX},{PY}) W={W}, AMP={AMP}, "
      f"res-{RES}  (uniform frac@P≈"
      f"{float((np.hypot(X0u[:,0]-PX,X0u[:,1]-PY)<1.5*W).mean()):.4f})")
report("undeformed", X0u, X0u)
panels = [("undeformed", X0u)]

for tag, name, fn in [
    ("ma", "scalar MA", lambda m, f, p: _winslow_elliptic(
        m, f, p, False, phi_degree=2)),
    ("sp", "spring", lambda m, f, p: _winslow_spring(
        m, f, p, False)),
    ("an", "(3) anisotropic", lambda m, f, p: _winslow_anisotropic(
        m, f, p, True, phi_degree=2))]:
    m, f, X0 = mk(tag)
    fn(m, f, _auto_pinned_labels(m))
    X = np.asarray(m.X.coords).copy()
    report(name, X, X0)
    panels.append((name, X))

fig, ax = plt.subplots(2, 4, figsize=(19, 9.6))
tr = mtri.Triangulation(X0u[:, 0], X0u[:, 1], tris)
for k, (name, X) in enumerate(panels):
    a = ax[0, k]
    a.triplot(mtri.Triangulation(X[:, 0], X[:, 1], tris),
              lw=0.3, color="#1f4e8c")
    a.plot(PX, PY, "o", ms=10, mfc="none", mec="#c0392b", mew=2)
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])
    a.set_title(name, fontsize=12)
    b = ax[1, k]
    b.triplot(mtri.Triangulation(X[:, 0], X[:, 1], tris),
              lw=0.5, color="#1f4e8c")
    b.plot(PX, PY, "o", ms=12, mfc="none", mec="#c0392b", mew=2)
    b.set_aspect("equal")
    b.set_xlim(PX - 0.28, PX + 0.28)
    b.set_ylim(PY - 0.28, PY + 0.28)
    b.set_xticks([])
    b.set_yticks([])
    b.set_title(f"{name} — zoom on blob", fontsize=11)
fig.suptitle(f"NON-SEPARABLE blob (the case (3) is for) — P="
             f"({PX},{PY}), res-{RES} AMP={AMP}  "
             f"(success = cleaner, blob-aligned cells / fewer "
             f"slivers, NOT a bigger far/near)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("/tmp/metric_mesh/aniso_nonsep.png", dpi=125)
print("\nsaved /tmp/metric_mesh/aniso_nonsep.png")
