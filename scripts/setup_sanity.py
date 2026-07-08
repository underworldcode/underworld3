"""Is the metric-grading SETUP correct? Checks the things common to
EVERY method (so a bug here would make them all look weak):

  1. fraction of vertices that are PINNED (over-pinning ⇒ nothing
     can move regardless of method)
  2. the metric ρ_tgt actually seen by the smoother
     (uw.function.evaluate(metric, coords)) — min/max + radial
     profile (flat ⇒ no target ⇒ no grading possible)
  3. the spring rest-length field L0 deep-vs-near ratio
     (should be ~ (ρ_near/ρ_deep)^(1/2) ≈ 3 for AMP=8)
  4. how far nodes actually move when smoothed, by radius band
"""
from __future__ import annotations
import numpy as np
import sympy
import underworld3 as uw
from underworld3.meshing import smoothing as S

R_I, R_O, W, AMP, RES = 0.5, 1.0, 0.12, 8.0, 16

mesh = uw.meshing.Annulus(radiusOuter=R_O, radiusInner=R_I,
                          cellSize=1.0 / RES, qdegree=3)
dm = mesh.dm
pS0, pE0 = dm.getDepthStratum(0)
nv = pE0 - pS0

r0 = uw.discretisation.MeshVariable(
    "r0s", mesh, vtype=uw.VarType.SCALAR, degree=1, continuous=True)
X0 = np.asarray(mesh.X.coords)
rad0 = np.sqrt((X0 ** 2).sum(axis=1))
r0.data[:, 0] = rad0
metric = 1.0 + AMP * sympy.exp(-(((r0.sym[0]) - R_O) / W) ** 2)

# --- 1. pinning ---
pinned_labels = S._auto_pinned_labels(mesh)
# re-fetch dm AFTER MeshVariable creation (stale-DM footgun)
dm = mesh.dm
is_pinned = S._pinned_mask(dm, pinned_labels)
print(f"boundaries on mesh : "
      f"{[getattr(b,'name',None) for b in mesh.boundaries]}")
print(f"auto-pinned labels : {pinned_labels}")
print(f"vertices           : {nv}")
print(f"pinned             : {is_pinned.sum()} "
      f"({100.0*is_pinned.sum()/nv:.1f}%)   "
      f"free: {(~is_pinned).sum()}")
# how many pinned are actually ON a boundary ring (r≈R_I or R_O)?
on_ring = (np.abs(rad0 - R_O) < 1e-6) | (np.abs(rad0 - R_I) < 1e-6)
print(f"verts on a ring    : {on_ring.sum()}   "
      f"pinned-but-interior: "
      f"{int((is_pinned & ~on_ring).sum())}")

# --- 2. metric actually seen by the smoother ---
coords = np.asarray(mesh.X.coords)
rho = np.asarray(uw.function.evaluate(metric, coords)).reshape(-1)
print(f"\nrho via uw.function.evaluate(metric, coords):")
print(f"  min={rho.min():.4f}  max={rho.max():.4f}  "
      f"mean={rho.mean():.4f}")
# radial profile
for lo in np.linspace(R_I, R_O, 6)[:-1]:
    hi = lo + (R_O - R_I) / 5
    m = (rad0 >= lo) & (rad0 < hi)
    if m.any():
        print(f"  r∈[{lo:.2f},{hi:.2f})  rho mean={rho[m].mean():.3f}"
              f"  (analytic f={1+AMP*np.exp(-(((lo+hi)/2-R_O)/W)**2):.3f})")

# --- 3. rest-length field L0 ---
edges = S._edge_pairs(dm)
v0, v1 = edges[:, 0], edges[:, 1]
w = np.maximum(rho, 1e-30) ** (-0.5)
w_edge = 0.5 * (w[v0] + w[v1])
e = coords[v1] - coords[v0]
Lc = np.linalg.norm(e, axis=1)
L0 = (Lc.sum() / w_edge.sum()) * w_edge
emid = 0.5 * (rad0[v0] + rad0[v1])
near = emid > (R_O - W)
deep = emid < (R_O - 0.30)
rho_edge = 0.5 * (rho[v0] + rho[v1])
print(f"\nspring rest length L0:")
print(f"  near-band mean L0={L0[near].mean():.4f}  "
      f"deep mean L0={L0[deep].mean():.4f}  "
      f"deep/near={L0[deep].mean()/L0[near].mean():.2f}  "
      f"(want (rho_near/rho_deep)^0.5 = "
      f"{(rho_edge[near].mean()/rho_edge[deep].mean())**0.5:.2f})")

# --- 4. actual node motion after smoothing ---
S.smooth_mesh_interior(mesh, metric=metric, verbose=False)
c1 = np.asarray(mesh.X.coords)
dr = c1 - X0
dmag = np.linalg.norm(dr, axis=1)
print(f"\nnode displacement after smooth_mesh_interior:")
print(f"  max|dx|={dmag.max():.4f}  mean|dx|={dmag.mean():.4f}  "
      f"moved(>1e-9): {(dmag>1e-9).sum()}/{nv}")
for lo in np.linspace(R_I, R_O, 6)[:-1]:
    hi = lo + (R_O - R_I) / 5
    m = (rad0 >= lo) & (rad0 < hi)
    if m.any():
        print(f"  r0∈[{lo:.2f},{hi:.2f})  mean|dx|={dmag[m].mean():.4f}"
              f"  mean Δr={(np.sqrt((c1[m]**2).sum(1))-rad0[m]).mean():+.4f}")

# --- 5. HONEST per-node radial-spacing metric (not centroid-band) ---
# For each FINAL node, its mean incident edge length, binned by its
# FINAL radius. Strong grading ⇒ near-surface mean edge ≪ deep.
rad1 = np.sqrt((c1 ** 2).sum(axis=1))
ev1 = c1[v1] - c1[v0]
Le1 = np.linalg.norm(ev1, axis=1)
node_edgelen = np.zeros(nv)
cnt = np.zeros(nv)
for a in (v0, v1):
    np.add.at(node_edgelen, a, Le1)
    np.add.at(cnt, a, 1.0)
node_edgelen /= np.maximum(cnt, 1.0)

# initial (uniform-ish) reference
ev0 = X0[v1] - X0[v0]
Le0 = np.linalg.norm(ev0, axis=1)
node_edgelen0 = np.zeros(nv)
cnt0 = np.zeros(nv)
for a in (v0, v1):
    np.add.at(node_edgelen0, a, Le0)
    np.add.at(cnt0, a, 1.0)
node_edgelen0 /= np.maximum(cnt0, 1.0)

print("\nHONEST metric — mean incident edge length by FINAL radius:")
print(f"{'radius band':>14} {'before':>9} {'after':>9} {'after/bef':>10}")
bands = [(0.50, 0.70), (0.70, 0.85), (0.85, 0.95), (0.95, 1.00)]
res = {}
for lo, hi in bands:
    mb = (rad1 >= lo) & (rad1 < hi)
    m0 = (rad0 >= lo) & (rad0 < hi)
    if mb.any() and m0.any():
        a_ = node_edgelen[mb].mean()
        b_ = node_edgelen0[m0].mean()
        res[(lo, hi)] = a_
        print(f"  [{lo:.2f},{hi:.2f})  {b_:9.4f} {a_:9.4f} "
              f"{a_/b_:10.3f}")
if (0.50, 0.70) in res and (0.95, 1.00) in res:
    print(f"\n  >>> deep/near edge ratio AFTER = "
          f"{res[(0.50,0.70)]/res[(0.95,1.00)]:.2f}  "
          f"(uniform start ≈ 1.0; exact equidistribution ~10)")
