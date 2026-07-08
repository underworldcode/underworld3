"""(3) mover — GAMG parity + cost-per-step characterisation.

Two extensibility questions:

1. GAMG.  The mover is NON-singular (homogeneous Dirichlet, no
   constant nullspace), so unlike the MA pure-Neumann path GAMG
   should be robust here. Check grading/quality PARITY (gamg must
   match direct) and cost across resolutions.
2. Cost per step.  Decompose into
     * COLD  — fresh mesh: MeshVariable + solver creation + 1st
               factorisation (one-off per remesh / topology change)
     * WARM  — same mesh object again (cache hit): the genuine
               per-timestep cost in a dynamic-adaptive loop
     * D-build (gproj solve + the per-node eigen-clamp) vs the
       per-outer-step displacement solves (n_outer=1 vs default
       slope).
   Scales with #triangles (res 16→48) → the parallel / 3D
   extrapolation.

Interior radial feature (clean cap=2 regime). minA/meanA is the
parity check.
"""
from __future__ import annotations
import time
import numpy as np
import sympy
import underworld3 as uw
from underworld3.meshing.smoothing import (
    _winslow_anisotropic, _auto_pinned_labels, _tri_cells,
    _signed_areas, _edge_pairs)

R_O, R_I, WIDTH, AMP, PEAK = 1.0, 0.5, 0.12, 8.0, 0.70


def case(tag, res):
    m = uw.meshing.Annulus(radiusOuter=R_O, radiusInner=R_I,
                           cellSize=1.0 / res, qdegree=3)
    r0 = uw.discretisation.MeshVariable(
        f"r0_{tag}", m, vtype=uw.VarType.SCALAR, degree=1,
        continuous=True)
    X0 = np.asarray(m.X.coords)
    r0.data[:, 0] = np.sqrt((X0 ** 2).sum(axis=1))
    f = 1.0 + AMP * sympy.exp(-(((r0.sym[0]) - PEAK) / WIDTH) ** 2)
    return m, f


def quality(m, tris):
    A = np.abs(_signed_areas(np.asarray(m.X.coords), tris))
    return A.min() / A.mean()


print(f"{'res':>4} {'ntri':>6} {'solver':>7} | {'cold':>7} "
      f"{'warm':>7} {'warm/out':>9} {'Dbuild':>7} | "
      f"{'minA/meanA':>10}")
print("-" * 74)

for res in (16, 24, 32, 48):
    m_probe, _ = case("probe", res)
    ntri = _tri_cells(m_probe.dm).shape[0]
    del m_probe
    row = {}
    for solver in ("direct", "gamg"):
        # COLD: fresh mesh, full setup, n_outer=12 (default)
        m, f = case(f"{solver}_c{res}", res)
        tris = _tri_cells(m.dm)
        pin = _auto_pinned_labels(m)
        X_und = np.asarray(m.X.coords).copy()   # undeformed coords
        t0 = time.perf_counter()
        _winslow_anisotropic(m, f, pin, False,
                             linear_solver=solver)
        t_cold = time.perf_counter() - t0
        mA = quality(m, tris)
        # WARM: SAME mesh object again (cache hit) — the real
        # per-adaptation-step cost in a dynamic loop. Restore the
        # undeformed coords so it does identical work.
        m._deform_mesh(X_und.copy())
        t0 = time.perf_counter()
        _winslow_anisotropic(m, f, pin, False,
                             linear_solver=solver)
        t_warm = time.perf_counter() - t0
        # n_outer=1 on the warm cache → fixed (D-build + 1 solve)
        m._deform_mesh(X_und.copy())
        t0 = time.perf_counter()
        _winslow_anisotropic(m, f, pin, False, n_outer=1,
                             linear_solver=solver)
        t_n1 = time.perf_counter() - t0
        per_out = (t_warm - t_n1) / 11.0          # 12 vs 1 slope
        row[solver] = (t_cold, t_warm, per_out, t_n1, mA)
        print(f"{res:4d} {ntri:6d} {solver:>7} | {t_cold:7.2f} "
              f"{t_warm:7.2f} {per_out:9.3f} {t_n1:7.2f} | "
              f"{mA:10.4f}")
    d = row["direct"]
    g = row["gamg"]
    print(f"{'':>4} {'':>6} {'Δ(g-d)':>7} | "
          f"{g[0]-d[0]:+7.2f} {g[1]-d[1]:+7.2f} "
          f"{g[2]-d[2]:+9.3f} {'':>7} | "
          f"parity |minA_g-minA_d|={abs(g[4]-d[4]):.2e}")

print("\n(cold = one-off per remesh: var+solver creation + 1st "
      "factorisation;\n warm = per-timestep cost in a dynamic "
      "loop = D-build + n_outer solves;\n warm/out = marginal cost "
      "of one extra MMPDE outer step;\n Dbuild ≈ n_outer=1 warm "
      "= gproj ∇ρ solve + per-node eigen-clamp + 1 disp solve.)")
