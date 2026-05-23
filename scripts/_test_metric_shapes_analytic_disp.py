"""Diagnostic probe: per-iter max|Δx| trajectory with an
*analytic* (Eulerian) sympy ρ vs. the Lagrangian
`metric_density_from_gradient` ρ.

If the user's hypothesis is correct, then with analytic ρ +
`metric_refresh_per_iter=True` the per-iter displacement should
decay roughly geometrically (true damped Picard) rather than
stalling at ~constant (the documented Lagrangian failure).

The analytic ρ mirrors the three synthetic shapes in
``_test_metric_shapes.py`` via sech²-banded smooth signed
distances — a pure sympy function of mesh.X, no MeshVariable
behind it.
"""
import io
import contextlib
import os
import re
import sys
import numpy as np
import sympy
import underworld3 as uw

sys.path.insert(0, os.path.dirname(__file__))
from _test_metric_shapes import build_mesh_with_field

EPS = 0.04       # band width in ρ for analytic Eulerian probe
AMP = 8.0        # peak (boundary) excess over bulk: ρ_peak ≈ 1+AMP
SOFT = 0.005     # smooth-max softness (units of coords)


def sym_smax(a, b, soft=SOFT):
    """Algebraic smooth max — sqrt of an Add, C-printable.

    (a+b+sqrt((a−b)² + s²))/2 → max(a,b) as s→0, smooth elsewhere.
    """
    return (a + b + sympy.sqrt((a - b) ** 2 + soft ** 2)) / 2


def sym_smin(a, b, soft=SOFT):
    return -sym_smax(-a, -b, soft)


def sym_smin3(a, b, c, soft=SOFT):
    return sym_smin(sym_smin(a, b, soft), c, soft)


def sym_sabs(x, soft=SOFT):
    """Smooth abs via sqrt (used for the square SDF)."""
    return sympy.sqrt(x * x + soft * soft)


def sym_sech2(z):
    """sech²(z) = 1/cosh²(z) — printable in C99 (cosh is supported)."""
    return 1 / sympy.cosh(z) ** 2


def analytic_rho(mesh):
    """Build sympy ρ(x,y) = 1 + AMP · Σ sech²(d_shape / EPS),
    mirroring the three shapes from _test_metric_shapes.py.

    Pure sympy expression in mesh.X — truly Eulerian.
    """
    X = mesh.CoordinateSystem.X
    x, y = X[0], X[1]

    # Square: centre (0.55, 0.35), side 0.4, angle 30°.
    cx_sq, cy_sq, side = 0.55, 0.35, 0.4
    ang_rad = 30.0 * np.pi / 180.0
    ct, st = float(np.cos(ang_rad)), float(np.sin(ang_rad))
    dxs, dys = x - cx_sq, y - cy_sq
    xp = ct * dxs + st * dys
    yp = -st * dxs + ct * dys
    d_sq = side / 2 - sym_smax(sym_sabs(xp), sym_sabs(yp))

    # Doughnut: centre (-0.55, 0.45), r ∈ [0.15, 0.30].
    cx_dh, cy_dh, r_in, r_out = -0.55, 0.45, 0.15, 0.30
    r = sympy.sqrt((x - cx_dh) ** 2 + (y - cy_dh) ** 2)
    d_dh = sym_smin(r - r_in, r_out - r)

    # Triangle: CCW vertices.
    v0 = (sympy.Float(0.05), sympy.Float(-0.65))
    v1 = (sympy.Float(0.55), sympy.Float(-0.35))
    v2 = (sympy.Float(-0.30), sympy.Float(-0.30))

    def half_plane(a, b):
        ex, ey = b[0] - a[0], b[1] - a[1]
        nx, ny = -ey, ex
        nl = sympy.sqrt(nx * nx + ny * ny)
        return ((x - a[0]) * nx + (y - a[1]) * ny) / nl

    d_tr = sym_smin3(
        half_plane(v0, v1), half_plane(v1, v2), half_plane(v2, v0))

    # ρ = 1 + AMP · Σ sech²(d / EPS). cosh-based, C-printable.
    rho = sympy.Integer(1)
    for d in (d_sq, d_dh, d_tr):
        rho = rho + AMP * sym_sech2(d / EPS)
    return rho


def _extract_disp(verbose_out: str):
    rows = []
    for line in verbose_out.splitlines():
        mch = re.search(
            r"outer\s+(\d+)/\d+:.*max\|Δx\|=([0-9.e+-]+)", line)
        if mch:
            rows.append((int(mch.group(1)), float(mch.group(2))))
    return rows


def _run_mover(label, metric, n_outer, relax, refresh):
    m, T = build_mesh_with_field()
    if isinstance(metric, str) and metric == "lagrangian":
        rho = uw.meshing.metric_density_from_gradient(
            m, T, refinement=3.0,
            name=f"lag_{label}_{n_outer}_{relax}_{refresh}")
    else:
        rho = analytic_rho(m)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        uw.meshing.smooth_mesh_interior(
            m, metric=rho, method="anisotropic", strategy="med",
            method_kwargs=dict(
                n_outer=n_outer, relax=relax,
                metric_refresh_per_iter=refresh),
            verbose=True)
    return _extract_disp(buf.getvalue())


CASES = [
    # (label, ρ kind, refresh)
    ("Lagrangian ρ — refresh=False (baseline)", "lagrangian", False),
    ("Lagrangian ρ — refresh=True", "lagrangian", True),
    ("Analytic ρ   — refresh=False", "analytic", False),
    ("Analytic ρ   — refresh=True (Eulerian)", "analytic", True),
]

N_OUTER = 12
RELAXES = [1.0, 0.5, 0.2]


if __name__ == "__main__":
  for relax in RELAXES:
    print(f"\n{'=' * 66}")
    print(f"  n_outer={N_OUTER}, relax={relax}")
    print(f"{'=' * 66}")
    for label, kind, refresh in CASES:
        rows = _run_mover(label, kind, N_OUTER, relax, refresh)
        traj = " ".join(f"{d:.2e}" for _, d in rows)
        print(f"\n  {label}")
        print(f"    {traj}")
  # end if __name__
