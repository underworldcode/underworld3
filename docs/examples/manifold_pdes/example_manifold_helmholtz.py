"""Manifold-mesh scalar PDE demonstrations on a SphericalManifold.

Three demonstrations of scalar PDE solves on a 2-manifold sphere
mesh (``dim=2, cdim=3``):

  (A) **Spherical-harmonic spectrum recovery** (steady Helmholtz).
      Solve :math:`(-\\Delta_S + I) T = Y_{lm}` for several ``(l, m)``
      and verify ``T = Y_{lm} / (l(l+1) + 1)`` to FE accuracy.

  (B) **h-convergence sweep**. Same problem (Y_10), refine cellSize
      from 0.4 → 0.05, fit the convergence rate.

  (C) **Closed-manifold Poisson with nullspace**. Solve
      :math:`-\\Delta_S T = z` on the closed unit sphere. The
      Laplace–Beltrami operator has a constant nullspace; PETSc
      handles it via ``petsc_use_constant_nullspace``. Recovers
      ``T = z/2 + const`` to FE accuracy.

A pyvista figure is rendered at the end showing T on the sphere
surface for the Y_10 Poisson case, with a few iso-contour lines.

Run:
    pixi run -e amr-dev python -u \\
        docs/examples/manifold_pdes/example_manifold_helmholtz.py

Optional headless rendering:
    PYVISTA_OFF_SCREEN=true pixi run -e amr-dev python -u \\
        docs/examples/manifold_pdes/example_manifold_helmholtz.py
"""

from __future__ import annotations

import os

import numpy as np
import sympy

import underworld3 as uw
from underworld3.systems import Poisson


# ---------------------------------------------------------------------------
# Spherical-harmonic helpers (real, orthonormal on the unit sphere)
# ---------------------------------------------------------------------------

def real_spherical_harmonic(l: int, m: int, x, y, z):
    r"""Real, unnormalised spherical-harmonic basis as a sympy expression.

    The 2-manifold sphere mesh stores Cartesian ``(x, y, z)`` coordinates,
    so we express :math:`Y_{lm}(\theta, \phi)` directly as polynomials
    in those — the natural form for this prototype (no charts, no
    pole-singularity worries). Each function is an eigenfunction of the
    Laplace–Beltrami operator on the unit sphere with eigenvalue
    :math:`-l(l+1)`:

    .. math::
        \Delta_S Y_{lm} = -l(l+1) Y_{lm}

    Conventions used here: each function is a spherical harmonic up
    to a normalisation constant. The L2 error against the analytic
    answer is computed relative to the dynamic range of the actual
    field on the mesh, so the absolute normalisation doesn't matter
    for the convergence assessment.

    Supported (l, m):
      (1, 0): z
      (1, 1): x
      (2, 0): 3 z^2 - 1
      (2, 1): x z
      (2, 2): x^2 - y^2
      (3, 0): z (5 z^2 - 3)
      (3, 2): z (x^2 - y^2)
    """
    if (l, m) == (1, 0):
        return z
    if (l, m) == (1, 1):
        return x
    if (l, m) == (2, 0):
        return 3 * z**2 - 1
    if (l, m) == (2, 1):
        return x * z
    if (l, m) == (2, 2):
        return x**2 - y**2
    if (l, m) == (3, 0):
        return z * (5 * z**2 - 3)
    if (l, m) == (3, 2):
        return z * (x**2 - y**2)
    raise ValueError(f"unsupported (l, m) = ({l}, {m})")


def _eval_at_dofs(expr_fn, mesh, dof_coords):
    """Evaluate a sympy expression at given physical coords via mesh.X.

    Substitutes ``mesh.X`` symbols with the coordinate columns so the
    analytic spherical harmonic can be compared element-wise against
    a MeshVariable's data array.
    """
    x_sym, y_sym, z_sym = mesh.X
    expr = expr_fn(x_sym, y_sym, z_sym)
    f = sympy.lambdify((x_sym, y_sym, z_sym), expr, modules="numpy")
    return f(dof_coords[:, 0], dof_coords[:, 1], dof_coords[:, 2])


# ---------------------------------------------------------------------------
# (A) Spherical-harmonic spectrum recovery
# ---------------------------------------------------------------------------

def helmholtz_recovery(sphere, mode, alpha=1.0, degree=2):
    r"""Solve :math:`(-\Delta_S + \alpha) T = Y_{lm}` and return the L2
    error against the analytic answer ``T = Y_{lm} / (l(l+1) + alpha)``.

    Non-singular for ``alpha > 0`` (the operator's constant kernel is
    broken by the reaction term), so no nullspace handling is needed.
    """
    l, m = mode
    T = uw.discretisation.MeshVariable(
        f"T_l{l}m{m}", sphere, num_components=1, degree=degree,
    )

    poisson = Poisson(sphere, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0

    x_sym, y_sym, z_sym = sphere.X
    Y_lm = real_spherical_harmonic(l, m, x_sym, y_sym, z_sym)
    poisson.f = Y_lm - alpha * T.sym[0]   # residual: -ΔT - Y_lm + αT = 0
    poisson.solve()

    eig = l * (l + 1)
    factor = eig + alpha
    coords = np.asarray(T.coords)
    expected = _eval_at_dofs(
        lambda x, y, z: real_spherical_harmonic(l, m, x, y, z) / factor,
        sphere, coords,
    )
    err = T.data[:, 0] - expected
    l2 = float(np.sqrt((err ** 2).mean()))
    scale = float(np.abs(expected).max())
    return l2, scale, T


def demo_A(sphere):
    print("\n" + "=" * 72)
    print("(A) Spherical-harmonic spectrum recovery on the unit sphere")
    print("    -Δ_S T + T = Y_lm   →   T = Y_lm / (l(l+1) + 1)")
    print("=" * 72)
    modes = [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (3, 2)]
    print(f"\n{'mode':>10} {'L2 err':>12} {'rel L2':>12} {'eigenvalue':>14}")
    print("-" * 54)
    for mode in modes:
        l2, scale, _ = helmholtz_recovery(sphere, mode)
        rel = l2 / scale if scale > 0 else l2
        eig = mode[0] * (mode[0] + 1)
        print(f"  ({mode[0]},{mode[1]}){'':<3} {l2:>12.3e} {rel:>12.3e} {eig:>14.1f}")


# ---------------------------------------------------------------------------
# (B) h-convergence sweep
# ---------------------------------------------------------------------------

def demo_B():
    print("\n" + "=" * 72)
    print("(B) h-convergence sweep — (-Δ_S + I) T = Y_10  →  T = z/2")
    print("=" * 72)
    print(f"\n{'cellSize':>10} {'cells':>8} {'L2 err':>12} {'rel L2':>12} {'order':>8}")
    print("-" * 56)
    prev_err = None
    prev_h = None
    for cs in (0.4, 0.2, 0.1, 0.05):
        sphere = uw.meshing.SphericalManifold(radius=1.0, cellSize=cs)
        l2, scale, _ = helmholtz_recovery(sphere, (1, 0))
        rel = l2 / scale if scale > 0 else l2
        cS, cE = sphere.dm.getHeightStratum(0)
        order = (
            np.log(prev_err / l2) / np.log(prev_h / cs) if prev_err is not None else float("nan")
        )
        order_str = f"{order:>8.2f}" if not np.isnan(order) else "  --   "
        print(f"  {cs:>8} {cE-cS:>8} {l2:>12.3e} {rel:>12.3e} {order_str}")
        prev_err = l2
        prev_h = cs


# ---------------------------------------------------------------------------
# (C) Closed Poisson with nullspace
# ---------------------------------------------------------------------------

def demo_C(sphere):
    print("\n" + "=" * 72)
    print("(C) Closed-manifold Poisson with constant-nullspace handling")
    print("    -Δ_S T = z   →   T = z/2 + const")
    print("=" * 72)
    T = uw.discretisation.MeshVariable("T_C", sphere, 1, degree=2)
    poisson = Poisson(sphere, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.petsc_use_constant_nullspace = True
    x_sym, y_sym, z_sym = sphere.X
    poisson.f = z_sym
    poisson.solve()

    coords = np.asarray(T.coords)
    expected = coords[:, 2] / 2.0
    diff = T.data[:, 0] - expected
    centred = diff - diff.mean()
    print(f"\n  T range:         [{T.data.min():.4f}, {T.data.max():.4f}]")
    print(f"  expected range:  [{expected.min():.4f}, {expected.max():.4f}]")
    print(f"  L_inf(T - z/2 - <diff>): {np.abs(centred).max():.3e}")
    print(f"  L_2  / |z/2|_max:        {np.sqrt((centred**2).mean()) / 0.5:.3e}")
    return T


# ---------------------------------------------------------------------------
# (D) pyvista visualisation
# ---------------------------------------------------------------------------

def render_pyvista(sphere, T_C, out_path=None):
    print("\n" + "=" * 72)
    print("(D) pyvista figure")
    print("=" * 72)
    try:
        import pyvista as pv
        import underworld3.visualisation as vis
    except ImportError:
        print("  pyvista not available — skipping render.")
        return

    pvmesh = vis.mesh_to_pv_mesh(sphere)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, T_C.sym)

    plotter = pv.Plotter(window_size=(900, 700), off_screen=True)
    plotter.set_background("white")
    plotter.add_mesh(
        pvmesh,
        scalars="T",
        cmap="coolwarm",
        show_edges=True,
        edge_color=(0.3, 0.3, 0.3),
        line_width=0.4,
        smooth_shading=False,
        scalar_bar_args={"title": "T = z/2", "color": "black"},
    )
    plotter.add_axes()
    plotter.camera_position = "yz"
    plotter.camera.azimuth = 30
    plotter.camera.elevation = 20

    if out_path is None:
        out_path = os.path.join(os.path.dirname(__file__), "_manifold_poisson.png")
    plotter.screenshot(out_path)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sphere_med = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.15)
    demo_A(sphere_med)
    demo_B()
    T_C = demo_C(sphere_med)
    render_pyvista(sphere_med, T_C)
    print("\n" + "=" * 72)
    print("Manifold PDE demonstration complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
