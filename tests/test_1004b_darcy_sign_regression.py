#!/usr/bin/env python3
"""Sign-convention regression tests for the Darcy solvers (UW3 issue #214).

Two distinct sign issues are locked here:

1. **Physical velocity sign.** The Darcy velocity is ``v = -kappa*(grad(h) - s)``
   (flow runs *down* the head gradient), i.e. minus the assembly flux
   ``F = kappa*(grad(h) - s)``. ``SNES_TransientDarcy`` previously projected
   ``+darcy_flux`` (the assembly flux) instead of ``-darcy_flux``, giving a
   velocity field with the *opposite* sign to the steady ``SNES_Darcy`` solver.

2. **Source-term / assembly sign for f != 0.** The steady solver must solve
   ``-div(kappa*(grad(h) - s)) = f``; a manufactured solution locks this so a
   future refactor of the flux sign cannot silently invert the source response
   (the existing Darcy tests all use ``f = 0`` and would not catch it).
"""
import numpy as np
import sympy
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _box(res=12):
    return uw.meshing.StructuredQuadBox(
        elementRes=(res, res), minCoords=(-1.0, -1.0), maxCoords=(0.0, 0.0), qdegree=2
    )


def _grad_h_integral_of_v_dot_gradh(mesh, p, v):
    """int_Omega v . grad(h) dV. For correct Darcy sign (s=0) this is
    int -kappa |grad h|^2 < 0 (flow down-gradient); a flipped velocity gives > 0."""
    x, y = mesh.X
    grad_h = sympy.Matrix([[p.sym[0].diff(x), p.sym[0].diff(y)]])
    return float(uw.maths.Integral(mesh, v.sym.dot(grad_h)).evaluate())


def _make_darcy(solver_cls, mesh, tag):
    p = uw.discretisation.MeshVariable(f"P_{tag}", mesh, 1, degree=2)
    v = uw.discretisation.MeshVariable(f"U_{tag}", mesh, mesh.dim, degree=1)
    darcy = solver_cls(mesh, p, v)
    darcy.constitutive_model = uw.constitutive_models.DarcyFlowModel
    darcy.constitutive_model.Parameters.permeability = 1.0
    darcy.constitutive_model.Parameters.s = sympy.Matrix([0.0, 0.0]).T
    darcy.f = 0.0
    # High head at the bottom, low at the top -> flow upward (down-gradient).
    darcy.add_dirichlet_bc([0.5], "Bottom")
    darcy.add_dirichlet_bc([0.0], "Top")
    return darcy, p, v


def test_steady_darcy_velocity_is_down_gradient():
    """Baseline: steady Darcy velocity flows down the head gradient (v . grad h < 0)."""
    mesh = _box()
    darcy, p, v = _make_darcy(uw.systems.SteadyStateDarcy, mesh, "s")
    darcy.solve()
    integ = _grad_h_integral_of_v_dot_gradh(mesh, p, v)
    assert integ < 0.0, f"steady Darcy velocity not down-gradient: int v.grad(h) = {integ:.3e}"


def test_transient_darcy_velocity_sign_matches_steady():
    """Regression #214: TransientDarcy projected velocity must have the SAME
    (physical, down-gradient) sign as the steady solver. Before the fix it
    projected +darcy_flux -> flipped sign -> int v.grad(h) > 0."""
    mesh = _box()
    darcy, p, v = _make_darcy(uw.systems.TransientDarcy, mesh, "t")
    darcy.storage = 1.0
    darcy.solve(timestep=1.0e6)  # large dt -> approaches steady
    integ = _grad_h_integral_of_v_dot_gradh(mesh, p, v)
    assert integ < 0.0, (
        f"TransientDarcy velocity sign is flipped (issue #214): int v.grad(h) = {integ:.3e} "
        "(should be < 0, matching steady)"
    )


def test_steady_darcy_source_term_sign():
    """Lock the f != 0 assembly sign with a manufactured solution.

    Solve  -div(grad h) = f  with  h_exact = sin(pi(x+1)) sin(pi(y+1))  (= 0 on
    the box boundary), so  f = -Laplacian(h_exact) = 2 pi^2 h_exact.  The
    recovered head must match h_exact; a flipped flux/source sign would return
    -h_exact (rel L2 ~ 2)."""
    mesh = _box(res=20)
    p = uw.discretisation.MeshVariable("P_mms", mesh, 1, degree=2)
    v = uw.discretisation.MeshVariable("U_mms", mesh, mesh.dim, degree=1)
    x, y = mesh.X
    h_exact = sympy.sin(sympy.pi * (x + 1)) * sympy.sin(sympy.pi * (y + 1))
    f_src = -(h_exact.diff(x, 2) + h_exact.diff(y, 2))  # 2 pi^2 h_exact

    darcy = uw.systems.SteadyStateDarcy(mesh, p, v)
    darcy.constitutive_model = uw.constitutive_models.DarcyFlowModel
    darcy.constitutive_model.Parameters.permeability = 1.0
    darcy.constitutive_model.Parameters.s = sympy.Matrix([0.0, 0.0]).T
    darcy.f = f_src
    for bnd in ["Top", "Bottom", "Left", "Right"]:
        darcy.add_dirichlet_bc([0.0], bnd)
    darcy.solve()

    err = float(uw.maths.Integral(mesh, (p.sym[0] - h_exact) ** 2).evaluate())
    nrm = float(uw.maths.Integral(mesh, h_exact ** 2).evaluate())
    rel_l2 = (err / nrm) ** 0.5
    assert rel_l2 < 0.05, (
        f"Darcy f!=0 manufactured-solution rel L2 error = {rel_l2:.3e} "
        "(source/assembly sign regression?)"
    )
