"""The consistent tangent must be finite at a state of rest.

The regression (measured, issue #507): differentiating an unwrapped
strain-rate invariant produces half-integer powers of
(grad v : grad v) whose value or derivative is 0/0 at v = 0, so EVERY
consistent-tangent assembly at a cold start filled the operator with
NaN — surfacing as GAMG's "Computed maximum singular value as zero"
(error 77) on the standard path, the rotated free-slip path, and the
split-node fault path alike. The alpha-blended continuation kernel
inherited it even in its Picard phase (IEEE 0*NaN = NaN). The guard in
``_jacobian_unwrap`` regularised the half-integer-power family (+1e-36
under the root), leaving the residual and the frozen (Picard) tangent
bit-identical. Since #841 it covers every fractional power: SymPy merges a
power of the bare invariant into one that is not a half-integer.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _vp_box(tag, tangent):
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.1)
    x, y = mesh.X
    v = uw.discretisation.MeshVariable(f"V{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P{tag}", mesh, 1, degree=0,
                                       continuous=False)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    # 1e4 never yields: the problem is secretly linear, which is exactly
    # the corner the yield-homotopy campaigns never visited
    stokes.constitutive_model.Parameters.yield_stress = 1.0e4
    stokes.consistent_jacobian = tangent
    stokes.bodyforce = [0.0, 0.0]
    for wall in ("Left", "Right", "Top", "Bottom"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0), wall)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1e-6
    return stokes


@pytest.mark.parametrize("tangent", [True, "continuation"])
def test_cold_consistent_tangent_is_finite_and_solves(tangent):
    stokes = _vp_box("a" if tangent is True else "b", tangent)
    stokes.solve(verbose=False)          # cold start — used to raise 77

    # and the assembled Jacobian at the rest state is finite
    snes = stokes.snes
    J = snes.getJacobian()[0].copy()
    U = stokes.dm.getGlobalVec()
    U.set(0.0)
    snes.computeJacobian(U, J)
    assert np.isfinite(J.norm()), "cold-state Jacobian contains NaN/Inf"


def _bounded_thinning_box(route, tangent):
    """A bounded shear-thinning law written on the bare strain-rate invariant,
    eta = 1 / (1 + edot**(1/3)). The residual is finite at rest. SymPy merges
    sqrt(g)**(1/3) into g**(1/6), a power the half-integer guard did not cover, so
    the Newton tangent was 0 * Inf = NaN at rest, on both JIT routes; under the
    default tangent the cold start's Picard step runs the blended kernel at
    alpha = 0, and 0 * NaN is NaN there too (#841)."""
    import sympy

    tag = f"{route[0]}{'n' if tangent is True else 'p'}"
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.25)
    v = uw.discretisation.MeshVariable(f"V1067{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P1067{tag}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    edot = stokes.Unknowns.Einv2
    stokes.constitutive_model.Parameters.shear_viscosity_0 = uw.expression(
        rf"\eta_{{1067{tag}}}", 1 / (1 + edot ** sympy.Rational(1, 3)),
        "bounded shear thinning")
    stokes.add_essential_bc((1.0, 0.0), "Top")
    stokes.add_essential_bc((0.0, 0.0), "Bottom")
    stokes.petsc_options.delValue("ksp_monitor")
    stokes.tolerance = 1.0e-8
    stokes.jit_route = route
    stokes.consistent_jacobian = tangent
    return stokes, v


@pytest.mark.parametrize("route", ["graph", "expanded"])
def test_a_fractional_power_of_the_invariant_solves_cold_under_newton(route):
    """Newton from a cold start reaches the solution the frozen tangent reaches, and
    its Jacobian at rest is finite."""
    frozen, v_frozen = _bounded_thinning_box(route, False)
    frozen.solve()
    assert frozen.snes.getConvergedReason() > 0

    newton, v_newton = _bounded_thinning_box(route, True)
    newton.solve()                       # cold: raised error 77 before the guard
    assert newton.snes.getConvergedReason() > 0

    a, b = np.asarray(v_newton.data), np.asarray(v_frozen.data)
    assert np.abs(a - b).max() <= 1.0e-6 * np.abs(b).max()

    J = newton.snes.getJacobian()[0].copy()
    U = newton.dm.getGlobalVec()
    U.set(0.0)
    newton.snes.computeJacobian(U, J)
    assert np.isfinite(J.norm()), "the Newton Jacobian at rest is not finite"
