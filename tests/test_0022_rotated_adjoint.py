r"""The adjoint of a solve that ran on the ROTATED operator.

Rotated free-slip does not solve :math:`K \delta = b`. It rotates into the
per-node boundary frame, strikes out the wall-normal rows, and inverts
:math:`Q K Q^T` in its own Krylov loop. The adjoint is the transpose of THAT
operator, and the multiplier comes back with no wall-normal component — the
dual of a strong constraint is a strong homogeneous constraint on the same
degrees of freedom.

The check that counts is the gradient against a central finite difference. Two
things in the setup are there to stop a wrong answer hiding, and both are
guarded rather than asserted in a comment:

  * an ANNULUS, so ``Q`` is a genuine per-node frame rather than the signed
    permutation an axis-aligned wall would give. Rotating the dual the wrong
    way (``Qᵀ b`` where ``Q b`` is meant) then breaks the gradient.
  * a POWER-LAW transversely isotropic viscosity under the CONSISTENT (Newton)
    tangent, so the tangent has no major symmetry and ``K ≠ Kᵀ``. A symmetric
    operator cannot tell a transpose from itself, and every isotropic Stokes
    case is symmetric — as is the frozen TI tangent, whose ``C`` keeps both
    minor and major symmetry. ``test_the_tangent_is_not_symmetric`` measures
    the asymmetry rather than trusting the setup to have produced it: without
    that guard, omitting the transpose entirely still passes (measured).
"""

import math

import numpy as np
import pytest
import sympy

from petsc4py import PETSc

import underworld3 as uw
from underworld3.adjoint import misfit_duals


R_I, R_O = 0.5, 1.0
ETA_1 = 0.2                      # the parameter, at the point the gradient is taken
FD_STEP = 1.0e-4
# The asymmetry the transpose is supposed to matter for. A tangent this close to
# symmetric could not distinguish K from Kᵀ, so the fixture measures it and the
# gradient test is only meaningful above this floor.
MIN_ASYMMETRY = 1.0e-3


@pytest.fixture(scope="module")
def rotated_gradient():
    """One rotated free-slip solve, its adjoint gradient, and the central finite
    difference to check it against. Module-scoped: the finite difference re-solves
    at two shifted parameters, so the state is not reusable afterwards and every
    contract has to be read off the record this builds."""
    mesh = uw.meshing.Annulus(radiusInner=R_I, radiusOuter=R_O,
                              cellSize=0.15, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x**2 + y**2)
    unit_r = sympy.Matrix([[x / r, y / r]])
    th = sympy.atan2(y, x)

    v = uw.discretisation.MeshVariable("v_rot_adj", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p_rot_adj", mesh, 1, degree=1)
    v_obs = uw.discretisation.MeshVariable("v_obs_rot_adj", mesh, 2, degree=2)

    eta_1 = uw.expression(r"\eta_1", ETA_1, "weak-plane viscosity ratio")

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    # Power law, so the CONSISTENT tangent picks up the dependence of the
    # anisotropic C on the strain rate — that term is what destroys the major
    # symmetry the frozen TI tangent keeps.
    edot = mesh.vector.strain_tensor(v.sym)
    eII = sympy.sqrt(sympy.Rational(1, 2) * (edot[0, 0] ** 2 + edot[1, 1] ** 2)
                     + edot[0, 1] ** 2)
    eta_0 = (sympy.Float(0.01) + eII) ** sympy.Rational(-1, 3)

    stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_0
    stokes.constitutive_model.Parameters.shear_viscosity_1 = eta_1 * eta_0
    # radial director: the weak planes are the tangential ones, and the director
    # turns with the boundary, so the anisotropy and the rotation frame are both
    # functions of position rather than constants that could coincidentally align.
    stokes.constitutive_model.Parameters.director = unit_r
    stokes.bodyforce = 1.0e2 * sympy.cos(3 * th) * (r - R_I) / (R_O - R_I) * unit_r
    stokes.add_dirichlet_bc((0.0, 0.0), "Lower")
    stokes.add_rotated_freeslip_bc(0.0, "Upper")
    stokes.consistent_jacobian = True
    stokes.tolerance = 1.0e-11

    def solve_at(value):
        eta_1.sym = sympy.Float(value)
        stokes.solve(zero_init_guess=True)

    # An observation set from a DIFFERENT parameter, so the misfit is not
    # stationary at the point the gradient is taken and the finite difference has
    # something to measure.
    solve_at(0.05)
    v_obs.array[...] = np.asarray(v.array)

    misfit = sympy.Rational(1, 2) * ((v.sym[0] - v_obs.sym[0]) ** 2
                                     + (v.sym[1] - v_obs.sym[1]) ** 2)

    def J_at(value):
        solve_at(value)
        return float(uw.maths.Integral(mesh, misfit).evaluate())

    J_at(ETA_1)
    supported, reason = stokes.adjoint_support()

    # Is the tangent this gradient is taken through actually non-symmetric?
    # Measured on the assembled Jacobian at the converged state, the same one
    # the adjoint transposes. Without this the whole file can pass with the
    # transpose deleted.
    K = stokes.snes.getJacobian()[0]
    Kt = K.transpose(PETSc.Mat())
    Kt.axpy(-1.0, K, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    asymmetry = Kt.norm(PETSc.NormType.FROBENIUS) / K.norm(PETSc.NormType.FROBENIUS)
    Kt.destroy()

    dual = misfit_duals(misfit, [v])[v]
    dual.array[...] = -np.asarray(dual.array)
    mu = uw.discretisation.MeshVariable("mu_rot_adj", mesh, 2, degree=2)
    lam = uw.discretisation.MeshVariable("lam_rot_adj", mesh, 1, degree=1)
    mu_global, reason_ksp = stokes.adjoint_solve((dual, None), target=(mu, lam))
    adjoint = stokes.sensitivity(mu, eta_1)

    # The wall-normal leak of the MULTIPLIER, measured in the discrete frame the
    # constraint is actually written in — Q's per-node rows, not an analytic
    # normal. Those two differ by the facet/true-normal discrepancy, which is a
    # property of the mesh and has nothing to say about the adjoint.
    info = stokes._rotated_freeslip_info
    vec = stokes.dm.createGlobalVec()
    vec.array[:] = mu_global
    rotated = vec.duplicate()
    info["Q"].mult(vec, rotated)
    lo, hi = rotated.getOwnershipRange()
    owned = [g - lo for g in info["normal_rows"] if lo <= g < hi]
    leak = float(np.abs(np.asarray(rotated.array)[owned]).max()) if owned else 0.0
    leak = uw.mpi.comm.allreduce(leak, op=uw.MPI.MAX)
    vec.destroy()
    rotated.destroy()

    fd = (J_at(ETA_1 + FD_STEP) - J_at(ETA_1 - FD_STEP)) / (2 * FD_STEP)
    eta_1.sym = sympy.Float(ETA_1)

    return {"supported": supported, "reason": reason, "ksp_reason": reason_ksp,
            "adjoint": adjoint, "fd": fd, "leak": leak,
            "asymmetry": float(asymmetry)}


@pytest.mark.level_2
@pytest.mark.tier_a
def test_the_tangent_is_not_symmetric(rotated_gradient):
    """The guard on the guard. A symmetric K satisfies Kᵀ = K, so the gradient
    test below would pass with the transpose deleted — as it does when the TI
    viscosity is constant. Assert the asymmetry is there before believing the
    gradient says anything about a transpose. Measured 6.7e-2 for this setup;
    deleting the transpose then moves the gradient by 1.9%, against a 0.2%
    bound."""
    assert rotated_gradient["asymmetry"] > MIN_ASYMMETRY, (
        f"tangent is (near-)symmetric at {rotated_gradient['asymmetry']:.2e} — "
        f"the gradient test cannot detect a missing transpose; strengthen the "
        f"anisotropy or the power-law exponent in the fixture")


@pytest.mark.level_2
@pytest.mark.tier_a
def test_the_gradient_through_a_rotated_freeslip_solve_matches_finite_differences(
        rotated_gradient):
    """The contract. A transpose taken on K rather than on Q K Qᵀ, or a dual
    rotated the wrong way, changes this number; the finite difference does not
    care how the constraint was imposed."""
    adjoint, fd = rotated_gradient["adjoint"], rotated_gradient["fd"]
    assert adjoint != 0.0
    assert abs(fd / adjoint - 1) < 2.0e-3, (fd, adjoint)


@pytest.mark.level_2
@pytest.mark.tier_a
def test_the_multiplier_has_no_wall_normal_component(rotated_gradient):
    """Exact, not converged. ``zeroRowsColumns`` decouples the constrained rows
    of the adjoint operator exactly as it does the forward one, so the
    multiplier's wall-normal component is set rather than iterated towards —
    a Krylov tolerance must not appear in this number."""
    assert rotated_gradient["leak"] < 1.0e-12, rotated_gradient["leak"]


@pytest.mark.level_2
@pytest.mark.tier_a
def test_the_adjoint_ksp_converged(rotated_gradient):
    assert rotated_gradient["ksp_reason"] > 0, rotated_gradient["ksp_reason"]


@pytest.mark.level_2
@pytest.mark.tier_a
def test_the_verdict_says_the_operator_is_the_rotated_one(rotated_gradient):
    """A supported verdict that did not mention the rotation would be describing
    a different solve — the one that transposes K."""
    assert rotated_gradient["supported"] is True
    assert "rotated free-slip" in rotated_gradient["reason"]
    assert "null space" in rotated_gradient["reason"]


@pytest.mark.level_1
@pytest.mark.tier_a
def test_a_released_rotation_is_refused_rather_than_read():
    """``_rotated_freeslip_info`` outlives the workspace cache it SHARES ``Q``
    with — deliberately, because the reaction vector in it is still wanted after
    a reset. The rotation is not still wanted: reading it there is a
    use-after-free, not a wrong answer. It has to be refused by name."""
    mesh = uw.meshing.Annulus(radiusInner=R_I, radiusOuter=R_O,
                              cellSize=0.3, qdegree=3)
    v = uw.discretisation.MeshVariable("v_rel", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p_rel", mesh, 1, degree=1)
    dual = uw.discretisation.MeshVariable("d_rel", mesh, 2, degree=2)

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
    stokes.bodyforce = sympy.Matrix([[0.0, -1.0]])
    stokes.add_dirichlet_bc((0.0, 0.0), "Lower")
    stokes.add_rotated_freeslip_bc(0.0, "Upper")
    stokes.tolerance = 1.0e-9
    stokes.solve()
    dual.array[...] = 1.0

    _, reason = stokes.adjoint_solve((dual, None))
    assert reason > 0, reason

    stokes._reset_rotated_solver_cache()
    with pytest.raises(RuntimeError, match="released"):
        stokes.adjoint_solve((dual, None))
