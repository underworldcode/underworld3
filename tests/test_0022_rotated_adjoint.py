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
    permutation an axis-aligned wall would give, and so the geometry is the one
    the free-slip adjoint is wanted for.
  * a POWER-LAW transversely isotropic viscosity under the CONSISTENT (Newton)
    tangent, so the tangent has no major symmetry and ``K ≠ Kᵀ``. A symmetric
    operator cannot tell a transpose from itself, and every isotropic Stokes
    case is symmetric — as is the frozen TI tangent, whose ``C`` keeps both
    minor and major symmetry. ``test_the_tangent_is_not_symmetric`` measures
    the asymmetry rather than trusting the setup to have produced it: without
    that guard, omitting the transpose entirely still passes (measured).

What this file does NOT check, so nobody reads it as covered: in 2-D with one
normal per node ``Q`` is EXACTLY symmetric — the frame ``numpy.linalg.svd``
returns for a single normal is the Householder reflection
``[[nx, ny], [ny, -nx]]``, and ``‖Q - Qᵀ‖`` measures zero on this mesh. So
rotating the dual the wrong way (``Qᵀ b`` where ``Q b`` is meant) is invisible
here; substituting it changes the gradient in the eighth digit. The direction is
right on the mathematics — ``Q`` is orthogonal, so ``μ̂ = Q μ`` and ``μ = Qᵀ μ̂``
— but it takes a 3-D boundary or a multi-normal corner, where the frame is no
longer a reflection, to make a test say so.

Serial. Under ``mpirun -n 2`` this file aborts about half the time in
``_jitextension`` with "JIT C-source hash differs across MPI ranks" — the
deliberate hard error for non-deterministic ``generate_c_source``. It is NOT an
adjoint failure: the abort lands in the fixture's FIRST FORWARD solve, before
any adjoint runs. Nor is it this file's expression — the pre-existing TI adjoint
test (``test_0021``) is stable at np=2, and building the fixture's expression
variants outside pytest never diverges (3/3). Fixing ``PYTHONHASHSEED`` does not
settle it. The CI parallel pass collects only ``tests/parallel/test_*.py``, so
this is not in the gate; the cause is a latent non-determinism in JIT source
generation — issue #752.
"""

import math

import numpy as np
import pytest
import sympy

from petsc4py import PETSc

import underworld3 as uw
from underworld3.adjoint import misfit_duals
from underworld3.utilities.rotated_bc import (_rotated_nullspace,
                                              _velocity_diag_scale)


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
    # The VELOCITY BLOCK, not the composite. UW3 assembles the velocity flux as
    # τ − p·I against +div u, so the operator is [[A, −Bᵀ], [B, 0]] and the
    # composite is structurally non-symmetric for EVERY rheology — measured
    # 2.5e-2 for constant isotropic viscosity, which would sail past any floor
    # set here and prove nothing. It is the A block whose symmetry decides
    # whether a transpose is detectable, and there it is 5.3e-17 isotropic,
    # 6.0e-17 for a CONSTANT TI viscosity (the frozen tangent keeps major
    # symmetry) and 5.7e-2 for the power-law TI below.
    vel_is = stokes._subdict["velocity"][0]
    A = K.createSubMatrix(vel_is, vel_is)
    dA = Kt.createSubMatrix(vel_is, vel_is)
    asymmetry = (dA.norm(PETSc.NormType.FROBENIUS)
                 / A.norm(PETSc.NormType.FROBENIUS))
    A.destroy()
    dA.destroy()

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
    # 1e-14, not 1e-12: the forward tolerance is 1e-11, so a bound of 1e-12 is
    # only one decade below a number that a DELETED `_zero_rows_local` would
    # leave at ~tolerance x ‖b̂‖ — which clears 1e-12 whenever that scale is
    # below 0.1, and the fixture does not measure it. What is genuinely left
    # here is round-off on the Q(Qᵀμ̂) round trip (measured 4.8e-18).
    assert rotated_gradient["leak"] < 1.0e-14, rotated_gradient["leak"]


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
    # The verdict warns about the null space unconditionally, because it is a
    # property of the problem the caller poses, not of this solve. This fixture
    # pins the inner boundary and HAS no null space (`_rotated_nullspace`
    # returns None) — which is exactly why its gradient is unambiguous.
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


@pytest.mark.level_2
@pytest.mark.tier_a
def test_the_null_space_serves_the_transposed_operator_too():
    """``_rotated_nullspace`` admits a mode by measuring ``‖Â·w‖``, and the
    adjoint attaches the result to ``Âᵀ`` as BOTH its null space and its
    transpose null space. Sound only if the modes are null from the left as well.

    The argument is that a rigid rotation has zero strain rate, so ``∫C:ε:ε``
    annihilates it read from either side whatever the symmetry of ``C``, and the
    constant-pressure mode couples only through an off-diagonal block that
    transposition moves but does not remove. On a tangent with no major symmetry
    that deserves measuring.

    Free slip on BOTH boundaries, because that is what admits a rigid rotation:
    the gradient fixture pins the inner boundary and has no null space at all."""
    mesh = uw.meshing.Annulus(radiusInner=R_I, radiusOuter=R_O,
                              cellSize=0.25, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x**2 + y**2)
    unit_r = sympy.Matrix([[x / r, y / r]])
    th = sympy.atan2(y, x)
    v = uw.discretisation.MeshVariable("v_nsp", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p_nsp", mesh, 1, degree=1)

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    edot = mesh.vector.strain_tensor(v.sym)
    eII = sympy.sqrt(sympy.Rational(1, 2) * (edot[0, 0] ** 2 + edot[1, 1] ** 2)
                     + edot[0, 1] ** 2)
    eta_0 = (sympy.Float(0.01) + eII) ** sympy.Rational(-1, 3)
    stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_0
    stokes.constitutive_model.Parameters.shear_viscosity_1 = ETA_1 * eta_0
    stokes.constitutive_model.Parameters.director = unit_r
    stokes.bodyforce = 1.0e2 * sympy.cos(3 * th) * (r - R_I) / (R_O - R_I) * unit_r
    stokes.add_rotated_freeslip_bc(0.0, "Lower")
    stokes.add_rotated_freeslip_bc(0.0, "Upper")
    stokes.petsc_use_pressure_nullspace = True
    stokes.consistent_jacobian = True
    stokes.tolerance = 1.0e-10
    stokes.solve(zero_init_guess=True)

    info = stokes._rotated_freeslip_info
    Q, Qt, rows = info["Q"], info["Qt"], info["normal_rows"]
    nsp = _rotated_nullspace(stokes, Q, rows)
    assert nsp is not None, "no null space here — the test would prove nothing"
    assert getattr(stokes, "_rotated_velocity_null_modes", 0) >= 1, (
        "no RIGID ROTATION was admitted; only the pressure mode is being "
        "measured, and that one is symmetric for a trivial reason")

    K = stokes.snes.getJacobian()[0]
    Ahat = K.ptap(Qt)
    Ahat.zeroRowsColumns(rows, diag=_velocity_diag_scale(Ahat, stokes))
    AhatT = Ahat.transpose(PETSc.Mat())
    try:
        for w in nsp.getVecs():
            out = w.duplicate()
            Ahat.mult(w, out)
            forward = out.norm() / w.norm()
            AhatT.mult(w, out)
            adjoint = out.norm() / w.norm()
            out.destroy()
            # Absolute, and then the identity: the same mode read from the two
            # sides must give the SAME residual, not merely a small one.
            assert adjoint < 1.0e-8, (forward, adjoint)
            assert abs(adjoint - forward) <= 0.01 * max(forward, 1.0e-16), (
                forward, adjoint)
    finally:
        Ahat.destroy()
        AhatT.destroy()
