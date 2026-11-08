"""A Stokes adjoint runs the velocity block's custom-P hierarchy, as the forward solve does (#1016).

The forward solve installs the hierarchy on the velocity block just before its SNES
solve (``custom_mg.auto_inject_custom_mg``). ``adjoint_solve`` switches the tangent and
solves with the same KSP, and the switch resets the preconditioner: the fieldsplit
rebuilt the velocity block as a one-level multigrid, with no coarse correction and no
FAC smoother. On a split fault box the adjoint took 23 velocity iterations at np 1 and
29 at np 2, against the forward's 4; a forward solve after it took 4 again, so only
the adjoint lost the hierarchy (hyperion, development ba3fc23e). The adjoint still
converged, so the cost is iterations, and they grow with the resolution where the
FAC patch matters.

The injection goes in before the adjoint assembles its tangent at the forward state.
The install assembles at a zero vector to reach the fieldsplit; done after the state's
assembly, it left a nonlinear adjoint solving with the tangent at u = 0, wrong by 12
to 75 % in the velocity and reported converged (hyperion's review of the first fix).
The nonlinear cases below check the multiplier against a dense solve of K^T mu = b
with K the consistent tangent at the forward state.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _velocity_pc(stokes):
    return stokes.snes.getKSP().getPC().getFieldSplitSubKSP()[0].getPC()


def _velocity_its(stokes):
    return stokes.snes.getKSP().getPC().getFieldSplitSubKSP()[0].getIterationNumber()


def test_the_adjoint_keeps_the_velocity_hierarchy_and_its_fac_smoother():
    base = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=1 / 8, regular=False, qdegree=2,
                                             refinement=1)
    levels = base._coarse_level_meshes()[:-1] + [base]
    mesh = base.add_fault(("Flt", np.array([[0.3, 0.45], [0.7, 0.55]])))
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("V1145", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P1145", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_dirichlet_bc((y - sympy.Rational(1, 2), 0), wall)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-6
    custom_mg.set_custom_fmg(stokes, levels, field_id=0)

    stokes.solve()
    forward = _velocity_pc(stokes)
    nlev = forward.getMGLevels()
    assert forward.getType() == "mg" and nlev == 3, (forward.getType(), nlev)
    fac_level = nlev - 1
    assert forward.getMGSmoother(fac_level).getPC().getType() == "composite"
    forward_its = _velocity_its(stokes)

    b = stokes.snes.getJacobian()[0].createVecLeft()
    b.setRandom(_seeded_random())
    _mu, reason = stokes.adjoint_solve(b.getArray())
    assert reason > 0, reason
    adjoint = _velocity_pc(stokes)
    assert adjoint.getType() == "mg" and adjoint.getMGLevels() == nlev, (
        "the adjoint's velocity block is not the forward's hierarchy",
        adjoint.getType(), adjoint.getMGLevels() if adjoint.getType() == "mg" else None)
    assert adjoint.getMGSmoother(fac_level).getPC().getType() == "composite", (
        "the adjoint's finest level lost its FAC smoother")
    # one-level, it took 23 against the forward's 4
    assert _velocity_its(stokes) <= forward_its + 2, (_velocity_its(stokes), forward_its)


def _seeded_random():
    """A seeded random context, so the right-hand side is the same on every run."""
    from petsc4py import PETSc
    rnd = PETSc.Random().create()
    rnd.setFromOptions()
    rnd.setSeed(1145)
    return rnd


def _nonlinear_fault_box(name, law, matrix_route, picard):
    base = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=1 / 8, regular=False, qdegree=2,
                                             refinement=1)
    levels = base._coarse_level_meshes()[:-1] + [base]
    mesh = base.add_fault(("Flt", np.array([[0.3, 0.45], [0.7, 0.55]])))
    x, y = mesh.X
    v = uw.discretisation.MeshVariable(f"V{name}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P{name}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    edot = mesh.vector.strain_tensor(v.sym)
    eII2 = sympy.Rational(1, 2) * (edot[0, 0] ** 2 + edot[1, 1] ** 2) + edot[0, 1] ** 2
    stokes.constitutive_model.Parameters.shear_viscosity_0 = {
        "powerlaw": (sympy.Float(1e-2) + eII2) ** sympy.Rational(-1, 3),
        # a velocity-dependent viscosity: the tangent's velocity block is not symmetric
        "eta_vx": 1 + 2 * v.sym[0] ** 2}[law]
    walls = ("Bottom", "Top", "Left", "Right")
    for wall in walls[:-1] if matrix_route else walls:
        stokes.add_dirichlet_bc((y - sympy.Rational(1, 2), 0), wall)
    if matrix_route:
        # a natural condition takes the adjoint off the transposed kernels
        stokes.add_natural_bc((0.0, -0.1), "Right")
    stokes.petsc_use_pressure_nullspace = not matrix_route
    stokes.tolerance = 1.0e-9
    if picard:
        stokes.consistent_jacobian = False
    custom_mg.set_custom_fmg(stokes, levels, field_id=0)
    return stokes


def _dense_copy(M):
    """M's entries as a numpy array. Mat.convert with no target converts IN PLACE,
    which turned the solver's own Jacobian dense; convert a copy instead."""
    C = M.copy()
    C.convert("dense")
    A = C.getDenseArray().copy()
    C.destroy()
    return A


def _dense_adjoint_error(stokes):
    """The adjoint's multiplier against a dense solve of K^T mu = b, K the consistent
    tangent at the forward state; the pressure constant removed where it is free."""
    gvec = stokes.dm.getGlobalVec()
    gvec.set(0.0)
    stokes._gather_fields_to_global(gvec)
    tangent = stokes._consistent_tangent_for_adjoint()
    J, P = stokes.snes.getJacobian()[:2]
    alpha = stokes._newton_alpha_for_adjoint()
    stokes.snes.computeJacobian(gvec, J, P)
    stokes._restore_newton_alpha(alpha)
    K = _dense_copy(J)
    stokes._restore_tangent(tangent)
    stokes.dm.restoreGlobalVec(gvec)
    pressure = np.asarray(stokes._subdict["pressure"][0].getIndices())
    b = np.random.default_rng(1145).standard_normal(K.shape[0])
    if stokes.petsc_use_pressure_nullspace:
        b[pressure] -= b[pressure].mean()
    reference = np.linalg.lstsq(K.T, b, rcond=None)[0]
    mu, reason = stokes.adjoint_solve(b.copy())

    def gauged(z):
        z = z.copy()
        if stokes.petsc_use_pressure_nullspace:
            z[pressure] -= z[pressure].mean()
        return z

    error = np.linalg.norm(gauged(mu) - gauged(reference)) / np.linalg.norm(gauged(reference))
    return error, reason


# The dense reference indexes the global pressure numbering into one rank's rows.
@pytest.mark.skipif(uw.mpi.size > 1, reason="the dense reference is serial")
@pytest.mark.parametrize("law,matrix_route,picard", [
    ("powerlaw", False, False), ("powerlaw", False, True), ("eta_vx", True, False),
    ("powerlaw", True, True)],
    ids=["powerlaw_newton", "powerlaw_picard", "eta_vx_matrix_route",
         "powerlaw_picard_matrix_route"])
def test_a_nonlinear_adjoint_is_the_transpose_of_the_tangent_at_the_state(
        law, matrix_route, picard):
    stokes = _nonlinear_fault_box(f"1145{law[0]}{int(matrix_route)}{int(picard)}",
                                  law, matrix_route, picard)
    stokes.solve()
    assert stokes.snes.getConvergedReason() > 0
    assert stokes._adjoint_by_kernels() is not matrix_route
    forward_its = _velocity_its(stokes)
    error, reason = _dense_adjoint_error(stokes)
    assert reason > 0, reason
    # 1e-10 measured; with the tangent left at u = 0 it was 2e-1 to 1.6
    assert error < 1.0e-6, error
    assert _velocity_pc(stokes).getMGLevels() == 3
    # on the matrix route the velocity block took J's values, not J^T's (#1021):
    # 9 to 11 iterations against 5 to 6, and after a Picard forward the cap of 200
    assert _velocity_its(stokes) <= forward_its + 2, (_velocity_its(stokes), forward_its)
