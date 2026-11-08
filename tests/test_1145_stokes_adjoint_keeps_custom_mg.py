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
