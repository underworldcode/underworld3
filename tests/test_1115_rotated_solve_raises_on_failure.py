"""A rotated free-slip solve whose linear solve fails raises (#907).

The rotated route runs its own Krylov loop and used to warn and go on when the
outer linear solve failed. With a preconditioner that fails at iteration 0
(``KSP_DIVERGED_PC_FAILED``, reason -11) it returned with the velocity left at
zero, and a caller that did not read ``_rotated_freeslip_info["converged"]``
took the zeros for a solution. Found with a singular multigrid coarse level
under plain LU (#901).

The failure is forced here by a coarse-level python PC that reports a failure
at every application, which fails the same way on any build. It fails the
first Newton step, so the fields would hold the initial guess. Contract: the
solve raises, naming the KSP reason; with ``solver.tolerate_linear_failure =
True`` it warns and keeps that guess, as before. A hard failure after an
accepted step leaves the fields at that iterate, the line search rejects the
failed increment, and the loop's convergence warning stands (test_0022's
fixture fails at round-off on its sixth step). An iteration cap
(``KSP_DIVERGED_ITS``) only warns.
"""
import pytest
import sympy
from petsc4py import PETSc

import underworld3 as uw
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


class _FailingCoarse:
    """A coarse-level PC that reports a failure at every application."""

    def setUp(self, pc):
        pass

    def apply(self, pc, b, x):
        x.set(0.0)
        pc.setFailedReason(PETSc.PC.FailedReason.SUBPC_ERROR)


def _rotated_annulus(monkeypatch, name):
    configure = custom_mg._configure_pcmg

    def configure_failing(pc, Ps, **kw):
        out = configure(pc, Ps, **kw)
        PETSc.Options()[(pc.getOptionsPrefix() or "") + "mg_coarse_pc_type"] = "python"
        coarse_pc = pc.getMGCoarseSolve().getPC()
        coarse_pc.setType("python")
        coarse_pc.setPythonContext(_FailingCoarse())
        return out

    monkeypatch.setattr(custom_mg, "_configure_pcmg", configure_failing)
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.4,
                              refinement=2, qdegree=3)
    v = uw.discretisation.MeshVariable(f"V{name}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P{name}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    X = mesh.X
    r = sympy.sqrt(X[0] ** 2 + X[1] ** 2)
    th = sympy.atan2(X[1], X[0])
    stokes.bodyforce = (1.0e3 * sympy.cos(4 * th) * sympy.sin(sympy.pi * (r - 0.5) / 0.5)
                        * sympy.Matrix([X[0] / r, X[1] / r]))
    for wall in ("Lower", "Upper"):
        stokes.add_rotated_freeslip_bc(0, wall)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-6
    custom_mg.set_custom_fmg(stokes, mesh._coarse_level_meshes()[:-1], field_id=0)
    return stokes


def test_a_failed_rotated_solve_raises(monkeypatch):
    stokes = _rotated_annulus(monkeypatch, "1115a")
    with pytest.raises(RuntimeError, match="DIVERGED_PC_FAILED"):
        stokes.solve(zero_init_guess=True)


def test_a_failed_rotated_solve_is_tolerated_when_asked(monkeypatch):
    stokes = _rotated_annulus(monkeypatch, "1115b")
    stokes.tolerate_linear_failure = True
    stokes.solve(zero_init_guess=True)
    info = stokes._rotated_freeslip_info
    assert not info["converged"], info
    assert int(info["ksp_reason"]) == -11, info    # KSP_DIVERGED_PC_FAILED
