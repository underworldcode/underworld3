"""A cold rotated free-slip solve whose first linear solve fails raises (#907).

The rotated route runs its own Krylov loop and used to warn and go on when the
outer linear solve failed. With a preconditioner that fails at iteration 0
(reason -11) on a cold start, it returned with the velocity left at zero, and a
caller that did not read ``_rotated_freeslip_info["converged"]`` took the zeros
for a solution. Found with a singular multigrid coarse level under plain LU
(#901).

The failure is forced by a coarse-level python PC that reports a failure at
every application, which fails the same way on any build. Contracts:

- a cold start that fails its first step raises, naming the reason, after the
  result is recorded: a caller that catches it reads a failed solve, and the
  fields are not written;
- ``solver.tolerate_linear_failure = True`` warns and continues, as before: the
  fields hold the zero guess;
- a warm start that fails its first step warns and keeps the caller's state, as
  does a failure after an accepted step (test_0022's fixture fails at round-off
  on its sixth step).
"""
import numpy as np
import pytest
import sympy
from petsc4py import PETSc

import underworld3 as uw
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


class _FailingCoarse:
    """A coarse-level PC: redundant LU, or a failure at every application."""

    def __init__(self):
        self.fail = True
        self.ksp = None

    def setUp(self, pc):
        A = pc.getOperators()[0]
        if self.ksp is None:
            self.ksp = PETSc.KSP().create(comm=A.getComm())
            self.ksp.setType("preonly")
            self.ksp.getPC().setType("redundant")
            self.ksp.getPC().setFactorShift(PETSc.Mat.FactorShiftType.NONZERO)
        self.ksp.setOperators(A)
        self.ksp.setUp()

    def apply(self, pc, b, x):
        if self.fail:
            x.set(0.0)
            pc.setFailedReason(PETSc.PC.FailedReason.SUBPC_ERROR)
        else:
            self.ksp.solve(b, x)


def _rotated_annulus(monkeypatch, name, coarse):
    configure = custom_mg._configure_pcmg

    def configure_with(pc, Ps, **kw):
        out = configure(pc, Ps, **kw)
        PETSc.Options()[(pc.getOptionsPrefix() or "") + "mg_coarse_pc_type"] = "python"
        coarse_pc = pc.getMGCoarseSolve().getPC()
        coarse_pc.setType("python")
        coarse_pc.setPythonContext(coarse)
        return out

    monkeypatch.setattr(custom_mg, "_configure_pcmg", configure_with)
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
    return v, stokes


def test_a_failed_cold_rotated_solve_raises_after_recording(monkeypatch):
    v, stokes = _rotated_annulus(monkeypatch, "1115a", _FailingCoarse())
    v.array[...] = 1.0                     # a value the solve must not overwrite
    with pytest.raises(RuntimeError, match="reason -11"):
        stokes.solve(zero_init_guess=True)
    info = stokes._rotated_freeslip_info
    assert not info["converged"] and int(info["ksp_reason"]) == -11, info
    assert stokes.snes.getConvergedReason() < 0
    assert np.all(np.asarray(v.array) == 1.0)


def test_a_failed_cold_rotated_solve_is_tolerated_when_asked(monkeypatch):
    v, stokes = _rotated_annulus(monkeypatch, "1115b", _FailingCoarse())
    stokes.tolerate_linear_failure = True
    stokes.solve(zero_init_guess=True)
    info = stokes._rotated_freeslip_info
    assert not info["converged"] and int(info["ksp_reason"]) == -11, info
    assert np.all(np.asarray(v.array) == 0.0)  # the zero guess, as before


def test_a_failed_warm_rotated_solve_keeps_the_state(monkeypatch):
    coarse = _FailingCoarse()
    coarse.fail = False
    v, stokes = _rotated_annulus(monkeypatch, "1115c", coarse)
    stokes.solve(zero_init_guess=True)
    assert stokes._rotated_freeslip_info["converged"]
    solved = np.array(v.array, copy=True)
    # a changed problem, so the warm start is not already its solution
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 2.0
    coarse.fail = True
    stokes.solve(zero_init_guess=False)        # warns, does not raise
    assert not stokes._rotated_freeslip_info["converged"]
    # the write-back re-applies the rotation gauge: 3e-10 relative, measured
    assert np.allclose(np.asarray(v.array), solved, rtol=0.0,
                       atol=1.0e-8 * np.abs(solved).max())
