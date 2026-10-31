"""A rotated solve sets its multigrid coarse level up once (#901).

The rotated route sets up the velocity multigrid itself (``vel_pc.setUp()``),
then turns off inode SOR on the rotated coarse levels and re-assembles their
operators. That loop included level 0, the coarse direct solve. Re-assembling
its operator moved its state, so the coarse PC was set up a second time at
its first application inside ``ksp.solve``. A coarse solve that factors at
setup paid that factorisation twice per solve: a dense SVD does, so half of
the SVD time quoted in #901 was this second factorisation.

Level 0 is not smoothed by SOR, and its direct solve factors its own gathered
copy, so it is left out of the loop. Contract: one linear rotated solve sets
the coarse PC up exactly once. The test observes it by putting a counting
python PC, which delegates to redundant LU, on the coarse level.
"""
import pytest
import sympy
from petsc4py import PETSc

import underworld3 as uw
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


class _CountingCoarse:
    """Redundant LU on the coarse level, counting its setups."""

    def __init__(self):
        self.setups = 0
        self.ksp = None

    def setUp(self, pc):
        self.setups += 1
        A = pc.getOperators()[0]
        if self.ksp is None:
            self.ksp = PETSc.KSP().create(comm=A.getComm())
            self.ksp.setType("preonly")
            self.ksp.getPC().setType("redundant")
            # the closed annulus's rotation makes this coarse operator nearly
            # singular; a shift only acts on an exact zero pivot
            self.ksp.getPC().setFactorShift(PETSc.Mat.FactorShiftType.NONZERO)
        self.ksp.setOperators(A)
        self.ksp.setUp()

    def apply(self, pc, b, x):
        self.ksp.solve(b, x)

    def destroy(self, pc):
        if self.ksp is not None:
            self.ksp.destroy()


def test_one_rotated_solve_sets_the_coarse_level_up_once(monkeypatch):
    counting = _CountingCoarse()
    configure = custom_mg._configure_pcmg

    def configure_counted(pc, Ps, **kw):
        out = configure(pc, Ps, **kw)
        # the bundle's coarse type is re-read from the options database at the
        # MG setup, so it must name the python PC too
        PETSc.Options()[(pc.getOptionsPrefix() or "") + "mg_coarse_pc_type"] = "python"
        coarse_pc = pc.getMGCoarseSolve().getPC()
        coarse_pc.setType("python")
        coarse_pc.setPythonContext(counting)
        return out

    monkeypatch.setattr(custom_mg, "_configure_pcmg", configure_counted)
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.4,
                              refinement=2, qdegree=3)
    v = uw.discretisation.MeshVariable("V1113", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P1113", mesh, 1, degree=1)
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
    stokes.solve(zero_init_guess=True)

    info = stokes._rotated_freeslip_info
    assert info["converged"] and info["velocity_pc"] == "custom-FMG", info
    assert info["nonlinear_iterations"] == 1, info
    assert getattr(stokes, "_custom_mg_rotated_levels", False)   # the path under test
    assert counting.setups == 1, f"the coarse level was set up {counting.setups} times"
