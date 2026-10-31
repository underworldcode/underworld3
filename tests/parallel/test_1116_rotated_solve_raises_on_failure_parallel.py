"""A failed cold rotated free-slip solve raises on every rank (#907).

test_1115's forced failure at np >= 2, reported on RANK 0 ONLY: PETSc reduces a
PC's failed reason over the communicator, so the outer KSP's reason is global,
and every rank must reach the same verdict and raise together. A rank that
raised alone would leave the others in the next collective. The thread timeout
fires on a rank waiting in Python (the allgather below), and the MPI launcher
then ends the others.

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1116_rotated_solve_raises_on_failure_parallel.py
"""
import pytest
import sympy
from petsc4py import PETSc

import underworld3 as uw
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(300, method="thread")]


class _FailingCoarse:
    """A coarse-level PC that reports a failure at every application, on rank 0."""

    def setUp(self, pc):
        pass

    def apply(self, pc, b, x):
        x.set(0.0)
        if uw.mpi.rank == 0:
            pc.setFailedReason(PETSc.PC.FailedReason.SUBPC_ERROR)


def test_a_failed_rotated_solve_raises_on_every_rank(monkeypatch):
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
    v = uw.discretisation.MeshVariable("V1116", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P1116", mesh, 1, degree=1)
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

    raised = 0
    try:
        stokes.solve(zero_init_guess=True)
    except RuntimeError as exc:
        raised = int("reason -11" in str(exc))
    assert uw.mpi.comm.allgather(raised) == [1] * uw.mpi.size
