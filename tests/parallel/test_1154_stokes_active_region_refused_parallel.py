"""Stokes.set_active_region refuses instead of ending in a segfault (#1154).

Configuring a region discrete system made the first residual evaluation segfault
inside PETSc (inserting the essential boundary values), and the region value was
never applied. The method now raises NotImplementedError on every rank, naming
mesh.extract_region, the supported route; the same solver without it solves.

    python scripts/mpi_supervisor.py --silence 300 -- mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1154_stokes_active_region_refused_parallel.py
"""
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(300)]


def test_set_active_region_is_refused_on_every_rank_and_the_solve_still_runs():
    mesh = uw.meshing.BoxInternalBoundary(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), zintCoord=0.5,
        cellSize=0.2, simplex=True, qdegree=3)
    v = uw.discretisation.MeshVariable("V1154", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P1154", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = sympy.Matrix([[0.0, sympy.cos(sympy.pi * mesh.X[0])]])
    for wall in ("Top", "Bottom", "Left", "Right"):
        stokes.add_essential_bc((0.0, 0.0), wall)

    message = ""
    try:
        stokes.set_active_region("Outer", mesh.regions.Outer.value)
    except NotImplementedError as error:
        message = str(error)
    messages = uw.mpi.comm.allgather(message)
    assert len(set(messages)) == 1 and "extract_region" in messages[0], messages

    stokes.solve()
    assert stokes.snes.getConvergedReason() > 0
