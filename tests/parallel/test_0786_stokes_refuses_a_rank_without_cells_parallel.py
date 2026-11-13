"""A Stokes solve on a mesh with a rank that holds no cells is refused on every
rank (#962).

The Stokes solver registers a Jacobian preconditioner, so SNES assembles two
matrices. PETSc's ``DMPlexComputeJacobianByKey`` (3.25.3 and ``main``) then leaves
early on a rank without cells and skips the assembly of the Jacobian that the other
ranks make. Before the refusal the solve stopped in ``MPI_Allreduce`` on the ranks
without cells, or hung: a 14-cell box on 8 ranks, a region extracted from a mesh
whose other ranks hold none of it, and a refined annulus whose base left two of 8
ranks empty.

Power: both fixtures leave a rank without cells at every np >= 2 (the triangle at
np 2 is one cell; from np 3 up, and for the region at every count, a shell
partition does it), and each test asserts that premise. The refusal is one gather
of the cell counts, so what the tests show is that every rank raises the same
error, named by those counts, and is still in step for the gather that follows. A
regression stops in MPI or hangs, and the MPI supervisor's silence limit ends it;
the decision and the text are asserted on one rank in
``tests/test_0786_stokes_refuses_a_rank_without_cells.py``.

A scalar solve has one matrix and is not refused: on the same region it converges
to the linear field its walls prescribe.

    python scripts/mpi_supervisor.py --silence 300 -- mpirun -n 3 python -m pytest --with-mpi tests/parallel/test_0786_stokes_refuses_a_rank_without_cells_parallel.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import sympy

import underworld3 as uw

from empty_rank_fixtures import (ALL_BOUNDARIES_VALUE, cells_along_a_side_leaving_one_rank_empty,
                                 cells_per_rank, right_triangle_of_cells)

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.timeout(300)]

REGION_WALLS = ("Top", "Left", "Right", "Internal")


def _triangle_with_an_empty_rank():
    size = uw.mpi.size
    n = cells_along_a_side_leaving_one_rank_empty(size)
    dm = right_triangle_of_cells(n, max(size - 1, 1))
    # a mesh made from a bare DM has the name All_Boundaries and no such label
    dm.createLabel("All_Boundaries")
    dm.markBoundaryFaces("All_Boundaries", ALL_BOUNDARIES_VALUE)
    return uw.discretisation.Mesh(dm, qdegree=3), ("All_Boundaries",)


def _region_held_by_rank_0():
    """The outer slab of the #858 tests' parent, all of it on rank 0."""
    path = Path(__file__).resolve().parent / "test_1087_submesh_prolongation_ownership.py"
    spec = importlib.util.spec_from_file_location("_submesh_ownership", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parent = module._thin_slab_parent("single_rank_region")
    return parent.extract_region("Outer"), REGION_WALLS


def _stokes(mesh, walls):
    v = uw.discretisation.MeshVariable("V0786", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P0786", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, _y = mesh.X
    stokes.bodyforce = sympy.Matrix([[0.0, sympy.cos(sympy.pi * x)]])
    for wall in walls:
        stokes.add_essential_bc((0.0, 0.0), wall)
    stokes.petsc_use_pressure_nullspace = True
    return stokes


@pytest.mark.tier_b
@pytest.mark.parametrize("fixture", (_triangle_with_an_empty_rank, _region_held_by_rank_0))
def test_stokes_is_refused_on_every_rank(fixture):
    mesh, walls = fixture()
    counts = cells_per_rank(mesh)
    stokes = _stokes(mesh, walls)
    try:
        stokes.solve()
        outcome = "solved"
    except RuntimeError as error:
        outcome = str(error)
    outcomes = uw.mpi.comm.allgather(outcome)

    assert counts.count(0) >= 1 and sum(counts) > 0, counts
    assert len(set(outcomes)) == 1, outcomes
    assert (f"ranks without cells of this mesh: {counts.count(0)} of {uw.mpi.size} "
            f"({sum(counts)} cells in all)") in outcome, outcome
    assert "#962" in outcome, outcome


@pytest.mark.tier_b
def test_a_scalar_solve_on_the_region_is_not_refused():
    mesh, walls = _region_held_by_rank_0()
    counts = cells_per_rank(mesh)
    t = uw.discretisation.MeshVariable("T0786", mesh, 1, degree=2)
    poisson = uw.systems.Poisson(mesh, u_Field=t)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 0.0
    x, _y = mesh.X
    for wall in walls:
        poisson.add_essential_bc((x,), wall)
    poisson.tolerance = 1.0e-10
    poisson.solve()

    reason = poisson.snes.getConvergedReason()
    local = np.abs(np.asarray(t.array)[:, 0, 0] - np.asarray(t.coords)[:, 0])
    worst = max(uw.mpi.comm.allgather(float(local.max(initial=0.0))))
    assert counts.count(0) >= 1 and sum(counts) > 0, counts
    assert reason > 0, reason
    # the walls prescribe x, which the elements hold exactly
    assert worst < 1.0e-8, (worst, counts)
