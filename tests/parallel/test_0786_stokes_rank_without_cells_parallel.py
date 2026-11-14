"""A Stokes solve completes on a mesh with a rank that holds no cells (#962).

The Stokes solver registers kernels for the Jacobian and for the matrix its
preconditioner is built from, so SNES assembles two matrices. PETSc's one-pass
assembly (``DMPlexComputeJacobianByKey``, 3.25.3 and ``main``) leaves early on a
rank without cells and skips an assembly of the Jacobian that the other ranks
make: the solve stopped in ``MPI_Allreduce`` on the ranks without cells, or hung.
The solver's local Jacobian routine hands PETSc one matrix to a call
(``UW_DMPlexSNESComputeJacobianFEM``), and every rank then assembles alike.

Power: the box of two cells and the region leave a rank without cells at every
np >= 2 and the triangle from np 3 up, and each test asserts that premise. With
PETSc's routine registered the run stops in MPI or hangs, and the MPI
supervisor's silence limit ends it. The value asserted is the largest nodal speed
of the same solve on one rank, where PETSc's routine is right. The box is a
structured mesh of two quadrilaterals, the one
``tests/parallel/test_0766_mover_refreshes_mesh_vector_parallel.py`` relies on
for a rank without cells. At np 2 the triangle leaves a rank empty only when it
is one cell, which has every velocity node on a wall and nothing to solve for, so
it is skipped there; ``scripts/test.sh`` runs this file at np 3 as well.
``tests/test_0786_stokes_jacobian_and_preconditioner_one_call_each.py`` compares
the matrices themselves on one rank.

A scalar solve has one matrix and never met the defect: on the same region it
converges to the linear field its walls prescribe.

A constrained solve removes its multipliers' interior unknowns from the system
and rebuilds the DM's global section, which every rank does together. A rank
with nothing of its own to remove left before the rebuild, and the solve hung
at set-up. On the region rank 0 removes 104 of each multiplier's 109 unknowns
and the other ranks have none.

    python scripts/mpi_supervisor.py --silence 300 -- mpirun -n 3 python -m pytest --with-mpi tests/parallel/test_0786_stokes_rank_without_cells_parallel.py
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

NO_SLIP = (0.0, 0.0)
REGION_WALLS = ("Top", "Left", "Right", "Internal")
# no flow through a wall, free to slide along it: with no slip the box's three
# interior velocity nodes do not move
BOX_WALLS = {"Top": (None, 0.0), "Bottom": (None, 0.0), "Left": (0.0, None), "Right": (0.0, None)}


# The largest nodal speed of each solve on one rank (macOS, PETSc 3.25.3): for the
# box, by cells along the triangle's side, and for the region. A triangle of one
# cell has no free velocity node.
SPEED_BOX = 3.5798925e-02
SPEED_TRIANGLE = {1: None, 4: 2.4923109e-03, 8: 2.5355659e-03}
SPEED_REGION = 8.1535981e-05
SPEED_REGION_CONSTRAINED = 1.2158645e-03


def _box_of_two_cells():
    """Both cells go to one rank: the partitioner decides that, and the test asserts it."""
    mesh = uw.meshing.StructuredQuadBox(elementRes=(1, 2), qdegree=3)
    return mesh, BOX_WALLS, SPEED_BOX


def _triangle_with_an_empty_rank():
    size = uw.mpi.size
    n = cells_along_a_side_leaving_one_rank_empty(size)
    dm = right_triangle_of_cells(n, max(size - 1, 1))
    # a mesh made from a bare DM has the name All_Boundaries and no such label (#1138)
    dm.createLabel("All_Boundaries")
    dm.markBoundaryFaces("All_Boundaries", ALL_BOUNDARIES_VALUE)
    return uw.discretisation.Mesh(dm, qdegree=3), {"All_Boundaries": NO_SLIP}, SPEED_TRIANGLE[n]


def _region_held_by_rank_0():
    """The outer slab of the #858 tests' parent, all of it on rank 0."""
    path = Path(__file__).resolve().parent / "test_1087_submesh_prolongation_ownership.py"
    spec = importlib.util.spec_from_file_location("_submesh_ownership", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parent = module._thin_slab_parent("single_rank_region")
    return parent.extract_region("Outer"), dict.fromkeys(REGION_WALLS, NO_SLIP), SPEED_REGION


def _stokes(mesh, walls):
    v = uw.discretisation.MeshVariable("V0786", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P0786", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, _y = mesh.X
    stokes.bodyforce = sympy.Matrix([[0.0, sympy.cos(sympy.pi * x)]])
    for wall, velocity in walls.items():
        stokes.add_essential_bc(velocity, wall)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-10
    return stokes, v


@pytest.mark.tier_b
@pytest.mark.parametrize(
    "fixture", (_box_of_two_cells, _triangle_with_an_empty_rank, _region_held_by_rank_0))
def test_stokes_solves_with_a_rank_without_cells(fixture):
    mesh, walls, speed_on_one_rank = fixture()
    if speed_on_one_rank is None:
        pytest.skip(f"np {uw.mpi.size}: the triangle is one cell, with no free velocity node")
    counts = cells_per_rank(mesh)
    stokes, v = _stokes(mesh, walls)
    stokes.solve()

    reason = stokes.snes.getConvergedReason()
    local = np.linalg.norm(np.asarray(v.array)[:, 0, :], axis=1)
    speed = max(uw.mpi.comm.allgather(float(local.max(initial=0.0))))
    assert counts.count(0) >= 1 and sum(counts) > 0, counts
    assert reason > 0, reason
    assert abs(speed - speed_on_one_rank) <= 1.0e-6 * speed_on_one_rank, (speed, counts)


@pytest.mark.tier_b
def test_a_scalar_solve_on_the_region_converges():
    mesh, walls, _speed = _region_held_by_rank_0()
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


@pytest.mark.tier_b
def test_a_constrained_solve_completes_with_a_rank_without_cells():
    """No flow through the region's side walls by multipliers, no slip on its base."""
    mesh, _walls, _speed = _region_held_by_rank_0()
    counts = cells_per_rank(mesh)
    v = uw.discretisation.MeshVariable("V0786c", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P0786c", mesh, 1, degree=1)
    stokes = uw.systems.Stokes_Constrained(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, _y = mesh.X
    stokes.bodyforce = sympy.Matrix([[0.0, sympy.cos(sympy.pi * x)]])
    stokes.add_essential_bc(NO_SLIP, "Internal")
    stokes.add_constraint_bc(0.0, "Left", normal=sympy.Matrix([[-1.0, 0.0]]))
    stokes.add_constraint_bc(0.0, "Right", normal=sympy.Matrix([[1.0, 0.0]]))
    stokes.tolerance = 1.0e-10
    stokes.solve()

    reason = stokes.snes.getConvergedReason()
    local = np.linalg.norm(np.asarray(v.array)[:, 0, :], axis=1)
    speed = max(uw.mpi.comm.allgather(float(local.max(initial=0.0))))
    section = stokes.dm.getLocalSection()
    multiplier = stokes._block_constraint_bcs[0].lam._solver_field_id
    removed = uw.mpi.comm.allgather(
        sum(section.getFieldConstraintDof(q, multiplier) for q in range(*section.getChart())))
    assert counts.count(0) >= 1 and sum(counts) > 0, counts
    # the premise: a rank that removes multiplier unknowns beside one that has none
    assert max(removed) > 0 and min(removed) == 0, removed
    assert reason > 0, reason
    assert abs(speed - SPEED_REGION_CONSTRAINED) <= 1.0e-6 * SPEED_REGION_CONSTRAINED, (speed, counts)
