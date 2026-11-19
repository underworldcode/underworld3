"""An integration-point variable on a rank that holds no cells.

``_BaseIntegrationPointVariable._create_petsc_fe`` took the reference cell of the
element from the type of the rank's first local cell. On a rank with no cells
there is no first cell: PETSc returns ``DM_POLYTOPE_UNKNOWN``, has no reference
cell for it, and raises (error 62) on that rank alone. The other ranks carry on
into the next collective (``DMClone`` in ``createSubDM``) and wait there, or the
job aborts when the failing rank's exit meets their allreduce. Measured on
development 35ac1397 at np 2 and np 3, on a 96-thread Linux workstation (MPICH),
by the parallel review desk. Proxies at nodes and cells were not affected.

The mesh is the right triangle of ``empty_rank_fixtures``, which leaves one rank
without cells (which one is PETSc's choice). The baselines are exact: every rank
creates the variable, and the rank without cells holds zero integration points,
zero values and a ``(0, Nq, ncomp)`` cell view, where ``Nq`` is the number of
points of the mesh rule, the same on every rank. In serial no rank is empty and
the same checks hold with the rank's own cell count.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw
from empty_rank_fixtures import (cells_along_a_side_leaving_one_rank_empty, cells_per_rank,
                                 right_triangle_of_cells)

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b,
              pytest.mark.timeout(240, method="thread")]


def _triangle():
    size = uw.mpi.size
    dm = right_triangle_of_cells(cells_along_a_side_leaving_one_rank_empty(size),
                                 max(size - 1, 1))
    mesh = uw.discretisation.Mesh(dm, degree=1, qdegree=2)
    counts = cells_per_rank(mesh)
    if uw.mpi.size > 1:
        assert 0 in counts, f"the fixture left no rank without cells: {counts}"
    return mesh, counts[uw.mpi.rank]


def _points_per_cell(mesh):
    """Points of the mesh rule, from the rule itself (the same on every rank)."""
    _, weights = mesh.integration_rule.getData()
    return len(weights)


def _assert_shapes(var, ncells, nq, ncomp, cdim):
    assert np.asarray(var.integration_points).shape == (ncells, nq, cdim)
    assert np.asarray(var.data).shape == (ncells * nq, ncomp)
    assert np.asarray(var.cell_data).shape == (ncells, nq, ncomp)


def test_an_integration_point_variable_is_created_on_every_rank():
    mesh, ncells = _triangle()
    nq = _points_per_cell(mesh)
    scalar = uw.discretisation.IntegrationPointVariable("q_scalar", mesh)
    # A second one takes the path for a mesh that already has fields (the DM is
    # cloned), which is collective too.
    vector = uw.discretisation.IntegrationPointVariable("q_vector", mesh, num_components=2)
    uw.mpi.barrier()
    _assert_shapes(scalar, ncells, nq, 1, mesh.cdim)
    _assert_shapes(vector, ncells, nq, 2, mesh.cdim)

    scalar.cell_data[...] = 2.0
    assert np.all(np.asarray(scalar.data) == 2.0)


def test_a_swarm_variable_with_an_integration_point_proxy_is_created_on_every_rank():
    mesh, ncells = _triangle()
    nq = _points_per_cell(mesh)
    swarm = uw.swarm.Swarm(mesh)
    s = uw.swarm.SwarmVariable("s", swarm, 1, proxy_location="integration_points")
    uw.mpi.barrier()
    _assert_shapes(s._meshVar, ncells, nq, 1, mesh.cdim)


def test_the_integration_point_semi_lagrangian_history_is_built_on_every_rank():
    mesh, ncells = _triangle()
    nq = _points_per_cell(mesh)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    T.data[:, 0] = 1.0
    V = sympy.Matrix([[0.0, 0.0]])
    ddt = uw.systems.ddt.IntegrationPointSemiLagrangian(mesh, T, V, degree=1, order=1)
    uw.mpi.barrier()
    for history in ddt.psi_star:
        _assert_shapes(history, ncells, nq, 1, mesh.cdim)


def test_a_material_swarm_builds_its_integration_point_level_sets_on_every_rank():
    mesh, ncells = _triangle()
    nq = _points_per_cell(mesh)
    materials = uw.swarm.MaterialSwarm(mesh, fill_param=2, name="ipvar_empty_rank")
    materials.add("lower", shear_viscosity_0=1.0)
    materials.add("upper", shear_viscosity_0=10.0)
    materials["upper"] = mesh.X[1] > 0.5
    # The level sets are built on the first read of a property; build them here.
    materials._ensure_built()
    uw.mpi.barrier()
    for level_set in materials._level_sets():
        _assert_shapes(level_set, ncells, nq, 1, mesh.cdim)
