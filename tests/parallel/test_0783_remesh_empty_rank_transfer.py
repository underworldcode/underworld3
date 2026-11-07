"""``Mesh.remesh`` transfers every variable past a rank holding none of its DOFs (#909).

The transfer loop asked for the new mesh's node coordinates, a collective call,
only on a rank where the old variable had DOFs. A rank with no cells skipped it
while the others entered it, and the ranks hung. The transfer now runs on every
rank, and each new node is evaluated on whichever rank holds it: a rank with no
old cells can hold new ones, with nothing of its own to evaluate them from.

Stand-in: no PETSc we can build adapts a simplex mesh on more than one rank
(``ptest_0763_mesh_adapt_transfer``), and conda-forge's has no adapter at all.
So ``adaptMetric`` is replaced, and returns a prepared finer mesh with a forced
partition. So is ``metricCreateIsotropic``, the adapter's input, after which a
later ``dm.distribute`` hangs when a rank has no cells (#975). All of ``remesh``
around them is the real one, and the transfer loop knows nothing of where its new
DM came from.

Power, each test asserting its premise:

- no old DOFs on one rank, new cells on every rank: every np >= 2, CI's np 2 too;
- no old DOFs and no new cells on one rank: np >= 3;
- old cells everywhere, no new cells on all ranks but one: every np >= 2;
- a transfer or restore failing on one rank: every np >= 2;
- a child mesh with empty ranks: np >= 4, which CI's default np 2 does not reach.

A regression hangs and the MPI supervisor's silence limit ends it, so every assert
follows the collectives it needs.

    python scripts/mpi_supervisor.py --silence 300 -- mpirun -n 3 python -m pytest --with-mpi tests/parallel/test_0783_remesh_empty_rank_transfer.py
"""
import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw
from underworld3.systems import ddt

from empty_rank_fixtures import (AdapterStandIn, cells_along_a_side_leaving_one_rank_empty,
                                 right_triangle_of_cells)

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(300)]

NEW_CELLS_ALONG_A_SIDE = 6


def _cells_per_rank(dm):
    start, end = dm.getHeightStratum(0)
    return uw.mpi.comm.allgather(end - start)


def _linear_field(coords):
    """Exact at every degree used here, so any error is the transfer's own and
    none of it is approximation."""
    return 1.0 + coords[:, 0] + 2.0 * coords[:, 1]


def _create_every_variable_type(mesh, tag):
    """One variable of each type, holding a different multiple of the linear field
    in each component, so a component arriving in the wrong place cannot pass."""
    variables = [
        uw.discretisation.MeshVariable(f"scalar_{tag}", mesh, 1, degree=2),
        uw.discretisation.MeshVariable(f"vector_{tag}", mesh, 2, degree=1),
        uw.discretisation.MeshVariable(f"discontinuous_{tag}", mesh, 1, degree=1, continuous=False),
        uw.discretisation.MeshVariable(f"tensor_{tag}", mesh, (2, 2), degree=1,
                                       vtype=uw.VarType.TENSOR),
        uw.discretisation.MeshVariable(f"symmetric_{tag}", mesh, (2, 2), degree=1,
                                       vtype=uw.VarType.SYM_TENSOR),
        uw.discretisation.MeshVariable(f"matrix_{tag}", mesh, (2, 3), degree=1,
                                       vtype=uw.VarType.MATRIX),
    ]
    for var in variables:
        _fill(var)
    return variables


def _component_factor(var, i, j):
    """Symmetric for a symmetric tensor, and otherwise different in each component."""
    return 1 + i + j if var.vtype == uw.VarType.SYM_TENSOR else 1 + i + 2 * j


def _fill(var):
    values = _linear_field(np.asarray(var.coords))
    for i in range(var.shape[0]):
        for j in range(var.shape[1]):
            var.array[:, i, j] = _component_factor(var, i, j) * values


def _worst_errors(variables):
    """The largest error in any component of each variable, over all ranks."""
    errors = {}
    for var in variables:
        values = _linear_field(np.asarray(var.coords))
        array = np.asarray(var.array)
        local = 0.0
        for i in range(var.shape[0]):
            for j in range(var.shape[1]):
                difference = array[:, i, j] - _component_factor(var, i, j) * values
                local = max(local, float(np.abs(difference).max(initial=0.0)))
        errors[var.clean_name] = uw.mpi.comm.allreduce(local, op=MPI.MAX)
    return errors


def _largest_value(var):
    return uw.mpi.comm.allreduce(float(np.abs(np.asarray(var.array)).max(initial=0.0)), op=MPI.MAX)


def _metric(mesh, tag):
    metric = uw.discretisation.MeshVariable(f"H_{tag}", mesh, 1, degree=1)
    metric.array[:, 0, 0] = float(NEW_CELLS_ALONG_A_SIDE ** 2)
    return metric


@pytest.mark.parametrize("new_cells_on", ["every_rank", "all_ranks_but_last"])
def test_remesh_transfers_past_a_rank_with_none_of_the_dofs(new_cells_on):
    """Every rank returns from ``remesh``, and a field exact in the new space
    arrives exact in every component on every rank, including one that had no old
    cells."""
    size = uw.mpi.size
    if new_cells_on == "all_ranks_but_last" and size == 2:
        pytest.skip("a DM on one rank alone is redistributed; see empty_rank_fixtures")

    n_old = cells_along_a_side_leaving_one_rank_empty(size)
    old_dm = right_triangle_of_cells(n_old, max(size - 1, 1), dm_class=AdapterStandIn)
    mesh = uw.discretisation.Mesh(old_dm, degree=1, qdegree=3)
    old_cells = _cells_per_rank(mesh.dm)

    variables = _create_every_variable_type(mesh, new_cells_on)
    metric = _metric(mesh, new_cells_on)
    old_dofs = uw.mpi.comm.allgather(variables[0].data.size)

    ranks_with_new_cells = size if new_cells_on == "every_rank" else size - 1
    AdapterStandIn.adapted = right_triangle_of_cells(NEW_CELLS_ALONG_A_SIDE, ranks_with_new_cells)
    mesh.remesh(metric)

    new_cells = _cells_per_rank(mesh.dm)
    errors = _worst_errors(variables)

    assert old_cells.count(0) == 1 and sum(old_cells) == n_old ** 2, old_cells
    assert [d == 0 for d in old_dofs] == [c == 0 for c in old_cells], (old_dofs, old_cells)
    assert sum(new_cells) == NEW_CELLS_ALONG_A_SIDE ** 2, new_cells
    assert (min(new_cells) > 0) == (new_cells_on == "every_rank"), new_cells
    assert all(e < 1.0e-9 for e in errors.values()), errors


def test_remesh_transfers_to_ranks_with_no_new_cells():
    """Every variable type where a rank has old DOFs and no new cells, at every
    np >= 2. A one-cell new mesh is the only way to empty a rank at np 2. The old
    grid stays at n = 4 because that keeps the new nodes clear of #976 at np 8, where
    n = 8, 12 and 20 each leave one or two of them extrapolated, out by up to 1.1e-2."""
    size = uw.mpi.size
    n_old = 4
    old_dm = right_triangle_of_cells(n_old, size, dm_class=AdapterStandIn)
    mesh = uw.discretisation.Mesh(old_dm, degree=1, qdegree=3)
    old_cells = _cells_per_rank(mesh.dm)

    variables = _create_every_variable_type(mesh, "no_new_cells")
    metric = _metric(mesh, "no_new_cells")

    AdapterStandIn.adapted = right_triangle_of_cells(1, 1)
    mesh.remesh(metric)

    new_cells = _cells_per_rank(mesh.dm)
    errors = _worst_errors(variables)

    assert min(old_cells) > 0, old_cells
    assert new_cells.count(0) == size - 1 and sum(new_cells) == 1, new_cells
    assert all(e < 1.0e-9 for e in errors.values()), errors


@pytest.mark.parametrize("stage", ["transfer", "restore"])
def test_a_failure_on_one_rank_resets_the_variable_on_every_rank(stage, monkeypatch):
    """A raise on one rank, after the collectives of its stage, resets that variable
    on all ranks and leaves the others alone. Each rank deciding for itself would
    send some ranks into the restore's collective without the others, or leave the
    variable part zero and part transferred."""
    size = uw.mpi.size
    n_old = cells_along_a_side_leaving_one_rank_empty(size)
    old_dm = right_triangle_of_cells(n_old, max(size - 1, 1), dm_class=AdapterStandIn)
    mesh = uw.discretisation.Mesh(old_dm, degree=1, qdegree=3)

    tag = f"fails_in_{stage}"
    victim = uw.discretisation.MeshVariable(f"victim_{tag}", mesh, 1, degree=2)
    bystander = uw.discretisation.MeshVariable(f"bystander_{tag}", mesh, 1, degree=2)
    _fill(victim)
    _fill(bystander)
    metric = _metric(mesh, tag)

    failing_rank = size - 1

    def raises_on_one_rank_for_the_victim(original, var_symbol_or_var):
        if uw.mpi.rank == failing_rank and victim.clean_name in str(var_symbol_or_var):
            raise RuntimeError("injected failure")

    if stage == "transfer":
        global_evaluate = uw.function.global_evaluate

        def failing_global_evaluate(expr, coords):
            result = global_evaluate(expr, coords)
            raises_on_one_rank_for_the_victim(global_evaluate, expr)
            return result

        monkeypatch.setattr(uw.function, "global_evaluate", failing_global_evaluate)
    else:
        layout_values = ddt._evaluated_in_storage_layout

        def failing_layout_values(var, values):
            raises_on_one_rank_for_the_victim(layout_values, var.clean_name)
            return layout_values(var, values)

        monkeypatch.setattr(ddt, "_evaluated_in_storage_layout", failing_layout_values)

    AdapterStandIn.adapted = right_triangle_of_cells(NEW_CELLS_ALONG_A_SIDE, size)
    mesh.remesh(metric)

    victim_largest = _largest_value(victim)
    bystander_errors = _worst_errors([bystander])

    assert victim_largest == 0.0, victim_largest
    assert all(e < 1.0e-9 for e in bystander_errors.values()), bystander_errors


def test_a_rank_with_no_cells_can_ask_about_points_elsewhere():
    """What the transfer asks of an empty rank that has new nodes. The rank's own
    classification used to hand back one array where its callers unpack two, so
    it raised alone while the others waited in the swarm's migration."""
    size = uw.mpi.size
    n = cells_along_a_side_leaving_one_rank_empty(size)
    mesh = uw.discretisation.Mesh(right_triangle_of_cells(n, max(size - 1, 1)),
                                  degree=1, qdegree=3)
    counts = _cells_per_rank(mesh.dm)
    field = uw.discretisation.MeshVariable("T_asks", mesh, 1, degree=2)
    field.array[:, 0, 0] = _linear_field(np.asarray(field.coords))

    points = np.array([[0.1, 0.1], [0.2, 0.3], [0.05, 0.6]])
    values = uw.function.global_evaluate(field.sym, points)
    error = np.abs(np.asarray(values).reshape(-1) - _linear_field(points)).max()
    worst = uw.mpi.comm.allreduce(float(error), op=MPI.MAX)

    assert counts.count(0) == 1, counts
    assert worst < 1.0e-12, worst


@pytest.mark.mpi(min_size=4)
def test_global_evaluate_on_a_child_mesh_with_empty_ranks():
    """#965: a region of a parent, where some ranks hold none of the child. Points
    asked for on one rank reach the empty ranks in the swarm's migration, whose
    classifier handed back the wrong shape there, and they raised alone."""
    parent = uw.meshing.BoxInternalBoundary(
        elementRes=(12, 12), zelementRes=(10, 2), zintCoord=0.85,
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), simplex=True, qdegree=3)
    child = parent.extract_region("Outer")
    counts = _cells_per_rank(child.dm)
    field = uw.discretisation.MeshVariable("T_child", child, 1, degree=2)
    field.array[:, 0, 0] = _linear_field(np.asarray(field.coords))

    points = np.array([[0.2, 0.9], [0.7, 0.95]]) if uw.mpi.rank == 0 else np.zeros((0, 2))
    values = uw.function.global_evaluate(field.sym, points)
    error = np.abs(np.asarray(values).reshape(-1) - _linear_field(points)).max(initial=0.0)
    worst = uw.mpi.comm.allreduce(float(error), op=MPI.MAX)

    assert counts.count(0) > 0, counts
    assert worst < 1.0e-12, worst
