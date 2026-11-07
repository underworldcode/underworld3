"""What ``global_evaluate`` returns reaches a variable's storage in the right place.

``global_evaluate`` returns ``(N, a, b)`` values, dimensional when the variable has
units. Storage is ``(N, components)`` in ``_data_layout`` order, non-dimensional,
and a MATRIX is stored by column.

- ``Mesh.remesh`` packed dimensional values as raw (#977), and zeroed a
  units-carrying variable on development.
- ``mesh.deform`` and the parallel refinement transfers reshape by row. That was
  right for a square MATRIX only while ``global_evaluate`` transposed, and it raised
  on a symmetric tensor. They now follow the layout. Their units are left as on
  development: the values arrive as plain numbers, as before, so a deform with units
  gives development's numbers. That is checked here too, on a constant field.

Power: every test runs in serial and at every np. The ``remesh`` test has an empty
rank from np 2 up (``empty_rank_fixtures``). The refinement transfers take their
parallel route only from np 2.

    python scripts/mpi_supervisor.py --silence 300 -- mpirun -n 3 python -m pytest --with-mpi tests/parallel/test_0785_evaluated_values_reach_storage_parallel.py
"""
import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw

from empty_rank_fixtures import (AdapterStandIn, cells_along_a_side_leaving_one_rank_empty,
                                 right_triangle_of_cells)

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(300)]

DEFORM_ERROR_BOUND = 0.2


def _linear_field(coords):
    return 1.0 + coords[:, 0] + 2.0 * coords[:, 1]


def _cells_per_rank(dm):
    start, end = dm.getHeightStratum(0)
    return uw.mpi.comm.allgather(end - start)


def _factor(var, i, j):
    return 1 + i + j if var.vtype == uw.VarType.SYM_TENSOR else 1 + i + 2 * j


def _raw_coords(var):
    return var.mesh._get_coords_for_basis(var.degree, var.continuous)


def _fill_raw(var, scale=1.0):
    values = scale * _linear_field(_raw_coords(var))
    data = np.array(var.data)
    for i in range(var.sym.shape[0]):
        for j in range(var.sym.shape[1]):
            data[:, var._data_layout(i, j)] = _factor(var, i, j) * values
    var.data[:] = data


def _worst_raw_error(var, scale=1.0):
    values = scale * _linear_field(_raw_coords(var))
    data = np.asarray(var.data)
    local = 0.0
    for i in range(var.sym.shape[0]):
        for j in range(var.sym.shape[1]):
            difference = data[:, var._data_layout(i, j)] - _factor(var, i, j) * values
            local = max(local, float(np.abs(difference).max(initial=0.0)))
    return uw.mpi.comm.allreduce(local, op=MPI.MAX)


@pytest.fixture
def reference_scales():
    model = uw.get_default_model()
    model.set_reference_quantities(
        length=uw.quantity(1000, "km"), time=uw.quantity(1, "Myr"),
        mass=uw.quantity(1e24, "kg"), temperature=uw.quantity(1000, "K"))
    yield model
    uw.reset_default_model()


def test_remesh_keeps_the_non_dimensional_values_of_variables_with_units(reference_scales):
    """Raw values are what the transfer must keep."""
    size = uw.mpi.size
    n = cells_along_a_side_leaving_one_rank_empty(size) if size > 1 else 4
    mesh = uw.discretisation.Mesh(
        right_triangle_of_cells(n, max(size - 1, 1), dm_class=AdapterStandIn), degree=1, qdegree=3)
    temperature = uw.discretisation.MeshVariable("T_u", mesh, 1, degree=2, units="K")
    velocity = uw.discretisation.MeshVariable("V_u", mesh, 2, degree=1, units="cm/yr")
    stress = uw.discretisation.MeshVariable("S_u", mesh, (2, 2), degree=1,
                                            vtype=uw.VarType.SYM_TENSOR, units="Pa")
    matrix = uw.discretisation.MeshVariable("M_u", mesh, (2, 3), degree=1,
                                            vtype=uw.VarType.MATRIX, units="Pa")
    metric = uw.discretisation.MeshVariable("H_u", mesh, 1, degree=1)
    variables = {temperature: 0.5, velocity: 0.25, stress: 0.125, matrix: 0.125}
    for var, scale in variables.items():
        _fill_raw(var, scale)
    metric.data[:, 0] = 36.0

    AdapterStandIn.adapted = right_triangle_of_cells(6, size)
    mesh.remesh(metric)

    errors = {var.clean_name: _worst_raw_error(var, scale) for var, scale in variables.items()}
    assert all(e < 1.0e-9 for e in errors.values()), errors


def test_deform_keeps_every_tensor_shape_in_its_components():
    """Square MATRIX (transposed by the flat reshape once ``global_evaluate`` was
    right) and SYM_TENSOR (which raised), with a TENSOR beside them, cells on every
    rank: an empty rank in ``mesh.deform`` is #959's.

    The bound is not this change's error. At np 2 and 3 the seam error is the
    transfer's, #976: the same with the limiter off. At np 8 it is the limiter's,
    which takes its bounds from each rank's own nodes. Measured 2026-11-06, this
    displacement, values up to 12: 0 in serial, 6.6e-2 at np 2 and 3, 0.125 at np 4,
    0.14 at np 8. A component in the wrong place is out by 1.6 or more. The MATRIX
    holds the same numbers as the TENSOR and must come out with the same error,
    which is the tight check.
    """
    size = uw.mpi.size
    mesh = uw.discretisation.Mesh(right_triangle_of_cells(4 if size <= 4 else 8, size),
                                  degree=1, qdegree=3)
    counts = _cells_per_rank(mesh.dm)
    variables = [
        uw.discretisation.MeshVariable("M_d", mesh, (2, 2), degree=1, vtype=uw.VarType.MATRIX),
        uw.discretisation.MeshVariable("T_d", mesh, (2, 2), degree=1, vtype=uw.VarType.TENSOR),
        uw.discretisation.MeshVariable("S_d", mesh, (2, 2), degree=1, vtype=uw.VarType.SYM_TENSOR),
    ]
    for var in variables:
        _fill_raw(var)

    nodes = np.array(mesh._get_coords_for_basis(mesh.degree, True))
    x, y = nodes[:, 0], nodes[:, 1]
    bubble = x * y * (1.0 - x - y)
    nodes[:, 0] += 0.5 * bubble
    nodes[:, 1] += 0.25 * bubble
    mesh.deform(nodes)

    errors = {var.clean_name: _worst_raw_error(var) for var in variables}
    uw.mpi.pprint(f"DEFORM_ERRORS np={size} cells={counts} {errors}")
    assert min(counts) > 0, counts
    assert abs(errors["M_d"] - errors["T_d"]) < 1.0e-12, errors
    assert all(e < DEFORM_ERROR_BOUND for e in errors.values()), errors


def test_deform_with_units_keeps_a_constant_field(reference_scales):
    """No worse than development: a constant T of raw 0.5 K stays 0.5 through a
    deform of a units box to 0.99 X. With the transferred values reduced by the
    temperature scale it came out at 0.0005 (review round 3, H1)."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(uw.quantity(0, "km"), uw.quantity(0, "km")),
        maxCoords=(uw.quantity(1000, "km"), uw.quantity(1000, "km")),
        cellSize=uw.quantity(250, "km"), regular=True, qdegree=2)
    temperature = uw.discretisation.MeshVariable("T_c", mesh, 1, degree=1, units="K")
    temperature.data[:, 0] = 0.5

    mesh.deform(0.99 * np.asarray(mesh.X.coords))

    data = np.asarray(temperature.data)
    lowest = uw.mpi.comm.allreduce(float(data.min(initial=np.inf)), op=MPI.MIN)
    highest = uw.mpi.comm.allreduce(float(data.max(initial=-np.inf)), op=MPI.MAX)
    assert np.isclose(lowest, 0.5, rtol=1e-12) and np.isclose(highest, 0.5, rtol=1e-12), (lowest, highest)


def test_the_refinement_transfers_keep_a_square_matrix_in_its_components():
    """Both transfers between a parent and its refinement child, on a field exact in
    both spaces. From np 2 they take the ``global_evaluate`` route."""
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, regular=True, qdegree=3,
        refinement=1)
    size_field = uw.discretisation.MeshVariable("h_r", base, 1, degree=1)
    size_field.array[:, 0, 0] = 0.1
    child = base.adapt(size_field, max_levels=1, engine="sbr")

    errors = {}
    for shape, vtype in (((2, 2), uw.VarType.MATRIX), ((2, 3), uw.VarType.MATRIX),
                         ((2, 2), uw.VarType.TENSOR), ((2, 2), uw.VarType.SYM_TENSOR)):
        tag = f"{vtype.name}_{shape[1]}"
        parent_var = uw.discretisation.MeshVariable(f"P_{tag}", base, shape, degree=1, vtype=vtype)
        child_var = uw.discretisation.MeshVariable(f"C_{tag}", child, shape, degree=1, vtype=vtype)
        _fill_raw(parent_var)
        child._refine_prolongate(parent_var, child_var)
        errors[f"prolongate_{tag}"] = _worst_raw_error(child_var)

        _fill_raw(child_var)
        parent_var.data[:] = 0.0
        child._refine_restrict(child_var, parent_var)
        errors[f"restrict_{tag}"] = _worst_raw_error(parent_var)

    assert all(e < 1.0e-9 for e in errors.values()), errors


def _units_refinement_pair(tag):
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, regular=True, qdegree=3,
        refinement=1)
    size_field = uw.discretisation.MeshVariable(f"h_{tag}", base, 1, degree=1)
    size_field.array[:, 0, 0] = 0.1
    child = base.adapt(size_field, max_levels=1, engine="sbr")
    parent_var = uw.discretisation.MeshVariable(f"Tp_{tag}", base, 1, degree=1, units="K")
    child_var = uw.discretisation.MeshVariable(f"Tc_{tag}", child, 1, degree=1, units="K")
    return child, parent_var, child_var


@pytest.mark.mpi(min_size=2)
def test_the_parallel_refinement_transfers_refuse_a_variable_with_units(reference_scales):
    """The parallel route gets units wrong on every shape (#991): a scalar in K came
    out by 968 at np 3. It refuses on every rank together rather than give a wrong
    value, and nothing is written."""
    child, parent_var, child_var = _units_refinement_pair("refuse")
    _fill_raw(parent_var, 0.5)
    _fill_raw(child_var, 0.25)

    outcomes = []
    for transfer, source, target in ((child._refine_prolongate, parent_var, child_var),
                                     (child._refine_restrict, child_var, parent_var)):
        try:
            transfer(source, target)
            outcomes.append("returned")
        except NotImplementedError as error:
            outcomes.append("refused" if "#991" in str(error) else str(error))
    all_outcomes = uw.mpi.comm.allgather(outcomes)
    child_error = _worst_raw_error(child_var, 0.25)
    parent_error = _worst_raw_error(parent_var, 0.5)

    assert all_outcomes == [["refused", "refused"]] * uw.mpi.size, (all_outcomes, child_error, parent_error)
    assert child_error < 1.0e-12 and parent_error < 1.0e-12, (child_error, parent_error)


def test_the_serial_refinement_transfers_keep_a_variable_with_units(reference_scales):
    """In serial the transfers take the custom-P route, which has no units fault."""
    if uw.mpi.size > 1:
        pytest.skip("the serial route")
    child, parent_var, child_var = _units_refinement_pair("serial")
    _fill_raw(parent_var, 0.5)
    child._refine_prolongate(parent_var, child_var)
    prolongate_error = _worst_raw_error(child_var, 0.5)
    _fill_raw(child_var, 0.25)
    child._refine_restrict(child_var, parent_var)
    restrict_error = _worst_raw_error(parent_var, 0.25)
    assert prolongate_error < 1.0e-9 and restrict_error < 1.0e-9, (prolongate_error, restrict_error)
