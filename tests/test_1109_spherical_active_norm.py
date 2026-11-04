"""Active-background norms: exact two-tet support, identities and unchanged physics."""

import ast
from contextlib import ExitStack
import inspect
import textwrap

import numpy as np
from petsc4py import PETSc
import pytest
import underworld3 as uw
import underworld3.utilities._spherical_stokes_gap as spherical_gap
from underworld3.utilities._cylindrical_gap import _layout
from underworld3.utilities._spherical_cut_cell import _inner_cell_relation

pytestmark = pytest.mark.level_2


def _two_tetrahedra():
    coordinates = np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.],
                            [.2, .2, .2], [1., 1., 1.]])
    cells = np.array([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=PETSc.IntType)
    if PETSc.COMM_WORLD.rank:
        coordinates = np.empty((0, 3))
        cells = np.empty((0, 4), dtype=PETSc.IntType)
    plex = PETSc.DMPlex().createFromCellList(
        3, cells, coordinates, interpolate=True, comm=PETSc.COMM_WORLD,
    )
    mesh = uw.discretisation.Mesh(plex, degree=1, qdegree=4, markVertices=True)
    velocity = uw.discretisation.MeshVariable("ActiveNormVelocity", mesh, 3, degree=2,
                                             vtype=uw.VarType.VECTOR)
    pressure = uw.discretisation.MeshVariable("ActiveNormPressure", mesh, 1, degree=1)
    vertices = np.asarray(mesh._get_coords_for_basis(1, True))[mesh._cell_node_indices(1, True)]
    velocity_rows = spherical_gap._canonical_rows(mesh, velocity, vertices)
    pressure_rows = spherical_gap._canonical_rows(mesh, pressure, vertices)
    velocity_map, velocity_size = _layout(velocity)
    pressure_map, pressure_size = _layout(pressure)
    return dict(mesh=mesh, velocity=velocity, pressure=pressure, vertices=vertices,
                uid=velocity_map.reshape(-1, 3)[velocity_rows].reshape(-1, 30),
                pid=pressure_map[pressure_rows]+velocity_size,
                owned=np.flatnonzero(mesh._get_owned_cells_mask()),
                velocity_map=velocity_map, velocity_size=velocity_size, pressure_size=pressure_size)


def _production_background(environment):
    # Exercise the real production restriction, not a test-side cell filter.
    # This pass needs no closed spherical wall and performs no Stokes solve.
    source = textwrap.dedent(inspect.getsource(spherical_gap._solve_constraints))
    loops = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.For)
             and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                     and isinstance(call.func.value, ast.Name)
                     and call.func.value.id == "norm_blocks" and call.func.attr == "add_volume"
                     for call in ast.walk(node))]
    assert len(loops) == 1, "Find exactly the production background-volume pass"
    namespace = dict(spherical_gap.__dict__)
    namespace.update(environment)
    exec(compile(ast.Module(body=loops, type_ignores=[]), spherical_gap.__file__, "exec"), namespace)


def _assemble(geometry, radius, cleanup, *, with_norm=True):
    velocity_size, pressure_size = geometry["velocity_size"], geometry["pressure_size"]
    size = velocity_size+pressure_size+1
    matrix = PETSc.Mat().createAIJ((size, size), nnz=size, comm=PETSc.COMM_WORLD)
    cleanup.callback(matrix.destroy)
    lower, upper = matrix.getOwnershipRange()
    for row in range(lower, upper):
        matrix.setValues(row, np.arange(size, dtype=PETSc.IntType), np.zeros(size))
    matrix.assemble()
    norm = None
    if with_norm:
        norm = spherical_gap._SphericalGapNormBlocks(
            geometry["velocity"], matrix, velocity_size, pressure_size, 1, 1, .125,
        )
        cleanup.callback(norm.destroy)
    mean, *rotations = [matrix.createVecRight() for _ in range(4)]
    for vector in (mean, *rotations):
        vector.set(0)
        cleanup.callback(vector.destroy)
    # Delta-sheet/pressure-lift paths can preload a nonzero RHS before this pass.
    rhs = matrix.createVecRight()
    cleanup.callback(rhs.destroy)
    rhs.array[:] = np.cos(np.arange(lower, upper)+.3)
    environment = dict(geometry, matrix=matrix, rhs=rhs, norm_blocks=norm, pmean=mean,
                       rotation_weights=rotations, viscosity=1., batch_size=32,
                       has_volume_bodyforce=False,
                       relation=_inner_cell_relation(geometry["vertices"], radius))
    _production_background(environment)
    matrix.assemble()
    for vector in (mean, *rotations):
        vector.assemble()
    if norm is not None:
        if PETSc.COMM_WORLD.rank == 0:
            norm.add_surface(np.ones((1, 1)), np.ones((1, 1, 1)),
                             np.array([[velocity_size+pressure_size]], dtype=PETSc.IntType))
        norm.assemble()
    return matrix, norm, rhs, mean, rotations


@pytest.mark.tier_b
def test_production_norm_restricts_void_cell_and_completes_only_unsupported_rows():
    geometry = _two_tetrahedra()
    with ExitStack() as cleanup:
        matrix, norm, rhs, mean, rotations = _assemble(geometry, 1., cleanup)
        unchanged_matrix, _, unchanged_rhs, unchanged_mean, unchanged_rotations = _assemble(
            geometry, 1., cleanup, with_norm=False,
        )
        for left, right in zip(matrix.getValuesCSR(), unchanged_matrix.getValuesCSR()):
            np.testing.assert_array_equal(left, right)
        for left, right in zip((rhs, mean, *rotations), (unchanged_rhs, unchanged_mean, *unchanged_rotations)):
            np.testing.assert_array_equal(left.array, right.array)
        # The known active tetrahedron is the one containing (1,1,1).
        # This classification defines the oracle only; both cells reached production.
        owned = geometry["owned"]
        active = np.max(np.linalg.norm(geometry["vertices"][owned], axis=-1), axis=1) > 1
        communicator = matrix.comm.tompi4py()
        supported = set(value for batch in communicator.allgather(
            geometry["uid"][owned][active].ravel().tolist()) for value in batch)
        supported_pressure = set(value-geometry["velocity_size"] for batch in communicator.allgather(
            geometry["pid"][owned][active].ravel().tolist()) for value in batch)
        unsupported = sorted(set(range(geometry["velocity_size"]))-supported)
        unsupported_pressure = sorted(set(range(geometry["pressure_size"]))-supported_pressure)
        assert len(unsupported) == 12 and len(unsupported_pressure) == 1
        coefficients = norm.h.createVecRight()
        image = coefficients.duplicate()
        cleanup.callback(coefficients.destroy)
        cleanup.callback(image.destroy)
        lower, upper = coefficients.getOwnershipRange()
        indices = np.arange(lower, upper)
        coefficients.array[:] = ((indices % 3 == 0) & np.isin(indices, list(supported)))
        norm.h.mult(coefficients, image)
        energy = float(coefficients.dot(image).real)
        # Constant x velocity on the active tetrahedron: zero viscous energy,
        # and mass energy exactly its affine volume1/3. Void-exclusive values0.
        np.testing.assert_allclose(energy, 1/3, rtol=1e-12, atol=0,
                                   err_msg="Production norm retains energy from a wholly void tetrahedron")
        diagonal = norm.h.getDiagonal()
        cleanup.callback(diagonal.destroy)
        owned_unsupported = [row for row in unsupported if lower <= row < upper]
        if PETSc.COMM_WORLD.size == 2:
            assert min(communicator.allgather(len(owned_unsupported))) == 0
        for row in owned_unsupported:
            columns, values = norm.h.getRow(row)
            assert values[columns == row].sum() == 1
            assert np.count_nonzero(values[columns != row]) == 0
        pressure_lower, pressure_upper = norm.pressure.getOwnershipRange()
        for row in range(pressure_lower, pressure_upper):
            expected = 1/30 if row in supported_pressure else 1.
            np.testing.assert_allclose(norm.pressure.array[row-pressure_lower], expected, rtol=1e-13)
        before = [array.copy() for array in norm.h.getValuesCSR()]
        pressure_before = norm.pressure.array.copy()
        norm.assemble()
        for left, right in zip(before, norm.h.getValuesCSR()):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(norm.pressure.array, pressure_before)
        space = PETSc.NullSpace().create(constant=False, comm=matrix.comm)
        cleanup.callback(space.destroy)
        coordinates = np.asarray(geometry["velocity"].coords)
        rotations_at_nodes = np.stack([np.cross(axis, coordinates) for axis in np.eye(3)], axis=-1).reshape(-1, 3)
        context = norm.context(geometry["velocity_map"], coordinates, rotations_at_nodes, space, "active_norm_regression_")
        cleanup.callback(context.destroy, None)
        assert context.velocity_ksp.getType() == "preonly"
        assert context.velocity_ksp.getPC().getType() == "gamg"
        result = rhs.duplicate()
        cleanup.callback(result.destroy)
        source_before = rhs.array.copy()
        context.apply(None, rhs, result)
        assert np.isfinite(result.norm())
        np.testing.assert_array_equal(rhs.array, source_before)
        for left, right in zip(matrix.getValuesCSR(), unchanged_matrix.getValuesCSR()):
            np.testing.assert_array_equal(left, right)


@pytest.mark.tier_b
def test_no_void_production_norm_matches_full_background_entries_bitwise():
    geometry = _two_tetrahedra()
    with ExitStack() as cleanup:
        matrix, production, _, _, _ = _assemble(geometry, .1, cleanup)
        direct = spherical_gap._SphericalGapNormBlocks(
            geometry["velocity"], matrix, geometry["velocity_size"], geometry["pressure_size"], 1, 1, .125,
        )
        cleanup.callback(direct.destroy)
        # The constructor borrows the current velocity block graph. Discard its
        # existing numerical entries to reproduce the production zero graph.
        direct.h.zeroEntries()
        cells = geometry["owned"]
        points, weights = spherical_gap._tetrahedron_quadrature(geometry["vertices"][cells], 4)
        blocks = spherical_gap._tetrahedral_stokes_blocks(geometry["vertices"][cells], points, weights)
        direct.add_volume(geometry["vertices"][cells], points, weights,
                          geometry["uid"][cells], geometry["pid"][cells], blocks["A"])
        if PETSc.COMM_WORLD.rank == 0:
            direct.add_surface(np.ones((1, 1)), np.ones((1, 1, 1)),
                               np.array([[geometry["velocity_size"]+geometry["pressure_size"]]], dtype=PETSc.IntType))
        direct.assemble()
        for left, right in zip(production.h.getValuesCSR(), direct.h.getValuesCSR()):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(production.pressure.array, direct.pressure.array)
        np.testing.assert_array_equal(production.multiplier.array, direct.multiplier.array)
