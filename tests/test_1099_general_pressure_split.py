"""Prescribed-jump exactness, numerical data, tips, junctions and Stokes signs."""

from enum import Enum

import numpy as np
import pytest
import sympy as sp
import underworld3 as uw
from petsc4py import PETSc

pytestmark = pytest.mark.level_1


class _Boundaries(Enum):
    Bottom = 1
    Top = 2
    Left = 3
    Right = 4
    Vertical = 5
    Horizontal = 6


def _cross_mesh(n=8, finite=False):
    """Triangles fitted to x=y=1/2, labelled before distribution."""
    grid = np.linspace(0, 1, n + 1)
    coords = np.array([(x, y) for y in grid for x in grid], dtype=float)
    cells = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            cells.extend([(a, a + 1, a + n + 2), (a, a + n + 2, a + n + 1)])
    cells = np.array(cells, dtype=PETSc.IntType)
    if uw.mpi.rank:
        coords, cells = np.empty((0, 2)), np.empty((0, 3), dtype=PETSc.IntType)
    dm = PETSc.DMPlex().createFromCellList(2, cells, coords, interpolate=True,
                                         comm=PETSc.COMM_WORLD)
    section, local_coords = dm.getCoordinateSection(), dm.getCoordinatesLocal().array
    vs, ve = dm.getDepthStratum(0)
    fs, fe = dm.getHeightStratum(1)
    for boundary in _Boundaries:
        dm.createLabel(boundary.name)
    for facet in range(fs, fe):
        vertices = [p for p in dm.getTransitiveClosure(facet)[0] if vs <= p < ve]
        xyz = np.array([local_coords[section.getOffset(v):section.getOffset(v) + 2]
                        for v in vertices])
        tests = {"Left": np.allclose(xyz[:, 0], 0), "Right": np.allclose(xyz[:, 0], 1),
                 "Bottom": np.allclose(xyz[:, 1], 0), "Top": np.allclose(xyz[:, 1], 1),
                 "Vertical": np.allclose(xyz[:, 0], 0.5),
                 "Horizontal": np.allclose(xyz[:, 1], 0.5)}
        if finite:
            tests["Horizontal"] = False
            tests["Vertical"] &= xyz[:, 1].min() >= 0.25 and xyz[:, 1].max() <= 0.75
        for name, selected in tests.items():
            if selected:
                dm.getLabel(name).setValue(facet, _Boundaries[name].value)
    return uw.discretisation.Mesh(dm, simplex=True, degree=1, qdegree=3,
                                  boundaries=_Boundaries, verbose=False)


def _stokes(mesh):
    u = uw.discretisation.MeshVariable("SplitVelocity", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("SplitRegularPressure", mesh, 1, degree=1)
    return uw.systems.Stokes(mesh, velocityField=u, pressureField=p)


def _check_lift(mesh, split, jumps):
    """Independent facet endpoint traces from the DG polynomial coefficients."""
    dm = mesh.dm
    cstart, cend = dm.getHeightStratum(0)
    rows = mesh._cell_node_indices(1, False).reshape(-1, mesh.dim + 1)
    coefficients = {}
    for c in range(cstart, cend):
        xyz = np.asarray(split.pressure_jump_field.coords)[rows[c - cstart]]
        coefficients[c] = np.linalg.solve(np.column_stack((np.ones(mesh.dim + 1), xyz)),
                                         split.pressure_jump_field.array[rows[c - cstart], 0, 0])
    fs, fe = dm.getHeightStratum(1)
    xyz = dm.getCoordinatesLocal().array
    cs = dm.getCoordinateSection()
    vs, ve = dm.getDepthStratum(0)
    maximum = 0.0
    for face in range(fs, fe):
        support = dm.getSupport(face)
        if len(support) != 2:
            continue  # Partition seams are covered by the distributed PDE test.
        points = [v for v in dm.getTransitiveClosure(face)[0] if vs <= v < ve]
        coordinates = np.array([xyz[cs.getOffset(v):cs.getOffset(v) + mesh.dim] for v in points])
        design = np.column_stack((np.ones(len(points)), coordinates))
        expected = np.zeros(len(points))
        for name, (jump, axis) in jumps.items():
            if dm.getLabel(name).getValue(face) == mesh.boundaries[name].value:
                left = np.mean(np.asarray(split.pressure_jump_field.coords)[rows[support[0] - cstart]], axis=0)
                right = np.mean(np.asarray(split.pressure_jump_field.coords)[rows[support[1] - cstart]], axis=0)
                expected = np.asarray(jump(coordinates)) * np.sign(right[axis] - left[axis])
        error = design @ (coefficients[support[1]] - coefficients[support[0]]) - expected
        maximum = max(maximum, float(np.max(abs(error))))
    assert uw.mpi.comm.allreduce(maximum, op=__import__("mpi4py").MPI.MAX) < 1e-11
    native_integral = uw.maths.Integral(mesh, split.pressure_jump).evaluate()
    assert native_integral == pytest.approx(split.pressure_jump_integral, abs=1e-11)


@pytest.mark.tier_b
def test_analytic_jump_and_numerical_field():
    mesh = _cross_mesh()
    stokes = _stokes(mesh)
    x, y = mesh.X
    samples = uw.discretisation.MeshVariable("JumpData", mesh, 1, degree=1)
    samples.array[:, 0, 0] = 1 + 2 * np.asarray(samples.coords)[:, 1]
    for jump in (1 + 2 * y, samples):
        split = stokes.pressure_split(jumps={"Vertical": jump}, normals={"Vertical": (1, 0)})
        _check_lift(mesh, split, {"Vertical": (lambda points: 1 + 2 * points[:, 1], 0)})
        assert split.checks["interface_facets"] == 8


@pytest.mark.tier_b
def test_intersecting_interfaces_and_callable_data():
    mesh = _cross_mesh()
    split = _stokes(mesh).pressure_split(
        jumps={"Vertical": lambda points: np.full(len(points), 2), "Horizontal": 3},
        normals={"Vertical": (1, 0), "Horizontal": (0, 1)})
    _check_lift(mesh, split, {"Vertical": (lambda points: 2, 0),
                              "Horizontal": (lambda points: 3, 1)})
    assert split.checks["interface_facets"] == 16


@pytest.mark.tier_b
def test_finite_interface_has_compatible_tips():
    mesh = _cross_mesh(finite=True)
    jump = lambda points: (points[:, 1] - 0.25) * (0.75 - points[:, 1])
    split = _stokes(mesh).pressure_split(jumps={"Vertical": jump}, normals={"Vertical": (1, 0)})
    _check_lift(mesh, split, {"Vertical": (jump, 0)})
    assert split.checks["interface_facets"] == 4


@pytest.mark.tier_b
@pytest.mark.parametrize("finite", [False, True])
def test_incompatible_junction_or_tip_is_rejected(finite):
    mesh = _cross_mesh(finite=finite)
    jumps = {"Vertical": 1} if finite else {
        "Vertical": lambda points: np.where(points[:, 1] < 0.5, 1, 2), "Horizontal": 3}
    normals = {name: (1, 0) if name == "Vertical" else (0, 1) for name in jumps}
    # A single-valued vertex sample is necessarily cycle-compatible at a cross.
    # Create incompatible sector data using opposite orientations on one branch.
    if not finite:
        normals["Vertical"] = lambda points: np.column_stack(
            (np.where(points[:, 1] < 0.5, 1, -1), np.zeros(len(points))))
    with pytest.raises(ValueError, match="Incompatible prescribed pressure jumps"):
        _stokes(mesh).pressure_split(jumps=jumps, normals=normals)


@pytest.mark.tier_b
def test_manufactured_stokes_with_two_pressure_jumps():
    mesh = _cross_mesh()
    stokes = _stokes(mesh)
    x, y = mesh.X
    split = stokes.pressure_split(jumps={"Vertical": 2, "Horizontal": 3},
                                  normals={"Vertical": (1, 0), "Horizontal": (0, 1)})
    velocity = sp.Matrix([y, 0])
    pressure = x + 2 * y + sp.Piecewise((2, x > sp.Rational(1, 2)), (0, True)) \
               + sp.Piecewise((3, y > sp.Rational(1, 2)), (0, True))
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.viscosity = 1
    stokes.bodyforce = sp.Matrix([1, 2]) + split.regular_force.sym
    for boundary in ("Left", "Right", "Top", "Bottom"):
        stokes.add_essential_bc(velocity, boundary)
    stokes.tolerance = 1e-10
    stokes.petsc_options["snes_type"] = "ksponly"
    stokes.petsc_options["ksp_type"] = "preonly"
    stokes.petsc_options["pc_type"] = "lu"
    stokes.petsc_options["pc_factor_mat_solver_type"] = "mumps"
    stokes.solve(verbose=False, debug=False)
    error = split.physical_pressure - pressure
    mean = uw.maths.Integral(mesh, error).evaluate()  # Unit-area domain.
    pressure_error = uw.maths.Integral(mesh, (error - mean) ** 2).evaluate()
    velocity_error = uw.maths.Integral(mesh, (stokes.u.sym - velocity).dot(stokes.u.sym - velocity)).evaluate()
    assert np.sqrt(abs(pressure_error)) < 1e-8
    assert np.sqrt(abs(velocity_error)) < 1e-9


@pytest.mark.tier_b
def test_tetrahedral_prescribed_jump():
    mesh = uw.meshing.BoxInternalBoundary(minCoords=(0, 0, 0), maxCoords=(1, 1, 1),
                                         zintCoord=0.5, cellSize=0.35, degree=1, qdegree=3,
                                         simplex=True)
    split = _stokes(mesh).pressure_split(jumps={"Internal": 2}, normals={"Internal": (0, 0, 1)})
    assert split.checks["interface_facets"] > 0
    _check_lift(mesh, split, {"Internal": (lambda points: 2, 2)})
