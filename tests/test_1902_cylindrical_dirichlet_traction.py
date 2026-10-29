"""Hard manufactured-stress contracts for full Dirichlet circular CBF recovery.

No Stokes solve or benchmark reference is required. Constant pressure fixes
the sign/gauge; affine Cartesian stress fixes every transferred vector moment.
"""

from enum import Enum
from types import SimpleNamespace

import numpy as np
import pytest
from petsc4py import PETSc
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
import sympy as sp
import underworld3 as uw
from underworld3.coordinates import CoordinateSystemType

pytestmark = pytest.mark.level_1


def _ring(count, radius, graded):
    spacing = 1+(.55*np.sin(np.arange(count)*.61) if graded else np.zeros(count))
    theta = 2*np.pi*np.r_[0, np.cumsum(spacing[:-1])]/spacing.sum()
    nodes = radius*np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    return np.stack((nodes, np.roll(nodes, -1, axis=0)), axis=1)


def _stress(points):
    x, y = points[..., 0], points[..., 1]
    pressure = .3+.2*x-.4*y
    result = np.empty(points.shape[:-1]+(2, 2))
    result[..., 0, 0], result[..., 1, 1] = 2*x-pressure, -2*x-pressure
    result[..., 0, 1] = result[..., 1, 0] = -y
    return result


def _polygon_cbf(rule, tensor):
    """Independent boundary Galerkin solve for a known Cartesian stress."""
    index, phi = rule["index"], rule["shape"]
    size = len(rule["nodes"])
    local_mass = np.einsum("eq,qi,qj->eij", rule["polygon_weight"], phi, phi)
    row = np.broadcast_to(index[:, :, None], local_mass.shape).ravel()
    col = np.broadcast_to(index[:, None, :], local_mass.shape).ravel()
    mass = coo_matrix((local_mass.ravel(), (row, col)), shape=(size, size)).tocsr()
    local_load = np.einsum("eq,qi,eqcd,ed->eic", rule["polygon_weight"], phi, tensor, rule["polygon_normal"])
    load = np.zeros((size, 2))
    np.add.at(load, index.ravel(), local_load.reshape(-1, 2))
    return spsolve(mass, load), load


@pytest.mark.tier_b
@pytest.mark.parametrize("radius,sign", [(1.22, -1), (2.22, 1)])
@pytest.mark.parametrize("graded", [False, True])
def test_constant_pressure_keeps_absolute_gauge_and_zero_shear(radius, sign, graded):
    from underworld3.utilities._cylindrical_dirichlet_traction import _traction_rule, _transfer_circle_traction

    rule = _traction_rule(_ring(29, radius, graded), radius, sign, 32)
    stress = np.broadcast_to(-.7*np.eye(2), rule["points"].shape[:-1]+(2, 2))
    cbf, _ = _polygon_cbf(rule, stress)
    actual, checks = _transfer_circle_traction(rule, cbf, stress, stress)
    np.testing.assert_allclose(actual[:, 0], -.7, rtol=0, atol=3e-13)
    np.testing.assert_allclose(actual[:, 1], 0, rtol=0, atol=3e-13)
    assert checks["maximum_cartesian_moment_residual"] < 3e-14
    assert checks["cartesian_resultant_residual"] < 3e-13


@pytest.mark.tier_b
@pytest.mark.parametrize("sign", [-1, 1])
def test_affine_stress_preserves_every_circle_vector_moment_and_quadrature(sign):
    from underworld3.utilities._cylindrical_dirichlet_traction import _traction_rule, _transfer_circle_traction

    edges, radius = _ring(23, 1.7, True), 1.7
    traces = []
    for quadrature in (24, 48):
        rule = _traction_rule(edges, radius, sign, quadrature)
        cbf, _ = _polygon_cbf(rule, _stress(rule["points"]))
        actual, checks = _transfer_circle_traction(rule, cbf, _stress(rule["points"]), _stress(rule["circle_points"]))
        sampled = np.einsum("qi,eik->eqk", rule["shape"], actual[rule["index"]])
        traction = np.einsum("eqk,eqck->eqc", sampled, rule["frame"])
        exact = np.einsum("eqcd,eqd->eqc", _stress(rule["circle_points"]), rule["normal"])
        defect = np.zeros((len(actual), 2))
        local = np.einsum("eq,qi,eqc->eic", rule["circle_weight"], rule["shape"], traction-exact)
        np.add.at(defect, rule["index"].ravel(), local.reshape(-1, 2))
        np.testing.assert_allclose(defect, 0, rtol=0, atol=2e-13)
        expected_resultant = sign*np.pi*radius**2*np.array([.8, .4])
        np.testing.assert_allclose(checks["cartesian_resultant"], expected_resultant, rtol=3e-13, atol=3e-13)
        traces.append(actual)
    np.testing.assert_allclose(traces[0], traces[1], rtol=3e-13, atol=3e-13)


@pytest.mark.tier_b
def test_regular_pressure_reaction_needs_regular_polygon_stress():
    from underworld3.utilities._cylindrical_dirichlet_traction import _traction_rule, _transfer_circle_traction

    radius = 2.22
    rule = _traction_rule(_ring(27, radius, True), radius, 1, 32)
    regular = _stress(rule["points"])
    cbf, _ = _polygon_cbf(rule, regular)
    x = rule["circle_points"]
    lift_circle = (radius-np.linalg.norm(x, axis=-1))*.4
    physical_circle = _stress(x)-lift_circle[..., None, None]*np.eye(2)
    actual, _ = _transfer_circle_traction(rule, cbf, regular, physical_circle)
    reference, _ = _transfer_circle_traction(rule, cbf, regular, _stress(x))
    np.testing.assert_allclose(actual, reference, rtol=0, atol=2e-13)
    # The prescribed lift vanishes at the true wall, but not at chord points.
    assert np.max(radius-np.linalg.norm(rule["points"], axis=-1)) > 1e-3


@pytest.mark.tier_b
def test_constant_pressure_uses_actual_facet_normals_with_tolerated_vertex_roundoff():
    from underworld3.utilities._cylindrical_dirichlet_traction import _traction_rule, _transfer_circle_traction

    radius = 1.7
    edges = _ring(31, radius, True)
    vertices = edges[:, 0]*(1+4e-11*np.sin(np.arange(len(edges))))[:, None]
    edges = np.stack((vertices, np.roll(vertices, -1, axis=0)), axis=1)
    rule = _traction_rule(edges, radius, 1, 32)
    stress = np.broadcast_to(-.7*np.eye(2), rule["points"].shape[:-1]+(2, 2))
    cbf, _ = _polygon_cbf(rule, stress)
    actual, _ = _transfer_circle_traction(rule, cbf, stress, stress)
    np.testing.assert_allclose(actual[:, 0], -.7, rtol=0, atol=3e-13)
    np.testing.assert_allclose(actual[:, 1], 0, rtol=0, atol=3e-13)


@pytest.mark.tier_b
def test_reconstruction_preserves_mean_and_combined_cartesian_resultants():
    from underworld3.utilities._cylindrical_dirichlet_traction import _traction_rule, _trace_change_checks
    from underworld3.utilities._cylindrical_reconstruction import _recover_circular_trace

    edges, radius = _ring(39, 1.7, True), 1.7
    rule = _traction_rule(edges, radius, -1, 48)
    theta = np.arctan2(rule["nodes"][:, 1], rule["nodes"][:, 0])
    source = np.stack((.4+np.cos(3*theta)+.03*np.cos(15*theta),
                       -.2+np.sin(2*theta)+.02*np.sin(17*theta)), axis=-1)
    target = np.stack([_recover_circular_trace(edges, source[:, k], radius, 6, 5, 16)[1]
                       for k in range(2)], axis=-1)
    checks = _trace_change_checks(rule, source, target)
    assert checks["maximum_cartesian_resultant_change"] < 2e-12
    np.testing.assert_allclose(checks["scalar_integral_changes"], 0, rtol=0, atol=2e-12)
    assert checks["maximum_changed_cartesian_nodal_moment"] > 1e-5


@pytest.mark.tier_b
@pytest.mark.parametrize("pressure_profile", ["zero", "affine"])
def test_real_mesh_native_reassembly_preserves_gauge_and_separate_targets(pressure_profile):
    """Exercise labelled facets, essential DOFs and shared MPI nodes without a solve."""
    mesh = uw.meshing.Annulus(radiusInner=.5, radiusOuter=1., cellSize=.18, qdegree=2)
    velocity = uw.discretisation.MeshVariable("TractionWitnessU", mesh, 2, degree=2)
    pressure = uw.discretisation.MeshVariable("TractionWitnessP", mesh, 1, degree=1, continuous=True)
    velocity.array[:, 0, :] = 0
    pressure.array[:, 0, 0] = .7
    stokes = uw.systems.Stokes(mesh, velocityField=velocity, pressureField=pressure)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.viscosity = 1
    stokes.bodyforce = sp.Matrix([0, 0])
    wall = stokes.add_cylindrical_zero_slip_transfer(boundary="Upper", radius=1., sign=1, quadrature=16)
    stokes.add_dirichlet_bc(sp.Matrix([[0, 0]]), "Lower")
    targets = [uw.discretisation.MeshVariable("TractionWitness"+name, mesh, 1, degree=2)
               for name in ("N", "T", "Nr", "Tr")]
    for target in targets:
        target.array[:, 0, 0] = 777
    jump = (sp.Rational(3, 10)+mesh.N.x/5-2*mesh.N.y/5
            if pressure_profile == "affine" else sp.S.Zero)
    checks = stokes.recover_cylindrical_zero_slip_traction(wall=wall,
        normal_field=targets[0], shear_field=targets[1],
        recovered_normal_field=targets[2], recovered_shear_field=targets[3],
        viscosity=1., pressure_jump=jump, degree=6, halfwidth_edges=3, quadrature=24)
    for target, exact in zip(targets, [-.7, 0, -.7, 0]):
        if pressure_profile == "zero":
            np.testing.assert_allclose(target.array[wall.local_boundary, 0, 0], exact, rtol=0, atol=2e-12)
        np.testing.assert_array_equal(target.array[~wall.local_boundary, 0, 0], 777)
    np.testing.assert_array_equal(pressure.array[:, 0, 0], .7)
    np.testing.assert_array_equal(velocity.array[:, 0, :], 0)
    assert checks["reconstruction"]["maximum_cartesian_resultant_change"] < 2e-12
    if pressure_profile == "affine":
        from underworld3.utilities._cylindrical_dirichlet_traction import _traction_rule

        rule = _traction_rule(wall.edges, 1., 1, 48)
        # Gather only trace values; no interior field participates in this audit.
        coords = np.concatenate(uw.mpi.comm.allgather(targets[0].coords[wall.local_boundary]))
        values = np.concatenate(uw.mpi.comm.allgather(np.stack([
            field.array[wall.local_boundary, 0, 0] for field in targets[:2]], axis=-1)))
        distance, index = cKDTree(coords).query(rule["nodes"])
        np.testing.assert_allclose(distance, 0, rtol=0, atol=1e-12)
        sample = np.einsum("qi,eik->eqk", rule["shape"], values[index][rule["index"]])
        traction = np.einsum("eqk,eqck->eqc", sample, rule["frame"])
        points = rule["circle_points"]
        exact_pressure = 1+points[..., 0]/5-2*points[..., 1]/5
        error = traction+exact_pressure[..., None]*rule["normal"]
        defect = np.zeros((len(rule["nodes"]), 2))
        np.add.at(defect, rule["index"].ravel(), np.einsum(
            "eq,qi,eqc->eic", rule["circle_weight"], rule["shape"], error).reshape(-1, 2))
        np.testing.assert_allclose(defect, 0, rtol=0, atol=2e-12)
        np.testing.assert_allclose(checks["cartesian_resultant"], -np.pi*np.array([.2, -.4]),
                                   rtol=2e-12, atol=2e-12)


@pytest.mark.tier_b
def test_pressure_adapter_gathers_boundary_cells_only_with_empty_local_owner(monkeypatch):
    from underworld3.utilities import _cylindrical_dirichlet_traction as module

    class BoundaryOnlyComm:
        def allgather(self, local):
            np.testing.assert_array_equal(local.shape, (0, 3))
            return [local, np.array([[.2, .3, .4]])]

    mesh = SimpleNamespace(_cell_node_indices=lambda degree, continuous: np.empty((0, 3), dtype=int))
    stokes = SimpleNamespace(mesh=mesh, p=SimpleNamespace(array=np.zeros((500, 1, 1))))
    wall = SimpleNamespace(cells=np.empty(0, dtype=int))
    monkeypatch.setattr(uw.mpi, "comm", BoundaryOnlyComm())
    np.testing.assert_array_equal(module._boundary_pressure_values(stokes, wall), [[.2, .3, .4]])


@pytest.mark.tier_b
@pytest.mark.mpi(min_size=4)
@pytest.mark.skipif(uw.mpi.size < 4, reason="Three labelled facets require >=4 ranks to force an empty owner.")
def test_real_three_facet_wall_recovery_with_ranks_owning_no_boundary_cells():
    """A complete real DMPlex wall has fewer facets than MPI ranks by construction."""
    angle = 2*np.pi*np.arange(3)/3
    coordinates = np.concatenate([radius*np.stack((np.cos(angle), np.sin(angle)), axis=-1)
                                   for radius in (.5, .75, 1.)])
    cells = []
    for layer in range(2):
        for i in range(3):
            a, b = 3*layer+i, 3*layer+(i+1)%3
            c, d = 3*(layer+1)+i, 3*(layer+1)+(i+1)%3
            cells.extend(((a, c, d), (a, d, b)))
    dm = PETSc.DMPlex().createFromCellList(2, np.asarray(cells, dtype=PETSc.IntType), coordinates)

    class Boundaries(Enum):
        Lower = 1
        Upper = 2

    for name in ("Lower", "Upper"):
        dm.createLabel(name)
    f0, f1 = dm.getDepthStratum(1)
    v0, v1 = dm.getDepthStratum(0)
    section = dm.getCoordinateSection()
    coords = dm.getCoordinatesLocal().array.reshape(-1, 2)
    for facet in range(f0, f1):
        vertices = [point for point in dm.getTransitiveClosure(facet)[0] if v0 <= point < v1]
        points = coords[[section.getOffset(point)//2 for point in vertices]]
        for name, radius, tag in (("Lower", .5, 1), ("Upper", 1., 2)):
            if np.allclose(np.linalg.norm(points, axis=-1), radius, rtol=0, atol=1e-12):
                dm.setLabelValue(name, facet, tag)
    mesh = uw.discretisation.Mesh(dm, degree=1, qdegree=2, simplex=True,
        boundaries=Boundaries, markVertices=True, coordinate_system_type=CoordinateSystemType.CYLINDRICAL2D)
    velocity = uw.discretisation.MeshVariable("EmptyTractionU", mesh, 2, degree=2)
    pressure = uw.discretisation.MeshVariable("EmptyTractionP", mesh, 1, degree=1)
    velocity.array[:, 0, :] = 0
    pressure.array[:, 0, 0] = .7
    stokes = uw.systems.Stokes(mesh, velocityField=velocity, pressureField=pressure)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.viscosity = 1
    stokes.bodyforce = sp.Matrix([0, 0])
    wall = stokes.add_cylindrical_zero_slip_transfer(boundary="Lower", radius=.5, sign=-1, quadrature=48)
    stokes.add_dirichlet_bc(sp.Matrix([[0, 0]]), "Upper")
    counts = uw.mpi.comm.allgather(len(wall.cells))
    assert sum(counts) == 3 and min(counts) == 0
    fields = [uw.discretisation.MeshVariable("EmptyTraction"+str(k), mesh, 1, degree=2) for k in range(4)]
    checks = stokes.recover_cylindrical_zero_slip_traction(wall=wall,
        normal_field=fields[0], shear_field=fields[1], recovered_normal_field=fields[2],
        recovered_shear_field=fields[3], viscosity=1., pressure_jump=sp.S.Zero,
        degree=6, halfwidth_edges=.5, quadrature=48)
    for field, value in zip(fields, [-.7, 0, -.7, 0]):
        np.testing.assert_allclose(field.array[wall.local_boundary, 0, 0], value, rtol=0, atol=2e-12)
    assert checks["cartesian_resultant_residual"] < 2e-12
