"""Independent polynomial and flux contracts for zero-slip circle correction.

These controls require no Stokes solve or analytical benchmark input. They test
quadratic continuation and the one global compatibility constraint directly.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import underworld3 as uw

pytestmark = pytest.mark.level_1


def _ring(count, radius, graded=False):
    spacing = 1 + (0.6*np.sin(np.arange(count)*0.73) if graded else np.zeros(count))
    theta = 2*np.pi*np.r_[0, np.cumsum(spacing[:-1])]/spacing.sum()
    vertices = radius*np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    edges = np.stack((vertices, np.roll(vertices, -1, axis=0)), axis=1)
    nodes = np.concatenate((vertices, edges.mean(axis=1)))
    index = np.stack((np.arange(count), (np.arange(count)+1)%count,
                      np.arange(count)+count), axis=-1)
    return SimpleNamespace(edges=edges, nodes=nodes, node_index=index,
                           values=np.zeros_like(nodes))


def _cells(wall, sign):
    midpoint = wall.edges.mean(axis=1)
    radius = np.linalg.norm(wall.edges[:, 0], axis=-1)
    interior = midpoint/np.linalg.norm(midpoint, axis=-1)[:, None]*(radius*(1-sign*.18))[:, None]
    vertices = np.concatenate((wall.edges, interior[:, None]), axis=1)
    nodes = np.concatenate((vertices, (vertices[:, [0, 1, 2]]+vertices[:, [1, 2, 0]])/2), axis=1)
    return vertices, nodes


def _quadratic(points):
    x, y = points[..., 0], points[..., 1]
    return np.stack((.7+x*x+2*x*y+3*y, -.2-x*x-2*x*y-y*y+2*x), axis=-1)


def _flux_reference(edges, edge_values, sign):
    """Exact Simpson integral of a vector quadratic against constant facet normal."""
    tangent = edges[:, 1]-edges[:, 0]
    length = np.linalg.norm(tangent, axis=-1)
    normal = np.stack((tangent[:, 1], -tangent[:, 0]), axis=-1)/length[:, None]
    normal *= (sign*np.sign(np.einsum("ei,ei->e", normal, edges.mean(axis=1))))[:, None]
    integral = np.einsum("q,eqi->ei", np.array([1., 1., 4.])/6, edge_values)
    return float(np.sum(length*np.einsum("ei,ei->e", integral, normal)))


@pytest.mark.tier_b
def test_quadratic_continuation_matches_global_cartesian_polynomial_outside_cells():
    from underworld3.utilities._cylindrical_dirichlet import _p2_velocity

    vertices = np.array([[[1.1, .1], [1.4, .1], [1.2, .4]],
                         [[1.8, -.2], [2.2, -.2], [1.9, .3]]])
    nodes = np.concatenate((vertices, (vertices[:, [0, 1, 2]]+vertices[:, [1, 2, 0]])/2), axis=1)
    values = _quadratic(nodes)
    original = values.copy()
    points = np.array([[[1.22, -.1], [1.8, .7]], [[2.22, .2], [1.6, -.4]]])
    np.testing.assert_allclose(_p2_velocity(vertices, values, points), _quadratic(points),
                               rtol=2e-13, atol=2e-13)
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("radius,sign", [(1.22, -1), (2.22, 1)])
@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.tier_b
def test_zero_slip_datum_is_exact_quadratic_circle_continuation(radius, sign, graded):
    from underworld3.utilities._cylindrical_dirichlet import _zero_slip_datum

    wall = _ring(17, radius, graded)
    vertices, nodes = _cells(wall, sign)
    edge_nodes = wall.nodes[wall.node_index]
    values = _quadratic(nodes)
    original = values.copy()
    actual = _zero_slip_datum(vertices, values, edge_nodes, radius)
    physical = radius*edge_nodes/np.linalg.norm(edge_nodes, axis=-1)[..., None]
    expected = _quadratic(edge_nodes)-_quadratic(physical)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-13)
    np.testing.assert_allclose(actual[:, :2], 0, rtol=0, atol=3e-13)
    np.testing.assert_array_equal(values, original)
    # An independent global quadratic vanishing on this circle has precisely
    # its chord value as the shifted datum; no analytical Stokes field is used.
    vanishing = (radius**2-np.sum(nodes**2, axis=-1))[..., None]*np.array([.4, -.7])
    exact = (radius**2-np.sum(edge_nodes**2, axis=-1))[..., None]*np.array([.4, -.7])
    np.testing.assert_allclose(_zero_slip_datum(vertices, vanishing, edge_nodes, radius),
                               exact, rtol=0, atol=3e-13)


@pytest.mark.tier_b
def test_translation_and_rigid_rotation_have_known_shifted_data():
    from underworld3.utilities._cylindrical_dirichlet import _zero_slip_datum

    wall = _ring(19, 1.7, True)
    vertices, nodes = _cells(wall, 1)
    edge_nodes = wall.nodes[wall.node_index]
    translation = np.broadcast_to([2.3, -.7], nodes.shape).copy()
    np.testing.assert_allclose(_zero_slip_datum(vertices, translation, edge_nodes, 1.7),
                               0, rtol=0, atol=2e-13)
    rotation = np.array([[0., -1.3], [1.3, 0.]])
    physical = 1.7*edge_nodes/np.linalg.norm(edge_nodes, axis=-1)[..., None]
    expected = (edge_nodes-physical)@rotation.T
    np.testing.assert_allclose(_zero_slip_datum(vertices, nodes@rotation.T+translation,
                                              edge_nodes, 1.7), expected, rtol=0, atol=3e-13)


@pytest.mark.tier_b
def test_datum_respects_edge_reversal_rotation_scaling_and_cell_permutation():
    from underworld3.utilities._cylindrical_dirichlet import _zero_slip_datum

    wall = _ring(23, 2.22, True)
    vertices, nodes = _cells(wall, 1)
    edge_nodes = wall.nodes[wall.node_index]
    values = _quadratic(nodes)
    expected = _zero_slip_datum(vertices, values, edge_nodes, 2.22)
    order = np.random.default_rng(31).permutation(len(vertices))
    reversed_cells = [1, 0, 2]
    reversed_p2 = [1, 0, 2, 3, 5, 4]
    actual = _zero_slip_datum(vertices[order][:, reversed_cells], values[order][:, reversed_p2],
                              edge_nodes[order][:, [1, 0, 2]], 2.22)
    np.testing.assert_allclose(actual, expected[order][:, [1, 0, 2]], rtol=0, atol=4e-13)
    angle = .391
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    actual = _zero_slip_datum(vertices@rotation.T, values@rotation.T,
                              edge_nodes@rotation.T, 2.22)
    np.testing.assert_allclose(actual, expected@rotation.T, rtol=0, atol=4e-13)
    actual = _zero_slip_datum(vertices*1.7, values, edge_nodes*1.7, 2.22*1.7)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=4e-13)


@pytest.mark.tier_b
def test_exact_facet_flux_obeys_polygon_divergence_theorem():
    from underworld3.utilities._cylindrical_dirichlet import _p2_facet_flux

    actual = 0.
    expected = 0.
    for count, radius, sign in [(21, 1.22, -1), (29, 2.22, 1)]:
        wall = _ring(count, radius, True)
        x, y = wall.nodes[..., 0], wall.nodes[..., 1]
        values = np.stack((x*x+x, y*y+y), axis=-1)[wall.node_index]
        actual += _p2_facet_flux(wall.edges, values, sign)
        np.testing.assert_allclose(_p2_facet_flux(wall.edges, values, sign),
                                   _flux_reference(wall.edges, values, sign), rtol=2e-13, atol=2e-13)
        a, b = wall.edges[:, 0], wall.edges[:, 1]
        cross = a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
        area = cross.sum()/2
        moment = np.sum((a+b)*cross[:, None], axis=0)/6
        expected += sign*(2*area+2*moment.sum())  # div(x²+x,y²+y)=2x+2y+2.
        order = np.random.default_rng(7).permutation(count)
        np.testing.assert_allclose(_p2_facet_flux(wall.edges[order, ::-1], values[order][:, [1, 0, 2]], sign),
                                   _flux_reference(wall.edges, values, sign), rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


@pytest.mark.tier_b
def test_joint_flux_projection_is_one_global_midpoint_constraint_not_wallwise():
    from underworld3.utilities._cylindrical_dirichlet import _project_joint_flux

    walls = dict(inner=_ring(19, 1.22, True), outer=_ring(27, 2.22, True))
    for name, wall in walls.items():
        wall.sign = -1 if name == "inner" else 1
        wall.values[len(wall.edges):] = (.7 if name == "inner" else -.2)*wall.nodes[len(wall.edges):]
    originals = {name: wall.values.copy() for name, wall in walls.items()}
    raw = sum(_flux_reference(w.edges, w.values[w.node_index], w.sign) for w in walls.values())
    length = sum(np.linalg.norm(w.edges[:, 1]-w.edges[:, 0], axis=-1).sum() for w in walls.values())
    coefficient = raw/(2*length/3)
    check = _project_joint_flux(walls)
    np.testing.assert_allclose(check["raw_total_flux"], raw, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(check["removed_midpoint_normal_value"], coefficient, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(check["l2_correction_norm"], abs(coefficient)*np.sqrt(8*length/15),
                               rtol=2e-13, atol=2e-13)
    for name, wall in walls.items():
        count = len(wall.edges)
        expected = originals[name].copy()
        midpoint = wall.nodes[count:]
        expected[count:] -= coefficient*wall.sign*midpoint/np.linalg.norm(midpoint, axis=-1)[:, None]
        np.testing.assert_allclose(wall.values, expected, rtol=0, atol=2e-13)
        np.testing.assert_array_equal(wall.values[:count], 0)
        assert abs(_flux_reference(wall.edges, wall.values[wall.node_index], wall.sign)) > 1e-4
    corrected = sum(_flux_reference(w.edges, w.values[w.node_index], w.sign) for w in walls.values())
    np.testing.assert_allclose(corrected, 0, rtol=0, atol=2e-13)
    np.testing.assert_allclose(check["corrected_total_flux"], 0, rtol=0, atol=2e-13)


@pytest.mark.tier_b
def test_zero_shift_and_zero_joint_flux_need_no_field_correction():
    from underworld3.utilities._cylindrical_dirichlet import _project_joint_flux, _zero_slip_datum

    wall = _ring(16, 1.22)
    vertices, nodes = _cells(wall, -1)
    # Nodes placed on the true circle have exactly zero displacement.
    points = wall.nodes[wall.node_index]
    points = 1.22*points/np.linalg.norm(points, axis=-1)[..., None]
    np.testing.assert_allclose(_zero_slip_datum(vertices, _quadratic(nodes), points, 1.22),
                               0, rtol=0, atol=3e-13)
    wall.sign = -1
    original = wall.values.copy()
    check = _project_joint_flux(dict(inner=wall))
    np.testing.assert_array_equal(wall.values, original)
    assert check["raw_total_flux"] == 0 and check["corrected_total_flux"] == 0


@pytest.mark.parametrize("layout", ["empty_other_ranks", "partitioned_with_ghosts"])
@pytest.mark.tier_b
def test_boundary_adapter_collects_owned_coefficients_and_preserves_velocity(layout, monkeypatch):
    from underworld3.utilities import _cylindrical_dirichlet as module
    _ZeroSlipWall = module._ZeroSlipWall

    radius = 2.22
    ring = _ring(23, radius, True)
    # A valid polygon triangulation with three circle vertices in each cell
    # includes internal diagonals joining circle vertices. Only labelled exterior
    # edges belong to this wall; a radius-only selector would incorrectly fail.
    polygon = ring.edges[:, 0]
    all_vertices = np.stack([np.stack((polygon[0], polygon[j], polygon[j+1]))
                             for j in range(1, len(polygon)-1)])
    all_nodes = np.concatenate((all_vertices,
        (all_vertices[:, [0, 1, 2]]+all_vertices[:, [1, 2, 0]])/2), axis=1)
    if layout == "empty_other_ranks":
        selected = np.arange(len(all_vertices)) if uw.mpi.rank == 0 else np.array([], dtype=int)
        owned = np.ones(len(selected), dtype=bool)
    else:
        chosen = np.arange(len(all_vertices))[uw.mpi.rank::uw.mpi.size]
        ghosts = np.setdiff1d(np.arange(3), chosen)
        selected = np.r_[chosen, ghosts]
        owned = np.r_[np.ones(len(chosen), dtype=bool), np.zeros(len(ghosts), dtype=bool)]
    local_vertices, local_nodes = all_vertices[selected], all_nodes[selected]
    coordinates, inverse = np.unique(local_nodes.reshape(-1, 2), axis=0, return_inverse=True)
    rows = inverse.reshape(-1, 6)
    # Deliberately use a noncanonical local P2 ordering. The adapter must discover
    # it geometrically while excluding every ghost cell from the gather.
    rows = rows[:, [3, 1, 5, 2, 0, 4]]
    p1_coordinates, p1_inverse = np.unique(local_vertices.reshape(-1, 2), axis=0, return_inverse=True)
    mesh = SimpleNamespace(dim=2, cdim=2, isSimplex=True, boundaries=[SimpleNamespace(name="Outer")],
        _cell_node_indices=lambda degree, continuous: p1_inverse.reshape(-1, 3) if degree == 1 else rows,
        _get_coords_for_basis=lambda degree, continuous: p1_coordinates,
        _get_owned_cells_mask=lambda: owned)
    velocity = SimpleNamespace(mesh=mesh, degree=2, continuous=True,
                               coords=coordinates, array=_quadratic(coordinates)[:, None, :])
    original = np.asarray(velocity.array).copy()

    def auxiliary(*args, **kwargs):
        return SimpleNamespace(mesh=mesh, coords=coordinates,
                               array=np.full_like(velocity.array, 777.), sym=None)

    # PETSc label extraction has a separate real-mesh test; here supply its
    # known fixture boundary so the owned/ghost coefficient adapter is isolated.
    monkeypatch.setattr(module, "_labelled_edge_keys", lambda mesh, name, radius, tolerance:
                        {module._edge_key(edge, tolerance) for edge in ring.edges})
    monkeypatch.setattr(uw.discretisation, "MeshVariable", auxiliary)
    monkeypatch.setattr(uw, "is_nondimensional_scaling_active", lambda: False)
    monkeypatch.setattr(uw, "get_default_model", lambda: SimpleNamespace(has_units_active=lambda: False))
    wall = _ZeroSlipWall(SimpleNamespace(instance_number=1, mesh=mesh, u=velocity, p=SimpleNamespace(degree=1, continuous=True)), "Outer", radius, 1, 12)
    assert len(wall.edges) == len(ring.edges)
    np.testing.assert_allclose(wall.cell_values(), _quadratic(np.concatenate((wall.vertices,
        (wall.vertices[:, [0, 1, 2]]+wall.vertices[:, [1, 2, 0]])/2), axis=1)), rtol=0, atol=1e-13)
    wall.update()
    circle = radius*wall.nodes/np.linalg.norm(wall.nodes, axis=-1)[:, None]
    expected = _quadratic(wall.nodes)-_quadratic(circle)
    is_vertex = np.isclose(np.linalg.norm(wall.nodes, axis=-1), radius, rtol=0, atol=1e-12)
    expected[is_vertex] = 0
    np.testing.assert_allclose(wall.values, expected, rtol=0, atol=5e-13)
    wall.write()
    np.testing.assert_allclose(wall.field.array[wall.local_boundary, 0],
                               expected[wall.local_index[wall.local_boundary]], rtol=0, atol=5e-13)
    np.testing.assert_array_equal(wall.field.array[~wall.local_boundary, 0], 777.)
    np.testing.assert_array_equal(velocity.array, original)
    check = wall.audit()
    theta = 2*np.pi*np.arange(32)/32
    exact = _quadratic(radius*np.stack((np.cos(theta), np.sin(theta)), axis=-1))
    rms = np.sqrt(np.mean(np.sum(exact**2, axis=-1)))
    np.testing.assert_allclose(check["true_circle_velocity_rms"], rms, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(check["true_circle_measure"], 2*np.pi*radius, rtol=3e-13, atol=3e-13)
    np.testing.assert_array_equal(velocity.array, original)


@pytest.mark.tier_b
def test_real_mesh_labelled_walls_continue_exact_polynomial_without_a_solve():
    from underworld3.utilities._cylindrical_dirichlet import _ZeroSlipWall, _project_joint_flux

    mesh = uw.meshing.Annulus(radiusInner=.5, radiusOuter=1., cellSize=1/8, qdegree=2)
    velocity = uw.discretisation.MeshVariable("DirichletPolynomialWitness", mesh, 2, degree=2, continuous=True)
    velocity.array[:, 0, :] = _quadratic(np.asarray(velocity.coords))
    original = np.asarray(velocity.array).copy()
    stokes = SimpleNamespace(instance_number=1, mesh=mesh, u=velocity, p=SimpleNamespace(degree=1, continuous=True))
    walls = {name: _ZeroSlipWall(stokes, name, radius, sign, 12)
             for name, radius, sign in [("Lower", .5, -1), ("Upper", 1., 1)]}
    for wall in walls.values():
        wall.update()
        np.testing.assert_allclose(wall.cell_values(), _quadratic(np.concatenate((wall.vertices,
            (wall.vertices[:, [0, 1, 2]]+wall.vertices[:, [1, 2, 0]])/2), axis=1)), rtol=0, atol=3e-13)
        check = wall.audit()
        theta = 2*np.pi*np.arange(32)/32
        exact = _quadratic(wall.radius*np.stack((np.cos(theta), np.sin(theta)), axis=-1))
        np.testing.assert_allclose(check["true_circle_velocity_rms"],
            np.sqrt(np.mean(np.sum(exact**2, axis=-1))), rtol=4e-13, atol=4e-13)
    projection = _project_joint_flux(walls)
    assert abs(projection["corrected_total_flux"]) < 2e-13
    for wall in walls.values():
        wall.write()
        np.testing.assert_allclose(wall.field.array[wall.local_boundary, 0],
            wall.values[wall.local_index[wall.local_boundary]], rtol=0, atol=3e-13)
    np.testing.assert_array_equal(velocity.array, original)


@pytest.mark.parametrize("degree,continuous", [(0, False), (1, False), (2, True)])
@pytest.mark.tier_b
def test_adapter_rejects_unsupported_regular_pressure_before_collectives(degree, continuous):
    from underworld3.utilities._cylindrical_dirichlet import _ZeroSlipWall

    mesh = SimpleNamespace(dim=2, cdim=2, isSimplex=True)
    stokes = SimpleNamespace(mesh=mesh, u=SimpleNamespace(degree=2, continuous=True),
                             p=SimpleNamespace(degree=degree, continuous=continuous))
    with pytest.raises(ValueError, match="continuous affine P1"):
        _ZeroSlipWall(stokes, "Outer", 1., 1, 12)


@pytest.mark.tier_b
def test_lagged_solve_requires_the_standard_pressure_nullspace():
    from underworld3.utilities._cylindrical_dirichlet import _solve_cylindrical_zero_slip_updates

    # Fail before any field modification, collective or solve; full Dirichlet
    # Stokes has the standard constant-pressure kernel.
    stokes = SimpleNamespace(petsc_use_pressure_nullspace=False)
    with pytest.raises(ValueError, match="petsc_use_pressure_nullspace=True"):
        _solve_cylindrical_zero_slip_updates(stokes, walls={}, updates=1)


@pytest.mark.parametrize("local_stratum", ["null_petsc_is", "false_sentinel", "valid_empty_is"])
@pytest.mark.tier_b
def test_absent_local_boundary_is_not_dereferenced_before_global_edge_gather(local_stratum, monkeypatch):
    """PETSc.IS() has a false null handle, distinct from a valid zero-length IS.

    The Gadi MPI16 gate first exposed the real null-handle crash. Never inspect
    its .indices here; a separate false sentinel safely makes any such access
    an assertion failure instead of triggering the native segmentation fault.
    """
    from petsc4py import PETSc
    from underworld3.utilities import _cylindrical_dirichlet as module

    class FalseHandle:
        def __bool__(self):
            return False

        @property
        def indices(self):
            raise AssertionError("A false PETSc handle must not be dereferenced.")

        def destroy(self):
            raise AssertionError("A false PETSc handle must not be destroyed.")

    if local_stratum == "null_petsc_is":
        facets = PETSc.IS()
        assert not facets
    elif local_stratum == "valid_empty_is":
        facets = PETSc.IS().createGeneral(np.empty(0, dtype=PETSc.IntType), comm=PETSc.COMM_SELF)
        assert bool(facets)  # Handle validity, not the length of its index array.
    else:
        facets = FalseHandle()
    radius = 1.7
    remote_edges = _ring(11, radius, True).edges

    class OtherRankOwnsWall:
        def allreduce(self, value, op):
            assert value is False
            return False

        def allgather(self, local):
            np.testing.assert_array_equal(local.shape, (0, 2, 2))
            return [local, remote_edges.copy()]

    empty_dm = SimpleNamespace(
        getDepthStratum=lambda depth: (0, 0),
        getCoordinateSection=lambda: SimpleNamespace(),
        getCoordinatesLocal=lambda: SimpleNamespace(array=np.empty(0)),
    )
    monkeypatch.setattr(module, "_boundary_stratum_is", lambda dm, mesh, boundary: facets)
    monkeypatch.setattr(uw.mpi, "comm", OtherRankOwnsWall())
    tolerance = 1e-10*radius
    actual = module._labelled_edge_keys(SimpleNamespace(dm=empty_dm), "Outer", radius, tolerance)
    assert actual == {module._edge_key(edge, tolerance) for edge in remote_edges}
    if local_stratum == "valid_empty_is":
        assert not facets  # The valid empty IS is released; null handles are untouched.
