"""Mesh variables the library makes for an owner are that owner's (#1118 follow-up).

Swarm proxies, index-variable level sets, the level-set interface thickness,
the decoded viscoelastic stress, the surface and fault collections' fields and
the free surface's fields were made under fixed names, so two owners on one
mesh shared one variable: the second overwrote the first's values. Once a
re-declaration that changes a variable refuses (#1118), the same sharing raised
instead. Each helper is now named after its owner, the same on every rank. Each
test puts two owners on one mesh and checks that each keeps a field of its own,
against a closed form where the field has one.
"""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


@pytest.fixture
def mesh():
    return uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.2)


def test_two_swarms_with_a_variable_of_one_name_keep_their_own_proxies(mesh):
    # Proxies of different degree, so a shared proxy cannot pass. Each swarm
    # carries one linear field.
    swarms, fields = [], []
    for axis, degree in ((0, 1), (1, 2)):
        swarm = uw.swarm.Swarm(mesh)
        s = uw.swarm.SwarmVariable("phase", swarm, 1, proxy_degree=degree)
        swarm.populate(fill_param=3)
        s.array[:, 0, 0] = swarm._particle_coordinates.array[:, 0, axis]
        swarms.append(swarm)
        fields.append((s, axis))
    assert fields[0][0]._meshVar._base_var is not fields[1][0]._meshVar._base_var
    # The first swarm keeps the plain name its checkpoints carry.
    assert fields[0][0]._meshVar.clean_name == "proxy_phase"
    assert fields[1][0]._meshVar.clean_name == f"proxy_{swarms[1].instance_number}_phase"
    for s, axis in fields:
        s._update_proxy_if_stale()
        proxy = s._meshVar
        error = np.abs(np.asarray(proxy.array).ravel() - np.asarray(proxy.coords)[:, axis]).max()
        assert error < 1.0e-12


def test_two_index_variables_of_one_name_keep_their_own_level_sets(mesh):
    # Nodal level sets of different degree (the default location, integration
    # points, has one layout for every degree).
    swarms = (uw.swarm.Swarm(mesh), uw.swarm.Swarm(mesh))      # held: a variable needs its swarm
    first = uw.swarm.IndexSwarmVariable("M", swarms[0], indices=2, proxy_degree=1,
                                        proxy_location="nodes")
    second = uw.swarm.IndexSwarmVariable("M", swarms[1], indices=2, proxy_degree=2,
                                         proxy_location="nodes")
    for a, b in zip(first._meshLevelSetVars, second._meshLevelSetVars):
        assert a._base_var is not b._base_var
    n = swarms[1].instance_number
    assert [v.clean_name for v in second._meshLevelSetVars] == [f"M_{n}_0", f"M_{n}_1"]
    assert [v.degree for v in first._meshLevelSetVars] == [1, 1]
    assert [v.degree for v in second._meshLevelSetVars] == [2, 2]


def test_interface_thickness_is_its_level_sets_own():
    # A regular box: every triangle has area h^2 / 2, so the thickness is the
    # closed form 0.35 V^(1/2) / sqrt(2) everywhere.
    h = 0.2
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=h, regular=True)
    user = uw.discretisation.MeshVariable("epsilon", mesh, 1, degree=2)
    user.array[:, 0, 0] = -1.0
    phi1 = uw.discretisation.MeshVariable("phi_a", mesh, 1, degree=1)
    phi2 = uw.discretisation.MeshVariable("phi_b", mesh, 1, degree=2)
    eps1 = uw.systems.level_set.interface_thickness(mesh, phi1)
    eps2 = uw.systems.level_set.interface_thickness(mesh, phi2)
    assert (eps1.degree, eps2.degree) == (1, 2)
    assert np.all(np.asarray(user.array) == -1.0)            # the user's field is untouched
    expect = 0.35 * np.sqrt(h * h / 2.0) / np.sqrt(2.0)
    for eps in (eps1, eps2):
        np.testing.assert_allclose(np.asarray(eps.array).ravel(), expect, rtol=1e-10)


def test_two_surface_collections_keep_their_own_distance_fields(mesh):
    traces = (np.array([[0.2, 0.2, 0.0], [0.8, 0.2, 0.0]]),
              np.array([[0.2, 0.8, 0.0], [0.8, 0.8, 0.0]]))
    fields = []
    for i, trace in enumerate(traces):
        collection = uw.meshing.SurfaceCollection()
        surface = uw.meshing.Surface(f"s0132_{i}", mesh, trace)
        surface.discretize()
        collection.add(surface)
        fields.append((collection.compute_distance_field(mesh), trace))
    assert fields[0][0]._base_var is not fields[1][0]._base_var
    for var, trace in fields:
        X = np.asarray(var.coords)
        a, b = trace[0, :2], trace[1, :2]
        t = np.clip(((X - a) @ (b - a)) / ((b - a) @ (b - a)), 0.0, 1.0)
        expect = np.linalg.norm(X - (a + t[:, None] * (b - a)), axis=1)
        np.testing.assert_allclose(np.asarray(var.array).ravel(), expect, rtol=0, atol=1e-10)


def test_two_fault_collections_keep_their_own_distance_fields():
    pytest.importorskip("pyvista")
    box = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
                                            cellSize=0.25)
    xx, yy = np.meshgrid(np.linspace(0, 1, 6), np.linspace(0, 1, 6))
    fields = []
    for i, z in enumerate((0.25, 0.75)):
        collection = uw.meshing.FaultCollection()
        fault = uw.meshing.FaultSurface(f"f0132_{i}",
                                        np.column_stack([xx.ravel(), yy.ravel(), np.full(36, z)]))
        fault.triangulate()
        collection.add(fault)
        fields.append((collection.compute_distance_field(box), z))
    assert fields[0][0]._base_var is not fields[1][0]._base_var
    for var, z in fields:
        # distance to the horizontal plane at height z
        np.testing.assert_allclose(np.abs(np.asarray(var.array).ravel()),
                                   np.abs(np.asarray(var.coords)[:, 2] - z), atol=1e-10)
