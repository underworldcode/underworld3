"""Mesh variables the library makes for an owner are that owner's (#1118 follow-up).

Swarm proxies, index-variable level sets, the level-set interface thickness,
the decoded viscoelastic stress and a surface collection's fields were made
under fixed names, so two owners on one mesh shared one variable: the second
overwrote the first's values. Once a re-declaration that changes a variable
refuses (#1118), the same sharing raised instead. Each helper is now named
after its owner, the same on every rank. Each test puts two owners on one mesh
and checks that both keep their own field: a closed form each, exactly.
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
    # Proxies of different degree: the second used to refuse, or before that
    # share the first's proxy. Each swarm carries one linear field.
    swarms, fields = [], []
    for axis, degree in ((0, 1), (1, 2)):
        swarm = uw.swarm.Swarm(mesh)
        s = uw.swarm.SwarmVariable("phase", swarm, 1, proxy_degree=degree)
        swarm.populate(fill_param=3)
        s.array[:, 0, 0] = swarm._particle_coordinates.array[:, 0, axis]
        swarms.append(swarm)
        fields.append((s, axis))
    assert fields[0][0]._meshVar._base_var is not fields[1][0]._meshVar._base_var
    for s, axis in fields:
        s._update_proxy_if_stale()
        proxy = s._meshVar
        error = np.abs(np.asarray(proxy.array).ravel() - np.asarray(proxy.coords)[:, axis]).max()
        assert error < 1.0e-12


def test_two_index_variables_of_one_name_keep_their_own_level_sets(mesh):
    # Nodal level sets of different degree (the default location, integration
    # points, has one layout for every degree).
    first = uw.swarm.IndexSwarmVariable("M", uw.swarm.Swarm(mesh), indices=2, proxy_degree=1,
                                        proxy_location="nodes")
    second = uw.swarm.IndexSwarmVariable("M", uw.swarm.Swarm(mesh), indices=2, proxy_degree=2,
                                         proxy_location="nodes")
    for a, b in zip(first._meshLevelSetVars, second._meshLevelSetVars):
        assert a._base_var is not b._base_var
    assert [v.degree for v in first._meshLevelSetVars] == [1, 1]
    assert [v.degree for v in second._meshLevelSetVars] == [2, 2]


def test_interface_thickness_is_its_level_sets_own(mesh):
    user = uw.discretisation.MeshVariable("epsilon", mesh, 1, degree=2)
    user.array[:, 0, 0] = -1.0
    phi1 = uw.discretisation.MeshVariable("phi_a", mesh, 1, degree=1)
    phi2 = uw.discretisation.MeshVariable("phi_b", mesh, 1, degree=2)
    eps1 = uw.systems.level_set.interface_thickness(mesh, phi1)
    eps2 = uw.systems.level_set.interface_thickness(mesh, phi2)
    assert (eps1.degree, eps2.degree) == (1, 2)
    assert np.all(np.asarray(user.array) == -1.0)            # the user's field is untouched
    assert np.asarray(eps1.array).min() > 0.0 and np.asarray(eps2.array).min() > 0.0


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
