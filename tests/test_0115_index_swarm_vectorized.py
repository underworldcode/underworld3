"""
Verify the vectorized IndexSwarmVariable._update_proxy_variables against
a reference implementation that mirrors the original (slow) per-particle loop.

We construct a small mesh, populate a swarm, assign materials based on
a horizontal interface, then compare the fractions produced by the
in-class (vectorized) method against an independent reference computed
by walking the particles one-by-one (the original algorithm).

Both update_type=0 and update_type=1 are tested.
"""
import numpy as np
import underworld3 as uw
import pytest

# All tests in this module are quick core tests
pytestmark = pytest.mark.level_1


def _reference_update_type0(swarm, material, nnn, radius_s):
    """
    Independent reimplementation of the ORIGINAL update_type=0 algorithm:
    for each particle, find its nearest mesh node(s) (np.isclose),
    accumulate 1/dist onto those nodes, normalize per node.
    Returns an (n_nodes, n_indices) array of fractions.
    """
    meshVar0 = material._meshLevelSetVars[0]
    kd_nodes = meshVar0._get_kdtree()
    n_distance, n_indices = kd_nodes.query(
        swarm._particle_coordinates.data, k=nnn, sqr_dists=False
    )
    kd_swarm = swarm._get_kdtree()
    _, nearest_particle_per_node = kd_swarm.query(
        meshVar0.coords, k=1, sqr_dists=False
    )

    n_nodes = meshVar0.data.shape[0]
    n_indices_mat = material.indices
    out = np.zeros((n_nodes, n_indices_mat))

    for ii in range(n_indices_mat):
        node_values = np.zeros(n_nodes)
        w = np.zeros(n_nodes)
        for i in range(swarm.local_size):
            tem = np.isclose(n_distance[i, :], n_distance[i, 0])
            dist = n_distance[i, tem]
            indices = n_indices[i, tem]
            tem = dist < radius_s
            dist = dist[tem]
            indices = indices[tem]
            for j, ind in enumerate(indices):
                node_values[ind] += (
                    np.isclose(material.data[i], ii) / (1.0e-16 + dist[j])
                )[0]
                w[ind] += 1.0 / (1.0e-16 + dist[j])
        node_values[w > 0] /= w[w > 0]
        # fallback for uncovered nodes
        ind_w0 = np.where(w == 0.0)[0]
        if len(ind_w0) > 0:
            ind_ = np.where(material.data[nearest_particle_per_node[ind_w0]] == ii)[0]
            if len(ind_) > 0:
                node_values[ind_w0[ind_]] = 1.0
        out[:, ii] = node_values
    return out


def _reference_update_type1(swarm, material, nnn, radius_s, nnn_bc=None, ind_bc=None):
    """
    Independent reimplementation of the ORIGINAL update_type=1 algorithm:
    for each mesh node, find its nnn nearest particles within radius_s,
    compute IDW-weighted material fractions.
    """
    meshVar0 = material._meshLevelSetVars[0]
    kd = uw.kdtree.KDTree(swarm._particle_coordinates.data)
    n_distance, n_indices = kd.query(meshVar0.coords, k=nnn, sqr_dists=False)

    n_nodes = meshVar0.data.shape[0]
    n_indices_mat = material.indices
    out = np.zeros((n_nodes, n_indices_mat))

    bc_set = set(ind_bc) if ind_bc is not None else set()

    for ii in range(n_indices_mat):
        node_values = np.zeros(n_nodes)
        w = np.zeros(n_nodes)
        for i in range(n_nodes):
            if i not in bc_set:
                ind = np.where(n_distance[i, :] < radius_s)
                a = 1.0 / (n_distance[i, ind] + 1.0e-16)
                w[i] = np.sum(a)
                b = np.isclose(material.data[n_indices[i, ind]], ii)
                node_values[i] = np.sum(np.dot(a, b))
                if ind[0].size == 0:
                    w[i] = 0
            else:
                ind = np.where(n_distance[i, :nnn_bc] < radius_s)
                a = 1.0 / (n_distance[i, :nnn_bc][ind] + 1.0e-16)
                w[i] = np.sum(a)
                b = np.isclose(material.data[n_indices[i, :nnn_bc][ind]], ii)
                node_values[i] = np.sum(np.dot(a, b))
                if ind[0].size == 0:
                    w[i] = 0
        node_values[w > 0] /= w[w > 0]
        out[:, ii] = node_values
    return out


def _build_mesh_and_swarm(fill_param=5, nnn=5, radius=0.5, update_type=0,
                           nnn_bc=None, ind_bc=None, interface_y=0.0,
                           mesh_res=(4, 4)):
    """Build a small mesh with a 2-material interface at y=interface_y."""
    uw.reset_default_model()
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=mesh_res,
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
    )
    swarm = uw.swarm.Swarm(mesh)
    material = uw.swarm.IndexSwarmVariable(
        "M_test", swarm, indices=2, proxy_degree=1, proxy_continuous=True,
        update_type=update_type, npoints=nnn, radius=radius,
        npoints_bc=nnn_bc if nnn_bc is not None else 2,
        ind_bc=ind_bc,
    )
    swarm.populate(fill_param=fill_param)

    # Assign materials based on y-coordinate relative to interface
    with swarm.access():
        material.data[:, 0] = np.where(
            swarm.data[:, 1] <= interface_y, 0, 1
        ).astype(int)
    return mesh, swarm, material


@pytest.mark.parametrize("update_type", [0, 1])
@pytest.mark.parametrize("radius", [0.1, 0.3, 0.5, 1.0])
@pytest.mark.parametrize("nnn", [3, 5, 10])
def test_vectorized_matches_reference(update_type, radius, nnn):
    """Compare the vectorized in-class update against the reference loop."""
    mesh, swarm, material = _build_mesh_and_swarm(
        fill_param=5, nnn=nnn, radius=radius, update_type=update_type,
        interface_y=0.5, mesh_res=(4, 4),
    )

    # Compute reference using the slow loop
    if update_type == 0:
        ref = _reference_update_type0(swarm, material, nnn, radius)
    else:
        ref = _reference_update_type1(swarm, material, nnn, radius)

    # Trigger the vectorized in-class update
    material._proxy_stale = True
    material._update_proxy_if_stale()

    # Read the in-class fractions
    vec = np.zeros_like(ref)
    for ii in range(material.indices):
        vec[:, ii] = material._meshLevelSetVars[ii].data[:, 0]

    # Compare — allow small tolerance for floating-point summation order
    assert np.allclose(ref, vec, atol=1e-10, rtol=1e-7), (
        f"Mismatch for update_type={update_type}, radius={radius}, nnn={nnn}\n"
        f"Max abs diff: {np.max(np.abs(ref - vec))}\n"
        f"ref sum per material: {ref.sum(axis=0)}\n"
        f"vec sum per material: {vec.sum(axis=0)}"
    )


def test_fractions_sum_to_one():
    """For nodes with at least one particle nearby, fractions should sum to ~1."""
    mesh, swarm, material = _build_mesh_and_swarm(
        fill_param=5, nnn=10, radius=0.5, update_type=1,
        interface_y=0.5, mesh_res=(6, 6),
    )
    material._proxy_stale = True
    material._update_proxy_if_stale()

    total = np.zeros(material._meshLevelSetVars[0].data.shape[0])
    for ii in range(material.indices):
        total += material._meshLevelSetVars[ii].data[:, 0]
    # Where there's at least some weight, the sum should be ~1.0
    # (allowing for the fallback behavior that may overshoot)
    assert np.all(total >= -1e-10), "Fractions should be non-negative"
    # Most interior nodes should sum to 1.0
    assert np.allclose(total[total > 0.5], 1.0, atol=1e-6), (
        f"Fractions don't sum to 1 where they should. "
        f"Min/Max of total: {total.min():.6f} / {total.max():.6f}"
    )


def test_interface_location():
    """The material interface should be near y=0.5."""
    mesh, swarm, material = _build_mesh_and_swarm(
        fill_param=5, nnn=10, radius=0.2, update_type=1,
        interface_y=0.5, mesh_res=(8, 8),
    )
    material._proxy_stale = True
    material._update_proxy_if_stale()

    # Material 0 should be 1.0 below the interface, 0.0 above
    coords = material._meshLevelSetVars[0].coords
    frac0 = material._meshLevelSetVars[0].data[:, 0]

    below = coords[:, 1] < 0.4
    above = coords[:, 1] > 0.6
    if below.any():
        assert np.mean(frac0[below]) > 0.9, (
            f"Material 0 fraction below interface too low: {np.mean(frac0[below]):.3f}"
        )
    if above.any():
        assert np.mean(frac0[above]) < 0.1, (
            f"Material 0 fraction above interface too high: {np.mean(frac0[above]):.3f}"
        )


if __name__ == "__main__":
    # Run a quick smoke test
    print("Running smoke test...")
    test_vectorized_matches_reference(0, 0.5, 5)
    test_vectorized_matches_reference(1, 0.5, 5)
    test_fractions_sum_to_one()
    test_interface_location()
    print("All smoke tests passed.")
