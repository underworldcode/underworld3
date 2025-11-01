import underworld3 as uw
import numpy as np
import pytest

from underworld3.kdtree import KDTree

test_single_data = []

coords = np.zeros((1, 2))
coords[0] = (0.5,) * 2
test_single_data.append((100000, 2, coords))

coords = np.zeros((1, 3))
coords[0] = (0.5,) * 3
test_single_data.append((100000, 3, coords))


# Single coord test
@pytest.mark.parametrize("n,dim,coords", test_single_data)
def test_single_coord(n, dim, coords):
    """
    Create set of randomly distributed points.
    Find the point closes to the provided coordinate.
    """
    pts = np.random.random(size=(n, dim))

    # Use brute force to find closest point
    repeatcoords = np.repeat(coords, n, 0)
    diff = pts - repeatcoords[:, 0:dim]
    brute_dist = np.sqrt(np.sum(np.multiply(diff, diff), 1))
    brute_id = np.argmin(brute_dist)

    # Build our index
    index = uw.kdtree.KDTree(pts)
    # Use KDTree to find closest point to a coord
    kd_dist, kd_id = index.query(coords, sqr_dists=False)

    assert np.any(kd_id[0] > index.n) == False, "Some point weren't found. Error"

    assert kd_id[0] == brute_id, "KDTree and brute force method did not find the same point."

    assert np.allclose(kd_dist[0], brute_dist[brute_id]), (
        "KDTree and Numpy did not find the same distance.\n"
        f"KDTree distance={kd_dist[0]} Numpy distance={brute_dist[brute_id]} "
    )


@pytest.mark.parametrize("n,dim", [(10000, 2), (10000, 3)])
def test_self_points(n, dim):
    """
    Create set of randomly distributed points.
    Use index to search for points closest to the
    points themselves! Should of course return
    their own index.
    """
    pts = np.random.random(size=(n, dim))

    # Build our index
    index = uw.kdtree.KDTree(pts)

    # Use KDTree to find closest point to a coord
    dist, kdpt = index.query(pts)

    assert np.any(kdpt > index.n) == False, "Some point weren't found. Error"
    # `find_closest_point` should return index of pts.
    assert np.allclose(np.arange(n), kdpt), "Point indices weren't as expected."
    # Distance to self should be zero!
    assert np.allclose(0.0, dist), "Point distances weren't as expected."


# Mesh vertex test
@pytest.mark.parametrize("res,dim", [(16, 2), (8, 3)])
def test_mesh_verts(res, dim):
    """
    Generate a mesh and create the kd index
    using the mesh vertices. Grab a copy of
    the mesh verts, add only a small amount noise,
    and confirm displaced coords are still closest
    to mesh verts.
    """
    mesh = uw.meshing.StructuredQuadBox(elementRes=(res,) * dim)
    index = uw.kdtree.KDTree(mesh.X.coords[...])
    # index = KDTree(mesh.X.coords[...])

    # Get copy of mesh vertices, and add some noise, but only a small
    # amount such that the copied data points are still closest to the
    # original points.
    elsize = 1.0 / float(res)
    coords = mesh.X.coords[...] + 0.5 * elsize * np.random.random(mesh.X.coords[...].shape)
    dist, kdpt = index.query(coords, sqr_dists=False)

    assert np.any(kdpt > index.n) == False, "Some point weren't found. Error"

    # assert np.allclose(True, found), "All points should have been found."
    # `find_closest_point` should return index of pts.
    assert np.allclose(
        np.arange(mesh.X.coords[...].shape[0]), kdpt
    ), "Point indices weren't as expected."
    # Calc distances
    diff = mesh.X.coords[...] - coords
    dot2 = np.sqrt(np.sum(np.multiply(diff, diff), 1))
    assert np.allclose(dist.squeeze(), dot2), "Point distances weren't as expected."


# Mesh centroid test
@pytest.mark.parametrize("res,dim,fill_param", [(16, 2, 4), (16, 3, 4)])
def test_mesh_centroid(res, dim, fill_param):
    """
    Generate a mesh and create the kd index
    using the mesh centroids. Generate a swarm,
    find closest centroid to particles, and confirm
    that closest centroid coincides with particle
    owning cell.
    """
    mesh = uw.meshing.StructuredQuadBox(elementRes=(res,) * dim)
    swarm = uw.swarm.Swarm(mesh)
    swarm.populate(fill_param=fill_param)

    pts_per_cell = swarm.dm.getSize() // mesh._centroids.shape[0]

    centroids = np.zeros((res**dim, dim))
    cellid = np.zeros((res**dim * pts_per_cell), dtype=int)

    for index in range(res**dim):
        centroids[index] = mesh._centroids[index]
        for i in range(pts_per_cell):
            cellid[index * pts_per_cell + i] = index

    index = uw.kdtree.KDTree(centroids)

    dist, kdpt = index.query(swarm._particle_coordinates.data[...])

    assert np.any(kdpt > index.n) == False, "Some point weren't found. Error"
    # `find_closest_point` should return index of pts.
    assert np.allclose(cellid, kdpt), "Point indices weren't as expected."
