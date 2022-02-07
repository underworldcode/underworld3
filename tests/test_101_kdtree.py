import underworld3 as uw
import numpy as np
import pytest


test_single_data = []

coords = np.zeros((1,2))
coords[0] = (0.5,)*2
test_single_data.append((100000,2,coords))

coords = np.zeros((1,3))
coords[0] = (0.5,)*3
test_single_data.append((100000,3,coords))

# Single coord test
@pytest.mark.parametrize("n,dim,coords", test_single_data)
def test_single_coord(n,dim,coords):
    """
    Create set of randomly distributed points. 
    Find the point closes to the provided coordinate.
    """
    pts = np.random.random( size=(n,dim) )

    # Build our index
    index = uw.algorithms.KDTree(pts)
    index.build_index()

    # Use KDTree to find closest point to a coord
    (kdpt, dist, found) = index.find_closest_point(coords)

    # Use brute force to find closest point
    repeatcoords = np.repeat(coords,n,0)
    diff = pts - repeatcoords[:,0:dim]
    dot2 = np.sum(np.multiply(diff,diff),1)
    brutept = np.argmin(dot2)

    assert found[0], "KDTree did not appear to find a point close to coord."
    assert kdpt[0] == brutept, "KDTree and brute force method did not find the same point."
    assert np.allclose(dist[0],dot2[brutept]), "KDTree and Numpy did not find the same distance squared.\n"\
                           f"KDTree distance={dist[0]} Numpy distance={dot2[brutept]} "


@pytest.mark.parametrize("n,dim", [(10000, 2), (10000, 3)])
def test_self_points(n,dim):
    """
    Create set of randomly distributed points. 
    Use index to search for points closest to the 
    points themselves! Should of course return 
    their own index.
    """
    pts = np.random.random( size=(n,dim) )

    # Build our index
    index = uw.algorithms.KDTree(pts)
    index.build_index()

    # Use KDTree to find closest point to a coord
    (kdpt, dist, found) = index.find_closest_point(pts)

    assert np.allclose(True, found), "All points should have been found."
    # `find_closest_point` should return index of pts.
    assert np.allclose(np.arange(n),kdpt), "Point indices weren't as expected."
    # Distance to self should be zero!
    assert np.allclose(0.,dist), "Point distances weren't as expected."


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
    mesh = uw.meshes.Box(elementRes=(res,)*dim)
    index = uw.algorithms.KDTree(mesh.data)
    index.build_index()

    # Get copy of mesh vertices, and add some noise, but only a small 
    # amount such that the copied data points are still closest to the
    # original points. 
    elsize = 1./float(res)
    coords = mesh.data.copy() + 0.5*elsize*np.random.random( mesh.data.shape )
    (kdpt, dist, found) = index.find_closest_point(coords)
    assert np.allclose(True, found), "All points should have been found."
    # `find_closest_point` should return index of pts.
    assert np.allclose(np.arange(mesh.data.shape[0]),kdpt), "Point indices weren't as expected."
    # Calc distances
    diff = mesh.data - coords
    dot2 = np.sum(np.multiply(diff,diff),1)
    assert np.allclose(dot2,dist), "Point distances weren't as expected."


# Mesh centroid test
@pytest.mark.parametrize("res,dim,ppcell", [(16, 2, 4), (16, 3, 4)])
def test_mesh_centroid(res,dim,ppcell):
    """
    Generate a mesh and create the kd index
    using the mesh centroids. Generate a swarm,
    find closest centroid to particles, and confirm
    that closest centroid coincides with particle 
    owning cell. 
    """
    mesh = uw.meshes.Box(elementRes=(res,)*dim)
    centroids = np.zeros((res**dim,dim))
    for index in range(res**dim):
        centroids[index] = mesh.dm.computeCellGeometryFVM(index)[1]
    index = uw.algorithms.KDTree(centroids)
    index.build_index()

    # add and populate swarm
    swarm  = uw.swarm.Swarm(mesh)
    swarm.populate(fill_param=ppcell)

    with swarm.access():
        kdpt, dist, found = index.find_closest_point(swarm.data)

        assert np.allclose(True, found), "All points should have been found."
        # `find_closest_point` should return index of pts.
        assert np.allclose(swarm.particle_cellid.data[:,0],kdpt), "Point indices weren't as expected."

