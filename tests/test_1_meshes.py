import pytest


def test_generate_2d_quads_mesh():
    from underworld3 import Mesh
    resX, resY = 10, 10
    minX, maxX = -5.0, 5.0
    minY, maxY = -5.0, 5.0

    mesh = Mesh(elementRes=(resX, resY), 
                minCoords=(minX, minY),
                maxCoords=(maxX, maxY),
                simplex=False)

    assert mesh.dim == 2, "Dimension should be 2"
    assert mesh.data.shape == ((resX+1)*(resY+1), mesh.dim)

def test_generate_3d_quads_mesh():
    from underworld3 import Mesh
    resX, resY, resZ = 10, 10, 10
    minX, maxX = -5.0, 5.0
    minY, maxY = -5.0, 5.0
    minZ, maxZ = -5.0, 5.0

    mesh = Mesh(elementRes=(resX, resY, resZ), 
                minCoords=(minX, minY, minZ),
                maxCoords=(maxX, maxY, maxZ),
                simplex=False)
    
    assert mesh.dim == 3, "Dimension should be 3"
    assert mesh.data.shape == ((resX+1)*(resY+1)*(resZ+1), mesh.dim)

def test_generate_2d_simplices_mesh():
    from underworld3 import Mesh
    resX, resY = 10, 10
    minX, maxX = -5.0, 5.0
    minY, maxY = -5.0, 5.0

    mesh = Mesh(elementRes=(resX, resY), 
                minCoords=(minX, minY),
                maxCoords=(maxX, maxY),
                simplex=True)
    
    assert mesh.dim == 2, "Dimension should be 2"
    assert mesh.data.shape == ((resX+1)*(resY+1), mesh.dim)

def test_generate_3d_simplices_mesh():
    from underworld3 import Mesh
    resX, resY, resZ = 10, 10, 10
    minX, maxX = -5.0, 5.0
    minY, maxY = -5.0, 5.0
    minZ, maxZ = -5.0, 5.0

    mesh = Mesh(elementRes=(resX, resY, resZ), 
                minCoords=(minX, minY, minZ),
                maxCoords=(maxX, maxY, maxZ),
                simplex=True)
    
    assert mesh.dim == 3, "Dimension should be 3"
    assert mesh.data.shape == ((resX+1)*(resY+1)*(resZ+1), mesh.dim)


def test_spherical_mesh():
    from underworld3 import Spherical
    mesh = Spherical(refinements=4)
    return

def test_mesh_save(load_multi_meshes):
    load_multi_meshes.save("mesh.h5")
    return
    