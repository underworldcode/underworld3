import underworld3 as uw
import pytest

@pytest.fixture(scope="module")
def load_2d_simplices_mesh():
    resX, resY = 10, 10
    minX, maxX = -5.0, 5.0
    minY, maxY = -5.0, 5.0

    mesh = uw.Mesh(elementRes=(resX, resY), 
                minCoords=(minX, minY),
                maxCoords=(maxX, maxY),
                simplex=True)
    return mesh

@pytest.fixture(scope="module")
def load_2d_quads_mesh():
    resX, resY = 10, 10
    minX, maxX = -5.0, 5.0
    minY, maxY = -5.0, 5.0

    mesh = uw.Mesh(elementRes=(resX, resY), 
                minCoords=(minX, minY),
                maxCoords=(maxX, maxY),
                simplex=False)
    return mesh

@pytest.fixture(scope="module")
def load_3d_quads_mesh():
    resX, resY, resZ = 10, 10, 10
    minX, maxX = -5.0, 5.0
    minY, maxY = -5.0, 5.0
    minZ, maxZ = -5.0, 5.0

    mesh = uw.Mesh(elementRes=(resX, resY, resZ), 
                minCoords=(minX, minY, minZ),
                maxCoords=(maxX, maxY, maxZ),
                simplex=False)
    return mesh

@pytest.fixture(scope="module")
def load_3d_simplices_mesh():
    resX, resY, resZ = 10, 10, 10
    minX, maxX = -5.0, 5.0
    minY, maxY = -5.0, 5.0
    minZ, maxZ = -5.0, 5.0

    mesh = uw.Mesh(elementRes=(resX, resY, resZ), 
                minCoords=(minX, minY, minZ),
                maxCoords=(maxX, maxY, maxZ),
                simplex=True)
    return mesh


@pytest.fixture(scope="module", params=["2DQuads", "3DQuads", "2DSimplices", "3DSimplices"])
def load_multi_meshes(request, load_2d_quads_mesh, load_3d_quads_mesh, load_2d_simplices_mesh, load_3d_simplices_mesh):
    mesh_dict = {"2DQuads": load_2d_quads_mesh, \
                 "3DQuads": load_3d_quads_mesh, \
                 "2DSimplices": load_2d_simplices_mesh,
                 "3DSimplices": load_3d_simplices_mesh}
    mesh_type = request.param
    return mesh_dict[mesh_type]