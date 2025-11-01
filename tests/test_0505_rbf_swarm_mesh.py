import underworld3 as uw
import numpy as np
import pytest


# calculate rbf interpolation by using a known analytic fields on var
# and calculating rbf interpolation of field to random points.
# Warning: if resolution of mesh can't capture analytic fluctuation interpolation will be bad.
# hence analytic functions are smooth and rtol is loose.
@pytest.mark.parametrize("dim,nnn,p", [(2, 1, 2), (2, 3, 2), (3, 4, 2)])
def test_rbf_NearestNeighbQuad(dim, nnn, p):

    # build a mesh and vector field, components size == dim, q1 discretisation
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(10,) * dim, minCoords=(1.0,) * dim, maxCoords=(2.0,) * dim
    )
    var = uw.discretisation.MeshVariable(
        mesh=mesh, varname="test", num_components=dim, degree=1, continuous=True
    )

    # create some simple functions
    f0 = lambda x0, x1: x0
    f1 = lambda x0, x1: 1.0 + np.cos(np.pi / 4 * x0)
    f2 = lambda x0, x1, x2: x0 + x1**2 + x2
    # evaluate functions on the var vector componets
    # Use new mesh.X.coords pattern for coordinate access
    with uw.synchronised_array_update():
        var.array[:, 0, 0] = f0(mesh.X.coords[:, 0], mesh.X.coords[:, 1])
        var.array[:, 0, 1] = f1(mesh.X.coords[:, 0], mesh.X.coords[:, 1])
        if dim == 3:
            var.array[:, 0, 2] = f2(mesh.X.coords[:, 0], mesh.X.coords[:, 1], mesh.X.coords[:, 2])

    # uncommennt for small perturbations from mesh verticies
    # small_fac = 0.1 * mesh._search_lengths[0]
    # rx = mesh.X.coords.copy() + small_fac * (np.random.random(mesh.X.coords.shape) - 0.5)

    # create 30 random coords (30,mesh.dim)
    rx = np.random.random(size=(30, dim)) + mesh.X.coords.min(axis=0)

    # will be nearest neighbour only
    rvals = var.rbf_interpolate(rx, nnn, p)

    assert np.allclose(rvals[:, 0], f0(rx[:, 0], rx[:, 1]), rtol=5e-2)
    assert np.allclose(rvals[:, 1], f1(rx[:, 0], rx[:, 1]), rtol=5e-2)
    if dim == 3:
        assert np.allclose(f2(rx[:, 0], rx[:, 1], rx[:, 2]), rvals[:, 2], rtol=5e-2)
