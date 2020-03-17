from petsc4py import PETSc
from .petsc_gen_xdmf import generateXdmf
import numpy as np

class FeMesh():

    def __init__(self, elementRes=(10, 10), minCoords=(0., 0.),
                 maxCoords=(1.0, 1.0), simplex=False):

        self.elementRes = elementRes
        self.minCoords = minCoords
        self.maxCoords = maxCoords
        self.isSimplex = simplex
        self.plex = PETSc.DMPlex().createBoxMesh(
            elementRes, 
            lower=minCoords, 
            upper=maxCoords,
            simplex=simplex)
        self.plex.distribute()

    @property
    def data(self):
        nnodes = np.prod([val + 1 for val in self.elementRes])
        return self.plex.getCoordinates().array.reshape((nnodes, self.dim))

    @property
    def dim(self):
        """ Number of dimensions of the mesh """
        return self.plex.getDimension()

    def save(self, filename):
        viewer = PETSc.Viewer().createHDF5(filename, "w")
        viewer(self.plex)
        generateXdmf(filename)

    def add_mesh_variable(self):
        return

