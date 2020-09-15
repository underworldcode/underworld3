from petsc4py.PETSc cimport DM, PetscDM, DS, PetscDS, Vec, PetscVec, PetscIS, PetscDM, PetscSF, MPI_Comm, DMLabel
from .petsc_types cimport PetscInt, PetscReal, PetscScalar, PetscErrorCode, PetscBool, DMBoundaryConditionType, PetscDSResidualFn, PetscDSJacobianFn
from petsc4py.PETSc cimport GetCommDefault, GetComm
from .petsc_gen_xdmf import generateXdmf
from petsc4py import PETSc

cdef extern from "petsc.h" nogil:
    PetscErrorCode DMPlexCreateBallMesh(MPI_Comm, PetscInt, PetscReal, PetscDM*)
    PetscErrorCode DMPlexCreateSphereMesh(MPI_Comm, PetscInt, PetscBool, PetscReal, PetscDM*)

class SphericalMesh():

    def __init__(self, simplex=True, radius=1.0, dimension=3):
        """ Create a ball mesh 
        
        import underworld3 as uw
        mesh = uw.SphericalMesh
        mesh.save("mesh.h5")

        """

        cdef DM dm = PETSc.DMPlex()
        cdef DM sdm = PETSc.DMPlex()
        cdef MPI_Comm ccomm = GetCommDefault()
        cdef PetscInt cdim = dimension - 1
        cdef PetscReal cradius = radius
        cdef PetscBool csimplex = simplex
        DMPlexCreateSphereMesh(ccomm, cdim, csimplex, cradius, &sdm.dm)
        self.sdm = sdm
        options = PETSc.Options()
        options.setValue("bd_dm_refine", 4)
        self.sdm.setOptionsPrefix("bd_")
        self.sdm.setFromOptions()
        self.dm = dm
        self.dm.generate(self.sdm, interpolate=True)
        self.sdm.destroy()
        self.dm.view()
    
    def save(self, filename):
        viewer = PETSc.Viewer().createHDF5(filename, "w")
        viewer(self.dm)
        generateXdmf(filename)


class BallMesh():

    def __init__(self, radius=1.0, dimension=3):
        """ Create a ball mesh 
        
        import underworld3 as uw
        mesh = uw.SphericalMesh
        mesh.save("mesh.h5")

        """

        options = PETSc.Options()
        options.setValue("bd_dm_refine", 4)
        cdef DM dm = PETSc.DMPlex()
        cdef MPI_Comm ccomm = GetCommDefault()
        cdef PetscInt cdim = dimension
        cdef PetscReal cradius = radius
        DMPlexCreateBallMesh(ccomm, cdim, cradius, &dm.dm)
        self.dm = dm
        self.dm.view()
    
    def save(self, filename):
        viewer = PETSc.Viewer().createHDF5(filename, "w")
        viewer(self.dm)
        generateXdmf(filename)
