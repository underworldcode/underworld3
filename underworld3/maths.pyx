from typing import Union
import sympy

import underworld3
import underworld3.timing as timing
from ._jitextension import getext

include "./petsc_extras.pxi"

cdef extern from "petsc.h" nogil:
    PetscErrorCode PetscDSSetObjective( PetscDS, PetscInt, PetscDSResidualFn )
    PetscErrorCode DMPlexComputeIntegralFEM( PetscDM, PetscVec, PetscScalar*, void* )
    PetscErrorCode DMPlexComputeCellwiseIntegralFEM( PetscDM, PetscVec, PetscVec, void* )


class Integral:
    """
    The `Integral` class constructs the volume integral

    .. math:: F_{i}  =   \int_V \, f(\mathbf{x}) \, \mathrm{d} V

    for some scalar function :math:`f` over the mesh domain :math:`V`.

    Parameters
    ----------
    mesh : 
        The mesh over which integration is performed.
    fn : 
        Function to be integrated.

    Example
    -------
    Calculate volume of mesh:

    >>> import underworld3 as uw
    >>> import numpy as np
    >>> mesh = uw.discretisation.Box()
    >>> volumeIntegral = uw.maths.Integral(mesh=mesh, fn=1.)
    >>> np.allclose( 1., volumeIntegral.evaluate(), rtol=1e-8)
    True
    """
    
    @timing.routine_timer_decorator
    def __init__( self,
                  mesh:  underworld3.discretisation.Mesh,
                  fn:    Union[float, int, sympy.Basic] ):

        self.mesh = mesh
        self.fn = sympy.sympify(fn)
        super().__init__()

    @timing.routine_timer_decorator
    def evaluate(self) -> float:
        if len(self.mesh.vars)==0:
            raise RuntimeError("The mesh requires at least a single variable for integration to function correctly.\n"
                               "This is a PETSc limitation.")
                               
        # Create JIT extension.
        # Note that we pass in the mesh variables as primary variables, as this
        # is how they are represented on the mesh DM.

        # Note that (at this time) PETSc does not support vector integrands, so 
        # if we wish to do vector integrals we'll need to split out the components
        # and calculate them individually. Let's support only scalars for now.
        if isinstance(self.fn, sympy.vector.Vector):
            raise RuntimeError("Integral evaluation for Vector integrands not supported.")
        elif isinstance(self.fn, sympy.vector.Dyadic):
            raise RuntimeError("Integral evaluation for Dyadic integrands not supported.")

        cdef PtrContainer ext = getext(self.mesh, [self.fn,], [], [], self.mesh.vars.values())

        # Pull out vec for variables, and go ahead with the integral
        self.mesh.update_lvec()
        a_global = self.mesh.dm.getGlobalVec()
        self.mesh.dm.localToGlobal(self.mesh.lvec, a_global)
        cdef Vec cgvec
        cgvec = a_global

        # # Now, find var with the highest degree. We will then configure the integration 
        # # to use this variable's quadrature object for all variables. 
        # # This needs to be double checked.  
        # deg = 0
        # for key, var in self.mesh.vars.items():
        #     if var.degree >= deg:
        #         deg = var.degree
        #         var_base = var

        # quad_base = var_base.petsc_fe.getQuadrature()
        # for fe in [var.petsc_fe for var in self.mesh.vars.values()]:
        #     fe.setQuadrature(quad_base)
        
        self.mesh.dm.clearDS()
        self.mesh.dm.createDS()

        cdef DM dm = self.mesh.dm
        cdef DS ds = self.mesh.dm.getDS()
        # Now set callback... note that we set the highest degree var_id (as determined
        # above) for the second parameter. 
        ierr = PetscDSSetObjective(ds.ds, 0, ext.fns_residual[0]); CHKERRQ(ierr)
        
        cdef PetscScalar val
        ierr = DMPlexComputeIntegralFEM(dm.dm, cgvec.vec, <PetscScalar*>&val, NULL); CHKERRQ(ierr)
        self.mesh.dm.restoreGlobalVec(a_global)

        # We're making an assumption here that PetscScalar is same as double.
        # Need to check where this may not be the case.
        cdef double vald = <double> val

        return vald


class CellWiseIntegral:
    """
    The `Integral` class constructs the cell wise volume integral

    .. math:: F_{i}  =   \int_V \, f(\mathbf{x}) \, \mathrm{d} V

    for some scalar function :math:`f` over the mesh domain :math:`V`.

    Parameters
    ----------
    mesh : 
        The mesh over which integration is performed.
    fn : 
        Function to be integrated.

    Example
    -------
    Calculate volume of mesh:

    >>> import underworld3 as uw
    >>> import numpy as np
    >>> mesh = uw.discretisation.Box()
    >>> volumeIntegral = uw.maths.Integral(mesh=mesh, fn=1.)
    >>> np.allclose( 1., volumeIntegral.evaluate(), rtol=1e-8)
    True
    """
    
    @timing.routine_timer_decorator
    def __init__( self,
                  mesh:  underworld3.discretisation.Mesh,
                  fn:    Union[float, int, sympy.Basic] ):

        self.mesh = mesh
        self.fn = sympy.sympify(fn)
        super().__init__()

    @timing.routine_timer_decorator
    def evaluate(self) -> float:
        if len(self.mesh.vars)==0:
            raise RuntimeError("The mesh requires at least a single variable for integration to function correctly.\n"
                               "This is a PETSc limitation.")
                               
        # Create JIT extension.
        # Note that we pass in the mesh variables as primary variables, as this
        # is how they are represented on the mesh DM.

        # Note that (at this time) PETSc does not support vector integrands, so 
        # if we wish to do vector integrals we'll need to split out the components
        # and calculate them individually. Let's support only scalars for now.
        if isinstance(self.fn, sympy.vector.Vector):
            raise RuntimeError("Integral evaluation for Vector integrands not supported.")
        elif isinstance(self.fn, sympy.vector.Dyadic):
            raise RuntimeError("Integral evaluation for Dyadic integrands not supported.")

        cdef PtrContainer ext = getext(self.mesh, [self.fn,], [], [], self.mesh.vars.values())

        # Pull out vec for variables, and go ahead with the integral
        self.mesh.update_lvec()
        a_global = self.mesh.dm.getGlobalVec()
        self.mesh.dm.localToGlobal(self.mesh.lvec, a_global)
        cdef Vec cgvec
        cgvec = a_global
       
        cdef DM dmc = self.mesh.dm.clone()
        cdef FE fec = FE().createDefault(self.mesh.dim, 1, False, -1)
        dmc.setField(0, fec)
        dmc.createDS()

        cdef DS ds = dmc.getDS()
        CHKERRQ( PetscDSSetObjective(ds.ds, 0, ext.fns_residual[0]) )
        
        cdef Vec rvec = dmc.createGlobalVec()
        CHKERRQ( DMPlexComputeCellwiseIntegralFEM(dmc.dm, cgvec.vec, rvec.vec, NULL) )
        self.mesh.dm.restoreGlobalVec(a_global)

        results = rvec.array.copy()
        rvec.destroy()

        return results


class mesh_vector_calculus:
    """Vector calculus on uw row matrices
         - this class is designed to augment the functionality of a mesh"""

    def __init__(self, mesh):
        self.mesh = mesh

    def curl(self, matrix):
        """
        $\nabla \cross \mathbf{v}$

        Returns the curl of a 3D vector field or the out-of-plane
        component of a 2D vector field
        """
            
        vector = self.to_vector(matrix)
        vector_curl = sympy.vector.curl(vector)

        if self.mesh.dim == 3:
            return self.to_matrix(vector_curl)
        else:   
            # if 2d, the out-of-plane vector is not defined in the basis so a scalar is returned (cf. vorticity)
            return vector_curl.dot(self.mesh.N.k)
        
    def divergence(self,matrix):
        """
        $\nabla \cdot \mathbf{v}$
        """
        vector = self.to_vector(matrix)
        scalar_div = sympy.vector.divergence(vector)
        return scalar_div

    def gradient(self, scalar):
        """
        $\nabla \phi$
        """

        if isinstance(scalar, sympy.Matrix) and scalar.shape==(1,1):
            scalar = scalar[0,0]

        vector_gradient = sympy.vector.gradient(scalar)
        return self.to_matrix(vector_gradient)

    def to_vector(self, matrix):

        if isinstance(matrix, sympy.vector.Vector):
            return matrix # No need to convert

        if matrix.shape == (1,self.mesh.dim):
            vector = sympy.vector.matrix_to_vector(matrix, self.mesh.N)
        elif matrix.shape == (1,1):
            vector = matrix[0,0]
        else:
            print(f"Unable to convert matrix of size {matrix.shape} to sympy.vector")
            vector = None

        return vector

    def to_matrix(self, vector):

        if isinstance(vector, sympy.Matrix) and vector.shape == (1, self.mesh.dim):
            return vector

        matrix = sympy.Matrix.zeros(1,self.mesh.dim)
        base_vectors = self.mesh.N.base_vectors()

        for i in range(self.mesh.dim):
            matrix[0,i] = vector.dot(base_vectors[i])

        return matrix

    def jacobian(self, vector):

        jac = vector.diff(self.mesh.X).reshape(self.mesh.X.shape[1], vector.shape[1]).tomatrix().T

        return jac
