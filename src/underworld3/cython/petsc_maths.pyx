from typing import Union
import sympy

import underworld3
import underworld3.timing as timing
from   underworld3.utilities._jitextension import getext

from petsc4py import PETSc

include "petsc_extras.pxi"

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
    def evaluate(self, verbose=False) -> float:
        if len(self.mesh.vars)==0:
            raise RuntimeError("The mesh requires at least a single variable for integration to function correctly.\n"
                               "This is a PETSc limitation.")

        # Create JIT extension.
        #
        # Note that - we pass in the mesh variables as primary variables, as this
        # is how they are represented on the mesh DM.

        # Note that -  (at this time) PETSc does not support vector integrands, so
        # if we wish to do vector integrals we'll need to split out the components
        # and calculate them individually. Let's support only scalars for now.

        # Note that - DMPlexComputeIntegralFEM returns an array even though we have set only
        # one objective function and only expect one non-zero value to be returned
        # Temporary workaround for this is to over-allocate the array we collect.

        if isinstance(self.fn, sympy.vector.Vector):
            raise RuntimeError("Integral evaluation for Vector integrands not supported.")
        elif isinstance(self.fn, sympy.vector.Dyadic):
            raise RuntimeError("Integral evaluation for Dyadic integrands not supported.")


        self.dm = self.mesh.dm  # .clone()
        mesh=self.mesh

        compiled_extns, dictionaries = getext(self.mesh, [self.fn,], [], [], [], [], self.mesh.vars.values(), verbose=verbose)
        cdef PtrContainer ext = compiled_extns

        # Pull out vec for variables, and go ahead with the integral

        self.mesh.update_lvec()
        a_global = self.dm.getGlobalVec()
        self.dm.localToGlobal(self.mesh.lvec, a_global)

        cdef Vec cgvec
        cgvec = a_global

        cdef DM dm = self.dm
        cdef DS ds = self.dm.getDS()
        cdef PetscScalar val_array[256]

        # Now set callback...
        ierr = PetscDSSetObjective(ds.ds, 0, ext.fns_residual[0]); CHKERRQ(ierr)
        ierr = DMPlexComputeIntegralFEM(dm.dm, cgvec.vec, &(val_array[0]), NULL); CHKERRQ(ierr)

        self.dm.restoreGlobalVec(a_global)

        # We're making an assumption here that PetscScalar is same as double.
        # Need to check where this may not be the case.
        cdef double vald = <double> val_array[0]

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

        ## Does this need to be consistent with everything else ?

        cdef DM dmc = self.mesh.dm.clone()
        cdef FE fec = FE().createDefault(self.dim, 1, False, -1)
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
