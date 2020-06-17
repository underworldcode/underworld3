from petsc4py.PETSc cimport DM, PetscDM, DS, PetscDS, Vec, PetscVec
from .petsc_types cimport PetscInt, PetscReal, PetscScalar, PetscErrorCode, PetscBool, DMBoundaryConditionType, PetscDSResidualFn, PetscDSJacobianFn
from .petsc_types cimport PtrContainer
import underworld3 as uw
from ._jitextension import getext, diff_fn1_wrt_fn2
from sympy import sympify
# TODO
# gil v nogil 
# ctypeds DMBoundaryConditionType etc.. is there a cleaner way? 

cdef extern from "petsc.h":
    PetscErrorCode PetscDSAddBoundary( PetscDS, DMBoundaryConditionType, const char[], const char[], PetscInt, PetscInt, const PetscInt *, void (*)(), PetscInt, const PetscInt *, void *)

cdef extern from "petsc.h" nogil:
    PetscErrorCode PetscDSSetResidual( PetscDS, PetscInt, PetscDSResidualFn, PetscDSResidualFn )
    PetscErrorCode PetscDSSetJacobian( PetscDS, PetscInt, PetscInt, PetscDSJacobianFn, PetscDSJacobianFn, PetscDSJacobianFn, PetscDSJacobianFn)
    PetscErrorCode DMPlexSetSNESLocalFEM( PetscDM, void *, void *, void *)
    PetscErrorCode DMPlexSNESComputeBoundaryFEM( PetscDM, void *, void *)

from petsc4py import PETSc
    
class Poisson:

    def __init__(self, mesh, degree=1):
        self.mesh = mesh
        options = PETSc.Options()
        options.setValue("u_petscspace_degree", degree)
        self._u = uw.MeshVariable( mesh = mesh, 
                                   num_components = 1,
                                   name = "u",
                                   vtype = uw.mesh.VarType.SCALAR, 
                                   isSimplex = False)
        mesh.plex.createDS()
        self._k = 1.
        self._h = 0.

        self.bcs = []

        super().__init__()

    @property
    def u(self):
        return self._u

    @property
    def k(self):
        return self._k
    @k.setter
    def k(self, value):
        # should add test here to make sure k is conformal
        self._k = sympify(value)

    @property
    def h(self):
        return self._h
    @h.setter
    def h(self, value):
        # should add test here to make sure h is conformal
        self._h = sympify(value)

    def add_dirichlet_bc(self, fn, boundaries, comps=[0]):
        # switch to numpy arrays
        # ndmin arg forces an array to be generated even
        # where comps/indices is a single value.
        import numpy as np
        comps      = np.array(comps,      dtype=np.int32, ndmin=1)
        boundaries = np.array(boundaries, dtype=object,   ndmin=1)
        from collections import namedtuple
        BC = namedtuple('BC', ['comps', 'fn', 'boundaries'])
        self.bcs.append(BC(comps,sympify(fn),boundaries))

    def _setup_terms(self):
        from sympy.vector import gradient
        import sympy

        N = self.mesh.N

        # f0 residual term
        self._f0 = self.h
        # f1 residual term
        self._f1 = gradient(self.u.fn)*self.k
        # g0 jacobian term
        self._g0 = -diff_fn1_wrt_fn2(self.h,self.u.fn)
        # g1 jacobian term
        dk_du = diff_fn1_wrt_fn2(self.k,self.u.fn)
        self._g1 = dk_du*gradient(self.u.fn)
        # g3 jacobian term
        dk_dux = diff_fn1_wrt_fn2(self.k, self.u.fn.diff(N.x))
        dk_duy = diff_fn1_wrt_fn2(self.k, self.u.fn.diff(N.y))
        dk_duz = diff_fn1_wrt_fn2(self.k, self.u.fn.diff(N.z))
        dk = dk_dux*N.i + dk_duy*N.j + dk_duz*N.k
        self._g3 = dk|gradient(self.u.fn)                        # outer product for nonlinear part
        self._g3 += self.k*( (N.i|N.i) + (N.j|N.j) + (N.k|N.k) )  # linear part using dyadic identity

        fns_residual = (self._f0, self._f1)
        fns_jacobian = (self._g0, self._g1, self._g3)
        fns_bcs      = [x[1] for x in self.bcs]

        # generate JIT code
        cdef PtrContainer ext = getext(self.mesh, fns_residual, fns_jacobian, fns_bcs)

        # set functions 
        cdef DS ds = self.mesh.plex.getDS()
        PetscDSSetResidual(ds.ds, 0, ext.fns_residual[0], ext.fns_residual[1])
        # TODO: check if there's a significant performance overhead in passing in 
        # identically `zero` pointwise functions instead of setting to `NULL`
        PetscDSSetJacobian(ds.ds, 0, 0, ext.fns_jacobian[0], ext.fns_jacobian[1], NULL, ext.fns_jacobian[2])
        cdef int ind=1
        cdef int [::1] comps_view  # for numpy memory view
        for index,bc in enumerate(self.bcs):
            comps_view = bc.comps
            for boundary in bc.boundaries:
                # use type 5 bc for `DM_BC_ESSENTIAL_FIELD` enum
                PetscDSAddBoundary(ds.ds, 5, NULL, str(boundary).encode('utf8'), 0, comps_view.shape[0], <const PetscInt *> &comps_view[0], <void (*)()>ext.fns_bcs[index], 1, <const PetscInt *> &ind, NULL)
        self.mesh.plex.setUp()

        self.mesh.plex.createClosureIndex(None)
        cdef DM dm = self.mesh.plex
        self.snes = PETSc.SNES().create(PETSc.COMM_WORLD)
        self.snes.setDM(self.mesh.plex)
        self.snes.setFromOptions()
        DMPlexSetSNESLocalFEM(dm.dm, NULL, NULL, NULL)
        self.u_global = self.mesh.plex.createGlobalVector()
        self.u_local  = self.mesh.plex.createLocalVector()


    def solve(self, setup=True):
        if setup:
            self._setup_terms()
        self.mesh.plex.localToGlobal(self.u_local, self.u_global, addv=PETSc.InsertMode.ADD_VALUES)
        self.snes.solve(None,self.u_global)
        self.mesh.plex.globalToLocal(self.u_global,self.u_local)
        # add back boundaries.. 
        cdef Vec lvec= self.u_local
        cdef DM dm = self.mesh.plex
        DMPlexSNESComputeBoundaryFEM(dm.dm, <void*>lvec.vec, NULL)