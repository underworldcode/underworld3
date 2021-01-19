from petsc4py.PETSc cimport DM, PetscDM, DS, PetscDS, Vec, PetscVec, PetscIS, FE, PetscFE, PetscQuadrature
from .petsc_types cimport PetscInt, PetscReal, PetscScalar, PetscErrorCode, PetscBool, DMBoundaryConditionType, PetscDSResidualFn, PetscDSJacobianFn
from .petsc_types cimport PtrContainer
import underworld3 as uw
from ._jitextension import getext, diff_fn1_wrt_fn2
from sympy import sympify
# TODO
# gil v nogil 
# ctypeds DMBoundaryConditionType etc.. is there a cleaner way? 

cdef CHKERRQ(PetscErrorCode ierr):
    cdef int interr = <int>ierr
    if ierr != 0: raise RuntimeError(f"PETSc error code '{interr}' was encountered.\nhttps://www.mcs.anl.gov/petsc/petsc-current/include/petscerror.h.html")

cdef extern from "petsc.h":
    PetscErrorCode PetscDSAddBoundary( PetscDS, DMBoundaryConditionType, const char[], const char[], PetscInt, PetscInt, const PetscInt *, void (*)(), void (*)(), PetscInt, const PetscInt *, void *)

cdef extern from "petsc.h" nogil:
    PetscErrorCode PetscDSSetResidual( PetscDS, PetscInt, PetscDSResidualFn, PetscDSResidualFn )
    PetscErrorCode PetscDSSetJacobian( PetscDS, PetscInt, PetscInt, PetscDSJacobianFn, PetscDSJacobianFn, PetscDSJacobianFn, PetscDSJacobianFn)
    PetscErrorCode DMPlexSetSNESLocalFEM( PetscDM, void *, void *, void *)
    PetscErrorCode DMPlexSNESComputeBoundaryFEM( PetscDM, void *, void *)
    PetscErrorCode PetscFEGetQuadrature(PetscFE fem, PetscQuadrature *q)
    PetscErrorCode PetscFESetQuadrature(PetscFE fem, PetscQuadrature q)


from petsc4py import PETSc
    
class Poisson:

    def __init__(self, mesh, degree=1):
        self.mesh = mesh
        self.dm   = mesh.dm.clone()

        self._u = uw.MeshVariable( mesh=mesh, num_components=1, name="u", vtype=uw.mesh.VarType.SCALAR, degree=degree )
        # create private variables
        options = PETSc.Options()
        options.setValue("uprivate_petscspace_degree", degree) # for private variables
        self.petsc_fe_u = PETSc.FE().createDefault(self.dm.getDimension(), 1, mesh.isSimplex, degree,"uprivate_", PETSc.COMM_WORLD)
        self.petsc_fe_u_id = self.dm.getNumFields()
        self.dm.setField( self.petsc_fe_u_id, self.petsc_fe_u )

        self._k = 1.
        self._h = 0.

        self.bcs = []

        self.is_setup = False
        super().__init__()

    @property
    def u(self):
        return self._u

    @property
    def k(self):
        return self._k
    @k.setter
    def k(self, value):
        self.is_setup = False
        # should add test here to make sure k is conformal
        self._k = sympify(value)

    @property
    def h(self):
        return self._h
    @h.setter
    def h(self, value):
        self.is_setup = False
        # should add test here to make sure h is conformal
        self._h = sympify(value)

    def add_dirichlet_bc(self, fn, boundaries, components=[0]):
        # switch to numpy arrays
        # ndmin arg forces an array to be generated even
        # where comps/indices is a single value.
        self.is_setup = False
        import numpy as np
        components = np.array(components, dtype=np.int32, ndmin=1)
        boundaries = np.array(boundaries, dtype=object,   ndmin=1)
        from collections import namedtuple
        BC = namedtuple('BC', ['components', 'fn', 'boundaries'])
        self.bcs.append(BC(components,sympify(fn),boundaries))

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

        # generate JIT code.
        # first, we must specify the primary fields.
        # these are fields for which the corresponding sympy functions 
        # should be replaced with the primary (instead of auxiliary) petsc 
        # field value arrays. in this instance, we want to switch out 
        # `self.u` and `self.p` for their primary field 
        # petsc equivalents. without specifying this list, 
        # the aux field equivalents will be used instead, which 
        # will give incorrect results for non-linear problems.
        # note also that the order here is important.
        prim_field_list = [self.u,]
        cdef PtrContainer ext = getext(self.mesh, tuple(fns_residual), tuple(fns_jacobian), [x[1] for x in self.bcs], primary_field_list=prim_field_list)

        # set functions 
        self.dm.createDS()
        cdef DS ds = self.dm.getDS()
        PetscDSSetResidual(ds.ds, 0, ext.fns_residual[0], ext.fns_residual[1])
        # TODO: check if there's a significant performance overhead in passing in 
        # identically `zero` pointwise functions instead of setting to `NULL`
        PetscDSSetJacobian(ds.ds, 0, 0, ext.fns_jacobian[0], ext.fns_jacobian[1], NULL, ext.fns_jacobian[2])
        cdef int ind=1
        cdef int [::1] comps_view  # for numpy memory view
        for index,bc in enumerate(self.bcs):
            comps_view = bc.components
            for boundary in bc.boundaries:
                # use type 5 bc for `DM_BC_ESSENTIAL_FIELD` enum
                PetscDSAddBoundary(ds.ds, 5, NULL, str(boundary).encode('utf8'), 0, comps_view.shape[0], <const PetscInt *> &comps_view[0], <void (*)()>ext.fns_bcs[index], NULL, 1, <const PetscInt *> &ind, NULL)
        self.dm.setUp()

        self.dm.createClosureIndex(None)
        self.snes = PETSc.SNES().create(PETSc.COMM_WORLD)
        self.snes.setDM(self.dm)
        self.snes.setFromOptions()
        cdef DM dm = self.dm
        DMPlexSetSNESLocalFEM(dm.dm, NULL, NULL, NULL)

        self.is_setup = True

    def solve(self, 
              zero_init_guess: bool =True, 
              _force_setup:    bool =False ):
        """
        Generates solution to constructed system.

        Params
        ------
        zero_init_guess:
            If `True`, a zero initial guess will be used for the 
            system solution. Otherwise, the current values of `self.u` 
            and `self.p` will be used.
        """
        if (not self.is_setup) or _force_setup:
            self._setup_terms()

        gvec = self.dm.getGlobalVec()
        lvec = self.dm.getLocalVec()

        if not zero_init_guess:
            with self.mesh.access():
                # TODO: Check if this approach is valid generally. 
                #       Alternative is to grab dm->subdm IS (from `DMCreateSubDM`), 
                #       use that for full_vec->sub_vec mapping (global), followed
                #       with global->local.
                lvec.array[:] = self.u.vec.array[:]
                self.dm.localToGlobal(lvec,gvec)
        else:
            gvec.array[:] = 0.

        # Set all quadratures to u quadrature.
        # Note that this needs to be done before the call to 
        # `getLocalVariableVec()`, as `createDS` 
        # TODO: Check if we need to unwind these quadratures. 
        #       I believe the PETSc reference counting should 
        #       do the job, but I'm not 100% sure.
        cdef PetscQuadrature quad
        cdef FE c_fe = self.petsc_fe_u
        ierr = PetscFEGetQuadrature(c_fe.fe, &quad); CHKERRQ(ierr)
        for fe in [var.petsc_fe for var in self.mesh.vars.values()]:
            c_fe = fe
            ierr = PetscFESetQuadrature(c_fe.fe,quad); CHKERRQ(ierr)        # set to vel quad

        # Call `createDS()` on aux dm. This is necessary after the 
        # quadratures are set above, as it generates the tablatures 
        # from the quadratures (among other things no doubt). 
        # TODO: What does createDS do?
        # TODO: What are the implications of calling this every solve.
        self.mesh.dm.clearDS()
        self.mesh.dm.createDS()

        a_local = self.mesh.getLocalVariableVec()
        self.dm.compose("A", a_local)
        self.dm.compose("dmAux", self.mesh.dm)

        # solve
        self.snes.solve(None,gvec)
        self.dm.globalToLocal(gvec,lvec)
        # add back boundaries.. 
        cdef Vec clvec = lvec
        cdef DM dm = self.dm
        DMPlexSNESComputeBoundaryFEM(dm.dm, <void*>clvec.vec, NULL)
        self.mesh.restoreLocalVariableVec()

        # This comment relates to previous implementation, but I'll leave it 
        # here for now as something to be aware of / investigate further.
        
        # # create SubDMs now to isolate velocity/pressure variables.
        # # this is currently problematic as the calls to DMCreateSubDM
        # # need to be executed after the solve above, as otherwise the 
        # # results are altered/corrupted. it's unclear to me why this is
        # # the case.  the migrationsf doesn't change anything.
        # # also, the SubDMs and associated vectors are potentially leaking
        # # due to our petsc4py/cython hacks below. 
        with self.mesh.access(self.u,):
            # TODO: Check if this approach is valid generally. 
            #       Alternative is to grab dm->subdm IS (from `DMCreateSubDM`), 
            #       use that for full_vec->sub_vec mapping (global), followed
            #       with global->local.
            self.u.vec.array[:] = lvec.array[:]

        self.dm.restoreLocalVec(lvec)
        self.dm.restoreGlobalVec(gvec)