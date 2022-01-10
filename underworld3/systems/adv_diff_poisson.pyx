import sympy
from sympy import sympify
from sympy.vector import gradient, divergence

from petsc4py import PETSc

import underworld3 
import underworld3 as uw
from .._jitextension import getext, diff_fn1_wrt_fn2
import underworld3.timing as timing

include "../petsc_extras.pxi"


class AdvDiffusion:
    @timing.routine_timer_decorator
    def __init__(self, 
                 mesh       : uw.mesh.MeshClass, 
                 u_Field    : uw.mesh.MeshVariable = None, 
                 ustar_Field: uw.mesh.MeshVariable = None, 
                 degree       = 2,
                 solver_name: str = "",
                 verbose      = False):


        ## Todo: this is obviously not particularly robust

        if solver_name != "" and not solver_name.endswith("_"):
            self.petsc_options_prefix = solver_name+"_"
        else:
            self.petsc_options_prefix = solver_name

        ## Todo: some validity checking on the size / type of u_Field supplied
        if not u_Field:
            self._u = uw.mesh.MeshVariable( mesh=mesh, num_components=1, name="u_adv_diff", vtype=uw.VarType.SCALAR, degree=degree )
        else:
            self._u = u_Field

        self.mesh = mesh
        self.k = 1.
        self.f = 0.
        self.dt = 1.0
        self.ustar = ustar_Field
        self.theta = 0.5

        self.bcs = []

        self.is_setup = False
        self.verbose = verbose

        # Build the DM / FE structures (should be done on remeshing)

        self._build_dm_and_mesh_discretisation()
        self._rebuild_after_mesh_update = self._build_dm_and_mesh_discretisation

        # Some other setup 

        self.mesh._equation_systems_register.append(self)

        super().__init__()


    def _build_dm_and_mesh_discretisation(self):

        degree = self._u.degree
        mesh = self.mesh

        self.dm   = mesh.dm.clone()

        # create private variables
        options = PETSc.Options()
        options.setValue("uprivate_petscspace_degree", degree) # for private variables
        self.petsc_fe_u = PETSc.FE().createDefault(mesh.dim, 1, mesh.isSimplex, degree, "uprivate_", PETSc.COMM_WORLD)
        self.petsc_fe_u_id = self.dm.getNumFields()
        self.dm.setField( self.petsc_fe_u_id, self.petsc_fe_u )

        self.is_setup = False

        return

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
    def f(self):
        return self._f
    @f.setter
    def f(self, value):
        self.is_setup = False
        # should add test here to make sure f is conformal
        self._f = sympify(value)

    @property
    def dt(self):
        return self._dt
    @dt.setter
    def dt(self, value):
        self.is_setup = False
        # should add test here to make sure f is conformal
        self._dt = sympify(value)

    @timing.routine_timer_decorator
    def add_dirichlet_bc(self, fn, boundaries, components=[0]):
        # switch to numpy arrays
        # ndmin arg forces an array to be generated even
        # where comps/indices is a single value.
        self.is_setup = False
        import numpy as np
        components = np.array(components, dtype=np.int32, ndmin=1)
        boundaries = np.array(boundaries, dtype=object,   ndmin=1)
        from collections import namedtuple
        BC = namedtuple('BC', ['components', 'fn', 'boundaries', 'type'])
        self.bcs.append(BC(components,sympify(fn),boundaries, 'dirichlet'))

    @timing.routine_timer_decorator
    def add_neumann_bc(self, fn, boundaries, components=[0]):
        # switch to numpy arrays
        # ndmin arg forces an array to be generated even
        # where comps/indices is a single value.
        self.is_setup = False
        import numpy as np
        components = np.array(components, dtype=np.int32, ndmin=1)
        boundaries = np.array(boundaries, dtype=object,   ndmin=1)
        from collections import namedtuple
        BC = namedtuple('BC', ['components', 'fn', 'boundaries', 'type'])
        self.bcs.append(BC(components,sympify(fn),boundaries, "neumann"))


    @timing.routine_timer_decorator
    def _setup_terms(self):
        from sympy.vector import gradient
        import sympy

        N = self.mesh.N

        # f0 residual term
        self._f0 = -self.f + (self.u.fn - self.ustar.fn) / self.dt

        # f1 residual term
        self._f1 = gradient(self.u.fn*self.theta + self.ustar.fn*(1.0-self.theta))*self.k

        # g0 jacobian term
        self._g0 = diff_fn1_wrt_fn2(self._f0, self.u.fn)  ## Should this one be -self._f0 rather than self.f ?

        # g1 jacobian term
        dk_du = diff_fn1_wrt_fn2(self.k,self.u.fn)
        self._g1 = dk_du*gradient(self.u.fn)

        # g3 jacobian term
        dk_dux = diff_fn1_wrt_fn2(self.k, self.u.fn.diff(N.x))
        dk_duy = diff_fn1_wrt_fn2(self.k, self.u.fn.diff(N.y))
        dk_duz = diff_fn1_wrt_fn2(self.k, self.u.fn.diff(N.z))
        dk = dk_dux*N.i + dk_duy*N.j + dk_duz*N.k
        self._g3 = dk | gradient(self.u.fn)                       # outer product for nonlinear part
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
        cdef DM cdm = self.dm

        for index,bc in enumerate(self.bcs):
            comps_view = bc.components
            for boundary in bc.boundaries:
                if self.verbose:
                    print("Setting bc {} ({})".format(index, bc.type))
                    print(" - components: {}".format(bc.components))
                    print(" - boundary:   {}".format(bc.boundaries))
                    print(" - fn:         {} ".format(bc.fn))

                # use type 5 bc for `DM_BC_ESSENTIAL_FIELD` enum
                # use type 6 bc for `DM_BC_NATURAL_FIELD` enum  (is this implemented for non-zero values ?)
                if bc.type == 'neumann':
                    bc_type = 6
                else:
                    bc_type = 5

                PetscDSAddBoundary_UW( cdm.dm, bc_type, str(boundary).encode('utf8'), str(boundary).encode('utf8'), 0, comps_view.shape[0], <const PetscInt *> &comps_view[0], <void (*)()>ext.fns_bcs[index], NULL, 1, <const PetscInt *> &ind, NULL)

        self.dm.setUp()

        self.dm.createClosureIndex(None)
        self.snes = PETSc.SNES().create(PETSc.COMM_WORLD)
        self.snes.setDM(self.dm)
        self.snes.setOptionsPrefix(self.petsc_options_prefix)
        self.snes.setFromOptions()
        cdef DM dm = self.dm
        DMPlexSetSNESLocalFEM(dm.dm, NULL, NULL, NULL)

        self.is_setup = True

    @timing.routine_timer_decorator
    def solve(self, 
              zero_init_guess: bool =True, 
              timestep = 1.0,
              _force_setup:    bool =False ):
        """
        Generates solution to constructed system.

        Params
        ------
        zero_init_guess:
            If `True`, a zero initial guess will be used for the 
            system solution. Otherwise, the current values of `self.u` will be used.
        """

        if timestep != self.dt:
            self.dt = timestep    # this will force an initialisation because the functions need to be updated

        if (not self.is_setup) or _force_setup:
            self._setup_terms()

        gvec = self.dm.getGlobalVec()

        if not zero_init_guess:
            with self.mesh.access():
                self.dm.localToGlobal(self.u.vec, gvec)
        else:
            gvec.array[:] = 0.

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

        cdef DM dm = self.dm

        self.mesh.update_lvec()
        cdef Vec cmesh_lvec
        # PETSc == 3.16 introduced an explicit interface 
        # for setting the aux-vector which we'll use when available.
        cmesh_lvec = self.mesh.lvec
        ierr = DMSetAuxiliaryVec(dm.dm, NULL, 0, cmesh_lvec.vec); CHKERRQ(ierr)

        # solve
        self.snes.solve(None,gvec)

        lvec = self.dm.getLocalVec()
        cdef Vec clvec = lvec
        # Copy solution back into user facing variable
        with self.mesh.access(self.u,):
            self.dm.globalToLocal(gvec, lvec)
            # add back boundaries.
            # Note that `DMPlexSNESComputeBoundaryFEM()` seems to need to use an lvec
            # derived from the system-dm (as opposed to the var.vec local vector), else 
            # failures can occur. 
            ierr = DMPlexSNESComputeBoundaryFEM(dm.dm, <void*>clvec.vec, NULL); CHKERRQ(ierr)
            self.u.vec.array[:] = lvec.array[:]

        self.dm.restoreLocalVec(lvec)
        self.dm.restoreGlobalVec(gvec)
