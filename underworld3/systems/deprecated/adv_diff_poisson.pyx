import sympy
from sympy import sympify
from sympy.vector import gradient, divergence

from petsc4py import PETSc

import underworld3 
import underworld3 as uw
from .._jitextension import getext, diff_fn1_wrt_fn2
import underworld3.timing as timing
import numpy as np
import mpi4py

from typing import Callable

include "../petsc_extras.pxi"


class AdvDiffusion:
    """ Characteristics-based advection diffusion solver:

    Uses a theta timestepping approach with semi-Lagrange sample backwards in time using 
    a mid-point advection scheme (based on our particle swarm implementation)
    """

    instances = 0   # count how many of these there are in order to create unique private mesh variable ids

    @timing.routine_timer_decorator
    def __init__(self, 
                 mesh       : uw.mesh.MeshClass, 
                 u_Field    : uw.mesh.MeshVariable = None, 
                 V_Field    : uw.mesh.MeshVariable = None, 
                 degree     : int  = 2,
                 theta      : float = 0.5,
                 solver_name: str = "adv_diff_",
                 restore_points_func: Callable = None,
                 verbose      = False):


        AdvDiffusion.instances += 1

        ## Todo: this is obviously not particularly robust

        if solver_name != "" and not solver_name.endswith("_"):
            self.petsc_options_prefix = solver_name+"_"
        else:
            self.petsc_options_prefix = solver_name

        self.petsc_options = PETSc.Options(self.petsc_options_prefix)

        # Here we can set some defaults for this set of KSP / SNES solvers
        self.petsc_options["snes_type"] = "newtonls"
        self.petsc_options["ksp_rtol"] = 1.0e-3
        self.petsc_options["ksp_monitor"] = None
        self.petsc_options["ksp_type"] = "fgmres"
        self.petsc_options["pc_type"] = "gamg"
        self.petsc_options["pc_gamg_type"] = "agg"
        self.petsc_options["snes_converged_reason"] = None
        self.petsc_options["snes_monitor_short"] = None
        # self.petsc_options["snes_view"] = None
        self.petsc_options["snes_rtol"] = 1.0e-3

        ## Todo: some validity checking on the size / type of u_Field supplied
        if not u_Field:
            self._u = uw.mesh.MeshVariable( mesh=mesh, num_components=1, name="u_adv_diff", vtype=uw.VarType.SCALAR, degree=degree )
        else:
            self._u = u_Field


        self._V = V_Field

        self.mesh = mesh
        self.k = 1.
        self.f = 0.
        self.delta_t = 1.0
        self.theta = theta
        self.restore_points_to_domain_func = restore_points_func

        self.bcs = []

        self.is_setup = False
        self.verbose = verbose

        # Build the DM / FE structures (should be done on remeshing)

        self._build_dm_and_mesh_discretisation()
        self._rebuild_after_mesh_update = self._build_dm_and_mesh_discretisation

        # Some other setup 

        self.mesh._equation_systems_register.append(self)

        # Add the nodal point swarm which we'll use to track the characteristics

        # There seems to be an issue with points launched from proc. boundaries
        # and managing the deletion of points, so a small perturbation to the coordinate
        # might fix this.

        nswarm = uw.swarm.Swarm(self.mesh)
        nT1 = uw.swarm.SwarmVariable("advdiff_Tstar_{}".format(self.instances), nswarm, 1)
        nX0 = uw.swarm.SwarmVariable("advdiff_X0_{}".format(self.instances), nswarm, nswarm.dim)

        nswarm.dm.finalizeFieldRegister()
        nswarm.dm.addNPoints(self._u.coords.shape[0]+1) # why + 1 ? That's the number of spots actually allocated
        cellid = nswarm.dm.getField("DMSwarm_cellid")
        coords = nswarm.dm.getField("DMSwarmPIC_coor").reshape( (-1, nswarm.dim) )
        coords[...] = self._u.coords[...] # + perturbation
        cellid[:] = self.mesh.get_closest_cells(coords)

        # Move slightly within the chosen cell to avoid edge effects 
        centroid_coords = self.mesh._centroids[cellid]
        shift = 1.0e-4 * self.mesh.get_min_radius()
        coords[...] = (1.0 - shift) * coords[...] + shift * centroid_coords[...]

        nswarm.dm.restoreField("DMSwarmPIC_coor")
        nswarm.dm.restoreField("DMSwarm_cellid")
        nswarm.dm.migrate(remove_sent_points=True)

        self._nswarm  = nswarm
        self._u_star  = nT1
        self._X0      = nX0

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
    def delta_t(self):
        return self._delta_t
    @delta_t.setter
    def delta_t(self, value):
        self.is_setup = False
        self._delta_t = sympify(value)

    @property
    def theta(self):
        return self._theta
    @theta.setter
    def theta(self, value):
        self.is_setup = False
        self._theta = sympify(value)


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
        self._f0 = -self.f + (self.u.fn - self._u_star.fn) / self.delta_t

        # f1 residual term
        self._f1 = gradient(self.u.fn*self.theta + self._u_star.fn*(1.0-self.theta))*self.k

        # g0 jacobian term
        self._g0 = diff_fn1_wrt_fn2(self._f0, self.u.fn)  

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

        ## LM - this would be more legible if we used a dictionary to name the functions / jacobians rather than
        ## refer to them via a tuple

        prim_field_list = [self.u,]
        cdef PtrContainer ext = getext(self.mesh, tuple(fns_residual), tuple(fns_jacobian), [x[1] for x in self.bcs], primary_field_list=prim_field_list)

        # set functions 
        self.dm.createDS()
        cdef DS ds = self.dm.getDS()
        PetscDSSetResidual(ds.ds, 0, ext.fns_residual[0], ext.fns_residual[1])

        # TODO: check if there's a significant performance overhead in passing in 
        # identically `zero` pointwise functions instead of setting to `NULL`
        # Since this will make everything slot together better !
        PetscDSSetJacobian(ds.ds, 0, 0, ext.fns_jacobian[0], ext.fns_jacobian[1], NULL, ext.fns_jacobian[2])
        
        
        cdef int ind=1
        cdef int [::1] comps_view  # for numpy memory view
        cdef DM cdm = self.dm

        for index,bc in enumerate(self.bcs):
            comps_view = bc.components
            for boundary in bc.boundaries:
                if self.verbose and  mpi4py.MPI.COMM_WORLD.rank==0:
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
              timestep       : float = 1.0,
              coords         : np.ndarray = None,
              _force_setup   : bool =False ):
        """
        Generates solution to constructed system.

        Params
        ------
        zero_init_guess:
            If `True`, a zero initial guess will be used for the 
            system solution. Otherwise, the current values of `self.u` will be used.
        """

        if timestep != self.delta_t:
            self.delta_t = timestep    # this will force an initialisation because the functions need to be updated

        if (not self.is_setup) or _force_setup:
            self._setup_terms()

        # mid pt update scheme should be preferred by default, but it is possible to supply
        # coords to over-ride this (e.g. rigid body rotation example)

        # placeholder definitions can be removed later
        nswarm = self._nswarm
        t_soln = self._u
        v_soln = self._V
        nX0 = self._X0
        nT1 = self._u_star
        delta_t = timestep

        with nswarm.access(nX0):
            nX0.data[...] = nswarm.data[...]

        with self.mesh.access():
            n_points = t_soln.data.shape[0]

        if coords is None: # Mid point method to find launch points (T*)

            # numerical_dt = self.estimate_dt()

            # if delta_t > 0.25*numerical_dt:
            #     substeps = int(np.floor(delta_t / (0.25* numerical_dt))) + 1
            # else:
            #    substeps = 1
           
            # with nswarm.access():
            #    if self.verbose and nswarm.data.shape[0] > 1.05 * n_points:
            #        print("1 - Swarm points {} = {} ({})".format(mpi4py.MPI.COMM_WORLD.rank,nswarm.data.shape[0], n_points), flush=True)

            with nswarm.access(nswarm.particle_coordinates):
                v_at_Vpts = uw.function.evaluate(v_soln.fn, nswarm.data).reshape(-1,self.mesh.dim)
                mid_pt_coords = nswarm.data[...] - 0.5 * delta_t * v_at_Vpts

                # validate_coords to ensure they live within the domain (or there will be trouble)
                if self.restore_points_to_domain_func is not None:
                    mid_pt_coords = self.restore_points_to_domain_func(mid_pt_coords)

                nswarm.data[...] = mid_pt_coords

            ## Let the swarm be updated, and then move the rest of the way

            with nswarm.access(nswarm.particle_coordinates):
                v_at_Vpts = uw.function.evaluate(v_soln.fn, nswarm.data).reshape(-1,self.mesh.dim)
                new_coords = nX0.data[...] - delta_t * v_at_Vpts

                # validate_coords to ensure they live within the domain (or there will be trouble)
                if self.restore_points_to_domain_func is not None:
                    new_coords = self.restore_points_to_domain_func(new_coords)

                nswarm.data[...] = new_coords
  

        else:  # launch points (T*) provided by omniscience user
            with nswarm.access(nswarm.particle_coordinates):
                nswarm.data[...] = coords[...]

        with nswarm.access(nT1):
            nT1.data[...] = uw.function.evaluate(t_soln.fn, nswarm.data).reshape(-1,1)

        # restore coords 
        with nswarm.access(nswarm.particle_coordinates):
            nswarm.data[...] = nX0.data[...]

        # Now solve the poisson equation which depends on the self._u_star field that
        # we have now computed


        gvec = self.dm.getGlobalVec()

        if not zero_init_guess:
            with self.mesh.access():
                self.dm.localToGlobal(self.u.vec, gvec)
        else:
            gvec.array[:] = 0.

        quad = self.petsc_fe_u.getQuadrature()
        for fe in [var.petsc_fe for var in self.mesh.vars.values()]:
            fe.setQuadrature(quad)

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


    @timing.routine_timer_decorator
    def estimate_dt(self):
        """
        Calculates an appropriate advective timestep for the given 
        mesh and velocity configuration.
        """
        # we'll want to do this on an element by element basis 
        # for more general mesh

        # first let's extract a max global velocity magnitude 
        import math
        with self.mesh.access():
            vel = self._V.data
            magvel_squared = vel[:,0]**2 + vel[:,1]**2 
            if self.mesh.dim ==3:
                magvel_squared += vel[:,2]**2 

            max_magvel = math.sqrt(magvel_squared.max())
        
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        max_magvel_glob = comm.allreduce( max_magvel, op=MPI.MAX)

        min_dx = self.mesh.get_min_radius()
        return min_dx/max_magvel_glob
