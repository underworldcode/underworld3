import sympy
from sympy import sympify
from sympy.vector import gradient, divergence
import numpy as np 

from typing import Optional, Callable

from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import SNES_Scalar, SNES_SaddlePoint
from .._jitextension import getext, diff_fn1_wrt_fn2
import underworld3.timing as timing


include "../petsc_extras.pxi"


class SNES_Poisson(SNES_Scalar):

    instances = 0

    @timing.routine_timer_decorator
    def __init__(self, 
                 mesh     : uw.mesh.MeshClass, 
                 u_Field  : uw.mesh.MeshVariable = None, 
                 degree     = 2,
                 solver_name: str = "",
                 verbose    = False):

        ## Keep track

        SNES_Poisson.instances += 1

        if solver_name == "":
            solver_name = "Poisson_{}_".format(self.instances)

        ## Parent class will set up default values etc
        super().__init__(mesh, u_Field, degree, solver_name, verbose)

        # Here we can set some defaults for this set of KSP / SNES solvers
        #self.petsc_options = PETSc.Options(self.petsc_options_prefix)
        #self.petsc_options["snes_type"] = "newtonls"
        #self.petsc_options["ksp_rtol"] = 1.0e-3
        #self.petsc_options["ksp_monitor"] = None
        #self.petsc_options["ksp_type"] = "fgmres"
        #self.petsc_options["pre_type"] = "gamg"
        #self.petsc_options["snes_converged_reason"] = None
        #self.petsc_options["snes_monitor_short"] = None
        #self.petsc_options["snes_view"] = None
        #self.petsc_options["snes_rtol"] = 1.0e-3

        # Register the problem setup function
        self._setup_problem_description = self.poisson_problem_description

        # default values for properties
        self.f = 0.0
        self.k = 1.0


    ## This function is the one we will typically over-ride to build specific solvers. 
    ## This example is a poisson-like problem with isotropic coefficients

    @timing.routine_timer_decorator
    def poisson_problem_description(self):

        dim = self.mesh.dim
        N   = self.mesh.N

        # f1 residual term (weighted integration) - scalar function
        self.F0  = -self.f

        # f1 residual term (integration by parts / gradients)
        # isotropic
        self.F1  = self.k * (self._L)

        return 

    @property
    def f(self):
        return self._f
    @f.setter
    def f(self, value):
        self.is_setup = False
        self._f = sympify(value)
    
    @property
    def k(self):
        return self._k
    @k.setter
    def k(self, value):
        self.is_setup = False
        self._k = sympify(value)
   


## --------------------------------
## Stokes saddle point solver plus
## ancilliary functions - note that 
## we need to update the description
## of the generic saddle pt solver
## to remove the Stokes-specific stuff
## --------------------------------


class SNES_Stokes(SNES_SaddlePoint):

    instances = 0

    def __init__(self, 
                 mesh          : uw.mesh.MeshClass, 
                 velocityField : Optional[uw.mesh.MeshVariable] =None,
                 pressureField : Optional[uw.mesh.MeshVariable] =None,
                 u_degree      : Optional[int]                           =2, 
                 p_degree      : Optional[int]                           =None,
                 solver_name   : Optional[str]                           ="",
                 verbose       : Optional[str]                           =False,
                 penalty       : Optional[float]                         = 0.0,
                 _Ppre_fn      = None
                ):
        """
        This class provides functionality for a discrete representation
        of the Stokes flow equations.

        Specifically, the class uses a mixed finite element implementation to
        construct a system of linear equations which may then be solved.

        The strong form of the given boundary value problem, for :math:`f`,
        :math:`g` and :math:`h` given, is

        .. math::
            \\begin{align}
            \\sigma_{ij,j} + f_i =& \\: 0  & \\text{ in }  \\Omega \\\\
            u_{k,k} =& \\: 0  & \\text{ in }  \\Omega \\\\
            u_i =& \\: g_i & \\text{ on }  \\Gamma_{g_i} \\\\
            \\sigma_{ij}n_j =& \\: h_i & \\text{ on }  \\Gamma_{h_i} \\\\
            \\end{align}

        where,

        * :math:`\\sigma_{i,j}` is the stress tensor
        * :math:`u_i` is the velocity,
        * :math:`p`   is the pressure,
        * :math:`f_i` is a body force,
        * :math:`g_i` are the velocity boundary conditions (DirichletCondition)
        * :math:`h_i` are the traction boundary conditions (NeumannCondition).

        The problem boundary, :math:`\\Gamma`,
        admits the decompositions :math:`\\Gamma=\\Gamma_{g_i}\\cup\\Gamma_{h_i}` where
        :math:`\\emptyset=\\Gamma_{g_i}\\cap\\Gamma_{h_i}`. The equivalent weak form is:

        .. math::
            \\int_{\Omega} w_{(i,j)} \\sigma_{ij} \\, d \\Omega = \\int_{\\Omega} w_i \\, f_i \\, d\\Omega + \sum_{j=1}^{n_{sd}} \\int_{\\Gamma_{h_j}} w_i \\, h_i \\,  d \\Gamma

        where we must find :math:`u` which satisfies the above for all :math:`w`
        in some variational space.

        Parameters
        ----------
        mesh : 
            The mesh object which forms the basis for problem discretisation,
            domain specification, and parallel decomposition.
        velocityField :
            Optional. Variable used to record system velocity. If not provided,
            it will be generated and will be available via the `u` stokes object property.
        pressureField :
            Optional. Variable used to record system pressure. If not provided,
            it will be generated and will be available via the `p` stokes object property.
            If provided, it is up to the user to ensure that it is of appropriate order
            relative to the provided velocity variable (usually one order lower degree).
        u_degree :
            Optional. The polynomial degree for the velocity field elements.
        p_degree :
            Optional. The polynomial degree for the pressure field elements. 
            If provided, it is up to the user to ensure that it is of appropriate order
            relative to the provided velocitxy variable (usually one order lower degree).
            If not provided, it will be set to one order lower degree than the velocity field.
        solver_name :
            Optional. The petsc options prefix for the SNES solve. This is important to provide
            a name space when multiples solvers are constructed that may have different SNES options.
            For example, if you name the solver "stokes", the SNES options such as `snes_rtol` become `stokes_snes_rtol`.
            The default is blank, and an underscore will be added to the end of the solver name if not already present.
 
        Notes
        -----
        Constructor must be called by collectively all processes.

        """       

        SNES_Stokes.instances += 1

        if solver_name == "":
            solver_name = "Stokes_{}_".format(self.instances)

        super().__init__(mesh, velocityField, pressureField, 
                         u_degree, p_degree,
                         solver_name,verbose, _Ppre_fn )

        self.penalty = 0.0

        # User-facing operations are matrices / vectors by preference
        self._E = self._L + self._L.transpose()/2
        self._Einv2 = sympy.sqrt((sympy.Matrix(self._E)**2).trace()) # scalar 2nd invariant


        # Here we can set some petsc defaults for this solver
        # self.petsc_options = PETSc.Options(self.petsc_options_prefix)
        # self.petsc_options["snes_type"] = "newtonls"
        # self.petsc_options["ksp_rtol"] = 1.0e-3
        # self.petsc_options["ksp_monitor"] = None
        # self.petsc_options["ksp_type"] = "fgmres"
        # self.petsc_options["pre_type"] = "gamg"
        # self.petsc_options["snes_converged_reason"] = None
        # self.petsc_options["snes_monitor_short"] = None
        # self.petsc_options["snes_view"] = None
        # self.petsc_options["snes_rtol"] = 1.0e-3
        # self.petsc_options["pc_type"] = "fieldsplit"
        # self.petsc_options["pc_fieldsplit_type"] = "schur"
        # self.petsc_options["pc_fieldsplit_schur_factorization_type"] = "full"
        # self.petsc_options["pc_fieldsplit_schur_precondition"] = "a11"
        # self.petsc_options["fieldsplit_velocity_ksp_type"] = "fgmres"
        # self.petsc_options["fieldsplit_velocity_ksp_rtol"] = 1.0e-4
        # self.petsc_options["fieldsplit_velocity_pc_type"]  = "gamg"
        # self.petsc_options["fieldsplit_pressure_ksp_rtol"] = 3.e-4
        # self.petsc_options["fieldsplit_pressure_pc_type"] = "gamg" 


        self._setup_problem_description = self.stokes_problem_description

        # this attrib records if we need to re-setup
        self.is_setup = False

        return
   
    @timing.routine_timer_decorator
    def stokes_problem_description(self):

        dim = self.mesh.dim
        N = self.mesh.N

        # residual terms can be redefined here 

        # terms that become part of the weighted integral
        self.UF0 = -self.bodyforce

        # Integration by parts into the stiffness matrix
        self.UF1 = self.stress + self.penalty * self.div_u * sympy.eye(dim)

        # forces in the constraint (pressure) equations
        self.PF0 = self.div_u

        return 

    ## note ... this is probably over-simple
    ## due to isotropy. Once anisotropy is allowed, sympy
    ## is going to require us to work with NDim arrays in place of
    ## matrices ... but they need to go back to matrices for the 
    ## pointwise function evaluation
     
    @property
    def strainrate(self):
        return sympy.Matrix(self._E)
    @property
    def stress_deviator(self):
        return 2*self.viscosity*self.strainrate
    @property
    def stress(self):
        return self.stress_deviator - sympy.eye(self.mesh.dim)*self.p.fn
    @property
    def div_u(self):
        return divergence(self.u.fn)
    @property
    def viscosity(self):
        return self._viscosity
    @viscosity.setter
    def viscosity(self, value):
        self.is_setup = False
        symval = sympify(value)
        if isinstance(symval, sympy.vector.Vector):
            raise RuntimeError("Viscosity appears to be a vector quantity. Scalars are required.")
        if isinstance(symval, sympy.Matrix):
            raise RuntimeError("Viscosity appears to be a matrix quantity. Scalars are required.")
        self._viscosity = symval

    @property
    def bodyforce(self):
        return self._bodyforce
    @bodyforce.setter
    def bodyforce(self, value):
        self.is_setup = False
        symval = sympify(value)
        # if not isinstance(symval, sympy.vector.Vector):
        #     raise RuntimeError("Body force term must be a vector quantity.")
        self._bodyforce = symval

    @property
    def penalty(self):
        return self._penalty
    @penalty.setter
    def penalty(self, value):
        self.is_setup = False
        symval = sympify(value)
        self._penalty = symval



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
            vel = self.u.data
            magvel_squared = vel[:,0]**2 + vel[:,1]**2 
            if self.mesh.dim ==3:
                magvel_squared += vel[:,2]**2 

            max_magvel = math.sqrt(magvel_squared.max())
        
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        max_magvel_glob = comm.allreduce( max_magvel, op=MPI.MAX)

        min_dx = self.mesh.get_min_radius()
        return min_dx/max_magvel_glob


## --------------------------------
## Project from pointwise functions
## nodal point unknowns 
## --------------------------------

class SNES_Projection(SNES_Scalar):
    """
    Generic solver that is used to project functions to nodal points.
    At present this is only a name-wrapper on the SNES_scalar class
    but we can make it specific to the job ... perhaps callable with
    the RHS as a parameter.
    """

    instances = 0

    @timing.routine_timer_decorator
    def __init__(self, 
                 mesh     : uw.mesh.MeshClass, 
                 u_Field  : uw.mesh.MeshVariable = None, 
                 degree     = 2,
                 solver_name: str = "projection_",
                 verbose    = False):

        SNES_Projection.instances += 1

        if solver_name == "":
            solver_name = "Projection_{}_".format(self.instances)

        super().__init__(mesh, 
                         u_Field,
                         degree, 
                         solver_name, verbose,  )

        # self._setup_problem_description = self.projection_problem_description

        self.is_setup = False

        return


    @timing.routine_timer_decorator
    def projection_problem_description(self):

        dim = self.mesh.dim
        N = self.mesh.N

        # residual terms - could add smoothing here ... 
        # otherwise there is nothing to do that is different from
        # the generic class

        return 

#################################################
# Characteristics-based advection-diffusion 
# solver based on SNES_Poisson and swarm-to-nodes
#
# Note that the solve() method has the swarm 
# handler. 
#################################################

class SNES_AdvectionDiffusion_SLCN(SNES_Poisson):

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
                 solver_name: str = "",
                 restore_points_func: Callable = None,
                 verbose      = False):


        SNES_AdvectionDiffusion_SLCN.instances += 1

        if solver_name == "":
            solver_name = "AdvDiff_slcn_{}_".format(self.instances)

        ## Parent class will set up default values etc
        super().__init__(mesh, u_Field, degree, solver_name, verbose)

        # These are unique to the advection solver
        self._V = V_Field
        self._Lstar =  sympy.derive_by_array(self._U, self._X).reshape(self.mesh.dim)

        self.delta_t = 1.0
        self.theta = theta

        self.restore_points_to_domain_func = restore_points_func
        self._setup_problem_description = self.adv_diff_slcn_problem_description

        self.is_setup = False

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

        return


    def adv_diff_slcn_problem_description(self):

        N = self.mesh.N

        # f0 residual term
        self.F0 = -self.f + (self.u.fn - self._u_star.fn) / self.delta_t

        # f1 residual term
        self.F1 = (self.theta * self._L + (1.0-self.theta) * self._Lstar) * self.k
        
        return

    @property
    def u(self):
        return self._u

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

        # Over to you Poisson Solver

        super().solve(zero_init_guess, _force_setup )

        return