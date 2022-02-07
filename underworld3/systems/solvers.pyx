import sympy
from sympy import sympify
from sympy.vector import gradient, divergence

from typing import Optional


from petsc4py import PETSc

import underworld3 
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


        # Should be done by super().__init__() ... 

        # self._build_dm_and_mesh_discretisation()
        # self._rebuild_after_mesh_update = self._build_dm_and_mesh_discretisation
        # self.mesh._equation_systems_register.append(self)

        # Register the problem setup function

        self._setup_problem_description = self.poisson_problem_description



    ## This function is the one we will typically over-ride to build specific solvers. 
    ## This example is a poisson-like problem with isotropic coefficients

    @timing.routine_timer_decorator
    def poisson_problem_description(self):

        dim = self.mesh.dim
        N   = self.mesh.N

        # f1 residual term (weighted integration) - scalar function
        self._f0 = -self.f

        # f1 residual term (integration by parts / gradients)
        self._f1 = self.k * (self._L)

        return 


## --------------------------------
## Stokes saddle point solver plus
## ancilliary functions 
## --------------------------------

class SNES_Stokes(SNES_SaddlePoint):

    instances = 0

    def __init__(self, 
                 mesh          : underworld3.mesh.MeshClass, 
                 velocityField : Optional[underworld3.mesh.MeshVariable] =None,
                 pressureField : Optional[underworld3.mesh.MeshVariable] =None,
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

    @property
    def penalty(self):
        return self._penalty
    @penalty.setter
    def penalty(self, value):
        self.is_setup = False
        symval = sympify(value)
        self._penalty = symval

    

    @timing.routine_timer_decorator
    def stokes_problem_description(self):

        dim = self.mesh.dim
        N = self.mesh.N

        # residual terms 

        # terms that become part of the weighted integral
        self._u_f0 = -self.bodyforce
        # Integration by parts into the stiffness matrix
        self._u_f1 = self.stress 
        # forces in the constraint (pressure) equations
        self._p_f0 = self.div_u

        return 


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

        self._g = sympy.Array([0.0, 0.0])

        self._setup_problem_description = self.projection_problem_description

        self.is_setup = False



        return

    @property
    def g(self):
        return self._g
    @g.setter
    def g(self, value):
        self.is_setup = False
        # should add test here to make sure g is conformal
        self._g = sympy.Array(sympify(value)).reshape(self.mesh.dim)

    @timing.routine_timer_decorator
    def projection_problem_description(self):

        dim = self.mesh.dim
        N = self.mesh.N

        # residual terms
        self._f0 = -self.f

        # f1 residual term
        self._f1 = self.g

        return 


 
