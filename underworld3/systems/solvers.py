import sympy
from sympy import sympify
from sympy.vector import gradient, divergence
import numpy as np

from typing import Optional, Callable

from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import SNES_Scalar, SNES_Vector, SNES_Stokes, SNES_SaddlePoint
import underworld3.timing as timing


class SNES_Poisson(SNES_Scalar):
    r"""
    SNES-based poisson equation solver

    """

    instances = 0

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        u_Field: uw.discretisation.MeshVariable = None,
        solver_name: str = "",
        verbose=False,
    ):

        ## Keep track

        SNES_Poisson.instances += 1

        if solver_name == "":
            solver_name = "Poisson_{}_".format(self.instances)

        ## Parent class will set up default values etc
        super().__init__(mesh, u_Field, solver_name, verbose)

        # Register the problem setup function
        self._setup_problem_description = self.poisson_problem_description

        # default values for properties
        self.f = sympy.Matrix.zeros(1, 1)

    ## This function is the one we will typically over-ride to build specific solvers.
    ## This example is a poisson-like problem with isotropic coefficients

    @timing.routine_timer_decorator
    def poisson_problem_description(self):

        dim = self.mesh.dim
        N = self.mesh.N

        # f1 residual term (weighted integration) - scalar function
        self._f0 = self.F0 - self.f

        # f1 residual term (integration by parts / gradients)
        # isotropic
        self._f1 = (
            self.F1 + self.constitutive_model.flux(self._L).T
        )  # self.k * (self._L)

        return

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        self.is_setup = False
        self._f = sympy.Matrix((value,))


class SNES_Darcy(SNES_Scalar):
    r"""
    Darcy docstring ...
    """

    instances = 0

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        u_Field: uw.discretisation.MeshVariable,
        v_Field: uw.discretisation.MeshVariable,
        solver_name: str = "",
        verbose=False,
    ):

        ## Keep track

        SNES_Darcy.instances += 1

        if solver_name == "":
            solver_name = "Darcy_{}_".format(self.instances)

        ## Parent class will set up default values etc
        super().__init__(mesh, u_Field, solver_name, verbose)

        # Register the problem setup function
        self._setup_problem_description = self.darcy_problem_description

        # default values for properties
        self._f = 0.0
        self._k = 1.0

        self._s = sympy.Matrix.zeros(rows=1, cols=self.mesh.dim)
        self._s[1] = -1.0

        self._v = v_Field

        ## Set up the projection operator that
        ## solves the flow rate

        self._v_projector = uw.systems.solvers.SNES_Vector_Projection(self.mesh, self.v)

        # If we add smoothing, it should be small relative to actual diffusion (self.viscosity)
        self._v_projector.smoothing = 0.0

    ## This function is the one we will typically over-ride to build specific solvers.
    ## This example is a poisson-like problem with isotropic coefficients

    @timing.routine_timer_decorator
    def darcy_problem_description(self):

        dim = self.mesh.dim
        N = self.mesh.N

        # f1 residual term (weighted integration)
        self._f0 = self.F0 - self.f

        # f1 residual term (integration by parts / gradients)
        self._f1 = self.F1 + self.darcy_flux

        # Flow calculation
        self._v_projector.uw_function = -self.darcy_flux

        return

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        self.is_setup = False
        self._f = sympy.Matrix((value,))

    @property
    def s(self):
        return self._s

    @s.setter
    def s(self, value):
        self.is_setup = False
        self._s = sympy.Matrix((value,))

    @property
    def darcy_flux(self):
        flux = self.constitutive_model.flux(self._L - self.s).T
        return flux

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, value):
        self._v_projector.is_setup = False
        self._v = sympify(value)

    @timing.routine_timer_decorator
    def solve(
        self,
        zero_init_guess: bool = True,
        timestep: float = None,
        _force_setup: bool = False,
    ):
        """
        Generates solution to constructed system.

        Params
        ------
        zero_init_guess:
            If `True`, a zero initial guess will be used for the
            system solution. Otherwise, the current values of `self.u` will be used.
        timestep:
            value used to evaluate inertial contribution
        """

        if (not self.is_setup) or _force_setup:
            self._setup_terms()

        # Solve pressure

        super().solve(zero_init_guess, _force_setup)

        # Now solve flow field

        self._v_projector.petsc_options[
            "snes_type"
        ] = "newtontr"  ## newtonls seems to be problematic when the previous guess is available
        self._v_projector.petsc_options["snes_rtol"] = 1.0e-5
        self._v_projector.petsc_options.delValue("ksp_monitor")
        self._v_projector.solve(zero_init_guess)

        return

    @timing.routine_timer_decorator
    def _setup_terms(self):
        self._v_projector.uw_function = self.darcy_flux
        self._v_projector._setup_terms()
        super()._setup_terms()


## --------------------------------
## Stokes saddle point solver plus
## ancilliary functions - note that
## we need to update the description
## of the generic saddle pt solver
## to remove the Stokes-specific stuff
## --------------------------------


class SNES_Stokes(SNES_Stokes):
    r"""
    This class provides functionality for a discrete representation
    of the Stokes flow equations assuming an incompressibility
    (or near-incompressibility) constraint.

    $$\frac{\partial}{\partial x_j} \left( \frac{\eta}{2} \left[ \frac{\partial u_i}{\partial x_j}  +
            \frac{\partial u_j}{\partial x_i} \right]\right) - \frac{\partial p}{\partial x_i} = f_i$$

    $$\frac{\partial u_i}{\partial x_i} = 0$$

    ## Properties

      - The viscosity, \( \eta \) is provided by setting the `constitutive_model` property to
    one of the `uw.systems.constitutive_models` classes and populating the parameters.
    It is usually a constant or a function of position / time and may also be non-linear
    or anisotropic.

      - The bodyforce term, \( f_i \) is provided through the `bodyforce` property.

      - The Augmented Lagrangian approach to application of the incompressibility
    constraint is to penalise incompressibility in the Stokes equation by adding
    \( \lambda \nabla \cdot \mathbf{u} \) when the weak form of the equations is constructed.
    (this is in addition to the constraint equation, unlike in the classical penalty method).
    This is activated by setting the `penalty` property to a non-zero floating point value.

      - A preconditioner is usually required for the saddle point system and this is provided
    though the `saddle_preconditioner` property. A common choice is \( 1/ \eta \) or
    \( 1 / \eta + 1/ \lambda \) if a penalty is used


    ## Notes

      - The interpolation order of the `pressureField` variable is used to determine the integration order of
    the mixed finite element method and is usually lower than the order of the `velocityField` variable.

      - It is possible to set discontinuous pressure variables by setting the `p_continous` option to `False`
    (currently this is not implemented).

      - The `solver_name` parameter sets the namespace for PETSc options and should be unique and
    compatible with the PETSc naming conventions.


    """

    instances = 0

    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        velocityField: uw.discretisation.MeshVariable,
        pressureField: uw.discretisation.MeshVariable,
        solver_name: Optional[str] = "",
        verbose: Optional[str] = False,
    ):

        SNES_Stokes.instances += 1

        if solver_name == "":
            solver_name = "Stokes_{}_".format(self.instances)

        super().__init__(mesh, velocityField, pressureField, solver_name, verbose)

        # User-facing operations are matrices / vectors by preference

        self._E = self.mesh.vector.strain_tensor(self._u.sym)
        self._Estar = None

        # scalar 2nd invariant (incompressible)
        self._Einv2 = sympy.sqrt((sympy.Matrix(self._E) ** 2).trace() / 2)
        self._penalty = 0.0
        self._constraints = sympy.Matrix(
            (self.div_u,)
        )  # by default, incompressibility constraint

        self._saddle_preconditioner = sympy.sympify(1)
        self._bodyforce = sympy.Matrix([0] * self.mesh.dim)

        self._setup_problem_description = self.stokes_problem_description

        # this attrib records if we need to re-setup
        self.is_setup = False

        return

    @timing.routine_timer_decorator
    def stokes_problem_description(self):

        dim = self.mesh.dim
        N = self.mesh.N

        # residual terms can be redefined here. We leave the
        # UF0, UF1, PF0 terms in place to allow injection of
        # additional terms. These are pre-defined to be zero

        # terms that become part of the weighted integral
        self._u_f0 = self.UF0 - self.bodyforce
        # Integration by parts into the stiffness matrix (constitutive terms)
        self._u_f1 = self.UF1 + self.stress + self.penalty * self.div_u * sympy.eye(dim)

        # forces in the constraint (pressure) equations
        self._p_f0 = self.PF0 + sympy.Matrix((self.constraints))

        return

    @property
    def strainrate(self):
        return sympy.Matrix(self._E)

    # Set this to something that supplies history if relevant
    @property
    def strainrate_star(self):
        return None

    # provide the strain-rate history in symbolic form
    @strainrate_star.setter
    def strainrate_star(self, strain_rate_fn):
        self._is_setup = False
        symval = sympify(strain_rate_fn)
        self._Estar = symval

    # This should return standard viscous behaviour if strainrate_star is None
    @property
    def stress_deviator(self):
        return self.constitutive_model.flux(self.strainrate, self.strainrate_star)

    @property
    def stress(self):
        return self.stress_deviator - sympy.eye(self.mesh.dim) * (self.p.sym[0])

    @property
    def div_u(self):
        E = self.strainrate
        divergence = E.trace()
        return divergence

    @property
    def constraints(self):
        return self._constraints

    @constraints.setter
    def constraints(self, constraints_matrix):
        self._is_setup = False
        symval = sympify(constraints_matrix)
        self._constraints = symval

    @property
    def bodyforce(self):
        return self._bodyforce

    @bodyforce.setter
    def bodyforce(self, value):
        self.is_setup = False
        self._bodyforce = self.mesh.vector.to_matrix(value)

    @property
    def saddle_preconditioner(self):
        return self._saddle_preconditioner

    @saddle_preconditioner.setter
    def saddle_preconditioner(self, value):
        self.is_setup = False
        symval = sympify(value)
        self._saddle_preconditioner = symval

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
            magvel_squared = vel[:, 0] ** 2 + vel[:, 1] ** 2
            if self.mesh.dim == 3:
                magvel_squared += vel[:, 2] ** 2

            max_magvel = math.sqrt(magvel_squared.max())

        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        max_magvel_glob = comm.allreduce(max_magvel, op=MPI.MAX)

        min_dx = self.mesh.get_min_radius()
        return min_dx / max_magvel_glob


## --------------------------------
## Project from pointwise functions
## nodal point unknowns
## --------------------------------


class SNES_Projection(SNES_Scalar):
    """
    Map underworld (pointwise) function to continuous
    nodal point values in least-squares sense.

    Solver can be given boundary conditions that
    the continuous function needs to satisfy and
    non-linear constraints will be handled by SNES.

    Consitutive model for this solver is the identity tensor (purely for validation)

    """

    instances = 0

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        u_Field: uw.discretisation.MeshVariable = None,
        solver_name: str = "",
        verbose=False,
    ):

        SNES_Projection.instances += 1

        if solver_name == "":
            solver_name = "SProj_{}_".format(self.instances)

        super().__init__(mesh, u_Field, solver_name, verbose)

        self._setup_problem_description = self.projection_problem_description
        self.is_setup = False
        self._smoothing = 0.0
        self._uw_weighting_function = 1.0
        self._constitutive_model = uw.systems.constitutive_models.Constitutive_Model(
            self.mesh.dim, 1
        )

        return

    @timing.routine_timer_decorator
    def projection_problem_description(self):

        dim = self.mesh.dim
        N = self.mesh.N

        # residual terms - defines the problem:
        # solve for a best fit to the continuous mesh
        # variable given the values in self.function
        # F0 is left in place for the user to inject
        # non-linear constraints if required

        self._f0 = (
            self.F0 + (self.u.sym - self.uw_function) * self.uw_weighting_function
        )

        # F1 is left in the users control ... e.g to add other gradient constraints to the stiffness matrix

        self._f1 = self.F1 + self.smoothing * self.mesh.vector.gradient(self.u.sym)

        return

    @property
    def uw_function(self):
        return self._uw_function

    @uw_function.setter
    def uw_function(self, user_uw_function):
        self.is_setup = False
        self._uw_function = sympy.Matrix([user_uw_function])

    @property
    def smoothing(self):
        return self._smoothing

    @smoothing.setter
    def smoothing(self, smoothing_factor):
        self.is_setup = False
        self._smoothing = sympify(smoothing_factor)

    @property
    def uw_weighting_function(self):
        return self._uw_weighting_function

    @uw_weighting_function.setter
    def uw_weighting_function(self, user_uw_function):
        self.is_setup = False
        self._uw_weighting_function = user_uw_function


## --------------------------------
## Project from pointwise vector
## functions to nodal point unknowns
##
## We can add boundary constraints in the usual way (parent class handles this)
## We can add smoothing (which takes the form of a viscosity term)
## We could consider an incompressibility constraint ... or remove null spaces
## --------------------------------


class SNES_Vector_Projection(SNES_Vector):
    """
    Map underworld (pointwise) function to continuous
    nodal point values in least-squares sense.

    Solver can be given boundary conditions that
    the continuous function needs to satisfy and
    non-linear constraints will be handled by SNES

    Consitutive model for this solver is the identity tensor (purely for validation)
    """

    instances = 0

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        u_Field: uw.discretisation.MeshVariable = None,
        solver_name: str = "",
        verbose=False,
    ):

        SNES_Vector_Projection.instances += 1

        if solver_name == "":
            solver_name = "VProj{}_".format(self.instances)

        super().__init__(mesh, u_Field, u_Field.degree, solver_name, verbose)

        self._setup_problem_description = self.projection_problem_description
        self.is_setup = False
        self._smoothing = 0.0
        self._penalty = 0.0
        self._uw_weighting_function = 1.0
        self._constitutive_model = uw.systems.constitutive_models.Constitutive_Model(
            self.mesh.dim, self.mesh.dim
        )

        return

    @timing.routine_timer_decorator
    def projection_problem_description(self):

        dim = self.mesh.dim
        N = self.mesh.N

        # residual terms - defines the problem:
        # solve for a best fit to the continuous mesh
        # variable given the values in self.function
        # F0 is left in place for the user to inject
        # non-linear constraints if required

        self._f0 = (
            self.F0 + (self.u.sym - self.uw_function) * self.uw_weighting_function
        )

        # F1 is left in the users control ... e.g to add other gradient constraints to the stiffness matrix

        E = 0.5 * (sympy.Matrix(self._L) + sympy.Matrix(self._L).T)  # ??
        self._f1 = (
            self.F1
            + self.smoothing * E
            + self.penalty
            * self.mesh.vector.divergence(self.u.sym)
            * sympy.eye(self.mesh.dim)
        )

        return

    @property
    def uw_function(self):
        return self._uw_function

    @uw_function.setter
    def uw_function(self, user_uw_function):
        self.is_setup = False
        self._uw_function = user_uw_function

    @property
    def smoothing(self):
        return self._smoothing

    @smoothing.setter
    def smoothing(self, smoothing_factor):
        self.is_setup = False
        self._smoothing = sympify(smoothing_factor)

    @property
    def penalty(self):
        return self._penalty

    @penalty.setter
    def penalty(self, value):
        self.is_setup = False
        symval = sympify(value)
        self._penalty = symval

    @property
    def uw_weighting_function(self):
        return self._uw_weighting_function

    @uw_weighting_function.setter
    def uw_weighting_function(self, user_uw_function):
        self.is_setup = False
        self._uw_weighting_function = user_uw_function


## --------------------------------
## Project from pointwise vector
## functions to nodal point unknowns
##
## We can add boundary constraints in the usual way (parent class handles this)
## We can add smoothing (which takes the form of a viscosity term)
## Here we add an incompressibility constraint
## --------------------------------


## Does not seem to be a well posed problem as currently written ...
## We will fall back to penalising the standard SNES_Vector

'''
class SNES_Solenoidal_Vector_Projection(SNES_Stokes):
    """
    Map underworld (pointwise) function to continuous
    nodal point values in least-squares sense.

    Solver can be given boundary conditions that
    the continuous function needs to satisfy and
    non-linear constraints will be handled by SNES"""

    instances = 0

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        u_Field: uw.discretisation.MeshVariable = None,
        solver_name: str = "",
        verbose=False,
    ):

        SNES_Solenoidal_Vector_Projection.instances += 1

        self._smoothing = 0.0
        self._uw_weighting_function = 1.0

        if solver_name == "":
            solver_name = "iVProj{}_".format(self.instances)

        self._constraint_field = uw.discretisation.MeshVariable(
            r"\lambda^{}".format(self.instances),
            mesh=mesh,
            num_components=1,
            vtype=uw.VarType.SCALAR,
            degree=u_Field.degree - 1,
            continuous=False,
        )

        super().__init__(
            mesh,
            u_Field,
            self._constraint_field,
            solver_name,
            verbose,
        )

        self._setup_problem_description = (
            self.constrained_projection_problem_description
        )
        self.is_setup = False

        return

    @timing.routine_timer_decorator
    def constrained_projection_problem_description(self):

        dim = self.mesh.dim
        N = self.mesh.N

        # residual terms - defines the problem:
        # solve for a best fit to the continuous mesh
        # variable given the values in self.function
        # F0 is left in place for the user to inject
        # non-linear constraints if required

        self._u_f0 = (
            self.UF0 + (self.u.sym - self.uw_function) * self.uw_weighting_function
        )

        # Integration by parts into the stiffness matrix
        self._u_f1 = (
            self.UF1
            + self.smoothing * (sympy.Matrix(self._L) + sympy.Matrix(self._L).T)
            - self._constraint_field.fn * sympy.Matrix.eye(dim)
        )

        # rhs in the constraint (pressure) equations
        self._p_f0 = self.PF0 + sympy.Matrix([self.mesh.vector.divergence(self.u.sym)])

        return

    @property
    def uw_function(self):
        return self._uw_function

    @uw_function.setter
    def uw_function(self, user_uw_function):
        self.is_setup = False
        self._uw_function = user_uw_function

    @property
    def uw_weighting_function(self):
        return self._uw_weighting_function

    @uw_weighting_function.setter
    def uw_weighting_function(self, user_uw_function):
        self.is_setup = False
        self._uw_weighting_function = user_uw_function

    @property
    def smoothing(self):
        return self._smoothing

    @smoothing.setter
    def smoothing(self, smoothing_factor):
        self.is_setup = False
        self._smoothing = sympify(smoothing_factor)
        if self._smoothing != 0.0:
            self.saddle_preconditioner = 1.0 / self._smoothing
'''

#################################################
# Characteristics-based advection-diffusion
# solver based on SNES_Poisson and swarm-to-nodes
#
# Note that the solve() method has the swarm
# handler.
#################################################


class SNES_AdvectionDiffusion_SLCN(SNES_Poisson):

    """Characteristics-based advection diffusion solver:

    Uses a theta timestepping approach with semi-Lagrange sample backwards in time using
    a mid-point advection scheme (based on our particle swarm implementation)
    """

    instances = 0  # count how many of these there are in order to create unique private mesh variable ids

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        u_Field: uw.discretisation.MeshVariable = None,
        V_Field: uw.discretisation.MeshVariable = None,
        theta: float = 0.5,
        solver_name: str = "",
        restore_points_func: Callable = None,
        verbose=False,
    ):

        SNES_AdvectionDiffusion_SLCN.instances += 1

        if solver_name == "":
            solver_name = "AdvDiff_slcn_{}_".format(self.instances)

        ## Parent class will set up default values etc
        super().__init__(mesh, u_Field, solver_name, verbose)

        # These are unique to the advection solver
        self._V = V_Field

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
        ks = str(self.instances)
        name = r"T^{*^{{[" + ks + "]}}}"
        nT1 = uw.swarm.SwarmVariable(name, nswarm, 1)
        name = r"X0^{*^{{[" + ks + "]}}}"
        nX0 = uw.swarm.SwarmVariable(name, nswarm, nswarm.dim)

        nswarm.dm.finalizeFieldRegister()
        nswarm.dm.addNPoints(
            self._u.coords.shape[0] + 1
        )  # why + 1 ? That's the number of spots actually allocated

        cellid = nswarm.dm.getField("DMSwarm_cellid")
        coords = nswarm.dm.getField("DMSwarmPIC_coor").reshape((-1, nswarm.dim))
        coords[...] = self._u.coords[...]
        cellid[:] = self.mesh.get_closest_cells(coords)

        # Move slightly within the chosen cell to avoid edge effects
        centroid_coords = self.mesh._centroids[cellid]
        shift = 1.0e-4 * self.mesh.get_min_radius()
        coords[...] = (1.0 - shift) * coords[...] + shift * centroid_coords[...]

        nswarm.dm.restoreField("DMSwarmPIC_coor")
        nswarm.dm.restoreField("DMSwarm_cellid")
        nswarm.dm.migrate(remove_sent_points=True)

        self._nswarm = nswarm
        self._u_star = nT1
        self._X0 = nX0

        # if we want u_star to satisfy the bcs then this will need to be
        # a projection-mesh variable but it should be ok given these points
        # are designed to land on the mesh

        self._Lstar = self.mesh.vector.jacobian(self._u_star.sym)

        ### add a mesh variable to project diffusivity values to, may want to modify name and degree (?)
        self.k = uw.discretisation.MeshVariable(
            f"k{self.instances}", self.mesh, num_components=1, degree=1
        )

        return

    def adv_diff_slcn_problem_description(self):

        N = self.mesh.N

        # f0 residual term
        self._f0 = self.F0 - self.f + (self.u.sym - self._u_star.sym) / self.delta_t

        # f1 residual term
        self._f1 = (
            self.F1
            + self.theta * self.constitutive_model.flux(self._L).T
            + (1.0 - self.theta) * self.constitutive_model.flux(self._Lstar).T
        )

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
    def estimate_dt(self):
        """
        Calculates an appropriate advective timestep for the given
        mesh and diffusivity configuration.
        """
        ## update the diffusivity values
        self.k_proj = uw.systems.Projection(self.mesh, self.k)
        self.k_proj.uw_function = self.constitutive_model.Parameters.diffusivity
        self.k_proj.smoothing = 0.0
        self.k_proj.solve(_force_setup=True)

        ### required modules
        import math
        from mpi4py import MPI

        with self.mesh.access(self.k):
            ## get local max diff value
            max_diffusivity = self.k.data[:, 0].max()
            ### get the velocity values
            vel = self._V.data

        ## get global max dif value
        comm = MPI.COMM_WORLD
        diffusivity_glob = comm.allreduce(max_diffusivity, op=MPI.MAX)

        ### get global velocity from velocity field
        max_magvel = np.linalg.norm(vel, axis=1).max()
        max_magvel_glob = comm.allreduce(max_magvel, op=MPI.MAX)

        ## get radius
        min_dx = self.mesh.get_min_radius()

        ## estimate dt of adv and diff components
        dt_diff = (min_dx**2) / diffusivity_glob
        dt_adv = min_dx / max_magvel_glob
        ### adv or diff could be 0 if only using one component
        if dt_adv == 0.0:
            dt_estimate = dt_diff
        elif dt_diff == 0.0:
            dt_estimate = dt_adv
        else:
            dt_estimate = min(dt_diff, dt_adv)

        return dt_estimate

    @timing.routine_timer_decorator
    def solve(
        self,
        zero_init_guess: bool = True,
        timestep: float = None,
        coords: np.ndarray = None,
        _force_setup: bool = False,
    ):
        """
        Generates solution to constructed system.

        Params
        ------
        zero_init_guess:
            If `True`, a zero initial guess will be used for the
            system solution. Otherwise, the current values of `self.u` will be used.
        """

        if timestep is not None and timestep != self.delta_t:
            self.delta_t = timestep  # this will force an initialisation because the functions need to be updated

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

        # replace this with built-in mid-point swarm advection (corrector term off)
        # and note the negative timestep

        if coords is None:  # Mid point method to find launch points (T*)
            nswarm.advection(
                self._V.sym,
                -timestep,
                order=2,
                corrector=False,
                restore_points_to_domain_func=self.restore_points_to_domain_func,
            )

        else:  # launch points (T*) provided by omniscience user
            with nswarm.access(nswarm.particle_coordinates):
                nswarm.data[...] = coords[...]

        # Sample the field at these locations

        with nswarm.access(nT1):
            nT1.data[...] = uw.function.evaluate(t_soln.sym[0], nswarm.data).reshape(
                -1, 1
            )

        # restore coords
        with nswarm.access(nswarm.particle_coordinates):
            nswarm.data[...] = nX0.data[...]

        # Over to you Poisson Solver

        super().solve(zero_init_guess, _force_setup)

        return


#################################################
# Swarm-based advection-diffusion
# solver based on SNES_Poisson and swarm-variable
# projection
#
#################################################


class SNES_AdvectionDiffusion_Swarm(SNES_Poisson):

    """Characteristics-based advection diffusion solver:

    Uses a theta timestepping approach with semi-Lagrange sample backwards in time using
    a mid-point advection scheme (based on our particle swarm implementation)
    """

    instances = 0  # count how many of these there are in order to create unique private mesh variable ids

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        u_Field: uw.discretisation.MeshVariable = None,
        u_Star_fn=None,
        theta: float = 0.5,
        solver_name: str = "",
        restore_points_func: Callable = None,
        projection: bool = True,
        verbose: bool = False,
    ):

        self.instance = SNES_AdvectionDiffusion_Swarm.instances
        SNES_AdvectionDiffusion_Swarm.instances += 1

        if solver_name == "":
            solver_name = "AdvDiff_swarm_{}_".format(self.instances)

        ## Parent class will set up default values etc
        super().__init__(mesh, u_Field, solver_name, verbose)

        self.delta_t = 1.0
        self.theta = theta
        self.projection = projection
        self._u_star_raw_fn = u_Star_fn

        self.restore_points_to_domain_func = restore_points_func
        self._setup_problem_description = self.adv_diff_swarm_problem_description

        self.is_setup = False
        self.u_star_is_valid = False

        ### add a mesh variable to project diffusivity values to, may want to modify name and degree (?)
        self.k = uw.discretisation.MeshVariable(
            f"k{self.instances}", self.mesh, num_components=1, degree=1
        )

        if projection:
            # set up a projection solver

            self._u_star_projected = uw.discretisation.MeshVariable(
                r"u^{{*}}{}".format(self.instances), self.mesh, 1, degree=u_Field.degree
            )
            self._u_star_projector = uw.systems.solvers.SNES_Projection(
                self.mesh, self._u_star_projected
            )

            # If we add smoothing, it should be small relative to actual diffusion (self.k)
            self._u_star_projector.smoothing = 0.0
            self._u_star_projector.uw_function = self._u_star_raw_fn

        # if we want u_star to satisfy the bcs then this will need to be
        # a projection

        self._Lstar = self.mesh.vector.jacobian(self.u_star_fn)
        # sympy.derive_by_array(self.u_star_fn, self._X).reshape(self.mesh.dim)

        return

    def adv_diff_swarm_problem_description(self):

        N = self.mesh.N

        # f0 residual term
        self._f0 = self.F0 - self.f + (self.u.sym - self.u_star_fn) / self.delta_t

        # f1 residual term
        self._f1 = (
            self.F1
            + self.theta * self.constitutive_model.flux(self._L).T
            + (1.0 - self.theta) * self.constitutive_model.flux(self._Lstar).T
        )

        return

    @property
    def u(self):
        return self._u

    @property
    def u_star_fn(self):
        if self.projection:
            return self._u_star_projected.sym
        else:
            return self._u_star_raw_fn

    @u_star_fn.setter
    def u_star_fn(self, u_star_fn):
        self.is_setup = False
        if self.projection:
            self._u_star_projector.is_setup = False
            self._u_star_projector.uw_function = u_star_fn

        self._u_star_raw_fn = u_star_fn

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
    def estimate_dt(self):
        """
        Calculates an appropriate advective timestep for the given
        mesh and diffusivity configuration.
        """
        ## update the diffusivity values
        self.k_proj = uw.systems.Projection(self.mesh, self.k)
        self.k_proj.uw_function = self.constitutive_model.Parameters.diffusivity
        self.k_proj.smoothing = 0.0
        self.k_proj.solve(_force_setup=True)

        ### required modules
        import math
        from mpi4py import MPI

        with self.mesh.access(self.k):
            ## get local max dif value
            max_diffusivity = self.k.data[:, 0].max()
            ### get the velocity values
            vel = self._V.data

        ## get global max dif value
        comm = MPI.COMM_WORLD
        diffusivity_glob = comm.allreduce(max_diffusivity, op=MPI.MAX)

        ### get global velocity from velocity field
        max_magvel = np.linalg.norm(vel, axis=1).max()
        max_magvel_glob = comm.allreduce(max_magvel, op=MPI.MAX)

        ## get radius
        min_dx = self.mesh.get_min_radius()

        ## estimate dt of adv and diff components
        dt_diff = (min_dx**2) / diffusivity_glob
        dt_adv = min_dx / max_magvel_glob
        ### adv or diff could be 0 if only using one component
        if dt_adv == 0.0:
            dt_estimate = dt_diff
        elif dt_diff == 0.0:
            dt_estimate = dt_adv
        else:
            dt_estimate = min(dt_diff, dt_adv)

        return dt_estimate

    @timing.routine_timer_decorator
    def solve(
        self,
        zero_init_guess: bool = True,
        timestep: float = None,
        _force_setup: bool = False,
    ):
        """
        Generates solution to constructed system.

        Params
        ------
        zero_init_guess:
            If `True`, a zero initial guess will be used for the
            system solution. Otherwise, the current values of `self.u` will be used.
        """

        if timestep is not None and timestep != self.delta_t:
            self.delta_t = timestep  # this will force an initialisation because the functions need to be updated

        if (not self.is_setup) or _force_setup:
            self._setup_terms()

        # Make sure we update the projection of the swarm variable if requested

        if self.projection:
            self._u_star_projector.solve(zero_init_guess)

        # Over to you Poisson Solver

        super().solve(zero_init_guess, _force_setup)

        return

    @timing.routine_timer_decorator
    def _setup_terms(self):

        if self.projection:
            self._u_star_projector.bcs = self.bcs
            self._u_star_projector._setup_terms()

        super()._setup_terms()


#################################################
# Swarm-based advection-diffusion
# solver based on SNES_Poisson and swarm-variable
# projection
#
#################################################


class SNES_NavierStokes_Swarm(SNES_Stokes):

    """Swarm-based Navier Stokes:

    Uses a theta timestepping approach with semi-Lagrange sample backwards in time using
    a mid-point advection scheme (based on our particle swarm implementation)

    Figure: ![](../../Jupyterbook/Figures/Diagrams/NS_Benchmark_DFG_2.png)

    """

    instances = 0  # count how many of these there are in order to create unique private mesh variable ids

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        velocityField: uw.discretisation.MeshVariable = None,
        pressureField: uw.discretisation.MeshVariable = None,
        velocityStar_fn=None,  # uw.function.UnderworldFunction = None,
        rho: Optional[float] = 0.0,
        theta: Optional[float] = 0.5,
        solver_name: Optional[str] = "",
        verbose: Optional[bool] = False,
        projection: Optional[bool] = False,
        restore_points_func: Callable = None,
    ):

        SNES_NavierStokes_Swarm.instances += 1

        if solver_name == "":
            solver_name = "NStokes_swarm_{}_".format(self.instances)

        self.delta_t = 1.0
        self.theta = theta
        self.projection = projection
        self.rho = rho
        self._u_star_raw_fn = velocityStar_fn

        # self._saddle_preconditioner = 1.0 / (
        #         self.viscosity + self.rho / self.delta_t
        #     )

        ## Parent class will set up default values etc
        super().__init__(
            mesh,
            velocityField,
            pressureField,
            solver_name,
            verbose,
        )

        if projection:
            # set up a projection solver
            self._u_star_projected = uw.discretisation.MeshVariable(
                rf"u^{{*[{self.instances}]}}",
                self.mesh,
                self.mesh.dim,
                degree=self._u.degree - 1,
            )
            self._u_star_projector = uw.systems.solvers.SNES_Vector_Projection(
                self.mesh, self._u_star_projected
            )

            options = self._u_star_projector.petsc_options
            options.delValue("ksp_monitor")

            # If we add smoothing, it should be small relative to actual diffusion (self.viscosity)
            self._u_star_projector.smoothing = 0.0
            self._u_star_projector.uw_function = self._u_star_raw_fn

            self._u_star_projector.add_dirichlet_bc(
                (self._u.sym[0],), "All_Boundaries", (0,)
            )
            self._u_star_projector.add_dirichlet_bc(
                (self._u.sym[1],), "All_Boundaries", (1,)
            )
            if self.mesh.dim == 3:
                self._u_star_projector.add_dirichlet_bc(
                    (self._u.sym[2],), "All_Boundaries", (2,)
                )

        self.restore_points_to_domain_func = restore_points_func
        self._setup_problem_description = self.navier_stokes_swarm_problem_description

        self.is_setup = False
        self.first_solve = True

        # self._Ustar = sympy.Array(
        #     (self.u_star_fn.to_matrix(self.mesh.N))[0 : self.mesh.dim]
        # )

        # This part is used to compute Jacobians (so probably not used)
        # self._Lstar = self.mesh.vector.jacobian(self._u_star.sym)

        # This is used
        self._Estar = self.mesh.vector.strain_tensor(self.u_star_fn)

        # self._Stress_star = (
        #     self.constitutive_model.flux(self.strainrate_star)
        #     - sympy.eye(self.mesh.dim) * self.p.fn
        # )

        return

    def navier_stokes_swarm_problem_description(self):

        N = self.mesh.N
        dim = self.mesh.dim

        # terms that become part of the weighted integral
        self._u_f0 = (
            self.UF0
            - 1.0 * self.bodyforce
            + self.rho * (self.u.sym - self.u_star_fn) / self.delta_t
        )

        # Integration by parts into the stiffness matrix
        self._u_f1 = (
            self.UF1
            + self.stress * self.theta
            + self.stress_star * (1.0 - self.theta)
            + self.penalty * self.div_u * sympy.eye(dim)
        )

        # forces in the constraint (pressure) equations
        self._p_f0 = self.PF0 + sympy.Matrix([self.div_u])

        return

    @property
    def u(self):
        return self._u

    @property
    def u_star_fn(self):
        if self.projection:
            return self._u_star_projected.sym
        else:
            return self._u_star_raw_fn

    @u_star_fn.setter
    def u_star_fn(self, uw_function):
        self.is_setup = False
        if self.projection:
            self._u_star_projector.is_setup = False
            self._u_star_projector.uw_function = uw_function

        self._u_star_raw_fn = uw_function

    @property
    def stress_star_deviator(self):
        return self.constitutive_model.flux(self.strainrate_star)

    @property
    def strainrate_star(self):
        return sympy.Matrix(self._Estar)

    @property
    def stress_star(self):
        return self.stress_deviator - sympy.eye(self.mesh.dim) * (self.p.sym[0])

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
    def solve(
        self,
        zero_init_guess: bool = True,
        timestep: float = None,
        _force_u_star_projection: bool = False,
        _force_setup: bool = False,
    ):
        """
        Generates solution to constructed system.

        Params
        ------
        zero_init_guess:
            If `True`, a zero initial guess will be used for the
            system solution. Otherwise, the current values of `self.u` will be used.
        timestep:
            value used to evaluate inertial contribution
        """

        if timestep is not None and timestep != self.delta_t:
            self.delta_t = timestep  # this will force an initialisation because the functions need to be updated

        if (not self.is_setup) or _force_setup:
            self._setup_terms()

        # Make sure we update the projection of the swarm variable if requested
        # But, this can break down on the first solve if there are constraints and bcs
        # (we might want to use v_star for checkpointing though)

        if self.projection and (not self.first_solve or _force_u_star_projection):
            # self._u_star_projector.petsc_options[
            #     "snes_type"
            # ] = "newtontr"  ## newtonls seems to be problematic when the previous guess is available

            self._u_star_projector.solve(zero_init_guess=False)

        # Over to you Stokes Solver
        super().solve(zero_init_guess)
        self.first_solve = False

        return

    @timing.routine_timer_decorator
    def _setup_terms(self):

        if self.projection:
            # self._u_star_projector.bcs = self.bcs
            # self._u_star_projector.
            self._u_star_projector._setup_terms()

        super()._setup_terms()
