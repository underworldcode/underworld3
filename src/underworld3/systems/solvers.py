import sympy
from sympy import sympify
import numpy as np

from typing import Optional, Callable, Union

import underworld3 as uw
from underworld3.systems import SNES_Scalar, SNES_Vector, SNES_Stokes_SaddlePt
from underworld3 import VarType
import underworld3.timing as timing
from underworld3.utilities._api_tools import uw_object


from .ddt import SemiLagrangian as SemiLagrangian_DDt
from .ddt import Lagrangian as Lagrangian_DDt


# class UW_Scalar_Temple(SNES_Scalar):

# class UW_Lagrangian_Helper:
#     """Mixin style ... add some functions to manage swarm updates etc"""

#     @property
#     def phi_star(self):
#         return "phi_star"

#     @property
#     def phi_star_star(self):
#         return "phi_star_star"


class SNES_Poisson(SNES_Scalar):
    r"""
    This class provides functionality for a discrete representation
    of the Poisson equation

    $$
    \nabla \cdot
            \color{Blue}{\underbrace{\Bigl[ \boldsymbol\kappa \nabla u \Bigr]}_{\mathbf{F}}} =
            \color{Maroon}{\underbrace{\Bigl[ f \Bigl] }_{\mathbf{f}}}
    $$

    The term $\mathbf{F}$ relates the flux to gradients in the unknown $u$

    ## Properties

      - The unknown is $u$

      - The diffusivity tensor, $\kappa$ is provided by setting the `constitutive_model` property to
    one of the scalar `uw.constitutive_models` classes and populating the parameters.
    It is usually a constant or a function of position / time and may also be non-linear
    or anisotropic.

      - $f$ is a volumetric source term
    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        u_Field: uw.discretisation.MeshVariable = None,
        solver_name: str = "",
        verbose=False,
        degree=2,
    ):
        ## Keep track

        ## Parent class will set up default values etc
        super().__init__(
            mesh,
            u_Field,
            None,
            None,
            degree,
            solver_name,
            verbose,
        )

        if solver_name == "":
            solver_name = "Poisson_{}_".format(self.instance_number)

        # Register the problem setup function
        self._setup_problem_description = self.poisson_problem_description

        # default values for properties
        self.f = sympy.Matrix.zeros(1, 1)

        self._constitutive_model = None

    @timing.routine_timer_decorator
    def poisson_problem_description(self):
        # f1 residual term (weighted integration) - scalar function
        self._f0 = self.F0 - self.f

        # f1 residual term (integration by parts / gradients)
        # isotropic
        self._f1 = self.F1 + self.constitutive_model.flux.T

        return

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        self.is_setup = False
        self._f = sympy.Matrix((value,))

    @property
    def CM_is_setup(self):
        return self._constitutive_model._solver_is_setup


class SNES_Darcy(SNES_Scalar):
    r"""
    This class provides functionality for a discrete representation
    of the Groundwater flow equations

    $$
    \color{Green}{\underbrace{ \Bigl[  S_s \frac{\partial h}{\partial t} \Bigr]}_{\dot{\mathbf{f}}}} -
    \nabla \cdot
            \color{Blue}{\underbrace{\Bigl[ \boldsymbol\kappa \nabla h  - \boldsymbol{s}\Bigr]}_{\mathbf{F}}} =
            \color{Maroon}{\underbrace{\Bigl[ W \Bigl] }_{\mathbf{f}}}
    $$

    The flux term, $\mathbf{F}$ relates the effective velocity to pressure gradients

    $$
    \boldsymbol{v} = \left( \boldsymbol\kappa \nabla h  - \boldsymbol{s} \right)
    $$

    The time-dependent term $\dot{\mathbf{f}}$ is not implemented in this version.

    ## Properties

      - The unknown is $h$, the hydraulic head

      - The permeability tensor, $\kappa$ is provided by setting the `constitutive_model` property to
    one of the scalar `uw.constitutive_models` classes and populating the parameters.
    It is usually a constant or a function of position / time and may also be non-linear
    or anisotropic.

      - Volumetric sources for the pressure gradient are supplied through
        the $s$ property [e.g. $s = \rho g$ ]

      - $W$ is a pressure source term

      - $S_s$ is the specific storage coefficient

    ## Notes

      - The solver returns the primary field and also the Darcy flux term (the mean-flow velocity)

    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        h_Field: Optional[uw.discretisation.MeshVariable] = None,
        v_Field: Optional[uw.discretisation.MeshVariable] = None,
        degree: int = 2,
        solver_name: str = "",
        verbose=False,
    ):
        ## Parent class will set up default values etc
        super().__init__(
            mesh,
            h_Field,
            None,  # DuDt
            None,  # DFDt
            degree,
            solver_name,
            verbose,
        )

        if solver_name == "":
            self.solver_name = "Darcy_{}_".format(self.instance_number)

        # Register the problem setup function
        self._setup_problem_description = self.darcy_problem_description

        # default values for properties
        self._f = sympy.Matrix([[0]])
        self._k = 1

        self._s = sympy.Matrix.zeros(rows=1, cols=self.mesh.dim)
        self._s[1] = -1

        self._constitutive_model = None

        ## Set up the projection operator that
        ## solves the flow rate

        self._v_projector = uw.systems.solvers.SNES_Vector_Projection(
            self.mesh,
            v_Field,
            degree=self.Unknowns.u.degree - 1,
        )

        self._v = self._v_projector.Unknowns.u

        # If we add smoothing, it should be small relative to actual diffusion (self.viscosity)
        self._v_projector.smoothing = 0

    ## This function is the one we will typically over-ride to build specific solvers.
    ## This example is a poisson-like problem with isotropic coefficients

    @timing.routine_timer_decorator
    def darcy_problem_description(self):
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

    #  ### W term not set up ???
    #  @property
    #  def W(self):
    #      return self._W

    #  @W.setter
    #  def W(self, value):
    #      self.is_setup = False
    #      self._W = sympy.Matrix((value,))

    @property
    def s(self):
        return self._s

    @s.setter
    def s(self, value):
        self.is_setup = False
        self._s = sympy.Matrix((value,))

    @property
    def darcy_flux(self):
        flux = self.constitutive_model.flux.T
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
        verbose: bool = False,
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
            self._setup_pointwise_functions(verbose)
            self._setup_discretisation(verbose)
            self._setup_solver(verbose)

        # Solve pressure

        super().solve(zero_init_guess, _force_setup)

        # Now solve flow field

        self._v_projector.petsc_options["snes_rtol"] = 1.0e-6
        self._v_projector.petsc_options.delValue("ksp_monitor")
        self._v_projector.solve(zero_init_guess)

        return

    # @timing.routine_timer_decorator
    # def _setup_terms(self):
    #     self._v_projector.uw_function = self.darcy_flux
    #     self._v_projector._setup_terms()  # _setup_pointwise_functions()  #
    #     super()._setup_terms()


## --------------------------------
## Stokes saddle point solver plus
## ancilliary functions - note that
## we need to update the description
## of the generic saddle pt solver
## to remove the Stokes-specific stuff
## --------------------------------


class SNES_Stokes(SNES_Stokes_SaddlePt):
    r"""
    This class provides functionality for a discrete representation
    of the Stokes flow equations assuming an incompressibility
    (or near-incompressibility) constraint.

    $$
    \nabla \cdot
            \color{Blue}{\underbrace{\Bigl[
                    \boldsymbol{\tau} -  p \mathbf{I} \Bigr]}_{\mathbf{F}}} =
            \color{Maroon}{\underbrace{\Bigl[ \mathbf{f} \Bigl] }_{\mathbf{f}}}
    $$

    $$
    \underbrace{\Bigl[ \nabla \cdot \mathbf{u} \Bigr]}_{\mathbf{f}_p} = 0
    $$

    The flux term is a deviatoric stress ( $\boldsymbol{\tau}$ ) related to velocity gradients
      ( $\nabla \mathbf{u}$ ) through a viscosity tensor, $\eta$, and a volumetric (pressure) part $p$

    $$
        \mathbf{F}: \quad \boldsymbol{\tau} = \frac{\eta}{2}\left( \nabla \mathbf{u} + \nabla \mathbf{u}^T \right)
    $$

    The constraint equation, $\mathbf{f}_p = 0$ is incompressible flow by default but can be set
    to any function of the unknown  $\mathbf{u}$ and  $\nabla\cdot\mathbf{u}$

    ## Properties

      - The unknowns are velocities $\mathbf{u}$ and a pressure-like constraint paramter $\mathbf{p}$

      - The viscosity tensor, $\boldsymbol{\eta}$ is provided by setting the `constitutive_model` property to
    one of the scalar `uw.constitutive_models` classes and populating the parameters.
    It is usually a constant or a function of position / time and may also be non-linear
    or anisotropic.

      - $\mathbf f$ is a volumetric source term (i.e. body forces)
      and is set by providing the `bodyforce` property.

      - An Augmented Lagrangian approach to application of the incompressibility
    constraint is to penalise incompressibility in the Stokes equation by adding
    $ \lambda \nabla \cdot \mathbf{u} $ when the weak form of the equations is constructed.
    (this is in addition to the constraint equation, unlike in the classical penalty method).
    This is activated by setting the `penalty` property to a non-zero floating point value which adds
    the term in the `sympy` expression.

      - A preconditioner is usually required for the saddle point system and this is provided
    though the `saddle_preconditioner` property. The default choice is $1/\eta$ for a scalar viscosity function.

    ## Notes

      - For problems with viscoelastic behaviour, the flux term contains the stress history as well as the
        stress and this term is a Lagrangian quantity that has to be tracked on a particle swarm.

      - The interpolation order of the `pressureField` variable is used to determine the integration order of
    the mixed finite element method and is usually lower than the order of the `velocityField` variable.

      - It is possible to set discontinuous pressure variables by setting the `p_continous` option to `False`

    """

    instances = 0

    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        velocityField: Optional[uw.discretisation.MeshVariable] = None,
        pressureField: Optional[uw.discretisation.MeshVariable] = None,
        degree: Optional[int] = 2,
        p_continuous: Optional[bool] = True,
        solver_name: Optional[str] = "",
        verbose: Optional[bool] = False,
    ):
        super().__init__(
            mesh,
            velocityField,
            pressureField,
            None,
            None,
            degree,
            p_continuous,
            solver_name,
            verbose,
        )

        # change this to be less generic
        if solver_name == "":
            self.name = "Stokes_{}_".format(self.instance_number)

        self._degree = degree
        # User-facing operations are matrices / vectors by preference

        # self.Unknowns.E = self.mesh.vector.strain_tensor(self.Unknowns.u.sym)
        self._Estar = None

        # scalar 2nd invariant (incompressible)
        self._penalty = 0.0
        self._constraints = sympy.Matrix(
            (self.div_u,)
        )  # by default, incompressibility constraint

        self._bodyforce = sympy.Matrix([[0] * self.mesh.dim])

        self._setup_problem_description = self.stokes_problem_description

        # this attrib records if we need to re-setup
        self.is_setup = False

        self._constitutive_model = None

        return

    @timing.routine_timer_decorator
    def stokes_problem_description(self):
        dim = self.mesh.dim

        # residual terms can be redefined here. We leave the
        # F0, F1, PF0 terms in place to allow injection of
        # additional terms. These are pre-defined to be zero

        # terms that become part of the weighted integral
        self._u_f0 = self.F0 - self.bodyforce
        # Integration by parts into the stiffness matrix (constitutive terms)
        self._u_f1 = sympy.simplify(
            self.F1 + self.stress + self.penalty * self.div_u * sympy.eye(dim)
        )

        # forces in the constraint (pressure) equations
        self._p_f0 = sympy.simplify(self.PF0 + sympy.Matrix((self.constraints)))

        return

    @property
    def DFDt(self):
        return self._DFDt

    @DFDt.setter
    def DFDt(self, DFDt: Union[SemiLagrangian_DDt, Lagrangian_DDt]):
        self._DFDt = DFDt

    ### add property for DuDt to be updated by the user if they wish
    @property
    def DuDt(self):
        return self._DuDt

    @DuDt.setter
    def DuDt(self, DuDt: Union[SemiLagrangian_DDt, Lagrangian_DDt]):
        self._DuDt = DuDt

    # @property
    # def constitutive_model(self):
    #     return self._constitutive_model

    ## Not keen on this - we ought to do this sort of thing
    ## on the generic classes - otherwise generic models do not
    ## have this property

    # @constitutive_model.setter
    # def constitutive_model(self, model):
    #     ### only setup history terms in it's a VEP model
    #     ### Need a better flag for this
    #     if model == uw.constitutive_models.ViscoElasticPlasticFlowModel:
    #         self._setup_history_terms

    #     ### checking if it's an instance
    #     if isinstance(model, uw.constitutive_models.Constitutive_Model):
    #         self._constitutive_model = model
    #         ### update history terms using setters
    #         self._constitutive_model.flux_dt = self.DFDt
    #         self._constitutive_model.DuDt = self.DuDt
    #     ### checking if it's a class
    #     elif type(model) == type(uw.constitutive_models.Constitutive_Model):
    #         self._constitutive_model = model(self.Unknowns)
    #         ### update history terms using setters
    #         self._constitutive_model.flux_dt = self.DFDt
    #         self._constitutive_model.DuDt = self.DuDt
    #     ### Raise an error if it's neither
    #     else:
    #         raise RuntimeError(
    #             "constitutive_model must be a valid class or instance of a valid class"
    #         )

    @property
    def CM_is_setup(self):
        return self._constitutive_model._solver_is_setup

    @property
    def strainrate(self):
        return sympy.Matrix(self.mesh.vector.strain_tensor(self.Unknowns.u.sym))

    @property
    def strainrate_1d(self):
        return uw.maths.tensor.rank2_to_voigt(self.strainrate, self.mesh.dim)

    @property
    def strainrate_star_1d(self):
        return uw.maths.tensor.rank2_to_voigt(self.strainrate_star, self.mesh.dim)

    # This should return standard viscous behaviour if strainrate_star is None
    @property
    def stress_deviator(self):
        return self.constitutive_model.flux  # strainrate, strain-rate history

    @property
    def stress_deviator_1d(self):
        return uw.maths.tensor.rank2_to_voigt(self.stress_deviator, self.mesh.dim)  ##

    @property
    def stress(self):
        return self.stress_deviator - sympy.eye(self.mesh.dim) * (self.p.sym[0])

    @property
    def stress_1d(self):
        return uw.maths.tensor.rank2_to_voigt(self.stress, self.mesh.dim)

    @property
    def div_u(self):
        # E = self.strainrate
        # divergence = E.trace()
        # return divergence

        return self.mesh.vector.divergence(self.Unknowns.u.sym)

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


class SNES_VE_Stokes(SNES_Stokes):
    r"""
    This class provides functionality for a discrete representation
    of the Stokes flow equations assuming an incompressibility
    (or near-incompressibility) constraint and with a flux history
    term included to allow for viscoelastic modelling.

    $$
    \nabla \cdot
            \color{Blue}{\underbrace{\Bigl[
                    \boldsymbol{\tau} -  p \mathbf{I} \Bigr]}_{\mathbf{F}}} =
            \color{Maroon}{\underbrace{\Bigl[ \mathbf{f} \Bigl] }_{\mathbf{f}}}
    $$

    $$
    \underbrace{\Bigl[ \nabla \cdot \mathbf{u} \Bigr]}_{\mathbf{f}_p} = 0
    $$

    The flux term is a deviatoric stress ( $\boldsymbol{\tau}$ ) related to velocity gradients
      ( $\nabla \mathbf{u}$ ) through a viscosity tensor, $\eta$, and a volumetric (pressure) part $p$

    $$
        \mathbf{F}: \quad \boldsymbol{\tau} = \frac{\eta}{2}\left( \nabla \mathbf{u} + \nabla \mathbf{u}^T \right)
    $$

    The constraint equation, $\mathbf{f}_p = 0$ is incompressible flow by default but can be set
    to any function of the unknown  $\mathbf{u}$ and  $\nabla\cdot\mathbf{u}$

    ## Properties

      - The unknowns are velocities $\mathbf{u}$ and a pressure-like constraint paramter $\mathbf{p}$

      - The viscosity tensor, $\boldsymbol{\eta}$ is provided by setting the `constitutive_model` property to
    one of the scalar `uw.constitutive_models` classes and populating the parameters.
    It is usually a constant or a function of position / time and may also be non-linear
    or anisotropic.

      - $\mathbf f$ is a volumetric source term (i.e. body forces)
      and is set by providing the `bodyforce` property.

      - An Augmented Lagrangian approach to application of the incompressibility
    constraint is to penalise incompressibility in the Stokes equation by adding
    $ \lambda \nabla \cdot \mathbf{u} $ when the weak form of the equations is constructed.
    (this is in addition to the constraint equation, unlike in the classical penalty method).
    This is activated by setting the `penalty` property to a non-zero floating point value which adds
    the term in the `sympy` expression.

      - A preconditioner is usually required for the saddle point system and this is provided
    though the `saddle_preconditioner` property. The default choice is $1/\eta$ for a scalar viscosity function.

    ## Notes

      - For problems with viscoelastic behaviour, the flux term contains the stress history as well as the
        stress and this term is a Lagrangian quantity that has to be tracked on a particle swarm.

      - The interpolation order of the `pressureField` variable is used to determine the integration order of
    the mixed finite element method and is usually lower than the order of the `velocityField` variable.

      - It is possible to set discontinuous pressure variables by setting the `p_continous` option to `False`

    """

    instances = 0

    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        velocityField: Optional[uw.discretisation.MeshVariable] = None,
        pressureField: Optional[uw.discretisation.MeshVariable] = None,
        DFDt: Union[SemiLagrangian_DDt, Lagrangian_DDt] = None,
        degree: Optional[int] = 2,
        order: Optional[int] = 2,
        p_continuous: Optional[bool] = True,
        solver_name: Optional[str] = "",
        verbose: Optional[bool] = False,
    ):
        super().__init__(
            mesh,
            velocityField,
            pressureField,
            None,
            DFDt,
            degree,
            p_continuous,
            solver_name,
            verbose,
        )

        # change this to be less generic
        if solver_name == "":
            self.name = "VE_Stokes_{}_".format(self.instance_number)

        self._order = order
        # User-facing operations are matrices / vectors by preference

        self._Estar = None

        # scalar 2nd invariant (incompressible)
        self._penalty = 0.0
        self._constraints = sympy.Matrix(
            (self.div_u,)
        )  # by default, incompressibility constraint

        self._bodyforce = sympy.Matrix([[0] * self.mesh.dim])

        self._setup_problem_description = self.stokes_problem_description

        # this attrib records if we need to re-setup
        self.is_setup = False
        self._constitutive_model = None

        if self.Unknowns.DFDt is None:
            self.Unknowns.DFDt = uw.systems.ddt.SemiLagrangian(
                self.mesh,
                sympy.Matrix.zeros(self.mesh.dim, self.mesh.dim),
                self.u.sym,
                vtype=uw.VarType.SYM_TENSOR,
                degree=self.u.degree - 1,
                continuous=True,
                varsymbol=rf"{{F[ {self.u.symbol} ] }}",
                verbose=self.verbose,
                bcs=None,
                order=self._order,
                smoothing=0.0,
            )

        return


class SNES_Projection(SNES_Scalar):
    r"""
    Solves $u = \tilde{f}$ where $\tilde{f}$ is a function that can be evaluated within an element and
    $u$ is a `meshVariable` with associated shape functions. Typically, the projection is used to obtain a
    continuous representation of a function that is not well defined at the mesh nodes. For example, functions of
    the spatial derivatives of one or more `meshVariable` (e.g. components of fluxes) can be mapped to continuous
    variables with a projection. More broadly it is a projection from one basis to another and its limitations should be
    evaluated within that context.

    The projection implemented by creating a solver for this problem

    $$
    \nabla \cdot
            \color{Blue}{\underbrace{\Bigl[ \boldsymbol\alpha \nabla u \Bigr]}_{\mathbf{F}}} -
            \color{Maroon}{\underbrace{\Bigl[ u - \tilde{f} \Bigl] }_{\mathbf{f}}} = 0
    $$

    Where the term $\mathbf{F}$ provides a smoothing regularization. $\alpha$ can be zero.
    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        u_Field: uw.discretisation.MeshVariable = None,
        degree=2,
        solver_name: str = "",
        verbose=False,
    ):
        super().__init__(
            mesh,
            u_Field,
            None,  # DuDt
            None,  # DFDt
            degree,
            solver_name,
            verbose,
        )

        if solver_name == "":
            self.name = "SProj_{}_".format(self.instance_number)

        self._setup_problem_description = self.projection_problem_description
        self.is_setup = False
        self._smoothing = 0.0
        self._uw_weighting_function = 1.0
        self._constitutive_model = uw.constitutive_models.Constitutive_Model(
            self.Unknowns
        )

        return

    @timing.routine_timer_decorator
    def projection_problem_description(self):
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
    r"""
    Solves $\mathbf{u} = \tilde{\mathbf{f}}$ where $\tilde{\mathbf{f}}$ is a vector function that can be evaluated within an element and
    $\mathbf{u}$ is a vector `meshVariable` with associated shape functions. Typically, the projection is used to obtain a
    continuous representation of a function that is not well defined at the mesh nodes. For example, functions of
    the spatial derivatives of one or more `meshVariable` (e.g. components of fluxes) can be mapped to continuous
    variables with a projection. More broadly it is a projection from one basis to another and its limitations should be
    evaluated within that context.

    The projection is implemented by creating a solver for this problem

    $$
    \nabla \cdot
            \color{Blue}{\underbrace{\Bigl[ \boldsymbol\alpha \nabla \mathbf{u} \Bigr]}_{\mathbf{F}}} -
            \color{Maroon}{\underbrace{\Bigl[ \mathbf{u} - \tilde{\mathbf{f}} \Bigl] }_{\mathbf{f}}} = 0
    $$

    Where the term $\mathbf{F}$ provides a smoothing regularization. $\alpha$ can be zero.
    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        u_Field: uw.discretisation.MeshVariable = None,
        degree=2,
        solver_name: str = "",
        verbose=False,
    ):
        super().__init__(
            mesh,
            u_Field,
            None,  # DuDt
            None,  # DFDt
            degree,
            solver_name,
            verbose,
        )

        if solver_name == "":
            solver_name = "VProj{}_".format(self.instance_number)

        self._setup_problem_description = self.projection_problem_description
        self.is_setup = False
        self._smoothing = 0.0
        self._penalty = 0.0
        self._uw_weighting_function = 1.0
        self._constitutive_model = uw.constitutive_models.Constitutive_Model(
            self.Unknowns
        )

        return

    @timing.routine_timer_decorator
    def projection_problem_description(self):
        # residual terms - defines the problem:
        # solve for a best fit to the continuous mesh
        # variable given the values in self.function
        # F0 is left in place for the user to inject
        # non-linear constraints if required

        self._f0 = (
            self.F0 + (self.u.sym - self.uw_function) * self.uw_weighting_function
        )

        # F1 is left in the users control ... e.g to add other gradient constraints to the stiffness matrix

        self._f1 = (
            self.F1
            + self.smoothing * self.Unknowns.E
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
        self._uw_function = sympy.Matrix(user_uw_function)

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


class SNES_Tensor_Projection(SNES_Projection):
    r"""
    Solves $\mathbf{u} = \tilde{\mathbf{f}}$ where $\tilde{\mathbf{f}}$ is a tensor-valued function that can be evaluated within an element and
    $\mathbf{u}$ is a tensor `meshVariable` with associated shape functions. Typically, the projection is used to obtain a
    continuous representation of a function that is not well defined at the mesh nodes. For example, functions of
    the spatial derivatives of one or more `meshVariable` (e.g. components of fluxes) can be mapped to continuous
    variables with a projection. More broadly it is a projection from one basis to another and its limitations should be
    evaluated within that context.

    The projection implemented by creating a solver for this problem

    $$
    \nabla \cdot
            \color{Blue}{\underbrace{\Bigl[ \boldsymbol\alpha \nabla \mathbf{u} \Bigr]}_{\mathbf{F}}} -
            \color{Maroon}{\underbrace{\Bigl[ \mathbf{u} - \tilde{\mathbf{f}} \Bigl] }_{\mathbf{f}}} = 0
    $$

    Where the term $\mathbf{F}$ provides a smoothing regularization. $\alpha$ can be zero.

    Note: this is currently implemented component-wise as we do not have a native solver for tensor unknowns.

    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        tensor_Field: uw.discretisation.MeshVariable = None,
        scalar_Field: uw.discretisation.MeshVariable = None,
        degree=2,
        solver_name: str = "",
        verbose=False,
    ):
        self.t_field = tensor_Field

        super().__init__(
            mesh=mesh,
            u_Field=scalar_Field,
            # DuDt=None,  # DuDt #### Not in in SNES_projection class
            # DFDt=None,  # DFDt #### Not in in SNES_projection class
            degree=degree,
            solver_name=solver_name,
            verbose=verbose,
        )

        if solver_name == "":
            solver_name = "TProj{}_".format(self.instance_number)

        return

    ## Need to over-ride solve method to run over all components

    def solve(self):
        # Loop over the components of the tensor. If this is a symmetric
        # tensor, we'll usually be given the 1d form to prevent duplication

        # if self.t_field.sym_1d.shape != self.uw_function.shape:
        #     raise ValueError(
        #         "Tensor shapes for uw_function and MeshVariable are not the same"
        #     )

        symm = self.t_field.sym.is_symmetric

        for i in range(self.uw_function.shape[0]):
            for j in range(self.uw_function.shape[1]):
                if symm and j > i:
                    continue

                self.uw_scalar_function = sympy.Matrix([[self.uw_function[i, j]]])

                with self.mesh.access(self.u):
                    self.u.data[:, 0] = self.t_field[i, j].data[:]

                # solver for the scalar problem

                super().solve()

                with self.mesh.access(self.t_field):
                    self.t_field[i, j].data[:] = self.u.data[:, 0]

        # That might be all ...

    # This is re-defined so it uses uw_scalar_function

    @timing.routine_timer_decorator
    def projection_problem_description(self):
        # residual terms - defines the problem:
        # solve for a best fit to the continuous mesh
        # variable given the values in self.function
        # F0 is left in place for the user to inject
        # non-linear constraints if required

        self._f0 = (
            self.F0
            + (self.u.sym - self.uw_scalar_function) * self.uw_weighting_function
        )

        # F1 is left in the users control ... e.g to add other gradient constraints to the stiffness matrix

        self._f1 = self.F1 + self.smoothing * self.mesh.vector.gradient(self.u.sym)

        return

    @property
    def uw_scalar_function(self):
        return self._uw_scalar_function

    @uw_scalar_function.setter
    def uw_scalar_function(self, user_uw_function):
        self.is_setup = False
        self._uw_scalar_function = user_uw_function


## --------------------------------
## Project from pointwise vector
## functions to nodal point unknowns
##
## We can add boundary constraints in the usual way (parent class handles this)
## We can add smoothing (which takes the form of a viscosity term)
## Here we add an incompressibility constraint
## --------------------------------


#################################################
# Characteristics-based advection-diffusion
# solver based on SNES_Poisson and swarm-to-nodes
#
# Note that the solve() method has the swarm
# handler.
#################################################


class SNES_AdvectionDiffusion_SLCN(SNES_Scalar):
    r"""
    This class provides a solver for the scalar Advection-Diffusion equation using the characteristics based Semi-Lagrange Crank-Nicholson method
    which is described in Spiegelman & Katz, (2006).

    $$
    \color{Green}{\underbrace{ \Bigl[ \frac{\partial u}{\partial t} - \left( \mathbf{v} \cdot \nabla \right) u \Bigr]}_{\dot{\mathbf{f}}}} -
    \nabla \cdot
            \color{Blue}{\underbrace{\Bigl[ \boldsymbol\kappa \nabla u \Bigr]}_{\mathbf{F}}} =
            \color{Maroon}{\underbrace{\Bigl[ f \Bigl] }_{\mathbf{f}}}
    $$

    The term $\mathbf{F}$ relates diffusive fluxes to gradients in the unknown $u$. The advective flux that results from having gradients along
    the direction of transport (given by the velocity vector field $\mathbf{v}$ ) are included in the $\dot{\mathbf{f}}$ term.

    The term $\dot{\mathbf{f}}$ involves upstream sampling to find the value $u^*$ which represents the value of $u$ at
    the points which later arrive at the nodal points of the mesh. This is achieved using a "hidden"
    swarm variable which is advected backwards from the nodal points automatically during the `solve` phase.

    ## Properties

      - The unknown is $u$.

      - The velocity field is $\mathbf{v}$ and is provided as a `sympy` function to allow operations such as time-averaging to be
        calculated in situ (e.g. `V_Field = v_solution.sym`) **NOTE: no it's not. Currently it is a MeshVariable** this is the desired behaviour though.

      - The diffusivity tensor, $\kappa$ is provided by setting the `constitutive_model` property to
        one of the scalar `uw.constitutive_models` classes and populating the parameters.
        It is usually a constant or a function of position / time and may also be non-linear
        or anisotropic.

      - Volumetric sources of $u$ are specified using the $f$ property and can be any valid combination of `sympy` functions of position and
        `meshVariable` or `swarmVariable` types.

      - The `theta` property sets $\theta$, the parameter that tunes between backward Euler $\theta=1$, forward Euler $\theta=0$ and
        Crank-Nicholson $\theta=1/2$. The default is to use the Crank-Nicholson value.

    ## Notes

      - The solver requires relatively high order shape functions to accurately interpolate the history terms.
        Spiegelman & Katz recommend cubic or higher degree for $u$ but this is not checked.

    ## Reference

    Spiegelman, M., & Katz, R. F. (2006). A semi-Lagrangian Crank-Nicolson algorithm for the numerical solution
    of advection-diffusion problems. Geochemistry, Geophysics, Geosystems, 7(4). https://doi.org/10.1029/2005GC001073

    """

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        ## feedback on this instance
        display(Latex(r"$\quad\mathrm{u} = $ " + self.u.sym._repr_latex_()))
        display(Latex(r"$\quad\mathbf{v} = $ " + self._V_fn._repr_latex_()))
        display(Latex(r"$\quad\Delta t = $ " + self.delta_t._repr_latex_()))
        display(Latex(r"$\quad\theta = $ " + self.theta._repr_latex_()))

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        u_Field: uw.discretisation.MeshVariable,
        V_fn: Union[
            uw.discretisation.MeshVariable, sympy.Basic
        ],  # Should be a sympy function
        order: int = 1,
        solver_name: str = "",
        restore_points_func: Callable = None,
        verbose=False,
    ):
        ## Parent class will set up default values etc
        super().__init__(
            mesh,
            u_Field,
            None,
            None,
            u_Field.degree,
            solver_name,
            verbose,
        )

        if solver_name == "":
            solver_name = "AdvDiff_slcn_{}_".format(self.instance_number)

        if isinstance(V_fn, uw.discretisation._MeshVariable):
            self._V_fn = V_fn.sym
        else:
            self._V_fn = V_fn

        # default values for properties
        self.f = sympy.Matrix.zeros(1, 1)

        self._constitutive_model = None

        # These are unique to the advection solver
        self.delta_t = 0.0
        self.is_setup = False

        self.restore_points_to_domain_func = restore_points_func
        self._setup_problem_description = self.adv_diff_slcn_problem_description

        ### Setup the history terms

        self.Unknowns.DuDt = SemiLagrangian_DDt(
            self.mesh,
            u_Field.sym,
            self._V_fn,
            vtype=uw.VarType.SCALAR,
            degree=u_Field.degree,
            continuous=u_Field.continuous,
            varsymbol=u_Field.symbol,
            verbose=verbose,
            bcs=self.essential_bcs,
            order=order,
            smoothing=0.0,
        )

        self.Unknowns.DFDt = SemiLagrangian_DDt(
            self.mesh,
            sympy.Matrix(
                [[0] * self.mesh.dim]
            ),  # Actual function is not defined at this stage
            self._V_fn,
            vtype=uw.VarType.VECTOR,
            degree=u_Field.degree - 1,
            continuous=True,
            varsymbol=rf"{{F[ {self.u.symbol} ] }}",
            verbose=verbose,
            bcs=None,
            order=order,
            smoothing=0.0,
        )

        return

    def adv_diff_slcn_problem_description(self):
        # f0 residual term
        self._f0 = self.F0 - self.f + self.DuDt.bdf() / self.delta_t

        # f1 residual term
        self._f1 = self.F1 + self.DFDt.adams_moulton_flux()

        return

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        self.is_setup = False
        self._f = sympy.Matrix((value,))

    @property
    def V_fn(self):
        return self._V_fn

    @property
    def delta_t(self):
        return self._delta_t

    @delta_t.setter
    def delta_t(self, value):
        self.is_setup = False
        self._delta_t = sympify(value)

    @timing.routine_timer_decorator
    def estimate_dt(self, v_factor=1.0, diffusivity=None):
        """
        Calculates an appropriate advective timestep for the given
        mesh and diffusivity configuration.
        """

        ## If k is non-linear and depends on gradients, then this evaluation
        ## does not work, we instead need a projection.

        if diffusivity is None:
            diffusivity = self.constitutive_model.Parameters.diffusivity

        if isinstance(diffusivity, uw.discretisation._MeshVariable):
            diffusivity = diffusivity.sym[0]

        if isinstance(diffusivity, sympy.Expr):
            k = uw.function.evaluate(
                sympy.sympify(diffusivity),
                self.mesh._centroids,
                self.mesh.N,
            )
            max_diffusivity = k.max()
        else:
            k = self.constitutive_model.Parameters.diffusivity
            max_diffusivity = k

        ### required modules
        from mpi4py import MPI

        ## get global max dif value
        comm = MPI.COMM_WORLD
        diffusivity_glob = comm.allreduce(max_diffusivity, op=MPI.MAX)

        ### get the velocity values
        vel = uw.function.evaluate(
            self.V_fn,
            self.mesh._centroids,
            self.mesh.N,
        )

        ### get global velocity from velocity field
        max_magvel = np.linalg.norm(vel, axis=1).max()
        max_magvel_glob = comm.allreduce(max_magvel, op=MPI.MAX)

        ## get radius
        min_dx = self.mesh.get_min_radius()

        ## estimate dt of adv and diff components

        if max_magvel_glob == 0.0:
            dt_diff = (min_dx**2) / diffusivity_glob
            dt_estimate = dt_diff
        elif diffusivity_glob == 0.0:
            dt_adv = min_dx / max_magvel_glob
            dt_estimate = dt_adv
        else:
            dt_diff = (min_dx**2) / diffusivity_glob
            dt_adv = min_dx / max_magvel_glob
            dt_estimate = min(dt_diff, v_factor * dt_adv)

        return dt_estimate

    @timing.routine_timer_decorator
    def solve(
        self,
        zero_init_guess: bool = True,
        timestep: float = None,
        _force_setup: bool = False,
        verbose=False,
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

        if _force_setup:
            self.is_setup = False

        if not self.constitutive_model._solver_is_setup:
            self.is_setup = False
            self.DFDt.psi_fn = self.constitutive_model.flux.T

        if not self.is_setup:
            self._setup_pointwise_functions(verbose)
            self._setup_discretisation(verbose)
            self._setup_solver(verbose)

        # Update SemiLagrange Unknown and Flux terms
        # self.DuDt.update(timestep, verbose=verbose, evalf=False)
        # self.DFDt.update(timestep, verbose=verbose, evalf=False)

        # super().solve(zero_init_guess, _force_setup)

        self.DuDt.update_pre_solve(timestep, verbose=verbose)
        self.DFDt.update_pre_solve(timestep, verbose=verbose)

        super().solve(zero_init_guess, _force_setup)

        self.DuDt.update_post_solve(timestep, verbose=verbose)
        self.DFDt.update_post_solve(timestep, verbose=verbose)

        self.is_setup = True
        self.constitutive_model._solver_is_setup = True

        return


#################################################
# Swarm-based advection-diffusion
# solver based on SNES_Poisson and swarm-variable
# projection
#
#################################################


class SNES_AdvectionDiffusion(SNES_Scalar):
    r"""
    This class provides a solver for the scalar Advection-Diffusion equation using the characteristics based Semi-Lagrange Crank-Nicholson method
    which is described in Spiegelman & Katz, (2006).

    $$
    \color{Green}{\underbrace{ \Bigl[ \frac{\partial u}{\partial t} - \left( \mathbf{v} \cdot \nabla \right) u \Bigr]}_{\dot{\mathbf{f}}}} -
    \nabla \cdot
            \color{Blue}{\underbrace{\Bigl[ \boldsymbol\kappa \nabla u \Bigr]}_{\mathbf{F}}} =
            \color{Maroon}{\underbrace{\Bigl[ f \Bigl] }_{\mathbf{f}}}
    $$

    The term $\mathbf{F}$ relates diffusive fluxes to gradients in the unknown $u$. The advective flux that results from having gradients along
    the direction of transport (given by the velocity vector field $\mathbf{v}$ ) are included in the $\dot{\mathbf{f}}$ term.

    The term $\dot{\mathbf{f}}$ involves upstream sampling to find the value $u^*$ which represents the value of $u$ at
    the points which later arrive at the nodal points of the mesh. This is achieved using a "hidden"
    swarm variable which is advected backwards from the nodal points automatically during the `solve` phase.

    ## Properties

      - The unknown is $u$.

      - The velocity field is $\mathbf{v}$ and is provided as a `sympy` function to allow operations such as time-averaging to be
        calculated in situ (e.g. `V_Field = v_solution.sym`) **NOTE: no it's not. Currently it is a MeshVariable** this is the desired behaviour though.

      - The diffusivity tensor, $\kappa$ is provided by setting the `constitutive_model` property to
        one of the scalar `uw.constitutive_models` classes and populating the parameters.
        It is usually a constant or a function of position / time and may also be non-linear
        or anisotropic.

      - Volumetric sources of $u$ are specified using the $f$ property and can be any valid combination of `sympy` functions of position and
        `meshVariable` or `swarmVariable` types.



    """

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        ## feedback on this instance
        display(Latex(r"$\quad\mathrm{u} = $ " + self.u.sym._repr_latex_()))
        display(Latex(r"$\quad\mathbf{v} = $ " + self._V_fn._repr_latex_()))
        display(Latex(r"$\quad\Delta t = $ " + self.delta_t._repr_latex_()))
        display(Latex(r"$\quad\theta = $ " + self.theta._repr_latex_()))

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        u_Field: uw.discretisation.MeshVariable,
        V_fn: Union[
            uw.discretisation.MeshVariable, sympy.Basic
        ],  # Should be a sympy function
        DuDt: Lagrangian_DDt = None,
        order: int = 1,
        solver_name: str = "",
        restore_points_func: Callable = None,
        verbose=False,
    ):
        ## Parent class will set up default values etc
        super().__init__(
            mesh,
            u_Field,
            None,
            None,
            u_Field.degree,
            solver_name,
            verbose,
        )

        if solver_name == "":
            solver_name = "AdvDiff_slcn_{}_".format(self.instance_number)

        if isinstance(V_fn, uw.discretisation._MeshVariable):
            self._V_fn = V_fn.sym
        else:
            self._V_fn = V_fn

        # default values for properties
        self.f = sympy.Matrix.zeros(1, 1)

        self._constitutive_model = None

        # These are unique to the advection solver
        self.delta_t = 0.0
        self.is_setup = False

        self.restore_points_to_domain_func = restore_points_func
        self._setup_problem_description = self.adv_diff_slcn_problem_description

        ### Setup the history terms ... This version should not build anything
        ### by default - it's the template / skeleton

        if DuDt is None:
            self.Unknowns.DuDt = SemiLagrangian_DDt(
                self.mesh,
                u_Field.sym,
                self._V_fn,
                vtype=uw.VarType.SCALAR,
                degree=u_Field.degree,
                continuous=u_Field.continuous,
                varsymbol=u_Field.symbol,
                verbose=verbose,
                bcs=self.essential_bcs,
                order=order,
                smoothing=0.0,
            )

        else:
            # validation
            if order is None:
                order = DuDt.order

            else:
                if DuDt.order < order:
                    raise RuntimeError(
                        f"DuDt supplied is order {DuDt.order} but order requested is {order}"
                    )

            self.Unknowns.DuDt = DuDt

        self.Unknowns.DFDt = SemiLagrangian_DDt(
            self.mesh,
            sympy.Matrix(
                [[0] * self.mesh.dim]
            ),  # Actual function is not defined at this point
            self._V_fn,
            vtype=uw.VarType.VECTOR,
            degree=u_Field.degree - 1,
            continuous=False,
            varsymbol=rf"{{F[ {self.u.symbol} ] }}",
            verbose=verbose,
            bcs=None,
            order=order,
            smoothing=0.0,
        )

        return

    def adv_diff_slcn_problem_description(self):
        # f0 residual term
        self._f0 = self.F0 - self.f + self.DuDt.bdf() / self.delta_t

        # f1 residual term
        self._f1 = self.F1 + self.DFDt.adams_moulton_flux()

        return

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        self.is_setup = False
        self._f = sympy.Matrix((value,))

    @property
    def V_fn(self):
        return self._V_fn

    @property
    def delta_t(self):
        return self._delta_t

    @delta_t.setter
    def delta_t(self, value):
        self.is_setup = False
        self._delta_t = sympify(value)

    @timing.routine_timer_decorator
    def estimate_dt(self):
        """
        Calculates an appropriate advective timestep for the given
        mesh and diffusivity configuration.
        """

        if isinstance(self.constitutive_model.Parameters.diffusivity, sympy.Expr):
            k = uw.function.evaluate(
                sympy.sympify(self.constitutive_model.Parameters.diffusivity),
                self.mesh._centroids,
                self.mesh.N,
            )
            max_diffusivity = k.max()
        else:
            k = self.constitutive_model.Parameters.diffusivity
            max_diffusivity = k

        ### required modules
        from mpi4py import MPI

        ## get global max dif value
        comm = MPI.COMM_WORLD
        diffusivity_glob = comm.allreduce(max_diffusivity, op=MPI.MAX)

        ### get the velocity values
        vel = uw.function.evaluate(
            self.V_fn,
            self.mesh._centroids,
            self.mesh.N,
        )

        ### get global velocity from velocity field
        max_magvel = np.linalg.norm(vel, axis=1).max()
        max_magvel_glob = comm.allreduce(max_magvel, op=MPI.MAX)

        ## get radius
        min_dx = self.mesh.get_min_radius()

        ## estimate dt of adv and diff components

        if max_magvel_glob == 0.0:
            dt_diff = (min_dx**2) / diffusivity_glob
            dt_estimate = dt_diff
        elif diffusivity_glob == 0.0:
            dt_adv = min_dx / max_magvel_glob
            dt_estimate = dt_adv
        else:
            dt_diff = (min_dx**2) / diffusivity_glob
            dt_adv = min_dx / max_magvel_glob
            dt_estimate = min(dt_diff, dt_adv)

        return dt_estimate

    @timing.routine_timer_decorator
    def solve(
        self,
        zero_init_guess: bool = True,
        timestep: float = None,
        _force_setup: bool = False,
        verbose=False,
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

        if _force_setup:
            self.is_setup = False

        if not self.constitutive_model._solver_is_setup:
            self.is_setup = False
            self.DFDt.psi_fn = self.constitutive_model.flux.T

        if not self.is_setup:
            self._setup_pointwise_functions(verbose)
            self._setup_discretisation(verbose)
            self._setup_solver(verbose)

        # Update History / Flux History terms
        # SemiLagrange and Lagrange may have different sequencing.
        self.DuDt.update_pre_solve(timestep, verbose=verbose)
        self.DFDt.update_pre_solve(timestep, verbose=verbose)

        super().solve(zero_init_guess, _force_setup)

        self.DuDt.update_post_solve(timestep, verbose=verbose)
        self.DFDt.update_post_solve(timestep, verbose=verbose)

        self.is_setup = True
        self.constitutive_model._solver_is_setup = True

        return


#################################################
# Swarm-based advection-diffusion
# solver based on SNES_Poisson and swarm-variable
# projection
#
#################################################


## This could be the parent class for AdvectionDiffusionSLCN


class SNES_AdvectionDiffusion_Swarm_old(SNES_Scalar):
    r"""
    This class provides a solver for the scalar Advection-Diffusion equation which is similar to that
    used in the Semi-Lagrange Crank-Nicholson method (Spiegelman & Katz, 2006) but using a
    distributed sampling of upstream values taken from an arbitrary swarm variable.

    $$
    \color{Green}{\underbrace{ \Bigl[ \frac{\partial u}{\partial t} - \left( \mathbf{v} \cdot \nabla \right) u \Bigr]}_{\dot{\mathbf{f}}}} -
    \nabla \cdot
            \color{Blue}{\underbrace{\Bigl[ \boldsymbol\kappa \nabla u \Bigr]}_{\mathbf{F}}} =
            \color{Maroon}{\underbrace{\Bigl[ f \Bigl] }_{\mathbf{f}}}
    $$

    The term $\mathbf{F}$ relates diffusive fluxes to gradients in the unknown $u$. The advective flux that results from having gradients along
    the direction of transport (given by the velocity vector field $\mathbf{v}$ ) are included in the $\dot{\mathbf{f}}$ term.

    The term $\dot{\mathbf{f}}$ involves upstream sampling to find the value $u^*$ which represents the value of $u$ at
    the beginning of the timestep. This is achieved using a `swarmVariable` that carries history information along the flow path.
    A dense sampling is required to achieve similar accuracy to the original SLCN approach but it allows the use of a single swarm
    for history tracking of variables with different interpolation order and for material tracking. The user is required to supply
    **and update** the swarmVariable representing $u^*$

    ## Properties

      - The unknown is $u$.

      - The velocity field is $\mathbf{v}$ and is provided as a `sympy` function to allow operations such as time-averaging to be
        calculated in situ (e.g. `V_Field = v_solution.sym`)

      - The history variable is $u^*$ and is provided in the form of a `sympy` function. It is the user's responsibility to keep this
        variable updated.

      - The diffusivity tensor, $\kappa$ is provided by setting the `constitutive_model` property to
        one of the scalar `uw.constitutive_models` classes and populating the parameters.
        It is usually a constant or a function of position / time and may also be non-linear
        or anisotropic.

      - Volumetric sources of $u$ are specified using the $f$ property and can be any valid combination of `sympy` functions of position and
        `meshVariable` or `swarmVariable` types.

      - The `theta` property sets $\theta$, the parameter that tunes between backward Euler $\theta=1$, forward Euler $\theta=0$ and
        Crank-Nicholson $\theta=1/2$. The default is to use the Crank-Nicholson value.

    ## Notes

      - The solver requires relatively high order shape functions to accurately interpolate the history terms.
        Spiegelman & Katz recommend cubic or higher degree for $u$ but this is not checked.

    ## Reference

    Spiegelman, M., & Katz, R. F. (2006). A semi-Lagrangian Crank-Nicolson algorithm for the numerical solution
    of advection-diffusion problems. Geochemistry, Geophysics, Geosystems, 7(4). https://doi.org/10.1029/2005GC001073
    """

    instances = 0  # count how many of these there are in order to create unique private mesh variable ids

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        u_Field: uw.discretisation.MeshVariable,
        V_fn: Union[
            uw.discretisation.MeshVariable, sympy.Basic
        ],  # Should be a sympy function
        DuDt: Lagrangian_DDt,
        solver_name: str = "",
        restore_points_func: Callable = None,
        verbose=False,
    ):
        if solver_name == "":
            solver_name = "AdvDiff_swarm_{}_".format(self.instance_number)

        ## Parent class will set up default values etc
        super().__init__(
            mesh,
            u_Field,
            DuDt,
            DFDt,
            solver_name,
            verbose,
        )

        self.delta_t = 1.0
        self.restore_points_to_domain_func = restore_points_func
        self._setup_problem_description = self.adv_diff_swarm_problem_description

        self.is_setup = False
        self.u_star_is_valid = False

        if isinstance(V_fn, uw.discretisation._MeshVariable):
            self._V_fn = V_fn.sym
        else:
            self._V_fn = V_fn

        # default values for properties
        self.f = sympy.Matrix.zeros(1, 1)
        self._constitutive_model = None

        return

    def adv_diff_swarm_problem_description(self):
        # f0 residual term
        self._f0 = self.F0 - self.f + self.DuDt.bdf() / self.delta_t

        # f1 residual term
        self._f1 = self.F1 + self.DFDt.adams_moulton_flux()

        return

    @property
    def V_fn(self):
        return self._V_fn

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

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        self.is_setup = False
        self._f = sympy.Matrix((value,))

    @property
    def constitutive_model(self):
        return self._constitutive_model

    @constitutive_model.setter
    def constitutive_model(self, model):
        ### checking if it's an instance
        if isinstance(model, uw.constitutive_models.Constitutive_Model):
            self._constitutive_model = model

            ### update history terms using setters
            self.DFDt.psi_fn = model._q(self.Unknowns.u)

        ### checking if it's a class
        elif type(model) == type(uw.constitutive_models.Constitutive_Model):
            self._constitutive_model = model(self.Unknowns)

            ### update history terms using setters
            self.DFDt.psi_fn = model._q(self.u)

        ### Raise an error if it's neither
        else:
            raise RuntimeError(
                "constitutive_model must be a valid class or instance of a valid class"
            )

    @timing.routine_timer_decorator
    def estimate_dt(self):
        """
        Calculates an appropriate advective timestep for the given
        mesh and diffusivity configuration.
        """
        # ## update the diffusivity values
        # self.k_proj = uw.systems.Projection(self.mesh, self.k)
        # self.k_proj.uw_function = self.constitutive_model.Parameters.diffusivity
        # self.k_proj.smoothing = 0.0
        # self.k_proj.solve(_force_setup=True)

        if isinstance(self.constitutive_model.Parameters.diffusivity, sympy.Expr):
            k = uw.function.evaluate(
                sympy.sympify(self.constitutive_model.Parameters.diffusivity),
                self.mesh._centroids,
                self.mesh.N,
            )
            max_diffusivity = k.max()
        else:
            k = self.constitutive_model.Parameters.diffusivity
            max_diffusivity = k

        ### required modules
        from mpi4py import MPI

        # with self.mesh.access(self.k):
        #     ## get local max diff value
        #     max_diffusivity = self.k.data[:, 0].max()

        ## get global max dif value
        comm = MPI.COMM_WORLD
        diffusivity_glob = comm.allreduce(max_diffusivity, op=MPI.MAX)

        ### get the velocity values
        vel = uw.function.evaluate(
            self.V_fn,
            self.mesh._centroids,
            self.mesh.N,
        )

        ### get global velocity from velocity field
        max_magvel = np.linalg.norm(vel, axis=1).max()
        max_magvel_glob = comm.allreduce(max_magvel, op=MPI.MAX)

        ## get radius
        min_dx = self.mesh.get_min_radius()

        ## estimate dt of adv and diff components

        if max_magvel_glob == 0.0:
            dt_diff = (min_dx**2) / diffusivity_glob
            dt_estimate = dt_diff
        elif diffusivity_glob == 0.0:
            dt_adv = min_dx / max_magvel_glob
            dt_estimate = dt_adv
        else:
            dt_diff = (min_dx**2) / diffusivity_glob
            dt_adv = min_dx / max_magvel_glob
            dt_estimate = min(dt_diff, dt_adv)

        return dt_estimate

    @timing.routine_timer_decorator
    def solve(
        self,
        zero_init_guess: bool = True,
        timestep: float = None,
        _force_setup: bool = False,
        verbose=False,
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

        if _force_setup:
            self.is_setup = False

        if not self.constitutive_model._solver_is_setup:
            self.is_setup = False
            self.DFDt.psi_fn = self.constitutive_model.flux.T

        if not self.is_setup:
            self._setup_pointwise_functions(verbose)
            self._setup_discretisation(verbose)
            self._setup_solver(verbose)

        # Update SemiLagrange Flux terms
        self.DuDt.update(timestep, verbose=verbose)
        self.DFDt.update(timestep, verbose=verbose)

        super().solve(zero_init_guess, _force_setup)

        self.is_setup = True
        self.constitutive_model._solver_is_setup = True

        return

    @timing.routine_timer_decorator
    def _setup_terms(self):
        if self.projection:
            self._u_star_projector.bcs = self.bcs
            self._u_star_projector._setup_pointwise_functions()  # _setup_terms()

        super()._setup_pointwise_functions()  # ._setup_terms()


## This one is already updated to work with the Semi-Lagrange D_Dt
class SNES_NavierStokes_SLCN(SNES_Stokes_SaddlePt):
    r"""
    This class provides a solver for the Navier-Stokes (vector Advection-Diffusion) equation which is similar to that
    used in the Semi-Lagrange Crank-Nicholson method (Spiegelman & Katz, 2006) but using a
    distributed sampling of upstream values taken from an arbitrary swarm variable.

    $$
    \color{Green}{\underbrace{ \Bigl[ \frac{\partial \mathbf{u} }{\partial t} -
                                      \left( \mathbf{u} \cdot \nabla \right) \mathbf{u} \ \Bigr]}_{\dot{\mathbf{f}}}} -
        \nabla \cdot
            \color{Blue}{\underbrace{\Bigl[ \frac{\boldsymbol{\eta}}{2} \left(
                    \nabla \mathbf{u} + \nabla \mathbf{u}^T \right) - p \mathbf{I} \Bigr]}_{\mathbf{F}}} =
            \color{Maroon}{\underbrace{\Bigl[ \mathbf{f} \Bigl] }_{\mathbf{f}}}
    $$

    The term $\mathbf{F}$ relates diffusive fluxes to gradients in the unknown $u$. The advective flux that results from having gradients along
    the direction of transport (given by the velocity vector field $\mathbf{v}$ ) are included in the $\dot{\mathbf{f}}$ term.

    The term $\dot{\mathbf{f}}$ involves upstream sampling to find the value $u^{ * }$ which represents the value of $u$ at
    the beginning of the timestep. This is achieved using a `swarmVariable` that carries history information along the flow path.
    A dense sampling is required to achieve similar accuracy to the original SLCN approach but it allows the use of a single swarm
    for history tracking of variables with different interpolation order and for material tracking. The user is required to supply
    **and update** the swarmVariable representing $u^{ * }$

    ## Properties

      - The unknown is $u$.

      - The velocity field is $\mathbf{v}$ and is provided as a `sympy` function to allow operations such as time-averaging to be
        calculated in situ (e.g. `V_Field = v_solution.sym`)

      - The history variable is $u^*$ and is provided in the form of a `sympy` function. It is the user's responsibility to keep this
        variable updated.

      - The diffusivity tensor, $\kappa$ is provided by setting the `constitutive_model` property to
        one of the scalar `uw.constitutive_models` classes and populating the parameters.
        It is usually a constant or a function of position / time and may also be non-linear
        or anisotropic.

      - Volumetric sources of $u$ are specified using the $f$ property and can be any valid combination of `sympy` functions of position and
        `meshVariable` or `swarmVariable` types.

    ## Notes

      - The solver requires relatively high order shape functions to accurately interpolate the history terms.
        Spiegelman & Katz recommend cubic or higher degree for $u$ but this is not checked.

    ## Reference

    Spiegelman, M., & Katz, R. F. (2006). A semi-Lagrangian Crank-Nicolson algorithm for the numerical solution
    of advection-diffusion problems. Geochemistry, Geophysics, Geosystems, 7(4). https://doi.org/10.1029/2005GC001073
    """

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        ## feedback on this instance
        display(Latex(r"$\quad\mathrm{u} = $ " + self.u.sym._repr_latex_()))
        display(Latex(r"$\quad\mathbf{p} = $ " + self.p.sym._repr_latex_()))
        display(Latex(r"$\quad\Delta t = $ " + self.delta_t._repr_latex_()))
        display(Latex(rf"$\quad\rho = $ {self.rho}"))

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        velocityField: uw.discretisation.MeshVariable,
        pressureField: uw.discretisation.MeshVariable,
        rho: Optional[float] = 0.0,
        solver_name: Optional[str] = "",
        p_continuous: Optional[bool] = True,
        restore_points_func: Callable = None,
        verbose: Optional[bool] = False,
        order: Optional[int] = 1,
    ):
        ## Parent class will set up default values and load u_Field into the solver

        super().__init__(
            mesh,
            velocityField,
            pressureField,
            None,
            None,
            order,
            p_continuous,
            solver_name,
            verbose,
        )

        # These are unique to the advection solver
        self.delta_t = sympy.oo
        self.is_setup = False
        self.rho = rho
        self._first_solve = True

        self._order = order
        self._constitutive_model = None

        self._penalty = 0.0
        self._bodyforce = sympy.Matrix([[0] * self.mesh.dim])

        # self._E = self.mesh.vector.strain_tensor(self.u.sym)
        self._Estar = None

        self._constraints = sympy.Matrix((self.div_u,))

        ### sets up DuDt and DFDt ...

        self.Unknowns.DuDt = uw.systems.ddt.SemiLagrangian(
            self.mesh,
            self.u.sym,
            self.u.sym * bc_mask,
            vtype=uw.VarType.VECTOR,
            degree=self.u.degree,
            continuous=self.u.continuous,
            varsymbol=self.u.symbol,
            verbose=self.verbose,
            bcs=self.essential_bcs,
            order=self._order,
            smoothing=0.0,
        )

        self.Unknowns.DFDt = uw.systems.ddt.SemiLagrangian(
            self.mesh,
            sympy.Matrix.zeros(self.mesh.dim, self.mesh.dim),
            self.u.sym,
            vtype=uw.VarType.SYM_TENSOR,
            degree=self.u.degree - 1,
            continuous=True,
            varsymbol=rf"{{F[ {self.u.symbol} ] }}",
            verbose=self.verbose,
            bcs=None,
            order=self._order,
            smoothing=0.0,
        )

        self.restore_points_to_domain_func = restore_points_func
        self._setup_problem_description = self.navier_stokes_slcn_problem_description

        return

    def navier_stokes_slcn_problem_description(self):
        N = self.mesh.N

        # f0 residual term
        self._u_f0 = (
            self.F0
            - self.bodyforce
            + self.rho * self.Unknowns.DuDt.bdf() / self.delta_t
        )

        # f1 residual term
        self._u_f1 = (
            self.F1
            + self.Unknowns.DFDt.adams_moulton_flux()
            - sympy.eye(self.mesh.dim) * (self.p.sym[0])
        )
        self._p_f0 = sympy.simplify(self.PF0 + sympy.Matrix((self.constraints)))

        return

    @property
    def delta_t(self):
        return self._delta_t

    @delta_t.setter
    def delta_t(self, value):
        self.is_setup = False
        self._delta_t = sympify(value)

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        self.is_setup = False
        self._f = sympy.Matrix((value,))

    @property
    def div_u(self):
        E = self.strainrate
        divergence = E.trace()
        return divergence

    @property
    def strainrate(self):
        return sympy.Matrix(self.Unknowns.E)

    @property
    def DuDt(self):
        return self._DuDt

    @DuDt.setter
    def DuDt(
        self,
        DuDt_value: Union[SemiLagrangian_DDt, Lagrangian_DDt],
    ):
        self._DuDt = DuDt_value
        self._solver_is_setup = False

    @property
    def DFDt(self):
        return self._DFDt

    @DFDt.setter
    def DFDt(
        self,
        DFDt_value: Union[SemiLagrangian_DDt, Lagrangian_DDt],
    ):
        self._DFDt = DFDt_value
        self._solver_is_setup = False

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
    def solve(
        self,
        zero_init_guess: bool = True,
        timestep: float = None,
        _force_setup: bool = False,
        verbose=False,
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

        if _force_setup:
            self.is_setup = False

        if not self.constitutive_model._solver_is_setup:
            self.is_setup = False
            self.Unknowns.DFDt.psi_fn = self.constitutive_model.flux.T

        if not self.is_setup:
            self._setup_pointwise_functions(verbose)
            self._setup_discretisation(verbose)
            self._setup_solver(verbose)

        # Update SemiLagrange Flux terms (May need to update to match pre-post solve pattern)
        self.Unknowns.DuDt.update(timestep, verbose=verbose)
        self.Unknowns.DFDt.update(timestep, verbose=verbose)

        super().solve(
            zero_init_guess,
            _force_setup=_force_setup,
            verbose=verbose,
            picard=0,
        )

        self.is_setup = True
        self.constitutive_model._solver_is_setup = True

        return


# This one is already updated to work with the Lagrange D_Dt
class SNES_NavierStokes(SNES_Stokes_SaddlePt):
    r"""
    This class provides a solver for the Navier-Stokes (vector Advection-Diffusion) equation which is similar to that
    used in the Semi-Lagrange Crank-Nicholson method (Spiegelman & Katz, 2006) but using a
    distributed sampling of upstream values taken from an arbitrary swarm variable.

    $$
    \color{Green}{\underbrace{ \Bigl[ \frac{\partial \mathbf{u} }{\partial t} -
                                      \left( \mathbf{u} \cdot \nabla \right) \mathbf{u} \ \Bigr]}_{\dot{\mathbf{f}}}} -
        \nabla \cdot
            \color{Blue}{\underbrace{\Bigl[ \frac{\boldsymbol{\eta}}{2} \left(
                    \nabla \mathbf{u} + \nabla \mathbf{u}^T \right) - p \mathbf{I} \Bigr]}_{\mathbf{F}}} =
            \color{Maroon}{\underbrace{\Bigl[ \mathbf{f} \Bigl] }_{\mathbf{f}}}
    $$

    The term $\mathbf{F}$ relates diffusive fluxes to gradients in the unknown $u$. The advective flux that results from having gradients along
    the direction of transport (given by the velocity vector field $\mathbf{v}$ ) are included in the $\dot{\mathbf{f}}$ term.

    The term $\dot{\mathbf{f}}$ involves upstream sampling to find the value $u^{ * }$ which represents the value of $u$ at
    the beginning of the timestep. This is achieved using a `swarmVariable` that carries history information along the flow path.
    A dense sampling is required to achieve similar accuracy to the original SLCN approach but it allows the use of a single swarm
    for history tracking of variables with different interpolation order and for material tracking. The user is required to supply
    **and update** the swarmVariable representing $u^{ * }$

    ## Properties

      - The unknown is $u$.

      - The velocity field is $\mathbf{v}$ and is provided as a `sympy` function to allow operations such as time-averaging to be
        calculated in situ (e.g. `V_Field = v_solution.sym`)

      - The history variable is $u^*$ and is provided in the form of a `sympy` function. It is the user's responsibility to keep this
        variable updated.

      - The diffusivity tensor, $\kappa$ is provided by setting the `constitutive_model` property to
        one of the scalar `uw.constitutive_models` classes and populating the parameters.
        It is usually a constant or a function of position / time and may also be non-linear
        or anisotropic.

      - Volumetric sources of $u$ are specified using the $f$ property and can be any valid combination of `sympy` functions of position and
        `meshVariable` or `swarmVariable` types.

    ## Notes

      - The solver requires relatively high order shape functions to accurately interpolate the history terms.
        Spiegelman & Katz recommend cubic or higher degree for $u$ but this is not checked.

    ## Reference

    Spiegelman, M., & Katz, R. F. (2006). A semi-Lagrangian Crank-Nicolson algorithm for the numerical solution
    of advection-diffusion problems. Geochemistry, Geophysics, Geosystems, 7(4). https://doi.org/10.1029/2005GC001073
    """

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        ## feedback on this instance
        display(Latex(r"$\quad\mathrm{u} = $ " + self.u.sym._repr_latex_()))
        display(Latex(r"$\quad\mathbf{p} = $ " + self.p.sym._repr_latex_()))
        display(Latex(r"$\quad\Delta t = $ " + self.delta_t._repr_latex_()))
        display(Latex(rf"$\quad\rho = $ {self.rho}"))

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        velocityField: uw.discretisation.MeshVariable,
        pressureField: uw.discretisation.MeshVariable,
        DuDt: Union[SemiLagrangian_DDt, Lagrangian_DDt] = None,
        DFDt: Union[SemiLagrangian_DDt, Lagrangian_DDt] = None,
        rho: Optional[float] = 0.0,
        restore_points_func: Callable = None,
        order: Optional[int] = 2,
        p_continuous: Optional[bool] = False,
        solver_name: Optional[str] = "",
        verbose: Optional[bool] = False,
        bc_mask_fn: Optional[sympy.Function] = 1,
    ):
        ## Parent class will set up default values and load u_Field into the solver
        super().__init__(
            mesh,
            velocityField,
            pressureField,
            DuDt,
            DFDt,
            order,
            p_continuous,
            solver_name,
            verbose,
        )

        # These are unique to the advection solver
        self.delta_t = sympy.oo

        self.is_setup = False
        self.rho = rho
        self._first_solve = True

        self.restore_points_to_domain_func = restore_points_func
        self._setup_problem_description = self.navier_stokes_problem_description

        self._order = order
        self._penalty = 0.0

        self._constitutive_model = None

        self.restore_points_to_domain_func = restore_points_func
        self._setup_problem_description = self.navier_stokes_problem_description

        self._bodyforce = sympy.Matrix([[0] * self.mesh.dim])

        self._constitutive_model = None

        # self._E = self.mesh.vector.strain_tensor(self.u.sym)
        self._Estar = None

        self._constraints = sympy.Matrix((self.div_u,))

        ### sets up DuDt and DFDt
        ## ._setup_history_terms()

        # If DuDt is not provided, then we can build a SLCN version
        if self.Unknowns.DuDt is None:
            self.Unknowns.DuDt = uw.systems.ddt.SemiLagrangian(
                self.mesh,
                self.u.sym,
                self.u.sym,
                vtype=uw.VarType.VECTOR,
                degree=self.u.degree,
                continuous=self.u.continuous,
                varsymbol=self.u.symbol,
                verbose=self.verbose,
                bcs=self.essential_bcs,
                order=self._order,
                smoothing=0.0,
                bc_mask_fn=bc_mask_fn,
            )

        # F (at least for N-S) is a nodal point variable so there is no benefit
        # to treating it as a swarm variable. We'll define and use our own SL tracker
        # as we do in the SLCN version. We'll leave the option for an over-ride.

        if self.Unknowns.DFDt is None:
            self.Unknowns.DFDt = uw.systems.ddt.SemiLagrangian(
                self.mesh,
                sympy.Matrix.zeros(self.mesh.dim, self.mesh.dim),
                self.u.sym,
                vtype=uw.VarType.SYM_TENSOR,
                degree=self.u.degree - 1,
                continuous=False,
                varsymbol=rf"{{F[ {self.u.symbol} ] }}",
                verbose=self.verbose,
                bcs=None,
                order=self._order,
                smoothing=0.0,
                bc_mask_fn=None,
            )

        ## Add in the history terms provided ...

        return

    def navier_stokes_problem_description(self):
        # f0 residual term

        self._u_f0 = (
            self.F0 - self.bodyforce + self.rho * self.DuDt.bdf() / self.delta_t
        )

        # f1 residual term
        self._u_f1 = (
            self.F1
            + self.DFDt.adams_moulton_flux()
            - sympy.eye(self.mesh.dim) * (self.p.sym[0])
        )
        self._p_f0 = sympy.simplify(self.PF0 + sympy.Matrix((self.constraints)))

        return

    @property
    def delta_t(self):
        return self._delta_t

    @delta_t.setter
    def delta_t(self, value):
        self.is_setup = False
        self._delta_t = sympify(value)

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        self.is_setup = False
        self._f = sympy.Matrix((value,))

    @property
    def div_u(self):
        E = self.strainrate
        divergence = E.trace()
        return divergence

    @property
    def strainrate(self):
        return sympy.Matrix(self.Unknowns.E)

    @property
    def DuDt(self):
        return self.Unknowns.DuDt

    @DuDt.setter
    def DuDt(
        self,
        DuDt_value: Union[SemiLagrangian_DDt, Lagrangian_DDt],
    ):
        self.Unknowns.DuDt = DuDt_value
        self._solver_is_setup = False

    @property
    def DFDt(self):
        return self.Unknowns.DFDt

    @DFDt.setter
    def DFDt(
        self,
        DFDt_value: Union[SemiLagrangian_DDt, Lagrangian_DDt],
    ):
        self.Unknowns.DFDt = DFDt_value
        self._solver_is_setup = False

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
    def solve(
        self,
        zero_init_guess: bool = True,
        timestep: float = None,
        _force_setup: bool = False,
        verbose=False,
        evalf=False,
        order=None,
    ):
        """
        Generates solution to constructed system.

        Params
        ------
        zero_init_guess:
            If `True`, a zero initial guess will be used for the
            system solution. Otherwise, the current values of `self.u` will be used.
        """

        if order is None or order > self._order:
            order = self._order

        if timestep is not None and timestep != self.delta_t:
            self.delta_t = timestep  # this will force an initialisation because the functions need to be updated

        if _force_setup:
            self.is_setup = False

        if not self.constitutive_model._solver_is_setup:
            self.is_setup = False
            self.DFDt.psi_fn = self.constitutive_model.flux.T

        if not self.is_setup:
            self._setup_pointwise_functions(verbose)
            self._setup_discretisation(verbose)
            self._setup_solver(verbose)

        if uw.mpi.rank == 0 and verbose:
            print(f"NS solver - pre-solve DuDt update", flush=True)

        # Update SemiLagrange Flux terms
        self.DuDt.update_pre_solve(timestep, verbose=verbose, evalf=evalf)
        self.DFDt.update_pre_solve(timestep, verbose=verbose, evalf=evalf)

        if uw.mpi.rank == 0 and verbose:
            print(f"NS solver - solve Stokes flow", flush=True)

        super().solve(
            zero_init_guess,
            _force_setup=_force_setup,
            verbose=verbose,
            picard=0,
        )

        if uw.mpi.rank == 0 and verbose:
            print(f"NS solver - post-solve DuDt update", flush=True)

        self.DuDt.update_post_solve(timestep, verbose=verbose, evalf=evalf)
        self.DFDt.update_post_solve(timestep, verbose=verbose, evalf=evalf)

        self.is_setup = True
        self.constitutive_model._solver_is_setup = True

        return