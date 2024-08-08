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
        DuDt: Union[SemiLagrangian_DDt, Lagrangian_DDt] = None,
        DFDt: Union[SemiLagrangian_DDt, Lagrangian_DDt] = None,
    ):
        ## Keep track

        ## Parent class will set up default values etc
        super().__init__(
            mesh,
            u_Field,
            degree,
            solver_name,
            verbose,
            DuDt=DuDt,
            DFDt=DFDt,
        )

        if solver_name == "":
            solver_name = "Poisson_{}_".format(self.instance_number)

        # Register the problem setup function
        self._setup_problem_description = self.poisson_problem_description

        # default values for properties
        self.f = sympy.Matrix.zeros(1, 1)

        self._constitutive_model = None

    @property
    def F0(self):

        f0_val = uw.function.expression(
            r"f_0 \left( \mathbf{u} \right)",
            -self.f,
            "Poisson pointwise force term: f_0(u)",
        )

        # backward compatibility
        self._f0 = f0_val

        return f0_val

    @property
    def F1(self):

        F1_val = uw.function.expression(
            r"\mathbf{F}_1\left( \mathbf{u} \right)",
            sympy.simplify(self.constitutive_model.flux.T),
            "Poisson pointwise flux term: F_1(u)",
        )

        # backward compatibility
        self._f1 = F1_val

        return F1_val

    @timing.routine_timer_decorator
    def poisson_problem_description(self):
        # f1 residual term (weighted integration) - scalar function
        self._f0 = self.F0.sym

        # f1 residual term (integration by parts / gradients)
        # isotropic
        self._f1 = self.F1.sym

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
        DuDt=None,
        DFDt=None,
    ):
        ## Parent class will set up default values etc
        super().__init__(
            mesh,
            h_Field,
            degree,
            solver_name,
            verbose,
            DuDt=DuDt,
            DFDt=DFDt,
        )

        if solver_name == "":
            self.solver_name = "Darcy_{}_".format(self.instance_number)

        # Register the problem setup function
        self._setup_problem_description = self.darcy_problem_description

        # default values for properties
        self._f = sympy.Matrix([0])
        self._k = 1

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

    @property
    def F0(self):

        f0_val = uw.function.expression(
            r"f_0 \left( \mathbf{u} \right)",
            -self.f,
            "Darcy pointwise force term: f_0(u)",
        )

        # backward compatibility
        self._f0 = f0_val.sym

        return f0_val

    @property
    def F1(self):

        F1_val = uw.function.expression(
            r"\mathbf{F}_1\left( \mathbf{u} \right)",
            sympy.simplify(self.darcy_flux),
            "Darcy pointwise flux term: F_1(u)",
        )

        # backward compatibility
        self._f1 = F1_val.sym
        self._v_projector.uw_function = -F1_val.sym

        return F1_val

    @timing.routine_timer_decorator
    def darcy_problem_description(self):
        # f1 residual term (weighted integration)
        self._f0 = self.F0.sym

        # f1 residual term (integration by parts / gradients)
        self._f1 = self.F1.sym

        # Flow calculation
        self._v_projector.uw_function = -self.F1.sym

        return

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        self.is_setup = False
        self._f = sympy.Matrix((value,))

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
        # Not used in Stokes, but may be used in NS, VE etc
        DuDt: Union[SemiLagrangian_DDt, Lagrangian_DDt] = None,
        DFDt: Union[SemiLagrangian_DDt, Lagrangian_DDt] = None,
    ):
        super().__init__(
            mesh,
            velocityField,
            pressureField,
            degree,
            p_continuous,
            solver_name,
            verbose,
            DuDt=DuDt,
            DFDt=DFDt,
        )

        # change this to be less generic
        if solver_name == "":
            self.name = "Stokes_{}_".format(self.instance_number)

        self._degree = degree
        # User-facing operations are matrices / vectors by preference

        self._Estar = None

        self._penalty = uw.function.expression(R"\uplambda", 0, "Numerical Penalty")
        self._constraints = sympy.Matrix(
            (self.div_u,)
        )  # by default, incompressibility constraint

        self._bodyforce = sympy.Matrix([[0] * self.mesh.dim])

        self._setup_problem_description = self.stokes_problem_description

        # this attrib records if we need to setup the problem (again)
        self.is_setup = False

        self._constitutive_model = None

        return

    ## Problem Description:
    ##  F0 - velocity equation forcing terms
    ##  F1 - velocity equation flux terms
    ##  PF0 - pressure / constraint equation forcing terms

    @property
    def F0(self):

        f0 = uw.function.expression(
            r"\mathbf{f}_0\left( \mathbf{u} \right)",
            -self.bodyforce,
            "Stokes pointwise force term: f_0(u)",
        )

        # backward compatibility
        self._u_f0 = f0

        return f0

    @property
    def F1(self):

        dim = self.mesh.dim

        F1_val = uw.function.expression(
            r"\mathbf{F}_1\left( \mathbf{u} \right)",
            sympy.simplify(self.stress + self.penalty * self.div_u * sympy.eye(dim)),
            "Stokes pointwise flux term: F_1(u)",
        )

        # backward compatibility
        self._u_f1 = F1_val

        return F1_val

    @property
    def PF0(self):

        f0 = uw.function.expression(
            r"\mathbf{f}_0\left( \mathbf{p} \right)",
            sympy.simplify(sympy.Matrix((self.constraints))),
            "Pointwise flux term: f_0(p)",
        )

        # backward compatibility
        self._p_f0 = f0

        return f0

    @timing.routine_timer_decorator
    def stokes_problem_description(self):

        # f0 residual term
        self._u_f0 = self.F0.sym

        # f1 residual term
        self._u_f1 = self.F1.sym

        # p0 residual term
        self._p_f0 = self.PF0.sym

        return

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
        self._penalty.sym = value


class SNES_VE_Stokes(SNES_Stokes):
    r"""
    This class provides functionality for a discrete representation
    of the Stokes flow equations assuming an incompressibility
    (or near-incompressibility) constraint and with a flux history
    term included to allow for viscoelastic modelling.

    All other functionality is inherited from SNES_Stokes

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
        order: Optional[int] = 2,
        p_continuous: Optional[bool] = True,
        solver_name: Optional[str] = "",
        verbose: Optional[bool] = False,
        # DuDt Not used in VE, but may be in child classes
        DuDt: Union[SemiLagrangian_DDt, Lagrangian_DDt] = None,
        DFDt: Union[SemiLagrangian_DDt, Lagrangian_DDt] = None,
    ):

        # Stokes is parent (will not build DuDt or DFDt)
        super().__init__(
            mesh,
            velocityField,
            pressureField,
            degree,
            p_continuous,
            solver_name,
            verbose,
            DuDt=DuDt,
            DFDt=DFDt,
        )

        # change this to be less generic
        if solver_name == "":
            self.name = "VE_Stokes_{}_".format(self.instance_number)

        self._order = order  # VE time-order

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

    @property
    def delta_t(self):

        return self.constitutive_model.Parameters.dt_elastic

    ## Solver needs to update the stress history terms as well as call the SNES solve:

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

        if timestep is None:
            timestep = self.delta_t.sym

        if timestep != self.delta_t:
            self._constitutive_model.Parameters.elastic_dt = timestep  # this will force an initialisation because the functions need to be updated

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
            print(f"VE Stokes solver - pre-solve DFDt update", flush=True)

        # Update SemiLagrange Flux terms
        self.DFDt.update_pre_solve(timestep, verbose=verbose, evalf=evalf)

        if uw.mpi.rank == 0 and verbose:
            print(f"VE Stokes solver - solve Stokes flow", flush=True)

        super().solve(
            zero_init_guess,
            _force_setup=_force_setup,
            verbose=verbose,
            picard=0,
        )

        if uw.mpi.rank == 0 and verbose:
            print(f"VEP Stokes solver - post-solve DFDt update", flush=True)

        self.DFDt.update_post_solve(timestep, verbose=verbose, evalf=evalf)

        self.is_setup = True
        self.constitutive_model._solver_is_setup = True

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

    @property
    def F0(self):

        f0_val = uw.function.expression(
            r"f_0 \left( \mathbf{u} \right)",
            (self.u.sym - self.uw_function) * self.uw_weighting_function,
            "Scalar Projection pointwise misfit term: f_0(u)",
        )

        # backward compatibility
        self._f0 = f0_val

        return f0_val

    @property
    def F1(self):

        F1_val = uw.function.expression(
            r"\mathbf{F}_1\left( \mathbf{u} \right)",
            self.smoothing * self.mesh.vector.gradient(self.u.sym),
            "Scalar projection pointwise smoothing term: F_1(u)",
        )

        # backward compatibility
        self._f1 = F1_val

        return F1_val

    @timing.routine_timer_decorator
    def projection_problem_description(self):
        # residual terms - defines the problem:
        # solve for a best fit to the continuous mesh
        # variable given the values in self.function
        # F0 is left in place for the user to inject
        # non-linear constraints if required

        self._f0 = self.F0.sym

        # F1 is left in the users control ... e.g to add other gradient constraints to the stiffness matrix

        self._f1 = self.F1.sym

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

    @property
    def F0(self):

        f0_val = uw.function.expression(
            r"f_0 \left( \mathbf{u} \right)",
            (self.u.sym - self.uw_function) * self.uw_weighting_function,
            "Vector projection pointwise misfit term: f_0(u)",
        )

        # backward compatibility
        self._f0 = f0_val

        return f0_val

    @property
    def F1(self):

        F1_val = uw.function.expression(
            r"\mathbf{F}_1\left( \mathbf{u} \right)",
            self.smoothing * self.Unknowns.E
            + self.penalty
            * self.mesh.vector.divergence(self.u.sym)
            * sympy.eye(self.mesh.dim),
            "Vector projection pointwise smoothing term: F_1(u)",
        )

        # backward compatibility
        self._f1 = F1_val

        return F1_val

    @timing.routine_timer_decorator
    def projection_problem_description(self):
        # residual terms - defines the problem:
        # solve for a best fit to the continuous mesh
        # variable given the values in self.function
        # F0 is left in place for the user to inject
        # non-linear constraints if required

        self._f0 = self.F0.sym

        # F1 is left in the users control ... e.g to add other gradient constraints to the stiffness matrix

        self._f1 = (
            self.F1.sym
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
            degree=degree,
            solver_name=solver_name,
            verbose=verbose,
        )

        if solver_name == "":
            solver_name = "TProj{}_".format(self.instance_number)

        return

    ## Need to over-ride solve method to run over all components

    def solve(self, verbose=False):
        # Loop over the components of the tensor. If this is a symmetric
        # tensor, we'll usually be given the 1d form to prevent duplication

        # if self.t_field.sym_1d.shape != self.uw_function.shape:
        #     raise ValueError(
        #         "Tensor shapes for uw_function and MeshVariable are not the same"
        #     )

        symm = self.t_field.sym.is_symmetric()

        for i in range(self.uw_function.shape[0]):
            for j in range(self.uw_function.shape[1]):

                if symm and j > i:
                    continue

                self.uw_scalar_function = sympy.Matrix([[self.uw_function[i, j]]])

                with self.mesh.access(self.u):
                    self.u.data[:, 0] = self.t_field[i, j].data[:]

                # solve the projection for the scalar sub-problem
                super().solve(verbose=verbose)

                with self.mesh.access(self.t_field):
                    self.t_field[i, j].data[:] = self.u.data[:, 0]

        # That might be all ...

    # This is re-defined so it uses uw_scalar_function

    @property
    def F0(self):

        f0_val = uw.function.expression(
            r"f_0 \left( \mathbf{u} \right)",
            (self.u.sym - self.uw_scalar_function) * self.uw_weighting_function,
            "Scalar subproblem of tensor projection: f_0(u)",
        )

        # backward compatibility
        self._f0 = f0_val

        return f0_val

    @property
    def F1(self):

        F1_val = uw.function.expression(
            r"\mathbf{F}_1\left( \mathbf{u} \right)",
            self.smoothing * self.mesh.vector.gradient(self.u.sym),
            "Scalar subproblem of tensor projection (smoothing): F_1(u)",
        )

        # backward compatibility
        self._f1 = F1_val

        return F1_val

    @property
    def uw_scalar_function(self):
        return self._uw_scalar_function

    @uw_scalar_function.setter
    def uw_scalar_function(self, user_uw_function):
        self.is_setup = False
        self._uw_scalar_function = user_uw_function


# #################################################
# # Swarm-based advection-diffusion
# # solver based on SNES_Poisson and swarm-variable
# # projection
# #
# #################################################


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
        DuDt: Union[SemiLagrangian_DDt, Lagrangian_DDt] = None,
        DFDt: Union[SemiLagrangian_DDt, Lagrangian_DDt] = None,
    ):
        ## Parent class will set up default values etc
        super().__init__(
            mesh,
            u_Field,
            u_Field.degree,
            solver_name,
            verbose,
            DuDt=DuDt,
            DFDt=DFDt,
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
        self._delta_t = uw.function.expression(
            R"\Delta t", 0, "Physically motivated timestep"
        )
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
            continuous=True,
            varsymbol=rf"{{F[ {self.u.symbol} ] }}",
            verbose=verbose,
            bcs=None,
            order=order,
            smoothing=0.0,
        )

        return

    @property
    def F0(self):

        f0 = uw.function.expression(
            r"f_0 \left( \mathbf{u} \right)",
            -self.f + self.DuDt.bdf() / self.delta_t,
            "Poisson pointwise force term: f_0(u)",
        )

        # backward compatibility
        self._f0 = f0

        return f0

    @property
    def F1(self):

        F1_val = uw.function.expression(
            r"\mathbf{F}_1\left( \mathbf{u} \right)",
            self.DFDt.adams_moulton_flux(),
            "Poisson pointwise flux term: F_1(u)",
        )

        # backward compatibility
        self._f1 = F1_val

        return F1_val

    def adv_diff_slcn_problem_description(self):
        # f0 residual term
        self._f0 = self.F0.sym

        # f1 residual term
        self._f1 = self.F1.sym

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
        self._delta_t.sym = value

    @timing.routine_timer_decorator
    def estimate_dt(self):
        r"""
        Calculates an appropriate timestep for the given
        mesh and diffusivity configuration. This is an implicit solver
        so the $\delta_t$ should be interpreted as:

            - ${\delta t}_\textrm{diff}: a typical time for the diffusion front to propagate across an element
            - ${\delta t}_\textrm{adv}: a typical element-crossing time for a fluid parcel

            returns (${\delta t}_\textrm{diff}, ${\delta t}_\textrm{adv})
        """

        if isinstance(self.constitutive_model.Parameters.diffusivity, sympy.Expr):
            if uw.function.fn_is_constant_expr(
                self.constitutive_model.Parameters.diffusivity
            ):
                max_diffusivity = uw.function.evaluate(
                    self.constitutive_model.Parameters.diffusivity
                )
            else:
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
        comm = uw.mpi.comm
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

        return dt_diff, dt_adv

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
        display(Latex(rf"$\quad\rho = $" + self.rho._repr_latex_()))

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        velocityField: uw.discretisation.MeshVariable,
        pressureField: uw.discretisation.MeshVariable,
        rho: Optional[float] = 0.0,
        restore_points_func: Callable = None,
        order: Optional[int] = 2,
        p_continuous: Optional[bool] = False,
        solver_name: Optional[str] = "",
        verbose: Optional[bool] = False,
        DuDt: Union[SemiLagrangian_DDt, Lagrangian_DDt] = None,
        DFDt: Union[SemiLagrangian_DDt, Lagrangian_DDt] = None,
    ):
        ## Parent class will set up default values and load u_Field into the solver
        super().__init__(
            mesh,
            velocityField,
            pressureField,
            order,
            p_continuous,
            solver_name,
            verbose,
            DuDt=DuDt,
            DFDt=DFDt,
        )

        # These are unique to the advection solver
        self._delta_t = uw.function.expression(
            r"\Delta t", sympy.oo, "Navier-Stokes timestep"
        )

        self.is_setup = False
        self._rho = uw.function.expression(R"{\uprho}", rho, "Density")
        self._first_solve = True

        self._order = order
        self._penalty = uw.function.expression(
            R"{\uplambda}", 0, "Incompressibility Penalty"
        )

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
            )

        # F (at least for N-S) is a nodal point variable so there is no benefit
        # to treating it as a swarm variable. We'll define and use our own SL tracker
        # as we do in the SLCN version. We'll leave the option for an over-ride.

        self.Unknowns.DFDt = uw.systems.ddt.SemiLagrangian(
            self.mesh,
            sympy.Matrix.zeros(self.mesh.dim, self.mesh.dim),
            self.u.sym,
            vtype=uw.VarType.SYM_TENSOR,
            degree=1,  # self.u.degree - 1,
            continuous=False,
            varsymbol=rf"{{F[ {self.u.symbol} ] }}",
            verbose=self.verbose,
            bcs=None,
            order=self._order,
            smoothing=0.0,
        )

        ## Add in the history terms provided ...

        return

    @property
    def F0(self):

        DuDt = self.Unknowns.DuDt

        # I think this should be bdf(1) ... the higher order
        # terms are introduced through the adams_moulton fluxes

        f0 = uw.function.expression(
            r"\mathbf{f}_0\left( \mathbf{u} \right)",
            -self.bodyforce + self.rho * DuDt.bdf(1) / self.delta_t,
            "NStokes pointwise force term: f_0(u)",
        )

        self._u_f0 = f0

        return f0

    @property
    def F1(self):
        dim = self.mesh.dim

        DFDt = self.Unknowns.DFDt

        if DFDt is not None:
            # We can flag to only do this if the constitutive model has been updated
            DFDt.psi_fn = self._constitutive_model.flux.T

            F1 = uw.function.expression(
                r"\mathbf{F}_1\left( \mathbf{u} \right)",
                DFDt.adams_moulton_flux()
                - sympy.eye(self.mesh.dim) * (self.p.sym[0])
                + self.penalty * self.div_u * sympy.eye(dim),
                "NStokes pointwise flux term: F_1(u)",
            )
        # Is the else condition useful - other than to prevent a crash ?
        # Yes, because then it can just live on the Stokes solver ...
        else:
            F1 = uw.function.expression(
                r"\mathbf{F}_1\left( \mathbf{u} \right)",
                self._constitutive_model.flux.T
                - sympy.eye(self.mesh.dim) * (self.p.sym[0]),
                "NStokes pressure gradient term: F_1(u) - No Flux history provided",
            )

        self._u_f1 = F1

        return F1

    @property
    def PF0(self):

        dim = self.mesh.dim

        f0 = uw.function.expression(
            r"\mathbf{F}_1\left( \mathbf{p} \right)",
            sympy.simplify(sympy.Matrix((self.constraints))),
            "NStokes pointwise flux term: f_0(p)",
        )

        self._p_f0 = f0

        return f0

    ## Deprecate this function
    def navier_stokes_problem_description(self):
        # f0 residual term
        self._u_f0 = self.F0.sym

        # f1 residual term
        self._u_f1 = self.F1.sym

        # p1 residual term
        self._p_f0 = self.PF0.sym

        return

    @property
    def delta_t(self):
        return self._delta_t

    @delta_t.setter
    def delta_t(self, value):
        self.is_setup = False
        self._delta_t.sym = value

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, value):
        self.is_setup = False
        self._rho.sym = value

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
        self._penalty.sym = value

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

    @timing.routine_timer_decorator
    def estimate_dt(self):
        r"""
        Calculates an appropriate timestep for the given
        mesh and viscosity configuration. This is an implicit solver
        so the $\delta_t$ should be interpreted as:

            - ${\delta t}_\textrm{diff}: a typical time for the diffusion of vorticity across an element
            - ${\delta t}_\textrm{adv}: a typical element-crossing time for a fluid parcel

        returns: ${\delta t}_\textrm{diff}, ${\delta t}_\textrm{adv}


        """

        # cf advection-diffusion. Here the diffusivity is represented by viscosity
        if isinstance(self.constitutive_model.viscosity, sympy.Expr):
            if uw.function.fn_is_constant_expr(self.constitutive_model.viscosity):
                max_diffusivity = uw.function.evaluate(
                    self.constitutive_model.viscosity
                )
            else:
                k = uw.function.evaluate(
                    sympy.sympify(self.constitutive_model.viscosity),
                    self.mesh._centroids,
                    self.mesh.N,
                )

                max_diffusivity = k.max()
        else:
            k = self.constitutive_model.viscosity / self.rho
            max_diffusivity = k

        ### required modules
        from mpi4py import MPI

        ## get global max dif value
        comm = uw.mpi.comm
        diffusivity_glob = comm.allreduce(max_diffusivity, op=MPI.MAX)

        ### get the velocity values
        vel = uw.function.evaluate(
            self.u.sym,
            self.mesh._centroids,
            self.mesh.N,
        )

        v_degree = self.u.degree

        ### get global velocity from velocity field
        max_magvel = np.linalg.norm(vel, axis=1).max()
        max_magvel_glob = comm.allreduce(max_magvel, op=MPI.MAX)

        ## get radius
        min_dx = self.mesh.get_min_radius()

        ## estimate dt of adv and diff components

        if max_magvel_glob == 0.0:
            dt_diff = ((min_dx / v_degree) ** 2) / diffusivity_glob
            dt_estimate = dt_diff
        elif diffusivity_glob == 0.0:
            dt_adv = min_dx / max_magvel_glob
            dt_estimate = dt_adv
        else:
            dt_diff = ((min_dx / v_degree) ** 2) / diffusivity_glob
            dt_adv = min_dx / max_magvel_glob
            dt_estimate = min(dt_diff, dt_adv)

        return dt_diff, dt_adv
