import sympy
from sympy import sympify
import numpy as np

from typing import Optional, Callable, Union

import underworld3 as uw
from underworld3.systems import SNES_Scalar, SNES_Vector, SNES_Stokes_SaddlePt
from underworld3 import VarType
import underworld3.timing as timing
from underworld3.utilities._api_tools import (
    uw_object,
    SymbolicProperty,
    Parameter,
    Template,
    ExpressionProperty,
)

from underworld3.function import expression as public_expression

expression = lambda *x, **X: public_expression(*x, _unique_name_generation=True, **X)


def _apply_unit_aware_scaling(dt_nondimensional, field, mesh):
    """
    Helper function to apply unit-aware scaling to timestep estimates.

    Detects the units of the velocity field and applies appropriate time scaling
    to convert nondimensional timestep to physical time units.

    Parameters
    ----------
    dt_nondimensional : float or np.ndarray
        The nondimensional timestep estimate
    field : MeshVariable or SymPy expression (often a Matrix)
        The velocity field - units are detected from this
    mesh : Mesh
        The mesh (may have reference to model with time scales)

    Returns
    -------
    float or UWQuantity
        Timestep with physical time units if detectable, otherwise nondimensional
    """
    try:
        from ..function.quantities import UWQuantity
        from ..units import get_units
        import sympy

        # Extract a component from field if it's a Matrix (common for velocity)
        field_to_check = field
        if isinstance(field, sympy.MatrixBase):
            # Extract first component: V_fn[0] or V_fn[0,0]
            if field.shape[0] > 0:
                field_to_check = field[0] if len(field.shape) == 1 else field[0, 0]

        # Try to get units from the field expression
        field_units = get_units(field_to_check)

        if field_units is not None:
            # Field has units - verify it has time dimension (as expected for velocity)
            # Get dimensionality: e.g., {'[length]': 1, '[time]': -1} for velocity
            field_dimensionality = field_units.dimensionality

            # Check if this has time dimension (velocity should have time^-1)
            if '[time]' in field_dimensionality:
                # Velocity field has time dimension - use model time scale for result
                # Don't try to match the velocity's specific time units (fragile string parsing)
                # Instead, always return in model's fundamental time scale and let user convert
                model = uw.get_default_model()
                if model and hasattr(model, "fundamental_scales"):
                    model_time_scale = model.fundamental_scales.get("time")
                    if model_time_scale is not None:
                        # Apply scaling: dt_physical = dt_nd * time_scale
                        dt_physical = dt_nondimensional * model_time_scale

                        # Return as UWQuantity
                        if not isinstance(dt_physical, UWQuantity):
                            dt_physical = UWQuantity._from_pint(dt_physical)
                        return dt_physical

        # Fallback: check if model has time scales (old behavior)
        model = uw.get_default_model()
        if model and hasattr(model, "fundamental_scales") and model.fundamental_scales:
            time_scale = model.fundamental_scales.get("time")
            if time_scale is not None:
                dt_physical = dt_nondimensional * time_scale
                if not isinstance(dt_physical, UWQuantity):
                    dt_physical = UWQuantity._from_pint(dt_physical)
                return dt_physical

    except Exception as e:
        # Silently fall back to nondimensional
        pass

    return dt_nondimensional


from .ddt import SemiLagrangian as SemiLagrangian_DDt
from .ddt import Lagrangian as Lagrangian_DDt
from .ddt import Eulerian as Eulerian_DDt
from .ddt import Symbolic as Symbolic_DDt


class SNES_Poisson(SNES_Scalar):
    r"""
    # Poisson Equation Solver

    This class provides functionality for a discrete representation
    of the Poisson equation

    $$
    \nabla \cdot
            \color{Blue}{\underbrace{\Bigl[ \boldsymbol\kappa \nabla u \Bigr]}_{\mathbf{F}}} =
            \color{Maroon}{\underbrace{\Bigl[ f \Bigl] }_{\mathbf{h}}}
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
            verbose,
            DuDt=DuDt,
            DFDt=DFDt,
        )

        # default values for properties
        self.f = sympy.Matrix.zeros(1, 1)

        self._constitutive_model = None

    # Use Template for persistent read-only template expressions
    F0 = Template(
        r"f_0 \left( \mathbf{u} \right)",
        lambda self: -self.f,
        "Poisson pointwise force term: f_0(u)",
    )

    F1 = Template(
        r"\mathbf{F}_1\left( \mathbf{u} \right)",
        lambda self: sympy.simplify(self.constitutive_model.flux.T),
        "Poisson pointwise flux term: F_1(u)",
    )

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

        # Handle UWQuantity with units - enforce "units everywhere" principle
        if hasattr(value, "value") and hasattr(value, "units"):
            # Extract the plain value
            plain_value = float(value.value)

            # If ND scaling is active, scale the constant
            import underworld3 as uw

            if uw.is_nondimensional_scaling_active():
                # The source term should have same dimensionality as the unknown field
                # Access via self.Unknowns.u (Poisson) or self.Unknowns.DuDt.u (Stokes)
                u = None
                if hasattr(self, "Unknowns") and hasattr(self.Unknowns, "u"):
                    u = self.Unknowns.u

                if u is not None and hasattr(u, "scaling_coefficient"):
                    scale = u.scaling_coefficient
                    if scale != 1.0 and scale != 0.0:
                        plain_value = plain_value / scale

            self._f = sympy.Matrix((plain_value,))

        # Also accept plain SymPy expressions (for variables, etc.)
        elif hasattr(value, "atoms"):  # It's a SymPy expression
            self._f = sympy.Matrix((value,))

        # Handle plain numbers - apply "units everywhere or nowhere" principle
        # Exception: allow zero always (dimensionless/no source)
        elif isinstance(value, (int, float)):
            if value == 0.0 or value == 0:
                # Zero is allowed - no source term
                self._f = sympy.Matrix((0.0,))
            else:
                # Check if model has reference quantities defined
                # If yes: enforce units everywhere
                # If no: allow plain numbers (user is responsible for consistency)
                import underworld3 as uw

                model = uw.get_default_model()

                if model.has_units():
                    # Reference quantities defined - enforce units everywhere
                    raise ValueError(
                        f"Units requirement enforced: Model has reference quantities defined.\n"
                        f"Source term 'f' requires units. Use:\n"
                        f"  solver.f = uw.quantity({value}, 'appropriate_units')\n"
                        f"Example: poisson.f = uw.quantity({value}, 'kelvin')"
                    )
                else:
                    # No reference quantities - allow plain numbers
                    # User is responsible for dimensional consistency
                    self._f = sympy.Matrix((float(value),))

        else:
            # Accept other types (might be symbolic)
            self._f = sympy.Matrix((value,))

    @property
    def CM_is_setup(self):
        return self._constitutive_model._solver_is_setup


class SNES_Darcy(SNES_Scalar):
    r"""
    # Darcy Flow Equation Solver

    This class provides functionality for a discrete representation
    of the Groundwater flow equations

    $$
    \color{Green}{\underbrace{ \Bigl[  S_s \frac{\partial h}{\partial t} \Bigr]}_{\dot{\mathbf{u}}}} -
    \nabla \cdot
            \color{Blue}{\underbrace{\Bigl[ \boldsymbol\kappa \nabla h  - \boldsymbol{s}\Bigr]}_{\mathbf{F}}} =
            \color{Maroon}{\underbrace{\Bigl[ W \Bigl] }_{\mathbf{h}}}
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
        verbose=False,
        DuDt=None,
        DFDt=None,
    ):
        ## Parent class will set up default values etc
        super().__init__(
            mesh,
            h_Field,
            degree,
            verbose,
            DuDt=DuDt,
            DFDt=DFDt,
        )

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

        # If we add smoothing, it should be small
        self._v_projector.smoothing = 1.0e-6

    ## This function is the one we will typically over-ride to build specific solvers.
    ## This example is a poisson-like problem with isotropic coefficients

    # Use Template for persistent read-only template expressions
    F0 = Template(
        r"f_0 \left( \mathbf{u} \right)", lambda self: -self.f, "Darcy pointwise force term: f_0(u)"
    )

    F1 = Template(
        r"\mathbf{F}_1\left( \mathbf{u} \right)",
        lambda self: self.darcy_flux,
        "Darcy pointwise flux term: F_1(u)",
    )

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

        # self._v_projector.petsc_options["snes_rtol"] = 1.0e-6
        # self._v_projector.petsc_options.delValue("ksp_monitor")
        self._v_projector.uw_function = self.darcy_flux
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
    Stokes equation solver for incompressible viscous flow.

    This class provides functionality for a discrete representation
    of the Stokes flow equations assuming an incompressibility
    (or near-incompressibility) constraint.

    The momentum equation is:

    .. math::

        -\nabla \cdot \left[ \boldsymbol{\tau} - p \mathbf{I} \right] = \mathbf{f}

    with the incompressibility constraint:

    .. math::

        \nabla \cdot \mathbf{u} = 0

    The flux term is a deviatoric stress (:math:`\boldsymbol{\tau}`) related to velocity
    gradients (:math:`\nabla \mathbf{u}`) through a viscosity tensor :math:`\eta`, and a
    volumetric (pressure) part :math:`p`:

    .. math::

        \boldsymbol{\tau} = \frac{\eta}{2}\left( \nabla \mathbf{u} + \nabla \mathbf{u}^T \right)

    The constraint equation gives incompressible flow by default but can be set
    to any function of the unknown :math:`\mathbf{u}` and :math:`\nabla \cdot \mathbf{u}`.

    Parameters
    ----------
    mesh : uw.discretisation.Mesh
        The computational mesh.
    velocityField : uw.discretisation.MeshVariable, optional
        Pre-existing velocity field. If None, one is created automatically.
    pressureField : uw.discretisation.MeshVariable, optional
        Pre-existing pressure field. If None, one is created automatically.
    degree : int, optional
        Polynomial degree for velocity interpolation. Default is 2.
    p_continuous : bool, optional
        If True (default), pressure is continuous. Set False for discontinuous pressure.
    verbose : bool, optional
        Enable verbose output during solving. Default is False.
    DuDt : SemiLagrangian_DDt or Lagrangian_DDt, optional
        Material derivative operator for velocity (used in derived classes).
    DFDt : SemiLagrangian_DDt or Lagrangian_DDt, optional
        Material derivative operator for flux (used in viscoelastic models).

    Attributes
    ----------
    u : MeshVariable
        The velocity field (accessed via ``solver.Unknowns.u``).
    p : MeshVariable
        The pressure field (accessed via ``solver.Unknowns.p``).
    bodyforce : UWexpression
        Volumetric body force vector :math:`\mathbf{f}`.
    constitutive_model : ConstitutiveModel
        Viscosity model providing the stress-strain relationship.
    penalty : UWexpression
        Augmented Lagrangian penalty parameter :math:`\lambda`.
    saddle_preconditioner : sympy.Expr
        Preconditioner for the saddle point system (default: :math:`1/\eta`).
    constraints : sympy.Matrix
        Constraint equation(s), default is :math:`\nabla \cdot \mathbf{u}`.

    Notes
    -----
    **Viscosity model**: The viscosity tensor :math:`\boldsymbol{\eta}` is provided by
    setting the ``constitutive_model`` property to one of the ``uw.constitutive_models``
    classes. It may be constant, spatially varying, non-linear, or anisotropic.

    **Augmented Lagrangian**: Setting ``penalty`` to a non-zero value adds
    :math:`\lambda \nabla \cdot \mathbf{u}` to the weak form, improving convergence
    for incompressible flow (in addition to the constraint equation).

    **Mixed finite elements**: The pressure field interpolation order determines
    the integration order of the mixed method and is typically lower than the
    velocity field order.

    **Viscoelastic models**: For viscoelastic behaviour, the flux term contains
    stress history tracked on a particle swarm. See :class:`SNES_VE_Stokes`.

    See Also
    --------
    SNES_VE_Stokes : Viscoelastic Stokes solver with flux history.
    SNES_NavierStokes : Navier-Stokes solver with inertial terms.
    uw.constitutive_models : Available viscosity models.

    Examples
    --------
    >>> import underworld3 as uw
    >>> mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0,0), maxCoords=(1,1), cellSize=0.1)
    >>> stokes = uw.systems.Stokes(mesh, degree=2)
    >>> stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel()
    >>> stokes.constitutive_model.Parameters.viscosity = 1.0
    >>> stokes.bodyforce = [0, -1]  # gravity
    >>> stokes.solve()
    """

    instances = 0

    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        velocityField: Optional[uw.discretisation.MeshVariable] = None,
        pressureField: Optional[uw.discretisation.MeshVariable] = None,
        degree: Optional[int] = 2,
        p_continuous: Optional[bool] = True,
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
            verbose,
            DuDt=DuDt,
            DFDt=DFDt,
        )

        self._degree = degree
        # User-facing operations are matrices / vectors by preference

        self._Estar = None

        self._penalty = expression(R"\uplambda", 0, "Numerical Penalty")
        self._constraints = sympy.Matrix((self.div_u,))  # by default, incompressibility constraint

        self._bodyforce = expression(
            Rf"\mathbf{{f}}_0\left( {self.Unknowns.u.symbol} \right)",
            sympy.Matrix([[0] * self.mesh.dim]),
            "Stokes pointwise force term: f_0(u)",
        )

        # this attrib records if we need to setup the problem (again)
        self.is_setup = False

        self._constitutive_model = None

        return

    ## Problem Description:
    ##  F0 - velocity equation forcing terms
    ##  F1 - velocity equation flux terms
    ##  PF0 - pressure / constraint equation forcing terms

    # Use Template for persistent read-only template expressions
    F0 = Template(
        r"\mathbf{f}_0\left( \mathbf{u} \right)",
        lambda self: -self.bodyforce.sym,
        "Stokes pointwise force term: f_0(u)",
    )

    F1 = Template(
        r"\mathbf{F}_1\left( \mathbf{u} \right)",
        lambda self: sympy.simplify(
            self.stress + self.penalty * self.div_u * sympy.eye(self.mesh.dim)
        ),
        "Stokes pointwise flux term: F_1(u)",
    )

    PF0 = Template(
        r"\mathbf{h}_0\left( \mathbf{p} \right)",
        lambda self: sympy.simplify(sympy.Matrix((self.constraints))),
        "Pointwise force term: h_0(p)",
    )

    # deprecated
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
        if isinstance(value, uw.function.expressions.UWexpression):
            self._bodyforce.sym = value.sym
        else:
            # Convert UWQuantity objects to SymPy expressions before Matrix creation
            converted_value = []
            for item in value:
                if isinstance(item, uw.function.quantities.UWQuantity):
                    sympified = item._sympify_()
                    # If UWQuantity contains a Matrix, extract the scalar element
                    if hasattr(sympified, "shape") and sympified.shape == (1, 1):
                        converted_value.append(sympified[0, 0])
                    else:
                        converted_value.append(sympified)
                else:
                    converted_value.append(item)
            self._bodyforce.sym = sympy.Matrix(converted_value)

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

    # @property
    # def continuity_rhs(self):
    #     return self._continuity_rhs

    # @continuity_rhs.setter
    # def continuity_rhs(self, value):
    #     self.is_setup = False
    #     self._continuity_rhs.sym = value

    @timing.routine_timer_decorator
    def estimate_dt(self):
        r"""
        Calculates an appropriate advective timestep for the Stokes solver.

        The Stokes equations are quasi-static (no time derivative ∂v/∂t),
        so there is no diffusive CFL constraint. The only relevant timescale
        is the advective one: how long it takes material to cross an element.

        This method computes a per-element timestep:
            dt_i = h_i / |v_i|

        where h_i is the element radius and v_i is the velocity at the element
        centroid, then returns the global minimum. This is more accurate than
        using global max velocity with global min element size, especially for
        non-uniform meshes with spatially varying velocity.

        Returns:
            Pint Quantity or float: The advective timestep with physical time units
            if a model with reference scales is available, otherwise nondimensional.
        """

        from mpi4py import MPI

        # Evaluate velocity at element centroids (consistent with AdvDiff)
        vel = uw.function.evaluate(self.u.sym, self.mesh._centroids)

        # If vel is unit-aware (UnitAwareArray), nondimensionalise it to get
        # consistent nondimensional values that match mesh._radii
        # Note: .magnitude returns physical units, which would be wrong here
        if hasattr(vel, "units") and vel.units is not None:
            vel = uw.non_dimensionalise(vel)
        elif hasattr(vel, "magnitude"):
            # Plain UWQuantity without units context - use magnitude
            vel = vel.magnitude

        # Ensure vel is a plain numpy array
        vel = np.asarray(vel)

        # Squeeze out any singleton dimensions from evaluate (shape: N,1,dim -> N,dim)
        vel = np.squeeze(vel)

        # Handle edge case of single element (ensure 2D)
        if vel.ndim == 1:
            vel = vel.reshape(1, -1)

        # Get per-element velocity magnitudes
        vel_magnitudes = np.linalg.norm(vel, axis=1)

        # Get per-element radii (characteristic element size)
        element_radii = self.mesh._radii

        # Compute per-element advective timestep: dt_i = h_i / |v_i|
        # Avoid division by zero for elements with zero velocity
        with np.errstate(divide='ignore', invalid='ignore'):
            dt_per_element = np.where(
                vel_magnitudes > 0,
                element_radii / vel_magnitudes,
                np.inf
            )

        # Get local minimum timestep
        min_dt_local = np.min(dt_per_element) if len(dt_per_element) > 0 else np.inf

        # Get global minimum timestep (parallel-safe)
        comm = uw.mpi.comm
        min_dt_glob = comm.allreduce(min_dt_local, op=MPI.MIN)

        # If all velocities are zero, return infinity
        if np.isinf(min_dt_glob):
            return np.inf

        # Dimensionalise to physical time
        try:
            return uw.dimensionalise(np.squeeze(min_dt_glob), {'[time]': 1})
        except Exception:
            return np.squeeze(min_dt_glob)


class SNES_VE_Stokes(SNES_Stokes):
    r"""
    # ViscoElastic Stokes Equation Solver

    This class provides functionality for a discrete representation
    of the Stokes flow equations assuming an incompressibility
    (or near-incompressibility) constraint and with a flux history
    term included to allow for viscoelastic modelling.

    All other functionality is inherited from SNES_Stokes

    $$
    -\nabla \cdot
            \color{Blue}{\underbrace{\Bigl[
                    \boldsymbol{\tau} -  p \mathbf{I} \Bigr]}_{\mathbf{F}}} =
            \color{Maroon}{\underbrace{\Bigl[ \mathbf{f} \Bigl] }_{\mathbf{h}}}
    $$

    $$
    \underbrace{\Bigl[ \nabla \cdot \mathbf{u} \Bigr]}_{\mathbf{h}_p} = 0
    $$

    The flux term is a deviatoric stress ( $\boldsymbol{\tau}$ ) related to velocity gradients
      ( $\nabla \mathbf{u}$ ) through a viscosity tensor, $\eta$, and a volumetric (pressure) part $p$

    $$
        \mathbf{F}: \quad \boldsymbol{\tau} = \frac{\eta}{2}\left( \nabla \mathbf{u} + \nabla \mathbf{u}^T \right)
    $$

    The constraint equation, $\mathbf{h}_p = 0$ is incompressible flow by default but can be set
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
    through the `saddle_preconditioner` property. The default choice is $1/\eta$ for a scalar viscosity function.

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
            verbose,
            DuDt=DuDt,
            DFDt=DFDt,
        )

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
                smoothing=0.0001,
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
    # Projection Solver

    Solves $u = \tilde{f}$ where $\tilde{f}$ is a function that can be evaluated within an element and
    $u$ is a `meshVariable` with associated shape functions. Typically, the projection is used to obtain a
    continuous representation of a function that is not well defined at the mesh nodes. For example, functions of
    the spatial derivatives of one or more `meshVariable` (e.g. components of fluxes) can be mapped to continuous
    variables with a projection. More broadly it is a projection from one basis to another and its limitations should be
    evaluated within that context.

    The projection implemented by creating a solver for this problem

    $$
    -\nabla \cdot
            \color{Blue}{\underbrace{\Bigl[ \boldsymbol\alpha \nabla u \Bigr]}_{\mathbf{F}}} -
            \color{Maroon}{\underbrace{\Bigl[ u - \tilde{f} \Bigl] }_{\mathbf{h}}} = 0
    $$

    Where the term $\mathbf{F}$ provides a smoothing regularization. $\alpha$ can be zero.
    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        u_Field: uw.discretisation.MeshVariable = None,
        degree=2,
        verbose=False,
    ):

        super().__init__(
            mesh,
            u_Field,
            degree,
            verbose,
        )

        self.is_setup = False
        self._smoothing = sympy.sympify(0)
        self._uw_weighting_function = sympy.sympify(1)
        self._uw_function = sympy.Matrix([0])  # Default: project zero
        self._constitutive_model = uw.constitutive_models.Constitutive_Model(self.Unknowns)

        return

    # Use Template for persistent read-only template expressions
    F0 = Template(
        r"f_0 \left( \mathbf{u} \right)",
        lambda self: (self.u.sym - self.uw_function) * self.uw_weighting_function,
        "Scalar Projection pointwise misfit term: f_0(u)",
    )

    F1 = Template(
        r"\mathbf{F}_1\left( \mathbf{u} \right)",
        lambda self: self.smoothing * self.mesh.vector.gradient(self.u.sym),
        "Scalar projection pointwise smoothing term: F_1(u)",
    )

    # Use SymbolicProperty for automatic unwrapping
    uw_function = SymbolicProperty(matrix_wrap=True, doc="Function to project onto mesh")

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
    # Projection Solver (Vector Variable)

    Solves $\mathbf{u} = \tilde{\mathbf{f}}$ where $\tilde{\mathbf{f}}$ is a vector function that can be evaluated within an element and
    $\mathbf{u}$ is a vector `meshVariable` with associated shape functions. Typically, the projection is used to obtain a
    continuous representation of a function that is not well defined at the mesh nodes. For example, functions of
    the spatial derivatives of one or more `meshVariable` (e.g. components of fluxes) can be mapped to continuous
    variables with a projection. More broadly it is a projection from one basis to another and its limitations should be
    evaluated within that context.

    The projection is implemented by creating a solver for this problem

    $$
    -\nabla \cdot
            \color{Blue}{\underbrace{\Bigl[ \boldsymbol\alpha \nabla \mathbf{u} \Bigr]}_{\mathbf{F}}} -
            \color{Maroon}{\underbrace{\Bigl[ \mathbf{u} - \tilde{\mathbf{f}} \Bigl] }_{\mathbf{h}}} = 0
    $$

    Where the term $\mathbf{F}$ provides a smoothing regularization. $\alpha$ can be zero.
    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        u_Field: uw.discretisation.MeshVariable = None,
        degree=2,
        verbose=False,
    ):
        super().__init__(
            mesh,
            u_Field,
            degree,
            verbose,
        )

        self.is_setup = False
        self._smoothing = 0.0
        self._penalty = 0.0
        self._uw_weighting_function = 1.0
        self._uw_function = sympy.Matrix([[0] * self.mesh.dim])  # Default: project zero vector
        self._constitutive_model = uw.constitutive_models.Constitutive_Model(self.Unknowns)

        return

    # Use Template for persistent read-only template expressions
    F0 = Template(
        r"f_0 \left( \mathbf{u} \right)",
        lambda self: (self.u.sym - self.uw_function) * self.uw_weighting_function,
        "Vector projection pointwise misfit term: f_0(u)",
    )

    F1 = Template(
        r"\mathbf{F}_1\left( \mathbf{u} \right)",
        lambda self: (
            self.smoothing * self.Unknowns.E
            + self.penalty * self.mesh.vector.divergence(self.u.sym) * sympy.eye(self.mesh.dim)
        ),
        "Vector projection pointwise smoothing term: F_1(u)",
    )

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
            + self.penalty * self.mesh.vector.divergence(self.u.sym) * sympy.eye(self.mesh.dim)
        )

        return

    # Use SymbolicProperty for automatic unwrapping
    uw_function = SymbolicProperty(matrix_wrap=True, doc="Vector function to project onto mesh")

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
    # Projection Solver (Tensor Variable)

    Solves $\mathbf{u} = \tilde{\mathbf{f}}$ where $\tilde{\mathbf{f}}$ is a tensor-valued function that can be evaluated within an element and
    $\mathbf{u}$ is a tensor `meshVariable` with associated shape functions. Typically, the projection is used to obtain a
    continuous representation of a function that is not well defined at the mesh nodes. For example, functions of
    the spatial derivatives of one or more `meshVariable` (e.g. components of fluxes) can be mapped to continuous
    variables with a projection. More broadly it is a projection from one basis to another and its limitations should be
    evaluated within that context.

    The projection implemented by creating a solver for this problem

    $$
    -\nabla \cdot
            \color{Blue}{\underbrace{\Bigl[ \boldsymbol\alpha \nabla \mathbf{u} \Bigr]}_{\mathbf{F}}} -
            \color{Maroon}{\underbrace{\Bigl[ \mathbf{u} - \tilde{\mathbf{f}} \Bigl] }_{\mathbf{h}}} = 0
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
        verbose=False,
    ):
        self.t_field = tensor_Field

        super().__init__(
            mesh=mesh,
            u_Field=scalar_Field,
            degree=degree,
            verbose=verbose,
        )

        return

    ## Need to over-ride solve method to run over all components

    @timing.routine_timer_decorator
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

                self.u.array[:, 0, 0] = self.t_field.array[:, i, j]

                # solve the projection for the scalar sub-problem
                super().solve(verbose=verbose)

                self.t_field.array[:, i, j] = self.u.array[:, 0, 0]

        # That might be all ...

    # This is re-defined so it uses uw_scalar_function

    @property
    def F0(self):

        f0_val = expression(
            r"f_0 \left( \mathbf{u} \right)",
            (self.u.sym - self.uw_scalar_function) * self.uw_weighting_function,
            "Scalar subproblem of tensor projection: f_0(u)",
        )

        # backward compatibility
        self._f0 = f0_val

        return f0_val

    @property
    def F1(self):

        F1_val = expression(
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
    # Advection-Diffusion Equation Solver (Scalar Variable)

    This class provides a solver for the scalar Advection-Diffusion equation using the characteristics based Semi-Lagrange Crank-Nicholson method
    which is described in Spiegelman & Katz, (2006).

    $$
    \color{Green}{\underbrace{ \Bigl[ \frac{\partial u}{\partial t} + \left( \mathbf{v} \cdot \nabla \right) u \Bigr]}_{\dot{\mathbf{u}}}} -
    \nabla \cdot
            \color{Blue}{\underbrace{\Bigl[ \boldsymbol\kappa \nabla u \Bigr]}_{\mathbf{F}}} =
            \color{Maroon}{\underbrace{\Bigl[ f \Bigl] }_{\mathbf{h}}}
    $$

    The term $\mathbf{F}$ relates diffusive fluxes to gradients in the unknown $u$. The advective flux that results from having gradients along
    the direction of transport (given by the velocity vector field $\mathbf{v}$ ) are included in the $\dot{\mathbf{u}}$ term.

    The term $\dot{\mathbf{u}}$ involves upstream sampling to find the value $u^*$ which represents the value of $u$ at
    the points which later arrive at the nodal points of the mesh. This is achieved using a "hidden"
    swarm variable which is advected backwards from the nodal points automatically during the `solve` phase.

    ## Properties

      - The unknown is $u$.

      - The velocity field is $\mathbf{v}$ and is provided as a `sympy` function to allow operations such as time-averaging to be
        calculated in situ (e.g. `V_Field = v_solution.sym`) **NOTE: no it's not.

      - The diffusivity tensor, $\kappa$ is provided by setting the `constitutive_model` property to
        one of the scalar `uw.constitutive_models` classes and populating the parameters.
        It is usually a constant or a function of position / time and may also be non-linear
        or anisotropic.

      - Volumetric sources of $u$ are specified using the $f$ property and can be any valid combination of `sympy` functions of position and
        `meshVariable` or `swarmVariable` types.

    ## References

    Spiegelman, M., & Katz, R. F. (2006). A semi-Lagrangian Crank-Nicolson algorithm for the numerical solution of advection-diffusion problems. Geochemistry, Geophysics, Geosystems, 7(4). https://doi.org/10.1029/2005GC001073

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
        V_fn: Union[uw.discretisation.MeshVariable, sympy.Basic],  # Should be a sympy function
        order: int = 1,
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
            verbose,
            DuDt=DuDt,
            DFDt=DFDt,
        )

        if isinstance(V_fn, uw.discretisation.MeshVariable):
            self._V_fn = V_fn.sym
        else:
            self._V_fn = V_fn

        # default values for properties
        self.f = sympy.Matrix.zeros(1, 1)

        self._constitutive_model = None

        # These are unique to the advection solver
        self._delta_t = expression(R"\Delta t", 0, "Physically motivated timestep")
        self.is_setup = False

        self.restore_points_to_domain_func = restore_points_func
        ### Setup the history terms ... This version should not build anything
        ### by default - it's the template / skeleton

        ## NB - Smoothing is generally required for stability. 0.0001 is effective
        ## at the various resolutions tested.

        if DuDt is None:
            self.Unknowns.DuDt = SemiLagrangian_DDt(
                self.mesh,
                u_Field.sym,  # Symbolic expression - SemiLagrangian evaluates this at each update
                self._V_fn,
                vtype=uw.VarType.SCALAR,
                degree=u_Field.degree,
                continuous=u_Field.continuous,
                varsymbol=u_Field.symbol,
                verbose=verbose,
                bcs=self.essential_bcs,
                order=1,
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
            sympy.Matrix([[0] * self.mesh.dim]),  # Actual function is not defined at this point
            self._V_fn,
            vtype=uw.VarType.VECTOR,
            degree=u_Field.degree,
            continuous=True,
            # The default is now to match the above and avoid
            # any use of projection.
            # swarm_degree=u_Field.degree - 1,
            # swarm_continuous=False,
            varsymbol=rf"{{F[ {self.u.symbol} ] }}",
            verbose=verbose,
            bcs=None,
            order=order,
            smoothing=0.0,
        )

        return

    @property
    def F0(self):

        f0 = expression(
            r"f_0 \left( \mathbf{u} \right)",
            -self.f + self.DuDt.bdf(0) / self.delta_t,
            "Poisson pointwise force term: f_0(u)",
        )

        # backward compatibility
        self._f0 = f0

        return f0

    @property
    def F1(self):

        F1_val = expression(
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

    @V_fn.setter
    def V_fn(self, value):
        """
        Set the velocity function for advection.

        Parameters:
        -----------
        value : uw.discretisation.MeshVariable or sympy.Basic
            Velocity field as either a MeshVariable or sympy expression
        """
        self.is_setup = False  # Mark as needing setup when velocity changes
        if isinstance(value, uw.discretisation.MeshVariable):
            self._V_fn = value.sym
        else:
            self._V_fn = value

    @property
    def delta_t(self):
        return self._delta_t

    @delta_t.setter
    def delta_t(self, value):
        self.is_setup = False

        # Handle Pint Quantities with time dimensions
        if hasattr(value, "dimensionality"):
            # This is a Pint Quantity - check if it has time dimensions
            try:
                from ..scaling import units as ureg

                time_dim = ureg.second.dimensionality
                if value.dimensionality == time_dim:
                    # Convert physical time to nondimensional using model time scale
                    model = uw.get_default_model()
                    if model and hasattr(model, "fundamental_scales") and model.fundamental_scales:
                        time_scale = model.fundamental_scales.get("time")
                        if time_scale is not None:
                            # Physical time / time scale = nondimensional time
                            # Must use to_reduced_units() to convert both quantities
                            # to same base units before extracting magnitude.
                            # Otherwise Pint keeps different unit bases (megayear/second)
                            # and .magnitude returns the unconverted number!
                            result = value / time_scale

                            # Get the internal Pint quantity for proper unit conversion
                            if hasattr(result, "_pint_qty"):
                                # UWQuantity - access internal Pint quantity
                                pint_result = result._pint_qty.to_reduced_units()
                            elif hasattr(result, "to_reduced_units"):
                                # Raw Pint Quantity
                                pint_result = result.to_reduced_units()
                            else:
                                pint_result = result

                            # Extract the dimensionless magnitude
                            if hasattr(pint_result, "magnitude"):
                                value = float(pint_result.magnitude)
                            else:
                                value = float(pint_result)
            except Exception:
                pass  # If anything fails, try to use value as-is

        self._delta_t.sym = value

    @timing.routine_timer_decorator
    def estimate_dt(self):
        r"""
        Calculates an appropriate timestep for the advection-diffusion solver.

        This is an implicit solver so the $\delta_t$ returned is the minimum of:

            - ${\delta t}_\textrm{diff}$: typical time for the diffusion front to propagate across an element
            - ${\delta t}_\textrm{adv}$: typical element-crossing time for a fluid parcel

        Returns:
            Pint Quantity or float: The recommended timestep with physical time units
            if a model with reference scales is available, otherwise nondimensional.
        """

        ### required modules
        from mpi4py import MPI

        # Use the unified .K property from the constitutive model
        # This provides diffusivity for diffusion models
        K = self.constitutive_model.K

        # Evaluate the diffusivity (handles constant and spatially-varying cases)
        if isinstance(K, sympy.Expr) or hasattr(K, 'sym'):
            K_sym = K.sym if hasattr(K, 'sym') else K
            if uw.function.fn_is_constant_expr(K_sym):
                diffusivity = uw.function.evaluate(
                    K_sym,
                    np.zeros((1, self.mesh.dim)),
                )
            else:
                diffusivity = uw.function.evaluate(
                    sympy.sympify(K_sym),
                    self.mesh._centroids,
                    self.mesh.N,
                )
                diffusivity = diffusivity.max()
        else:
            diffusivity = K

        # If diffusivity is unit-aware (UnitAwareArray), nondimensionalise it to get
        # consistent nondimensional values that match mesh._radii
        # Note: .magnitude returns physical units, which would be wrong here
        if hasattr(diffusivity, "units") and diffusivity.units is not None:
            diffusivity = uw.non_dimensionalise(diffusivity)
        elif hasattr(diffusivity, "magnitude"):
            # Plain UWQuantity without units context - use magnitude
            diffusivity = diffusivity.magnitude

        # Ensure diffusivity is a plain numpy scalar
        max_diffusivity = float(np.asarray(diffusivity).max())

        ## get global max diffusivity value
        comm = uw.mpi.comm
        diffusivity_glob = comm.allreduce(max_diffusivity, op=MPI.MAX)

        ### get the velocity values at element centroids (nondimensional)
        vel = uw.function.evaluate(
            self.V_fn,
            self.mesh._centroids,
        )

        # If vel is unit-aware (UnitAwareArray), nondimensionalise it to get
        # consistent nondimensional values that match mesh._radii
        # Note: .magnitude returns physical units, which would be wrong here
        if hasattr(vel, "units") and vel.units is not None:
            vel = uw.non_dimensionalise(vel)
        elif hasattr(vel, "magnitude"):
            # Plain UWQuantity without units context - use magnitude
            vel = vel.magnitude

        # Ensure vel is a plain numpy array
        vel = np.asarray(vel)

        # Squeeze out any singleton dimensions from evaluate (shape: N,1,dim -> N,dim)
        vel = np.squeeze(vel)

        # Handle edge case of single element (ensure 2D)
        if vel.ndim == 1:
            vel = vel.reshape(1, -1)

        # Get per-element velocity magnitudes
        vel_magnitudes = np.linalg.norm(vel, axis=1)

        # Get per-element radii (characteristic element size)
        element_radii = self.mesh._radii

        ## estimate dt of adv and diff components using per-element approach
        ## dt_adv_i = h_i / |v_i| for advection
        ## dt_diff_i = h_i^2 / κ for diffusion (using global κ for now)

        # Per-element diffusive timestep (all elements use same diffusivity)
        if diffusivity_glob > 0:
            dt_diff_per_element = (element_radii ** 2) / diffusivity_glob
            min_dt_diff_local = np.min(dt_diff_per_element) if len(dt_diff_per_element) > 0 else np.inf
        else:
            min_dt_diff_local = np.inf

        # Per-element advective timestep
        with np.errstate(divide='ignore', invalid='ignore'):
            dt_adv_per_element = np.where(
                vel_magnitudes > 0,
                element_radii / vel_magnitudes,
                np.inf
            )
        min_dt_adv_local = np.min(dt_adv_per_element) if len(dt_adv_per_element) > 0 else np.inf

        # Get global minimum timesteps (parallel-safe)
        min_dt_diff_glob = comm.allreduce(min_dt_diff_local, op=MPI.MIN)
        min_dt_adv_glob = comm.allreduce(min_dt_adv_local, op=MPI.MIN)

        # Store for user inspection
        self.dt_adv = min_dt_adv_glob if not np.isinf(min_dt_adv_glob) else 0.0
        self.dt_diff = min_dt_diff_glob if not np.isinf(min_dt_diff_glob) else 0.0

        # Take overall minimum (respecting infinity for zero velocity/diffusivity cases)
        dt_estimate = min(min_dt_diff_glob, min_dt_adv_glob)

        # If both are infinite (no velocity and no diffusivity), return infinity
        if np.isinf(dt_estimate):
            return np.inf

        # Dimensionalise the result to physical time
        try:
            return uw.dimensionalise(np.squeeze(dt_estimate), {'[time]': 1})
        except Exception:
            # Fallback: return plain nondimensional number
            return np.squeeze(dt_estimate)

    @timing.routine_timer_decorator
    def solve(
        self,
        zero_init_guess: bool = True,
        timestep: float = None,
        _force_setup: bool = False,
        _evalf=False,
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

        self.DuDt.update_pre_solve(timestep, verbose=verbose, evalf=_evalf)
        self.DFDt.update_pre_solve(timestep, verbose=verbose, evalf=_evalf)

        super().solve(zero_init_guess, _force_setup)

        # Invalidate cached data views - PETSc may have replaced underlying buffers
        # This ensures .data and .array properties return fresh data from PETSc
        # Handle both EnhancedMeshVariable (has _base_var) and direct _MeshVariable
        target_var = getattr(self.u, "_base_var", self.u)
        if hasattr(target_var, "_canonical_data"):
            target_var._canonical_data = None

        self.DuDt.update_post_solve(timestep, verbose=verbose, evalf=_evalf)
        self.DFDt.update_post_solve(timestep, verbose=verbose, evalf=_evalf)

        self.is_setup = True
        self.constitutive_model._solver_is_setup = True

        return


class SNES_Diffusion(SNES_Scalar):
    r"""
    # Diffusion Equation Solver (Scalar Variable)

    This class provides a solver for the scalar Diffusion equation using mesh-based finite elements.

    $$
    \color{Green}{\underbrace{ \Bigl[ \frac{\partial u}{\partial t} - \left( \mathbf{v} \cdot \nabla \right) u \Bigr]}_{\dot{\mathbf{f}}}} -
    \nabla \cdot
            \color{Blue}{\underbrace{\Bigl[ \boldsymbol\kappa \nabla u \Bigr]}_{\mathbf{F}}} =
            \color{Maroon}{\underbrace{\Bigl[ f \Bigl] }_{\mathbf{f}}}
    $$

    The term $\mathbf{F}$ relates diffusive fluxes to gradients in the unknown $u$.

    ## Properties

      - The unknown is $u$.

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
        display(Latex(r"$\quad\Delta t = $ " + self.delta_t._repr_latex_()))

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        u_Field: uw.discretisation.MeshVariable,
        order: int = 1,
        theta: float = 0.0,
        evalf: Optional[bool] = False,
        verbose=False,
        DuDt: Union[Eulerian_DDt, SemiLagrangian_DDt, Lagrangian_DDt] = None,
        DFDt: Union[Eulerian_DDt, SemiLagrangian_DDt, Lagrangian_DDt] = None,
    ):
        ## Parent class will set up default values etc
        super().__init__(
            mesh,
            u_Field,
            u_Field.degree,
            verbose,
            DuDt=DuDt,
            DFDt=DFDt,
        )

        # default values for properties
        self.f = sympy.Matrix.zeros(1, 1)

        self._constitutive_model = None

        self.theta = theta

        # These are unique to the advection solver
        self._delta_t = expression(R"\Delta t", 0, "Physically motivated timestep")
        self.is_setup = False

        ### Setup the history terms ... This version should not build anything
        ### by default - it's the template / skeleton

        ## NB - Smoothing is generally required for stability. 0.0001 is effective
        ## at the various resolutions tested.

        if DuDt is None:
            self.Unknowns.DuDt = Eulerian_DDt(
                self.mesh,
                u_Field,
                vtype=uw.VarType.SCALAR,
                degree=u_Field.degree,
                continuous=u_Field.continuous,
                varsymbol=u_Field.symbol,
                verbose=verbose,
                evalf=evalf,
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

        if DFDt is None:
            self.Unknowns.DFDt = Symbolic_DDt(
                sympy.Matrix([[0] * self.mesh.dim]),
                varsymbol=rf"{{F[ {self.u.symbol} ] }}",
                theta=theta,
                bcs=None,
                order=order,
            )
            ### solution unable to solve after n timesteps, due to the projection of flux term (???)
            # self.Unknowns.DFDt = Eulerian_DDt(
            #     self.mesh,
            #     sympy.Matrix(
            #         [[0] * self.mesh.dim]
            #     ),  # Actual function is not defined at this point
            #     vtype=uw.VarType.VECTOR,
            #     degree=u_Field.degree,
            #     continuous=u_Field.continuous,
            #     varsymbol=rf"{{F[ {self.u.symbol} ] }}",
            #     theta=theta,
            #     evalf=evalf,
            #     verbose=verbose,
            #     bcs=None,
            #     order=order,
            #     smoothing=0.0,
            # )

        return

    @property
    def F0(self):

        f0 = expression(
            r"f_0 \left( \mathbf{u} \right)",
            -self.f + sympy.simplify(self.DuDt.bdf()) / self.delta_t,
            "Diffusion pointwise force term: f_0(u)",
        )

        # backward compatibility
        self._f0 = f0

        return f0

    @property
    def F1(self):

        F1_val = expression(
            r"\mathbf{F}_1\left( \mathbf{u} \right)",
            self.DFDt.adams_moulton_flux(),
            "Diffusion pointwise flux term: F_1(u)",
        )

        # backward compatibility
        self._f1 = F1_val

        return F1_val

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        self.is_setup = False
        self._f = sympy.Matrix((value,))

    @property
    def delta_t(self):
        return self._delta_t

    @delta_t.setter
    def delta_t(self, value):
        self.is_setup = False

        # Handle Pint Quantities with time dimensions
        if hasattr(value, "dimensionality"):
            # This is a Pint Quantity - check if it has time dimensions
            try:
                from ..scaling import units as ureg

                time_dim = ureg.second.dimensionality
                if value.dimensionality == time_dim:
                    # Convert physical time to nondimensional using model time scale
                    model = uw.get_default_model()
                    if model and hasattr(model, "fundamental_scales") and model.fundamental_scales:
                        time_scale = model.fundamental_scales.get("time")
                        if time_scale is not None:
                            # Physical time / time scale = nondimensional time
                            # Must use to_reduced_units() to convert both quantities
                            # to same base units before extracting magnitude.
                            # Otherwise Pint keeps different unit bases (megayear/second)
                            # and .magnitude returns the unconverted number!
                            result = value / time_scale

                            # Get the internal Pint quantity for proper unit conversion
                            if hasattr(result, "_pint_qty"):
                                # UWQuantity - access internal Pint quantity
                                pint_result = result._pint_qty.to_reduced_units()
                            elif hasattr(result, "to_reduced_units"):
                                # Raw Pint Quantity
                                pint_result = result.to_reduced_units()
                            else:
                                pint_result = result

                            # Extract the dimensionless magnitude
                            if hasattr(pint_result, "magnitude"):
                                value = float(pint_result.magnitude)
                            else:
                                value = float(pint_result)
            except Exception:
                pass  # If anything fails, try to use value as-is

        self._delta_t.sym = value

    @timing.routine_timer_decorator
    def estimate_dt(self):
        r"""
        Calculates an appropriate timestep for the diffusion solver.

        This solver only has a diffusive component, so the $\delta_t$ returned is:

            - ${\delta t}_\textrm{diff}$: typical time for the diffusion front to propagate across an element

        Returns:
            Pint Quantity or float: The diffusive timestep with physical time units
            if a model with reference scales is available, otherwise nondimensional.
        """

        ### required modules
        from mpi4py import MPI

        # Use the unified .K property from the constitutive model
        # This provides diffusivity for diffusion models
        K = self.constitutive_model.K

        # Evaluate the diffusivity (handles constant and spatially-varying cases)
        if isinstance(K, sympy.Expr) or hasattr(K, 'sym'):
            K_sym = K.sym if hasattr(K, 'sym') else K
            if uw.function.fn_is_constant_expr(K_sym):
                diffusivity = uw.function.evaluate(
                    K_sym,
                    np.zeros((1, self.mesh.dim)),
                )
            else:
                diffusivity = uw.function.evaluate(
                    sympy.sympify(K_sym),
                    self.mesh._centroids,
                    self.mesh.N,
                )
                diffusivity = diffusivity.max()
        else:
            diffusivity = K

        # If diffusivity is unit-aware (UnitAwareArray), nondimensionalise it to get
        # consistent nondimensional values that match mesh._radii
        # Note: .magnitude returns physical units, which would be wrong here
        if hasattr(diffusivity, "units") and diffusivity.units is not None:
            diffusivity = uw.non_dimensionalise(diffusivity)
        elif hasattr(diffusivity, "magnitude"):
            # Plain UWQuantity without units context - use magnitude
            diffusivity = diffusivity.magnitude

        # Ensure diffusivity is a plain numpy scalar
        diffusivity = float(np.asarray(diffusivity).max())

        ## get global max diffusivity value
        comm = uw.mpi.comm
        diffusivity_glob = comm.allreduce(diffusivity, op=MPI.MAX)

        ## get mesh spacing (nondimensional, consistent with diffusivity)
        min_dx = self.mesh.get_min_radius()

        ## estimate dt of diff component (all in nondimensional units)
        self.dt_diff = 0.0

        if diffusivity_glob != 0.0:
            dt_diff = (min_dx**2) / diffusivity_glob
            self.dt_diff = dt_diff
        else:
            dt_diff = np.inf

        dt_estimate = dt_diff

        # Apply unit-aware scaling if the solver has unit-aware variables
        return _apply_unit_aware_scaling(np.squeeze(dt_estimate), self.u, self.mesh)

    @timing.routine_timer_decorator
    def solve(
        self,
        zero_init_guess: bool = True,
        timestep: float = None,
        evalf: bool = False,
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
            # self._flux =  self.constitutive_model.flux.T
            # self._flux_star =  self._flux.copy()

        if not self.is_setup:
            self._setup_pointwise_functions(verbose)
            self._setup_discretisation(verbose)
            self._setup_solver(verbose)

        # Update History / Flux History terms
        # SemiLagrange and Lagrange may have different sequencing.
        # self.DuDt.update_pre_solve(verbose=verbose)
        # self.DFDt.update_pre_solve(verbose=verbose)

        super().solve(zero_init_guess, _force_setup)

        self.DuDt.update_post_solve(evalf=evalf, verbose=verbose)
        self.DFDt.update_post_solve(evalf=evalf, verbose=verbose)

        # if isinstance(self.DFDt, Eulerian_DDt):
        #     for i in range(order):
        #         ### have to substitute the unknown history term into the symbolic flux term
        #         self.DFDt.psi_star[i].subs({self.DuDt.psi_fn:self.DuDt.psi_star[i]})

        # self._flux_star =  self._flux.copy()

        self.is_setup = True
        self.constitutive_model._solver_is_setup = True

        return


# This one is already updated to work with the Lagrange D_Dt
class SNES_NavierStokes(SNES_Stokes_SaddlePt):
    r"""
    # Navier-Stokes Equation Solver

    This class provides a solver for the Navier-Stokes (vector Advection-Diffusion) equation which is similar to that
    used in the Semi-Lagrange Crank-Nicholson method (Spiegelman & Katz, 2006) but using a
    distributed sampling of upstream values taken from an arbitrary swarm variable.

    $$
    \color{Green}{\underbrace{ \Bigl[ \frac{\partial \mathbf{u} }{\partial t} +
                                      \left( \mathbf{u} \cdot \nabla \right) \mathbf{u} \ \Bigr]}_{\dot{\mathbf{u}}}} -
        \nabla \cdot
            \color{Blue}{\underbrace{\Bigl[ \frac{\boldsymbol{\eta}}{2} \left(
                    \nabla \mathbf{u} + \nabla \mathbf{u}^T \right) - p \mathbf{I} \Bigr]}_{\mathbf{F}}} =
            \color{Maroon}{\underbrace{\Bigl[ \mathbf{f} \Bigl] }_{\mathbf{h}}}
    $$

    The term $\mathbf{F}$ relates diffusive fluxes to gradients in the unknown $u$. The advective flux that results from having gradients along
    the direction of transport (given by the velocity vector field $\mathbf{v}$ ) are included in the $\dot{\mathbf{u}}$ term.

    The term $\dot{\mathbf{u}}$ involves upstream sampling to find the value $u^{ * }$ which represents the value of $u$ at
    the beginning of the timestep. This is achieved using a `swarmVariable` that carries history information along the flow path.
    A dense sampling is required to achieve similar accuracy to the original SLCN approach but it allows the use of a single swarm
    for history tracking of variables with different interpolation order and for material tracking. The user is required to supply
    **and update** the swarmVariable representing $u^{ * }$

    ## Properties

      - The unknown is $u$.

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

    ## References

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
            verbose,
            DuDt=DuDt,
            DFDt=DFDt,
        )

        # These are unique to the advection solver
        self._delta_t = expression(r"\Delta t", sympy.oo, "Navier-Stokes timestep")

        self.is_setup = False
        self._rho = expression(R"{\uprho}", rho, "Density")
        self._first_solve = True

        self._order = order
        self._penalty = expression(R"{\uplambda}", 0, "Incompressibility Penalty")

        self.restore_points_to_domain_func = restore_points_func
        self._bodyforce = sympy.Matrix([[0] * self.mesh.dim]).T
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
                self.u.sym,  # Symbolic expression - SemiLagrangian evaluates this at each update
                self.u.sym,
                vtype=uw.VarType.VECTOR,
                degree=self.u.degree,
                continuous=self.u.continuous,
                varsymbol=self.u.symbol,
                verbose=self.verbose,
                bcs=self.essential_bcs,
                order=self._order,
                smoothing=0.0001,
            )

        # F (at least for N-S) is a nodal point variable so there is no benefit
        # to treating it as a swarm variable. We'll define and use our own SL tracker
        # as we do in the SLCN version. We'll leave the option for an over-ride.
        #
        # Maybe u.degree-1. The scalar equivalent seems to show
        # little benefit from specific choices here other than
        # discontinuous flux variables amplifying instabilities.

        self.Unknowns.DFDt = uw.systems.ddt.SemiLagrangian(
            self.mesh,
            sympy.Matrix.zeros(self.mesh.dim, self.mesh.dim),
            self.u.sym,
            vtype=uw.VarType.SYM_TENSOR,
            degree=self.u.degree,
            continuous=self.u.continuous,
            varsymbol=rf"{{ F[ {self.u.symbol} ] }}",
            verbose=self.verbose,
            bcs=None,
            order=self._order,
        )

        ## Add in the history terms provided ...

        return

    @property
    def F0(self):

        DuDt = self.Unknowns.DuDt

        # I think this should be bdf(1) ... the higher order
        # terms are introduced through the adams_moulton fluxes

        f0 = expression(
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

            F1 = expression(
                r"\mathbf{F}_1\left( \mathbf{u} \right)",
                DFDt.adams_moulton_flux()
                - sympy.eye(self.mesh.dim) * (self.p.sym[0])
                + self.penalty * self.div_u * sympy.eye(dim),
                "NStokes pointwise flux term: F_1(u)",
            )
        # Is the else condition useful - other than to prevent a crash ?
        # Yes, because then it can just live on the Stokes solver ...
        else:
            F1 = expression(
                r"\mathbf{F}_1\left( \mathbf{u} \right)",
                self._constitutive_model.flux.T - sympy.eye(self.mesh.dim) * (self.p.sym[0]),
                "NStokes pressure gradient term: F_1(u) - No Flux history provided",
            )

        self._u_f1 = F1

        return F1

    @property
    def PF0(self):

        dim = self.mesh.dim

        f0 = expression(
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
        _evalf=False,
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
        self.DuDt.update_pre_solve(timestep, verbose=verbose, evalf=_evalf)
        self.DFDt.update_pre_solve(timestep, verbose=verbose, evalf=_evalf)

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

        self.DuDt.update_post_solve(timestep, verbose=verbose, evalf=_evalf)
        self.DFDt.update_post_solve(timestep, verbose=verbose, evalf=_evalf)

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

        The Navier-Stokes equations include momentum diffusion via kinematic viscosity
        (ν = η/ρ), so the diffusive timestep is computed from this quantity.

        returns: (${\delta t}_\textrm{diff}$, ${\delta t}_\textrm{adv}$)
        """

        ### required modules
        from mpi4py import MPI

        # For Navier-Stokes, diffusivity is the kinematic viscosity: ν = η/ρ
        # Use the unified .K property from the constitutive model (returns viscosity)
        K = self.constitutive_model.K

        # Evaluate the viscosity (handles constant and spatially-varying cases)
        if isinstance(K, sympy.Expr) or hasattr(K, 'sym'):
            K_sym = K.sym if hasattr(K, 'sym') else K
            if uw.function.fn_is_constant_expr(K_sym):
                diffusivity = uw.function.evaluate(
                    K_sym,
                    np.zeros((1, self.mesh.dim)),
                )
            else:
                diffusivity = uw.function.evaluate(
                    sympy.sympify(K_sym),
                    self.mesh._centroids,
                    self.mesh.N,
                )
                diffusivity = diffusivity.max()
        else:
            diffusivity = K

        # If diffusivity is unit-aware (UnitAwareArray), nondimensionalise it to get
        # consistent nondimensional values that match mesh._radii
        # Note: .magnitude returns physical units, which would be wrong here
        if hasattr(diffusivity, "units") and diffusivity.units is not None:
            diffusivity = uw.non_dimensionalise(diffusivity)
        elif hasattr(diffusivity, "magnitude"):
            # Plain UWQuantity without units context - use magnitude
            diffusivity = diffusivity.magnitude

        # Ensure diffusivity is a plain numpy scalar
        max_diffusivity = float(np.asarray(diffusivity).max())

        ## get global max diffusivity value
        comm = uw.mpi.comm
        diffusivity_glob = comm.allreduce(max_diffusivity, op=MPI.MAX)

        ### get the velocity values
        vel = uw.function.evaluate(
            self.u.sym,
            self.mesh._centroids,
            self.mesh.N,
        )

        # If vel is unit-aware (UnitAwareArray), nondimensionalise it to get
        # consistent nondimensional values that match mesh._radii
        # Note: .magnitude returns physical units, which would be wrong here
        if hasattr(vel, "units") and vel.units is not None:
            vel = uw.non_dimensionalise(vel)
        elif hasattr(vel, "magnitude"):
            # Plain UWQuantity without units context - use magnitude
            vel = vel.magnitude

        # Ensure vel is a plain numpy array
        vel = np.asarray(vel)

        ### get global velocity from velocity field
        max_magvel = np.linalg.norm(vel, axis=1).max()
        max_magvel_glob = comm.allreduce(max_magvel, op=MPI.MAX)

        ## get radius
        min_dx = self.mesh.get_min_radius()

        ## estimate dt of adv and diff components
        dt_diff = np.inf
        dt_adv = np.inf

        if diffusivity_glob != 0.0:
            dt_diff = (min_dx**2) / diffusivity_glob

        if max_magvel_glob != 0.0:
            dt_adv = min_dx / max_magvel_glob

        return (
            _apply_unit_aware_scaling(np.squeeze(dt_diff), self.u, self.mesh),
            _apply_unit_aware_scaling(np.squeeze(dt_adv), self.u, self.mesh),
        )
