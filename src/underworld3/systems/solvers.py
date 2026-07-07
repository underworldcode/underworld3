r"""
PDE solvers for Underworld3.

This module provides finite element solvers for common PDEs in geodynamics
and computational mechanics. All solvers use PETSc's SNES (Scalable Nonlinear
Equation Solvers) framework.

Scalar Equations
----------------
SNES_Poisson
    Poisson/diffusion equation: :math:`\nabla \cdot (k \nabla T) = f`
SNES_Darcy
    Darcy flow: pressure-driven flow through porous media
SNES_TransientDarcy
    Transient groundwater flow with constant storage
SNES_Richards
    Richards equation for variably-saturated porous media flow

Vector Equations
----------------
SNES_Stokes
    Stokes flow: :math:`\nabla \cdot \boldsymbol{\tau} - \nabla p = f`
SNES_VE_Stokes
    Viscoelastic Stokes with stress history
SNES_Stokes_SaddlePt
    Stokes with saddle-point preconditioner

Transport Equations
-------------------
SNES_AdvectionDiffusion
    Scalar advection-diffusion with Lagrangian tracking

Notes
-----
Solvers support unit-aware calculations when reference quantities are
set on the model. Physical units are preserved through the solve process
and timestep estimates are returned with appropriate time units.

See Also
--------
underworld3.systems.ddt : Time derivative schemes for transient problems.
underworld3.constitutive_models : Material constitutive laws.

Examples
--------
Set up a Stokes solver:

>>> stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
>>> stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
>>> stokes.viscosity = 1e21
>>> stokes.solve()
"""

import sympy
from sympy import sympify
import numpy as np

from typing import Optional, Callable, Union

import underworld3 as uw
from underworld3.systems import SNES_Scalar, SNES_Vector, SNES_Stokes_SaddlePt
from underworld3.cython.generic_solvers import SNES_MultiComponent
from underworld3 import VarType
import underworld3.timing as timing
from underworld3.utilities import memprobe
from underworld3.utilities._api_tools import (
    uw_object,
    SymbolicProperty,
    Parameter,
    Template,
    ExpressionProperty,
)

from underworld3.function import expression as public_expression


def expression(*args, **kwargs):
    """UWexpression factory with per-instance unique symbol names.

    Solver residual templates re-create expressions carrying the same LaTeX
    symbol every time they are rebuilt; unique name generation keeps each
    instance distinct so symbols from different solvers/meshes cannot collide
    (see docs/developer/design/SYMBOL_DISAMBIGUATION_2025-12.md).
    """
    return public_expression(*args, _unique_name_generation=True, **kwargs)


def _apply_unit_aware_scaling(dt_nondimensional, field, mesh):
    """
    Convert a nondimensional timestep estimate to physical time units.

    Multiplies by the model's fundamental time scale when one is configured;
    otherwise the nondimensional value is returned unchanged.

    Parameters
    ----------
    dt_nondimensional : float or np.ndarray
        The nondimensional timestep estimate.
    field, mesh : unused
        Retained for call-site stability: the estimate_dt implementation in
        ``cython/petsc_generic_snes_solvers.pyx`` calls this helper with
        ``(dt_nd, self.u, self.mesh)``. (An earlier version inspected the
        field's units before scaling, but both branches multiplied by the
        same ``fundamental_scales['time']``, so the inspection changed
        nothing and was removed.)

    Returns
    -------
    float or UWQuantity
        Timestep with physical time units if a time scale is configured,
        otherwise the nondimensional input.
    """
    try:
        from ..function.quantities import UWQuantity

        orchestration_model = uw.get_default_model()
        if (
            orchestration_model
            and hasattr(orchestration_model, "fundamental_scales")
            and orchestration_model.fundamental_scales
        ):
            time_scale = orchestration_model.fundamental_scales.get("time")
            if time_scale is not None:
                dt_physical = dt_nondimensional * time_scale
                if not isinstance(dt_physical, UWQuantity):
                    dt_physical = UWQuantity._from_pint(dt_physical)
                return dt_physical
    except Exception:
        # Sanctioned swallow: units machinery unavailable or misconfigured
        # (no default model, Pint conversion failure) — a timestep estimate
        # must always come back, so fall back to the nondimensional value.
        pass

    return dt_nondimensional


def _nondimensionalise_timestep(value):
    """Convert a dimensional (time-valued) timestep to its nondimensional value.

    Parameters
    ----------
    value : float, sympy expression, Pint Quantity, or UWQuantity
        Timestep. A Pint/UWQuantity with time dimensionality is divided by
        the model's fundamental time scale; any other input is returned
        unchanged (assumed already nondimensional).

    Returns
    -------
    float or the original object
        The nondimensional magnitude when the conversion applies, otherwise
        ``value`` unchanged.
    """
    if not hasattr(value, "dimensionality"):
        return value
    try:
        from ..scaling import units as ureg

        if value.dimensionality != ureg.second.dimensionality:
            return value
        orchestration_model = uw.get_default_model()
        if not (
            orchestration_model
            and hasattr(orchestration_model, "fundamental_scales")
            and orchestration_model.fundamental_scales
        ):
            return value
        time_scale = orchestration_model.fundamental_scales.get("time")
        if time_scale is None:
            return value

        # Physical time / time scale = nondimensional time.
        # Must use to_reduced_units() to convert both quantities to the same
        # base units before extracting the magnitude. Otherwise Pint keeps
        # different unit bases (megayear/second) and .magnitude returns the
        # unconverted number!
        result = value / time_scale
        if hasattr(result, "_pint_qty"):
            # UWQuantity - reduce via the internal Pint quantity
            pint_result = result._pint_qty.to_reduced_units()
        elif hasattr(result, "to_reduced_units"):
            # Raw Pint Quantity
            pint_result = result.to_reduced_units()
        else:
            pint_result = result

        if hasattr(pint_result, "magnitude"):
            return float(pint_result.magnitude)
        return float(pint_result)
    except Exception:
        # Sanctioned swallow: units machinery unavailable or misconfigured —
        # fall back to using the value as-is (legacy behaviour of the
        # delta_t setters this was extracted from).
        return value


def _global_max_diffusivity(constitutive_K, mesh):
    r"""Global (all-rank) maximum of the diffusivity over the mesh.

    Shared by the transient solvers' ``estimate_dt``: the diffusive CFL
    limit needs one representative :math:`\kappa_{\max}` for
    :math:`\delta t = h^2 / \kappa_{\max}`.

    Parameters
    ----------
    constitutive_K : sympy expression, UWexpression, or number
        The constitutive model's ``.K`` (diffusivity, or kinematic
        viscosity for Navier-Stokes).
    mesh : Mesh
        Supplies the sampling points (cell centroids) and the coordinate
        basis.

    Returns
    -------
    float
        Nondimensional global maximum (MPI allreduce MAX across ranks).
    """
    from mpi4py import MPI

    K = constitutive_K
    if isinstance(K, sympy.Expr) or hasattr(K, "sym"):
        K_sym = K.sym if hasattr(K, "sym") else K
        if uw.function.fn_is_constant_expr(K_sym):
            diffusivity = uw.function.evaluate(
                K_sym, np.zeros((1, mesh.dim)))
        else:
            # Spatially varying: sample at cell centroids, take the max.
            diffusivity = uw.function.evaluate(
                sympy.sympify(K_sym), mesh._centroids, mesh.N)
            diffusivity = diffusivity.max()
    else:
        diffusivity = K

    # If unit-aware (UnitAwareArray), nondimensionalise so the value is
    # consistent with mesh._radii. Note: .magnitude alone would keep the
    # physical-units number, which would be wrong here.
    if hasattr(diffusivity, "units") and diffusivity.units is not None:
        diffusivity = uw.non_dimensionalise(diffusivity)
    elif hasattr(diffusivity, "magnitude"):
        # Plain UWQuantity without units context - use magnitude
        diffusivity = diffusivity.magnitude

    local_max = float(np.asarray(diffusivity).max())
    return uw.mpi.comm.allreduce(local_max, op=MPI.MAX)


def _centroid_velocities_nd(V_fn, mesh, basis=None, ensure_2d=True):
    """Nondimensional velocities sampled at element centroids.

    Shared by the ``estimate_dt`` implementations: the advective CFL limit
    needs per-element centroid velocities in the same (nondimensional)
    scale as ``mesh._radii``.

    Parameters
    ----------
    V_fn : sympy Matrix expression
        Velocity function to sample.
    mesh : Mesh
        Supplies the centroid sample points.
    basis : optional
        Coordinate basis forwarded to ``uw.function.evaluate`` (the
        Navier-Stokes caller passes ``mesh.N``; the others use the default).
    ensure_2d : bool, default True
        Squeeze evaluate's singleton axes ((N, 1, dim) -> (N, dim)) and
        re-expand a single-element result to 2-D. The Navier-Stokes caller
        historically skips this and reduces the raw array.

    Returns
    -------
    numpy.ndarray
        Velocity samples, shape ``(N, dim)`` when ``ensure_2d``.
    """
    if basis is not None:
        vel = uw.function.evaluate(V_fn, mesh._centroids, basis)
    else:
        vel = uw.function.evaluate(V_fn, mesh._centroids)

    # If unit-aware (UnitAwareArray), nondimensionalise so the values are
    # consistent with mesh._radii. Note: .magnitude alone would keep the
    # physical-units numbers, which would be wrong here.
    if hasattr(vel, "units") and vel.units is not None:
        vel = uw.non_dimensionalise(vel)
    elif hasattr(vel, "magnitude"):
        # Plain UWQuantity without units context - use magnitude
        vel = vel.magnitude

    vel = np.asarray(vel)
    if ensure_2d:
        # Squeeze singleton axes from evaluate ((N,1,dim) -> (N,dim)) and
        # handle the single-element edge case (ensure 2D).
        vel = np.squeeze(vel)
        if vel.ndim == 1:
            vel = vel.reshape(1, -1)
    return vel


def _invalidate_solution_cache(u):
    """Drop the cached data view of a just-solved variable.

    PETSc may replace the underlying vector buffers during a solve, so the
    lazily-built ``.data`` / ``.array`` view on the unknown must be rebuilt
    on next access. Handles both EnhancedMeshVariable (the base variable
    sits behind ``_base_var``) and a bare ``_MeshVariable``.

    Parameters
    ----------
    u : MeshVariable
        The solver unknown whose cached view is invalidated.
    """
    target_var = getattr(u, "_base_var", u)
    if hasattr(target_var, "_canonical_data"):
        target_var._canonical_data = None


from .ddt import SemiLagrangian as SemiLagrangian_DDt
from .ddt import Lagrangian as Lagrangian_DDt
from .ddt import Eulerian as Eulerian_DDt
from .ddt import Symbolic as Symbolic_DDt


class SNES_Poisson(SNES_Scalar):
    r"""Poisson equation solver.

    Provides a discrete representation of the Poisson equation:

    .. math::

        - \nabla \cdot \left[ \boldsymbol{\kappa} \nabla u \right] = f

    where :math:`\mathbf{F} = \boldsymbol{\kappa} \nabla u` relates the flux to
    gradients in the unknown :math:`u`.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh.
    u_Field : MeshVariable, optional
        Pre-existing mesh variable for the solution. If None, one is created.
    degree : int, optional
        Polynomial degree for the solution field (default: 2).
    verbose : bool, optional
        Enable verbose output during solve.
    DuDt : SemiLagrangian_DDt or Lagrangian_DDt, optional
        Time derivative operator for time-dependent problems.
    DFDt : SemiLagrangian_DDt or Lagrangian_DDt, optional
        Time derivative operator for the flux.

    Notes
    -----
    The constructor follows the family-wide parameter order
    ``(mesh, u_Field, degree, verbose, ...)`` (Style Charter, API
    conventions); SNES_Poisson historically inverted ``(verbose, degree)``.
    A legacy positional call is detected by ``type(degree) is bool`` (a
    polynomial degree is never a bool) and shimmed with one
    DeprecationWarning.

    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        u_Field: uw.discretisation.MeshVariable = None,
        degree=2,
        verbose=False,
        DuDt: Union[SemiLagrangian_DDt, Lagrangian_DDt] = None,
        DFDt: Union[SemiLagrangian_DDt, Lagrangian_DDt] = None,
    ):
        if type(degree) is bool:
            # Legacy positional order (mesh, u_Field, verbose, degree): the
            # bool in the degree slot is the legacy verbose flag; a legacy
            # positional degree, if present, landed in the verbose slot.
            import warnings
            warnings.warn(
                "SNES_Poisson(mesh, u_Field, verbose, degree) is deprecated; "
                "use SNES_Poisson(mesh, u_Field, degree, verbose)",
                DeprecationWarning, stacklevel=2)
            degree, verbose = (
                verbose if type(verbose) is not bool else 2), degree

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

    # =========================================================================
    # PETSc Residual Templates
    # For the Poisson equation: -∇·(k∇u) = f
    # =========================================================================

    F0 = Template(
        r"f_0 \left( \mathbf{u} \right)",
        lambda self: -self.f,
        r"""Source term for the Poisson equation (pointwise).

        The $f_0$ term represents the source/sink in the diffusion equation.
        Set via the ``f`` property (e.g., ``poisson.f = heat_source``).
        """,
    )

    F1 = Template(
        r"\mathbf{F}_1\left( \mathbf{u} \right)",
        lambda self: self.constitutive_model.flux.T,
        r"""Diffusive flux term for the Poisson equation (pointwise).

        The $\mathbf{F}_1$ vector represents the flux $k \nabla u$
        from the constitutive model, where $k$ is the diffusivity.
        """,
    )

    @timing.routine_timer_decorator
    def poisson_problem_description(self):
        """Build residual terms for Poisson FEM assembly."""
        # f1 residual term (weighted integration) - scalar function
        self._f0 = self.F0.sym

        # f1 residual term (integration by parts / gradients)
        # isotropic
        self._f1 = self.F1.sym

        return

    @property
    def f(self):
        r"""Source term for the Poisson equation.

        The source term :math:`f` appears on the right-hand side:

        .. math::
            - \nabla \cdot (\kappa \nabla u) = f

        Returns
        -------
        sympy.Matrix
            Source term expression (scalar, shape ``(1, 1)``).

        See Also
        --------
        constitutive_model : Provides the diffusivity :math:`\kappa`.
        """
        return self._f

    @f.setter
    def f(self, value):
        """Set the source term (handles units and scaling)."""
        self._needs_function_rewire = True

        # Handle UWQuantity with units - enforce "units everywhere" principle
        if hasattr(value, "value") and hasattr(value, "units"):
            # Extract the plain value
            plain_value = float(value.value)

            # If ND scaling is active, scale the constant
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
                orchestration_model = uw.get_default_model()

                if orchestration_model.has_units():
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
        """Whether the constitutive model is configured for this solver."""
        return self._constitutive_model._solver_is_setup


class SNES_Darcy(SNES_Scalar):
    r"""
    Darcy flow equation solver for groundwater problems.

    Provides a discrete representation of the groundwater flow equations:

    .. math::

        \underbrace{S_s \frac{\partial h}{\partial t}}_{\dot{u}}
        - \nabla \cdot \underbrace{\left[ \boldsymbol{\kappa} \left(
        \nabla h - \boldsymbol{s} \right) \right]}_{\mathbf{F}}
        = \underbrace{W}_{h}

    The physical Darcy velocity is minus the assembly flux
    :math:`\mathbf{F} = \boldsymbol{\kappa}(\nabla h - \boldsymbol{s})`
    (flow runs *down* the head gradient):

    .. math::

        \boldsymbol{v} = - \boldsymbol{\kappa} \left( \nabla h - \boldsymbol{s} \right)

    Parameters
    ----------
    mesh : Mesh
        The computational mesh.
    h_Field : MeshVariable, optional
        Mesh variable for hydraulic head. Created automatically if not provided.
    v_Field : MeshVariable, optional
        Mesh variable for Darcy velocity. Created automatically if not provided.
    degree : int, default=2
        Polynomial degree for the finite element discretization.
    verbose : bool, default=False
        Enable verbose output.
    DuDt : optional
        Time derivative operator for the unknown.
    DFDt : optional
        Time derivative operator for the flux.

    Notes
    -----
    - The unknown is :math:`h`, the hydraulic head
    - The permeability tensor :math:`\kappa` is set via the ``constitutive_model``
      property using one of the ``uw.constitutive_models`` classes
    - :math:`W` is a pressure source term
    - :math:`S_s` is the specific storage coefficient
    - The time-dependent term :math:`\dot{f}` is not implemented in this version
    - The solver returns both the primary field and the Darcy flux (mean-flow velocity)

    See Also
    --------
    SNES_Poisson : Related diffusion-only solver.
    uw.constitutive_models.DarcyFlowModel : Constitutive model for Darcy flow.
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

    # =========================================================================
    # PETSc Residual Templates
    # For Darcy flow: -∇·(K∇p) = f, with velocity v = -K∇p
    # =========================================================================

    F0 = Template(
        r"f_0 \left( \mathbf{u} \right)",
        lambda self: -self.f,
        r"""Source term for the Darcy pressure equation (pointwise).

        The $f_0$ term represents fluid sources/sinks.
        Set via the ``f`` property.
        """,
    )

    F1 = Template(
        r"\mathbf{F}_1\left( \mathbf{u} \right)",
        lambda self: self.darcy_flux,
        r"""Darcy flux term (pointwise).

        The $\mathbf{F}_1$ vector is the Darcy flux $K \nabla p$,
        where $K$ is the permeability.
        """,
    )

    @timing.routine_timer_decorator
    def darcy_problem_description(self):
        """Build residual terms for Darcy flow FEM assembly."""
        # f1 residual term (weighted integration)
        self._f0 = self.F0.sym

        # f1 residual term (integration by parts / gradients)
        self._f1 = self.F1.sym

        # Flow calculation
        self._v_projector.uw_function = -self.F1.sym

        return

    @property
    def f(self):
        """Source term for the Darcy equation."""
        return self._f

    @f.setter
    def f(self, value):
        """Set the source term."""
        self._needs_function_rewire = True
        self._f = sympy.Matrix((value,))

    @property
    def darcy_flux(self):
        """Darcy flux velocity computed from pressure gradient."""
        flux = self.constitutive_model.flux.T
        return flux

    @property
    def v(self):
        """Projected Darcy velocity field."""
        return self._v

    @v.setter
    def v(self, value):
        """Set the velocity projection target."""
        self._v_projector.is_setup = False
        self._v = sympify(value)

    @timing.routine_timer_decorator
    def solve(
        self,
        zero_init_guess: bool = True,
        timestep: float = None,
        verbose: bool = False,
        _force_setup: bool = False,
        divergence_retries: int = 0,
    ):
        r"""Solve the Darcy flow system.

        Computes the pressure field and Darcy flux velocity.

        Parameters
        ----------
        zero_init_guess : bool, optional
            If True (default), start from zero initial guess.
            If False, use current field values as initial guess.
        timestep : float, optional
            Timestep value for inertial terms (if applicable).
        verbose : bool, optional
            If True, print solver progress information.
        _force_setup : bool, optional
            Force re-setup of solver even if already configured.
        divergence_retries : int, optional
            If SNES reports DIVERGED, retry with warm start up to this
            many times. 0 preserves legacy behaviour.

        Notes
        -----
        After solving, the pressure field ``self.u`` and velocity field
        ``self.v`` contain the solution.
        """

        if (not self.is_setup) or _force_setup:
            self._setup_pointwise_functions(verbose)
            self._setup_discretisation(verbose)
            self._setup_solver(verbose)

        # Solve pressure

        super().solve(zero_init_guess, _force_setup,
                      divergence_retries=divergence_retries)

        # Now solve flow field: v = -flux = -K(grad(h) - s)

        # self._v_projector.petsc_options["snes_rtol"] = 1.0e-6
        # self._v_projector.petsc_options.delValue("ksp_monitor")
        self._v_projector.uw_function = -self.darcy_flux
        self._v_projector.solve(zero_init_guess)

        return

    # @timing.routine_timer_decorator
    # def _setup_terms(self):
    #     self._v_projector.uw_function = self.darcy_flux
    #     self._v_projector._setup_terms()  # _setup_pointwise_functions()  #
    #     super()._setup_terms()


class SNES_TransientDarcy(SNES_Darcy):
    r"""
    Transient Darcy flow solver for time-dependent groundwater problems.

    Solves the transient groundwater flow equation:

    .. math::

        S_s \frac{\partial h}{\partial t}
        - \nabla \cdot \left[ K (\nabla h - \mathbf{s}) \right] = f

    where :math:`S_s` is the specific storage (constant), :math:`K` is the
    hydraulic conductivity, :math:`\mathbf{s}` is the body force (gravity),
    and :math:`f` is a source/sink term.

    Inherits velocity projection from :class:`SNES_Darcy`.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh.
    h_Field : MeshVariable, optional
        Mesh variable for hydraulic head.
    v_Field : MeshVariable, optional
        Mesh variable for Darcy velocity.
    order : int, default=1
        Time integration order (BDF/Adams-Moulton history depth).
    theta : float, default=0.5
        Implicit time weighting (0=explicit, 0.5=Crank-Nicolson, 1=implicit).
    degree : int, default=2
        Polynomial degree for finite element discretization.
    verbose : bool, default=False
        Enable verbose output.
    DuDt : optional
        Time derivative operator for the unknown.
    DFDt : optional
        Time derivative operator for the flux.

    Attributes
    ----------
    storage : sympy.Expr
        Specific storage :math:`S_s` (default 1).
    delta_t : UWexpression
        Current timestep.

    See Also
    --------
    SNES_Darcy : Steady-state parent solver.
    SNES_Richards : Nonlinear extension for unsaturated flow.
    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        h_Field: Optional[uw.discretisation.MeshVariable] = None,
        v_Field: Optional[uw.discretisation.MeshVariable] = None,
        order: int = 1,
        theta: float = 0.5,
        degree: int = 2,
        verbose=False,
        DuDt=None,
        DFDt=None,
    ):
        super().__init__(mesh, h_Field, v_Field, degree, verbose, DuDt=DuDt, DFDt=DFDt)

        self.theta = theta
        self._delta_t = expression(R"\Delta t", 0, "Physically motivated timestep")
        self._storage = sympy.sympify(1)
        self._needs_function_rewire = True

        if DuDt is None:
            self.Unknowns.DuDt = Eulerian_DDt(
                self.mesh,
                h_Field,
                vtype=uw.VarType.SCALAR,
                degree=h_Field.degree,
                continuous=h_Field.continuous,
                varsymbol=h_Field.symbol,
                verbose=verbose,
                order=order,
                smoothing=0.0,
            )
        else:
            if order is not None and DuDt.order < order:
                raise RuntimeError(
                    f"DuDt supplied is order {DuDt.order} but order requested is {order}"
                )
            self.Unknowns.DuDt = DuDt

        if DFDt is None:
            self.Unknowns.DFDt = Symbolic_DDt(
                sympy.Matrix([[0] * self.mesh.dim]),
                varsymbol=rf"{{F[ {self.u.symbol} ] }}",
                theta=theta,
                order=order,
            )
        else:
            self.Unknowns.DFDt = DFDt

    # --- Properties ---

    @property
    def storage(self):
        """Specific storage coefficient :math:`S_s`."""
        return self._storage

    @storage.setter
    def storage(self, value):
        self._needs_function_rewire = True
        self._storage = sympy.sympify(value)

    @property
    def delta_t(self):
        """Current timestep."""
        return self._delta_t

    @delta_t.setter
    def delta_t(self, value):
        self._needs_function_rewire = True
        if hasattr(value, "value"):
            self._delta_t.sym = value.value
        elif hasattr(value, "magnitude"):
            self._delta_t.sym = value.magnitude
        else:
            self._delta_t.sym = value

    @property
    def DuDt(self):
        return self.Unknowns.DuDt

    @property
    def DFDt(self):
        return self.Unknowns.DFDt

    # --- Template expressions (override Darcy's steady-state Templates) ---

    @property
    def F0(self):
        """Pointwise storage + source: :math:`S_s \\dot{h} / \\Delta t - f`."""
        f0 = expression(
            r"f_0(h)",
            -self.f + self.storage * self.DuDt.bdf() / self.delta_t,
            "Transient Darcy storage + source term",
        )
        self._f0 = f0
        return f0

    @property
    def F1(self):
        """Pointwise flux: Adams-Moulton time-weighted Darcy flux."""
        F1_val = expression(
            r"\mathbf{F}_1(h)",
            self.DFDt.adams_moulton_flux(),
            "Transient Darcy flux (time-weighted)",
        )
        self._f1 = F1_val
        return F1_val

    # --- Timestep estimation ---

    @timing.routine_timer_decorator
    def estimate_dt(self):
        r"""
        Estimate a stable timestep based on diffusive CFL.

        Returns
        -------
        float or pint.Quantity
            Diffusive timestep :math:`\delta t = (\Delta x)^2 / K_{\max}`.
        """
        diffusivity_glob = _global_max_diffusivity(
            self.constitutive_model.K, self.mesh)

        min_dx = self.mesh.get_min_radius()

        if diffusivity_glob != 0.0:
            dt_diff = (min_dx**2) / diffusivity_glob
        else:
            dt_diff = np.inf

        return _apply_unit_aware_scaling(np.squeeze(dt_diff), self.u, self.mesh)

    # --- Solve ---

    @timing.routine_timer_decorator
    def solve(
        self,
        zero_init_guess: bool = True,
        timestep=None,
        _force_setup: bool = False,
        verbose=False,
        divergence_retries: int = 0,
    ):
        r"""
        Solve the transient Darcy system for one timestep.

        Parameters
        ----------
        zero_init_guess : bool, optional
            Start from zero initial guess (default True).
        timestep : float, optional
            Timestep size. Updates ``self.delta_t`` if provided.
        _force_setup : bool, optional
            Force re-setup of solver.
        verbose : bool, optional
            Print solver progress.
        divergence_retries : int, optional
            If SNES reports DIVERGED, retry with warm start up to this
            many times. 0 preserves legacy behaviour.
        """
        if timestep is not None and timestep != self.delta_t:
            self.delta_t = timestep

        if not self.constitutive_model._solver_is_setup:
            self._needs_function_rewire = True
            self.DFDt.psi_fn = self.constitutive_model.flux.T

        if not self.is_setup:
            self._setup_pointwise_functions(verbose)
            self._setup_discretisation(verbose)
            self._setup_solver(verbose)

        # Pre-solve: update history terms
        self.DuDt.update_pre_solve(timestep, verbose=verbose)
        self.DFDt.update_pre_solve(timestep, verbose=verbose)

        # Solve PDE (bypass SNES_Darcy.solve to avoid double setup/projection)
        SNES_Scalar.solve(self, zero_init_guess, _force_setup,
                          divergence_retries=divergence_retries)

        _invalidate_solution_cache(self.u)

        # Post-solve: shift history
        self.DuDt.update_post_solve(timestep, verbose=verbose)
        self.DFDt.update_post_solve(timestep, verbose=verbose)

        # Velocity projection (inherited from Darcy). The physical Darcy
        # velocity is v = -flux = -kappa(grad(h) - s); the assembly flux F1 =
        # darcy_flux = +kappa(grad(h) - s), so project the NEGATED flux to match
        # the steady SNES_Darcy.solve sign (UW3 issue #214: transient previously
        # projected +darcy_flux, giving a sign-flipped velocity field).
        self._v_projector.uw_function = -self.darcy_flux
        self._v_projector.solve(zero_init_guess)

        self.is_setup = True
        self.constitutive_model._solver_is_setup = True

        return


class SNES_Richards(SNES_TransientDarcy):
    r"""
    Richards equation solver for variably-saturated porous media flow.

    Two formulations are supported:

    **Mixed form** (mass-conservative, preferred) — set ``water_content``:

    .. math::

        \frac{\partial \theta}{\partial t}
        - \nabla \cdot \left[ K(\psi) (\nabla \psi - \mathbf{s}) \right] = f

    discretised as :math:`(\theta(\psi^{n+1}) - \theta(\psi^n)) / \Delta t`,
    which is exactly conservative by construction (Celia et al., 1990).

    **Head-based form** (backward compatible) — set ``capacity``:

    .. math::

        C(\psi) \frac{\partial \psi}{\partial t}
        - \nabla \cdot \left[ K(\psi) (\nabla \psi - \mathbf{s}) \right] = f

    where :math:`C(\psi) = d\theta/d\psi` is the specific moisture capacity.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh.
    psi_Field : MeshVariable, optional
        Mesh variable for pressure head :math:`\psi`.
    v_Field : MeshVariable, optional
        Mesh variable for Darcy velocity.
    order : int, default=1
        Time integration order.
    theta : float, default=0.5
        Implicit time weighting.
    degree : int, default=2
        Polynomial degree.
    verbose : bool, default=False
        Enable verbose output.
    DuDt : optional
        Time derivative operator for the unknown.
    DFDt : optional
        Time derivative operator for the flux.

    Attributes
    ----------
    water_content : sympy.Expr or None
        Water content function :math:`\theta(\psi)` for the mixed form.
        When set, the storage term uses
        :math:`(\theta(\psi^{n+1}) - \theta(\psi^n)) / \Delta t`.
        Takes precedence over ``capacity`` if both are set.
    capacity : sympy.Expr
        Specific moisture capacity :math:`C(\psi)` for the head-based form.
        Used only when ``water_content`` is None.
    psi : MeshVariable
        Alias for ``self.u`` (pressure head).

    See Also
    --------
    SNES_TransientDarcy : Linear parent solver.
    underworld3.utilities.retention_curves : Van Genuchten and Gardner functions.

    Examples
    --------
    Mixed form (preferred):

    >>> from underworld3.utilities.retention_curves import (
    ...     van_genuchten_theta, van_genuchten_K,
    ... )
    >>> richards = uw.systems.Richards(mesh, psi_Field=psi, v_Field=v)
    >>> richards.constitutive_model = uw.constitutive_models.DarcyFlowModel
    >>> richards.constitutive_model.Parameters.permeability = van_genuchten_K(
    ...     psi.sym[0], Ks=1e-4, alpha=3.35, n=2.0,
    ... )
    >>> richards.water_content = van_genuchten_theta(
    ...     psi.sym[0], theta_r=0.045, theta_s=0.43, alpha=3.35, n=2.0,
    ... )

    Head-based form (backward compatible):

    >>> from underworld3.utilities.retention_curves import van_genuchten_C
    >>> richards.capacity = van_genuchten_C(
    ...     psi.sym[0], theta_r=0.045, theta_s=0.43, alpha=3.35, n=2.0,
    ... )
    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        psi_Field: Optional[uw.discretisation.MeshVariable] = None,
        v_Field: Optional[uw.discretisation.MeshVariable] = None,
        order: int = 1,
        theta: float = 0.5,
        degree: int = 2,
        verbose=False,
        DuDt=None,
        DFDt=None,
    ):
        super().__init__(
            mesh, psi_Field, v_Field, order, theta, degree, verbose, DuDt, DFDt
        )
        self._capacity = sympy.sympify(1)
        self._water_content = None  # None → head-based form; set → mixed form

    @property
    def water_content(self):
        r"""Water content function :math:`\theta(\psi)` for the mixed form.

        When set, the storage term uses
        :math:`(\theta(\psi^{n+1}) - \theta(\psi^n)) / \Delta t`
        instead of :math:`C(\psi) \cdot (\psi^{n+1} - \psi^n) / \Delta t`,
        giving exact mass conservation (Celia et al., 1990).

        The expression should be a SymPy function of ``psi.sym[0]``.
        The Jacobian :math:`\partial\theta/\partial\psi = C(\psi)` is
        computed automatically by PETSc (finite-difference colouring),
        so there is no need to provide :math:`C(\psi)` separately.
        """
        return self._water_content

    @water_content.setter
    def water_content(self, value):
        self._needs_function_rewire = True
        self._water_content = sympy.sympify(value) if value is not None else None

    @property
    def capacity(self):
        r"""Specific moisture capacity :math:`C(\psi) = d\theta/d\psi`.

        Used only when ``water_content`` is None (head-based form).
        Typically a nonlinear SymPy expression depending on ``psi.sym[0]``.
        """
        return self._capacity

    @capacity.setter
    def capacity(self, value):
        self._needs_function_rewire = True
        self._capacity = sympy.sympify(value)

    @property
    def psi(self):
        """Alias for ``self.u`` (pressure head)."""
        return self.u

    @property
    def F0(self):
        r"""Pointwise storage + source term.

        Mixed form: :math:`(\theta(\psi^{n+1}) - \theta(\psi^n)) / \Delta t - f`

        Head-based form: :math:`C(\psi) (\psi^{n+1} - \psi^n) / \Delta t - f`
        """
        if self._water_content is not None:
            # Mixed (mass-conservative) form:
            # θ(ψ^{n+1}) is self._water_content (already in terms of psi.sym[0])
            # θ(ψ^n) is water_content with psi.sym[0] → psi_star[0].sym[0]
            psi_sym = self.psi.sym[0]
            psi_star_sym = self.DuDt.psi_star[0].sym[0]
            theta_new = self._water_content
            theta_old = self._water_content.subs(psi_sym, psi_star_sym)
            storage_term = sympy.Matrix([[theta_new - theta_old]]) / self.delta_t
        else:
            # Head-based form (backward compatible)
            storage_term = self.capacity * self.DuDt.bdf() / self.delta_t

        f0 = expression(
            r"f_0(\psi)",
            -self.f + storage_term,
            "Richards storage + source term",
        )
        self._f0 = f0
        return f0


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
        self._needs_function_rewire = True

        self._constitutive_model = None

        # Optional: alternative F1 expression to autodiff for the
        # Jacobian.  When None, autodiff F1 itself (default).  Used for
        # inexact-Newton tricks like a smooth-Jacobian / sharp-residual
        # split at a VEP yield kink.  See ``set_jacobian_F1_source``.
        self._F1_jacobian_source = None

        # Last-seen constitutive effective_order (VE/VEP DDt history
        # ramp-up); solve() re-wires the pointwise functions when it changes.
        self._prev_effective_order = None

        return

    def set_jacobian_F1_source(self, F1_source, linesearch="cp"):
        r"""Override the F1 expression used to build the Jacobian blocks.

        By default, the Stokes Jacobian's uu / up G2, G3 blocks are
        autodiff'd from the residual F1.  Some problems benefit from
        differentiating a *different* but related expression — e.g. a
        smooth (softmin) viscosity formula for the Jacobian while the
        residual F1 keeps a sharp Min, so Newton sees a continuous
        derivative even when the iterate sits exactly on the yield kink.

        Setting ``F1_source`` triggers a JIT recompile (the Jacobian
        symbols change).  Pass ``None`` to revert to autodiff of F1.

        Parameters
        ----------
        F1_source : sympy.Matrix or None
            Alternative expression of the same shape as ``F1.sym``.
        linesearch : str or None, default ``"cp"``
            SNES linesearch type to install when ``F1_source`` is set.
            Defaults to ``"cp"`` (critical-point) because inexact-Newton
            steps don't reliably reduce the residual norm and PETSc's
            default ``bt`` (backtracking) consequently rejects useful
            steps with ``DIVERGED_LINE_SEARCH``.  ``cp`` accepts the
            predicted step at the optimum of the local linearisation and
            converges cleanly on the same problems where ``bt`` flails.
            Set to ``None`` to leave the linesearch type untouched (e.g.
            if you've already configured one via ``petsc_options``).
            Has no effect when ``F1_source is None``.
        """
        self._F1_jacobian_source = F1_source
        self._needs_function_rewire = True
        if F1_source is not None and linesearch is not None:
            self.petsc_options["snes_linesearch_type"] = linesearch

    def _create_stress_history_ddt(self, order=2):
        """Create DFDt for stress history tracking (VE/VEP models).

        Called automatically when a constitutive model with
        ``requires_stress_history = True`` is assigned. Can also be called
        explicitly to pre-create the DFDt with a specific order.

        Constitutive models can inject extra SemiLagrangian kwargs via the
        ``stress_history_ddt_kwargs`` property — used e.g. by
        ``MaxwellExponentialFlowModel`` to set ``with_forcing_history=True``.
        """
        if self.Unknowns.DFDt is not None:
            return  # already created

        self._order = order
        # Constitutive model may request extra SemiLagrangian kwargs (e.g.
        # with_forcing_history for ETD-2 integration).
        cm = getattr(self, "constitutive_model", None)
        ddt_kwargs = {}
        if cm is not None:
            ddt_kwargs = dict(getattr(cm, "stress_history_ddt_kwargs", {}))

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
            order=order,
            smoothing=0.0001,
            **ddt_kwargs,
        )
        # Stress flux = 2·viscosity·E_eff references psi_star[0] in E_eff's
        # history term — without snapshot substitution the projection of
        # flux→psi_star[0] becomes implicit in psi_star[0] and Min-mode at
        # yield admits the wrong fixed point under timestep change.
        self.Unknowns.DFDt.enable_source_snapshot()

    @timing.routine_timer_decorator
    @memprobe.instrument("Stokes.solve")
    def solve(
        self,
        zero_init_guess: bool = True,
        timestep: float = None,
        _force_setup: bool = False,
        verbose: bool = False,
        debug: bool = False,
        debug_name: str = None,
        evalf=False,
        order=None,
        picard: int = 0,
        divergence_retries: int = 0,
    ):
        """Solve the Stokes system, with optional viscoelastic stress history.

        When a constitutive model with stress history is active (DFDt is not None),
        the solve includes pre/post hooks for advecting stress history, updating
        BDF coefficients, and projecting the actual stress after the solve.

        Parameters
        ----------
        zero_init_guess : bool
            If True, use zero initial guess. Otherwise use current field values.
        timestep : float, optional
            Advection timestep. Required when stress history is active.
        _force_setup : bool
            Force rebuild of pointwise functions.
        verbose : bool
            Enable verbose output.
        evalf : bool
            Force numerical evaluation during history updates.
        order : int, optional
            Override the VE time integration order.
        picard : int, default=0
            Number of Picard iterations before switching to Newton.
            Picard uses a simplified Jacobian and can help convergence
            for strongly nonlinear problems like VEP at yield onset.
        divergence_retries : int, default=0
            If SNES returns a DIVERGED reason after the main solve, re-call
            the underlying Newton up to this many times with a warm start
            (``zero_init_guess=False``) to try to rescue. Each retry uses
            the just-computed iterate plus the freshly-advected stress
            history, which is often enough for VEP at yield onset (Min/softmin
            kinks) to step off a bad Newton iterate. ``0`` preserves legacy
            behaviour (divergence is terminal). Typical useful value is 1.
            Only applies in the VE/VEP branch (``DFDt is not None``).
        """

        has_stress_history = self.Unknowns.DFDt is not None

        if has_stress_history:
            if timestep is None:
                raise ValueError(
                    "timestep is required for viscoelastic solve. "
                    "Call stokes.solve(timestep=dt)"
                )

            # dt_elastic must always equal the solve timestep. The constitutive
            # model's VE formulas (eta_eff, stress history terms) all reference
            # Parameters.dt_elastic. If it differs from the actual timestep,
            # the stress computation is inconsistent with the time integration.
            self.constitutive_model.Parameters.dt_elastic = timestep

            if order is None or order > self._order:
                order = self._order

            if _force_setup:
                self._needs_function_rewire = True

            # Re-setup when effective_order changes (DDt history ramp-up)
            _current_eff_order = self.constitutive_model.effective_order
            if _current_eff_order != self._prev_effective_order:
                self._needs_function_rewire = True
                self.constitutive_model._solver_is_setup = False
            self._prev_effective_order = _current_eff_order

            if not self.constitutive_model._solver_is_setup:
                self._needs_function_rewire = True
                self.DFDt.psi_fn = self.constitutive_model.flux.T

            if not self.is_setup:
                self._setup_pointwise_functions(verbose)
                self._setup_discretisation(verbose)
                self._setup_solver(verbose)

            # 1. ADVECT stress history along characteristics
            if uw.mpi.rank == 0 and verbose:
                print(f"Stokes solver - advect stress history", flush=True)

            self.DFDt.update_pre_solve(timestep, verbose=verbose, evalf=evalf,
                                       store_result=False)
            # Uniform pre-solve coefficient hook: VEP delegates to
            # _update_bdf_coefficients(); MaxwellExponentialFlowModel updates
            # α, φ on the DDt via _update_exp_coefficients(). No isinstance
            # checks at the solver layer.
            self.constitutive_model._update_history_coefficients()

            # 2. SOLVE
            if uw.mpi.rank == 0 and verbose:
                print(f"Stokes solver - solve", flush=True)

            super().solve(
                zero_init_guess,
                _force_setup=_force_setup,
                verbose=verbose,
                picard=picard,
                divergence_retries=divergence_retries,
            )

            # 3. PROJECT actual stress and SHIFT history
            if uw.mpi.rank == 0 and verbose:
                print(f"Stokes solver - store stress and shift history", flush=True)

            _advected_sigma_star = np.copy(self.DFDt.psi_star[0].array[...])

            if getattr(self.DFDt, '_psi_star_use_multicomponent', False):
                # Multi-component projection of flux → psi_star[0].
                #
                # The DFDt's source-snapshot machinery (enabled once in
                # _create_stress_history_ddt) intercepts psi_fn assignment
                # to substitute psi_star[0] symbols with a frozen
                # psi_snapshot variable, refreshed each step in
                # update_pre_solve. So the projection's compiled source
                # reads from psi_snapshot (not psi_star[0] itself) and is a
                # true one-shot Galerkin projection — no implicit
                # fixed-point iteration.
                self.DFDt._psi_star_projection_solver.smoothing = 0.0
                self.DFDt._psi_star_projection_solver.solve(verbose=verbose)
                # Fan flat result back to psi_star[0] tensor variable
                for k, (i, j) in enumerate(self.DFDt._psi_star_indep_indices):
                    vals = self.DFDt._psi_star_flat_var.array[:, 0, k]
                    self.DFDt.psi_star[0].array[:, i, j] = vals
                    if i != j:
                        self.DFDt.psi_star[0].array[:, j, i] = vals
            else:
                self.DFDt._psi_star_projection_solver.uw_function = self.constitutive_model.flux
                self.DFDt._psi_star_projection_solver.smoothing = 0.0
                self.DFDt._psi_star_projection_solver.solve(verbose=verbose)

            for i in range(self.DFDt.order - 1, 0, -1):
                if i == 1:
                    self.DFDt.psi_star[i].array[...] = _advected_sigma_star
                else:
                    self.DFDt.psi_star[i].array[...] = self.DFDt.psi_star[i - 1].array[...]

            self.DFDt.update_post_solve(timestep, verbose=verbose, evalf=evalf)

            # Uniform post-solve hook for any extra integrator-state storage.
            # VEP: no-op. ETD-2 / MaxwellExponentialFlowModel: refresh
            # forcing_star with current ε̇^{n+1} so the next step's history
            # term has access to ε̇ⁿ.
            self.constitutive_model._update_history_post_solve()

            self.is_setup = True
            self.constitutive_model._solver_is_setup = True

        else:
            # Plain Stokes — no stress history
            super().solve(
                zero_init_guess,
                _force_setup=_force_setup,
                verbose=verbose,
                picard=picard,
                divergence_retries=divergence_retries,
            )

    @property
    def tau(self):
        r"""Deviatoric stress from the most recent solve.

        When stress history is active (VEP), returns ``psi_star[0]`` which
        contains the actual projected stress. Otherwise falls through to the
        base class lazy projection.
        """
        if self.Unknowns.DFDt is not None:
            return self.DFDt.psi_star[0]
        return super().tau

    # =========================================================================
    # PETSc Residual Templates
    # These define the weak form terms assembled by PETSc's finite element system.
    # F0/F1 are velocity equation terms, PF0 is the pressure/continuity equation.
    # =========================================================================

    F0 = Template(
        r"\mathbf{f}_0\left( \mathbf{u} \right)",
        lambda self: -self.bodyforce.sym,
        r"""Velocity equation body force term (pointwise).

        The $\mathbf{f}_0$ term represents external body forces acting on the
        fluid, typically gravity. This is the negative of the ``bodyforce`` parameter.
        """,
    )

    F1 = Template(
        r"\mathbf{F}_1\left( \mathbf{u} \right)",
        lambda self: (
            self.stress
            + self.penalty * self.constitutive_model.K * self.div_u * sympy.eye(self.mesh.dim)
        ),
        r"""Velocity equation flux/stress term (pointwise).

        The $\mathbf{F}_1$ tensor represents the stress response of the fluid,
        combining deviatoric stress $\boldsymbol{\tau}$, pressure $p$,
        and the **viscosity-scaled** augmented-Lagrangian penalty term
        $\lambda\,\mu\,(\nabla\cdot\mathbf{u})\,\mathbf{I}$ for weak
        incompressibility. The penalty is multiplied by the local viscosity
        $\mu$ (``constitutive_model.K``) so that, under spatially-variable
        viscosity, the ratio penalty/$\mu$ stays uniform — a bare constant
        would over-stiffen low-viscosity regions and lock the velocity there.
        """,
    )

    PF0 = Template(
        r"\mathbf{h}_0\left( \mathbf{p} \right)",
        lambda self: sympy.Matrix((self.constraints)),
        r"""Pressure equation constraint term (continuity).

        The $h_0$ term enforces the incompressibility constraint
        $\nabla \cdot \mathbf{u} = 0$. Additional constraints can be
        added via ``add_condition()``.
        """,
    )

    # deprecated
    @timing.routine_timer_decorator
    def stokes_problem_description(self):
        """Build residual terms for Stokes FEM assembly (deprecated)."""
        # f0 residual term
        self._u_f0 = self.F0.sym

        # f1 residual term
        self._u_f1 = self.F1.sym

        # p0 residual term
        self._p_f0 = self.PF0.sym

        return

    @property
    def CM_is_setup(self):
        """Whether the constitutive model is configured for this solver."""
        return self._constitutive_model._solver_is_setup

    @property
    def strainrate(self):
        r"""Symmetric strain rate tensor from velocity gradients.

        The strain rate tensor :math:`\dot{\varepsilon}` is computed as:

        .. math::
            \dot{\varepsilon}_{ij} = \frac{1}{2}\left(\frac{\partial u_i}{\partial x_j}
            + \frac{\partial u_j}{\partial x_i}\right)

        Returns
        -------
        sympy.Matrix
            Symmetric tensor of shape ``(dim, dim)`` where ``dim`` is the
            mesh dimensionality.

        See Also
        --------
        strainrate_1d : Voigt notation (vector) form.
        stress_deviator : Deviatoric stress computed from strain rate.
        """
        return sympy.Matrix(self.mesh.vector.strain_tensor(self.Unknowns.u.sym))

    @property
    def strainrate_1d(self):
        r"""Strain rate in Voigt notation (vector form).

        Converts the symmetric strain rate tensor to Voigt notation for
        use in constitutive model calculations. In 2D, returns a 3-component
        vector; in 3D, a 6-component vector.

        Returns
        -------
        sympy.Matrix
            Strain rate in Voigt notation.

        See Also
        --------
        strainrate : Full tensor form.
        """
        return uw.maths.tensor.rank2_to_voigt(self.strainrate, self.mesh.dim)

    @property
    def strainrate_star_1d(self):
        r"""Historical strain rate in Voigt notation (for viscoelastic models).

        Used in viscoelastic formulations where the stress depends on
        both current and historical strain rates.

        Returns
        -------
        sympy.Matrix
            Historical strain rate in Voigt notation.

        See Also
        --------
        strainrate_1d : Current strain rate.
        """
        return uw.maths.tensor.rank2_to_voigt(self.strainrate_star, self.mesh.dim)

    @property
    def stress_deviator(self):
        r"""Deviatoric stress tensor from the constitutive model.

        The deviatoric stress :math:`\boldsymbol{\tau}` is the traceless part
        of the stress tensor, computed by the constitutive model from the
        strain rate:

        .. math::
            \boldsymbol{\tau} = \boldsymbol{\eta} : \dot{\boldsymbol{\varepsilon}}

        For a Newtonian fluid: :math:`\boldsymbol{\tau} = 2\eta\dot{\boldsymbol{\varepsilon}}`

        Returns
        -------
        sympy.Matrix
            Deviatoric stress tensor of shape ``(dim, dim)``.

        See Also
        --------
        stress : Total stress (deviatoric + pressure).
        constitutive_model : Provides the viscosity relationship.
        """
        return self.constitutive_model.flux

    @property
    def stress_deviator_1d(self):
        r"""Deviatoric stress in Voigt notation.

        Returns
        -------
        sympy.Matrix
            Deviatoric stress in Voigt notation.

        See Also
        --------
        stress_deviator : Full tensor form.
        """
        return uw.maths.tensor.rank2_to_voigt(self.stress_deviator, self.mesh.dim)

    @property
    def stress(self):
        r"""Total Cauchy stress tensor.

        The total stress combines the deviatoric stress and pressure:

        .. math::
            \boldsymbol{\sigma} = \boldsymbol{\tau} - p\mathbf{I}

        where :math:`\boldsymbol{\tau}` is the deviatoric stress and
        :math:`p` is the pressure (positive in compression).

        Returns
        -------
        sympy.Matrix
            Total stress tensor of shape ``(dim, dim)``.

        See Also
        --------
        stress_deviator : Deviatoric (traceless) part.
        """
        return self.stress_deviator - sympy.eye(self.mesh.dim) * (self.p.sym[0])

    @property
    def stress_1d(self):
        r"""Total stress in Voigt notation.

        Returns
        -------
        sympy.Matrix
            Total stress in Voigt notation.

        See Also
        --------
        stress : Full tensor form.
        """
        return uw.maths.tensor.rank2_to_voigt(self.stress, self.mesh.dim)

    @property
    def div_u(self):
        r"""Velocity divergence.

        For incompressible flow, this should be zero:

        .. math::
            \nabla \cdot \mathbf{u} = 0

        Returns
        -------
        sympy.Expr
            Scalar divergence expression.

        Notes
        -----
        Non-zero divergence indicates compressibility or mass sources/sinks.
        """
        return self.mesh.vector.divergence(self.Unknowns.u.sym)

    @property
    def constraints(self):
        r"""Constraint equation for the saddle-point system.

        By default, this is the incompressibility constraint
        :math:`\nabla \cdot \mathbf{u} = 0`. Can be modified for
        compressible or other constrained formulations.

        Returns
        -------
        sympy.Expr
            Constraint expression.
        """
        return self._constraints

    @constraints.setter
    def constraints(self, constraints_matrix):
        """Set the constraint equation (e.g., incompressibility)."""
        self._is_setup = False
        symval = sympify(constraints_matrix)
        self._constraints = symval

    @property
    def bodyforce(self):
        r"""Body force vector (source term).

        The volumetric body force :math:`\mathbf{f}` appears on the
        right-hand side of the momentum equation:

        .. math::
            -\nabla \cdot \boldsymbol{\sigma} = \mathbf{f}

        Common examples include gravity (:math:`\rho\mathbf{g}`) or
        buoyancy forces.

        Returns
        -------
        UWexpression
            Body force vector expression.
        """
        return self._bodyforce

    @bodyforce.setter
    def bodyforce(self, value):
        """Set the body force vector (e.g., gravity, buoyancy)."""
        self._needs_function_rewire = True
        if isinstance(value, uw.function.expressions.UWexpression):
            self._bodyforce.sym = value.sym
        else:
            # Convert UWQuantity components to their NON-DIMENSIONAL value before
            # Matrix creation — the body force feeds the non-dimensional solver
            # (see the ND<->units boundary contract / #282). The component may be
            # symbolic (e.g. a buoyancy ρα g (T − T_ref) carries the T field), so
            # non_dimensionalise yields a (1×1) array/Matrix whose element we
            # extract. (The previous code called item._sympify_(), which does not
            # exist on UWQuantity; _sympy_ raises on a dimensional quantity.)
            converted_value = []
            for item in value:
                if isinstance(item, uw.function.quantities.UWQuantity):
                    nd = uw.non_dimensionalise(item)
                    sympified = nd.magnitude if hasattr(nd, "magnitude") else nd
                    # If the non-dimensional value is a 1x1 array/Matrix, extract
                    # the scalar element.
                    if hasattr(sympified, "shape") and tuple(sympified.shape) == (1, 1):
                        converted_value.append(sympified[0, 0])
                    else:
                        converted_value.append(sympified)
                else:
                    converted_value.append(item)
            self._bodyforce.sym = sympy.Matrix(converted_value)

    def set_pressure_gauge(self, boundary, reference=0.0):
        r"""Pin the pressure gauge by removing the surface-mean pressure each iteration.

        On enclosed / all-Dirichlet-velocity problems the pressure is determined
        only up to an additive constant (a constant null space). This registers a
        per-iteration callback (see :meth:`add_update_callback`) that subtracts the
        mean pressure over ``boundary`` from the whole pressure field at every
        nonlinear iteration, fixing a *specific, physical* gauge:

        .. math:: \frac{1}{|\Gamma|}\int_\Gamma p \, dS = \texttt{reference}

        e.g. ``stokes.set_pressure_gauge("Top")`` makes the mean pressure on the
        top surface zero. This is an alternative (or complement) to a constant
        pressure null space when you want the gauge tied to a surface rather than
        to an arbitrary constant chosen by the solver.

        Parameters
        ----------
        boundary : str
            Mesh boundary label over which the mean pressure is fixed.
        reference : float, default 0.0
            Target mean pressure on ``boundary``.

        Returns
        -------
        the registered callback (so it can be identified/removed later).
        """
        p = self.Unknowns.p
        area = uw.maths.BdIntegral(self.mesh, 1.0, boundary).evaluate()
        p_surface_integral = uw.maths.BdIntegral(self.mesh, p.sym[0, 0], boundary)

        def _pressure_gauge(solver, iteration):
            mean = p_surface_integral.evaluate() / area
            p.data[...] -= (mean - reference)

        return self.add_update_callback(_pressure_gauge)

    @property
    def saddle_preconditioner(self):
        r"""Pressure Schur-complement preconditioner — **usually leave unset**.

        Approximates the Schur complement
        :math:`\mathbf{S} \approx \mathbf{B}\mathbf{A}^{-1}\mathbf{B}^T \sim 1/\eta`
        (inverse viscosity). When this is unset (the default, ``None``) the solver
        automatically uses :math:`1/\eta = 1/`\ ``constitutive_model.K`` — the
        correct local-viscosity scaling for any constitutive model. Setting it
        explicitly to ``1/viscosity`` is therefore **redundant** (and only a chance
        to supply an inconsistent viscosity); just leave it at the default.

        Returns
        -------
        sympy.Expr or None
            Preconditioner expression, or ``None`` to use the automatic
            ``1/constitutive_model.K``.

        Notes
        -----
        This is an **advanced override**, useful only when the automatic
        ``1/K`` is not the right Schur scaling — e.g. an anisotropic/tensorial
        ``K``, or deliberate preconditioner tuning. (``Stokes_Constrained`` builds
        its Schur preconditioner automatically and does not expose this property.)
        """
        return self._saddle_preconditioner

    @saddle_preconditioner.setter
    def saddle_preconditioner(self, value):
        """Set the Schur complement preconditioner."""
        self._needs_function_rewire = True
        symval = sympify(value)
        self._saddle_preconditioner = symval

    @property
    def penalty(self):
        r"""Augmented Lagrangian penalty parameter (dimensionless, viscosity-scaled).

        The penalty adds a **viscosity-weighted** grad-div term to the weak form
        that penalizes non-zero divergence:

        .. math::
            \lambda \int \mu\,(\nabla \cdot \mathbf{u})(\nabla \cdot \mathbf{v}) \, dV

        where :math:`\mu` is the local viscosity (``constitutive_model.K``). The
        :math:`\mu`-weighting keeps the effective penalty proportional to the
        local stress scale, so the ratio penalty/:math:`\mu` is uniform across a
        spatially-variable viscosity field. (A bare *constant* penalty would be
        huge relative to the stress in low-:math:`\mu` regions and negligible in
        high-:math:`\mu` regions — over-stiffening the former into velocity
        locking. See the design note ``CONSTRAINED_FREESLIP_MULTIPLIER``.) So the
        parameter here is a **dimensionless** base of :math:`O(1)`.

        Returns
        -------
        UWexpression
            Dimensionless augmented-Lagrangian penalty base (typically :math:`O(1)`).

        Notes
        -----
        Set to zero (the default) for a standard saddle-point Stokes solve; the
        ``saddle_preconditioner`` already conditions the pressure Schur, so the
        penalty is usually unnecessary for convergence.

        **Pressure correction.** This term is part of the *operator*, so when it
        is non-zero the recovered pressure ``p`` is the Lagrange multiplier, not
        the mechanical pressure. The total isotropic stress is
        :math:`-p + \lambda\,\mu\,(\nabla\cdot\mathbf{u})`, so the mechanical
        pressure is

        .. math::
            p_\text{mech} = p - \lambda\,\mu\,(\nabla\cdot\mathbf{u}).

        At convergence :math:`\nabla\cdot\mathbf{u}\to 0` so the two agree, but
        *pointwise* they differ by the penalty term (≈ a couple of percent of
        :math:`|p|` at :math:`\lambda=O(1)`). For a pressure-dependent
        constitutive law (yield, density, rheology), use :math:`p_\text{mech}`;
        for visualisation or weak pressure dependence the raw ``p`` is adequate.

        References
        ----------
        Glowinski, R., & Le Tallec, P. (1989). *Augmented Lagrangian and
        Operator-Splitting Methods in Nonlinear Mechanics*. SIAM.
        https://doi.org/10.1137/1.9781611970838
        """
        return self._penalty

    @penalty.setter
    def penalty(self, value):
        """Set the augmented Lagrangian penalty parameter."""
        self._needs_function_rewire = True
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

        .. math::
            \delta t_i = h_i / \|\mathbf{v}_i\|

        where $h_i$ is the element radius and $\mathbf{v}_i$ is the velocity at the element
        centroid, then returns the global minimum. This is more accurate than
        using global max velocity with global min element size, especially for
        non-uniform meshes with spatially varying velocity.

        Returns:
            Pint Quantity or float: The advective timestep with physical time units
            if a model with reference scales is available, otherwise nondimensional.
        """

        from mpi4py import MPI

        # Velocity at element centroids (consistent with AdvDiff)
        vel = _centroid_velocities_nd(self.u.sym, self.mesh)

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
    r"""Viscoelastic Stokes solver (backward-compatibility wrapper).

    .. deprecated::
        Use ``uw.systems.Stokes`` directly with a
        ``ViscoElasticPlasticFlowModel`` constitutive model. The Stokes
        solver now creates stress history infrastructure automatically
        when the constitutive model is assigned (the lazy-creation
        pathway also reads ``stress_history_ddt_kwargs`` from the model
        — required for ``integrator='etd'`` to allocate
        ``forcing_star``). VE_Stokes pre-creates the DDt at solver
        ``__init__`` time, before the model exists, so it can't see
        those kwargs and is incompatible with ``integrator='etd'``.

    Migration: replace::

        stokes = uw.systems.VE_Stokes(mesh, velocityField=v, pressureField=p, order=2)
        stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel

    with::

        stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
        stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
            stokes.Unknowns, order=2,
        )

    Parameters
    ----------
    mesh : Mesh
        The computational mesh.
    order : int, default=2
        BDF time integration order for stress history.
    **kwargs
        All other arguments are passed to :class:`SNES_Stokes`.
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
        DuDt: Union[SemiLagrangian_DDt, Lagrangian_DDt] = None,
        DFDt: Union[SemiLagrangian_DDt, Lagrangian_DDt] = None,
    ):
        import warnings
        warnings.warn(
            "VE_Stokes is deprecated. Use uw.systems.Stokes(...) directly and "
            "assign the constitutive model afterwards — the Stokes solver creates "
            "DDt infrastructure lazily when the model is assigned, and the lazy "
            "path correctly forwards stress_history_ddt_kwargs from the model "
            "(required for integrator='etd' to allocate forcing_star). "
            "VE_Stokes pre-creates DDt at __init__ time, before the model exists, "
            "so it cannot use ETD-2. See the class docstring for migration.",
            DeprecationWarning, stacklevel=2,
        )
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

        # Pre-create DFDt so it's available before constitutive model is set
        self._create_stress_history_ddt(order=order)

        return

    @property
    def delta_t(self):
        """Elastic timestep from the constitutive model."""
        return self.constitutive_model.Parameters.dt_elastic


class _BlockConstraintBC:
    """Bookkeeping for one in-saddle-point multiplier constraint.

    Holds the multiplier field ``lam`` (a full-domain scalar MeshVariable
    registered as an extra DM field), the prescribed normal velocity ``g``, the
    (symbolic, row-vector) constraint normal, and the mutable PETSc bookkeeping
    (``petsc_id``/``label_val``) plus the JIT-compiled boundary functions
    (``fns``) populated during assembly. Mirrors the natural-BC namedtuple but
    stays a mutable object so the Cython assembly can stamp ids onto it.
    """

    __slots__ = ("boundary", "g", "normal", "lam", "augmentation",
                 "petsc_id_u", "petsc_id_h", "label_val", "fns")

    def __init__(self, boundary, g, normal, lam, augmentation):
        self.boundary = boundary
        self.g = g
        self.normal = normal
        self.lam = lam
        # Augmented-Lagrangian parameter r: a penalty r(n·u−g)·n added to the
        # u-row, giving a uu boundary stiffness r·(n⊗n) that conditions the
        # [p,h] Schur complement WITHOUT biasing the multiplier (the h-row is
        # still the exact constraint). 0 disables it (bare KKT).
        self.augmentation = augmentation
        # PETSc needs ONE boundary entry per (boundary, test-field): each entry
        # evaluates only its registered field's residual + that field's Jacobian
        # row. So a constraint registers the boundary twice — for field 0 (the
        # u-row traction h·n and the uh Jacobian) and for the multiplier field
        # (the h-row constraint n·u−g and the hu Jacobian).
        self.petsc_id_u = -1
        self.petsc_id_h = -1
        self.label_val = -1
        self.fns = {}


class SNES_Stokes_Constrained(SNES_Stokes):
    r"""
    Stokes solver that enforces :math:`\mathbf{u}\cdot\mathbf{n} = g` on a
    boundary via a Lagrange multiplier living **inside** the saddle-point
    system — a single coupled solve, not an outer loop.

    A scalar multiplier field :math:`h` is added as a third DM field and grouped
    with pressure into a 2-way :math:`\mathbf{u}\,|\,[p,h]` Schur split. The
    coupled block system is

    .. math::

        \begin{bmatrix} A & B^{T} & C^{T} \\ B & 0 & 0 \\ C & 0 & \varepsilon M
        \end{bmatrix}
        \begin{bmatrix} \mathbf{u} \\ p \\ h \end{bmatrix} =
        \begin{bmatrix} \mathbf{f} \\ 0 \\ g \end{bmatrix},

    where :math:`C = \int_\Gamma (\mathbf{n}\cdot\mathbf{v})\,\psi` couples the
    multiplier to the boundary normal velocity, and :math:`\varepsilon M =
    \varepsilon\int_\Omega h\,\psi` is an interior screening mass that
    de-singularises the otherwise-empty interior :math:`h` block (it does not
    bias the boundary multiplier). At convergence :math:`h|_\Gamma = -\,
    \mathbf{n}\cdot\boldsymbol{\sigma}\cdot\mathbf{n}` is the boundary normal
    traction = dynamic topography; access it via :meth:`multiplier` /
    :meth:`topography`.

    The constraint is enforced in one coupled solve (no outer iteration). The
    augmented-Lagrangian term conditions the :math:`[p,h]` Schur complement
    without biasing the multiplier, and the interior multiplier DOFs are reduced
    away so the solved block is boundary-sized.

    Runs in parallel: the interior-multiplier reduction is rank-local section
    surgery so the global system, velocity solve, and gauge-fixed topography are
    partition-independent (validated in
    ``tests/parallel/test_1063_constrained_freeslip_parallel.py``). On an
    **enclosed** problem (a pressure null space is active) the constant pressure
    and constant multiplier are gauge-free, and the solver lands on a
    partition-dependent level for each. To keep the raw **pressure** reproducible
    across ranks the solver pins the surface-mean pressure automatically (see
    :attr:`auto_pressure_gauge`); the raw **multiplier** keeps its own gauge level,
    so read dynamic topography through :meth:`topography` with ``reference="mean"``
    (gauge-invariant). See
    ``docs/developer/design/CONSTRAINED_FREESLIP_MULTIPLIER.md``.

    See Also
    --------
    SNES_Stokes : The unconstrained saddle-point solver this extends.
    """

    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        velocityField: Optional[uw.discretisation.MeshVariable] = None,
        pressureField: Optional[uw.discretisation.MeshVariable] = None,
        degree: Optional[int] = 2,
        p_continuous: Optional[bool] = True,
        verbose: Optional[bool] = False,
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

        # In-saddle multiplier constraints (see add_constraint_bc). Empty for an
        # ordinary Stokes solve, so the base assembly is unaffected.
        self._block_constraint_bcs = []

        # Guarded automatic pressure gauge (see _install_auto_pressure_gauge). On an
        # enclosed constrained problem the constant pressure and constant
        # multiplier are gauge-free; the solver lands on a partition-dependent
        # level for each, so the raw fields are not reproducible across ranks.
        # When a pressure null space is active we pin the surface-mean pressure,
        # making the raw PRESSURE reproducible at zero cost to the velocity (the
        # raw multiplier keeps its own gauge — read topography via reference="mean").
        # Set to False to opt out.
        self.auto_pressure_gauge = True
        self._auto_gauge_callback = None

        # Default the grouped [p, lambda] Schur preconditioner to `selfp` (the base
        # Stokes default is `a11`). The constraint Schur complement
        # S_lambda = C A^-1 C^T needs a preconditioner built from the actual operator
        # blocks; selfp forms S ~ A11 - A10 diag(A00)^-1 A01, whose lambda-lambda corner
        # is -C diag(A)^-1 C^T = the true constraint Schur, automatically. This cracks
        # strong boundary-viscosity-contrast walls that the bare `a11` mass cannot, makes
        # the augmentation optional on moderate contrast, and keeps the recovered
        # multiplier (= dynamic topography) clean at small augmentation. Override via
        # `solver.petsc_options["pc_fieldsplit_schur_precondition"] = "a11"` if desired.
        self.petsc_options["pc_fieldsplit_schur_precondition"] = "selfp"

        # Flexible outer Krylov (fgmres) is REQUIRED, not a tuning choice. The
        # grouped [p, lambda] Schur preconditioner runs inner *iterative* sub-solves
        # (velocity multigrid + the Schur KSP), so the preconditioner operator
        # varies from one outer iteration to the next — a variable/nonlinear
        # preconditioner. A non-flexible Krylov method (the base's gmres) is
        # invalid for that: the preconditioned residual it minimises false-converges
        # (it reported CONVERGED_RTOL in ~2 iterations while the TRUE residual
        # ||r||/||b|| blew up to ~10^3), landing on a wrong, partition-dependent
        # answer (the ~0.4% serial-vs-parallel velocity spread, and the failure to
        # reproduce the Zhong response). fgmres handles the variable preconditioner
        # correctly: the true residual converges and the solve is
        # partition-independent. Override via petsc_options["ksp_type"] if needed.
        self.petsc_options["ksp_type"] = "fgmres"

        # Judge outer convergence on the TRUE (unpreconditioned) residual. With a
        # variable preconditioner the preconditioned-residual norm is not a fixed
        # inner product and false-converges (it can report convergence while the
        # true residual is still large), so the default preconditioned-norm test
        # would stop the solve early even with a tight rtol — re-introducing the
        # partition-dependent velocity. The unpreconditioned norm costs one extra
        # matvec per iteration and makes the stopping test honest.
        self.petsc_options["ksp_norm_type"] = "unpreconditioned"

        # Bound the outer iteration count. The well-set-up solve converges in a
        # handful of outer iterations (~6 on the 3-D shell at cellSize 1/8); this
        # generous cap just guarantees a pathological case fails loudly
        # (DIVERGED_ITS) rather than silently grinding. Raise it for genuinely
        # hard problems via petsc_options["ksp_max_it"].
        self.petsc_options["ksp_max_it"] = 100

        # Tighten the Eisenstat-Walker adaptive linear tolerance (the base saddle
        # point enables snes_ksp_ew with the PETSc default initial/max rtol 0.3).
        # EW's loose-early heuristic is right for a NONLINEAR Newton solve, but the
        # constrained solver is normally run linear (snes_type=ksponly), where EW
        # then caps the single outer solve at rtol 0.3 — one outer iteration, true
        # residual ~1e-5 relative. The augmented saddle point is ill-conditioned
        # (augmentation ~1e4), so that loose residual is amplified into a ~0.1%
        # partition-dependent velocity (GAMG's per-partition aggregation makes the
        # preconditioner partition-dependent, which only cancels once the TRUE
        # residual is driven down). Pinning EW's initial = max rtol to the solver
        # tolerance makes the outer fgmres iterate until genuinely converged, so
        # the velocity is partition-independent to round-off. Kept in sync by the
        # `tolerance` setter below.
        self.petsc_options["snes_ksp_ew_rtol0"] = self._tolerance * 1.0e-1
        self.petsc_options["snes_ksp_ew_rtolmax"] = self._tolerance * 1.0e-1
        return

    @property
    def tolerance(self):
        """Solver tolerance (see :class:`SNES_Stokes_SaddlePt.tolerance`).

        Overridden so that, in addition to ``snes_rtol`` / ``ksp_rtol`` /
        ``ksp_atol``, the Eisenstat-Walker initial and max relative tolerances are
        pinned to ``tolerance * 0.1`` — otherwise EW's default (0.3) under-solves
        the ill-conditioned augmented constrained system on a linear solve and the
        velocity becomes partition-dependent (see ``__init__``).
        """
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        self._tolerance = value
        self.petsc_options["snes_rtol"] = value
        self.petsc_options["ksp_rtol"] = value * 1.0e-1
        self.petsc_options["ksp_atol"] = value * 1.0e-6
        self.petsc_options["snes_ksp_ew_rtol0"] = value * 1.0e-1
        self.petsc_options["snes_ksp_ew_rtolmax"] = value * 1.0e-1

    def solve(self, *args, **kwargs):
        """Solve the constrained Stokes system (see :meth:`SNES_Stokes.solve`).

        Installs the guarded automatic pressure gauge (if eligible) before
        delegating to the base solve, so the raw pressure and multiplier are
        partition-reproducible by construction on enclosed problems.
        """
        self._warn_if_monolithic_direct()
        self._install_auto_pressure_gauge()
        return super().solve(*args, **kwargs)

    def _warn_if_monolithic_direct(self):
        """Warn that a monolithic direct solve of the constrained system is a
        serial diagnostic only.

        Setting ``pc_type = lu``/``cholesky`` factorises the whole
        :math:`[\\,u, p, h\\,]` saddle point as one block. On the constrained
        system this is a KNOWN-BAD path: it does not reproduce the validated
        grouped-Schur (``selfp``) velocity response (e.g. surface velocity ~4e-3
        vs the ~1e-2 Zhong reference on the 3-D shell) AND it segfaults in
        parallel (the indefinite KKT factorisation is not robust here). The
        grouped :math:`\\mathbf{u}\\,|\\,[p,h]` field-split is the supported path;
        use ``pc_type = lu`` only as a serial cross-check, knowing the response is
        not the benchmark reference.
        """
        if not self._block_constraint_bcs:
            return
        try:
            pc_type = self.petsc_options["pc_type"]
        except KeyError:
            return
        if pc_type in ("lu", "cholesky"):
            import warnings
            warnings.warn(
                "Stokes_Constrained: a monolithic direct solve "
                f"(pc_type='{pc_type}') of the constrained saddle point is a "
                "serial diagnostic only — it does not reproduce the validated "
                "grouped-Schur velocity response and segfaults in parallel. Use "
                "the default field-split (u | [p,h]) path for production runs.",
                stacklevel=2,
            )

    def _install_auto_pressure_gauge(self):
        """Pin the (p, h) gauge consistently on an enclosed constrained problem.

        Conservative — installs ``set_pressure_gauge`` on the first constraint
        boundary (``reference=0``) only when ALL of the following hold, otherwise
        it is a no-op and the solve path is unchanged:

        * ``auto_pressure_gauge`` is left True (the default),
        * we have not already installed it (idempotent across re-solves),
        * at least one multiplier constraint is registered,
        * a pressure null space is active (``petsc_use_nullspace`` / enclosed) —
          this is the gauge-freedom flag; with it off the pressure is determined
          and must not be shifted,
        * no pressure Dirichlet BC already pins the gauge,
        * the user has registered no update callback of their own (e.g. an
          explicit ``set_pressure_gauge`` or a smoother) — we never clobber it.

        Pinning the surface-mean pressure makes the raw **pressure** field
        partition-reproducible (measured: the enclosed-shell mean pressure goes
        from a ~10% serial-vs-np8 spread to the assembly floor, ~0.4%). It is
        physics-neutral: the velocity is bit-identical with and without the pin.
        It does NOT fix the raw **multiplier** ``h`` — the constant multiplier is
        an independent gauge freedom the pressure pin does not touch — so dynamic
        topography must still be read with ``topography(..., reference="mean")``,
        which is gauge-invariant by construction.
        """
        if not self.auto_pressure_gauge:
            return
        if self._auto_gauge_callback is not None:
            return
        if not self._block_constraint_bcs:
            return
        if not self._petsc_use_pressure_nullspace:
            return
        if self._pressure_dirichlet_bcs():
            return
        if self._snes_update_callbacks:
            return
        boundary = self._block_constraint_bcs[0].boundary
        self._auto_gauge_callback = self.set_pressure_gauge(boundary, 0.0)

    @property
    def saddle_preconditioner(self):
        """Not used by ``Stokes_Constrained`` — the Schur preconditioner is built
        automatically.

        The grouped :math:`[p,\\lambda]` Schur preconditioner is formed by
        ``selfp`` from the operator blocks, and the pressure mass it needs is the
        ``1/viscosity`` (``1/constitutive_model.K``) term supplied automatically.
        There is nothing for the user to set; this property is inert and assigning
        to it raises. (The base :class:`SNES_Stokes` keeps a settable
        ``saddle_preconditioner`` as an advanced override.)
        """
        return None

    @saddle_preconditioner.setter
    def saddle_preconditioner(self, value):
        raise AttributeError(
            "Stokes_Constrained does not use `saddle_preconditioner`: the Schur "
            "preconditioner is built automatically (selfp + the 1/viscosity mass "
            "from constitutive_model.K). Remove this assignment."
        )

    def _viscosity_scale(self):
        """A representative scalar viscosity, for sizing the interior screening."""
        try:
            mu = self.constitutive_model.Parameters.shear_viscosity_0
            return float(mu)
        except (TypeError, ValueError, AttributeError):
            return 1.0

    def add_constraint_bc(self, conds=None, boundary=None, normal=None, screening=None,
                          augmentation=None, augmentation_base=1.0e4, degree=None,
                          g=None):
        r"""Register a multiplier-enforced normal-velocity constraint on ``boundary``.

        Adds a scalar multiplier field ``h`` coupled into the saddle-point system
        so that :math:`\mathbf{u}\cdot\mathbf{n}=\mathrm{conds}` is enforced on
        ``boundary`` in the coupled solve; at convergence ``h`` on the boundary
        is the normal traction (dynamic topography), recoverable via
        :meth:`multiplier` / :meth:`topography`.

        Parameters
        ----------
        conds : float or sympy expression, optional
            Prescribed normal velocity :math:`\mathbf{u}\cdot\mathbf{n}`,
            in the canonical value-first BC order. Default ``None`` means zero
            (free-slip).
        boundary : str
            Mesh boundary label (e.g. ``"Upper"``).
        normal : sympy matrix, optional
            Row-vector constraint normal. Defaults to ``mesh.Gamma_P1``.
        screening : float or sympy expression, optional
            Interior screening coefficient :math:`\varepsilon` (de-singularises
            the interior multiplier DOFs). Defaults to ``1e-6``.
        augmentation : float or sympy expression, optional
            Augmented-Lagrangian parameter :math:`r`. Adds a penalty
            :math:`r(\mathbf{n}\cdot\mathbf{u}-g)\,\mathbf{n}` to the u-row,
            giving a ``uu`` boundary stiffness :math:`r\,(\mathbf{n}\otimes
            \mathbf{n})` that conditions the :math:`[p,h]` Schur complement
            **without biasing the multiplier** (the h-row is still the exact
            constraint). Defaults to ``augmentation_base · μ(x)`` (viscosity-
            weighted, mesh-independent). Pass ``0`` for the bare KKT system.
        augmentation_base : float, default 1e4
            Base multiple used when ``augmentation`` is not given. Accuracy is
            independent of this value (the multiplier carries the exact
            constraint); larger values reduce the iteration count up to a broad
            plateau, well below the roundoff limit.
        g : float or sympy expression, optional
            Deprecated keyword alias for ``conds`` (one DeprecationWarning).

        Returns
        -------
        h : MeshVariable
            The scalar multiplier field.

        Notes
        -----
        The legacy boundary-first call ``add_constraint_bc(boundary, g=...)``
        is detected conservatively (first positional argument a string while
        the second is not — a BC datum is never a string) and shimmed with one
        DeprecationWarning; see
        ``SolverBaseClass._value_first_bc_args``.
        """
        conds, boundary = self._value_first_bc_args(
            "add_constraint_bc", conds, boundary, alias=g)
        g = conds if conds is not None else 0.0
        # Parallel-safe: the interior-multiplier reduction
        # (_constrain_interior_multipliers_in_section) is rank-local section
        # surgery (it uses the distributed boundary label IS and iterates the
        # local chart), so the global system — and hence the velocity solve and
        # the gauge-invariant boundary traction — are partition-independent.
        # Validated bit-identical at np=1/2/4 (velocity L2 and mean-stripped
        # boundary topography) in
        # tests/parallel/test_1063_constrained_freeslip_parallel.py.
        # NOTE: on enclosed problems the constant pressure and constant multiplier
        # are gauge-free and land on a partition-dependent level. The automatic
        # pressure gauge (auto_pressure_gauge, see _install_auto_pressure_gauge) pins
        # the raw PRESSURE reproducibly, but the raw multiplier h keeps its own
        # gauge level — read dynamic topography via topography(..., reference="mean"),
        # which is gauge-invariant and partition-reproducible by construction.

        if not hasattr(self.mesh.boundaries, boundary):
            raise ValueError(
                f"'{boundary}' is not a boundary of this mesh. "
                f"Available: {[b.name for b in self.mesh.boundaries]}."
            )

        if normal is None:
            normal = self.mesh.Gamma_P1
        normal = sympy.Matrix(normal)
        if normal.shape[0] != 1:
            normal = normal.reshape(1, self.mesh.dim)

        if screening is None:
            # Small interior screening: de-singularises the interior h block
            # without biasing the boundary multiplier. A viscosity-aware scale
            # is chosen in a later milestone.
            screening = 1.0e-6

        g = sympy.sympify(g)

        if augmentation is None:
            # Viscosity-weighted augmentation r = augmentation_base · μ(x): keeps
            # the conditioning ratio uniform across viscosity contrasts (same
            # rationale as the outer-loop solver; r ∝ μ is mesh-independent).
            try:
                viscosity = self.constitutive_model.Parameters.shear_viscosity_0
                augmentation = augmentation_base * viscosity
            except (AttributeError, TypeError):
                augmentation = augmentation_base * self._viscosity_scale()
        augmentation = sympy.sympify(augmentation)

        idx = len(self._block_constraint_bcs)
        field_name = f"multiplier_{idx}"
        # Multiplier degree defaults to the velocity degree so its trace reaches
        # every velocity normal-trace DOF (no constraint floor on the P2 mid-edge
        # component). A lower degree trades a little constraint accuracy for far
        # fewer multiplier DOFs (cheaper [p,h] Schur block).
        h_degree = self._degree if degree is None else degree
        h = uw.discretisation.MeshVariable(
            f"H{self.instance_number}_{idx}",
            self.mesh,
            1,
            degree=h_degree,
        )
        h.data[:] = 0.0
        h._solver_field_name = field_name

        self._multipliers.append(h)
        self._multiplier_screening.append(screening)
        self.fields[field_name] = h
        # New DM field → the discretisation and solver must be rebuilt.
        self.is_setup = False
        self._needs_function_rewire = True

        cbc = _BlockConstraintBC(boundary, g, normal, h, augmentation)

        self._block_constraint_bcs.append(cbc)
        return h

    def multiplier(self, boundary):
        """Return the multiplier field for ``boundary`` (None if not constrained).

        After :meth:`solve`, the multiplier's boundary trace is the normal
        traction holding the constraint. Divide by :math:`\\Delta\\rho\\,g` for
        dynamic topography (see :meth:`topography`).
        """
        for cbc in self._block_constraint_bcs:
            if cbc.boundary == boundary:
                return cbc.lam
        return None

    def topography(self, boundary, buoyancy_scale=1.0, reference=None):
        r"""Dynamic topography expression :math:`h / (\Delta\rho\, g)` on ``boundary``.

        For an **enclosed** problem (no net normal flow through any boundary) the
        multiplier :math:`h` is determined only up to the :math:`[p,\lambda]` gauge
        constant, and the solver lands on a **partition-dependent representative**
        of it — the velocity and the *deviation* of :math:`h` are unaffected, but
        the absolute level of :math:`h` is not reproducible across ranks. For such
        problems pass ``reference="mean"`` to subtract the boundary mean and obtain
        a gauge-fixed, partition-independent topography. The default
        (``reference=None``) returns the raw multiplier — correct for problems with
        **no** gauge freedom (e.g. an open boundary), where the mean of :math:`h` is
        the physical mean traction and must NOT be removed.

        Note that the automatic pressure gauge (:attr:`auto_pressure_gauge`) fixes
        the raw *pressure* level but NOT the raw *multiplier* level (the constant
        multiplier is an independent gauge freedom). So on an enclosed problem the
        raw multiplier (``reference=None``) is still partition-dependent —
        ``reference="mean"`` is the gauge-invariant, partition-reproducible read
        for dynamic topography and is the recommended path.

        Parameters
        ----------
        boundary : str
            Constrained boundary label.
        buoyancy_scale : float, default 1.0
            Divide by :math:`\Delta\rho\,g` to convert traction to length.
        reference : {None, "mean"}, default None
            ``None`` returns the raw multiplier (correct when there is no gauge
            freedom). ``"mean"`` subtracts the boundary mean (gauge-fixed,
            reproducible) — use for enclosed problems.

        Notes
        -----
        ``reference="mean"`` evaluates two ``BdIntegral`` reductions immediately
        to compute the boundary mean, which are **collective** MPI operations —
        in parallel it must be called on every rank (do not guard it behind a
        single-rank branch). ``reference=None`` is a pure symbolic accessor with
        no reduction.
        """
        lam = self.multiplier(boundary)
        if lam is None:
            raise ValueError(f"No constraint registered on boundary '{boundary}'.")
        expr = lam.sym[0]
        if reference == "mean":
            # Subtract the boundary mean of h via parallel-safe surface integrals
            # (BdIntegral handles the cross-rank reduction); this fixes the gauge.
            blen = uw.maths.BdIntegral(
                mesh=self.mesh, fn=sympy.Integer(1), boundary=boundary).evaluate()
            hbar = uw.maths.BdIntegral(
                mesh=self.mesh, fn=lam.sym[0], boundary=boundary).evaluate() / blen
            expr = expr - hbar
        elif reference is not None:
            raise ValueError(
                f"reference must be 'mean' or None, got {reference!r}")
        return expr / buoyancy_scale


class _SmoothingLengthMixin:
    r"""Shared :math:`\alpha \leftrightarrow L` bookkeeping for the projection solvers.

    The projection solvers all implement the screened-Poisson smoother
    :math:`u - \nabla\!\cdot\!(\alpha\,\nabla u) = \tilde f`, whose natural
    filter scale is :math:`L = \sqrt{\alpha}` (Green's function
    :math:`\propto e^{-r/L}`). This mixin holds the single implementation of
    the ``smoothing`` (:math:`\alpha`, length²) setter and the unit-aware
    ``smoothing_length`` (:math:`L`, length) accessors — including the
    ``_smoothing_is_dimensional`` flag that lets a dimensional input
    round-trip as a Pint Quantity while plain-float input round-trips as a
    plain float. Subclasses keep their own property docstrings as thin
    wrappers delegating here.
    """

    def _set_smoothing(self, value):
        """Store the smoothing coefficient alpha (units length squared)."""
        self._needs_function_rewire = True
        self._smoothing = sympify(value)

    def _get_smoothing_length(self):
        r"""Return :math:`L = \sqrt{\alpha}` (unit-aware, see mixin docstring)."""
        s = self._smoothing
        try:
            sval = float(s)
        except (TypeError, ValueError):
            # Symbolic alpha (e.g. an expression): return the symbolic sqrt.
            return sympy.sqrt(s)
        if sval < 0:
            raise ValueError(
                f"smoothing is negative ({sval}); smoothing_length is undefined")
        L_nd = sval ** 0.5
        # Return a Pint Quantity only if the user set a dimensional value via
        # the setter; plain-float input round-trips as a plain float so the
        # meaning of the number the user passed is preserved.
        if getattr(self, "_smoothing_is_dimensional", False):
            return uw.scaling.dimensionalise(
                L_nd, uw.scaling.units.meter)
        return L_nd

    def _set_smoothing_length(self, L):
        r"""Store :math:`\alpha = L^2` from a unit-aware length :math:`L`."""
        self._needs_function_rewire = True
        # Unit-aware: a dimensional input (Pint Quantity / UnitAware) is
        # non-dimensionalised through the active scaling context and reduced to
        # a plain float (uw.non_dimensionalise returns a dimensionless
        # UWQuantity, which sympify can't square); a plain number is taken as
        # already non-dimensional.
        is_dim = hasattr(L, "magnitude") or hasattr(L, "units")
        if is_dim:
            L_nd = uw.non_dimensionalise(L)
            L_nd = float(getattr(L_nd, "magnitude", L_nd))
        else:
            L_nd = float(L)
        if L_nd < 0:
            raise ValueError(f"smoothing_length must be ≥ 0, got {L_nd}")
        self._smoothing_is_dimensional = is_dim
        self._smoothing = sympify(L_nd) ** 2


class SNES_Projection(_SmoothingLengthMixin, SNES_Scalar):
    r"""
    Scalar projection solver for mapping functions to mesh variables.

    Solves :math:`u = \tilde{f}` where :math:`\tilde{f}` is a function that
    can be evaluated within an element and :math:`u` is a mesh variable with
    associated shape functions.

    Typically used to obtain a continuous representation of a function not
    well-defined at mesh nodes (e.g., derivatives or flux components). More
    broadly, it is a projection from one basis to another.

    Strong form (the screened-Poisson smoother)
    -------------------------------------------

    The projection is implemented by solving

    .. math::

        u - \nabla \cdot \left( \alpha \nabla u \right) = \tilde{f},

    or equivalently, in the more familiar Helmholtz form

    .. math::

        u - \alpha \, \nabla^{2} u \;=\; \tilde{f}.

    With :math:`\alpha = 0` this is a pure pointwise L2 projection of
    :math:`\tilde f` onto the discrete space of :math:`u`. With
    :math:`\alpha > 0` it is a *screened-Poisson smoother*: the equation
    enforces a balance between fidelity to :math:`\tilde f` and curvature of
    :math:`u`. The natural length scale that emerges from this balance is

    .. math::

        L \;=\; \sqrt{\alpha},

    so :math:`\alpha` has dimensions of **length squared**. The free-space
    Green's function of the operator decays as
    :math:`\exp(-r/L)` (in 2D it is
    :math:`G(r) \propto K_{0}(r/L)/L^{2}`), so the solution behaves like a
    Gaussian-like convolution of :math:`\tilde f` of width :math:`L` —
    obtained implicitly by one elliptic solve, without ever assembling
    the kernel. Features in :math:`\tilde f` of scale much smaller than
    :math:`L` are attenuated; features much larger than :math:`L` pass
    through essentially unchanged.

    Weak form
    ---------

    Multiplying by a test function :math:`v` and integrating by parts gives
    the symmetric weak form actually assembled by PETSc:

    .. math::

        \int_\Omega (u - \tilde f)\, v \; + \;
        \int_\Omega \alpha \, \nabla u \cdot \nabla v
        \;=\; 0,

    which is exactly minimising
    :math:`\tfrac{1}{2}\!\int (u-\tilde f)^2 + \tfrac{\alpha}{2}\!\int
    |\nabla u|^2` — a Tikhonov-regularised L2 projection.

    Setting the smoothing length
    ----------------------------

    Two equivalent accessors are provided:

    * :attr:`smoothing` — set the coefficient :math:`\alpha` directly
      (units of length²). Historically used with tiny values (e.g.
      :math:`10^{-6}`) as a *numerical* regulariser, which corresponds to
      a sub-grid :math:`L` and produces no physical smoothing.
    * :attr:`smoothing_length` — set :math:`L` directly (length units,
      unit-aware via :func:`underworld3.non_dimensionalise`). This is the
      recommended path when you actually want the projection to act as
      a low-pass filter of a chosen physical scale.

    See Also
    --------
    SNES_Vector_Projection : Vector field projection.
    SNES_Tensor_Projection : Tensor field projection.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh.
    u_Field : MeshVariable, optional
        Target mesh variable for the projection.
    scalar_Field : MeshVariable, optional
        Alternative name for the target field.
    degree : int, default=2
        Polynomial degree for the finite element space.
    solver_name : str, optional
        Name for the solver instance.
    verbose : bool, default=False
        Enable verbose output.
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

        self._needs_function_rewire = True
        self._smoothing = sympy.sympify(0)
        self._uw_weighting_function = sympy.sympify(1)
        self._uw_function = sympy.Matrix([0])  # Default: project zero
        self._constitutive_model = uw.constitutive_models.Constitutive_Model(self.Unknowns)

        return

    # =========================================================================
    # PETSc Residual Templates
    # L2 projection: minimize ||u - f||^2 with optional smoothing
    # =========================================================================

    F0 = Template(
        r"f_0 \left( \mathbf{u} \right)",
        lambda self: (self.u.sym - self.uw_function) * self.uw_weighting_function,
        r"""Projection misfit term (pointwise).

        The $f_0$ term measures the weighted difference between the mesh
        variable and the target function for L2 projection.
        """,
    )

    F1 = Template(
        r"\mathbf{F}_1\left( \mathbf{u} \right)",
        lambda self: self.smoothing * self.mesh.vector.gradient(self.u.sym),
        r"""Projection smoothing term (pointwise).

        The $\mathbf{F}_1$ term provides optional regularization via
        $\alpha \nabla u$, where $\alpha$ is the ``smoothing`` parameter.
        """,
    )

    # Use SymbolicProperty for automatic unwrapping
    uw_function = SymbolicProperty(matrix_wrap=True, doc="Function to project onto mesh")

    def linear_solver(self, pc="jacobi", rtol=1.0e-10):
        """Switch this projector to a lightweight *linear* (SPD) solve.

        An L2 projection (and the screened-Poisson smoother) is a **linear,
        symmetric-positive-definite** problem, so the inherited
        ``newtonls / gmres / gamg`` default is unnecessarily heavy — GAMG
        setup/repartition dominates cost and memory at MPI scale, which is the
        bottleneck for repeated post-processing projections (UW3 issue #156).
        This replaces it with ``ksponly + CG + a cheap preconditioner`` — the
        right tool for the mass/Helmholtz matrix — and removes the now-unused
        GAMG options.

        Opt-in: the default ``SNES_Scalar`` solver stack is unchanged for code
        that relies on it. Call this on a projector used purely for output /
        post-processing.

        Parameters
        ----------
        pc : str, default "jacobi"
            Preconditioner. ``"jacobi"`` is fine for a well-conditioned mass
            matrix; use ``"bjacobi"`` or ``"icc"`` if CG iteration counts climb
            on distorted or high-degree meshes.
        rtol : float, default 1e-10
            KSP relative tolerance.

        Returns
        -------
        self (so the call can be chained).
        """
        self.petsc_options["snes_type"] = "ksponly"
        self.petsc_options["ksp_type"] = "cg"
        self.petsc_options["pc_type"] = pc
        self.petsc_options["ksp_rtol"] = rtol
        # GAMG-specific options are now unused; remove them to avoid PETSc
        # "unused option" warnings (and any stale AMG configuration).
        for _k in (
            "pc_gamg_type",
            "pc_gamg_repartition",
            "pc_gamg_agg_nsmooths",
            "pc_mg_type",
        ):
            try:
                self.petsc_options.delValue(_k)
            except Exception:
                pass
        return self

    @property
    def smoothing(self):
        r"""Smoothing coefficient :math:`\alpha` of the screened-Poisson form.

        The projection solves
        :math:`u - \nabla\!\cdot\!(\alpha\,\nabla u) = \tilde f`, so
        :math:`\alpha` has dimensions of **length²** and the implied
        smoothing length is :math:`L = \sqrt{\alpha}`. The free-space
        Green's function decays as :math:`\exp(-r/L)`, giving the
        projection the action of a Gaussian-like convolution of width
        :math:`L` — without ever forming the kernel.

        See the class docstring for the full derivation.

        Two usage patterns
        ~~~~~~~~~~~~~~~~~~

        * **Pure L2 projection** — ``smoothing = 0`` (or omit). No
          regularisation; ``u`` is the best L2 fit to ``ũ``.
        * **Genuine length-scale smoother** — set
          :attr:`smoothing_length` to the desired physical length
          (unit-aware), or equivalently
          ``smoothing = L**2``. The output is ``ũ`` smoothed at
          scale ``L``.

        .. note::

           Historical UW3 code occasionally sets ``smoothing`` to a
           tiny number (e.g. ``1e-6``) as a *numerical* regulariser
           against rank deficiency. That value corresponds to
           :math:`L \approx 10^{-3}` in the problem's ND length
           units — almost always well below the cell size, so it
           does no useful smoothing. Use a true sub-grid value only
           if you need that regularisation; for filtering, use
           :attr:`smoothing_length`.
        """
        return self._smoothing

    @smoothing.setter
    def smoothing(self, smoothing_factor):
        """Set the smoothing regularization parameter."""
        self._set_smoothing(smoothing_factor)

    @property
    def smoothing_length(self):
        r"""Smoothing length scale :math:`L` (length units, **unit-aware**).

        L-valued view on :attr:`smoothing` with the convention
        :math:`L = \sqrt{\alpha}`. Setting ``smoothing_length = L``
        is equivalent to setting ``smoothing = L**2``, but is the
        natural physical knob because :math:`L` is what the
        smoother actually filters by.

        Mathematical meaning
        --------------------

        The projection then solves the screened-Poisson equation

        .. math::

            u \;-\; L^{2}\,\nabla^{2} u \;=\; \tilde f,

        whose Green's function decays as :math:`\exp(-r/L)`. In
        practice this acts like a Gaussian convolution of width
        :math:`L`:

        * features in :math:`\tilde f` with spatial scale
          :math:`\ll L` are damped (roughly as
          :math:`1/(1+k^{2}L^{2})` for a wavenumber :math:`k`);
        * features with scale :math:`\gg L` are preserved;
        * features at :math:`\sim L` are attenuated by a factor
          near ``1/2``.

        Choosing :math:`L` smaller than the local mesh size has
        essentially no effect (the field is already band-limited
        by the discretisation). A useful default for *light*
        de-noising is :math:`L \approx 1\!-\!2\,h`, where
        :math:`h` is a representative cell size.

        Units
        -----

        The setter accepts a plain number (assumed already
        non-dimensional), a pint ``Quantity`` with length units,
        or any unit-aware object understood by
        :func:`underworld3.non_dimensionalise`. Internally the
        squared non-dimensional value is stored in
        ``self._smoothing`` (so ``smoothing`` and
        ``smoothing_length`` stay consistent).

        The getter returns a Pint ``Quantity`` with length units
        when a scaling context is configured; otherwise the plain
        non-dimensional float :math:`\sqrt{\alpha}`.
        """
        return self._get_smoothing_length()

    @smoothing_length.setter
    def smoothing_length(self, L):
        """Set the smoothing length scale.

        Accepts a Pint Quantity (with length units), a UnitAware
        scalar, or a plain non-dimensional number. The value is
        non-dimensionalised through the active scaling context
        before being squared and stored as ``self._smoothing``.
        """
        self._set_smoothing_length(L)

    @property
    def uw_weighting_function(self):
        """Weighting function applied during projection."""
        return self._uw_weighting_function

    @uw_weighting_function.setter
    def uw_weighting_function(self, user_uw_function):
        """Set the weighting function for the projection."""
        self._needs_function_rewire = True
        self._uw_weighting_function = user_uw_function


## --------------------------------
## Project from pointwise vector
## functions to nodal point unknowns
##
## We can add boundary constraints in the usual way (parent class handles this)
## We can add smoothing (which takes the form of a viscosity term)
## We could consider an incompressibility constraint ... or remove null spaces
## --------------------------------


class SNES_Vector_Projection(_SmoothingLengthMixin, SNES_Vector):
    r"""
    Vector projection solver for mapping vector functions to mesh variables.

    Solves :math:`\mathbf{u} = \tilde{\mathbf{f}}` where
    :math:`\tilde{\mathbf{f}}` is a vector function that can be evaluated
    within an element and :math:`\mathbf{u}` is a vector mesh variable
    with associated shape functions.

    Typically used to obtain a continuous representation of a vector
    function not well-defined at mesh nodes (e.g., gradient or flux
    vectors), or as a length-scale-aware smoother of an existing vector
    field.

    Strong form (screened-Poisson, vector-valued)
    ---------------------------------------------

    .. math::

        \mathbf{u} \;-\; \nabla \cdot \left( \alpha\, \nabla \mathbf{u}
        \right)
        \;+\; \lambda \left( \nabla \cdot \mathbf{u} \right) \mathbf{I}
        \;=\; \tilde{\mathbf{f}} .

    The :math:`\alpha`-term is the same screened-Poisson smoother as
    in :class:`SNES_Projection`, applied component-wise: it has the same
    :math:`L = \sqrt{\alpha}` smoothing-length interpretation and the
    same :math:`\exp(-r/L)` Green's function. The extra :math:`\lambda`
    term is a divergence penalty (see :attr:`penalty`) — set it nonzero
    when you want an approximately solenoidal projection of
    :math:`\tilde{\mathbf{f}}`.

    Setting :math:`\alpha = 0` (and :math:`\lambda = 0`) gives a pure
    pointwise L2 projection. See :class:`SNES_Projection` for the full
    mathematical context; the relationship between :attr:`smoothing`
    (units length²) and :attr:`smoothing_length` (units length) is the
    same as for the scalar projection.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh.
    u_Field : MeshVariable, optional
        Target vector mesh variable for the projection.
    degree : int, default=2
        Polynomial degree for the finite element space.
    verbose : bool, default=False
        Enable verbose output.

    See Also
    --------
    SNES_Projection : Scalar field projection (full mathematical detail).
    SNES_Tensor_Projection : Tensor field projection.
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

        self._needs_function_rewire = True
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
            + self.penalty * self.mesh.vector.divergence(self.u.sym) * sympy.eye(self.mesh.cdim)
        ),
        "Vector projection pointwise smoothing term: F_1(u)",
    )

    # Use SymbolicProperty for automatic unwrapping
    uw_function = SymbolicProperty(matrix_wrap=True, doc="Vector function to project onto mesh")

    @property
    def smoothing(self):
        r"""Smoothing coefficient :math:`\alpha` (units **length²**).

        Coefficient of the :math:`\nabla\!\cdot\!(\alpha\,\nabla
        \mathbf u)` term in the vector screened-Poisson equation
        (see the class docstring). Acts component-wise; the
        smoothing length is :math:`L = \sqrt{\alpha}` and the
        Green's function decays as :math:`\exp(-r/L)`.

        Use :attr:`smoothing_length` for the L-valued, unit-aware
        knob. See :attr:`SNES_Projection.smoothing` for the full
        derivation and usage patterns.
        """
        return self._smoothing

    @smoothing.setter
    def smoothing(self, smoothing_factor):
        """Set the smoothing regularization parameter."""
        self._set_smoothing(smoothing_factor)

    @property
    def smoothing_length(self):
        r"""Smoothing length :math:`L` (length units, **unit-aware**).

        L-valued view on :attr:`smoothing`, with
        :math:`L = \sqrt{\alpha}`. The projection then acts as a
        component-wise screened-Poisson smoother
        :math:`\mathbf u - L^{2}\,\nabla^{2}\mathbf u =
        \tilde{\mathbf f}`, i.e. a Gaussian-like convolution of
        width :math:`L` applied to each Cartesian component of
        the input vector field.

        Choose :math:`L \gtrsim h` (a cell size) for noticeable
        smoothing; :math:`L < h` does essentially nothing because
        the discretisation already band-limits at that scale.
        See :attr:`SNES_Projection.smoothing_length` for the full
        mathematical and units discussion.
        """
        return self._get_smoothing_length()

    @smoothing_length.setter
    def smoothing_length(self, L):
        """Set the smoothing length scale (unit-aware)."""
        self._set_smoothing_length(L)

    @property
    def penalty(self):
        r"""Divergence penalty :math:`\lambda` for (approx.) incompressibility.

        Coefficient of the :math:`\lambda (\nabla\!\cdot\!\mathbf u)
        \mathbf I` term in the vector projection. Large positive
        values bias the projection toward
        :math:`\nabla\!\cdot\!\mathbf u = 0`, i.e. a solenoidal
        approximation of :math:`\tilde{\mathbf f}`. Has no length
        interpretation — unlike :attr:`smoothing` it does not
        introduce a filter scale.
        """
        return self._penalty

    @penalty.setter
    def penalty(self, value):
        """Set the divergence penalty parameter."""
        self._needs_function_rewire = True
        symval = sympify(value)
        self._penalty = symval

    @property
    def uw_weighting_function(self):
        """Weighting function applied during projection."""
        return self._uw_weighting_function

    @uw_weighting_function.setter
    def uw_weighting_function(self, user_uw_function):
        """Set the weighting function for the projection."""
        self._needs_function_rewire = True
        self._uw_weighting_function = user_uw_function


class SNES_Tensor_Projection(SNES_Projection):
    r"""
    Tensor projection solver for mapping tensor functions to mesh variables.

    Solves :math:`\mathbf{u} = \tilde{\mathbf{f}}` where
    :math:`\tilde{\mathbf{f}}` is a tensor-valued function that can be
    evaluated within an element and :math:`\mathbf{u}` is a tensor mesh
    variable with associated shape functions.

    Typically used to obtain a continuous representation of a tensor
    function not well-defined at mesh nodes (e.g., stress or strain
    tensors), with optional length-scale smoothing.

    Strong form (screened-Poisson, applied component-wise)
    ------------------------------------------------------

    Internally the solve is decomposed into scalar sub-problems, one per
    tensor component :math:`u_{ij}`; each sub-problem is

    .. math::

        u_{ij} \;-\; \nabla \cdot \left( \alpha\, \nabla u_{ij}
        \right) \;=\; \tilde{f}_{ij},

    identical to :class:`SNES_Projection`. The smoothing length is
    :math:`L = \sqrt{\alpha}` and the Green's function decays as
    :math:`\exp(-r/L)`; setting :math:`\alpha = 0` gives a pure
    pointwise L2 projection. See :class:`SNES_Projection` for the full
    derivation, the choice between :attr:`smoothing` (length²) and
    :attr:`smoothing_length` (length, unit-aware), and guidance on
    picking :math:`L` relative to the cell size.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh.
    tensor_Field : MeshVariable, optional
        Target tensor mesh variable for the projection.
    scalar_Field : MeshVariable, optional
        Scalar work variable used internally.
    degree : int, default=2
        Polynomial degree for the finite element space.
    verbose : bool, default=False
        Enable verbose output.

    Notes
    -----
    Currently implemented component-wise as there is no native solver
    for tensor unknowns. Each component sees the same :math:`\alpha`,
    so the effective smoothing length :math:`L` is uniform across the
    tensor entries.

    See Also
    --------
    SNES_Projection : Scalar field projection (full mathematical detail).
    SNES_Vector_Projection : Vector field projection.
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
    def solve(self, verbose=False, divergence_retries: int = 0):
        """Solve by projecting each tensor component sequentially.

        Parameters
        ----------
        verbose : bool
        divergence_retries : int, default=0
            Forwarded to each per-component SNES solve. 0 preserves
            legacy behaviour.
        """
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
        """Pointwise misfit term for scalar subproblem."""
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
        """Pointwise smoothing flux term for scalar subproblem."""
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
        """Current scalar component function being projected."""
        return self._uw_scalar_function

    @uw_scalar_function.setter
    def uw_scalar_function(self, user_uw_function):
        """Set the scalar component function for current tensor element."""
        self._needs_function_rewire = True
        self._uw_scalar_function = user_uw_function


class SNES_MultiComponent_Projection(_SmoothingLengthMixin, SNES_MultiComponent):
    r"""
    Multi-component projection solver.

    Projects an N-component row-matrix expression onto an N-component
    mesh variable in a **single** SNES solve, sharing one DM across all
    components. Replaces the per-component cycling used by
    :class:`SNES_Tensor_Projection`, which tears down and rebuilds the
    PETSc DM on every inner iteration.

    Strong form (block-diagonal screened Poisson)
    ---------------------------------------------

    There is no cross-component coupling, so each component
    :math:`u_k,\ k=1,\dots,N_c` satisfies the same scalar
    screened-Poisson equation as :class:`SNES_Projection`,

    .. math::

        u_k \;-\; \nabla \cdot \left( \alpha\, \nabla u_k \right)
        \;=\; \tilde f_k.

    Setting :math:`\alpha = 0` gives a pure pointwise L2 projection per
    component. The smoothing length is :math:`L = \sqrt{\alpha}` and the
    Green's function decays as :math:`\exp(-r/L)` — the same Gaussian-like
    convolution interpretation as the scalar case. All components share a
    single :math:`\alpha`, so the smoothing scale :math:`L` is uniform
    across the multi-component target. Use :attr:`smoothing` to set
    :math:`\alpha` (units length²) or :attr:`smoothing_length` for the
    L-valued, unit-aware knob. See :class:`SNES_Projection` for the full
    derivation and guidance.

    Parameters
    ----------
    mesh : Mesh
    u_Field : MeshVariable, optional
        Target ``(1, n_components)`` MATRIX variable. If ``None``, one is
        created.
    n_components : int, optional
        Number of scalar components; required if ``u_Field`` is ``None``.
    degree : int, default=2
    verbose : bool, default=False

    See Also
    --------
    SNES_Projection : Scalar projection (Nc=1) — full mathematical detail.
    SNES_Tensor_Projection : Legacy per-component cycling projector.
    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        u_Field: uw.discretisation.MeshVariable = None,
        n_components: int = None,
        degree=2,
        verbose=False,
    ):
        super().__init__(
            mesh,
            u_Field=u_Field,
            n_components=n_components,
            degree=degree,
            verbose=verbose,
        )

        self._needs_function_rewire = True
        self._smoothing = sympify(0)
        self._uw_weighting_function = sympify(1)
        self._uw_function = sympy.zeros(1, self._n_components)
        self._constitutive_model = uw.constitutive_models.Constitutive_Model(self.Unknowns)

    F0 = Template(
        r"f_0 \left( \mathbf{u} \right)",
        lambda self: (self.u.sym - self.uw_function) * self.uw_weighting_function,
        "Per-component projection misfit (row matrix shape (1, Nc)).",
    )

    F1 = Template(
        r"\mathbf{F}_1 \left( \mathbf{u} \right)",
        lambda self: self.smoothing * self.Unknowns.L,
        "Per-component block-diagonal Laplacian smoothing (shape (Nc, dim)).",
    )

    uw_function = SymbolicProperty(
        matrix_wrap=True,
        doc="Row matrix (1, n_components) of expressions to project.",
    )

    @property
    def smoothing(self):
        r"""Smoothing coefficient :math:`\alpha` (units **length²**).

        Coefficient of the :math:`\nabla\!\cdot\!(\alpha\,\nabla
        u_k)` term in each component's screened-Poisson sub-problem
        (see the class docstring). One :math:`\alpha` is shared
        across all :math:`N_c` components, so the implied smoothing
        length :math:`L = \sqrt{\alpha}` is uniform.

        Use :attr:`smoothing_length` for the L-valued, unit-aware
        knob. See :attr:`SNES_Projection.smoothing` for the full
        derivation and the Gaussian-like convolution interpretation.
        """
        return self._smoothing

    @smoothing.setter
    def smoothing(self, value):
        self._set_smoothing(value)

    @property
    def smoothing_length(self):
        r"""Smoothing length :math:`L` (length units, **unit-aware**).

        L-valued view on :attr:`smoothing`, with
        :math:`L = \sqrt{\alpha}`. Each component then satisfies
        :math:`u_k - L^{2}\,\nabla^{2} u_k = \tilde f_k`, i.e. a
        Gaussian-like convolution of width :math:`L` applied
        independently to each component of the multi-component
        target. See :attr:`SNES_Projection.smoothing_length` for the
        full mathematical and units discussion.
        """
        return self._get_smoothing_length()

    @smoothing_length.setter
    def smoothing_length(self, L):
        """Set the smoothing length scale (unit-aware)."""
        self._set_smoothing_length(L)

    @property
    def uw_weighting_function(self):
        """Weighting function applied during projection."""
        return self._uw_weighting_function

    @uw_weighting_function.setter
    def uw_weighting_function(self, value):
        self._needs_function_rewire = True
        self._uw_weighting_function = value


# #################################################
# # Swarm-based advection-diffusion
# # solver based on SNES_Poisson and swarm-variable
# # projection
# #
# #################################################


class SNES_AdvectionDiffusion(SNES_Scalar):
    r"""
    Advection-diffusion equation solver using semi-Lagrangian Crank-Nicolson.

    Implements the characteristics-based method described in Spiegelman & Katz (2006):

    .. math::

        \underbrace{\frac{\partial u}{\partial t} + \left( \mathbf{v} \cdot \nabla
        \right) u}_{\dot{u}} - \nabla \cdot \underbrace{\left[ \boldsymbol{\kappa}
        \nabla u \right]}_{\mathbf{F}} = \underbrace{f}_{\mathbf{h}}

    The flux term :math:`\mathbf{F}` relates diffusive fluxes to gradients in
    :math:`u`. Advective fluxes along the velocity field :math:`\mathbf{v}` are
    handled in the :math:`\dot{u}` term.

    The time derivative :math:`\dot{u}` involves upstream sampling to find
    :math:`u^*`, the value of :math:`u` at points which later arrive at mesh
    nodes. This is achieved using a hidden swarm variable advected backwards
    from nodal points automatically during solve.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh.
    u_Field : MeshVariable
        Mesh variable for the transported scalar.
    V_fn : MeshVariable or sympy.Basic
        Velocity field for advection.
    order : int, default=1
        Time integration order. Note the scheme has **two** order knobs that
        must be paired consistently (see ``theta``):

        - the advective time-derivative ``DuDt`` carries the **BDF** order of
          the backward difference along the characteristic;
        - the diffusive flux ``DFDt`` carries the **Adams-Moulton/θ** flux
          integrator.

        When ``DuDt`` is built internally (``DuDt=None``) it is fixed at BDF
        order 1, so this ``order`` raises only the *flux* AM order — i.e.
        ``order=1, theta=0.5`` is the canonical **SLCN** (BDF1 difference +
        Crank-Nicolson flux), the standard second-order scheme. To run
        **SL-BDF2** (BDF2 difference + Backward-Euler-centred flux, second
        order without CN's spurious resonance) supply an explicit order-2
        ``DuDt`` and set ``theta=1.0`` on the flux:

        .. code-block:: python

           duDt = uw.systems.ddt.SemiLagrangian(
               mesh, T.sym, V_fn, vtype=uw.VarType.SCALAR,
               degree=T.degree, continuous=T.continuous, order=2)
           adv = uw.systems.AdvDiffusionSLCN(mesh, T, V_fn, DuDt=duDt, order=1)
           adv.DFDt.theta = 1.0   # flux implicit at n+1 (BDF2-consistent)

        A BDF2 stencil with a Crank-Nicolson flux (``order=2`` + ``theta=0.5``)
        centres the two sides at different times and is **not** a consistent
        second-order scheme. See
        ``docs/advanced/semi-lagrangian-time-integration.md``.
    restore_points_func : callable, optional
        Function to restore particles to valid domain.
    verbose : bool, default=False
        Enable verbose output.
    DuDt : SemiLagrangian_DDt or Lagrangian_DDt, optional
        Time derivative operator for the unknown.
    DFDt : SemiLagrangian_DDt or Lagrangian_DDt, optional
        Time derivative operator for the flux.
    monotone_mode : str or None, optional
        Monotonicity limiter for the semi-Lagrangian trace-back.
        Forwarded to the internally-constructed ``SemiLagrangian_DDt``
        instances for ``DuDt`` and ``DFDt``.

        - ``None`` (default): pure FE trace-back. Can overshoot at
          non-nodal upstream points in cells with sharp gradients
          (e.g. thin boundary layers in deformed cells); legacy
          behaviour.
        - ``"clamp"``: clip the FE trace-back result to
          ``[nbr_min, nbr_max]`` of the ``k = dim + 1`` nearest
          ``psi_star`` DOFs. Bit-identical to pure FE in smooth
          regions; bounds overshoots where the FE interpolant would
          otherwise leave the local data range.
        - ``"pick"``: keep the FE result where in-bounds; for
          out-of-bounds DOFs only, re-evaluate via Shepard's-method
          RBF. More conservative than clamp at the catastrophe edge.

        When a user supplies a pre-built ``DuDt``, this kwarg is
        applied to the internally-constructed ``DFDt`` only — the
        user's ``DuDt`` already encodes whatever ``monotone_mode``
        it was constructed with.
    theta : float, default=0.5
        Adams-Moulton theta for the diffusive flux at order 1.
        Forwarded to the internally-constructed
        ``SemiLagrangian_DDt`` instances (same forwarding rule as
        ``monotone_mode``).

        - ``0.5`` (default): Crank-Nicolson, A-stable but not
          L-stable. Second-order accurate. Rings on stiff modes
          with sharp gradients (negative amplification factor).
        - ``1.0``: Backward Euler, L-stable, monotone for the
          diffusive flux. First-order accurate. Recommended when
          SLCN+CN ringing dominates the discretisation error.
        - ``0.0``: Forward Euler — unstable for stiff diffusion;
          included for completeness.
    old_frame_traceback : bool, default=False
        Use the old-frame semi-Lagrangian reach-back for the advective
        ``DuDt`` history on a moving mesh (free surface or interior-node
        adaptation). Forwarded to the internally-constructed ``DuDt``
        only — the diffusive ``DFDt`` keeps the standard ALE path
        (validated sufficient at Ra=1e5; the unstable mode rides the
        advective scalar, not the dissipative flux).

        When ``True``, the trace-back computes the departure foot from
        the PHYSICAL velocity and samples ``psi_star`` on the mesh
        ephemerally restored to the previous-step geometry, instead of
        the lossy ``v_mesh = Δx/dt`` fold on the new mesh. This cures the
        high-Ra free-surface convection blow-up (T leaves [0,1] ~step 20
        at Ra=1e5). **Contract:** pass the physical velocity as
        ``V_fn`` (NOT ``v − v_mesh``) — the old-frame trace-back must not
        double-compensate the mesh motion. See
        ``docs/developer/design/lagged-clone-sl-history.md``.

    Notes
    -----
    - The diffusivity :math:`\kappa` is set via the ``constitutive_model`` property
    - Sources :math:`f` can be any sympy expression involving mesh/swarm variables

    References
    ----------
    Spiegelman, M., & Katz, R. F. (2006). A semi-Lagrangian Crank-Nicolson
    algorithm for the numerical solution of advection-diffusion problems.
    *Geochemistry, Geophysics, Geosystems*, 7(4).
    https://doi.org/10.1029/2005GC001073

    Bonaventura, L., Calzola, E., Carlini, E., & Ferretti, R. (2021). Second
    order fully semi-Lagrangian discretizations of advection-diffusion-reaction
    systems. *Journal of Scientific Computing*, 88, 23.
    https://doi.org/10.1007/s10915-021-01518-8 (SL-BDF2 vs SL-CN).

    See Also
    --------
    SNES_Diffusion : Pure diffusion solver without advection.
    SNES_Navier_Stokes : Full momentum advection-diffusion.
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
        monotone_mode: Optional[str] = None,
        theta: float = 0.5,
        old_frame_traceback: bool = False,
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
        self._needs_function_rewire = True

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
                monotone_mode=monotone_mode,
                theta=theta,
                old_frame_traceback=old_frame_traceback,
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

        # Flux placeholder: cdim-sized to match the embedded-coord
        # flux vector (volume meshes have dim==cdim so this is
        # unchanged; manifold meshes have cdim > dim and need the
        # extra component).
        self.Unknowns.DFDt = SemiLagrangian_DDt(
            self.mesh,
            sympy.Matrix([[0] * self.mesh.cdim]),  # Actual function is not defined at this point
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
            monotone_mode=monotone_mode,
            theta=theta,
        )

        # Phase-2 remesh: the advected field must ride the mesh (CARRY)
        # coherently with its own semi-Lagrangian history (psi_star,
        # already CARRY + managed). Under the default REMAP it would be
        # geometrically re-interpolated onto the new node positions AND
        # then have v_mesh = Δx/dt subtracted in the next trace-back —
        # a DOUBLE compensation for the mesh motion, inconsistent with
        # the once-compensated CARRY'd history. Stamping it CARRY +
        # managed-by-DuDt makes deform()/adapt transfer it on the single
        # ALE trace-back path (and REMAP it correctly on an OT opt-out
        # reset). Only meaningful when DuDt traces back (SemiLagrangian);
        # Eulerian/Lagrangian fields keep the default policy.
        if isinstance(self.Unknowns.DuDt, SemiLagrangian_DDt):
            from underworld3.discretisation.remesh import RemeshPolicy
            self.u.remesh_policy = RemeshPolicy.CARRY
            self.u._remesh_managed_by = self.Unknowns.DuDt

        return

    @property
    def F0(self):
        """Pointwise source term including time derivative."""
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
        """Pointwise diffusive flux term (Adams-Moulton integration)."""
        F1_val = expression(
            r"\mathbf{F}_1\left( \mathbf{u} \right)",
            self.DFDt.adams_moulton_flux(),
            "Poisson pointwise flux term: F_1(u)",
        )

        # backward compatibility
        self._f1 = F1_val

        return F1_val

    def adv_diff_slcn_problem_description(self):
        """Build residual terms for advection-diffusion FEM assembly."""
        # f0 residual term
        self._f0 = self.F0.sym

        # f1 residual term
        self._f1 = self.F1.sym

        return

    @property
    def f(self):
        r"""Source term for the advection-diffusion equation.

        The source :math:`f` appears on the right-hand side:

        .. math::
            \frac{\partial u}{\partial t} + \mathbf{V} \cdot \nabla u
            = \nabla \cdot (\kappa \nabla u) + f

        Returns
        -------
        sympy.Matrix
            Source term expression.
        """
        return self._f

    @f.setter
    def f(self, value):
        """Set the volumetric source term."""
        self._needs_function_rewire = True
        self._f = sympy.Matrix((value,))

    @property
    def V_fn(self):
        r"""Velocity field for advection.

        The advection velocity :math:`\mathbf{V}` transports the scalar
        field :math:`u`. Can be a MeshVariable or symbolic expression.

        Returns
        -------
        sympy.Matrix
            Velocity vector expression.
        """
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
        r"""Timestep for time integration.

        The timestep :math:`\Delta t` controls the temporal discretization.
        For explicit advection, this should satisfy the CFL condition:

        .. math::
            \Delta t < \frac{h}{|\mathbf{V}|}

        where :math:`h` is the element size and :math:`|\mathbf{V}|` is the
        velocity magnitude.

        Returns
        -------
        UWexpression
            Timestep value.

        See Also
        --------
        estimate_dt : Computes a stable timestep automatically.
        """
        return self._delta_t

    @delta_t.setter
    def delta_t(self, value):
        """Set the timestep (handles unit conversion if provided)."""
        # Skip the (expensive) function rewire when the timestep is unchanged.
        # The comparison must tolerate UWexpressions / UWQuantities, hence the
        # float() coercion via .data.
        try:
            old_dt = float(self._delta_t.data)
            new_dt = float(value.data) if hasattr(value, 'data') else float(value)
            if np.isclose(old_dt, new_dt, rtol=1e-12, atol=1e-15):
                return
        except (TypeError, ValueError):
            # Non-numeric (e.g. symbolic) timestep: no stable comparison is
            # possible — fall through and assign.
            pass

        self._needs_function_rewire = True
        self._delta_t.sym = _nondimensionalise_timestep(value)

    @timing.routine_timer_decorator
    def estimate_dt(self, direction_aware: bool = False, percentile: float = 0.0):
        r"""
        Estimate an appropriate timestep for the advection-diffusion solver.

        This is an implicit solver so the returned :math:`\delta t` is the
        minimum of:

        - :math:`\delta t_{\textrm{diff}}`: typical time for diffusion across an element
        - :math:`\delta t_{\textrm{adv}}`: typical element-crossing time for a fluid parcel

        Parameters
        ----------
        direction_aware : bool, default False
            If True, the advective dt uses the per-cell extent
            *along the local velocity direction* — `h_eff_c =
            max_i(s_i) - min_i(s_i)` where `s_i = (x_i -
            centroid) · v̂` over the cell vertices. This is the
            distance material actually traverses through the cell
            per unit ``|v|``, and is **always ≥ the isotropic
            mesh._radii estimate**, by 1.5–3× for equant cells
            (geometric factor) and up to ~10× for cells that the
            mover has stretched along the flow direction. On
            adapted meshes the gain is substantial; on uniform
            meshes it's the geometric factor only. Off by
            default to preserve historical behaviour; safe to
            enable everywhere once validated.

        Returns
        -------
        pint.Quantity or float
            The recommended timestep with physical time units if a model
            with reference scales is available, otherwise nondimensional.
        """

        ### required modules
        from mpi4py import MPI

        comm = uw.mpi.comm

        ## global max diffusivity (unified .K property: diffusivity for
        ## diffusion models)
        diffusivity_glob = _global_max_diffusivity(
            self.constitutive_model.K, self.mesh)

        ### velocity values at element centroids (nondimensional)
        vel = _centroid_velocities_nd(self.V_fn, self.mesh)

        # Get per-element velocity magnitudes
        vel_magnitudes = np.linalg.norm(vel, axis=1)

        # Get per-element radii (characteristic element size)
        element_radii = self.mesh._radii

        ## estimate dt of adv and diff components using per-element approach
        ## dt_adv_i = h_i / |v_i| for advection
        ## dt_diff_i = h_i^2 / κ for diffusion (using global κ for now)

        # Reduce per-element dt to one global value. Default (percentile=0) =
        # strict global MINIMUM — one cell sets the limit. percentile>0 takes the
        # Nth global percentile (50 = median) of the per-element dt instead, so a
        # few anisotropic SLIVER cells (velocity ACROSS a thin cell) don't collapse
        # dt. SLCN is unconditionally stable, and ``direction_aware`` already
        # credits cells stretched ALONG the flow — together they give the
        # orientation-aware + sliver-robust timestep.
        def _reduce_dt(per_elem):
            fin = per_elem[np.isfinite(per_elem)] if len(per_elem) else per_elem
            if percentile and percentile > 0:
                gathered = comm.allgather(np.ascontiguousarray(fin, dtype=float))
                allv = (np.concatenate([a for a in gathered if a.size])
                        if any(a.size for a in gathered) else np.empty(0))
                return float(np.percentile(allv, percentile)) if allv.size else np.inf
            loc = float(np.min(fin)) if len(fin) else np.inf
            return comm.allreduce(loc, op=MPI.MIN)

        # Per-element diffusive timestep (all elements use same diffusivity)
        if diffusivity_glob > 0:
            dt_diff_per_element = (element_radii ** 2) / diffusivity_glob
        else:
            dt_diff_per_element = np.array([np.inf])

        # Per-element advective timestep — either isotropic
        # (mesh._radii / |v|) or direction-aware (v-aligned cell
        # extent / |v|).
        if direction_aware:
            # Per-cell vertex indices (triangle / tet).
            from underworld3.meshing.smoothing import _tri_cells
            tris = _tri_cells(self.mesh.dm)
            if tris is None:
                # Fall back to isotropic for non-triangle meshes.
                h_per_element = element_radii
            else:
                coords = np.asarray(self.mesh.X.coords)
                centroids = coords[tris].mean(axis=1)
                # v-hat per cell (use centroid v we already have)
                vhat = np.where(
                    vel_magnitudes[:, None] > 0,
                    vel / np.maximum(vel_magnitudes[:, None],
                                      1.0e-30),
                    0.0)
                D = coords[tris] - centroids[:, None, :]
                # Signed projections along v̂ per cell vertex
                s = np.einsum('cvd,cd->cv', D, vhat)
                h_per_element = s.max(axis=1) - s.min(axis=1)
                # Sanity-floor — for zero-velocity cells s=0
                # ⇒ h_eff=0 ⇒ dt_adv=inf via the where below
                h_per_element = np.maximum(
                    h_per_element, 0.0)
        else:
            h_per_element = element_radii

        with np.errstate(divide='ignore', invalid='ignore'):
            dt_adv_per_element = np.where(
                vel_magnitudes > 0,
                h_per_element / vel_magnitudes,
                np.inf
            )
        # Global reduction — strict min (percentile=0) or Nth percentile (median).
        min_dt_diff_glob = _reduce_dt(dt_diff_per_element)
        min_dt_adv_glob = _reduce_dt(dt_adv_per_element)

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
        divergence_retries: int = 0,
    ):
        """
        Generates solution to constructed system.

        Params
        ------
        zero_init_guess:
            If `True`, a zero initial guess will be used for the
            system solution. Otherwise, the current values of `self.u` will be used.
        divergence_retries:
            If SNES reports DIVERGED, retry with warm start up to this
            many times. 0 preserves legacy behaviour.
        """

        if timestep is not None and timestep != self.delta_t:
            self.delta_t = timestep  # this will force an initialisation because the functions need to be updated

        if _force_setup:
            self._needs_function_rewire = True

        if not self.constitutive_model._solver_is_setup:
            self._needs_function_rewire = True
            self.DFDt.psi_fn = self.constitutive_model.flux.T

        if not self.is_setup:
            self._setup_pointwise_functions(verbose)
            self._setup_discretisation(verbose)
            self._setup_solver(verbose)

        # Update History / Flux History terms
        # SemiLagrange and Lagrange may have different sequencing.

        self.DuDt.update_pre_solve(timestep, verbose=verbose, evalf=_evalf)
        self.DFDt.update_pre_solve(timestep, verbose=verbose, evalf=_evalf)

        super().solve(zero_init_guess, _force_setup,
                      divergence_retries=divergence_retries)

        _invalidate_solution_cache(self.u)

        self.DuDt.update_post_solve(timestep, verbose=verbose, evalf=_evalf)
        self.DFDt.update_post_solve(timestep, verbose=verbose, evalf=_evalf)

        self.is_setup = True
        self.constitutive_model._solver_is_setup = True

        return


class SNES_Diffusion(SNES_Scalar):
    r"""
    Diffusion equation solver using mesh-based finite elements.

    Solves the scalar diffusion equation:

    .. math::

        \underbrace{\frac{\partial u}{\partial t}}_{\dot{f}}
        - \nabla \cdot \underbrace{\left[ \boldsymbol{\kappa} \nabla u
        \right]}_{\mathbf{F}} = \underbrace{f}_{h}

    The flux term :math:`\mathbf{F}` relates diffusive fluxes to gradients
    in the unknown :math:`u`.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh.
    u_Field : MeshVariable
        Mesh variable for the diffusing scalar.
    order : int, default=1
        Time integration order.
    theta : float, default=0.0
        Time integration parameter (0=explicit, 0.5=Crank-Nicolson, 1=implicit).
    evalf : bool, default=False
        Numerically evaluate symbolic expressions during setup.
    verbose : bool, default=False
        Enable verbose output.
    DuDt : Eulerian_DDt, SemiLagrangian_DDt, or Lagrangian_DDt, optional
        Time derivative operator for the unknown.
    DFDt : Eulerian_DDt, SemiLagrangian_DDt, or Lagrangian_DDt, optional
        Time derivative operator for the flux.

    Notes
    -----
    - The diffusivity :math:`\kappa` is set via the ``constitutive_model`` property
    - Sources :math:`f` can be any sympy expression involving mesh/swarm variables

    See Also
    --------
    SNES_AdvectionDiffusion : Adds advection transport.
    SNES_Poisson : Steady-state diffusion (no time derivative).
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
        self._needs_function_rewire = True

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
            # Flux placeholder must match the flux dimension (cdim — the
            # embedded coordinate space), not the topological dim. On
            # volume meshes dim==cdim so this is unchanged.
            self.Unknowns.DFDt = Symbolic_DDt(
                sympy.Matrix([[0] * self.mesh.cdim]),
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
        """Pointwise source term including time derivative."""
        f0 = expression(
            r"f_0 \left( \mathbf{u} \right)",
            -self.f + self.DuDt.bdf() / self.delta_t,
            "Diffusion pointwise force term: f_0(u)",
        )

        # backward compatibility
        self._f0 = f0

        return f0

    @property
    def F1(self):
        """Pointwise diffusive flux term."""
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
        """Source term for the diffusion equation."""
        return self._f

    @f.setter
    def f(self, value):
        """Set the volumetric source term."""
        self._needs_function_rewire = True
        self._f = sympy.Matrix((value,))

    @property
    def delta_t(self):
        """Timestep for time integration."""
        return self._delta_t

    @delta_t.setter
    def delta_t(self, value):
        """Set the timestep (handles unit conversion if provided)."""
        # Skip the (expensive) function rewire when the timestep is unchanged.
        # The comparison must tolerate UWexpressions / UWQuantities, hence the
        # float() coercion via .data.
        try:
            old_dt = float(self._delta_t.data)
            new_dt = float(value.data) if hasattr(value, 'data') else float(value)
            if np.isclose(old_dt, new_dt, rtol=1e-12, atol=1e-15):
                return
        except (TypeError, ValueError):
            # Non-numeric (e.g. symbolic) timestep: no stable comparison is
            # possible — fall through and assign.
            pass

        self._needs_function_rewire = True
        self._delta_t.sym = _nondimensionalise_timestep(value)

    @timing.routine_timer_decorator
    def estimate_dt(self):
        r"""
        Estimate an appropriate timestep for the diffusion solver.

        This solver only has a diffusive component, so the returned
        :math:`\delta t` is:

        - :math:`\delta t_{\textrm{diff}}`: typical time for diffusion across an element

        Returns
        -------
        pint.Quantity or float
            The diffusive timestep with physical time units if a model
            with reference scales is available, otherwise nondimensional.
        """

        ## global max diffusivity (unified .K property: diffusivity for
        ## diffusion models)
        diffusivity_glob = _global_max_diffusivity(
            self.constitutive_model.K, self.mesh)

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
        divergence_retries: int = 0,
    ):
        """
        Generates solution to constructed system.

        Params
        ------
        zero_init_guess:
            If `True`, a zero initial guess will be used for the
            system solution. Otherwise, the current values of `self.u` will be used.
        divergence_retries:
            If SNES reports DIVERGED, retry with warm start up to this
            many times. 0 preserves legacy behaviour.
        """

        if timestep is not None and timestep != self.delta_t:
            self.delta_t = timestep  # this will force an initialisation because the functions need to be updated

        if _force_setup:
            self._needs_function_rewire = True

        if not self.constitutive_model._solver_is_setup:
            self._needs_function_rewire = True
            self.DFDt.psi_fn = self.constitutive_model.flux.T
            # self._flux =  self.constitutive_model.flux.T
            # self._flux_star =  self._flux.copy()

        if not self.is_setup:
            self._setup_pointwise_functions(verbose)
            self._setup_discretisation(verbose)
            self._setup_solver(verbose)

        # Update History / Flux History terms
        # SemiLagrange and Lagrange may have different sequencing.
        self.DuDt.update_pre_solve(timestep, evalf=evalf, verbose=verbose)
        self.DFDt.update_pre_solve(timestep, evalf=evalf, verbose=verbose)

        super().solve(zero_init_guess, _force_setup,
                      divergence_retries=divergence_retries)

        _invalidate_solution_cache(self.u)

        self.DuDt.update_post_solve(timestep, evalf=evalf, verbose=verbose)
        self.DFDt.update_post_solve(timestep, evalf=evalf, verbose=verbose)

        self.is_setup = True
        self.constitutive_model._solver_is_setup = True

        return


# This one is already updated to work with the Lagrange D_Dt
class SNES_NavierStokes(SNES_Stokes_SaddlePt):
    r"""
    Navier-Stokes equation solver with momentum advection.

    Provides a solver for the Navier-Stokes (vector advection-diffusion) equation
    similar to the Semi-Lagrange Crank-Nicolson method (Spiegelman & Katz, 2006)
    but using distributed upstream sampling from a swarm variable.

    .. math::

        \underbrace{\frac{\partial \mathbf{u}}{\partial t}
        + \left( \mathbf{u} \cdot \nabla \right) \mathbf{u}}_{\dot{\mathbf{u}}}
        - \nabla \cdot \underbrace{\left[ \frac{\boldsymbol{\eta}}{2}
        \left( \nabla \mathbf{u} + \nabla \mathbf{u}^T \right)
        - p \mathbf{I} \right]}_{\mathbf{F}} = \underbrace{\mathbf{f}}_{\mathbf{h}}

    The flux term :math:`\mathbf{F}` relates viscous stresses to velocity gradients.
    Advective momentum transport is handled in the :math:`\dot{\mathbf{u}}` term.

    The time derivative :math:`\dot{\mathbf{u}}` involves upstream sampling to find
    :math:`\mathbf{u}^*`, representing velocity at the start of the timestep. This is
    achieved using a swarm variable that carries history information along flow paths.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh.
    velocityField : MeshVariable
        Mesh variable for velocity.
    pressureField : MeshVariable
        Mesh variable for pressure.
    rho : float or sympy.Expr
        Fluid density.
    order : int, default=1
        Time integration order.
    theta : float, default=0.5
        Time integration parameter.
    p_continuous : bool, default=True
        If False, use discontinuous pressure elements.
    verbose : bool, default=False
        Enable verbose output.
    DuDt : SemiLagrangian_DDt or Lagrangian_DDt, optional
        Time derivative operator for velocity.
    DFDt : SemiLagrangian_DDt or Lagrangian_DDt, optional
        Time derivative operator for stress.

    Notes
    -----
    - The viscosity :math:`\eta` is set via the ``constitutive_model`` property
    - High-order shape functions (cubic or higher) are recommended for accurate
      history term interpolation
    - The user must supply and update the swarm variable representing :math:`\mathbf{u}^*`

    References
    ----------
    Spiegelman, M., & Katz, R. F. (2006). A semi-Lagrangian Crank-Nicolson
    algorithm for the numerical solution of advection-diffusion problems.
    *Geochemistry, Geophysics, Geosystems*, 7(4).
    https://doi.org/10.1029/2005GC001073

    See Also
    --------
    SNES_Stokes : Steady-state Stokes flow (no inertia).
    SNES_AdvectionDiffusion : Scalar advection-diffusion.
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
        flux_order: Optional[int] = None,
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

        self._needs_function_rewire = True
        self._rho = expression(R"{\uprho}", rho, "Density")
        self._first_solve = True

        self._order = order
        self._flux_order = flux_order  # None means follow effective_order
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
        """Pointwise momentum source term (body force + inertia)."""
        DuDt = self.Unknowns.DuDt

        f0 = expression(
            r"\mathbf{f}_0\left( \mathbf{u} \right)",
            -self.bodyforce + self.rho * DuDt.bdf() / self.delta_t,
            "NStokes pointwise force term: f_0(u)",
        )

        self._u_f0 = f0

        return f0

    @property
    def F1(self):
        """Pointwise stress flux term (viscous + pressure)."""
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
        """Pointwise constraint term (continuity equation)."""
        dim = self.mesh.dim

        f0 = expression(
            r"\mathbf{F}_1\left( \mathbf{p} \right)",
            sympy.Matrix((self.constraints)),
            "NStokes pointwise flux term: f_0(p)",
        )

        self._p_f0 = f0

        return f0

    ## Deprecate this function
    def navier_stokes_problem_description(self):
        """Build residual terms for Navier-Stokes FEM assembly (deprecated)."""
        # f0 residual term
        self._u_f0 = self.F0.sym

        # f1 residual term
        self._u_f1 = self.F1.sym

        # p1 residual term
        self._p_f0 = self.PF0.sym

        return

    @property
    def delta_t(self):
        """Timestep for time integration."""
        return self._delta_t

    @delta_t.setter
    def delta_t(self, value):
        """Set the timestep value."""
        self._needs_function_rewire = True
        self._delta_t.sym = value

    @property
    def rho(self):
        """Fluid density."""
        return self._rho

    @rho.setter
    def rho(self, value):
        """Set the fluid density."""
        self._needs_function_rewire = True
        self._rho.sym = value

    @property
    def f(self):
        """Source term for the momentum equation."""
        return self._f

    @f.setter
    def f(self, value):
        """Set the volumetric source term."""
        self._needs_function_rewire = True
        self._f = sympy.Matrix((value,))

    @property
    def div_u(self):
        r"""Velocity divergence: :math:`\nabla \cdot \mathbf{u} = \mathrm{tr}(\dot{\varepsilon})`."""
        E = self.strainrate
        divergence = E.trace()
        return divergence

    @property
    def strainrate(self):
        r"""Symmetric strain rate tensor (with 1/2 factor).

        .. math::
            \dot{\varepsilon}_{ij} = \frac{1}{2}\left(\frac{\partial u_i}{\partial x_j}
            + \frac{\partial u_j}{\partial x_i}\right)
        """
        return sympy.Matrix(self.Unknowns.E)

    @property
    def DuDt(self):
        """Time derivative operator for velocity."""
        return self.Unknowns.DuDt

    @DuDt.setter
    def DuDt(
        self,
        DuDt_value: Union[SemiLagrangian_DDt, Lagrangian_DDt],
    ):
        """Set the time derivative operator for velocity."""
        self.Unknowns.DuDt = DuDt_value
        self._solver_is_setup = False

    @property
    def DFDt(self):
        """Time derivative operator for stress flux."""
        return self.Unknowns.DFDt

    @property
    def constraints(self):
        """Constraint equation (typically incompressibility)."""
        return self._constraints

    @constraints.setter
    def constraints(self, constraints_matrix):
        """Set the constraint equation."""
        self._is_setup = False
        symval = sympify(constraints_matrix)
        self._constraints = symval

    @property
    def bodyforce(self):
        """Body force vector (e.g., gravity)."""
        return self._bodyforce

    @bodyforce.setter
    def bodyforce(self, value):
        """Set the body force vector."""
        self._needs_function_rewire = True
        self._bodyforce = self.mesh.vector.to_matrix(value)

    @property
    def saddle_preconditioner(self):
        """Preconditioner for the Schur complement."""
        return self._saddle_preconditioner

    @saddle_preconditioner.setter
    def saddle_preconditioner(self, value):
        """Set the Schur complement preconditioner."""
        self._needs_function_rewire = True
        symval = sympify(value)
        self._saddle_preconditioner = symval

    @property
    def penalty(self):
        """Augmented Lagrangian penalty parameter."""
        return self._penalty

    @penalty.setter
    def penalty(self, value):
        """Set the augmented Lagrangian penalty parameter."""
        self._needs_function_rewire = True
        self._penalty.sym = value

    @timing.routine_timer_decorator
    @memprobe.instrument("NavierStokes.solve")
    def solve(
        self,
        zero_init_guess: bool = True,
        timestep: float = None,
        _force_setup: bool = False,
        verbose=False,
        _evalf=False,
        order=None,
        divergence_retries: int = 0,
    ):
        """
        Generates solution to constructed system.

        Params
        ------
        zero_init_guess:
            If `True`, a zero initial guess will be used for the
            system solution. Otherwise, the current values of `self.u` will be used.
        divergence_retries:
            If SNES reports DIVERGED, retry with warm start up to this
            many times. 0 preserves legacy behaviour.
        """

        if order is None or order > self._order:
            order = self._order

        if timestep is not None and timestep != self.delta_t:
            self.delta_t = timestep  # this will force an initialisation because the functions need to be updated

        if _force_setup:
            self._needs_function_rewire = True

        if not self.constitutive_model._solver_is_setup:
            self._needs_function_rewire = True
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

        # Override AM coefficients if flux_order is explicitly set
        if self._flux_order is not None:
            from underworld3.systems.ddt import _update_am_values
            fo = min(self._flux_order, self.DFDt.effective_order)
            _update_am_values(self.DFDt._am_coeffs, fo, 0.5)

        if uw.mpi.rank == 0 and verbose:
            print(f"NS solver - solve Stokes flow", flush=True)

        super().solve(
            zero_init_guess,
            _force_setup=_force_setup,
            verbose=verbose,
            picard=0,
            divergence_retries=divergence_retries,
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
        Estimate an appropriate timestep for the Navier-Stokes solver.

        This is an implicit solver, so the returned :math:`\delta t` should
        be interpreted as:

        - :math:`\delta t_{\textrm{diff}}`: typical time for vorticity diffusion across an element
        - :math:`\delta t_{\textrm{adv}}`: typical element-crossing time for a fluid parcel

        The Navier-Stokes equations include momentum diffusion via kinematic
        viscosity :math:`\nu = \eta/\rho`, so the diffusive timestep is computed
        from this quantity.

        Returns
        -------
        tuple
            (:math:`\delta t_{\textrm{diff}}`, :math:`\delta t_{\textrm{adv}}`)
        """

        ### required modules
        from mpi4py import MPI

        comm = uw.mpi.comm

        # For Navier-Stokes, diffusivity is the kinematic viscosity: ν = η/ρ
        # (the unified .K property of the constitutive model returns viscosity)
        diffusivity_glob = _global_max_diffusivity(
            self.constitutive_model.K, self.mesh)

        ### get the velocity values at element centroids.
        # ensure_2d=False preserves the historical reduction below, which
        # operates on the raw (un-squeezed) evaluate result.
        vel = _centroid_velocities_nd(
            self.u.sym, self.mesh, basis=self.mesh.N, ensure_2d=False)

        ### get global velocity from velocity field
        # TODO(DESIGN): if evaluate returns the (N, 1, dim) packed shape here,
        # this axis-1 norm reduces over the singleton axis, making max_magvel
        # the maximum |component| rather than the maximum vector magnitude
        # (the other estimate_dt implementations squeeze first). Preserved
        # as-is (Wave D is behaviour-neutral); revisit with a numerical check.
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
