r"""
Constitutive models for Underworld3 solvers.

This module provides constitutive relationships that define how material
properties (viscosity, diffusivity, etc.) relate fluxes to gradients of
unknowns. These models are plugged into SNES solvers to complete the
governing equations.

Classes
-------
Constitutive_Model
    Base class for all constitutive models.
ViscousFlowModel
    Isotropic viscous flow with scalar or tensor viscosity.
ViscoPlasticFlowModel
    Viscous flow with yield stress (plastic behavior).
ViscoElasticPlasticFlowModel
    Combined viscous, elastic, and plastic rheology.
DiffusionModel
    Scalar diffusion (heat, chemical species).
DarcyFlowModel
    Porous media flow (Darcy's law).
TransverseIsotropicFlowModel
    Anisotropic viscosity with directional weakness.
MultiMaterialConstitutiveModel
    Level-set weighted composite of multiple materials.

See Also
--------
underworld3.systems.solvers : Solvers that use these constitutive models.
"""

from typing_extensions import Self
import sympy
from sympy import sympify
from sympy.vector import gradient, divergence
import numpy as np

from typing import Optional, Callable
from typing import NamedTuple, Union

from petsc4py import PETSc

import underworld3 as uw
import underworld3.timing as timing
import underworld3.cython
from underworld3.utilities._api_tools import uw_object
from underworld3.swarm import IndexSwarmVariable
from underworld3.discretisation import MeshVariable
from underworld3.systems.ddt import SemiLagrangian as SemiLagrangian_DDt
from underworld3.systems.ddt import _bdf_coefficients
from underworld3.function.quantities import UWQuantity
from underworld3.systems.ddt import Lagrangian as Lagrangian_DDt

from underworld3.function import expression as public_expression

expression = lambda *x, **X: public_expression(*x, _unique_name_generation=True, **X)


class _ParameterBase:
    """Base class for all constitutive model ``_Parameters`` containers.

    All ``_Parameters`` nested classes must inherit from this.  It provides a
    ``__setattr__`` guard that **rejects** any public attribute assignment
    that doesn't match a defined ``Parameter`` descriptor or ``@property``
    on the class.  Without this guard, a typo like
    ``Parameters.viscocity = 1`` silently creates an instance attribute
    that the solver never reads — producing wrong results with no error.

    How to define parameters in a new constitutive model
    ----------------------------------------------------
    1. Inherit from ``_ParameterBase`` (and ``_ViscousParameterAlias`` if
       the model has a ``shear_viscosity_0`` parameter)::

           class _Parameters(_ParameterBase, _ViscousParameterAlias):
               ...

    2. Define each parameter as a **class-level** ``Parameter`` descriptor.
       The **attribute name IS the user API name** — users will set it via
       ``model.Parameters.<attribute_name> = value``::

           import underworld3.utilities._api_tools as api_tools

           shear_viscosity_0 = api_tools.Parameter(
               r"\\eta",              # LaTeX display name (cosmetic only)
               lambda self: 1,        # default value factory
               "Shear viscosity",     # description
               units="Pa*s",          # expected units
           )

    3. To add a **convenience alias** (e.g. ``viscosity`` → ``shear_viscosity_0``),
       either use a mixin like ``_ViscousParameterAlias`` or define a
       ``@property`` with getter and setter on the ``_Parameters`` class.
       The guard recognises both descriptors and properties.
    """

    @staticmethod
    def _list_valid_parameters(cls_type):
        """List valid parameter names for error messages."""
        from underworld3.utilities._api_tools import ExpressionDescriptor

        valid = []
        for cls in cls_type.__mro__:
            for k, v in cls.__dict__.items():
                if isinstance(v, (ExpressionDescriptor, property)) and k not in valid:
                    valid.append(k)
        return valid

    def __setattr__(self, name, value):
        # Private/internal attributes are always allowed
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        from underworld3.utilities._api_tools import ExpressionDescriptor

        # Walk the MRO looking for a matching descriptor or property
        for cls in type(self).__mro__:
            if name in cls.__dict__:
                attr = cls.__dict__[name]
                if isinstance(attr, ExpressionDescriptor):
                    # Valid descriptor — let it handle the set
                    attr.__set__(self, value)
                    return
                elif isinstance(attr, property):
                    if attr.fset is not None:
                        attr.fset(self, value)
                        return
                    raise AttributeError(
                        f"Parameter '{name}' is read-only"
                    )

        # Not a known descriptor — likely a name mismatch bug
        valid = _ParameterBase._list_valid_parameters(type(self))

        raise AttributeError(
            f"No parameter '{name}' on {type(self).__name__}. "
            f"Valid parameters: {valid}"
        )


class _ViscousParameterAlias:
    """Mixin providing ``viscosity`` as a read/write alias for ``shear_viscosity_0``.

    Add this to the inheritance of any ``_Parameters`` class that defines
    a ``shear_viscosity_0`` descriptor, so that the established
    ``Parameters.viscosity`` API continues to work::

        class _Parameters(_ParameterBase, _ViscousParameterAlias):
            shear_viscosity_0 = api_tools.Parameter(...)

    To create similar aliases for other parameters, define a ``@property``
    with a setter — the ``_ParameterBase`` guard recognises properties
    automatically.
    """

    @property
    def viscosity(self):
        return self.shear_viscosity_0

    @viscosity.setter
    def viscosity(self, value):
        self.shear_viscosity_0 = value


# How do we use the default here if input is required ?
def validate_parameters(symbol, input, default=None, allow_number=True, allow_expression=True):
    """Convert input to a UWexpression for use in constitutive models.

    Parameters
    ----------
    symbol : str
        LaTeX symbol for display (e.g., r"\\eta" for viscosity).
    input : various
        Value to convert (UWexpression, UWQuantity, float, int, sympy expr).
    default : optional
        Default value if input is None.
    allow_number : bool
        If True, accept plain numbers (int/float).
    allow_expression : bool
        If True, accept raw sympy expressions.

    Returns
    -------
    UWexpression or None
        Wrapped expression, or None if conversion failed.
    """
    # CRITICAL: Check for UWexpression FIRST, before checking sympy.Basic
    # UWexpression inherits from sympy.Symbol, so it would match the Basic check
    # and cause double-wrapping, losing unit information
    from .function.expressions import UWexpression
    if isinstance(input, UWexpression):
        # Already a UWexpression - return as-is, no wrapping needed
        return input

    elif isinstance(input, UWQuantity):
        # Convert UWQuantity to UWexpression - this is the beautiful symmetry!
        # The UWexpression constructor will handle unit conversion automatically
        input = expression(
            symbol,
            input,
            f"(converted from UWQuantity with units {input.units if input.has_units else 'dimensionless'})",
        )

    elif allow_number and isinstance(input, (float)):
        # print(f"{symbol}: Converting number to uw expression {input}")
        input = expression(symbol, input, "(converted from float)")

    elif allow_number and isinstance(input, (int)):
        # print(f"{symbol}: Converting number to uw expression {input}")
        input = expression(symbol, input, "(converted from int)")

    elif allow_expression and isinstance(input, sympy.core.basic.Basic):
        # print(f"{symbol}: Converting sympy fn to uw expression {input}")
        input = expression(symbol, input, "(imported sympy expression)")

    elif input is None and default is not None:
        input = expression(symbol, default, "(default value)")

    else:
        # That's about all we can fix automagically
        print(f"Unable to set parameter: {symbol} from {input}")
        print(f"An underworld `expression`, `UWQuantity`, or `function` is required", flush=True)
        return None

    return input


class Constitutive_Model(uw_object):
    r"""
    Base class for constitutive laws relating gradients to fluxes.

    Constitutive laws relate gradients in the unknowns to fluxes of quantities
    (for example, heat fluxes are related to temperature gradients through a
    thermal conductivity). This class is a base class for building Underworld
    constitutive laws.

    In a scalar problem, the relationship is:

    .. math::

        q_i = k_{ij} \frac{\partial T}{\partial x_j}

    and the constitutive parameters describe :math:`k_{ij}`. The template
    assumes :math:`k_{ij} = \delta_{ij}`.

    In a vector problem (such as the Stokes problem), the relationship is:

    .. math::

        t_{ij} = c_{ijkl} \frac{\partial u_k}{\partial x_l}

    but is usually written to eliminate the anti-symmetric part of the
    displacement or velocity gradients:

    .. math::

        t_{ij} = c_{ijkl} \frac{1}{2} \left[ \frac{\partial u_k}{\partial x_l}
        + \frac{\partial u_l}{\partial x_k} \right]

    and the constitutive parameters describe :math:`c_{ijkl}`. The template
    assumes :math:`k_{ij} = \frac{1}{2}(\delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk})`
    which is the 4th rank identity tensor accounting for symmetry in the flux
    and the gradient terms.
    """

    # Class-level instance counter for automatic symbol uniqueness across all constitutive models
    _global_instance_count = 0
    # Per-class instance counters for class-specific numbering
    _class_instance_counts = {}

    @timing.routine_timer_decorator
    def __init__(self, unknowns, material_name: str = None):
        """
        Initialize a constitutive model.

        Parameters
        ----------
        unknowns : UnknownSet
            The solver's unknowns (velocity, pressure, etc.)
        material_name : str, optional
            A distinguishing name for this material's symbols.
            If provided, symbols will be subscripted: η → η_{name}
            Useful when bundling multiple models in MultiMaterialModel.
        """
        # Define / identify the various properties in the class but leave
        # the implementation to child classes. The constitutive tensor is
        # defined as a template here, but should be instantiated via class
        # properties as required.

        # We provide a function that converts gradients / gradient history terms
        # into the relevant flux term.

        # Store material name for symbol disambiguation
        self._material_name = material_name

        # Track instance numbers for automatic symbol uniqueness
        Constitutive_Model._global_instance_count += 1
        self._global_instance_number = Constitutive_Model._global_instance_count

        # Track per-class instance numbers (0-based indexing)
        class_name = self.__class__.__name__
        if class_name not in Constitutive_Model._class_instance_counts:
            Constitutive_Model._class_instance_counts[class_name] = 0
        self._class_instance_number = Constitutive_Model._class_instance_counts[class_name]
        Constitutive_Model._class_instance_counts[class_name] += 1

        # Backing attribute for settable flux_jacobian (set to a custom
        # SymPy expression to override the Jacobian tangent independently
        # of the residual flux).  None by default, meaning the solver
        # differentiates the exact flux.
        self._flux_jacobian = None

        self.Unknowns = unknowns

        u = self.Unknowns.u
        self._DFDt = self.Unknowns.DFDt
        self._DuDt = self.Unknowns.DuDt

        # Constitutive tensors relate gradients to fluxes; both live in
        # the embedded coordinate space (cdim-dimensional), not the
        # topological one. ``u.sym.jacobian(mesh.N)`` produces a
        # (u_dim x cdim) matrix, so the conductivity / viscosity
        # tensor must also be cdim-sized to multiply against it. For
        # volume meshes ``dim == cdim`` so this is a no-op; for
        # manifold meshes (e.g. SphericalManifold: dim=2, cdim=3) the
        # constitutive tensor correctly acts on the 3-component flux.
        self.dim = u.mesh.cdim
        self.u_dim = u.num_components

        self.Parameters = self._Parameters(self)
        self.Parameters._solver = None
        self.Parameters._reset = self._reset
        self._material_properties = None

        ## Default consitutive tensor is the identity

        if self.u_dim == 1:
            self._c = sympy.Matrix.eye(self.dim)
        else:  # vector problem
            self._c = uw.maths.tensor.rank4_identity(self.dim)

        self._K = sympy.sympify(1)
        self._C = None

        self._reset()

        super().__init__()

    def create_unique_symbol(self, base_symbol, value, description):
        """
        Create a unique symbol name for constitutive model parameters.

        Symbol naming priority:
        1. If material_name is set: η → η_{material_name}
        2. Else if multiple instances of same class: η → η^{(n)}
        3. Else: use base symbol as-is

        Parameters
        ----------
        base_symbol : str
            The base LaTeX symbol name (e.g., r"\\eta", r"\\kappa")
        value : float or expression
            The initial value for the symbol
        description : str
            Description of the parameter

        Returns
        -------
        UWexpression
            Expression with unique symbol name
        """
        # Priority 1: User-specified material name (subscript notation)
        if self._material_name is not None:
            symbol_name = rf"{{{base_symbol}}}_{{\mathrm{{{self._material_name}}}}}"
        # Priority 2: Multiple instances of same class (superscript notation)
        elif self._class_instance_number > 0:
            symbol_name = rf"{{{base_symbol}}}^{{({self._class_instance_number})}}"
        # Priority 3: First/only instance - clean symbol
        else:
            symbol_name = base_symbol

        return expression(symbol_name, value, description)

    class _Parameters(_ParameterBase):
        """Any material properties that are defined by a constitutive relationship are
        collected in the parameters which can then be defined/accessed by name in
        individual instances of the class.
        """

        def __init__(inner_self, _owning_model):
            inner_self._owning_model = _owning_model
            return

    @property
    def Unknowns(self):
        r"""Reference to the solver's unknown fields.

        Returns
        -------
        Unknowns
            Container holding the primary unknown field(s) (e.g., velocity,
            pressure, temperature) that this constitutive model operates on.
        """
        return self._Unknowns

    # We probably should not be changing this ever ... does this setter even belong here ?
    @Unknowns.setter
    def Unknowns(self, unknowns):
        """Set the solver unknowns (invalidates setup)."""
        self._Unknowns = unknowns
        self._solver_is_setup = False
        return

    @property
    def K(self):
        r"""Primary constitutive property (viscosity, diffusivity, etc.).

        Returns
        -------
        UWexpression
            The material property defining the flux-gradient relationship.
        """
        return self._K

    @property
    def u(self):
        r"""The primary unknown field from the solver.

        Returns
        -------
        MeshVariable
            The unknown field (velocity, temperature, etc.).
        """
        return self.Unknowns.u

    @property
    def grad_u(self):
        r"""Gradient of the unknown field.

        For scalar fields, this is a vector. For vector fields (velocity),
        this is the velocity gradient tensor :math:`\nabla \mathbf{u}`.

        Returns
        -------
        sympy.Matrix
            Gradient/Jacobian of the unknown field.
        """
        mesh = self.Unknowns.u.mesh
        # return mesh.vector.gradient(self.Unknowns.u.sym)
        return self.Unknowns.u.sym.jacobian(mesh.CoordinateSystem.N)

    @property
    def DuDt(self):
        r"""Material derivative operator for the unknown field.

        Used in time-dependent problems to track Lagrangian or
        semi-Lagrangian derivatives.

        Returns
        -------
        SemiLagrangian_DDt or Lagrangian_DDt or None
            The material derivative operator, or None if not set.
        """
        return self._DuDt

    @DuDt.setter
    def DuDt(
        self,
        DuDt_value: Union[SemiLagrangian_DDt, Lagrangian_DDt],
    ):
        """Set the material derivative operator for the unknown."""
        self._DuDt = DuDt_value
        self._solver_is_setup = False
        return

    @property
    def DFDt(self):
        """Material derivative operator for the flux history."""
        return self._DFDt

    # Do we want to lock this down ?
    @DFDt.setter
    def DFDt(
        self,
        DFDt_value: Union[SemiLagrangian_DDt, Lagrangian_DDt],
    ):
        """Set the material derivative operator for flux history."""
        self._DFDt = DFDt_value
        self._solver_is_setup = False
        return

    ## Properties on all sub-classes

    @property
    def C(self):
        """The matrix form of the constitutive model (the `c` property)
        that relates fluxes to gradients.
        For scalar problem, this is the matrix representation of the rank 2 tensor.
        For vector problems, the Mandel form of the rank 4 tensor is returned.
        NOTE: this is an immutable object that is _a view_ of the underlying tensor
        """
        if not self._is_setup:
            self._build_c_tensor()

        d = self.dim
        rank = len(self.c.shape)

        if rank == 2:
            return sympy.Matrix(self._c).as_immutable()
        else:
            return uw.maths.tensor.rank4_to_mandel(self._c, d).as_immutable()

    @property
    def c(self):
        """The tensor form of the constitutive model that relates fluxes to gradients. In scalar
        problems, `c` and `C` are equivalent (matrices), but in vector problems, `c` is a
        rank 4 tensor. NOTE: `c` is the canonical form of the constitutive relationship.
        """

        if not self._is_setup:
            self._build_c_tensor()
        if hasattr(self._c, "sym"):
            return sympy.Matrix(self._c.sym).as_immutable()
        else:
            return self._c.as_immutable()

    @property
    def flux(self):
        """Computes the effect of the constitutive tensor on the gradients of the unknowns.
        (always uses the `c` form of the tensor). In general cases, the history of the gradients
        may be required to evaluate the flux.
        """

        ddu = self.grad_u

        return self._q(ddu)

    def _q(self, ddu):
        """Generic flux term"""

        if not self._is_setup:
            self._build_c_tensor()

        c = self.c
        rank = len(c.shape)

        # tensor multiplication

        if rank == 2:
            flux = c * ddu.T
        else:  # rank==4
            flux = sympy.tensorcontraction(
                sympy.tensorcontraction(sympy.tensorproduct(c, ddu), (1, 5)), (0, 3)
            )

        return sympy.Matrix(flux)

    @property
    def flux_1d(self):
        """Computes the effect of the constitutive tensor on the gradients of the unknowns.
        (always uses the `c` form of the tensor). In general cases, the history of the gradients
        may be required to evaluate the flux. Returns the Voigt form that is flattened so as to
        match the PETSc field storage pattern for symmetric tensors.
        """

        flux = self.flux

        if flux.shape[0] == 1:
            return flux

        if flux.shape[1] == 1:
            return flux.T

        assert (
            flux.is_symmetric()
        ), "The conversion of tensors to Voigt form is only defined for symmetric tensors in underworld\
            but for non-symmetric tensors, the .flat() method is a potential replacement"

        return uw.maths.tensor.rank2_to_voigt(flux, dim=self.dim)

    @property
    def flux_jacobian(self):
        """Optional smooth surrogate flux for Jacobian assembly.

        Returns ``None`` by default, meaning the solver differentiates the
        exact :attr:`flux` (the Newton fix unwraps it first; a generic Min/Max
        kink-smoothing fallback then rounds any remaining yield kink).

        Set this to a custom SymPy expression to override the Jacobian tangent
        independently of the residual flux.  The residual still uses the exact
        :attr:`flux`, so the converged solution satisfies the true constitutive
        law — only the Newton search direction is smoothed, giving a robust,
        line-search-friendly tangent without changing the answer.

        Use cases:

        * A model whose flux has a non-smooth yield kink (e.g. hard-``Min``
          viscoplasticity) supplies a physically-motivated *smooth law for the
          tangent only*.

        * A model whose flux contains composition-dependent coefficients
          (e.g. multicomponent diffusion) may supply a *constant-coefficient*
          version to give the solver a clean SPD Jacobian while keeping the
          exact physics in the residual.

        Shape must match :attr:`flux` (the solver substitutes it for ``F1`` when
        forming the velocity-gradient Jacobian blocks).
        """
        return self._flux_jacobian

    @flux_jacobian.setter
    def flux_jacobian(self, value):
        """Set a custom Jacobian flux expression (or None to revert)."""
        self._flux_jacobian = value
        # Signal the solver to rebuild its pointwise Jacobian function.
        # Without this, the change is silently ignored until something
        # else triggers a re-setup.
        self._solver_is_setup = False

    def _reset(self):
        """Flags that the expressions in the consitutive tensor need to be refreshed and also that the
        solver will need to rebuild the stiffness matrix and jacobians"""

        self._solver_is_setup = False
        self._is_setup = False

        # Propagate invalidation to solver if we have a reference.
        # A constitutive parameter change affects the F1 pointwise function
        # only — the DM, fields, and BCs are unchanged. The solver's
        # _build() can swap functions in place without DM teardown.
        if hasattr(self, "Parameters") and hasattr(self.Parameters, "_solver"):
            if self.Parameters._solver is not None:
                self.Parameters._solver._needs_function_rewire = True

        return

    @property
    def requires_stress_history(self):
        """Whether this model needs DFDt stress history tracking.

        Models that return True require a solver with stress history
        management (e.g. VE_Stokes). Assigning such a model to a plain
        Stokes solver will raise an error.
        """
        return False

    @property
    def supports_yield_homotopy(self):
        """Whether this model can be solved by a single-parameter yield homotopy.

        ``True`` on the yielding models, which carry a δ-parameterised soft-min
        yield law that sharpens to the exact ``Min`` as δ→0 — see
        :meth:`_yield_homotopy_control`. ``solver.solve(homotopy=True)`` refuses a
        model that returns ``False`` (a purely viscous model has no yield surface to
        sharpen, so there is nothing to march).
        """
        return False

    @property
    def stress_history_ddt_kwargs(self):
        """Extra kwargs passed to the auto-DDt creation when this model
        triggers it via ``requires_stress_history = True``.

        Default: empty dict (BDF-only models). ETD-2 / exponential models
        override to inject ``with_forcing_history=True`` so the DDt
        allocates a forcing-history slot.
        """
        return {}

    def _update_history_coefficients(self):
        """Uniform pre-solve hook the Stokes solver calls before each solve.

        Default: no-op (non-stress-history models have nothing to update).
        BDF-style stress-history subclasses (VEP, TI-VEP) override to
        delegate to ``_update_bdf_coefficients``. ETD-2 / exponential
        subclasses (e.g. ``MaxwellExponentialFlowModel``) override to
        update α, φ on the DDt. The solver dispatches uniformly through
        this method — no ``isinstance`` checks at the solver layer.
        """
        return

    def _update_history_post_solve(self):
        """Uniform post-solve hook the Stokes solver calls after each solve.

        Default: no-op. Subclasses that store extra integrator state for
        the next step (e.g. ETD-2 storing ε̇ⁿ in ``forcing_star``) override.
        """
        return

    @property
    def plastic_fraction(self):
        """Fraction of strain rate that is plastic (0 for non-plastic models).

        Returns a sympy expression that can be evaluated post-solve via
        ``uw.function.evaluate(cm.plastic_fraction, coords)``.
        """
        return sympy.Integer(0)

    def _build_c_tensor(self):
        """Return the identity tensor of appropriate rank (e.g. for projections)"""

        self._c = self._K * uw.maths.tensor.rank4_identity(self.dim)
        self._is_setup = True

        return

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display
        from textwrap import dedent

        display(
            Markdown(
                rf"This consititutive model is formulated for {self.dim} dimensional equations"
            )
        )


class ViscousFlowModel(Constitutive_Model):
    r"""
    Viscous flow constitutive model for Stokes-type solvers.

    Defines the relationship between deviatoric stress and strain rate:

    .. math::

        \tau_{ij} = \eta_{ijkl} \cdot \frac{1}{2} \left[ \frac{\partial u_k}{\partial x_l}
        + \frac{\partial u_l}{\partial x_k} \right]

    where :math:`\eta` is the viscosity, which can be a scalar constant, SymPy
    function, Underworld mesh variable, or any valid combination. This results
    in an isotropic (but not necessarily homogeneous or linear) relationship
    between :math:`\tau` and the velocity gradients.

    Parameters
    ----------
    unknowns : Unknowns
        The solver unknowns (typically velocity and pressure fields).
    material_name : str, optional
        Name identifier for this material (used in multi-material setups).

    Examples
    --------
    >>> import underworld3 as uw
    >>> stokes = uw.systems.Stokes(mesh)
    >>> viscous = uw.constitutive_models.ViscousFlowModel(stokes.Unknowns)
    >>> viscous.Parameters.shear_viscosity_0 = 1e21  # Pa.s
    >>> stokes.constitutive_model = viscous

    See Also
    --------
    ViscoPlasticFlowModel : Adds yield stress for plastic behavior.
    ViscoElasticPlasticFlowModel : Adds viscoelastic memory.
    """

    #     ```python
    # class ViscousFlowModel(Constitutive_Model)
    # ...
    # ```
    # ### Example

    # ```python
    # viscous_model = ViscousFlowModel(dim)
    # viscous_model.material_properties = viscous_model.Parameters(viscosity=viscosity_fn)
    # solver.constititutive_model = viscous_model
    # ```

    # ```python
    # tau = viscous_model.flux(gradient_matrix)
    # ```

    def __init__(self, unknowns, material_name: str = None):
        # All this needs to do is define the
        # viscosity property and init the parent(s)
        # In this case, nothing seems to be needed.
        # The viscosity is completely defined
        # in terms of the Parameters

        super().__init__(unknowns, material_name=material_name)

        # self._viscosity = expression(
        #     R"{\eta_0}",
        #     1,
        #     " Apparent viscosity",
        # )

    class _Parameters(_ParameterBase, _ViscousParameterAlias):
        """Any material properties that are defined by a constitutive relationship are
        collected in the parameters which can then be defined/accessed by name in
        individual instances of the class.

        Now uses Parameter descriptor pattern for automatic lazy evaluation preservation
        with unit-aware quantities.
        """

        # Import Parameter descriptor (must use absolute import inside nested class)
        import underworld3.utilities._api_tools as api_tools

        # Define shear_viscosity_0 as a Parameter descriptor
        # The lambda receives the _Parameters instance and creates the expression via the owning model
        shear_viscosity_0 = api_tools.Parameter(
            r"\eta",
            lambda params_instance: params_instance._owning_model.create_unique_symbol(
                r"\eta", 1, "Shear viscosity"
            ),
            "Shear viscosity",
            units="Pa*s"
        )

        def __init__(
            inner_self,
            _owning_model,
        ):
            inner_self._owning_model = _owning_model
            # Note: shear_viscosity_0 is now a descriptor, no need to create it here

    @property
    def viscosity(self):
        """Whatever the consistutive model defines as the effective value of viscosity
        in the form of an uw.expression"""

        return self.Parameters.shear_viscosity_0

    @property
    def K(self):
        """Effective stiffness parameter (viscosity for viscous flow)"""
        return self.viscosity

    @property
    def flux(self):
        r"""Viscous stress tensor: :math:`\boldsymbol{\tau} = 2\eta\dot{\varepsilon}`."""
        edot = self.grad_u
        return self._q(edot)

    def _q(self, edot):
        """Apply constitutive tensor to strain rate to compute stress."""

        if not self._is_setup:
            self._build_c_tensor()

        c = self.c
        rank = len(c.shape)

        # tensor multiplication

        if rank == 2:
            flux = c * edot
        else:  # rank==4
            flux = sympy.tensorcontraction(
                sympy.tensorcontraction(sympy.tensorproduct(c, edot), (1, 5)), (0, 3)
            )

        return sympy.Matrix(flux)

    ## redefine the gradient for the viscous law as it relates to
    ## the symmetric part of the tensor only

    @property
    def grad_u(self):
        r"""Symmetric strain rate tensor (with 1/2 factor).

        .. math::
            \dot{\varepsilon}_{ij} = \frac{1}{2}\left(\frac{\partial u_i}{\partial x_j}
            + \frac{\partial u_j}{\partial x_i}\right)
        """
        mesh = self.Unknowns.u.mesh

        return mesh.vector.strain_tensor(self.Unknowns.u.sym)

        # ddu = self.Unknowns.u.sym.jacobian(mesh.CoordinateSystem.N)
        # edot = (ddu + ddu.T) / 2
        # return edot

    @property
    def plastic_fraction(self):
        """Fraction of strain rate that is plastic: 1 - η_vp / η_viscous."""
        return sympy.Max(0, 1 - self.viscosity / self.Parameters.shear_viscosity_0)

    def _build_c_tensor(self):
        """For this constitutive law, we expect just a viscosity function"""

        if self._is_setup:
            return

        d = self.dim
        viscosity = self.viscosity

        # Check for tensor forms first (Mandel matrix or full rank-4 tensor)
        dv = uw.maths.tensor.idxmap[d][0]
        if isinstance(viscosity, sympy.Matrix) and viscosity.shape == (dv, dv):
            # Mandel form of constitutive tensor
            self._c = 2 * uw.maths.tensor.mandel_to_rank4(viscosity, d)
        elif isinstance(viscosity, sympy.Array) and viscosity.shape == (d, d, d, d):
            # Full rank-4 tensor
            self._c = 2 * viscosity
        else:
            # Scalar viscosity case
            # UWexpression has __getitem__ from MathematicalMixin, making it Iterable,
            # which causes SymPy's array multiplication operator to reject it.
            # Solution: Use element-wise loop construction instead of operator overloading.
            # The multiplication creates Mul(scalar, UWexpression) objects which are NOT
            # Iterable, so array assignment accepts them. JIT unwrapper finds the
            # UWexpression atoms inside and substitutes correctly.

            identity = uw.maths.tensor.rank4_identity(d)
            result = sympy.MutableDenseNDimArray.zeros(d, d, d, d)

            # Element-wise multiplication: c_ijkl = 2 * I_ijkl * viscosity
            for i in range(d):
                for j in range(d):
                    for k in range(d):
                        for l in range(d):
                            val = 2 * identity[i, j, k, l] * viscosity
                            # If simplification returns bare UWexpression (e.g., 2*(1/2)*visc = visc),
                            # wrap it to avoid Iterable check failure during assignment
                            if hasattr(val, '__getitem__') and not isinstance(val, (sympy.MatrixBase, sympy.NDimArray)):
                                val = sympy.Mul(sympy.S.One, val, evaluate=False)
                            result[i, j, k, l] = val

            self._c = result

        self._is_setup = True
        self._solver_is_setup = False

        return

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        ## feedback on this instance
        display(
            Latex(
                r"$\quad\eta_\textrm{eff} = $ " + sympy.sympify(self.viscosity.sym)._repr_latex_()
            )
        )

    # --- Yield soft-min smoother (shared by the visco-plastic subclasses) -----------
    # The δ soft-min regularisation and the smooth-min FAMILY selection live on the
    # base class so every yielding model inherits one implementation. δ is held as a
    # constants[] UWexpression atom (not a baked float) so a homotopy can ramp it at
    # runtime via PetscDSSetConstants with no JIT recompile.

    def _get_yield_softness(self):
        r"""The soft-min regularisation δ as a ``constants[]`` UWexpression atom.

        Created lazily and kept in sync with ``self._yield_softness`` (the
        configured numeric value).  Storing δ as a UWexpression — rather than
        baking the float into the compiled flux — lets a yield homotopy ramp
        δ at runtime via ``PetscDSSetConstants`` with no JIT recompile.
        ``δ = 0`` makes the sqrt law identically ``Min``.
        """
        delta_value = getattr(self, "_yield_softness", 0.0)
        if getattr(self, "_yield_softness_expr", None) is None:
            self._yield_softness_expr = expression(
                R"{\updelta_{y}}",
                sympy.Float(delta_value),
                "Yield soft-min regularisation δ (rampable constant; δ=0 ⇒ exact Min)",
            )
            # The offset fixes WHERE the smoothed curve is pinned to the exact law.
            # Held as its OWN constant atom — a single symbol in the stress tensor —
            # so it does not blow the tensor up, while still tracking δ symbolically
            # (one δ update repacks both constants). See `yield_anchor`.
            self._yield_offset_expr = expression(
                R"{\updelta_{y,0}}",
                self._yield_offset_for_anchor(),
                "Yield soft-min offset; tracks δ so the chosen anchor is exact",
            )
        else:
            self._yield_softness_expr.sym = sympy.Float(delta_value)
        return self._yield_softness_expr

    def _yield_offset_for_anchor(self):
        r"""The sqrt soft-min offset that pins the chosen anchor, symbolic in δ."""
        delta = self._yield_softness_expr
        if getattr(self, "_yield_anchor", "onset") == "yield":
            return delta / 2
        return (-1 + sympy.sqrt(1 + delta**2)) / 2

    def _get_yield_offset(self):
        """The offset constant atom (lazily created alongside δ)."""
        if getattr(self, "_yield_offset_expr", None) is None:
            self._get_yield_softness()
        return self._yield_offset_expr

    @property
    def yield_anchor(self):
        r"""Which point of the ``"softmin"`` yield law is pinned to the exact ``Min``.

        A soft-min rounds the yield corner by δ, and rounding a corner moves the whole
        curve — so the family has one free constant, fixed by choosing WHICH point the
        smoothed law must reproduce exactly. Both choices apply to both smoother
        families, and both recover ``Min`` as :math:`\delta \to 0`; they differ only in
        which SIDE of the exact law the curve sits on in between. Writing
        :math:`f = \eta_{ve} / \eta_{pl}` for the overstress ratio (so :math:`f = 1` is
        the yield point and :math:`f < 1` is unyielded):

        ``"onset"`` (default, historical)
            Pins the unyielded limit :math:`f \to 0`, where :math:`\eta = \eta_{ve}`
            exactly. The curve then sits BELOW the exact law at and above the yield
            point — the sqrt family undershoots for :math:`f < 2` (at
            :math:`\delta = 64`, :math:`\eta/\eta_{ve} = 0.80` at :math:`f = 0.5` where
            the exact law gives 1.0, and the yield point itself is a third
            under-stressed, :math:`\tau/\tau_y = 0.67`), and the power mean degenerates
            to the p-norm :math:`(\eta_{ve}^{-s} + \eta_{pl}^{-s})^{-1/s}`, which falls
            away to :math:`\eta \to 0` as δ grows. So neither family approaches the
            yield surface from above under this anchor. That matters whenever the
            problem's overstress ratio is O(1), because then the smoothed problem is
            WEAKER than the sharp one it is supposed to be an easy version of.

        ``"yield"``
            Pins :math:`f = 1`, so :math:`\tau/\tau_y = 1` exactly at the yield point
            for every δ and the curve is :math:`\ge` the exact law everywhere — a
            genuine approach from above, which is the safe direction for a homotopy
            entry. The sqrt family gets there with the offset :math:`\delta/2`; the
            power mean needs no offset at all, since averaging the two terms,
            :math:`\eta = ((\eta_{ve}^{-s} + \eta_{pl}^{-s})/2)^{-1/s}`, makes it a
            generalised mean, and a generalised mean returns the common value when its
            arguments are equal. The cost is stiffened unyielded material, bounded by a
            factor 2 for the sqrt family and by :math:`2^{\delta}` for the power mean,
            both decaying to 1 as :math:`\delta \to 0`. The power-mean bound is why its
            entry δ is O(1) and not O(10).
        """
        return getattr(self, "_yield_anchor", "onset")

    @yield_anchor.setter
    def yield_anchor(self, value):
        if value not in ("onset", "yield"):
            raise ValueError(
                f"yield_anchor must be 'onset' or 'yield', got {value!r}")
        self._yield_anchor = value
        # The sqrt family's offset atom holds the anchor's FORMULA, so it must be
        # rebuilt, not merely revalued — and _get_yield_softness only builds it
        # alongside δ. (The power mean carries the anchor in _combine_yield itself, so
        # for that family the _reset is the whole of the update.)
        self._yield_offset_expr = None
        self._yield_softness_expr = None
        self._reset()

    def _combine_yield(self, eta_ve, eta_pl):
        r"""Combine the visco-elastic/viscous viscosity ``eta_ve`` with the plastic
        (yield) viscosity ``eta_pl`` according to ``self._yield_mode``:

        - ``"min"``: exact hard ``Min(η_ve, η_pl)`` (sharp yield surface).
        - ``"harmonic"``: ``1/(1/η_ve + 1/η_pl)`` (a distinct smooth blend).
        - ``"softmin"``: the δ-parameterised soft-min, in the family chosen by
          ``self.yield_smoother`` (``"sqrt"`` or ``"powermean"``) and on the side of
          exact ``Min`` chosen by ``self.yield_anchor``. Which side is the property
          that matters for a homotopy, and it belongs to the ANCHOR, not the family:
          ``"onset"`` puts both families below ``Min`` near the yield point,
          ``"yield"`` puts both on or above it.
          The ``sqrt`` family is exactly ``Min`` at ``δ = 0``; the ``powermean``
          family is not — its sharpness ``s = 1/(δ + 0.001)`` saturates at 1000, so it
          lands within a factor :math:`2^{\pm(\delta + 0.001)}` of ``Min``, i.e. 0.07 %
          at ``δ = 0``. Measured on a 45 %-yielded box, that leaves the converged
          solution satisfying the exact yield law to 6e-10 of the initial residual — an
          order of magnitude INSIDE a 1e-8 solver tolerance, so the difference is not
          observable in practice (2026-07-27).

        Behaviour at the stock settings (``yield_mode`` per model default,
        ``yield_smoother="sqrt"``) is identical to the previous inline law — δ merely
        moves from a baked float to a ``constants[]`` atom (same value).
        """
        mode = getattr(self, "_yield_mode", "softmin")
        if mode == "harmonic":
            return 1 / (1 / eta_ve + 1 / eta_pl)
        if mode == "min":
            return sympy.Min(eta_ve, eta_pl)

        # "softmin": δ-parameterised smooth-min family.
        smoother = getattr(self, "_yield_smoother", "sqrt")
        delta = self._get_yield_softness()
        f = eta_ve / eta_pl
        if smoother == "powermean":
            # Soft-min of order -s in an overflow-safe harmonic-normalised form. δ is
            # floored SMOOTHLY (+ε, not Max()) so 1/δ stays finite as δ→0 (a Max on
            # the δ atom triggers an unsupported symbolic numeric comparison).
            s = 1 / (delta + sympy.Float(0.001))
            a = 1 + f
            b = 1 + 1 / f
            # Harmonic mean written as eta_ve/(1+f), NOT eta_ve*eta_pl/(eta_ve+eta_pl).
            # The two are algebraically identical, but the product-over-sum form
            # evaluates to inf/inf = NaN when eta_pl is infinite — which is exactly
            # what a rigid (unyielded) point gives, since eta_pl = tau_y/(2 edot_II)
            # and edot_II = 0 there. That includes every point of a cold v=0 start.
            # In this form f -> 0 and N -> eta_ve, the correct viscous limit.
            N = eta_ve / a
            softmin = a ** (-s) + b ** (-s)
            if self.yield_anchor == "yield":
                # Averaging the two terms is what makes this a power MEAN rather than
                # a p-NORM, and it is the whole of the "yield" anchor for this family:
                # a generalised mean returns the common value when its arguments are
                # equal, so η = η_ve = η_pl exactly at f = 1 — no offset atom needed.
                softmin = softmin / 2
            return N * softmin ** (-1 / s)

        # default "sqrt" soft-min: η_ve / g(f), g(0)=1, g ≈ max(1, f), exact Min at δ=0.
        offset = self._get_yield_offset()
        g = 1 + (f - 1 + sympy.sqrt((f - 1) ** 2 + delta**2)) / 2 - offset
        return eta_ve / g

    @property
    def yield_smoother(self):
        r"""Which smooth-min FAMILY regularises the ``"softmin"`` yield mode.

        Both families use the same softness parameter ``δ`` (``yield_softness``)
        and approach exact ``Min`` as ``δ → 0``, but differ in how they round
        the kink:

        - ``"sqrt"`` (default): ``η_ve / g(f, δ)`` with
          ``g = 1 + ½(f−1+√((f−1)²+δ²)) − offset``, ``f = η_ve/η_pl``.  Exact
          ``Min`` at ``δ=0``.  Its smoothing authority saturates: as ``δ → ∞``,
          ``η → 2 η_ve/(f+2)``, so the most it can move a point is a factor
          ``2f/(f+2)`` — which tends to 1 as ``f → 1``.  On a problem whose
          overstress ratio is O(1) this family therefore has very little room to
          build an easier problem, however large δ is.
        - ``"powermean"``: the soft-min of order ``−s``,
          ``η_eff = (η_ve^(−s) + η_pl^(−s))^(−1/s)`` with ``s = 1/(δ + 0.001)``
          (``s→∞`` ⇒ ``Min``).  Unlike the sqrt family it does not saturate — it
          keeps moving with δ — which is what makes it usable where the sqrt family
          has no range.

        ``δ`` is NOT the same parameter in the two families and their schedules do
        not transfer: for the power mean ``s = 1/(δ + 0.001)``, so ``δ = 1`` is
        already the harmonic mean and useful entries are ``δ ≤ 1``; for the sqrt
        family δ is a percentage stress deviation and a generous entry is O(10).
        Take the entry from ``_yield_homotopy_control``, which asks the family.

        Which SIDE of ``Min`` a family sits on is set by ``yield_anchor``, not by the
        family — see that property.

        Selecting ``"powermean"`` bumps a zero ``yield_softness`` to ``1.0``
        (``s≈1``, the parameter-free harmonic mean) since ``δ=0`` is the sharp
        hard-``Min`` limit it only *approaches*.
        """
        return getattr(self, "_yield_smoother", "sqrt")

    @yield_smoother.setter
    def yield_smoother(self, value):
        if value not in ("sqrt", "powermean"):
            raise ValueError(
                f"yield_smoother must be 'sqrt' or 'powermean', got '{value}'"
            )
        self._yield_smoother = value
        if value == "powermean" and getattr(self, "_yield_softness", 0.0) == 0.0:
            # δ=0 ⇒ s=1/δ=∞ is the singular Min limit; default to the
            # parameter-free harmonic mean (s=1) instead.
            self.yield_softness = 1.0
        self._reset()

    # --- Smooth lower bounds (viscosity / yield floors) -----------------------------
    # A hard sympy.Max cutoff is non-differentiable at the corner. When the flux is
    # differentiated for the consistent-Newton tangent that kink breaks the tangent —
    # and, because one operand is a UWexpression over the fields, sympy's fuzzy `>=`
    # comparison cannot resolve it and recurses. In the smooth yield modes we therefore
    # round the floor with the same δ that regularises the yield transition, so the
    # whole effective viscosity stays differentiable and δ→0 recovers the sharp bound.

    @property
    def viscosity_min_rounding(self):
        r"""Rounding scale :math:`\epsilon` for the VISCOSITY floor, in viscosity units.

        A floor is a corner, and at :math:`\epsilon = 0` the smooth-max expression is
        algebraically exactly ``Max`` — non-differentiable, which the consistent-Newton
        tangent cannot cope with (it dies at ``nl=0, DIVERGED_LINEAR_SOLVE``). Without
        this property the only rounding available came from ``yield_mode``: zero under
        ``"min"``, and :math:`\delta\,|floor|` under the smooth modes, so the smoothing
        **vanished as δ → 0** — exactly where a yield homotopy lands.

        The floor is a different corner from the yield surface and needs a scale of its
        own. Set this to a small fraction of the floor (a few per cent is usually ample)
        and the cutoff becomes usable with ``consistent_jacobian=True`` at any δ,
        including δ = 0.

        ``None`` (the default) keeps the historical ``yield_mode``-derived behaviour, so
        nothing changes for existing models.

        Notes
        -----
        This matters for more than differentiability. A viscosity floor bounds how weak
        yielded material can become, and therefore bounds the viscosity CONTRAST the
        solution can develop — which is to say it bounds how localised the solution can
        be. A floor that is relaxed toward zero is a continuation from a diffuse,
        uniquely-solvable viscous problem toward the localised (near rigid-plastic)
        limit, tracking one branch as it sharpens. That is a solution-SELECTION
        homotopy, not a smoothness fix, and it is why the floor needs to be usable
        independently of δ.
        """
        return getattr(self, "_viscosity_min_rounding", None)

    @viscosity_min_rounding.setter
    def viscosity_min_rounding(self, value):
        self._viscosity_min_rounding = value
        self._reset()

    def _apply_floor(self, value, floor, rounding=None):
        r"""Impose the lower bound :math:`value \ge floor`.

        In ``yield_mode="min"`` this is the exact hard ``sympy.Max(value, floor)`` —
        the sharp cutoff the model has always used. In the smooth yield modes
        (``"softmin"``/``"harmonic"``) it is the differentiable ``uw.maths.smooth_max``
        rounded by the yield softness δ *relative to the floor* (:math:`\epsilon =
        \delta\,|floor|`), so the whole effective viscosity — not only the yield
        transition — is differentiable for the consistent-Newton tangent. The relative
        rounding needs a non-zero ``floor`` for a length scale, which the numerical
        viscosity/yield floors provide; a cutoff *at zero* (a tension cutoff) has no
        such scale and is rounded by its own physical parameter via
        ``uw.maths.smooth_max`` at the call site instead.
        """
        # smooth_max is 1/2 (a + b + sqrt((a-b)^2 + eps^2)) — pure arithmetic on the
        # operands, so `floor` stays the symbolic Parameter that the JIT routes
        # through constants[], and sympy is never asked for the ordering test that
        # recurses on an opaque UWexpression. At eps = 0 this is exactly
        # Max(value, floor), but written without a comparison.
        #
        # TODO(DESIGN): the rounding scale is RELATIVE (delta * floor), so it
        # collapses for a tension cutoff at floor = 0 — now the default for
        # yield_stress_min. That leaves a hard corner: the floored yield stress is
        # exactly 0 in tension, hence eta_pl = tau_y/(2 edot_II) is exactly 0 and the
        # tangent through the soft-min is undefined there. A properly rounded cap
        # (Griffith / parabolic) needs an ABSOLUTE stress scale, which this signature
        # cannot supply. Maintainer decision pending (2026-07-26).
        if rounding is None:
            rounding = 0 if getattr(self, "_yield_mode", "min") == "min" \
                else self._get_yield_softness() * floor
        return uw.maths.smooth_max(value, floor, rounding)

    # Tangent this model wants while the yield homotopy marches. Newton is right for
    # a purely viscous-plastic yield; the elastic (VEP) subclasses override to the
    # frozen/Picard tangent, because the consistent yield tangent taken across the
    # elastic stress-history block makes the Jacobian indefinite and the linear
    # solve fails outright (DIVERGED_LINEAR_SOLVE).
    _yield_homotopy_tangent = True

    #: Entry δ per soft-min family. These are NOT interchangeable numbers: for the
    #: power mean the sharpness is s = 1/(δ + 0.001), so δ ≤ 1 and δ = 1 IS the harmonic
    #: mean; for the sqrt family δ is a percentage stress deviation and a generous entry
    #: is O(10). A sqrt-family δ handed to the power mean asks for a law a factor 2^δ
    #: away from Min — 65000 at δ=16 — which is why the entry has to come from the
    #: family, not the caller.
    _YIELD_HOMOTOPY_DELTA0 = {"powermean": 1.0, "sqrt": 16.0}

    def _yield_homotopy_control(self, smoother=None, anchor=None):
        """Put this model in its smooth (δ-parameterised) yield mode and describe
        how to march it.

        The model owns what the homotopy *means* for it: which knob is the
        continuation parameter, how to set it, and which tangent to pair it with.
        The solver only marches the number. Called by
        ``solver.solve(homotopy=True)``; see
        :doc:`nonlinear-solver-homotopy-warmstart` (Layer 2).

        Defaults to the power-mean family, whose large-δ limit is the harmonic mean —
        bounded by the background viscosity even as :math:`\\dot\\varepsilon \\to 0`, so
        the first (cold) solve of the march is well posed and no separate viscous
        pre-solve is needed.

        **Which family suits a given problem is not settled by theory** — it is set by
        the problem's overstress ratio :math:`f = \\eta_{ve}/\\eta_{pl}`, which is cheap
        to measure on the viscous seed before choosing. The sqrt family's smoothing
        saturates at a factor :math:`2f/(f+2)`, so on a problem where f is O(1) it can
        barely build an easier problem at all, however large δ is; the power mean does
        not saturate and has range there. Where f is large the position reverses.

        The side the regularised law sits on is set by ``anchor=``, not by the family.
        An entry problem should be no WEAKER than the sharp problem it precedes, so
        ``anchor="yield"`` is the safe direction for both families — see
        :attr:`yield_anchor`.

        Parameters
        ----------
        smoother : {"powermean", "sqrt"}, optional
            Soft-min family to march. ``None`` keeps the default (power mean).
        anchor : {"onset", "yield"}, optional
            Which point the smoothed law reproduces exactly. ``None`` leaves the
            model's current :attr:`yield_anchor` alone.

        Returns
        -------
        YieldHomotopyControl
            ``set_delta`` (model-owned setter for δ), ``tangent`` (the
            ``consistent_jacobian`` value to use), ``delta`` (the ``constants[]``
            atom itself, for diagnostics) and ``delta0`` (the family's entry δ).
        """
        from underworld3.systems.yield_continuation import YieldHomotopyControl

        if smoother is None:
            smoother = "powermean"
        if smoother not in self._YIELD_HOMOTOPY_DELTA0:
            raise ValueError(
                f"smoother must be one of {sorted(self._YIELD_HOMOTOPY_DELTA0)}, "
                f"got {smoother!r}"
            )

        self.yield_mode = "softmin"
        self.yield_smoother = smoother
        if anchor is not None:
            self.yield_anchor = anchor

        # No strain-rate floor is needed for the cold (v = 0) start the march begins
        # from: eta_pl = tau_y/(2 edot_II) is +inf there, which the soft-min carries
        # correctly to the viscous branch. See the harmonic-mean note in
        # _combine_yield for the one form that must be written carefully to keep it
        # so.

        def set_delta(value):
            # Go through the property, not the atom: `yield_softness` updates BOTH
            # the stored value and the constants[] atom, so a later
            # _get_yield_softness() cannot silently reset δ to a stale number.
            self.yield_softness = value

        return YieldHomotopyControl(
            set_delta=set_delta,
            tangent=self._yield_homotopy_tangent,
            delta=self._get_yield_softness(),
            delta0=self._YIELD_HOMOTOPY_DELTA0[smoother],
        )


## NOTE - retrofit VEP into here


class ViscoPlasticFlowModel(ViscousFlowModel):
    r"""
    Viscoplastic flow constitutive model with yield stress.

    Extends :class:`ViscousFlowModel` with a yield stress that limits the
    maximum deviatoric stress. When stress would exceed the yield stress,
    the effective viscosity is reduced to cap the stress.

    .. math::

        \tau_{ij} = \eta_\mathrm{eff} \cdot \dot{\varepsilon}_{ij}

    where the effective viscosity is:

    .. math::

        \eta_\mathrm{eff} = \min\left(\eta_0, \frac{\tau_y}{2\dot{\varepsilon}_{II}}\right)

    and :math:`\tau_y` is the yield stress and :math:`\dot{\varepsilon}_{II}`
    is the second invariant of the strain rate.

    Parameters
    ----------
    unknowns : Unknowns
        The solver unknowns (typically velocity and pressure fields).
    material_name : str, optional
        Name identifier for this material.

    Notes
    -----
    If yield stress is not defined, this model behaves identically to
    :class:`ViscousFlowModel`. The message ``not~yet~defined`` in the
    effective viscosity indicates missing parameters.

    See Also
    --------
    ViscousFlowModel : Base viscous model without yielding.
    ViscoElasticPlasticFlowModel : Adds viscoelastic memory.
    """

    def __init__(self, unknowns, material_name: str = None):
        # All this needs to do is define the
        # non-paramter properties that we want to
        # use in other expressions and init the parent(s)
        #

        super().__init__(unknowns, material_name=material_name)

        self._strainrate_inv_II = expression(
            r"\dot\varepsilon_{II}",
            sympy.sqrt((self.grad_u**2).trace() / 2),
            "Strain rate 2nd Invariant",
        )

        self._plastic_eff_viscosity = expression(
            R"{\eta_\textrm{eff,p}}",
            1,
            "Effective viscosity (plastic)",
        )

        # Yield-combination mode (see _combine_yield on the base class). Default
        # "min" = the exact hard Min(η_0, η_yield) this model has always used, so the
        # default behaviour is unchanged. Opt into "softmin" (+ yield_smoother /
        # yield_softness) for the δ-parameterised smooth-min homotopy.
        self._yield_mode = "min"
        self._yield_softness = 0.0        # δ; 0 ⇒ exact Min
        self._yield_smoother = "sqrt"     # smooth-min family: "sqrt" | "powermean"
        self._yield_softness_expr = None  # constants[] δ atom (created lazily)
        self._yield_offset_expr = None    # onset-offset atom (created lazily)

    class _Parameters(_ParameterBase, _ViscousParameterAlias):
        """Any material properties that are defined by a constitutive relationship are
        collected in the parameters which can then be defined/accessed by name in
        individual instances of the class.

        `sympy.oo` (infinity) for default values ensures that sympy.Min simplifies away
        the conditionals when they are not required.

        Uses Parameter descriptor pattern for automatic lazy evaluation preservation
        with unit-aware quantities.
        """

        # Import Parameter descriptor (must use absolute import inside nested class)
        import underworld3.utilities._api_tools as api_tools

        shear_viscosity_0 = api_tools.Parameter(
            R"{\eta}",
            lambda inner_self: 1,
            "Shear viscosity",
            units="Pa*s",
        )

        shear_viscosity_min = api_tools.Parameter(
            R"{\eta_{\textrm{min}}}",
            lambda inner_self: -sympy.oo,
            "Shear viscosity, minimum cutoff",
            units="Pa*s",
        )

        yield_stress = api_tools.Parameter(
            R"{\tau_{y}}",
            lambda inner_self: sympy.oo,
            "Yield stress (DP)",
            units="Pa",
        )

        yield_stress_min = api_tools.Parameter(
            R"{\tau_{y, \mathrm{min}}}",
            lambda inner_self: 0,
            "Yield stress (DP) minimum cutoff",
            units="Pa",
        )

        strainrate_inv_II_min = api_tools.Parameter(
            R"{\dot\varepsilon_{\mathrm{min}}}",
            lambda inner_self: 0,
            "Strain rate invariant minimum value",
            units="1/s",
        )

        def __init__(inner_self, _owning_model):
            inner_self._owning_model = _owning_model
            # Parameters are now descriptors - no manual initialization needed

    @property
    def viscosity(self):
        r"""Effective viscosity with plastic yielding.

        .. math::
            \eta_{\mathrm{eff}} = \min\left(\eta_0, \frac{\tau_y}{2\dot{\varepsilon}_{II}}\right)

        where :math:`\dot{\varepsilon}_{II}` is the second invariant of strain rate.
        """
        inner_self = self.Parameters
        # detect if values we need are defined or are placeholder symbols

        if inner_self.yield_stress.sym == sympy.oo:
            self._plastic_eff_viscosity.symbol = inner_self.shear_viscosity_0.symbol
            self._plastic_eff_viscosity._sym = inner_self.shear_viscosity_0._sym
            return self._plastic_eff_viscosity

        # Don't put conditional behaviour in the constitutive law
        # when it is not needed

        # Lower bound on the yield stress, defaulting to ZERO and therefore normally
        # active. tau_y is compared against the second invariant of the stress, so a
        # negative tau_y is meaningless — a pressure-dependent Drucker-Prager yield
        # C + sin(phi)*p goes negative in tension and must be cut off there rather
        # than propagated into tau_y/(2 edot_II). (The ±oo defaults elsewhere in
        # Parameters exist so sympy can cancel an unused term away; that trick is
        # wrong here, so this one defaults to 0 — maintainer ruling 2026-07-26.)
        # An explicit -oo still disables the floor.
        if inner_self.yield_stress_min.sym != -sympy.oo:
            yield_stress = self._apply_floor(
                inner_self.yield_stress, inner_self.yield_stress_min
            )
        else:
            yield_stress = inner_self.yield_stress

        # Rate regularisation. eta_pl = tau_y / (2 edot_II) is unbounded as edot -> 0;
        # adding a floor to the strain rate caps it at tau_y/(2 edot_min) and so bounds
        # the viscosity CONTRAST, which is what conditions the velocity block (the same
        # role the Perzyna/rate-strengthening xi plays in the Spiegelman studies). The
        # parameter was declared on this model but never applied — the elastic models
        # (ViscoElasticPlastic, TransverseIsotropicVEP) have always used it. Default 0
        # = off, so the unregularised law is unchanged.
        if inner_self.strainrate_inv_II_min.sym != 0:
            viscosity_yield = yield_stress / (
                2 * (self._strainrate_inv_II + inner_self.strainrate_inv_II_min)
            )
        else:
            viscosity_yield = yield_stress / (2 * self._strainrate_inv_II)

        # Combine the viscous and plastic (yield) viscosities. The default
        # yield_mode="min" gives the exact hard Min(η_0, η_yield); yield_mode="softmin"
        # opts into the δ-parameterised smooth-min (sqrt or powermean family) for a
        # scalable homotopy toward the sharp yield surface.
        effective_viscosity = self._combine_yield(
            inner_self.shear_viscosity_0, viscosity_yield
        )

        # If we want to apply limits to the viscosity but see caveat above
        # Keep this as an sub-expression for clarity

        if inner_self.shear_viscosity_min.sym != -sympy.oo:
            self._plastic_eff_viscosity._sym = self._apply_floor(
                effective_viscosity, inner_self.shear_viscosity_min,
                rounding=self.viscosity_min_rounding,
            )

        else:
            self._plastic_eff_viscosity._sym = effective_viscosity

        # Returns an expression that has a different description
        return self._plastic_eff_viscosity

    @property
    def supports_yield_homotopy(self):
        """This model carries a δ-parameterised yield law — ``solve(homotopy=True)``
        can march it. See :meth:`_yield_homotopy_control`."""
        return True

    @property
    def yield_mode(self):
        r"""How the viscous and plastic (yield) viscosities are combined.

        - ``"min"`` (default): exact hard ``Min(η_0, η_yield)`` — the sharp yield
          surface this model has always used.
        - ``"harmonic"``: ``1/(1/η_0 + 1/η_yield)`` — a smooth blend.
        - ``"softmin"``: the δ-parameterised smooth-min (family set by
          ``yield_smoother``), for a scalable homotopy toward the sharp surface.
        """
        return self._yield_mode

    @yield_mode.setter
    def yield_mode(self, value):
        if value not in ("min", "harmonic", "softmin"):
            raise ValueError(
                f"yield_mode must be 'min', 'harmonic', or 'softmin', got '{value}'"
            )
        self._yield_mode = value
        self._reset()

    @property
    def yield_softness(self):
        r"""Soft-min regularisation δ for ``yield_mode="softmin"`` (0 ⇒ exact Min).

        δ is held as a ``constants[]`` atom, so ramping it at runtime (this setter,
        or ``cm._get_yield_softness().sym = ...`` + ``solver._update_constants()``)
        does not trigger a JIT recompile — the basis of a scalable yield homotopy.
        """
        return self._yield_softness

    @yield_softness.setter
    def yield_softness(self, value):
        self._yield_softness = float(value)
        if getattr(self, "_yield_softness_expr", None) is not None:
            self._yield_softness_expr.sym = sympy.Float(self._yield_softness)
        self._reset()

    def plastic_correction(self) -> float:
        r"""Scaling factor to reduce stress to yield surface.

        .. math::
            f = \frac{\tau_y}{\tau_{II}}

        where :math:`\tau_{II}` is the second invariant of deviatoric stress.
        Returns 1 if no yield stress is set.
        """
        parameters = self.Parameters

        if parameters.yield_stress == sympy.oo:
            return sympy.sympify(1)

        stress = self.stress_projection()

        # The yield criterion in this case is assumed to be a bound on the second invariant of the stress

        stress_II = sympy.sqrt((stress**2).trace() / 2)

        correction = parameters.yield_stress / stress_II

        return correction

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        ## feedback on this instance
        display(
            Latex(
                r"$\quad\eta_\textrm{0} = $"
                + sympy.sympify(self.Parameters.shear_viscosity_0.sym)._repr_latex_()
            ),
            Latex(
                r"$\quad\tau_\textrm{y} = $"
                + sympy.sympify(self.Parameters.yield_stress.sym)._repr_latex_(),
            ),
        )

        return


class ViscoElasticPlasticFlowModel(ViscousFlowModel):
    r"""
    Viscoelastic-plastic flow constitutive model.

    The stress (flux term) is given by:

    .. math::

        \tau_{ij} = \eta_{ijkl} \cdot \frac{1}{2} \left[ \frac{\partial u_k}{\partial x_l}
        + \frac{\partial u_l}{\partial x_k} \right]

    where :math:`\eta` is the viscosity, a scalar constant, SymPy function,
    Underworld mesh variable, or any valid combination. This results in an
    isotropic (but not necessarily homogeneous or linear) relationship between
    :math:`\tau` and the velocity gradients. You can also supply :math:`\eta_{IJ}`,
    the Mandel form of the constitutive tensor, or :math:`\eta_{ijkl}`, the rank-4 tensor.

    The Mandel constitutive matrix is available in `viscous_model.C` and the rank 4 tensor form is
    in `viscous_model.c`.  Apply the constitutive model using:

    """

    def __init__(self, unknowns, order=1, integrator: str = "bdf",
                 material_name: str = None):
        """Construct a viscoelastic-plastic flow model.

        Parameters
        ----------
        unknowns : Unknowns
            The solver unknowns (velocity, pressure).
        order : int, default 1
            Time-integration order. Combines with ``integrator``:

            - ``integrator='bdf', order=1``: BDF-1 (backward Euler).
            - ``integrator='bdf', order=2``: BDF-2.
            - ``integrator='etd', order=1``: ETD-1 (single-step,
              fully L-stable, recommended default for VEP+yield).
            - ``integrator='etd', order=2``: ETD-2 (single-step
              with linear-quadrature forcing history; accurate on
              smooth VE but **unstable in tight-yield VEP** —
              produces global runaway, see EXPONENTIAL_VE_INTEGRATOR.md
              lessons #7, #9).
        integrator : str, default "bdf"
            Time-integration scheme:

            - ``"bdf"``: backward differentiation formula on the
              deviatoric-stress rate equation. Production default.
            - ``"etd"``: exponential time-differencing — integrates the
              Maxwell relaxation operator analytically (``α = exp(-Δt/τ)``).
              ``order=1`` is the recommended default for new code: same
              stability as BDF-1, exact handling of the relaxation factor
              at large ``Δt/τ``. ``order=2`` adds linear quadrature on
              the forcing history for higher accuracy on smooth VE
              (4.3× more accurate than BDF-2 on ``bench_ve_harmonic``)
              but blows up under active yield in tight-yield TI faults.
              See ``docs/developer/design/EXPONENTIAL_VE_INTEGRATOR.md``.
        material_name : str, optional
            Name identifier for this material.
        """
        if integrator not in ("bdf", "etd"):
            raise ValueError(
                f"integrator must be 'bdf' or 'etd', got '{integrator!r}'"
            )
        if integrator == "etd" and order not in (1, 2):
            raise ValueError(
                f"integrator='etd' supports order=1 (ETD-1, default-recommended) "
                f"or order=2 (ETD-2, accurate for smooth VE; avoid in tight-yield "
                f"VEP where it produces global runaway). Got order={order}."
            )
        self._integrator = integrator

        # Store material_name before creating expressions (needed by create_unique_symbol)
        self._material_name = material_name

        # This may not be defined at initialisation time, set to None until used
        self._stress_star = expression(
            r"{\tau^{*}}",
            None,
            r"Lagrangian Stress at $t - \delta_t$",
        )

        # This may not be defined at initialisation time, set to None until used
        self._stress_2star = expression(
            r"{\tau^{**}}",
            None,
            r"Lagrangian Stress at $t - 2\delta_t$",
        )

        # This may not be well-defined at initialisation time, set to None until used
        self._E_eff = expression(
            r"{\dot{\varepsilon}_{\textrm{eff}}}",
            None,
            "Equivalent value of strain rate (accounting for stress history)",
        )

        # This may not be well-defined at initialisation time, set to None until used
        self._E_eff_inv_II = expression(
            r"{\dot{\varepsilon}_{II,\textrm{eff}}}",
            None,
            "Equivalent value of strain rate 2nd invariant (accounting for stress history)",
        )

        self._order = order
        self._yield_mode = "softmin"  # "min", "harmonic", "smooth", or "softmin"
        self._yield_softness = 0.1  # δ parameter for "softmin" mode
        self._yield_smoother = "sqrt"     # smooth-min family: "sqrt" | "powermean"
        self._yield_softness_expr = None  # constants[] δ atom (created lazily)
        self._yield_offset_expr = None    # onset-offset atom (created lazily)

        # Timestep — set by the solver before each solve(). Not a user parameter.
        # Initialised to oo (viscous limit). The solver overwrites this with the
        # actual timestep on every call to solve(timestep=dt).
        self._dt = expression(r"{\Delta t}", sympy.oo, "Timestep (set by solver)")

        # BDF coefficients as UWexpressions — route through PetscDS constants[].
        # Updated each step by _update_bdf_coefficients() before solve.
        # Initialised to BDF-1 values: [1, -1, 0, 0].
        self._bdf_c0 = expression(r"{c_0^{\mathrm{BDF}}}", sympy.Integer(1), "BDF leading coefficient")
        self._bdf_c1 = expression(r"{c_1^{\mathrm{BDF}}}", sympy.Integer(-1), "BDF history coefficient 1")
        self._bdf_c2 = expression(r"{c_2^{\mathrm{BDF}}}", sympy.Integer(0), "BDF history coefficient 2")
        self._bdf_c3 = expression(r"{c_3^{\mathrm{BDF}}}", sympy.Integer(0), "BDF history coefficient 3")

        self._reset()

        super().__init__(unknowns, material_name=material_name)

        return

    class _Parameters(_ParameterBase, _ViscousParameterAlias):
        """Any material properties that are defined by a constitutive relationship are
        collected in the parameters which can then be defined/accessed by name in
        individual instances of the class.

        Uses Parameter descriptor pattern for automatic lazy evaluation preservation
        with unit-aware quantities.
        """

        # Import Parameter descriptor (must use absolute import inside nested class)
        import underworld3.utilities._api_tools as api_tools

        # Basic parameters with Parameter descriptors
        shear_viscosity_0 = api_tools.Parameter(
            R"{\eta}",
            lambda inner_self: 1,
            "Shear viscosity",
            units="Pa*s",
        )

        shear_modulus = api_tools.Parameter(
            R"{\mu}",
            lambda inner_self: sympy.oo,
            "Shear modulus",
            units="Pa",
        )

        @property
        def dt_elastic(inner_self):
            """Timestep for VE formulas. Set by the solver, not a user parameter.

            Returns the UWexpression that the solver updates before each solve.
            This flows through PetscDS constants[] so the JIT-compiled pointwise
            functions always see the current timestep.
            """
            return inner_self._owning_model._dt

        @dt_elastic.setter
        def dt_elastic(inner_self, value):
            """Allow the solver to set dt via Parameters.dt_elastic = timestep."""
            if hasattr(value, 'sym'):
                inner_self._owning_model._dt.sym = value.sym
            else:
                inner_self._owning_model._dt.sym = value

        shear_viscosity_min = api_tools.Parameter(
            R"{\eta_{\textrm{min}}}",
            lambda inner_self: -sympy.oo,
            "Shear viscosity, minimum cutoff",
            units="Pa*s",
        )

        yield_stress = api_tools.Parameter(
            R"{\tau_{y}}",
            lambda inner_self: sympy.oo,
            "Yield stress (DP)",
            units="Pa",
        )

        yield_stress_min = api_tools.Parameter(
            R"{\tau_{y, \mathrm{min}}}",
            lambda inner_self: 0,
            "Yield stress (DP) minimum cutoff",
            units="Pa",
        )

        strainrate_inv_II_min = api_tools.Parameter(
            R"{\dot\varepsilon_{II,\mathrm{min}}}",
            lambda inner_self: 0,
            "Strain rate invariant minimum value",
            units="1/s",
        )

        def __init__(
            inner_self,
            _owning_model,
        ):
            inner_self._owning_model = _owning_model

            # Internal symbols for stress history (not parameters, internal state)
            strainrate_inv_II = sympy.symbols(
                r"\left|\dot\epsilon\right|\rightarrow\textrm{not\ defined}"
            )
            stress_star = sympy.symbols(r"\sigma^*\rightarrow\textrm{not\ defined}")
            inner_self._stress_star = stress_star
            inner_self._not_yielded = sympy.sympify(1)

            ## The following expressions are containers for derived/computed values.
            ## They have @property calls to retrieve / calculate them.
            ## We keep them as expression containers for lazy evaluation.

            inner_self._ve_effective_viscosity = expression(
                R"{\eta_{\mathrm{eff}}}",
                None,
                "Effective viscosity (elastic)",
            )

            inner_self._t_relax = expression(
                R"{t_{\mathrm{relax}}}",
                None,
                "Maxwell relaxation time",
            )

        ## Derived parameters of the constitutive model (these have no setters)
        ## Note, do not return new expressions, keep the old objects as containers
        ## the correct values are used in existing expressions. These really are
        ## parameters - they are solely combinations of other parameters.

        @property
        def ve_effective_viscosity(inner_self):
            r"""Visco-elastic effective viscosity: :math:`\eta_{\mathrm{eff}} = \frac{\eta G \Delta t}{\eta + G \Delta t}`."""
            # the dt_elastic defaults to infinity, t_relax to zero,
            # so this should be well behaved in the viscous limit

            if inner_self.shear_modulus == sympy.oo:
                return inner_self.shear_viscosity_0

            # BDF-k effective viscosity: eta_eff = eta*mu*dt / (c0*eta + mu*dt)
            # c0 is a UWexpression routed through PetscDS constants[],
            # updated each step by _update_bdf_coefficients().
            eta = inner_self.shear_viscosity_0
            mu = inner_self.shear_modulus
            dt_e = inner_self.dt_elastic
            c0 = inner_self._owning_model._bdf_c0

            el_eff_visc = eta * mu * dt_e / (c0 * eta + mu * dt_e)

            inner_self._ve_effective_viscosity.sym = el_eff_visc

            return inner_self._ve_effective_viscosity

        @property
        def t_relax(inner_self):
            r"""Maxwell relaxation time: :math:`t_{\mathrm{relax}} = \eta / G`."""
            # shear modulus defaults to infinity so t_relax goes to zero
            # in the viscous limit

            inner_self._t_relax.sym = inner_self.shear_viscosity_0 / inner_self.shear_modulus
            return inner_self._t_relax

    ## End of parameters definition

    @property
    def order(self):
        """Time integration order (1 or 2)."""
        return self._order

    @order.setter
    def order(self, value):
        """Set the time integration order.

        If the model is already attached to a solver with a DFDt, this will
        warn if the DFDt was created with a lower order (since it can't be
        changed after creation — the DFDt allocates history buffers at init).
        """
        self._order = value
        self._reset()

        # Propagate to connected solver if present
        solver = getattr(self.Parameters, '_solver', None)
        if solver is not None:
            ddt = getattr(solver.Unknowns, 'DFDt', None)
            if ddt is not None and ddt.order < value:
                import warnings
                warnings.warn(
                    f"Setting order={value} but the solver's DFDt was already "
                    f"created with order={ddt.order}. The DFDt order cannot be "
                    f"changed after creation. To use order={value}, create the "
                    f"model with the desired order before assigning to the solver:\n"
                    f"  cm = ViscoElasticPlasticFlowModel(stokes.Unknowns, order={value})\n"
                    f"  stokes.constitutive_model = cm",
                    UserWarning,
                    stacklevel=2,
                )
            elif ddt is not None:
                solver._order = value
        return

    @property
    def effective_order(self):
        """Effective order accounting for DDt history startup.

        During the first few timesteps, the DDt may not have enough history
        to support the requested order. This property returns the lower of
        the requested order and the DDt's effective order (which ramps from
        1 to self.order as history accumulates).
        """
        if self.Unknowns is not None and self.Unknowns.DFDt is not None:
            return min(self._order, self.Unknowns.DFDt.effective_order)
        return self._order

    # Maximum timestep ratio (dt_new / dt_old) for which BDF-2+ is safe.
    # Beyond this, fall back to BDF-1 to avoid negative-stress extrapolation
    # when stress history is non-smooth (e.g. yield events).
    _max_dt_ratio_for_higher_order = 2.0

    def _update_bdf_coefficients(self):
        """Update BDF coefficient UWexpressions from current dt_elastic and DDt history.

        Call this before each solve so that the constants[] array carries the
        correct coefficients to the compiled pointwise functions. The coefficient
        UWexpressions (_bdf_c0..c3) are referenced symbolically in ve_effective_viscosity,
        E_eff, and stress() — their numeric values flow through PetscDSSetConstants.

        When the timestep ratio exceeds ``_max_dt_ratio_for_higher_order``,
        BDF-2+ coefficients can cause negative stress extrapolation if the
        stress history is non-smooth (e.g. after a yield event). In this case
        we fall back to BDF-1 coefficients for safety.
        """
        order = self.effective_order

        if self.Unknowns is not None and self.Unknowns.DFDt is not None:
            dt_current = self.Parameters.dt_elastic
            if hasattr(dt_current, 'sym'):
                dt_current = dt_current.sym

            # Guard: fall back to BDF-1 when timestep increases too rapidly
            dt_history = self.Unknowns.DFDt._dt_history
            if order >= 2 and len(dt_history) > 0 and dt_history[0] is not None:
                try:
                    ratio = float(dt_current) / float(dt_history[0])
                    if ratio > self._max_dt_ratio_for_higher_order:
                        order = 1
                except (TypeError, ZeroDivisionError):
                    pass  # symbolic dt — can't evaluate, keep requested order

            coeffs = _bdf_coefficients(order, dt_current, dt_history)
        else:
            coeffs = _bdf_coefficients(order, None, [])

        # Pad to length 4
        while len(coeffs) < 4:
            coeffs.append(sympy.Integer(0))

        self._bdf_c0.sym = coeffs[0]
        self._bdf_c1.sym = coeffs[1]
        self._bdf_c2.sym = coeffs[2]
        self._bdf_c3.sym = coeffs[3]

    def _update_history_coefficients(self):
        """Pre-solve hook: refresh integrator coefficients.

        Dispatches on ``(self._integrator, self._order)``:
        - ``"bdf"`` (order 1 or 2): updates BDF c-coefficients via
          :py:meth:`_update_bdf_coefficients`.
        - ``"etd"`` order=2 (Phase B ETD-2): updates α, φ on the DDt
          from ``τ_VE = η/μ``; forcing-history slot active.
        - ``"etd"`` order=1 (ETD-1): updates α, φ as for ETD-2 then
          forces ``φ = α`` so the ``(φ-α)·ε̇*`` term zeros out — fully
          L-stable single-step, no forcing-history slot needed.
        """
        if self._integrator == "etd":
            if self.Unknowns.DFDt is None:
                return
            params = self.Parameters
            if params.shear_modulus.sym is sympy.oo:
                tau_eff = sympy.oo
            else:
                try:
                    eta_val = float(params.shear_viscosity_0.sym)
                    mu_val = float(params.shear_modulus.sym)
                    tau_eff = eta_val / mu_val if mu_val > 0 else sympy.oo
                except (TypeError, ValueError):
                    tau_eff = None
            try:
                dt_val = (
                    float(params.dt_elastic.sym)
                    if params.dt_elastic.sym is not sympy.oo
                    else None
                )
            except (TypeError, ValueError):
                dt_val = None
            self.Unknowns.DFDt.update_exp_coefficients(dt_val, tau_eff)
            if self._order == 1:
                # ETD-1 reduction: φ = α makes the (φ-α)·ε̇* term zero
                # AND turns (1-φ)·ε̇ into (1-α)·ε̇.
                self.Unknowns.DFDt._exp_phi.sym = self.Unknowns.DFDt._exp_alpha.sym
        else:
            self._update_bdf_coefficients()

    def _update_history_post_solve(self):
        """Post-solve hook.

        - BDF / ETD-1: no-op (no forcing-history slot).
        - ETD-2: refresh ``forcing_star`` from the just-solved ε̇ for
          the next step's history term.
        """
        if self._integrator == "etd" and self._order == 2 and self.Unknowns.DFDt is not None:
            if self.Unknowns.DFDt.forcing_star is not None:
                self.Unknowns.DFDt.update_forcing_history(forcing_fn=self.Unknowns.E)

    @property
    def stress_history_ddt_kwargs(self):
        """SemiLagrangian DDt kwargs based on integrator selection.

        ETD-2 (order=2) needs the forcing-history slot; BDF and ETD-1
        (order=1) do not.
        """
        if self._integrator == "etd" and self._order == 2:
            return {"with_forcing_history": True}
        return {}

    # The following should have no setters
    @property
    def stress_star(self):
        r"""Previous timestep stress :math:`\boldsymbol{\sigma}^*` from history."""
        if self.Unknowns.DFDt is not None:
            self._stress_star.sym = self.Unknowns.DFDt.psi_star[0].sym

        return self._stress_star

    @property
    def stress_2star(self):
        r"""Second-order stress history :math:`\boldsymbol{\sigma}^{**}` (for 2nd order integration)."""
        # Check if we have enough information in DFDt to update _stress_star,
        # otherwise it will be defined as zero

        if self.Unknowns.DFDt is not None:
            if self.Unknowns.DFDt.order >= 2:
                self._stress_2star.sym = self.Unknowns.DFDt.psi_star[1].sym
            else:
                self._stress_2star.sym = sympy.sympify(0)

        return self._stress_2star

    @property
    def E_eff(self):
        r"""Effective strain rate including elastic-history coupling.

        For BDF integration:

        .. math::
            \dot{\varepsilon}_\mathrm{eff} = \dot{\varepsilon}
                - \sum_i c_i \frac{\sigma^{*(i)}}{2 \mu \Delta t}

        For ETD-2 (exponential) integration:

        .. math::
            \dot{\varepsilon}_\mathrm{eff} = (1-\varphi)\,\dot{\varepsilon}
                + \frac{\alpha}{2\eta}\,\sigma^*
                + (\varphi-\alpha)\,\dot{\varepsilon}^*

        Both forms reduce to bare ``ε̇`` when no elastic history is
        active. The yield criterion ``η_pl = τ_y/(2|E_eff|_II)`` is the
        same expression structure for both — it adapts naturally to the
        integrator's ``E_eff``.
        """
        E = self.Unknowns.E

        if self.Unknowns.DFDt is None or not self.is_elastic:
            self._E_eff.sym = E
            return self._E_eff

        DDt = self.Unknowns.DFDt

        if self._integrator == "etd":
            # ETD-2 effective strain rate carrying α·σ*/(2η) and (φ-α)·ε̇*.
            # ETD-1 (order=1): φ = α (set in _update_history_coefficients),
            # so the (φ-α)·ε̇* term zeros out and (1-φ)·ε̇ → (1-α)·ε̇ — same
            # expression tree, no separate code path needed.
            alpha = DDt._exp_alpha
            phi = DDt._exp_phi
            sigma_star = DDt.psi_star[0].sym
            if DDt.forcing_star is not None:
                edot_star = DDt.forcing_star.sym
            else:
                edot_star = sympy.zeros(*E.shape)
            eta_raw = self.Parameters.shear_viscosity_0
            self._E_eff.sym = (
                (1 - phi) * E
                + (alpha / (2 * eta_raw)) * sigma_star
                + (phi - alpha) * edot_star
            )
            return self._E_eff

        # BDF default
        mu_dt = self.Parameters.dt_elastic * self.Parameters.shear_modulus
        bdf_cs = [self._bdf_c1, self._bdf_c2, self._bdf_c3]
        for i in range(DDt.order):
            E += -bdf_cs[i] * DDt.psi_star[i].sym / (2 * mu_dt)
        self._E_eff.sym = E
        return self._E_eff

    @property
    def E_eff_inv_II(self):
        r"""Second invariant of effective strain rate: :math:`\dot{\varepsilon}_{II} = \sqrt{\frac{1}{2}\dot{\varepsilon}_{ij}\dot{\varepsilon}_{ij}}`."""
        E_eff = self.E_eff.sym
        self._E_eff_inv_II.sym = sympy.sqrt((E_eff**2).trace() / 2)

        return self._E_eff_inv_II

    @property
    def K(self):
        """Effective stiffness parameter (viscosity for visco-elastic-plastic flow)."""
        return self.viscosity

    @property
    def _unclipped_ve_viscosity(self):
        """Unclipped viscoelastic viscosity (no yield wrap), depends on integrator.

        - BDF: ``ve_effective_viscosity = η·μΔt/(c₀·η + μΔt)`` —
          baked-in time-integration factor for backward differentiation.
        - ETD-2: ``η`` (raw) — the time-integration factor ``(1-φ)`` is
          carried symbolically in :py:attr:`E_eff`, not folded into
          this viscosity.
        """
        if self._integrator == "etd":
            return self.Parameters.shear_viscosity_0
        return self.Parameters.ve_effective_viscosity

    @property
    def viscosity(self):
        r"""Effective viscosity combining visco-elastic and plastic limits.

        The yield mode controls how η_ve and η_pl are combined:

        - ``"smooth"`` (default): corrected harmonic ``η_ve·(1+f)/(1+f+f²)``
          where ``f = η_ve/η_pl``. Converges to η_pl at deep yielding,
          no Min/Max discontinuities.
        - ``"harmonic"``: ``1/(1/η_ve + 1/η_pl)``. Smooth but undershoots τ_y
          when η_ve is small relative to η_pl.
        - ``"min"``: sharp ``Min(η_ve, η_pl)``. Exact yield stress but can
          cause SNES divergence with higher-order BDF time integration.

        The unclipped η_ve depends on ``self._integrator`` —
        :py:attr:`_unclipped_ve_viscosity` returns the correct base for
        BDF or ETD-2 (raw η for ETD-2 since the time factor lives in
        ``E_eff``; ``ve_effective_viscosity`` for BDF).
        """

        inner_self = self.Parameters

        if inner_self.yield_stress.sym == sympy.oo:
            return self._unclipped_ve_viscosity

        effective_viscosity = self._unclipped_ve_viscosity

        if self.is_viscoplastic:
            vp_effective_viscosity = self._plastic_effective_viscosity
            # Combine η_ve with the plastic viscosity per yield_mode (harmonic / exact
            # Min / δ-soft-min). The soft-min softness δ now lives in a constants[] atom
            # (see _combine_yield / _get_yield_softness) — value-identical to the former
            # inline float law at the same δ, but runtime-rampable with no recompile.
            effective_viscosity = self._combine_yield(
                effective_viscosity, vp_effective_viscosity
            )

        # Apply viscosity floor — but skip for smooth-blend yield modes
        # where the outer Max creates a nested Min/Max that breaks the
        # BDF-2 Jacobian. Those modes are already smooth and bounded.

        if inner_self.shear_viscosity_min.sym != -sympy.oo:
            rounding = self.viscosity_min_rounding
            if rounding is not None:
                # An explicit rounding scale makes the cutoff differentiable, which is
                # what the skip below existed to avoid needing: there is no outer Max to
                # nest, so the floor can be honoured in the smooth yield modes too.
                return self._apply_floor(
                    effective_viscosity, inner_self.shear_viscosity_min,
                    rounding=rounding,
                )
            if self.is_viscoplastic and self._yield_mode in ("harmonic", "softmin"):
                return effective_viscosity
            else:
                return sympy.Max(
                    effective_viscosity,
                    inner_self.shear_viscosity_min,
                )

        else:
            return effective_viscosity

    # NOTE: a hard-Min smooth-tangent override (flux_jacobian = harmonic) was
    # prototyped here but deferred to the yield-law / δ-homotopy follow-up. The
    # smooth-Jacobian-with-Min-residual tangent is inconsistent (it is the
    # consistent tangent of the *harmonic* problem) and converges WORSE than
    # Picard on hard-yield VEP; the robust route is problem-space homotopy
    # (ramp the softmin softness δ→0), not a smooth tangent. See the design doc
    # docs/developer/design/jacobian-unwrap-constants-bug.md. The generic
    # Constitutive_Model.flux_jacobian hook (default None) remains available.

    @property
    def _plastic_effective_viscosity(self):
        parameters = self.Parameters

        if parameters.yield_stress == sympy.oo:
            return sympy.oo

        # Use the effective strain rate (including elastic history) for the
        # yield criterion. This must use the same order-dependent BDF
        # coefficients as the stress formula.
        Edot = self.E_eff.sym

        strainrate_inv_II = expression(
            R"{\dot\varepsilon_{II}'}",
            sympy.sqrt((Edot**2).trace() / 2),
            "Strain rate 2nd Invariant including elastic strain rate term",
        )

        # Guard on the DISABLING sentinel, not on zero: zero is the default and a
        # physically meaningful floor (the yield stress is compared against the second
        # invariant of the stress, so a negative tau_y is meaningless). Only an
        # explicit -oo turns the floor off.
        if parameters.yield_stress_min.sym != -sympy.oo:
            # Literal 0 for the default floor, not the parameter atom: sympy cannot
            # fuzzy-compare an opaque UWexpression and Max canonicalisation recurses.
            # smooth_max keeps both operands symbolic (the JIT routes the floor
            # through constants[]) and needs no ordering test, which sympy cannot
            # resolve against an opaque UWexpression. eps = 0 makes it exactly Max.
            yield_stress = uw.maths.smooth_max(
                parameters.yield_stress, parameters.yield_stress_min, 0
            )
        else:
            yield_stress = parameters.yield_stress

        if parameters.strainrate_inv_II_min.sym != 0:
            viscosity_yield = yield_stress / (
                2 * (strainrate_inv_II + parameters.strainrate_inv_II_min)
            )
        else:
            viscosity_yield = yield_stress / (2 * strainrate_inv_II)

        return viscosity_yield

    def plastic_correction(self):
        r"""Scaling factor to reduce stress to yield surface: :math:`f = \tau_y / \tau_{II}`."""
        parameters = self.Parameters

        if parameters.yield_stress == sympy.oo:
            return sympy.sympify(1)

        stress = self.stress_projection()

        # The yield criterion in this case is assumed to be a bound on the second invariant of the stress
        stress_inv_II = sympy.sqrt((stress**2).trace() / 2)
        correction = parameters.yield_stress / stress_inv_II

        return correction
        # return sympy.Min(1, correction)

    ## Is this really different from the original ?

    def _build_c_tensor(self):
        """For this constitutive law, we expect just a viscosity function"""

        if self._is_setup:
            print("Using cached value of c matrix", flush=True)
            return

        print("Building c matrix", flush=True)

        d = self.dim
        # inner_self = self.Parameters
        viscosity = self.viscosity

        try:
            # CRITICAL: Use .sym property to avoid UWexpression array corruption issues
            # See ViscousFlowModel._build_c_tensor() for detailed explanation
            viscosity_sym = viscosity.sym if hasattr(viscosity, "sym") else viscosity
            self._c = 2 * uw.maths.tensor.rank4_identity(d) * viscosity_sym
        except:
            d = self.dim
            dv = uw.maths.tensor.idxmap[d][0]
            if isinstance(viscosity, sympy.Matrix) and viscosity.shape == (dv, dv):
                self._c = 2 * uw.maths.tensor.mandel_to_rank4(viscosity, d)
            elif isinstance(viscosity, sympy.Array) and viscosity.shape == (d, d, d, d):
                self._c = 2 * viscosity
            else:
                raise RuntimeError(
                    "Viscosity is not a known type (scalar, Mandel matrix, or rank 4 tensor"
                )

        self._is_setup = True
        self._solver_is_setup = False

        return

    # Modify flux to use the stress history term
    # This may be preferable to using strain rate which can be discontinuous
    # and harder to map back and forth between grid and particles without numerical smoothing

    @property
    def flux(self):
        r"""Computes the effect of the constitutive tensor on the gradients of the unknowns.
        (always uses the `c` form of the tensor). In general cases, the history of the gradients
        may be required to evaluate the flux. For viscoelasticity, the
        """

        stress = self.stress()

        # if self.is_viscoplastic:
        #     plastic_scale_factor = sympy.Max(1, self.plastic_overshoot())
        #     stress /= plastic_scale_factor

        return stress

    def stress_projection(self):
        """viscoelastic stress projection (no plastic response)"""

        edot = self.grad_u

        # This is a scalar viscosity ...

        stress = 2 * self.Parameters.ve_effective_viscosity * edot

        if self.Unknowns.DFDt is not None:
            stress_star = self.Unknowns.DFDt.psi_star[0]

            if self.is_elastic:
                # 1st order
                stress += (
                    self.Parameters.ve_effective_viscosity
                    * stress_star.sym
                    / (self.Parameters.dt_elastic * self.Parameters.shear_modulus)
                )

        return stress

    def stress(self):
        """Viscoelastic(-plastic) deviatoric stress for the weak form.

        Both BDF and ETD-2 are written as ``σ = 2·viscosity·E_eff``.
        :py:attr:`E_eff` carries the integrator-specific elastic-history
        coupling, and :py:attr:`viscosity` returns the appropriate yield-
        wrapped effective viscosity (``ve_effective_viscosity`` for BDF,
        raw ``η`` for ETD-2 since the time factor is in ``E_eff``).
        """
        if not self.is_elastic or self.Unknowns.DFDt is None:
            return 2 * self.viscosity * self.grad_u

        # ETD-1 (order=1) uses the same E_eff machinery but with φ=α, so
        # forcing_star is not required (the (φ-α)·ε̇* term zeros out).
        # Only ETD-2 (order=2) needs forcing_star.
        if (
            self._integrator == "etd"
            and self._order == 2
            and self.Unknowns.DFDt.forcing_star is None
        ):
            raise RuntimeError(
                "integrator='etd' requires a SemiLagrangian DDt with "
                "with_forcing_history=True. The auto-DDt creation path "
                "reads stress_history_ddt_kwargs — re-create the solver/"
                "model so the kwargs propagate."
            )

        return 2 * self.viscosity * self.E_eff.sym

    # def eff_edot(self):

    #     edot = self.grad_u

    #     if self.Unknowns.DFDt is not None:
    #         stress_star = self.Unknowns.DFDt.psi_star[0]

    #         if self.is_elastic:
    #             edot += stress_star.sym / (
    #                 2 * self.Parameters.dt_elastic * self.Parameters.shear_modulus
    #             )

    #     return edot

    # def eff_edot_inv_II(self):

    #     edot = self.eff_edot()
    #     edot_inv_II = sympy.sqrt((edot**2).trace() / 2)

    #     return edot_inv_II

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        # super()._object_viewer()

        display(Markdown(r"### Viscous deformation"))
        display(
            Latex(
                r"$\quad\eta_\textrm{0} = $ "
                + sympy.sympify(self.Parameters.shear_viscosity_0.sym)._repr_latex_()
            ),
        )

        display(Markdown(r"#### Elastic deformation"))
        display(
            Latex(
                r"$\quad\mu = $ " + sympy.sympify(self.Parameters.shear_modulus.sym)._repr_latex_(),
            ),
            Latex(
                r"$\quad\Delta t_e = $ "
                + sympy.sympify(self.Parameters.dt_elastic.sym)._repr_latex_(),
            ),
        )

        display(Markdown(r"#### Plastic deformation"))
        display(
            Latex(
                r"$\quad\tau_\textrm{y} = $ "
                + sympy.sympify(self.Parameters.yield_stress.sym)._repr_latex_(),
            )
            ## Todo: add all the other properties in here
        )

    @property
    def yield_mode(self):
        r"""How to combine VE and plastic viscosities.

        ``"softmin"`` (default): smooth approximation to Min —
            ``η_ve / g(f)`` where ``g(f) ≈ max(1, f)`` with smoothing
            parameter δ (``yield_softness``, default 0.1).  Approaches
            exact Min as δ → 0; smooth derivatives at the kink.
            Recommended default: gets within ~2 % of the true yield
            surface while avoiding the SNES kink penalties of ``"min"``.
        ``"harmonic"``: parallel blending — ``1/(1/η_ve + 1/η_pl)``.
            Smooth but undershoots τ_y for soft materials.
        ``"min"``: sharp cutoff — ``Min(η_ve, η_pl)``.
            Exact yield but can cause SNES divergence with BDF-2 and
            BDF-2 phase-lag at BC discontinuities (see benchmarks).

        Note: the previous ``"smooth"`` mode (corrected harmonic
        ``η_ve·(1+f)/(1+f+f²)``) was retired — it under-clipped the
        yield surface by ~50 % under realistic forcing, with no
        compensating benefit over ``softmin``.  Recover from git
        history if needed (commit message keyword: ``smooth_yield``).
        """
        return self._yield_mode

    @yield_mode.setter
    def yield_mode(self, value):
        if value == "smooth":
            raise ValueError(
                "yield_mode='smooth' has been retired — it under-clipped "
                "the yield surface by ~50%. Use 'softmin' instead "
                "(default; close to exact Min with smooth derivatives)."
            )
        if value not in ("min", "harmonic", "softmin"):
            raise ValueError(
                f"yield_mode must be 'min', 'harmonic', or 'softmin', got '{value}'"
            )
        self._yield_mode = value
        self._reset()

    @property
    def yield_softness(self):
        r"""Regularisation parameter δ for ``"softmin"`` yield mode.

        Controls how closely the soft minimum approximates the sharp Min.
        Smaller values → sharper yield (closer to Min, less robust).
        Larger values → smoother transition (more robust, lower stress).

        Default 0.1. Only used when ``yield_mode == "softmin"``.
        Increase toward 0.5 if SNES convergence is difficult at yield onset.
        """
        return self._yield_softness

    @yield_softness.setter
    def yield_softness(self, value):
        self._yield_softness = float(value)
        # Keep the constants[] δ atom in sync (created lazily on first use) so a
        # numeric δ assignment is reflected without a JIT recompile.
        if getattr(self, "_yield_softness_expr", None) is not None:
            self._yield_softness_expr.sym = sympy.Float(self._yield_softness)
        self._reset()

    @property
    def requires_stress_history(self):
        """VEP models always require stress history tracking."""
        return True

    @property
    def supports_yield_homotopy(self):
        """This model carries a δ-parameterised yield law — ``solve(homotopy=True)``
        can march it. See :meth:`_yield_homotopy_control`."""
        return True

    # Picard, not Newton: the consistent yield tangent taken across the elastic
    # stress-history block makes the Jacobian indefinite, and the linear solve fails
    # outright (DIVERGED_LINEAR_SOLVE at 0 iterations). The frozen tangent is
    # contractive, and with the δ-march it still converges to the exact yield surface.
    _yield_homotopy_tangent = False

    @property
    def plastic_fraction(self):
        """Fraction of strain rate that is plastic: 1 - η_vep / η_ve."""
        return sympy.Max(0, 1 - self.viscosity / self.Parameters.ve_effective_viscosity.sym)

    @property
    def is_elastic(self):
        """True if elastic behavior is active (finite dt_elastic and shear_modulus)."""
        # If any of these is not defined, elasticity is switched off

        if self.Parameters.dt_elastic.sym is sympy.oo:
            return False

        if self.Parameters.shear_modulus.sym is sympy.oo:
            return False

        return True

    @property
    def is_viscoplastic(self):
        """True if plastic yielding is active (finite yield_stress)."""
        if self.Parameters.yield_stress.sym is sympy.oo:
            return False

        return True


###


class MaxwellExponentialFlowModel(ViscoElasticPlasticFlowModel):
    r"""Thin alias: ``ViscoElasticPlasticFlowModel(integrator='etd', order=1)``.

    .. deprecated:: Phase B
        Use the canonical form ``ViscoElasticPlasticFlowModel(unknowns,
        integrator='etd', order=1)`` directly. This sibling class
        survives as a thin scaffold so existing scripts continue to
        work; defaults to ETD-1 (recommended).

    See ``docs/developer/design/EXPONENTIAL_VE_INTEGRATOR.md`` for the
    formulation.
    """

    def __init__(self, unknowns, material_name=None):
        super().__init__(
            unknowns, order=1, integrator="etd",
            material_name=material_name,
        )



###


class DiffusionModel(Constitutive_Model):
    r"""
    Diffusion (Fourier/Fick) constitutive model for scalar transport.

    Defines the flux-gradient relationship for scalar diffusion:

    .. math::

        q_{i} = \kappa_{ij} \frac{\partial \phi}{\partial x_j}

    For isotropic diffusion, :math:`\kappa_{ij} = \kappa \delta_{ij}`.

    Parameters
    ----------
    unknowns : Unknowns
        The solver unknowns (the scalar field being diffused).
    material_name : str, optional
        Name identifier for this material.

    Examples
    --------
    >>> diffusion = uw.constitutive_models.DiffusionModel(poisson.Unknowns)
    >>> diffusion.Parameters.diffusivity = 1e-6  # m^2/s
    >>> poisson.constitutive_model = diffusion

    See Also
    --------
    AnisotropicDiffusionModel : For direction-dependent diffusivity.
    """

    class _Parameters(_ParameterBase):
        """Any material properties that are defined by a constitutive relationship are
        collected in the parameters which can then be defined/accessed by name in
        individual instances of the class.

        Now uses Parameter descriptor pattern for automatic lazy evaluation preservation
        with unit-aware quantities.
        """

        # Import Parameter descriptor (must use absolute import inside nested class)
        import underworld3.utilities._api_tools as api_tools

        # Define diffusivity as a Parameter descriptor
        # The lambda receives the _Parameters instance and creates the expression via the owning model
        diffusivity = api_tools.Parameter(
            r"\upkappa",
            lambda params_instance: params_instance._owning_model.create_unique_symbol(
                r"\upkappa", 1, "Diffusivity"
            ),
            "Diffusivity",
            units="m**2/s"  # Thermal or mass diffusivity
        )

        def __init__(
            inner_self,
            _owning_model,
        ):
            inner_self._owning_model = _owning_model
            # Note: diffusivity is now a descriptor, no need to create it here

    @property
    def K(self):
        r"""Diffusivity :math:`\kappa` (alias for ``diffusivity``)."""
        return self.Parameters.diffusivity

    @property
    def diffusivity(self):
        r"""Scalar or tensor diffusivity :math:`\kappa`."""
        return self.Parameters.diffusivity

    def _build_c_tensor(self):
        """Build isotropic diffusivity tensor from scalar."""

        d = self.dim
        kappa = self.Parameters.diffusivity

        # Scalar diffusivity case
        # Use element-wise construction (consistent with ViscousFlowModel pattern)
        # to handle UWexpression properly and preserve for JIT unwrapping
        result = sympy.Matrix.zeros(d, d)

        for i in range(d):
            for j in range(d):
                if i == j:
                    # Diagonal element: kappa
                    val = kappa
                    # Wrap if bare UWexpression to avoid Iterable check failure
                    if hasattr(val, '__getitem__') and not isinstance(val, (sympy.MatrixBase, sympy.NDimArray)):
                        val = sympy.Mul(sympy.S.One, val, evaluate=False)
                    result[i, j] = val
                # Off-diagonal elements remain 0

        self._c = result

        return

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        ## feedback on this instance
        display(
            Latex(r"$\quad\kappa = $ " + sympy.sympify(self.Parameters.diffusivity)._repr_latex_())
        )

        return


# AnisotropicDiffusionModel: expects a diffusivity vector and builds a diagonal tensor.
class AnisotropicDiffusionModel(DiffusionModel):
    r"""Anisotropic diffusion with direction-dependent diffusivities.

    Defines a diagonal diffusivity tensor :math:`\kappa_{ij} = \text{diag}(\kappa_0, \kappa_1, ...)`
    for direction-dependent diffusion rates.
    """

    class _Parameters(_ParameterBase):
        def __init__(inner_self, _owning_model):
            dim = _owning_model.dim
            inner_self._owning_model = _owning_model
            # Set default diffusivity as an identity matrix wrapped in an expression
            default_diffusivity = sympy.ones(_owning_model.dim, 1)
            elements = [default_diffusivity[i] for i in range(dim)]
            validated = []
            for i, v in enumerate(elements):
                comp = validate_parameters(
                    rf"\upkappa_{{{i}}}", v, f"Diffusivity in x_{i}", allow_number=True
                )
                if comp is not None:
                    validated.append(comp)
            # Store the validated diffusivity as a diagonal matrix
            inner_self._diffusivity = sympy.diag(*validated)

        @property
        def diffusivity(inner_self):
            """Diagonal diffusivity tensor."""
            return inner_self._diffusivity

        @diffusivity.setter
        def diffusivity(inner_self, value: sympy.Matrix):
            """Set diffusivity from a vector of per-direction values."""
            dim = inner_self._owning_model.dim

            # Accept shape (dim, 1) or (1, dim)
            if value.shape not in [(dim, 1), (1, dim)]:
                raise ValueError(
                    f"Diffusivity must be a vector of length {dim}. Got shape {value.shape}."
                )
            # Validate each component using validate_parameters
            elements = [value[i] for i in range(dim)]
            validated = []
            for i, v in enumerate(elements):
                diff = validate_parameters(
                    rf"\upkappa_{{{i}}}", v, f"Diffusivity in x_{i}", allow_number=True
                )
                if diff is not None:
                    validated.append(diff)
            # Store the validated diffusivity as a diagonal matrix
            inner_self._diffusivity = sympy.diag(*validated)
            inner_self._reset()

    def _build_c_tensor(self):
        """Constructs the anisotropic (diagonal) tensor from the diffusivity vector."""
        self._c = self.Parameters.diffusivity
        self._is_setup = True

    def _object_viewer(self):
        from IPython.display import Latex, display

        super()._object_viewer()

        diagonal = self.Parameters.diffusivity.diagonal()
        latex_entries = ", ".join([sympy.latex(k) for k in diagonal])
        kappa_latex = r"\kappa = \mathrm{diag}\left(" + latex_entries + r"\right)"
        display(Latex(r"$\quad " + kappa_latex + r"$"))


class GenericFluxModel(Constitutive_Model):
    r"""
    A generic constitutive model with symbolic flux expression.

    Example usage:
    ```python
    grad_phi = sympy.Matrix([sp.Symbol("dphi_dx"), sp.Symbol("dphi_dy")])
    flux_expr = sympy.Matrix([[kappa_11, kappa_12], [kappa_21, kappa_22]]) * grad_phi

    model = GenericFluxModel(dim=2)
    model.flux = flux_expr
    scalar_solver.constititutive_model = model
    ```
    """

    class _Parameters(_ParameterBase):
        def __init__(inner_self, _owning_model):
            inner_self._owning_model = _owning_model

            default_flux = sympy.zeros(_owning_model.dim, 1)
            elements = [default_flux[i] for i in range(_owning_model.dim)]
            validated = []
            for i, v in enumerate(elements):
                flux_component = validate_parameters(
                    rf"q_{{{i}}}", v, f"Flux component in x_{i}", allow_number=True
                )
                if flux_component is not None:
                    validated.append(flux_component)

            inner_self._flux = sympy.Matrix(validated)

        @property
        def flux(inner_self):
            """User-defined flux expression."""
            return inner_self._flux

        @flux.setter
        def flux(inner_self, value: sympy.Matrix):
            """Set the flux expression (must be a vector of length dim)."""
            dim = inner_self._owning_model.dim

            # Accept shape (dim, 1) or (1, dim)
            if value.shape not in [(dim, 1), (1, dim)]:
                raise ValueError(
                    f"Flux must be a symbolic vector of length {dim}. " f"Got shape {value.shape}."
                )

            # Flatten and validate
            elements = [value[i] for i in range(dim)]
            validated = []
            for i, v in enumerate(elements):
                flux_component = validate_parameters(
                    rf"q_{{{i}}}", v, f"Flux component in x_{i}", allow_number=True
                )
                if flux_component is not None:
                    validated.append(flux_component)

            inner_self._flux = sympy.Matrix(validated).reshape(dim, 1)
            inner_self._reset()

    @property
    def flux(self):
        """The user-defined flux expression."""
        # if self._flux is None:
        #     raise RuntimeError("Flux expression has not been set.")
        return self.Parameters.flux

    def _object_viewer(self):
        from IPython.display import display, Latex

        super()._object_viewer()
        if self.flux is not None:
            display(Latex(r"$\vec{q} = " + sympy.latex(self.flux) + "$"))
        else:
            display(Latex(r"No flux expression set."))


class DarcyFlowModel(Constitutive_Model):
    r"""
    Darcy flow constitutive model for porous media flow.

    The ``flux`` property returns the assembly flux
    :math:`\mathbf{F}` that enters the PDE as :math:`-\nabla\cdot\mathbf{F} = f`:

    .. math::

        F_{i} = \kappa_{ij} \left( \frac{\partial p}{\partial x_j} - s_j \right)

    where :math:`\kappa` is the permeability (or hydraulic conductivity),
    :math:`p` is the pressure (or hydraulic head), and :math:`s` is the
    body force term (e.g., gravity: :math:`s = \rho g`). The **physical Darcy
    velocity** is minus this (flow runs *down* the head gradient):

    .. math::

        q_{i} = -F_{i} = -\kappa_{ij} \left( \frac{\partial p}{\partial x_j} - s_j \right)

    Parameters
    ----------
    unknowns : Unknowns
        The solver unknowns (the pressure/head field).
    material_name : str, optional
        Name identifier for this material.

    Examples
    --------
    >>> darcy = uw.constitutive_models.DarcyFlowModel(solver.Unknowns)
    >>> darcy.Parameters.permeability = 1e-12  # m^2
    >>> darcy.Parameters.s = [0, -rho * g]  # Gravity in y-direction
    >>> solver.constitutive_model = darcy

    See Also
    --------
    DiffusionModel : For pure diffusion without body forces.
    """

    class _Parameters(_ParameterBase):
        """Any material properties that are defined by a constitutive relationship are
        collected in the parameters which can then be defined/accessed by name in
        individual instances of the class.

        Uses Parameter descriptor pattern for scalar permeability.
        Matrix-valued `s` remains instance-level (special case).
        """

        # Import Parameter descriptor (must use absolute import inside nested class)
        import underworld3.utilities._api_tools as api_tools

        # Define permeability as a Parameter descriptor
        permeability = api_tools.Parameter(
            r"\kappa",
            lambda params_instance: params_instance._owning_model.create_unique_symbol(
                r"\kappa", 1, "Permeability"
            ),
            "Permeability",
            units="m**2"  # Intrinsic permeability
        )

        def __init__(
            inner_self,
            _owning_model,
            permeabililty: Union[float, sympy.Function] = 1,  # Note: typo in param name preserved for compatibility
        ):

            inner_self._s = expression(
                R"{s}",
                sympy.Matrix.zeros(
                    rows=1, cols=_owning_model.dim
                ),  # Row matrix (1, dim) to match grad_u from jacobian
                "Gravitational forcing",
            )

            inner_self._owning_model = _owning_model
            # Note: permeability is now a descriptor, no need to create it here

        @property
        def s(inner_self):
            r"""Body force vector (e.g., gravitational source term :math:`\rho \mathbf{g}`)."""
            return inner_self._s

        @s.setter
        def s(inner_self, value: sympy.Matrix):
            """Set the body force vector."""
            # Update expression content in-place to preserve object identity
            # Cannot use validate_parameters() as it doesn't handle matrices
            # UWexpression.sym setter handles sympy.Matrix directly
            inner_self._s.sym = value
            inner_self._reset()

    @property
    def K(self):
        r"""Permeability :math:`\kappa` [m²] - the primary constitutive parameter."""
        return self.Parameters.permeability

    def _build_c_tensor(self):
        """For this constitutive law, we expect just a permeability function"""

        d = self.dim
        kappa = self.Parameters.permeability

        # Scalar permeability case
        # Use element-wise construction (consistent with ViscousFlowModel and DiffusionModel)
        # to handle UWexpression properly and preserve for JIT unwrapping
        result = sympy.Matrix.zeros(d, d)

        for i in range(d):
            for j in range(d):
                if i == j:
                    # Diagonal element: kappa
                    val = kappa
                    # Wrap if bare UWexpression to avoid Iterable check failure
                    if hasattr(val, '__getitem__') and not isinstance(val, (sympy.MatrixBase, sympy.NDimArray)):
                        val = sympy.Mul(sympy.S.One, val, evaluate=False)
                    result[i, j] = val
                # Off-diagonal elements remain 0

        self._c = result

        return

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        ## feedback on this instance
        display(
            Latex(r"$\quad\kappa = $ " + sympy.sympify(self.Parameters.diffusivity)._repr_latex_())
        )

        return

    @property
    def flux(self):
        """Computes the effect of the constitutive tensor on the gradients of the unknowns.
        (always uses the `c` form of the tensor). In general cases, the history of the gradients
        may be required to evaluate the flux.
        """

        ddu = self.grad_u - self.Parameters.s.sym

        return self._q(ddu)


class TransverseIsotropicFlowModel(ViscousFlowModel):
    r"""
    Transversely isotropic (anisotropic) viscous flow model.

    .. math::

        \tau_{ij} = \eta_{ijkl} \cdot \frac{1}{2} \left[ \frac{\partial u_k}{\partial x_l}
        + \frac{\partial u_l}{\partial x_k} \right]

    where :math:`\eta` is the viscosity tensor defined as:

    .. math::

        \eta_{ijkl} = \eta_0 \cdot I_{ijkl} + (\eta_0-\eta_1) \left[ \frac{1}{2} \left[
        n_i n_l \delta_{jk} + n_j n_k \delta_{il} + n_i n_l \delta_{jk}
        + n_j n_l \delta_{ik} \right] - 2 n_i n_j n_k n_l \right]

    and :math:`\hat{\mathbf{n}} \equiv \{n_i\}` is the unit vector defining
    the local orientation of the weak plane (a.k.a. the director).

    The Mandel constitutive matrix is available in ``viscous_model.C`` and the
    rank-4 tensor form is in ``viscous_model.c``.

    Examples
    --------
    >>> viscous_model = TransverseIsotropicFlowModel(dim)
    >>> viscous_model.material_properties = viscous_model.Parameters(
    ...     eta_0=viscosity_fn,
    ...     eta_1=weak_viscosity_fn,
    ...     director=orientation_vector_fn
    ... )
    >>> solver.constitutive_model = viscous_model
    >>> tau = viscous_model.flux(gradient_matrix)
    ---
    """

    def __init__(self, unknowns, material_name: str = None):
        # All this needs to do is define the
        # viscosity property and init the parent(s)
        # In this case, nothing seems to be needed.
        # The viscosity is completely defined
        # in terms of the Parameters

        super().__init__(unknowns, material_name=material_name)

        # self._viscosity = expression(
        #     R"{\eta_0}",
        #     1,
        #     " Apparent viscosity",
        # )

    class _Parameters(_ParameterBase, _ViscousParameterAlias):
        """Any material properties that are defined by a constitutive relationship are
        collected in the parameters which can then be defined/accessed by name in
        individual instances of the class.

        Uses Parameter descriptor pattern for automatic lazy evaluation preservation
        with unit-aware quantities.
        """

        # Import Parameter descriptor (must use absolute import inside nested class)
        import underworld3.utilities._api_tools as api_tools

        shear_viscosity_0 = api_tools.Parameter(
            r"\eta_0",
            lambda inner_self: 1,
            "Shear viscosity",
            units="Pa*s",
        )

        shear_viscosity_1 = api_tools.Parameter(
            r"\eta_1",
            lambda inner_self: 1,
            "Second viscosity",
            units="Pa*s",
        )

        director = api_tools.Parameter(
            r"\hat{n}",
            lambda inner_self: sympy.Matrix([0] * (inner_self._owning_model.dim - 1) + [1]),
            "Director orientation",
            units=None,  # Dimensionless unit vector
        )

        def __init__(
            inner_self,
            _owning_model,
        ):
            inner_self._owning_model = _owning_model
            # Parameters are now descriptors - no manual initialization needed

    ## End of parameters

    @property
    def viscosity(self):
        """Whatever the consistutive model defines as the effective value of viscosity
        in the form of an uw.expression"""

        return self.Parameters.shear_viscosity_0

    @property
    def K(self):
        """Whatever the consistutive model defines as the effective value of viscosity
        in the form of an uw.expression"""

        return self.Parameters.shear_viscosity_0

    @property
    def grad_u(self):
        r"""Symmetric strain rate tensor (with 1/2 factor).

        .. math::
            \dot{\varepsilon}_{ij} = \frac{1}{2}\left(\frac{\partial u_i}{\partial x_j}
            + \frac{\partial u_j}{\partial x_i}\right)
        """
        mesh = self.Unknowns.u.mesh

        return mesh.vector.strain_tensor(self.Unknowns.u.sym)

    def _build_c_tensor(self):
        """For this constitutive law, we expect two viscosity functions
        and a sympy row-matrix that describes the director components n_{i}"""

        if self._is_setup:
            return

        d = self.dim
        dv = uw.maths.tensor.idxmap[d][0]

        # Use .sym to get sympy expressions from Parameters
        eta_0 = self.Parameters.shear_viscosity_0.sym
        eta_1 = self.Parameters.shear_viscosity_1.sym
        n = self.Parameters.director.sym

        Delta = eta_0 - eta_1

        # Use element-wise construction (same pattern as ViscousFlowModel).
        # UWexpression has __getitem__ from MathematicalMixin, making it appear
        # "Iterable" to SymPy's array multiplication operator, which rejects it.
        # Element-wise construction avoids this by creating Mul objects that
        # don't have __getitem__.
        identity = uw.maths.tensor.rank4_identity(d)
        lambda_mat = sympy.MutableDenseNDimArray.zeros(d, d, d, d)

        for i in range(d):
            for j in range(d):
                for k in range(d):
                    for l in range(d):
                        # Build isotropic part element-wise
                        base_val = 2 * identity[i, j, k, l] * eta_0

                        # Anisotropic correction term
                        aniso_correction = (
                            2
                            * Delta
                            * (
                                (
                                    n[i] * n[k] * int(j == l)
                                    + n[j] * n[k] * int(l == i)
                                    + n[i] * n[l] * int(j == k)
                                    + n[j] * n[l] * int(k == i)
                                )
                                / 2
                                - 2 * n[i] * n[j] * n[k] * n[l]
                            )
                        )

                        val = base_val - aniso_correction

                        # Wrap if needed to avoid Iterable check during assignment
                        if hasattr(val, '__getitem__') and not isinstance(val, (sympy.MatrixBase, sympy.NDimArray)):
                            val = sympy.Mul(sympy.S.One, val, evaluate=False)

                        lambda_mat[i, j, k, l] = val

        lambda_mat = uw.maths.tensor.rank4_to_mandel(lambda_mat, d)

        self._c = uw.maths.tensor.mandel_to_rank4(lambda_mat, d)

        self._is_setup = True
        self._solver_is_setup = False

        return

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        ## feedback on this instance
        display(Latex(r"$\quad\eta_0 = $ " + sympy.sympify(self.Parameters.shear_viscosity_0)._repr_latex_()))
        display(Latex(r"$\quad\eta_1 = $ " + sympy.sympify(self.Parameters.shear_viscosity_1)._repr_latex_()))
        display(
            Latex(
                r"$\quad\hat{\mathbf{n}} = $ "
                + sympy.sympify(self.Parameters.director.T)._repr_latex_()
            )
        )


class TransverseIsotropicVEPFlowModel(TransverseIsotropicFlowModel):
    r"""Transversely isotropic viscoelastic-plastic flow model for fault mechanics.

    Combines the anisotropic viscosity tensor from :class:`TransverseIsotropicFlowModel`
    with viscoelastic stress history and plastic yield limiting on the fault plane.

    The anisotropic viscosity tensor uses two viscosities (η₀ for the bulk,
    η₁ for fault-plane shear) and a director n̂ defining the weak plane.
    The yield stress τ_y limits the shear stress resolved on the fault plane.

    Parameters
    ----------
    unknowns : Unknowns
        Solver unknowns (velocity, pressure).
    order : int, default=1
        Time integration order for stress history (1 or 2).
    material_name : str, optional
        Name for disambiguation in multi-material setups.

    See Also
    --------
    TransverseIsotropicFlowModel : Anisotropic viscous model (no yield/elasticity).
    ViscoElasticPlasticFlowModel : Isotropic VEP model.
    """

    def __init__(self, unknowns, order=1, integrator: str = "bdf",
                 fault_weight=None,
                 material_name: str = None):
        """Construct a transversely isotropic VEP flow model.

        Parameters
        ----------
        unknowns : Unknowns
            Solver unknowns (velocity, pressure).
        order : int, default 1
            Time-integration order. Combines with ``integrator``:

            - ``integrator='bdf', order=1``: BDF-1 (backward Euler).
            - ``integrator='bdf', order=2``: BDF-2.
            - ``integrator='etd', order=1``: ETD-1 (single-step,
              fully L-stable, **recommended default for VEP+yield**).
              Reproduces BDF-1 byte-identically on the killer test;
              wins at large ``Δt/τ`` where the analytical relaxation
              factor matters.
            - ``integrator='etd', order=2``: ETD-2 (linear-quadrature
              forcing history). 4× more accurate than BDF-2 on smooth
              VE but blows up under active yield in tight-yield TI
              faults — see lessons #7, #9 in EXPONENTIAL_VE_INTEGRATOR.md.

            ``integrator='hybrid'`` pins ``order=1``.
        integrator : str, default "bdf"
            Time integration scheme:

            - ``'bdf'``: Backward differentiation (default, robust for VEP).
            - ``'etd'``: Exponential time-differencing — analytical
              relaxation factor ``α = exp(-Δt/τ)``. Pair with ``order=1``
              for the recommended default; ``order=2`` is available but
              unsafe under active yield (see lessons #7, #9).
            - ``'hybrid'``: **EXPERIMENTAL — DO NOT USE FOR PRODUCTION.**
              Spatial blend of BDF (inside fault) and ETD (outside
              fault). Phase E investigation: σ enforcement is
              BDF-class but |u_y| drifts monotonically over cycles
              from shared-history coupling between the two branches.
              See ``docs/developer/design/EXPONENTIAL_VE_INTEGRATOR.md``
              lesson #11. Use ``'bdf'`` for deep-yield fault problems.
              Requires ``fault_weight`` parameter.
        fault_weight : sympy expression, optional
            Spatial weight ``w(x) ∈ [0, 1]`` selecting BDF (``w=1``) vs
            ETD (``w=0``) per quadrature point. Required when
            ``integrator='hybrid'``. Typically built from the
            ``influence_function`` used to construct ``yield_stress``,
            normalised so that ``w=1`` inside the fault zone (where
            yielding can happen) and ``w=0`` in the bulk (where
            ``τ_y → τ_y_bulk`` and yielding is structurally
            unreachable). The flux blend is
            ``σ = w·σ_BDF + (1-w)·σ_ETD``.
        material_name : str, optional
            Name identifier for this material.
        """
        if integrator not in ("bdf", "etd", "hybrid"):
            raise ValueError(
                f"integrator must be 'bdf', 'etd', or 'hybrid', "
                f"got '{integrator!r}'"
            )
        if integrator == "etd" and order not in (1, 2):
            raise ValueError(
                f"integrator='etd' supports order=1 (ETD-1, default-recommended) "
                f"or order=2 (ETD-2; avoid in tight-yield VEP). "
                f"Got order={order}."
            )
        self._integrator = integrator
        if integrator == "hybrid" and order != 1:
            import warnings
            warnings.warn(
                f"integrator='hybrid' uses one stress history slot; "
                f"``order`` is pinned to 1 (you passed order={order}).",
                UserWarning, stacklevel=2,
            )
            order = 1
        if integrator == "hybrid" and fault_weight is None:
            raise ValueError(
                "integrator='hybrid' requires a ``fault_weight`` sympy "
                "expression in [0, 1] (1 inside fault → BDF; 0 outside → ETD)."
            )
        self._fault_weight = fault_weight

        self._material_name = material_name

        # Stress history expressions
        self._stress_star = expression(
            r"{\tau^{*}}", None,
            r"Lagrangian Stress at $t - \delta_t$",
        )
        self._stress_2star = expression(
            r"{\tau^{**}}", None,
            r"Lagrangian Stress at $t - 2\delta_t$",
        )
        self._E_eff = expression(
            r"{\dot{\varepsilon}_{\textrm{eff}}}", None,
            "Equivalent value of strain rate (accounting for stress history)",
        )
        self._E_eff_inv_II = expression(
            r"{\dot{\varepsilon}_{II,\textrm{eff}}}", None,
            "Equivalent value of strain rate 2nd invariant (accounting for stress history)",
        )

        self._order = order
        self._yield_mode = "softmin"
        self._yield_softness = 0.1
        # BDF order-blending α ∈ [0, 1].  α=1 → pure BDF-2 (default);
        # α=0 → pure BDF-1 coefficients; intermediate → linear blend.
        #
        # NOTE: TI-VEP at order=2 with a spatially varying ``yield_stress``
        # field (e.g. ``influence_function``-localised faults) is unstable
        # for α > ~0.25.  The recommended user-facing fix is ``order=1``
        # (the class default).  This knob is left as an *explicit* damping
        # option for users who must use order=2 — it doesn't paper over
        # the bug at the default.  Empirical stability threshold:
        # α ≤ 0.25 stable, α ≥ 0.5 blow-up.  Investigated 2026-04.
        self._bdf_blend = 1.0
        self._max_dt_ratio_for_higher_order = 2.0

        # Timestep (set by solver)
        self._dt = expression(r"{\Delta t}", sympy.oo, "Timestep (set by solver)")

        # BDF coefficients (initialised to BDF-1)
        self._bdf_c0 = expression(r"{c_0^{\mathrm{BDF}}}", sympy.Integer(1), "BDF leading coefficient")
        self._bdf_c1 = expression(r"{c_1^{\mathrm{BDF}}}", sympy.Integer(-1), "BDF history coefficient 1")
        self._bdf_c2 = expression(r"{c_2^{\mathrm{BDF}}}", sympy.Integer(0), "BDF history coefficient 2")
        self._bdf_c3 = expression(r"{c_3^{\mathrm{BDF}}}", sympy.Integer(0), "BDF history coefficient 3")

        self._reset()

        super().__init__(unknowns, material_name=material_name)

        return

    class _Parameters(_ParameterBase, _ViscousParameterAlias):
        """Parameters for transverse isotropic VEP model.

        Combines anisotropic parameters (η₀, η₁, director) with VEP
        parameters (shear_modulus, yield_stress, etc.).
        """

        import underworld3.utilities._api_tools as api_tools

        # Anisotropic parameters
        shear_viscosity_0 = api_tools.Parameter(
            r"\eta_0", lambda inner_self: 1,
            "Bulk shear viscosity", units="Pa*s",
        )
        shear_viscosity_1 = api_tools.Parameter(
            r"\eta_1", lambda inner_self: 1,
            "Fault-plane shear viscosity", units="Pa*s",
        )
        director = api_tools.Parameter(
            r"\hat{n}",
            lambda inner_self: sympy.Matrix([0] * (inner_self._owning_model.dim - 1) + [1]),
            "Director orientation (fault normal)", units=None,
        )

        # Elastic parameter
        shear_modulus = api_tools.Parameter(
            R"{\mu}", lambda inner_self: sympy.oo,
            "Shear modulus", units="Pa",
        )

        # Timestep (managed by solver)
        @property
        def dt_elastic(inner_self):
            """Timestep for VE formulas. Set by the solver."""
            return inner_self._owning_model._dt

        @dt_elastic.setter
        def dt_elastic(inner_self, value):
            if hasattr(value, 'sym'):
                inner_self._owning_model._dt.sym = value.sym
            else:
                inner_self._owning_model._dt.sym = value

        # Viscosity limits
        shear_viscosity_min = api_tools.Parameter(
            R"{\eta_{\textrm{min}}}",
            lambda inner_self: -sympy.oo,
            "Shear viscosity, minimum cutoff", units="Pa*s",
        )

        # Yield parameters (applied to fault-plane shear)
        yield_stress = api_tools.Parameter(
            R"{\tau_{y}}", lambda inner_self: sympy.oo,
            "Yield stress (fault-plane shear)", units="Pa",
        )
        yield_stress_min = api_tools.Parameter(
            R"{\tau_{y, \mathrm{min}}}",
            lambda inner_self: 0,
            "Yield stress minimum cutoff", units="Pa",
        )
        strainrate_inv_II_min = api_tools.Parameter(
            R"{\dot\varepsilon_{II,\mathrm{min}}}",
            lambda inner_self: 0,
            "Strain rate invariant minimum value", units="1/s",
        )

        def __init__(inner_self, _owning_model):
            inner_self._owning_model = _owning_model

            inner_self._ve_effective_viscosity = expression(
                R"{\eta_{\mathrm{eff}}}", None,
                "Effective viscosity (elastic, fault-plane)",
            )
            inner_self._t_relax = expression(
                R"{t_{\mathrm{relax}}}", None,
                "Maxwell relaxation time",
            )

        @property
        def ve_effective_viscosity(inner_self):
            r"""VE effective viscosity using η₁ (fault-plane viscosity)."""
            mu_val = inner_self.shear_modulus.sym if hasattr(inner_self.shear_modulus, 'sym') else inner_self.shear_modulus
            if mu_val is sympy.oo:
                return inner_self.shear_viscosity_1

            eta = inner_self.shear_viscosity_1
            mu = inner_self.shear_modulus
            dt_e = inner_self.dt_elastic
            c0 = inner_self._owning_model._bdf_c0

            el_eff_visc = eta * mu * dt_e / (c0 * eta + mu * dt_e)
            inner_self._ve_effective_viscosity.sym = el_eff_visc
            return inner_self._ve_effective_viscosity

        @property
        def t_relax(inner_self):
            r"""Maxwell relaxation time: η₁ / μ."""
            inner_self._t_relax.sym = inner_self.shear_viscosity_1 / inner_self.shear_modulus
            return inner_self._t_relax

    ## End of parameters

    @property
    def is_elastic(self):
        """True if elastic behavior is active (finite shear_modulus)."""
        if self.Parameters.shear_modulus.sym is sympy.oo:
            return False
        return True

    @property
    def is_viscoplastic(self):
        """True if plastic yielding is active (finite yield_stress)."""
        if self.Parameters.yield_stress.sym is sympy.oo:
            return False
        return True

    @property
    def order(self):
        """Time integration order (1 or 2)."""
        return self._order

    @order.setter
    def order(self, value):
        """Set time integration order (warns if DFDt already created)."""
        self._order = value
        self._reset()
        solver = getattr(self.Parameters, '_solver', None)
        if solver is not None:
            ddt = getattr(solver.Unknowns, 'DFDt', None)
            if ddt is not None and ddt.order < value:
                import warnings
                warnings.warn(
                    f"Setting order={value} but DFDt was created with order={ddt.order}. "
                    f"Create the model with the desired order before assigning to the solver.",
                    UserWarning, stacklevel=2,
                )
            elif ddt is not None:
                solver._order = value
        return

    @property
    def effective_order(self):
        """Effective order accounting for DDt history startup."""
        if self.Unknowns is not None and self.Unknowns.DFDt is not None:
            ddt_eff = self.Unknowns.DFDt.effective_order
            return min(self._order, ddt_eff)
        return self._order

    def _update_etd_coefficients(self):
        """Refresh DDt's (α, φ) UWexpressions from τ_eff = η_1/μ."""
        if self.Unknowns.DFDt is None:
            return
        params = self.Parameters
        if params.shear_modulus.sym is sympy.oo:
            tau_eff = sympy.oo
        else:
            try:
                eta_val = float(params.shear_viscosity_1.sym)
                mu_val = float(params.shear_modulus.sym)
                tau_eff = eta_val / mu_val if mu_val > 0 else sympy.oo
            except (TypeError, ValueError):
                tau_eff = None
        try:
            dt_val = (
                float(params.dt_elastic.sym)
                if params.dt_elastic.sym is not sympy.oo
                else None
            )
        except (TypeError, ValueError):
            dt_val = None
        self.Unknowns.DFDt.update_exp_coefficients(dt_val, tau_eff)

    def _update_history_coefficients(self):
        r"""Pre-solve hook — dispatches BDF, ETD (order 1/2), or hybrid.

        BDF: refresh ``_bdf_c0..c3``. ETD-2 (order=2): refresh ``α, φ``
        on the DDt from ``η_1/μ``. ETD-1 (order=1): same plus force
        ``φ = α`` so the ``(φ-α)·ε̇*`` history term vanishes. Hybrid:
        refresh both BDF and ETD-2 — flux uses both per-spatial weight.
        """
        if self._integrator == "etd":
            self._update_etd_coefficients()
            if self._order == 1:
                # ETD-1 reduction: φ = α zeros (φ-α)·ε̇* and turns
                # (1-φ)·ε̇ into (1-α)·ε̇.
                DDt = self.Unknowns.DFDt
                if DDt is not None:
                    DDt._exp_phi.sym = DDt._exp_alpha.sym
        elif self._integrator == "hybrid":
            # Hybrid uses BOTH integrators with spatial blend; update
            # both coefficient sets each step.
            self._update_bdf_coefficients()
            self._update_etd_coefficients()
        else:
            self._update_bdf_coefficients()

    def _update_history_post_solve(self):
        """Post-solve hook — refresh forcing_star for ETD-2 / hybrid.

        ETD-1 (order=1) doesn't need this; the (φ-α)·ε̇* term zeros out.
        """
        needs_forcing = (
            (self._integrator == "etd" and self._order == 2)
            or self._integrator == "hybrid"
        )
        if needs_forcing and self.Unknowns.DFDt is not None:
            if self.Unknowns.DFDt.forcing_star is not None:
                self.Unknowns.DFDt.update_forcing_history(forcing_fn=self.Unknowns.E)

    @property
    def stress_history_ddt_kwargs(self):
        """ETD-2 (order=2) and hybrid need the forcing-history slot;
        BDF and ETD-1 do not.
        """
        if self._integrator == "hybrid":
            return {"with_forcing_history": True}
        if self._integrator == "etd" and self._order == 2:
            return {"with_forcing_history": True}
        return {}

    def _update_bdf_coefficients(self):
        """Update BDF coefficient UWexpressions with blending."""
        order = self.effective_order

        if self.Unknowns is not None and self.Unknowns.DFDt is not None:
            dt_current = self.Parameters.dt_elastic
            if hasattr(dt_current, 'sym'):
                dt_current = dt_current.sym

            dt_history = self.Unknowns.DFDt._dt_history
            if order >= 2 and len(dt_history) > 0 and dt_history[0] is not None:
                try:
                    ratio = float(dt_current) / float(dt_history[0])
                    if ratio > self._max_dt_ratio_for_higher_order:
                        order = 1
                except (TypeError, ZeroDivisionError):
                    pass

            coeffs = _bdf_coefficients(order, dt_current, dt_history)
        else:
            coeffs = _bdf_coefficients(order, None, [])

        # BDF order blending — see ``self._bdf_blend`` docstring.
        # Linear mix of BDF-1 and the requested-order coefficients.
        # α=0 → pure BDF-1; α=1 → no blend (skip).
        alpha = self._bdf_blend
        if alpha < 1 and order >= 2:
            coeffs_o1 = _bdf_coefficients(1, dt_current, dt_history) \
                if (self.Unknowns is not None and self.Unknowns.DFDt is not None) \
                else _bdf_coefficients(1, None, [])
            while len(coeffs_o1) < len(coeffs):
                coeffs_o1.append(sympy.Integer(0))
            coeffs = [
                (1 - alpha) * c1 + alpha * ck
                for c1, ck in zip(coeffs_o1, coeffs)
            ]

        while len(coeffs) < 4:
            coeffs.append(sympy.Integer(0))

        self._bdf_c0.sym = coeffs[0]
        self._bdf_c1.sym = coeffs[1]
        self._bdf_c2.sym = coeffs[2]
        self._bdf_c3.sym = coeffs[3]

    @property
    def bdf_blend(self):
        r"""BDF coefficient blending α ∈ [0, 1].  **Damping knob.**

        Linearly mixes BDF-1 and the requested-order coefficients:
        ``c = (1-α)·c_BDF1 + α·c_requested_order``.  α=1 is no blend.

        **For TI-VEP fault simulations, prefer ``order=1`` over tuning
        this knob.**  At ``order=2`` with a spatially varying
        ``yield_stress`` field, the simulation drifts unstably:
        |σ_xy| → 10⁸ over ~10 t_r.  Empirically the instability is
        gated by the magnitude of the ψ*_{n-1} weight: α ≤ 0.25 stable,
        α ≥ 0.5 blow-up.  Lower α values stabilise but throw away
        most of BDF-2's accuracy advantage — at α=0.10 the trace
        difference vs ``order=1`` is ~0.1% of peak, for ~50% wall-time
        overhead.  Use this knob only if you specifically need
        order-2 behaviour on a problem with uniform yield_stress.
        """
        return self._bdf_blend

    @bdf_blend.setter
    def bdf_blend(self, value):
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"bdf_blend must be in [0, 1], got {value}")
        self._bdf_blend = float(value)

    @property
    def stress_star(self):
        r"""Previous timestep stress from history."""
        if self.Unknowns.DFDt is not None:
            self._stress_star.sym = self.Unknowns.DFDt.psi_star[0].sym
        return self._stress_star

    def _e_eff_for(self, integrator_mode):
        r"""Build E_eff for a given integrator mode (without storing on
        ``self._E_eff``). Used by both the public :py:attr:`E_eff` and
        the hybrid ``stress()`` path which needs both forms in one
        evaluation.
        """
        E = self.Unknowns.E
        if self.Unknowns.DFDt is None or not self.is_elastic:
            return E
        DDt = self.Unknowns.DFDt

        if integrator_mode == "etd":
            alpha = DDt._exp_alpha
            phi = DDt._exp_phi
            sigma_star = DDt.psi_star[0].sym
            if DDt.forcing_star is not None:
                edot_star = DDt.forcing_star.sym
            else:
                edot_star = sympy.zeros(*E.shape)
            eta_1 = self.Parameters.shear_viscosity_1
            return (
                (1 - phi) * E
                + (alpha / (2 * eta_1)) * sigma_star
                + (phi - alpha) * edot_star
            )

        # BDF default
        mu_dt = self.Parameters.dt_elastic * self.Parameters.shear_modulus
        bdf_cs = [self._bdf_c1, self._bdf_c2, self._bdf_c3]
        out = E
        for i in range(DDt.order):
            out = out - bdf_cs[i] * DDt.psi_star[i].sym / (2 * mu_dt)
        return out

    @property
    def E_eff(self):
        r"""Effective strain rate including elastic-history coupling.

        BDF: ``E_eff = ε̇ - Σc_i·σ*/(2μΔt)``.
        ETD-2: ``E_eff = (1-φ)·ε̇ + α·σ*/(2η₁) + (φ-α)·ε̇*``.
        Hybrid: returns the BDF form (E_eff is consumed by yield-clip
        code that should see the BDF rate metric); the actual flux uses
        both forms inside :py:meth:`stress`.
        """
        mode = "bdf" if self._integrator in ("bdf", "hybrid") else "etd"
        self._E_eff.sym = self._e_eff_for(mode)
        return self._E_eff

    @property
    def E_eff_inv_II(self):
        r"""Second invariant of effective strain rate."""
        E_eff = self.E_eff.sym
        self._E_eff_inv_II.sym = sympy.sqrt((E_eff**2).trace() / 2)
        return self._E_eff_inv_II

    @property
    def viscosity(self):
        r"""Effective viscosity for the fault-plane shear component.

        Applies the yield mode (softmin/min/harmonic) to η₁, leaving
        η₀ (bulk) unchanged. The anisotropic tensor handles the
        directional dependence.
        """
        inner_self = self.Parameters

        if inner_self.yield_stress.sym == sympy.oo:
            return inner_self.shear_viscosity_0

        # η₁ is the fault-plane viscosity that gets yield-limited
        eta_1_eff = inner_self.ve_effective_viscosity

        if self.is_viscoplastic:
            vp_eff = self._plastic_effective_viscosity
            # Same yield envelope as the isotropic models: _combine_yield owns the
            # harmonic / soft-min / hard-Min choice, reads delta from the
            # constants[] atom (so a homotopy can ramp it without a recompile) and
            # honours yield_smoother. Previously inlined here, which pinned this
            # model to the sqrt family and to a baked float delta.
            eta_1_eff = self._combine_yield(eta_1_eff, vp_eff)

        return inner_self.shear_viscosity_0

    @property
    def K(self):
        """Effective stiffness for preconditioner."""
        return self.Parameters.shear_viscosity_0

    @property
    def _plastic_effective_viscosity(self):
        """Plastic viscosity from resolved fault-plane shear strain rate.

        Computes the in-plane shear magnitude using Pythagoras:
          T = ε̇_eff · n       (traction-like vector on fault)
          ε̇_n = T · n          (normal component)
          |γ̇| = √(|T|² - ε̇_n²) (in-plane shear magnitude)

        This works in both 2D and 3D — no explicit tangent vector needed.
        The formula 2η₁_pl = τ_y / |γ̇| is the same pattern as isotropic
        Drucker-Prager but projected onto the fault plane.
        """
        parameters = self.Parameters

        ty_val = parameters.yield_stress.sym if hasattr(parameters.yield_stress, 'sym') else parameters.yield_stress
        if ty_val is sympy.oo:
            return sympy.oo

        Edot = self.E_eff.sym

        # Resolve strain rate onto fault plane via Pythagoras
        n = parameters.director.sym
        T = Edot * n                        # "traction" vector on fault
        edot_n = (n.T * T)[0, 0]            # normal component
        T_sq = (T.T * T)[0, 0]              # |T|²
        gamma_dot_sq = T_sq - edot_n**2     # in-plane shear²
        gamma_dot_abs = sympy.sqrt(sympy.Max(gamma_dot_sq, 0))

        tau_y = parameters.yield_stress
        # Guard on the DISABLING sentinel, not on zero — zero is the default and a
        # real floor (a negative yield stress is meaningless against an invariant).
        if parameters.yield_stress_min.sym != -sympy.oo:
            # Literal 0 for the default floor (sympy fuzzy-compare recursion on atoms).
            # smooth_max: keeps the floor symbolic, no ordering test (see the note
            # in Constitutive_Model._apply_floor). eps = 0 is exactly Max.
            tau_y = uw.maths.smooth_max(tau_y, parameters.yield_stress_min, 0)

        if parameters.strainrate_inv_II_min.sym != 0:
            viscosity_yield = tau_y / (
                2 * (gamma_dot_abs + parameters.strainrate_inv_II_min)
            )
        else:
            viscosity_yield = tau_y / (2 * gamma_dot_abs)

        return viscosity_yield

    def _eta_for_tensor(self, integrator_mode, apply_yield):
        """Return ``(eta_0, eta_1_eff)`` for tensor build, parameterised
        by integrator mode and whether to apply yield clipping.

        - ``integrator_mode='bdf'``: η₀, η₁ are VE-effective (c₀-baked
          Δt scaling — needed for BDF's E_eff structure).
        - ``integrator_mode='etd'``: η₀, η₁ are raw (time factor lives
          in α/φ symbolically). Used for both ETD-1 and ETD-2 — the
          C tensor is identical; only the symbolic E_eff differs (via
          ``self._order``).
        - ``apply_yield=True``: softmin/min/harmonic clip on η₁_eff.
        - ``apply_yield=False``: no clipping (use this for the ETD-VE
          branch of the hybrid integrator, where the bulk is
          structurally non-yieldable so clipping is a no-op anyway).
        """
        if integrator_mode == "etd":
            eta_0 = self.Parameters.shear_viscosity_0.sym
            eta_1_eff = self.Parameters.shear_viscosity_1
        else:  # bdf
            eta_0_raw = self.Parameters.shear_viscosity_0
            mu = self.Parameters.shear_modulus
            dt_e = self.Parameters.dt_elastic
            c0 = self._bdf_c0
            mu_val = mu.sym if hasattr(mu, 'sym') else mu
            if mu_val is sympy.oo:
                eta_0 = eta_0_raw.sym if hasattr(eta_0_raw, 'sym') else eta_0_raw
            else:
                eta_0 = eta_0_raw * mu * dt_e / (c0 * eta_0_raw + mu * dt_e)
            eta_1_eff = self.Parameters.ve_effective_viscosity

        if apply_yield and self.is_viscoplastic:
            vp_eff = self._plastic_effective_viscosity
            # Same yield envelope as the isotropic models: _combine_yield owns the
            # harmonic / soft-min / hard-Min choice, reads delta from the
            # constants[] atom (so a homotopy can ramp it without a recompile) and
            # honours yield_smoother. Previously inlined here, which pinned this
            # model to the sqrt family and to a baked float delta.
            eta_1_eff = self._combine_yield(eta_1_eff, vp_eff)
        return eta_0, eta_1_eff

    def _assemble_c_tensor(self, eta_0, eta_1_eff):
        """Build the anisotropic rank-4 tensor from ``(eta_0, eta_1_eff)``.

        Loop body identical to :py:meth:`_build_c_tensor_ve`; refactored
        into a helper so the hybrid path can call it twice (BDF tensor
        with yield clip, ETD tensor without) without code duplication.
        """
        d = self.dim
        n = self.Parameters.director.sym
        Delta = eta_0 - eta_1_eff
        identity = uw.maths.tensor.rank4_identity(d)
        lambda_mat = sympy.MutableDenseNDimArray.zeros(d, d, d, d)
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    for l in range(d):
                        base_val = 2 * identity[i, j, k, l] * eta_0
                        aniso_correction = (
                            2 * Delta * (
                                (n[i] * n[k] * int(j == l)
                                 + n[j] * n[k] * int(l == i)
                                 + n[i] * n[l] * int(j == k)
                                 + n[j] * n[l] * int(k == i)) / 2
                                - 2 * n[i] * n[j] * n[k] * n[l]
                            )
                        )
                        val = base_val - aniso_correction
                        if hasattr(val, '__getitem__') and not isinstance(val, (sympy.MatrixBase, sympy.NDimArray)):
                            val = sympy.Mul(sympy.S.One, val, evaluate=False)
                        lambda_mat[i, j, k, l] = val
        lambda_mat = uw.maths.tensor.rank4_to_mandel(lambda_mat, d)
        return uw.maths.tensor.mandel_to_rank4(lambda_mat, d)

    def _build_c_tensor(self):
        """Build the anisotropic tensor(s) for the active integrator.

        - ``'bdf'`` / ``'etd'``: single tensor ``self._c``.
        - ``'hybrid'``: two tensors ``self._c_bdf`` (yield-clipped) and
          ``self._c_etd`` (no clip — bulk is non-yieldable). The flux
          blend ``w·σ_BDF + (1-w)·σ_ETD`` happens in :py:meth:`stress`.
        """
        if self._is_setup:
            return

        if self._integrator == "hybrid":
            eta_0_bdf, eta_1_bdf = self._eta_for_tensor("bdf", apply_yield=True)
            self._c_bdf = self._assemble_c_tensor(eta_0_bdf, eta_1_bdf)
            eta_0_etd, eta_1_etd = self._eta_for_tensor("etd", apply_yield=False)
            self._c_etd = self._assemble_c_tensor(eta_0_etd, eta_1_etd)
            self._c = self._c_bdf  # default for any callers reading self._c
        else:
            eta_0, eta_1_eff = self._eta_for_tensor(
                self._integrator, apply_yield=self.is_viscoplastic
            )
            self._c = self._assemble_c_tensor(eta_0, eta_1_eff)

        self._is_setup = True
        self._solver_is_setup = False
        return

    @property
    def flux(self):
        """Stress flux for the weak form."""
        return self.stress()

    def stress_projection(self):
        """VE stress without plastic correction (for history storage).

        Uses the anisotropic tensor with VE effective viscosities but
        no yield limiting (η₁_ve, not η₁_eff). This is the stress that
        should be stored in the DFDt history for the next timestep.
        """
        edot = self.grad_u
        self._build_c_tensor_ve()
        # Contract with the VE-only tensor (not self._c which has yield)
        c_ve = self._c_ve
        if len(c_ve.shape) == 2:
            flux = c_ve * edot
        else:
            flux = sympy.tensorcontraction(
                sympy.tensorcontraction(sympy.tensorproduct(c_ve, edot), (1, 5)), (0, 3)
            )
        return sympy.Matrix(flux)

    def _build_c_tensor_ve(self):
        """Build anisotropic tensor with VE η₁ only (no yield)."""
        d = self.dim
        eta_0 = self.Parameters.shear_viscosity_0.sym
        eta_1_ve = self.Parameters.ve_effective_viscosity
        n = self.Parameters.director.sym
        Delta = eta_0 - eta_1_ve

        identity = uw.maths.tensor.rank4_identity(d)
        lambda_mat = sympy.MutableDenseNDimArray.zeros(d, d, d, d)

        for i in range(d):
            for j in range(d):
                for k in range(d):
                    for l in range(d):
                        base_val = 2 * identity[i, j, k, l] * eta_0
                        aniso_correction = (
                            2 * Delta * (
                                (n[i] * n[k] * int(j == l)
                                 + n[j] * n[k] * int(l == i)
                                 + n[i] * n[l] * int(j == k)
                                 + n[j] * n[l] * int(k == i)) / 2
                                - 2 * n[i] * n[j] * n[k] * n[l]
                            )
                        )
                        val = base_val - aniso_correction
                        if hasattr(val, '__getitem__') and not isinstance(val, (sympy.MatrixBase, sympy.NDimArray)):
                            val = sympy.Mul(sympy.S.One, val, evaluate=False)
                        lambda_mat[i, j, k, l] = val

        lambda_mat = uw.maths.tensor.rank4_to_mandel(lambda_mat, d)
        self._c_ve = uw.maths.tensor.mandel_to_rank4(lambda_mat, d)

    def stress(self):
        """Viscoelastic-plastic anisotropic stress for the weak form.

        Matches isotropic VEP pattern: tensor contraction for current strain
        rate, scalar yield-limited viscosity for VE history terms. The tensor
        C(η₁_eff) handles anisotropy; the history uses the same η₁_eff as
        a scalar multiplier (consistent with how isotropic VEP uses
        self.viscosity for both).

        Hybrid path: ``σ = w·σ_BDF + (1-w)·σ_ETD`` with the spatial
        weight from ``self._fault_weight``. Each branch contracts its
        own E_eff with its own C-tensor; the blend lives at the flux
        level so neither integrator's structure is compromised.
        """
        self._build_c_tensor()

        if self._integrator == "hybrid":
            # σ_BDF: BDF flux with yield-clipped C tensor
            edot_eff_bdf = self._e_eff_for("bdf")
            sigma_bdf = self._q_with(self._c_bdf, edot_eff_bdf)
            # σ_ETD: ETD flux with no-yield C tensor
            edot_eff_etd = self._e_eff_for("etd")
            sigma_etd = self._q_with(self._c_etd, edot_eff_etd)
            # Spatial blend
            w = self._fault_weight
            return w * sigma_bdf + (1 - w) * sigma_etd

        # Apply the anisotropic tensor to the effective strain rate
        # (current + VE history): σ = C(η₀_ve, η₁_eff) : ε̇_eff
        # This is the correct VE formula — the tensor handles anisotropy
        # for both current and history contributions uniformly.
        edot_eff = self.E_eff.sym if hasattr(self.E_eff, 'sym') else self.E_eff
        stress = self._q(edot_eff)

        return stress

    def _q_with(self, c, edot):
        """Apply a given rank-4 tensor to a strain rate (helper for
        the hybrid flux that needs to contract two distinct C tensors
        in one ``stress()`` call).
        """
        rank = len(c.shape)
        if rank == 2:
            flux = c * edot
        else:
            flux = sympy.tensorcontraction(
                sympy.tensorcontraction(sympy.tensorproduct(c, edot), (1, 5)),
                (0, 3),
            )
        return sympy.Matrix(flux)

    @property
    def yield_mode(self):
        r"""How to apply yield limiting to the fault-plane viscosity.

        Same options as :class:`ViscoElasticPlasticFlowModel`:
        ``"softmin"`` (default), ``"harmonic"``, ``"min"``.  The
        ``"smooth"`` option was retired (under-clipped by ~50 %); see
        the parent class's :attr:`yield_mode` docstring for details.
        """
        return self._yield_mode

    @yield_mode.setter
    def yield_mode(self, value):
        if value == "smooth":
            raise ValueError(
                "yield_mode='smooth' has been retired — it under-clipped "
                "the yield surface by ~50%. Use 'softmin' instead "
                "(default; close to exact Min with smooth derivatives)."
            )
        if value not in ("min", "harmonic", "softmin"):
            raise ValueError(
                f"yield_mode must be 'min', 'harmonic', or 'softmin', got '{value}'"
            )
        self._yield_mode = value
        self._reset()

    @property
    def yield_softness(self):
        """Regularisation parameter δ for softmin mode."""
        return self._yield_softness

    @yield_softness.setter
    def yield_softness(self, value):
        self._yield_softness = float(value)
        # Keep the constants[] atom in step: this model now shares the isotropic
        # _combine_yield, which reads δ from that atom rather than from the float.
        if getattr(self, "_yield_softness_expr", None) is not None:
            self._yield_softness_expr.sym = sympy.Float(self._yield_softness)
        self._reset()

    @property
    def requires_stress_history(self):
        """Transverse isotropic VEP requires stress history tracking."""
        return True

    @property
    def supports_yield_homotopy(self):
        """This model carries a δ-parameterised yield law — ``solve(homotopy=True)``
        can march it. It shares the isotropic yield envelope (:meth:`_combine_yield`),
        so ``yield_smoother`` applies here too; its ``yield_softness`` setter still
        triggers a rebuild, so a δ step costs a recompile the isotropic models avoid.
        See :meth:`_yield_homotopy_control`."""
        return True

    # Picard, not Newton — as for the isotropic VEP model, the consistent yield
    # tangent over the elastic stress-history block is indefinite.
    _yield_homotopy_tangent = False

    @property
    def plastic_fraction(self):
        """Fraction of strain rate that is plastic."""
        eta_1_ve = self.Parameters.ve_effective_viscosity
        eta_1_eff = self.viscosity
        # viscosity property returns η₀, need to compare η₁ effective vs η₁ ve
        # This is approximate for the anisotropic case
        return sympy.Max(0, 1 - eta_1_eff / eta_1_ve.sym if hasattr(eta_1_ve, 'sym') else 0)


class TransverseIsotropicMaxwellExponentialFlowModel(TransverseIsotropicVEPFlowModel):
    r"""Thin alias: ``TransverseIsotropicVEPFlowModel(integrator='etd', order=1)``.

    .. deprecated:: Phase B
        Use the canonical form ``TransverseIsotropicVEPFlowModel(unknowns,
        integrator='etd', order=1)`` directly. This sibling class
        survives as a thin scaffold for existing scripts; defaults to
        ETD-1 (recommended).
    """

    def __init__(self, unknowns, material_name=None):
        super().__init__(
            unknowns, order=1, integrator="etd",
            material_name=material_name,
        )


class TransverseIsotropicVEPSplitFlowModel(TransverseIsotropicVEPFlowModel):
    r"""**EXPERIMENTAL — DO NOT USE FOR PRODUCTION.**

    Phase D investigation artefact. The σ_∥ enforcement reaches BDF-class
    fidelity (1.21·τ_y vs BDF's 1.04·τ_y at τ_y=0.05) but the velocity
    field overshoots the boundary value and ratchets monotonically over
    cycles to ~21× BDF-1's |u_y|. The fault-tip stress concentrations
    in the PyVista field plot are non-physical for this loading.

    Retained on the branch for reproducibility of the investigation; see
    ``docs/developer/design/EXPONENTIAL_VE_INTEGRATOR.md`` lessons #9 and
    #10 for why this doesn't ship.

    For deep-yield TI fault problems use
    ``TransverseIsotropicVEPFlowModel(integrator='bdf')``.

    --

    Phase D — per-component ``(α_⊥, φ_⊥)/(α_∥, φ_∥)`` ETD-2 for TI VEP.

    The rank-4 modulus splits into two orthogonal projectors:

    .. math::
        C(\eta_0, \eta_\parallel) = 2\eta_0 \, \mathbf{P}_\perp
            + 2\eta_\parallel \, \mathbf{P}_\parallel

    where :math:`\mathbf{P}_\parallel` is the director-aligned projector
    (the ``K`` kernel built in :py:meth:`_build_c_tensor`) and
    :math:`\mathbf{P}_\perp = \mathbf{I}_4 - \mathbf{P}_\parallel`.
    Each branch has its own Maxwell relaxation time:

    .. math::
        \tau_\perp = \eta_0 / \mu, \qquad
        \tau_\parallel = \eta_\parallel^\text{eff} / \mu

    so the analytical exponential factors differ:

    .. math::
        \alpha_k = e^{-\Delta t / \tau_k}, \qquad
        \varphi_k = (1 - \alpha_k) \tau_k / \Delta t

    The split flux integrates each branch independently and sums:

    .. math::
        \sigma^{n+1} = (\alpha_\perp \mathbf{P}_\perp + \alpha_\parallel
            \mathbf{P}_\parallel) : \sigma^*
            + 2[\eta_0(1-\varphi_\perp) \mathbf{P}_\perp
                 + \eta_\parallel^\text{eff}(1-\varphi_\parallel)
                   \mathbf{P}_\parallel] : \dot\varepsilon^{n+1}
            + 2[\eta_0(\varphi_\perp - \alpha_\perp) \mathbf{P}_\perp
                 + \eta_\parallel^\text{eff}(\varphi_\parallel - \alpha_\parallel)
                   \mathbf{P}_\parallel] : \dot\varepsilon^*

    Phase B uses a single lumped ``(α, φ)`` from ``η_∥_eff/μ`` for the
    whole tensor — empirically blows up at tight yield surfaces because
    the matrix branch has no business being yielded. The split scheme
    relaxes each channel on its proper timescale; the analytical floor
    on ``σ_∥`` is then ``≲ τ_y`` by construction.

    Implementation choice: ``α_⊥, φ_⊥`` come from the DDt's existing
    scalar ``_exp_coeffs`` (matrix viscosity is fixed and spatially
    uniform — a single per-step scalar is right). ``α_∥, φ_∥`` are
    inlined as sympy expressions of the yield-clipped ``η_∥_eff`` so
    the JIT evaluates them per quadrature point (spatial heterogeneity
    captured automatically). No DDt changes, no solver changes.
    """

    # Default cap on τ_∥/Δt — recommendation 4 from the practical
    # stabilisation strategy. α_∥ ≥ exp(-1/c); c=1 → α_∥ ≥ 0.37
    # (37% of σ* retained each step, matches BDF-style elastic
    # damping). Set to 0 to disable (recovers the un-capped behaviour
    # where boundary motion goes straight into plastic slip).
    _tau_par_cap_factor = 1.0

    def __init__(self, unknowns, material_name=None):
        super().__init__(
            unknowns, order=1, integrator="etd",
            material_name=material_name,
        )

    @property
    def tau_par_cap_factor(self):
        r"""Lower bound ``c`` such that ``τ_∥ ≥ c·Δt`` in the parallel
        branch's exponential factor.

        Caps how aggressively the parallel-branch's elastic memory is
        relaxed during yielding. Without this cap, ``η_∥_eff → 0`` at
        deep yield drives ``α_∥ = exp(-Δt/τ_∥) → 0`` — boundary
        motion goes straight into slip each step with no elastic
        spring-back, ratcheting fault displacement at the BC rate.
        With ``c=1`` (default), ``α_∥ ≥ 1/e``; with ``c=2``,
        ``α_∥ ≥ exp(-0.5)``.

        Set to ``0`` to disable (recovers the explicit-plasticity
        behaviour). The cap applies only to the (α_∥, φ_∥) factors —
        ``C_∥`` keeps the natural yield-clipped η_∥ so ``σ_∥`` still
        sits at the yield surface.
        """
        return self._tau_par_cap_factor

    @tau_par_cap_factor.setter
    def tau_par_cap_factor(self, value):
        self._tau_par_cap_factor = float(value)

    def _update_history_coefficients(self):
        r"""Update ``(α_⊥, φ_⊥)`` only — the matrix branch.

        ``(α_∥, φ_∥)`` are inlined per-quadrature in :py:meth:`stress`.
        Picks ``τ_⊥ = η_0/μ`` (raw matrix viscosity).
        """
        if self._integrator != "etd" or self.Unknowns.DFDt is None:
            return super()._update_history_coefficients()
        params = self.Parameters
        if params.shear_modulus.sym is sympy.oo:
            tau_perp = sympy.oo
        else:
            try:
                eta_val = float(params.shear_viscosity_0.sym)
                mu_val = float(params.shear_modulus.sym)
                tau_perp = eta_val / mu_val if mu_val > 0 else sympy.oo
            except (TypeError, ValueError):
                tau_perp = None
        try:
            dt_val = (
                float(params.dt_elastic.sym)
                if params.dt_elastic.sym is not sympy.oo
                else None
            )
        except (TypeError, ValueError):
            dt_val = None
        self.Unknowns.DFDt.update_exp_coefficients(dt_val, tau_perp)

    def _eta_par_eff(self):
        """Yield-clipped ``η_∥_eff`` — same softmin/min/harmonic as parent.

        For ETD the base is the raw ``η_1`` (no VE pre-clip); the yield
        envelope is then applied via the configured yield_mode.
        """
        params = self.Parameters
        eta_par = params.shear_viscosity_1
        if hasattr(eta_par, 'sym'):
            eta_par = eta_par.sym

        if not self.is_viscoplastic or params.yield_stress.sym is sympy.oo:
            return eta_par

        vp_eff = self._plastic_effective_viscosity
        # Same yield envelope as the isotropic models: _combine_yield owns the
        # harmonic / soft-min / hard-Min choice, reads delta from the
        # constants[] atom (so a homotopy can ramp it without a recompile) and
        # honours yield_smoother. Previously inlined here, which pinned this
        # model to the sqrt family and to a baked float delta.
        return self._combine_yield(eta_par, vp_eff)

    def _eta_par_eff_lagged(self):
        """Yield-clipped ``η_∥_eff`` using the **lagged** strain rate
        (``forcing_star``, projected post-solve) instead of the current
        ``E_eff``.

        Same softmin/harmonic/min envelope as :py:meth:`_eta_par_eff`,
        but the plastic estimate ``vp_eff_lag = τ_y/(2|γ̇*|)`` uses the
        *previous step's* fault-plane shear magnitude. This is the
        proper "lag η_∥_eff": elastic regime → η_1_raw (low |γ̇*|);
        yielded regime → ``τ_y/(2|γ̇*|)`` (saturated stress, large rate).
        Using the parent's E_eff-based formula here would couple α_∥
        back into the current Newton iterate, which collapses to a
        1-iteration trivial residual (over-damping). Holding the rate
        fixed at the post-solve value breaks that feedback.
        """
        params = self.Parameters
        DDt = self.Unknowns.DFDt

        eta_par = params.shear_viscosity_1
        if hasattr(eta_par, 'sym'):
            eta_par = eta_par.sym

        if not self.is_viscoplastic or params.yield_stress.sym is sympy.oo:
            return eta_par
        if DDt is None or DDt.forcing_star is None:
            return self._eta_par_eff()  # no history yet — fall back to current

        # Lagged plastic estimate: |γ̇*_∥| from forcing_star + director
        Edot_lag = DDt.forcing_star.sym
        n = params.director.sym
        T_lag = Edot_lag * n
        edot_n_lag = (n.T * T_lag)[0, 0]
        T_sq_lag = (T_lag.T * T_lag)[0, 0]
        gamma_dot_sq_lag = T_sq_lag - edot_n_lag ** 2
        gamma_dot_abs_lag = sympy.sqrt(sympy.Max(gamma_dot_sq_lag, 0))

        # Strip Pint from the ε_min floor (forcing_star is unitless storage)
        edot_min_raw = params.strainrate_inv_II_min.sym
        if hasattr(edot_min_raw, 'magnitude'):
            edot_min_val = float(edot_min_raw.magnitude)
        else:
            try:
                edot_min_val = float(edot_min_raw)
            except (TypeError, ValueError):
                edot_min_val = 1.0e-6

        tau_y_sym = params.yield_stress.sym
        vp_eff_lag = tau_y_sym / (
            2 * (gamma_dot_abs_lag + sympy.Float(edot_min_val))
        )

        # Same yield envelope as the isotropic models: _combine_yield owns the
        # harmonic / soft-min / hard-Min choice, reads delta from the
        # constants[] atom (so a homotopy can ramp it without a recompile) and
        # honours yield_smoother. Previously inlined here, which pinned this
        # model to the sqrt family and to a baked float delta.
        return self._combine_yield(eta_par, vp_eff_lag)

    def _build_split_c_tensors(self, eta_perp, eta_par):
        r"""Build ``C_⊥ = 2·η_⊥·P_⊥`` and ``C_∥ = 2·η_∥·P_∥``.

        Identical loop structure to :py:meth:`_build_c_tensor`, but
        each tensor isolates one projector by zeroing the other
        viscosity coefficient.
        """
        d = self.dim
        n = self.Parameters.director.sym
        identity = uw.maths.tensor.rank4_identity(d)

        c_perp_arr = sympy.MutableDenseNDimArray.zeros(d, d, d, d)
        c_par_arr = sympy.MutableDenseNDimArray.zeros(d, d, d, d)

        for i in range(d):
            for j in range(d):
                for k in range(d):
                    for l in range(d):
                        I_ijkl = identity[i, j, k, l]
                        K_ijkl = (
                            (n[i] * n[k] * int(j == l)
                             + n[j] * n[k] * int(l == i)
                             + n[i] * n[l] * int(j == k)
                             + n[j] * n[l] * int(k == i)) / 2
                            - 2 * n[i] * n[j] * n[k] * n[l]
                        )
                        # 2·η_⊥·P_⊥ = 2·η_⊥·(I - K)
                        v_perp = 2 * eta_perp * (I_ijkl - K_ijkl)
                        # 2·η_∥·P_∥ = 2·η_∥·K
                        v_par = 2 * eta_par * K_ijkl
                        # Same guard as parent _build_c_tensor — sympy
                        # NDimArray.__setitem__ refuses iterable RHS.
                        if hasattr(v_perp, '__getitem__') and not isinstance(
                            v_perp, (sympy.MatrixBase, sympy.NDimArray)
                        ):
                            v_perp = sympy.Mul(sympy.S.One, v_perp, evaluate=False)
                        if hasattr(v_par, '__getitem__') and not isinstance(
                            v_par, (sympy.MatrixBase, sympy.NDimArray)
                        ):
                            v_par = sympy.Mul(sympy.S.One, v_par, evaluate=False)
                        c_perp_arr[i, j, k, l] = v_perp
                        c_par_arr[i, j, k, l] = v_par

        c_perp = uw.maths.tensor.mandel_to_rank4(
            uw.maths.tensor.rank4_to_mandel(c_perp_arr, d), d)
        c_par = uw.maths.tensor.mandel_to_rank4(
            uw.maths.tensor.rank4_to_mandel(c_par_arr, d), d)
        return c_perp, c_par

    def stress(self):
        r"""Per-component ETD-2 flux with **lagged** ``(α_∥, φ_∥)``.

        Each branch's E_eff is built and contracted with its own
        sub-modulus; the two are summed.

        ``α_∥, φ_∥`` are derived from a *lagged* parallel viscosity
        computed from the projected stress and strain-rate histories:

        .. math::
            \eta_\parallel^{\,\mathrm{lag}}
              = \frac{|\sigma^*_\parallel|}
                     {2\,\max(|\dot\varepsilon^*_\parallel|, \dot\varepsilon_\min)}

        where ``|·|_∥`` is the Pythagorean fault-plane shear magnitude
        (same pattern as :py:meth:`_plastic_effective_viscosity`). This
        sits naturally on the yield surface during yielding (because
        ``|σ^*_∥|`` saturates near ``τ_y`` while ``|ε̇^*_∥|`` is large)
        and tracks the elastic VE response otherwise. Critically the
        expression depends only on previous-step storage — no
        plasticity-clip recursion through ``E_eff`` — keeping the JIT
        codegen tree shallow.

        The C_∥ sub-modulus still uses the **current** yield-clipped
        ``η_∥_eff`` (preserves Newton's nonlinear plasticity Jacobian
        through the multiplicative response weights).
        """
        params = self.Parameters
        DDt = self.Unknowns.DFDt
        E = self.Unknowns.E

        # Matrix branch: scalar (α_⊥, φ_⊥) from DDt
        alpha_perp = DDt._exp_alpha
        phi_perp = DDt._exp_phi
        eta_perp = params.shear_viscosity_0
        eta_perp_sym = eta_perp.sym

        # Histories
        sigma_star = DDt.psi_star[0].sym
        if DDt.forcing_star is not None:
            edot_star = DDt.forcing_star.sym
        else:
            edot_star = sympy.zeros(*E.shape)

        # ── Lagged η_∥_eff via parent's softmin envelope, evaluated
        # against forcing_star (previous-step ε̇) instead of E_eff
        # (current Newton iterate). Breaks the per-quad-split's
        # 1-iteration trivial-Newton failure mode.
        eta_par_lag_chain = self._eta_par_eff_lagged()
        if hasattr(eta_par_lag_chain, 'sym'):
            eta_par_lagged = eta_par_lag_chain.sym
        else:
            eta_par_lagged = eta_par_lag_chain

        mu_sym = params.shear_modulus.sym
        dt_sym = params.dt_elastic.sym if hasattr(params.dt_elastic, 'sym') else params.dt_elastic
        # x_par = Δt/τ_∥ — natural value (no cap)
        tau_par_natural = eta_par_lagged / mu_sym
        x_par_natural = dt_sym / tau_par_natural
        # Soft cap on x_par: x_eff = (1 - exp(-c·x))/c
        # • x → 0:    x_eff → x          (elastic, no cap)
        # • x → ∞:    x_eff → 1/c        (capped → α_∥ ≥ exp(-1/c))
        # • smooth derivatives everywhere; pre-evaluates to a finite
        #   scalar at codegen-time defaults (dt=∞, μ=∞, η=Pint) where
        #   x_natural = oo, exp(-c·oo) = 0, x_eff = 1/c — avoids the
        #   oo-vs-Pint dimensional clash that breaks sympy.Max/+ caps.
        # Equivalent to capping τ_∥ ≥ c·Δt (recommendation #4).
        if self._tau_par_cap_factor > 0.0:
            c = sympy.Float(self._tau_par_cap_factor)
            x_par = (1 - sympy.exp(-c * x_par_natural)) / c
        else:
            x_par = x_par_natural
        alpha_par = sympy.exp(-x_par)
        phi_par = (1 - alpha_par) / x_par

        # C_∥ uses the natural lagged η (no cap) — preserves the
        # σ_∥ ≈ τ_y enforcement on the yield surface. The cap only
        # tames the elastic-memory factor (α_∥, φ_∥) so the parallel
        # branch retains some spring-back instead of fully releasing
        # in one step. Recovers BDF-style behaviour where boundary
        # motion is partly absorbed by elastic accumulation rather
        # than dumped entirely into slip.
        eta_par_current = eta_par_lagged

        # Build the split sub-moduli with the lagged η_∥
        c_perp, c_par = self._build_split_c_tensors(eta_perp_sym, eta_par_current)

        # E_eff_⊥ = (1-φ_⊥)·ε̇ + α_⊥/(2η_⊥)·σ* + (φ_⊥-α_⊥)·ε̇*
        e_eff_perp = (
            (1 - phi_perp) * E
            + (alpha_perp / (2 * eta_perp)) * sigma_star
            + (phi_perp - alpha_perp) * edot_star
        )
        # E_eff_∥ uses lagged α_∥, φ_∥ but current η_∥_eff in the
        # σ*-projection denominator (so the projected history reads
        # off the same modulus the next step's flux is built on).
        e_eff_par = (
            (1 - phi_par) * E
            + (alpha_par / (2 * eta_par_current)) * sigma_star
            + (phi_par - alpha_par) * edot_star
        )

        def _contract(c, x):
            if len(c.shape) == 2:
                return sympy.Matrix(c * x)
            return sympy.Matrix(sympy.tensorcontraction(
                sympy.tensorcontraction(sympy.tensorproduct(c, x), (1, 5)),
                (0, 3),
            ))

        return _contract(c_perp, e_eff_perp) + _contract(c_par, e_eff_par)


class MultiMaterialConstitutiveModel(Constitutive_Model):
    r"""
    Multi-material constitutive model using level-set weighted flux averaging.

    Mathematical Foundation:

    .. math::

        \mathbf{f}_{\text{composite}}(\mathbf{x}) = \sum_{i=1}^{N}
        \phi_i(\mathbf{x}) \cdot \mathbf{f}_i(\mathbf{x})

    Critical Architecture:

    - Solver owns Unknowns (including :math:`D\mathbf{F}/Dt` stress history)
    - All constituent models share solver's Unknowns
    - Composite flux becomes stress history for all materials
    """

    def __init__(
        self,
        unknowns,
        material_swarmVariable: IndexSwarmVariable,
        constitutive_models: list,
        normalize_levelsets: bool = False,
    ):
        r"""
        Parameters
        ----------
        unknowns : UnknownSet
            The solver's authoritative unknowns (:math:`\mathbf{u}`,
            :math:`D\mathbf{F}/Dt`, :math:`D\mathbf{u}/Dt`).
        material_swarmVariable : IndexSwarmVariable
            Index variable tracking material distribution on particles.
        constitutive_models : list of Constitutive_Model
            Pre-configured constitutive models for each material.
        normalize_levelsets : bool, optional
            Whether to normalize level-set functions to enforce partition of unity.
            Set to True if IndexSwarmVariable does not maintain partition of unity.
            Default: False (assumes IndexSwarmVariable maintains partition of unity)
        """
        # Validate compatibility before initialization
        self._validate_model_compatibility(constitutive_models)

        self._material_var = material_swarmVariable
        self._constitutive_models = constitutive_models
        self._normalize_levelsets = normalize_levelsets

        # Ensure model count matches material indices
        if len(constitutive_models) != material_swarmVariable.indices:
            raise ValueError(
                f"Model count ({len(constitutive_models)}) must match "
                f"material indices ({material_swarmVariable.indices})"
            )

        # CRITICAL: Share solver's unknowns with all constituent models
        self._setup_shared_unknowns(constitutive_models, unknowns)

        # Composite model doesn't have its own material_name - constituents do
        super().__init__(unknowns, material_name=None)

    def _setup_shared_unknowns(self, constitutive_models, unknowns):
        """
        Ensure all constituent models share the solver's authoritative unknowns.
        This is critical for proper stress history management.
        """
        for i, model in enumerate(constitutive_models):
            # Share solver's unknowns - this gives access to composite D(F)/Dt history
            model.Unknowns = unknowns

            # Validation: Ensure sharing worked correctly
            assert model.Unknowns is unknowns, f"Model {i} failed to share unknowns - memory issue?"

            # For elastic models, verify DFDt access
            if hasattr(model, "_stress_star"):
                assert hasattr(
                    unknowns, "DFDt"
                ), f"Model {i} needs stress history but DFDt not available"

    def _validate_model_compatibility(self, models: list) -> bool:
        """
        Ensure all constituent models are compatible for flux averaging.

        Checks:
        - Same u_dim (scalar vs vector problem compatibility)
        - Same spatial dimension (2D/3D consistency)
        - Compatible flux tensor shapes
        - All models properly initialized
        """
        if not models:
            raise ValueError("At least one constitutive model required")

        reference_model = models[0]
        reference_u_dim = reference_model.u_dim
        reference_dim = reference_model.dim

        for i, model in enumerate(models):
            if model.u_dim != reference_u_dim:
                raise ValueError(f"Model {i} has u_dim={model.u_dim}, expected {reference_u_dim}")
            if model.dim != reference_dim:
                raise ValueError(f"Model {i} has dim={model.dim}, expected {reference_dim}")
            # Validate model is properly initialized
            if not hasattr(model, "Unknowns"):
                raise ValueError(f"Model {i} is not properly initialized")

        return True

    @property
    def flux(self):
        r"""
        Compute level-set weighted average of constituent model fluxes.

        CRITICAL: This composite flux becomes the stress history that
        all constituent models (including elastic ones) will read via
        ``DFDt.psi_star[0]`` in the next time step.
        """
        # Get reference flux shape from first model
        reference_flux = self._constitutive_models[0].flux
        combined_flux = sympy.Matrix.zeros(*reference_flux.shape)

        if self._normalize_levelsets:
            # Compute normalization factor to ensure partition of unity
            total_levelset = sum(
                self._material_var.sym[i] for i in range(self._material_var.indices)
            )

            for i in range(self._material_var.indices):
                # Get normalized level-set function for material i
                material_fraction = self._material_var.sym[i] / total_levelset

                # Get flux contribution from constituent model i
                model_flux = self._constitutive_models[i].flux

                # Add weighted contribution to composite flux
                combined_flux += material_fraction * model_flux
        else:
            # Use level-sets directly (assuming they already maintain partition of unity)
            for i in range(self._material_var.indices):
                # Get flux contribution from constituent model i
                model_flux = self._constitutive_models[i].flux

                # Add weighted contribution using level-set directly
                combined_flux += self._material_var.sym[i] * model_flux

        # This combined_flux will become the stress history for ALL materials
        return combined_flux

    @property
    def K(self):
        r"""
        Effective stiffness using level-set weighted harmonic average.

        For composite materials, harmonic averaging gives the correct effective
        stiffness for preconditioning: $1/K_{eff} = \sum(\phi_i / K_i) / \sum(\phi_i)$
        """
        # Harmonic average: 1/K_eff = sum(phi_i / K_i) / sum(phi_i)
        combined_inv_K = sympy.sympify(0)

        if self._normalize_levelsets:
            # Compute normalization factor to ensure partition of unity
            total_levelset = sum(
                self._material_var.sym[i] for i in range(self._material_var.indices)
            )

            for i in range(self._material_var.indices):
                # Get normalized level-set function for material i
                material_fraction = self._material_var.sym[i] / total_levelset

                # Get stiffness from constituent model i
                model_K = self._constitutive_models[i].K

                # Add weighted contribution to inverse stiffness
                combined_inv_K += material_fraction / model_K
        else:
            # Use level-sets directly (assuming they already maintain partition of unity)
            for i in range(self._material_var.indices):
                # Get stiffness from constituent model i
                model_K = self._constitutive_models[i].K

                # Add weighted contribution using level-set directly
                combined_inv_K += self._material_var.sym[i] / model_K

        # Return harmonic average
        return 1 / combined_inv_K

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        display(Markdown(f"**Multi-Material Model**: {len(self._constitutive_models)} materials"))

        for i, model in enumerate(self._constitutive_models):
            display(Markdown(f"**Material {i}**: {type(model).__name__}"))

        if self.flux is not None:
            display(Latex(r"$\mathbf{f}_{\text{composite}} = " + sympy.latex(self.flux) + "$"))
