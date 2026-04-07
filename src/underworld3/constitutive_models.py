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

        self.Unknowns = unknowns

        u = self.Unknowns.u
        self._DFDt = self.Unknowns.DFDt
        self._DuDt = self.Unknowns.DuDt

        self.dim = u.mesh.dim
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

    def _reset(self):
        """Flags that the expressions in the consitutive tensor need to be refreshed and also that the
        solver will need to rebuild the stiffness matrix and jacobians"""

        self._solver_is_setup = False
        self._is_setup = False

        # Propagate is_setup flag to solver if we have a reference
        if hasattr(self, "Parameters") and hasattr(self.Parameters, "_solver"):
            if self.Parameters._solver is not None:
                self.Parameters._solver.is_setup = False

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
            lambda inner_self: -sympy.oo,
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

        if inner_self.yield_stress_min.sym != 0:
            yield_stress = sympy.Max(inner_self.yield_stress_min, inner_self.yield_stress)
        else:
            yield_stress = inner_self.yield_stress

        viscosity_yield = yield_stress / (2 * self._strainrate_inv_II)

        ## Question is, will sympy reliably differentiate something
        ## with so many Max / Min statements. The smooth version would
        ## be a reasonable alternative:

        # effective_viscosity = sympy.sympify(
        #     1 / (1 / inner_self.shear_viscosity_0 + 1 / viscosity_yield),
        # )

        effective_viscosity = sympy.Min(inner_self.shear_viscosity_0, viscosity_yield)

        # If we want to apply limits to the viscosity but see caveat above
        # Keep this as an sub-expression for clarity

        if inner_self.shear_viscosity_min.sym != -sympy.oo:
            self._plastic_eff_viscosity._sym = sympy.Max(
                effective_viscosity, inner_self.shear_viscosity_min
            )

        else:
            self._plastic_eff_viscosity._sym = effective_viscosity

        # Returns an expression that has a different description
        return self._plastic_eff_viscosity

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

    def __init__(self, unknowns, order=1, material_name: str = None):

        ## We just need to add the expressions for the stress history terms in here.\
        ## They are properties to hold expressions that are persistent for this instance
        ## (i.e. we only update the value, not the object)

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
        self._yield_mode = "smooth"  # "min", "harmonic", "smooth", or "softmin"
        self._yield_softness = 0.5  # δ parameter for "softmin" mode
        self._bdf_blend = None  # auto: 1.0 for VE, 0.75 for VEP

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
            lambda inner_self: -sympy.oo,
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

            # Blend with O1 coefficients for stability
            # 0 = pure O1, 0.5 = balanced (default), 1 = pure requested order
            alpha = self.bdf_blend  # property resolves None → auto-detect
            if 0 < alpha < 1 and order >= 2:
                coeffs_o1 = _bdf_coefficients(1, dt_current, dt_history)
                while len(coeffs_o1) < len(coeffs):
                    coeffs_o1.append(sympy.Integer(0))
                coeffs = [
                    (1 - alpha) * c1 + alpha * ck
                    for c1, ck in zip(coeffs_o1, coeffs)
                ]
        else:
            coeffs = _bdf_coefficients(order, None, [])

        # Pad to length 4
        while len(coeffs) < 4:
            coeffs.append(sympy.Integer(0))

        self._bdf_c0.sym = coeffs[0]
        self._bdf_c1.sym = coeffs[1]
        self._bdf_c2.sym = coeffs[2]
        self._bdf_c3.sym = coeffs[3]

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
        r"""Effective strain rate including elastic contribution.

        .. math::
            \dot{\varepsilon}_{\mathrm{eff}} = \dot{\varepsilon} + \frac{\boldsymbol{\sigma}^*}{2 G \Delta t}
        """
        E = self.Unknowns.E

        if self.Unknowns.DFDt is not None:

            if self.is_elastic:
                mu_dt = self.Parameters.dt_elastic * self.Parameters.shear_modulus
                # BDF history coefficients as UWexpressions (route through constants[])
                bdf_cs = [self._bdf_c1, self._bdf_c2, self._bdf_c3]

                # History contribution: -Σ cᵢ·σ_star[i-1] / (2·μ·dt)
                for i in range(self.Unknowns.DFDt.order):
                    E += -bdf_cs[i] * self.Unknowns.DFDt.psi_star[i].sym / (2 * mu_dt)

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
        """

        inner_self = self.Parameters

        if inner_self.yield_stress.sym == sympy.oo:
            return inner_self.ve_effective_viscosity

        effective_viscosity = inner_self.ve_effective_viscosity

        if self.is_viscoplastic:
            vp_effective_viscosity = self._plastic_effective_viscosity
            if self._yield_mode == "harmonic":
                effective_viscosity = 1 / (1 / effective_viscosity + 1 / vp_effective_viscosity)
            elif self._yield_mode == "smooth":
                # Corrected harmonic: cancels the excess 1/η_ve contribution
                # at deep yielding while staying smooth everywhere.
                #   η_eff = η_ve · (1+f) / (1 + f + f²)
                # where f = η_ve/η_pl measures yield overshoot.
                #
                # f → 0 (elastic): η_eff → η_ve   (no correction)
                # f → ∞ (yielding): η_eff → η_pl  (exact yield)
                # No Min/Max — just arithmetic. Continuous derivatives.
                f = effective_viscosity / vp_effective_viscosity
                effective_viscosity = effective_viscosity * (1 + f) / (1 + f + f**2)
            elif self._yield_mode == "softmin":
                # Smooth approximation to Min(η_ve, η_pl):
                #   η_eff = η_ve / g(f)
                #   g(f) = (1+f)/2 + √((f-1)² + δ²)/2  ≈ max(1, f)
                # where f = η_ve/η_pl and δ = yield_softness.
                # Approaches exact Min as δ→0. No Min/Max in expression.
                delta = self._yield_softness
                f = effective_viscosity / vp_effective_viscosity
                g = (1 + f) / 2 + sympy.sqrt((f - 1)**2 + delta**2) / 2
                effective_viscosity = effective_viscosity / g
            else:
                effective_viscosity = sympy.Min(effective_viscosity, vp_effective_viscosity)

        # Apply viscosity floor — but skip for smooth/harmonic yield modes
        # where the outer Max creates a nested Min/Max that breaks the
        # BDF-2 Jacobian. Those modes are already smooth and bounded.

        if inner_self.shear_viscosity_min.sym != -sympy.oo:
            if self.is_viscoplastic and self._yield_mode in ("harmonic", "smooth", "softmin"):
                return effective_viscosity
            else:
                return sympy.Max(
                    effective_viscosity,
                    inner_self.shear_viscosity_min,
                )

        else:
            return effective_viscosity

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

        if parameters.yield_stress_min.sym != 0:
            yield_stress = sympy.Max(
                parameters.yield_stress_min, parameters.yield_stress
            )  # .rewrite(sympy.Piecewise)
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
        """Viscoelastic(-plastic) deviatoric stress for the weak form."""

        edot = self.grad_u

        stress = 2 * self.viscosity * edot

        if self.Unknowns.DFDt is not None:

            if self.is_elastic:
                mu_dt = self.Parameters.dt_elastic * self.Parameters.shear_modulus
                bdf_cs = [self._bdf_c1, self._bdf_c2, self._bdf_c3]

                for i in range(self.Unknowns.DFDt.order):
                    stress += 2 * self.viscosity * (
                        -bdf_cs[i] * self.Unknowns.DFDt.psi_star[i].sym / (2 * mu_dt)
                    )

        return stress

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

        ``"smooth"`` (default): corrected harmonic —
            ``η_ve · (1+f) / (1+f+f²)`` where ``f = η_ve/η_pl``.
            Smooth, no Min/Max. Best balance of accuracy and robustness.
        ``"softmin"``: smooth approximation to Min —
            ``η_ve / g(f)`` where ``g(f) ≈ max(1, f)`` with smoothing
            parameter δ (``yield_softness``, default 0.5).
            Closer to exact yield than ``"smooth"`` but less robust.
        ``"harmonic"``: parallel blending — ``1/(1/η_ve + 1/η_pl)``.
            Smooth but undershoots τ_y for soft materials.
        ``"min"``: sharp cutoff — ``Min(η_ve, η_pl)``.
            Exact yield but can cause SNES divergence with BDF-2.
        """
        return self._yield_mode

    @yield_mode.setter
    def yield_mode(self, value):
        if value not in ("min", "harmonic", "smooth", "softmin"):
            raise ValueError(f"yield_mode must be 'min', 'harmonic', 'smooth', or 'softmin', got '{value}'")
        self._yield_mode = value
        self._reset()

    @property
    def yield_softness(self):
        r"""Regularisation parameter δ for ``"softmin"`` yield mode.

        Controls how closely the soft minimum approximates the sharp Min.
        Smaller values → sharper yield (closer to Min, less robust).
        Larger values → smoother transition (more robust, lower stress).

        Default 0.5. Only used when ``yield_mode == "softmin"``.
        """
        return self._yield_softness

    @yield_softness.setter
    def yield_softness(self, value):
        self._yield_softness = value
        self._reset()

    @property
    def bdf_blend(self):
        r"""Blending parameter α for BDF history coefficients.

        Blends O1 and O2 BDF coefficients: ``c = (1-α)·c_O1 + α·c_O2``.

        - ``α = 0``: pure BDF-1 (most stable, first-order accurate)
        - ``α = 0.75``: default for VEP (stable, near-optimal accuracy)
        - ``α = 1``: pure BDF-2 (default for pure VE, second-order accurate)
        - ``None`` (default): auto-detect — 1.0 for VE, 0.75 for VEP
        """
        if self._bdf_blend is None:
            return 0.75 if self.is_viscoplastic else 1.0
        return self._bdf_blend

    @bdf_blend.setter
    def bdf_blend(self, value):
        self._bdf_blend = value

    @property
    def requires_stress_history(self):
        """VEP models always require stress history tracking."""
        return True

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

    Relates the Darcy flux to pressure gradients and body forces:

    .. math::

        q_{i} = \kappa_{ij} \left( \frac{\partial p}{\partial x_j} - s_j \right)

    where :math:`\kappa` is the permeability (or hydraulic conductivity),
    :math:`p` is the pressure (or hydraulic head), and :math:`s` is the
    body force term (e.g., gravity: :math:`s = \rho g`).

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
            lambda inner_self: 1,
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
