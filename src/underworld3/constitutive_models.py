## This file has constitutive models that can be plugged into the SNES solvers

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
from underworld3.systems.ddt import Lagrangian as Lagrangian_DDt

from underworld3.function import expression as public_expression

expression = lambda *x, **X: public_expression(*x, _unique_name_generation=True, **X)


# How do we use the default here if input is required ?
def validate_parameters(
    symbol, input, default=None, allow_number=True, allow_expression=True
):

    if allow_number and isinstance(input, (float)):
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
        print(f"An underworld `expression` or `function` is required", flush=True)
        return None

    return input


class Constitutive_Model(uw_object):
    r"""
    Constititutive laws relate gradients in the unknowns to fluxes of quantities
    (for example, heat fluxes are related to temperature gradients through a thermal conductivity)
    The `Constitutive_Model` class is a base class for building `underworld` constitutive laws

    In a scalar problem, the relationship is

    $$
    q_i = k_{ij} \frac{\partial T}{\partial x_j}
    $$

    and the constitutive parameters describe $ k_{ij}$. The template assumes $ k_{ij} = \delta_{ij} $

    In a vector problem (such as the Stokes problem), the relationship is

    $$
    t_{ij} = c_{ijkl} \frac{\partial u_k}{\partial x_l}
    $$

    but is usually written to eliminate the anti-symmetric part of the displacement or velocity gradients (for example):

    $$
    t_{ij} = c_{ijkl} \frac{1}{2} \left[ \frac{\partial u_k}{\partial x_l} + \frac{\partial u_l}{\partial x_k} \right]
    $$

    and the constitutive parameters describe $c_{ijkl}$. The template assumes
    $ k_{ij} = \frac{1}{2} \left( \delta_{ik} \delta_{jl} + \delta_{il} \delta_{jk} \right) $ which is the
    4th rank identity tensor accounting for symmetry in the flux and the gradient terms.
    """

    @timing.routine_timer_decorator
    def __init__(self, unknowns):
        # Define / identify the various properties in the class but leave
        # the implementation to child classes. The constitutive tensor is
        # defined as a template here, but should be instantiated via class
        # properties as required.

        # We provide a function that converts gradients / gradient history terms
        # into the relevant flux term.

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

    class _Parameters:
        """Any material properties that are defined by a constitutive relationship are
        collected in the parameters which can then be defined/accessed by name in
        individual instances of the class.
        """

        def __init__(inner_self, _owning_model):
            inner_self._owning_model = _owning_model
            return

    @property
    def Unknowns(self):
        return self._Unknowns

    # We probably should not be changing this ever ... does this setter even belong here ?
    @Unknowns.setter
    def Unknowns(self, unknowns):
        self._Unknowns = unknowns
        self._solver_is_setup = False
        return

    @property
    def K(self):
        """The constitutive property for this flow law"""
        return self._K

    ## Not sure about setters for these, I suppose it would be a good idea
    @property
    def u(self):
        return self.Unknowns.u

    @property
    def grad_u(self):
        mesh = self.Unknowns.u.mesh
        # return mesh.vector.gradient(self.Unknowns.u.sym)
        return self.Unknowns.u.sym.jacobian(mesh.CoordinateSystem.N)

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
        return

    @property
    def DFDt(self):
        return self._DFDt

    # Do we want to lock this down ?
    @DFDt.setter
    def DFDt(
        self,
        DFDt_value: Union[SemiLagrangian_DDt, Lagrangian_DDt],
    ):
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

        return

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
    ### Viscous Flow Model

    $$\tau_{ij} = \eta_{ijkl} \cdot \frac{1}{2} \left[ \frac{\partial u_k}{\partial x_l} + \frac{\partial u_l}{\partial x_k} \right]$$

    where $\eta$ is the viscosity, a scalar constant, `sympy` function, `underworld` mesh variable or
    any valid combination of those types. This results in an isotropic (but not necessarily homogeneous or linear)
    relationship between $\tau$ and the velocity gradients. You can also supply $\eta_{IJ}$, the Mandel form of the
    constitutive tensor, or $\eta_{ijkl}$, the rank 4 tensor.

    The Mandel constitutive matrix is available in `viscous_model.C` and the rank 4 tensor form is
    in `viscous_model.c`.


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

    def __init__(self, unknowns):
        # All this needs to do is define the
        # viscosity property and init the parent(s)
        # In this case, nothing seems to be needed.
        # The viscosity is completely defined
        # in terms of the Parameters

        super().__init__(unknowns)

        # self._viscosity = expression(
        #     R"{\eta_0}",
        #     1,
        #     " Apparent viscosity",
        # )

    class _Parameters:
        """Any material properties that are defined by a constitutive relationship are
        collected in the parameters which can then be defined/accessed by name in
        individual instances of the class.
        """

        def __init__(
            inner_self,
            _owning_model,
        ):

            inner_self._owning_model = _owning_model
            inner_self._shear_viscosity_0 = expression(r"\eta", 1, "Shear viscosity")

        @property
        def shear_viscosity_0(inner_self):
            return inner_self._shear_viscosity_0

        @shear_viscosity_0.setter
        def shear_viscosity_0(inner_self, value: Union[float, sympy.Function]):

            visc_expr = validate_parameters(
                R"\eta", value, default=None, allow_number=True
            )

            inner_self._shear_viscosity_0.copy(visc_expr)
            del visc_expr

            return

    @property
    def viscosity(self):
        """Whatever the consistutive model defines as the effective value of viscosity
        in the form of an uw.expression"""

        return self.Parameters._shear_viscosity_0

    @property
    def flux(self):
        edot = self.grad_u
        return self._q(edot)

    def _q(self, edot):
        """Computes the effect of the constitutive tensor on the gradients of the unknowns.
        (always uses the `c` form of the tensor). In general cases, the history of the gradients
        may be required to evaluate the flux.
        """

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
        mesh = self.Unknowns.u.mesh

        return mesh.vector.strain_tensor(self.Unknowns.u.sym)

        # ddu = self.Unknowns.u.sym.jacobian(mesh.CoordinateSystem.N)
        # edot = (ddu + ddu.T) / 2
        # return edot

    def _build_c_tensor(self):
        """For this constitutive law, we expect just a viscosity function"""

        if self._is_setup:
            return

        d = self.dim
        viscosity = self.viscosity

        try:
            self._c = 2 * uw.maths.tensor.rank4_identity(d) * viscosity
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

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        ## feedback on this instance
        display(
            Latex(
                r"$\quad\eta_\textrm{eff} = $ "
                + sympy.sympify(self.viscosity.sym)._repr_latex_()
            )
        )


## NOTE - retrofit VEP into here


class ViscoPlasticFlowModel(ViscousFlowModel):
    r"""

    $$\tau_{ij} = \eta_{ijkl} \cdot \frac{1}{2} \left[ \frac{\partial u_k}{\partial x_l} + \frac{\partial u_l}{\partial x_k} \right]$$

    where $\eta$ is the viscosity, a scalar constant, `sympy` function, `underworld` mesh variable or
    any valid combination of those types. This results in an isotropic (but not necessarily homogeneous or linear)
    relationship between $\tau$ and the velocity gradients. You can also supply $\eta_{IJ}$, the Mandel form of the
    constitutive tensor, or $\eta_{ijkl}$, the rank 4 tensor.

    In a viscoplastic model, this viscosity is actually defined to cap the value of the overall stress at a value known as the *yield stress*.
    In this constitutive law, we are assuming that the yield stress is a scalar limit on the 2nd invariant of the stress. A general, anisotropic
    model needs to define the yield surface carefully and only a sub-set of possible cases is available in `Underworld`

    This constitutive model is a convenience function that simplifies the code at run-time but can be reproduced easily by using the appropriate
    `sympy` functions in the standard viscous constitutive model. **If you see `not~yet~defined` in the definition of the effective viscosity, this means
    that you have not yet defined all the required functions. The behaviour is to default to the standard viscous constitutive law if yield terms are
    not specified.

    The Mandel constitutive matrix is available in `viscoplastic_model.C` and the rank 4 tensor form is
    in `viscoplastic_model.c`.  Apply the constitutive model using:

    ---
    """

    def __init__(self, unknowns):
        # All this needs to do is define the
        # non-paramter properties that we want to
        # use in other expressions and init the parent(s)
        #

        super().__init__(unknowns)

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

    class _Parameters:
        """Any material properties that are defined by a constitutive relationship are
        collected in the parameters which can then be defined/accessed by name in
        individual instances of the class.

        `sympy.oo` (infinity) for default values ensures that sympy.Min simplifies away
        the conditionals when they are not required
        """

        def __init__(
            inner_self,
            _owning_model,
        ):
            inner_self._owning_model = _owning_model

            # Default / placeholder values for constitutive parameters

            inner_self._shear_viscosity_0 = expression(
                R"{\eta}",
                1,
                "Shear viscosity",
            )
            inner_self._shear_viscosity_min = expression(
                R"{\eta_{\textrm{min}}}",
                -sympy.oo,
                "Shear viscosity, minimum cutoff",
            )

            inner_self._yield_stress = expression(
                R"{\tau_{y}}",
                sympy.oo,
                "Yield stress (DP)",
            )
            inner_self._yield_stress_min = expression(
                R"{\tau_{y, \mathrm{min}}}",
                -sympy.oo,
                "Yield stress (DP) minimum cutoff ",
            )

            inner_self._strainrate_inv_II_mi = expression(
                R"{\dot\varepsilon_{\mathrm{min}}}",
                0,
                "Strain rate invariant minimum value ",
            )

            return

        #

        @property
        def shear_viscosity_0(inner_self):
            return inner_self._shear_viscosity_0

        @shear_viscosity_0.setter
        def shear_viscosity_0(inner_self, value):
            expr = validate_parameters(R"\eta_0", value, allow_number=True)
            inner_self._shear_viscosity_0.copy(expr)
            inner_self._reset()

            return

        @property
        def shear_viscosity_min(inner_self):
            return inner_self._shear_viscosity_min

        @shear_viscosity_min.setter
        def shear_viscosity_min(inner_self, value):

            expr = validate_parameters(
                R"{\eta_{\textrm{min}}}", value, allow_number=True
            )
            inner_self._shear_viscosity_min.copy(expr)
            inner_self._reset()

            return

        @property
        def yield_stress(inner_self):
            return inner_self._yield_stress

        @yield_stress.setter
        def yield_stress(inner_self, value):

            expr = validate_parameters(R"{\tau_\textrm{y}}", value, allow_number=True)
            inner_self._yield_stress.copy(expr)
            inner_self._reset()

            return

        @property
        def yield_stress_min(inner_self):
            return inner_self._yield_stress_min

        @yield_stress_min.setter
        def yield_stress_min(inner_self, value):
            expr = validate_parameters(
                R"{\tau_\textrm{y, min}}", value, allow_number=True
            )
            inner_self._yield_stress_min.copy(expr)
            inner_self._reset()

            return

        @property
        def strainrate_inv_II_min(inner_self):
            return inner_self._strainrate_inv_II_min

        @strainrate_inv_II_min.setter
        def strainrate_inv_II_min(inner_self, value):
            expr = validate_parameters(
                R"{II(\tau)_{\textrm{min}}}", value, allow_number=True
            )
            inner_self._strainrate_inv_II_min.copy(expr)
            inner_self._reset()

            return

    @property
    def viscosity(self):
        inner_self = self.Parameters
        # detect if values we need are defined or are placeholder symbols

        if inner_self.yield_stress.sym == sympy.oo:
            self._plastic_eff_viscosity.symbol = inner_self._shear_viscosity_0.symbol
            self._plastic_eff_viscosity._sym = inner_self._shear_viscosity_0._sym
            return self._plastic_eff_viscosity

        # Don't put conditional behaviour in the constitutive law
        # when it is not needed

        if inner_self.yield_stress_min.sym != 0:
            yield_stress = sympy.Max(
                inner_self.yield_stress_min, inner_self.yield_stress
            )
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
            self._plastic_eff_viscosity._sym = sympy.simplify(
                sympy.Max(effective_viscosity, inner_self.shear_viscosity_min)
            )

        else:
            self._plastic_eff_viscosity._sym = sympy.simplify(effective_viscosity)

        # Returns an expression that has a different description
        return self._plastic_eff_viscosity

    def plastic_correction(self) -> float:
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

    ### Formulation

    The stress (flux term) is given by

    $$\tau_{ij} = \eta_{ijkl} \cdot \frac{1}{2} \left[ \frac{\partial u_k}{\partial x_l} + \frac{\partial u_l}{\partial x_k} \right]$$

    where $\eta$ is the viscosity, a scalar constant, `sympy` function, `underworld` mesh variable or
    any valid combination of those types. This results in an isotropic (but not necessarily homogeneous or linear)
    relationship between $\tau$ and the velocity gradients. You can also supply $\eta_{IJ}$, the Mandel form of the
    constitutive tensor, or $\eta_{ijkl}$, the rank 4 tensor.

    The Mandel constitutive matrix is available in `viscous_model.C` and the rank 4 tensor form is
    in `viscous_model.c`.  Apply the constitutive model using:

    """

    def __init__(self, unknowns, order=1):

        ## We just need to add the expressions for the stress history terms in here.\
        ## They are properties to hold expressions that are persistent for this instance
        ## (i.e. we only update the value, not the object)

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

        self._reset()

        super().__init__(unknowns)

        return

    class _Parameters:
        """Any material properties that are defined by a constitutive relationship are
        collected in the parameters which can then be defined/accessed by name in
        individual instances of the class.
        """

        def __init__(
            inner_self,
            _owning_model,
        ):
            inner_self._owning_model = _owning_model

            strainrate_inv_II = sympy.symbols(
                r"\left|\dot\epsilon\right|\rightarrow\textrm{not\ defined}"
            )

            stress_star = sympy.symbols(r"\sigma^*\rightarrow\textrm{not\ defined}")

            ## These all need to be expressions that can be replaced (for lazy evaluation)
            ## So do any derived quantities like relaxation time. They will need all
            ## getters that use expression.copy()

            inner_self._stress_star = stress_star
            inner_self._not_yielded = sympy.sympify(1)

            inner_self._shear_viscosity_0 = expression(
                R"{\eta}",
                1,
                "Shear viscosity",
            )

            inner_self._shear_modulus = expression(
                R"{\mu}",
                sympy.oo,
                "Shear modulus",
            )

            inner_self._dt_elastic = expression(
                R"{\Delta t_{e}}",
                sympy.oo,
                "Elastic timestep",
            )

            inner_self._shear_viscosity_min = expression(
                R"{\eta_{\textrm{min}}}",
                -sympy.oo,
                "Shear viscosity, minimum cutoff",
            )

            inner_self._yield_stress = expression(
                R"{\tau_{y}}",
                sympy.oo,
                "Yield stress (DP)",
            )
            inner_self._yield_stress_min = expression(
                R"{\tau_{y, \mathrm{min}}}",
                -sympy.oo,
                "Yield stress (DP) minimum cutoff ",
            )

            inner_self._strainrate_inv_II_min = expression(
                R"{\dot\varepsilon_{II,\mathrm{min}}}",
                0,
                "Strain rate invariant minimum value ",
            )

            ## The following expressions are not pure parameters, but
            ## combinations. We set them up here and they will then
            ## have @property calls to retrieve / calculate them
            ## It is useful to have each as a separate expression
            ## as it is these can be used in many derivations

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

            return

        @property
        def shear_viscosity_0(inner_self):
            return inner_self._shear_viscosity_0

        @shear_viscosity_0.setter
        def shear_viscosity_0(inner_self, value):
            expr = validate_parameters(R"\eta", value, allow_number=True)
            inner_self._shear_viscosity_0.copy(expr)
            inner_self._reset()

            return

        @property
        def shear_modulus(inner_self):
            return inner_self._shear_modulus

        @shear_modulus.setter
        def shear_modulus(inner_self, value):
            expr = validate_parameters(R"\mu", value, allow_number=True)
            inner_self._shear_modulus.copy(expr)
            del expr
            inner_self._reset()

            return

        @property
        def dt_elastic(inner_self):
            return inner_self._dt_elastic

        @dt_elastic.setter
        def dt_elastic(inner_self, value):
            expr = validate_parameters(R"{\Delta t_e}", value, allow_number=True)
            inner_self._dt_elastic.copy(expr)
            inner_self._reset()

            return

        @property
        def shear_viscosity_min(inner_self):
            return inner_self._shear_viscosity_min

        @shear_viscosity_min.setter
        def shear_viscosity_min(inner_self, value):

            expr = validate_parameters(
                R"{\eta_{\textrm{min}}}", value, allow_number=True
            )
            inner_self._shear_viscosity_min.copy(expr)
            inner_self._reset()

            return

        @property
        def yield_stress(inner_self):
            return inner_self._yield_stress

        @yield_stress.setter
        def yield_stress(inner_self, value):

            expr = validate_parameters(R"{\tau_\textrm{y}}", value, allow_number=True)
            inner_self._yield_stress.copy(expr)
            inner_self._reset()

            return

        @property
        def yield_stress_min(inner_self):
            return inner_self._yield_stress_min

        @yield_stress_min.setter
        def yield_stress_min(inner_self, value):
            expr = validate_parameters(
                R"{\tau_{\textrm{y, min}}}", value, allow_number=True
            )
            inner_self._yield_stress_min.copy(expr)
            inner_self._reset()

            return

        @property
        def strainrate_inv_II_min(inner_self):
            return inner_self._strainrate_inv_II_min

        @strainrate_inv_II_min.setter
        def strainrate_inv_II_min(inner_self, value):
            expr = validate_parameters(
                R"{II(\tau)_{\textrm{min}}}", value, allow_number=True
            )
            inner_self._strainrate_inv_II_min.copy(expr)
            inner_self._reset()

            return

        ## Derived parameters of the constitutive model (these have no setters)
        ## Note, do not return new expressions, keep the old objects as containers
        ## the correct values are used in existing expressions. These really are
        ## parameters - they are solely combinations of other parameters.

        @property
        def ve_effective_viscosity(inner_self):
            # the dt_elastic defaults to infinity, t_relax to zero,
            # so this should be well behaved in the viscous limit

            if inner_self.shear_modulus == sympy.oo:
                return inner_self.shear_viscosity_0

            # Note, 1st order only here but we should add higher order versions of this

            # 1st Order version (default)
            if inner_self._owning_model.order != 2:
                el_eff_visc = (
                    inner_self.shear_viscosity_0
                    * inner_self.shear_modulus
                    * inner_self.dt_elastic
                    / (
                        inner_self.shear_viscosity_0
                        + inner_self.dt_elastic * inner_self.shear_modulus
                    )
                )

            # 2nd Order version (need to ask for this one)
            else:
                el_eff_visc = (
                    2
                    * inner_self.shear_viscosity_0
                    * inner_self.shear_modulus
                    * inner_self.dt_elastic
                    / (
                        3 * inner_self.shear_viscosity_0
                        + 2 * inner_self.dt_elastic * inner_self.shear_modulus
                    )
                )

            inner_self._ve_effective_viscosity.sym = el_eff_visc

            return inner_self._ve_effective_viscosity

        @property
        def t_relax(inner_self):
            # shear modulus defaults to infinity so t_relax goes to zero
            # in the viscous limit

            inner_self._t_relax.sym = (
                inner_self.shear_viscosity_0 / inner_self.shear_modulus
            )
            return inner_self._t_relax

    ## End of parameters definition

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order = value
        self._reset()
        return

    # The following should have no setters
    @property
    def stress_star(self):
        if self.Unknowns.DFDt is not None:
            self._stress_star.sym = self.Unknowns.DFDt.psi_star[0].sym

        return self._stress_star

    @property
    def stress_2star(self):
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

        E = self.Unknowns.E

        if self.Unknowns.DFDt is not None:

            if self.is_elastic:
                if self.order != 2:
                    stress_star = self.Unknowns.DFDt.psi_star[0].sym
                    E += stress_star / (
                        2 * self.Parameters.dt_elastic * self.Parameters.shear_modulus
                    )

                else:
                    stress_star = self.Unknowns.DFDt.psi_star[0].sym
                    stress_2star = self.Unknowns.DFDt.psi_star[1].sym
                    E += stress_star / (
                        self.Parameters.dt_elastic * self.Parameters.shear_modulus
                    ) - stress_2star / (
                        4 * self.Parameters.dt_elastic * self.Parameters.shear_modulus
                    )

        self._E_eff.sym = E

        return self._E_eff

    @property
    def E_eff_inv_II(self):

        E_eff = self.E_eff.sym
        self._E_eff_inv_II.sym = sympy.sqrt((E_eff**2).trace() / 2)

        return self._E_eff_inv_II

    @property
    def K(self):
        return self.viscosity

    @property
    def viscosity(self):
        # detect if values we need are defined or are placeholder symbols

        ## Do we want this to be an expression of its own ? If so, define above in __init__() and
        ## make sure it is updated in this call, rather than being replaced.

        inner_self = self.Parameters

        if inner_self.yield_stress.sym == sympy.oo:
            return inner_self.ve_effective_viscosity

        effective_viscosity = inner_self.ve_effective_viscosity

        if self.is_viscoplastic:
            vp_effective_viscosity = self._plastic_effective_viscosity
            effective_viscosity = sympy.Min(effective_viscosity, vp_effective_viscosity)

            ## Why is it p**2 here ?
            # p = self.plastic_correction()
            # effective_viscosity *= 2 * p**2 / (1 + p**2)

            # effective_viscosity *= self.plastic_correction()

        # If we want to apply limits to the viscosity but see caveat above

        if inner_self.shear_viscosity_min.sym != -sympy.oo:
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

        Edot = self.Unknowns.E
        if self.Unknowns.DFDt is not None:

            ## First order ...
            stress_star = self.Unknowns.DFDt.psi_star[0]

            if self.is_elastic:
                Edot += stress_star.sym / (
                    2 * self.Parameters.dt_elastic * self.Parameters.shear_modulus
                )

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
            self._c = 2 * uw.maths.tensor.rank4_identity(d) * viscosity
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

        stress = sympy.simplify(stress)

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

        stress = sympy.simplify(stress)

        return stress

    def stress(self):
        """viscoelastic stress projection (no plastic response)"""

        edot = self.grad_u

        # This is a scalar viscosity ...

        stress = 2 * self.viscosity * edot

        if self.Unknowns.DFDt is not None:

            if self.is_elastic:
                if self.order != 2:
                    stress_star = self.Unknowns.DFDt.psi_star[0].sym
                    stress += (
                        2
                        * self.viscosity
                        * (
                            stress_star
                            / (
                                2
                                * self.Parameters.dt_elastic
                                * self.Parameters.shear_modulus
                            )
                        )
                    )

                else:
                    stress_star = self.Unknowns.DFDt.psi_star[0].sym
                    stress_2star = self.Unknowns.DFDt.psi_star[1].sym

                    stress += (
                        2
                        * self.viscosity
                        * (
                            stress_star
                            / (
                                self.Parameters.dt_elastic
                                * self.Parameters.shear_modulus
                            )
                            - stress_2star
                            / (
                                4
                                * self.Parameters.dt_elastic
                                * self.Parameters.shear_modulus
                            )
                        )
                    )

        stress = sympy.simplify(stress)

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
                r"$\quad\mu = $ "
                + sympy.sympify(self.Parameters.shear_modulus.sym)._repr_latex_(),
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
    def is_elastic(self):
        # If any of these is not defined, elasticity is switched off

        if self.Parameters.dt_elastic.sym is sympy.oo:
            return False

        if self.Parameters.shear_modulus.sym is sympy.oo:
            return False

        return True

    @property
    def is_viscoplastic(self):
        if self.Parameters.yield_stress == sympy.oo:
            return False

        return True


###


class DiffusionModel(Constitutive_Model):
    r"""
    ```python
    class DiffusionModel(Constitutive_Model)
    ...
    ```
    ```python
    diffusion_model = DiffusionModel(dim)
    diffusion_model.material_properties = diffusion_model.Parameters(diffusivity=diffusivity_fn)
    scalar_solver.constititutive_model = diffusion_model
    ```
    $$q_{i} = \kappa_{ij} \cdot \frac{\partial \phi}{\partial x_j}$$

    where $\kappa$ is a diffusivity, a scalar constant, `sympy` function, `underworld` mesh variable or
    any valid combination of those types. Access the constitutive model using:

    ```python
    flux = diffusion_model.flux(gradient_matrix)
    ```
    ---
    """

    class _Parameters:
        """Any material properties that are defined by a constitutive relationship are
        collected in the parameters which can then be defined/accessed by name in
        individual instances of the class.
        """

        def __init__(
            inner_self,
            _owning_model,
        ):

            inner_self._diffusivity = expression(R"\upkappa", 1, "Diffusivity")
            inner_self._owning_model = _owning_model

        @property
        def diffusivity(inner_self):
            return inner_self._diffusivity

        @diffusivity.setter
        def diffusivity(inner_self, value: Union[int, float, sympy.Function]):

            diff = validate_parameters(
                R"{\upkappa}", value, "Diffusivity", allow_number=True
            )

            if diff is not None:
                inner_self._diffusivity.copy(diff)
                inner_self._reset()

            return

    @property
    def K(self):
        return self.Parameters.diffusivity

    @property
    def diffusivity(self):
        return self.Parameters.diffusivity

    def _build_c_tensor(self):
        """For this constitutive law, we expect just a diffusivity function"""

        d = self.dim
        kappa = self.Parameters.diffusivity
        self._c = sympy.Matrix.eye(d) * kappa

        return

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        ## feedback on this instance
        display(
            Latex(
                r"$\quad\kappa = $ "
                + sympy.sympify(self.Parameters.diffusivity)._repr_latex_()
            )
        )

        return


class DarcyFlowModel(Constitutive_Model):
    r"""
    ```python
    class DarcyFlowModel(Constitutive_Model)
    ...
    ```
    ```python
    diffusion_model = DiffusionModel(dim)
    diffusion_model.material_properties = diffusion_model.Parameters(diffusivity=diffusivity_fn)
    scalar_solver.constititutive_model = diffusion_model
    ```
    $$q_{i} = \kappa_{ij} \cdot \left( \frac{\partial \phi}{\partial x_j} - s\right)$$

    where $\kappa$ is the permeability, a scalar constant, `sympy` function, `underworld` mesh variable or
    any valid combination of those types. $s$ is the body force 'source' of pressure gradients.

    Access the constitutive model using:

    ```python
    flux = darcy_flow_model.flux
    ```
    ---
    """

    class _Parameters:
        """Any material properties that are defined by a constitutive relationship are
        collected in the parameters which can then be defined/accessed by name in
        individual instances of the class.
        """

        def __init__(
            inner_self,
            _owning_model,
            permeabililty: Union[float, sympy.Function] = 1,
        ):

            inner_self._s = expression(
                R"{s}",
                sympy.Matrix.zeros(rows=1, cols=_owning_model.dim),
                "Gravitational forcing",
            )

            inner_self._permeability = expression(
                R"{\kappa}",
                1,
                "Permeability",
            )

            inner_self._owning_model = _owning_model

        @property
        def s(inner_self):
            return inner_self._s

        @s.setter
        def s(inner_self, value: sympy.Matrix):
            inner_self._s.sym = value
            inner_self._reset()

        @property
        def permeability(inner_self):
            return inner_self._permeability

        @permeability.setter
        def permeability(inner_self, value: Union[int, float, sympy.Function]):

            perm = validate_parameters(
                R"{\upkappa}", value, "Permeability", allow_number=True
            )

            if perm is not None:
                inner_self._permeability.copy(perm)
                inner_self._reset()

            return

    @property
    def K(self):
        return self.Parameters.permeability

    def _build_c_tensor(self):
        """For this constitutive law, we expect just a diffusivity function"""

        d = self.dim
        kappa = self.Parameters.permeability
        self._c = sympy.Matrix.eye(d) * kappa

        return

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        ## feedback on this instance
        display(
            Latex(
                r"$\quad\kappa = $ "
                + sympy.sympify(self.Parameters.diffusivity)._repr_latex_()
            )
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
    ```python
    class TransverseIsotropicFlowModel(Constitutive_Model)
    ...
    ```
    ```python
    viscous_model = TransverseIsotropicFlowModel(dim)
    viscous_model.material_properties = viscous_model.Parameters(eta_0=viscosity_fn,
                                                                eta_1=weak_viscosity_fn,
                                                                director=orientation_vector_fn)
    solver.constititutive_model = viscous_model
    ```
    $$ \tau_{ij} = \eta_{ijkl} \cdot \frac{1}{2} \left[ \frac{\partial u_k}{\partial x_l} + \frac{\partial u_l}{\partial x_k} \right] $$

    where $\eta$ is the viscosity tensor defined as:

    $$ \eta_{ijkl} = \eta_0 \cdot I_{ijkl} + (\eta_0-\eta_1) \left[ \frac{1}{2} \left[
        n_i n_l \delta_{jk} + n_j n_k \delta_{il} + n_i n_l \delta_{jk} + n_j n_l \delta_{ik} \right] - 2 n_i n_j n_k n_l \right] $$

    and $ \hat{\mathbf{n}} \equiv \left\{ n_i \right\} $ is the unit vector
    defining the local orientation of the weak plane (a.k.a. the director).

    The Mandel constitutive matrix is available in `viscous_model.C` and the rank 4 tensor form is
    in `viscous_model.c`.  Apply the constitutive model using:

    ```python
    tau = viscous_model.flux(gradient_matrix)
    ```
    ---
    """

    def __init__(self, unknowns):
        # All this needs to do is define the
        # viscosity property and init the parent(s)
        # In this case, nothing seems to be needed.
        # The viscosity is completely defined
        # in terms of the Parameters

        super().__init__(unknowns)

        # self._viscosity = expression(
        #     R"{\eta_0}",
        #     1,
        #     " Apparent viscosity",
        # )

    class _Parameters:
        """Any material properties that are defined by a constitutive relationship are
        collected in the parameters which can then be defined/accessed by name in
        individual instances of the class.
        """

        def __init__(
            inner_self,
            _owning_model,
        ):
            inner_self._owning_model = _owning_model

            inner_self._eta_0 = expression(r"\eta_0", 1, "Shear viscosity")
            inner_self._eta_1 = expression(r"\eta_1", 1, "Second viscosity")
            inner_self._director = expression(r"\hat{n}", 1, "Director orientation")

        ## Note the inefficiency below if we change all these values one after the other

        @property
        def eta_0(inner_self):
            return inner_self._eta_0

        @eta_0.setter
        def eta_0(
            inner_self,
            value: Union[float, sympy.Function],
        ):
            visc_expr = validate_parameters(
                R"\eta_0", value, default=None, allow_number=True
            )

            inner_self._eta_0.copy(visc_expr)
            del visc_expr
            inner_self._reset()

        @property
        def eta_1(inner_self):
            return inner_self._eta_1

        @eta_1.setter
        def eta_1(
            inner_self,
            value: Union[float, sympy.Function],
        ):
            visc_expr = validate_parameters(
                R"\eta_1", value, default=None, allow_number=True
            )

            inner_self._eta_1.copy(visc_expr)
            del visc_expr
            inner_self._reset()

        @property
        def director(inner_self):
            return inner_self._director

        @director.setter
        def director(
            inner_self,
            value: Union[sympy.Matrix, sympy.Function, expression],
        ):

            inner_self._director._sym = value
            inner_self._reset()

    ## End of parameters

    @property
    def viscosity(self):
        """Whatever the consistutive model defines as the effective value of viscosity
        in the form of an uw.expression"""

        return self.Parameters._eta_0

    @property
    def K(self):
        """Whatever the consistutive model defines as the effective value of viscosity
        in the form of an uw.expression"""

        return self.Parameters._eta_0

    @property
    def grad_u(self):
        mesh = self.Unknowns.u.mesh

        return mesh.vector.strain_tensor(self.Unknowns.u.sym)

    def _build_c_tensor(self):
        """For this constitutive law, we expect two viscosity functions
        and a sympy row-matrix that describes the director components n_{i}"""

        if self._is_setup:
            return

        d = self.dim
        dv = uw.maths.tensor.idxmap[d][0]

        eta_0 = self.Parameters.eta_0
        eta_1 = self.Parameters.eta_1
        n = self.Parameters.director.sym

        Delta = eta_0 - eta_1
        lambda_mat = 2 * uw.maths.tensor.rank4_identity(d) * eta_0

        for i in range(d):
            for j in range(d):
                for k in range(d):
                    for l in range(d):
                        lambda_mat[i, j, k, l] -= (
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

        lambda_mat = sympy.simplify(uw.maths.tensor.rank4_to_mandel(lambda_mat, d))

        self._c = uw.maths.tensor.mandel_to_rank4(lambda_mat, d)

        self._is_setup = True
        self._solver_is_setup = False

        return

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        ## feedback on this instance
        display(
            Latex(
                r"$\quad\eta_0 = $ "
                + sympy.sympify(self.Parameters.eta_0)._repr_latex_()
            )
        )
        display(
            Latex(
                r"$\quad\eta_1 = $ "
                + sympy.sympify(self.Parameters.eta_1)._repr_latex_()
            )
        )
        display(
            Latex(
                r"$\quad\hat{\mathbf{n}} = $ "
                + sympy.sympify(self.Parameters.director.T)._repr_latex_()
            )
        )


class MultiMaterial_ViscoElasticPlastic(Constitutive_Model):
    r"""
    Manage multiple materials in a constitutive framework.

    Bundles multiple materials into a single consitutive law. The expectation
    is that these all have compatible flux terms.
    """

    def __init__(
        self,
        material_swarmVariable: Optional[IndexSwarmVariable] = None,
        constitutive_models: Optional[list] = [],
    ):
        self._constitutive_models = constitutive_models
        self._material_var = material_swarmVariable

        return

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        ## feedback on this instance

        ## IndexVariables etc

    @property
    def flux(
        self,
    ):
        combined_flux = sympy.sympify(0)

        for i in range(self._material_var.indices):
            M = self._material_var[i]
            combined_flux += self._constitutive_models[i].flux(ddu, ddu_dt, u, u_dt) * M

        return
