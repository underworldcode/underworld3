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
from underworld3.utilities._api_tools import uw_object
from underworld3.systems import SNES_Scalar, SNES_Vector, SNES_Stokes_SaddlePt
from underworld3.swarm import IndexSwarmVariable
from underworld3.discretisation import MeshVariable


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
    def __init__(
        self,
        u: MeshVariable,
        flux_dt: uw.swarm.Lagrangian_Updater = None,
    ):
        # Define / identify the various properties in the class but leave
        # the implementation to child classes. The constitutive tensor is
        # defined as a template here, but should be instantiated via class
        # properties as required.

        # We provide a function that converts gradients / gradient history terms
        # into the relevant flux term.

        self._u = u
        self.dim = u.mesh.dim
        self.u_dim = u.num_components
        self._flux_dt = flux_dt

        self.Parameters = self._Parameters(self)
        self.Parameters._solver = None
        self.Parameters._reset = self._reset
        self._is_setup = False
        self._solver_is_setup = False

        self._material_properties = None

        ## Default consitutive tensor is the identity

        if self.u_dim == 1:
            self._c = sympy.Matrix.eye(self.dim)
        else:  # vector problem
            self._c = uw.maths.tensor.rank4_identity(self.dim)

        self._K = sympy.sympify(1)
        self._C = None

        super().__init__()

    class _Parameters:
        """Any material properties that are defined by a constitutive relationship are
        collected in the parameters which can then be defined/accessed by name in
        individual instances of the class.
        """

        def __init__(inner_self, owning_model):
            # inner_self._solver = None  #### should this be here?

            inner_self.owning_model = owning_model
            return

    @property
    def K(self):
        """The constitutive property for this flow law"""
        return self._K

    ## Not sure about setters for these, I suppose it would be a good idea
    @property
    def u(self):
        return self._u

    @property
    def grad_u(self):
        mesh = self._u.mesh

        return self._u.sym.jacobian(mesh.CoordinateSystem.N)

    @property
    def flux_dt(self):
        return self._flux_dt

    ## This breaks the solver, but does it break the constitutive term ?
    @flux_dt.setter
    def flux_dt(self, flux_dt_value: uw.swarm.Lagrangian_Updater):
        self._flux_dt = flux_dt_value
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
                sympy.tensorcontraction(sympy.tensorproduct(c, ddu), (3, 5)), (0, 1)
            )

        return sympy.Matrix(flux)

    @property
    def flux_1d(self):
        """Computes the effect of the constitutive tensor on the gradients of the unknowns.
        (always uses the `c` form of the tensor). In general cases, the history of the gradients
        may be required to evaluate the flux. Returns the Voigt form that is flattened so as to
        match the PETSc field storage pattern for symmetric tensors.
        """

        flux = self.flux()

        assert (
            flux.is_symmetric()
        ), "The conversion to Voigt form is only defined for symmetric tensors in underworld\
            but for non-symmetric tensors, the .flatten() method is a potential replacement"

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
    ```python
    class ViscousFlowModel(Constitutive_Model)
    ...
    ```
    ### Example

    ```python
    viscous_model = ViscousFlowModel(dim)
    viscous_model.material_properties = viscous_model.Parameters(viscosity=viscosity_fn)
    solver.constititutive_model = viscous_model
    ```
    ### Formulation

    $$
    \tau_{ij} = \eta_{ijkl} \cdot \frac{1}{2} \left[ \frac{\partial u_k}{\partial x_l} + \frac{\partial u_l}{\partial x_k} \right]
    $$

    where $ \eta $ is the viscosity, a scalar constant, `sympy` function, `underworld` mesh variable or
    any valid combination of those types. This results in an isotropic (but not necessarily homogeneous or linear)
    relationship between $\tau$ and the velocity gradients. You can also supply $\eta_{IJ}$, the Mandel form of the
    constitutive tensor, or $\eta_{ijkl}$, the rank 4 tensor.

    The Mandel constitutive matrix is available in `viscous_model.C` and the rank 4 tensor form is
    in `viscous_model.c`.  Apply the constitutive model using:

    ```python
    tau = viscous_model.flux(gradient_matrix)
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
            owning_model,
            viscosity: Union[float, sympy.Function] = None,
        ):
            if viscosity is None:
                viscosity = sympy.sympify(1)

            inner_self.owning_model = owning_model

            inner_self._shear_viscosity_0 = sympy.sympify(viscosity)

        @property
        def shear_viscosity_0(inner_self):
            return inner_self._shear_viscosity_0

        @shear_viscosity_0.setter
        def shear_viscosity_0(inner_self, value: Union[float, sympy.Function]):
            inner_self._shear_viscosity_0 = value
            inner_self._reset()

    @property
    def viscosity(self):
        return self.Parameters.shear_viscosity_0

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
                sympy.tensorcontraction(sympy.tensorproduct(c, edot), (3, 5)), (0, 1)
            )

        return sympy.Matrix(flux)

    ## redefine the gradient for the viscous law as it relates to
    ## the symmetric part of the tensor only

    @property
    def grad_u(self):
        mesh = self._u.mesh
        ddu = self._u.sym.jacobian(mesh.CoordinateSystem.N)
        edot = (ddu + ddu.T) / 2

        return edot

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
                r"$\quad\eta = $ "
                + sympy.sympify(self.Parameters.shear_viscosity_0)._repr_latex_()
            )
        )


## NOTE - retrofit VEP into here


class ViscoPlasticFlowModel(ViscousFlowModel):
    r"""
    ```python
    class ViscoPlasticFlowModel(Constitutive_Model)
    ...
    ```
    ```python
    viscoplastic_model = ViscoPlasticFlowModel(dim)
    viscoplastic_model.Parameters(
            viscosity: Union[float, sympy.Function] = sympy.sympify(1),
            yield_stress: Union[float, sympy.Function] = None,
            min_viscosity: Union[float, sympy.Function] = sympy.oo,
            yield_stress_min: Union[float, sympy.Function] = sympy.oo,
            edot_II_fn: sympy.Function = None,
            epsilon_edot_II: float = None,
        )
    solver.constititutive_model = viscoplastic_model
    ```
    $$
    \tau_{ij} = \eta_{ijkl} \cdot \frac{1}{2} \left[ \frac{\partial u_k}{\partial x_l} + \frac{\partial u_l}{\partial x_k} \right]
    $$

    where $ \eta $ is the viscosity, a scalar constant, `sympy` function, `underworld` mesh variable or
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

    ```python
    tau = viscoplastic_model.flux(gradient_matrix)
    ```
    ---
    """

    class _Parameters:
        """Any material properties that are defined by a constitutive relationship are
        collected in the parameters which can then be defined/accessed by name in
        individual instances of the class.

        `sympy.oo` (infinity) for default values ensures that sympy.Min simplifies away
        the conditionals when they are not required
        """

        def __init__(
            inner_self,
            owning_model,
            shear_viscosity_0: Union[float, sympy.Function] = 1,
            shear_viscosity_min: Union[float, sympy.Function] = -sympy.oo,
            yield_stress: Union[float, sympy.Function] = sympy.oo,
            yield_stress_min: Union[float, sympy.Function] = -sympy.oo,
            strainrate_inv_II_min: float = 0.0,
        ):
            inner_self.owning_model = owning_model

            inner_self._shear_viscosity_0 = sympy.sympify(shear_viscosity_0)
            inner_self._yield_stress = sympy.sympify(yield_stress)
            inner_self._yield_stress_min = sympy.sympify(yield_stress_min)
            inner_self._shear_viscosity_min = sympy.sympify(shear_viscosity_min)
            inner_self._strainrate_inv_II_min = sympy.sympify(strainrate_inv_II_min)

            return

        @property
        def shear_viscosity_0(inner_self):
            return inner_self._shear_viscosity_0

        @shear_viscosity_0.setter
        def shear_viscosity_0(inner_self, value: Union[float, sympy.Function]):
            inner_self._shear_viscosity_0 = value
            inner_self._reset()

        @property
        def shear_viscosity_min(inner_self):
            return inner_self._shear_viscosity_min

        @shear_viscosity_min.setter
        def shear_viscosity_min(inner_self, value: Union[float, sympy.Function]):
            inner_self._shear_viscosity_min = value
            inner_self._reset()

        @property
        def yield_stress(inner_self):
            return inner_self._yield_stress

        @yield_stress.setter
        def yield_stress(inner_self, value: Union[float, sympy.Function]):
            inner_self._yield_stress = value
            inner_self._reset()

        @property
        def yield_stress_min(inner_self):
            return inner_self._yield_stress_min

        @yield_stress_min.setter
        def yield_stress_min(inner_self, value: Union[float, sympy.Function]):
            inner_self._yield_stress_min = value
            inner_self._reset()

        @property
        def strainrate_inv_II_min(inner_self):
            return inner_self._epsilon_edot_II

        @strainrate_inv_II_min.setter
        def strainrate_inv_II_min(inner_self, value: float):
            inner_self._epsilon_edot_II = sympy.sympify(value)
            inner_self._reset()

    @property
    def viscosity(self):
        # detect if values we need are defined or are placeholder symbols

        inner_self = self.Parameters

        if inner_self.yield_stress == sympy.oo:
            return inner_self.shear_viscosity_0

        if self.is_viscoplastic:
            ## Why is it p**2 here ?
            p = self.plastic_correction()
            effective_viscosity *= 2 * p**2 / (1 + p**2)

            # effective_viscosity *= sympy.Min(1, self.plastic_correction())

        # If we want to apply limits to the viscosity but see caveat above

        if inner_self.shear_viscosity_min is not None:
            return sympy.Max(
                effective_viscosity,
                inner_self.shear_viscosity_min,
            )  # .rewrite(sympy.Piecewise)

        else:
            return effective_viscosity

    @property
    def viscosity(self):
        inner_self = self.Parameters
        # detect if values we need are defined or are placeholder symbols

        Edot = self.grad_u
        strainrate_inv_II = sympy.sqrt((Edot**2).trace() / 2)

        if isinstance(inner_self.yield_stress, sympy.core.symbol.Symbol):
            return inner_self._shear_viscosity_0

        if isinstance(inner_self.edot_II_fn, sympy.core.symbol.Symbol):
            return inner_self._shear_viscosity_0

        # Don't put conditional behaviour in the constitutive law
        # where it is not needed

        if inner_self.yield_stress_min is not None:
            yield_stress = sympy.Max(
                inner_self.yield_stress_min, inner_self.yield_stress
            )
        else:
            yield_stress = inner_self.yield_stress

        viscosity_yield = yield_stress / (2.0 * strainrate_inv_II)

        ## Question is, will sympy reliably differentiate something
        ## with so many Max / Min statements. The smooth version would
        ## be a reasonable alternative:

        # effective_viscosity = sympy.sympify(
        #     1 / (1 / inner_self.bg_viscosity + 1 / viscosity_yield),
        # )

        effective_viscosity = sympy.Min(inner_self._shear_viscosity_0, viscosity_yield)

        # If we want to apply limits to the viscosity but see caveat above

        if inner_self.min_viscosity is not None:
            return sympy.simplify(
                sympy.Max(
                    effective_viscosity,
                    inner_self.min_viscosity,
                )
            )

        else:
            return sympy.simplify(effective_viscosity)

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        ## feedback on this instance
        display(
            Latex(
                r"$\quad\eta_\textrm{0} = $ "
                + sympy.sympify(self.Parameters.shear_viscosity_0)._repr_latex_()
            ),
            Latex(
                r"$\quad\tau_\textrm{y} = $ "
                + sympy.sympify(self.Parameters.yield_stress)._repr_latex_(),
            ),
            Latex(
                r"$\quad|\dot\epsilon| = $ "
                + sympy.sympify(self.Parameters.strainrate_inv_II)._repr_latex_(),
            ),
        )

        return


class ViscoElasticPlasticFlowModel(ViscousFlowModel):
    r"""
    ```python
    class ViscoElasticFlowModel(Constitutive_Model)
    ...
    ```

    ### Example

    ```python
    viscoelastic_model = ViscoElasticPlasticFlowModel(u=velocity_variable)
    viscoelastic_model.Parameters(
            shear_viscosity_0: Union[float, sympy.Function] = 1,
            shear_viscosity_min: Union[float, sympy.Function] = sympy.oo,
            shear_modulus: Union[float, sympy.Function] = sympy.oo,
            dt_elastic: Union[float, sympy.Function] = sympy.oo,
            yield_stress: Union[float, sympy.Function] = sympy.oo,
            yield_stress_min: Union[float, sympy.Function] = sympy.oo,
            strainrate_inv_II: sympy.Function = None,
            stress_star: sympy.Function = None,
            strainrate_inv_II_min: float = 0
        )
    solver.constititutive_model = viscoelastic_model
    ```

    ### Formulation

    The stress (flux term) is given by

    $$\tau_{ij} = \eta_{ijkl} \cdot \frac{1}{2} \left[ \frac{\partial u_k}{\partial x_l} + \frac{\partial u_l}{\partial x_k} \right]$$

    where $\eta$ is the viscosity, a scalar constant, `sympy` function, `underworld` mesh variable or
    any valid combination of those types. This results in an isotropic (but not necessarily homogeneous or linear)
    relationship between $\tau$ and the velocity gradients. You can also supply $\eta_{IJ}$, the Mandel form of the
    constitutive tensor, or $\eta_{ijkl}$, the rank 4 tensor.

    The Mandel constitutive matrix is available in `viscous_model.C` and the rank 4 tensor form is
    in `viscous_model.c`.  Apply the constitutive model using:

    ```python
    tau = viscous_model.flux()
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
            owning_model,
            shear_viscosity_0: Union[float, sympy.Function] = 1,
            shear_viscosity_min: Union[float, sympy.Function] = -sympy.oo,
            shear_modulus: Union[float, sympy.Function] = sympy.oo,
            dt_elastic: Union[float, sympy.Function] = sympy.oo,
            yield_stress: Union[float, sympy.Function] = sympy.oo,
            yield_stress_min: Union[float, sympy.Function] = -sympy.oo,
            stress_star: sympy.Function = None,
            strainrate_inv_II_min: float = 0.0,
        ):
            inner_self.owning_model = owning_model

            if strainrate_inv_II_min is None:
                strainrate_inv_II = sympy.symbols(
                    r"\left|\dot\epsilon\right|\rightarrow\textrm{not\ defined}"
                )

            if stress_star is None:
                stress_star = sympy.symbols(r"\sigma^*\rightarrow\textrm{not\ defined}")

            inner_self._shear_viscosity_0 = sympy.sympify(shear_viscosity_0)
            inner_self._shear_modulus = sympy.sympify(shear_modulus)
            inner_self._dt_elastic = sympy.sympify(dt_elastic)
            inner_self._yield_stress = sympy.sympify(yield_stress)
            inner_self._yield_stress_min = sympy.sympify(yield_stress_min)
            inner_self._shear_viscosity_min = sympy.sympify(shear_viscosity_min)
            inner_self._stress_star = sympy.sympify(stress_star)
            inner_self._strainrate_inv_II_min = sympy.sympify(strainrate_inv_II_min)
            inner_self._not_yielded = sympy.sympify(1)

            return

        @property
        def shear_viscosity_0(inner_self):
            return inner_self._shear_viscosity_0

        @shear_viscosity_0.setter
        def shear_viscosity_0(inner_self, value: Union[float, sympy.Function]):
            inner_self._shear_viscosity_0 = value
            inner_self._reset()

        @property
        def shear_modulus(inner_self):
            return inner_self._shear_modulus

        @shear_modulus.setter
        def shear_modulus(inner_self, value: Union[float, sympy.Function]):
            inner_self._shear_modulus = value
            inner_self._reset()

        @property
        def dt_elastic(inner_self):
            return inner_self._dt_elastic

        @dt_elastic.setter
        def dt_elastic(inner_self, value: Union[float, sympy.Function]):
            inner_self._dt_elastic = value
            inner_self._reset()

        @property
        def ve_effective_viscosity(inner_self):
            # the dt_elastic defaults to infinity, t_relax to zero,
            # so this should be well behaved in the viscous limit

            el_eff_visc = inner_self.shear_viscosity_0 / (
                1
                + inner_self.shear_viscosity_0
                / (inner_self.dt_elastic * inner_self.shear_modulus)
            )

            return sympy.simplify(el_eff_visc)

        @property
        def t_relax(inner_self):
            # shear modulus defaults to infinity so t_relax goes to zero
            # in the viscous limit
            return inner_self.shear_viscosity_0 / inner_self.shear_modulus

        @property
        def shear_viscosity_min(inner_self):
            return inner_self._shear_viscosity_min

        @shear_viscosity_min.setter
        def shear_viscosity_min(inner_self, value: Union[float, sympy.Function]):
            inner_self._shear_viscosity_min = value
            inner_self._reset()

        @property
        def yield_stress(inner_self):
            return inner_self._yield_stress

        @yield_stress.setter
        def yield_stress(inner_self, value: Union[float, sympy.Function]):
            inner_self._yield_stress = value
            inner_self._reset()

        @property
        def yield_stress_min(inner_self):
            return inner_self._yield_stress_min

        @yield_stress_min.setter
        def yield_stress_min(inner_self, value: Union[float, sympy.Function]):
            inner_self._yield_stress_min = value
            inner_self._reset()

        # # This one should only be set internally
        # @property
        # def strainrate_inv_II(inner_self):
        #     return inner_self._strainrate_inv_II

        # @strainrate_inv_II.setter
        # def strainrate_inv_II(inner_self, value: sympy.Function):
        #     inner_self._strainrate_inv_II = value
        #     inner_self._reset()

        @property
        def stress_star(inner_self):
            return inner_self._stress_star

        @stress_star.setter
        def stress_star(inner_self, value: sympy.Function):
            inner_self._stress_star = value
            inner_self._reset()

        @property
        def strainrate_inv_II_min(inner_self):
            return inner_self._epsilon_edot_II

        @strainrate_inv_II_min.setter
        def strainrate_inv_II_min(inner_self, value: float):
            inner_self._epsilon_edot_II = sympy.sympify(value)
            inner_self._reset()

    ## End of parameters definition

    @property
    def K(self):
        return self.viscosity

    # This has no setter !!
    @property
    def viscosity(self):
        # detect if values we need are defined or are placeholder symbols

        inner_self = self.Parameters

        if inner_self.yield_stress == sympy.oo:
            return inner_self.ve_effective_viscosity

        effective_viscosity = inner_self.ve_effective_viscosity

        if self.is_viscoplastic:
            # vp_effective_viscosity = self._plastic_effective_viscosity
            # effective_viscosity = sympy.Min(effective_viscosity, vp_effective_viscosity)

            ## Why is it p**2 here ?
            p = self.plastic_correction()
            effective_viscosity *= 2 * p**2 / (1 + p**2)

            # effective_viscosity *= self.plastic_correction()

        # If we want to apply limits to the viscosity but see caveat above

        if inner_self.shear_viscosity_min is not None:
            return sympy.Max(
                effective_viscosity,
                inner_self.shear_viscosity_min,
            )  # .rewrite(sympy.Piecewise)

        else:
            return effective_viscosity

    @property
    def _plastic_effective_viscosity(self):
        parameters = self.Parameters

        if parameters.yield_stress == sympy.oo:
            return sympy.oo

        Edot = self.grad_u

        ## Assume just the one history term

        if self.flux_dt is not None:
            stress_star = self.flux_dt.psi_star[0]

            if self.is_elastic:
                print("Adding stress history in plastic term", flush=True)
                Edot += stress_star.sym / (
                    2 * self.Parameters.dt_elastic * self.Parameters.shear_modulus
                )

        strainrate_inv_II = sympy.sqrt((Edot**2).trace() / 2)

        if parameters.yield_stress_min is not None:
            yield_stress = sympy.Max(
                parameters.yield_stress_min, parameters.yield_stress
            )  # .rewrite(sympy.Piecewise)
        else:
            yield_stress = parameters.yield_stress

        viscosity_yield = yield_stress / (
            2.0 * (strainrate_inv_II + parameters.strainrate_inv_II_min)
        )

        return viscosity_yield

    def plastic_correction(self):
        parameters = self.Parameters

        if parameters.yield_stress == sympy.oo:
            return sympy.sympify(1)

        stress = self.stress_projection()

        # The yield criterion in this case is assumed to be a bound on the second invariant of the stress

        stress_II = sympy.sqrt((stress**2).trace() / 2)

        correction = parameters.yield_stress / stress_II

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
        inner_self = self.Parameters
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

        if self.flux_dt is not None:
            stress_star = self.flux_dt.psi_star[0]

            if self.is_elastic:
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

        if self.flux_dt is not None:
            stress_star = self.flux_dt.psi_star[0]

            if self.is_elastic:
                stress += (
                    self.viscosity
                    * stress_star.sym
                    / (self.Parameters.dt_elastic * self.Parameters.shear_modulus)
                )

        stress = sympy.simplify(stress)

        return stress

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        # super()._object_viewer()

        display(Markdown(r"### Viscous deformation"))
        display(
            Latex(
                r"$\quad\eta_\textrm{0} = $ "
                + sympy.sympify(self.Parameters.shear_viscosity_0)._repr_latex_()
            ),
        )

        ## If elasticity is active:
        display(Markdown(r"#### Elastic deformation"))
        display(
            Latex(
                r"$\quad\mu = $ "
                + sympy.sympify(self.Parameters.shear_modulus)._repr_latex_(),
            ),
            Latex(
                r"$\quad\Delta t_e = $ "
                + sympy.sympify(self.Parameters.dt_elastic)._repr_latex_(),
            ),
            Latex(
                r"$\quad \sigma^* = $ "
                + sympy.sympify(self.Parameters.stress_star)._repr_latex_(),
            ),
        )

        # If plasticity is active
        display(Markdown(r"#### Plastic deformation"))
        display(
            Latex(
                r"$\quad\tau_\textrm{y} = $ "
                + sympy.sympify(self.Parameters.yield_stress)._repr_latex_(),
            )
            ## Todo: add all the other properties in here
        )

    @property
    def is_elastic(self):
        # If any of these is not defined, elasticity is switched off

        if self.Parameters.dt_elastic is sympy.oo:
            return False

        if self.Parameters.shear_modulus is sympy.oo:
            return False

        if isinstance(self.Parameters.stress_star, sympy.core.symbol.Symbol):
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
            owning_model,
            diffusivity: Union[float, sympy.Function] = 1,
        ):
            inner_self._diffusivity = diffusivity
            inner_self.owning_model = owning_model

        @property
        def diffusivity(inner_self):
            return inner_self._diffusivity

        @diffusivity.setter
        def diffusivity(inner_self, value: Union[float, sympy.Function]):
            inner_self._diffusivity = value
            inner_self._reset()

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


class TransverseIsotropicFlowModel(Constitutive_Model):
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

    class _Parameters:
        """Any material properties that are defined by a constitutive relationship are
        collected in the parameters which can then be defined/accessed by name in
        individual instances of the class.
        """

        def __init__(
            inner_self,
            owning_model,
            eta_0: Union[float, sympy.Function] = 1,
            eta_1: Union[float, sympy.Function] = 1,
            director: Union[sympy.Matrix, sympy.Function] = sympy.Matrix([0, 0, 1]),
        ):
            inner_self.owning_model = owning_model

            inner_self._eta_0 = eta_0
            inner_self._eta_1 = eta_1
            inner_self._director = director
            # inner_self.constitutive_model_class = const_model

        ## Note the inefficiency below if we change all these values one after the other

        @property
        def eta_0(inner_self):
            return inner_self._eta_0

        @eta_0.setter
        def eta_0(
            inner_self,
            value: Union[float, sympy.Function],
        ):
            inner_self._eta_0 = value
            inner_self._reset()

        @property
        def eta_1(inner_self):
            return inner_self._eta_1

        @eta_1.setter
        def eta_1(
            inner_self,
            value: Union[float, sympy.Function],
        ):
            inner_self._eta_1 = value
            inner_self._reset()

        @property
        def director(inner_self):
            return inner_self._director

        @director.setter
        def director(
            inner_self,
            value: Union[sympy.Matrix, sympy.Function],
        ):
            inner_self._director = value
            inner_self._reset()

    def __init__(self, dim):
        u_dim = dim
        super().__init__(dim, u_dim)

        # default values ... maybe ??
        return

    def _build_c_tensor(self):
        """For this constitutive law, we expect two viscosity functions
        and a sympy matrix that describes the director components n_{i}"""

        if self._is_setup:
            return

        d = self.dim
        dv = uw.maths.tensor.idxmap[d][0]

        eta_0 = self.Parameters.eta_0
        eta_1 = self.Parameters.eta_1
        n = self.Parameters.director

        Delta = eta_1 - eta_0

        lambda_mat = uw.maths.tensor.rank4_identity(d) * eta_0

        for i in range(d):
            for j in range(d):
                for k in range(d):
                    for l in range(d):
                        lambda_mat[i, j, k, l] += Delta * (
                            (
                                n[i] * n[k] * int(j == l)
                                + n[j] * n[k] * int(l == i)
                                + n[i] * n[l] * int(j == k)
                                + n[j] * n[l] * int(k == i)
                            )
                            / 2
                            - 2 * n[i] * n[j] * n[k] * n[l]
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
