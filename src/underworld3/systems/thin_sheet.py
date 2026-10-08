r"""Thin-sheet solver: a vertically averaged (plane-stress) viscous or
viscoelastic sheet, flat or on a surface embedded in three dimensions.

The sheet carries a tangential velocity :math:`\mathbf{u}` and a thickness
:math:`L`. Vertical averaging with the plane-stress closure
:math:`\sigma_{nn} = 0` on the dynamic stress eliminates the pressure:
with :math:`\boldsymbol{\tau}` the deviatoric stress of an incompressible
material, :math:`p = \tau_{nn} = -\operatorname{tr}_s \boldsymbol{\tau}`, so the
in-plane stress is

.. math::

    \boldsymbol{\sigma} = \boldsymbol{\tau}
    + \operatorname{tr}(\boldsymbol{\tau})\,\mathbf{P},
    \qquad \mathbf{P} = \mathbf{I} - \hat{\mathbf{n}}\hat{\mathbf{n}}^T,

(England & McKenzie 1982). Any deviatoric constitutive law applies unchanged,
a Maxwell viscoelastic one included; for a Newtonian sheet
:math:`\boldsymbol{\sigma} = 2\eta\,(\dot{\boldsymbol\varepsilon}_s
+ \operatorname{tr}\dot{\boldsymbol\varepsilon}_s\,\mathbf{P})`, whose uniaxial
limit is the Trouton viscosity :math:`3\eta`. The momentum balance solved is

.. math::

    -\nabla_s \cdot \left( L\,\boldsymbol{\sigma} \right) + c\,\mathbf{u}
    = \mathbf{f},

with :math:`c` a basal drag coefficient (:math:`\eta_a/h_a` for a viscous
substrate) and :math:`\mathbf{f}` a force per unit area. There is no pressure
unknown, so this is a vector problem, not a saddle point.

On a flat mesh :math:`\mathbf{P} = \mathbf{I}` and the velocity has two
components. On a manifold (``mesh.dim < mesh.cdim``, e.g.
:class:`~underworld3.meshing.SphericalManifold`) the velocity is carried in the
embedding space with :math:`\mathbf{P}` built from the surface normal; the
in-plane strain rate is :math:`\dot{\boldsymbol\varepsilon}_s = \mathbf{P}\,
\operatorname{sym}(\nabla_s \mathbf{u})\,\mathbf{P}`, the outer projection
supplying the curvature term :math:`-\mathbf{u}\cdot\nabla_s\hat{\mathbf{n}}`.
"""

from typing import Optional, Union

import sympy

import underworld3 as uw
import underworld3.timing as timing
from underworld3.coordinates import CoordinateSystemType
from underworld3.function import expression
from underworld3.systems import SNES_Vector
from underworld3.utilities._api_tools import Template

from .solvers import _ConstitutiveModelStateMixin
from .ddt import SemiLagrangian as SemiLagrangian_DDt
from .ddt import Lagrangian as Lagrangian_DDt


def _nondimensional_row(value, cdim):
    """A ``(1, cdim)`` sympy row from a sequence whose entries may be
    dimensional ``UWQuantity`` values (converted to the solver's non-dimensional
    scale) or plain numbers / expressions."""
    # TODO(DESIGN): SNES_Stokes.bodyforce repeats this conversion inline; share it.
    row = []
    for item in value:
        if isinstance(item, uw.function.quantities.UWQuantity):
            nd = uw.non_dimensionalise(item)
            nd = nd.magnitude if hasattr(nd, "magnitude") else nd
            if hasattr(nd, "shape") and tuple(nd.shape) == (1, 1):
                nd = nd[0, 0]
            row.append(nd)
        else:
            row.append(item)
    return sympy.Matrix(row).reshape(1, cdim)


def _sheet_normal(mesh):
    r"""Unit normal :math:`\hat{\mathbf{n}}` of the sheet as a ``(1, cdim)``
    row, or ``None`` on a flat (volume) mesh."""
    if mesh.dim == mesh.cdim:
        return None
    if mesh.CoordinateSystem.coordinate_type is CoordinateSystemType.SPHERICAL:
        return sympy.Matrix(mesh.CoordinateSystem.unit_e_0).reshape(1, mesh.cdim)
    raise NotImplementedError(
        "ThinSheet needs the surface normal of the sheet: only a flat mesh "
        "or a spherical manifold (CoordinateSystemType.SPHERICAL) is supported.")


def _tangent_projector(mesh):
    r""":math:`\mathbf{P} = \mathbf{I} - \hat{\mathbf{n}}\hat{\mathbf{n}}^T`
    (the identity on a flat mesh), ``(cdim, cdim)``."""
    n = _sheet_normal(mesh)
    if n is None:
        return sympy.eye(mesh.cdim)
    return sympy.eye(mesh.cdim) - n.T * n


class SNES_ThinSheet(_ConstitutiveModelStateMixin, SNES_Vector):
    r"""Vertically averaged thin sheet (plane stress), viscous or viscoelastic.

    Solves

    .. math::

        -\nabla_s \cdot \left( L\,\boldsymbol{\sigma} \right) + c\,\mathbf{u}
        = \mathbf{f},
        \qquad
        \boldsymbol{\sigma} = \boldsymbol{\tau}
        + \operatorname{tr}(\boldsymbol{\tau})\,\mathbf{P},

    for the sheet velocity :math:`\mathbf{u}`, with :math:`\boldsymbol{\tau}`
    the deviatoric stress of the constitutive model evaluated on the in-plane
    strain rate :math:`\dot{\boldsymbol\varepsilon}_s`. See the module
    docstring for the closure.

    Parameters
    ----------
    mesh : Mesh
        A flat mesh (``dim == cdim``) or a spherical manifold.
    velocityField : MeshVariable, optional
        Sheet velocity, ``cdim`` components. Created if not supplied.
    degree : int, default 2
        Polynomial degree of the velocity space.
    verbose : bool, default False
    DuDt, DFDt : optional
        Time-derivative managers; a viscoelastic constitutive model creates
        its own stress history when none is supplied.

    Examples
    --------
    A Newtonian sheet stretched along :math:`x` with free lateral edges
    carries :math:`\sigma_{xx} = 3\eta\dot\varepsilon` (the Trouton ratio):

    >>> sheet = uw.systems.ThinSheet(mesh)
    >>> sheet.constitutive_model = uw.constitutive_models.ViscousFlowModel
    >>> sheet.constitutive_model.Parameters.shear_viscosity_0 = 1
    >>> sheet.add_essential_bc([-1, None], "Left")
    >>> sheet.add_essential_bc([1, None], "Right")
    >>> sheet.solve()
    """

    class _Unknowns(SNES_Vector._Unknowns):
        """The sheet's kinematics: the strain rate handed to the constitutive
        model is the in-plane one, :math:`\mathbf{P}\,\mathrm{sym}(\nabla_s
        \mathbf{u})\,\mathbf{P}`."""

        @SNES_Vector._Unknowns.u.setter
        def u(inner_self, new_u):
            SNES_Vector._Unknowns.u.fset(inner_self, new_u)
            if new_u is not None:
                P = _tangent_projector(new_u.mesh)
                inner_self._E = P * inner_self._E * P
                inner_self._Einv2 = sympy.sqrt(
                    (sympy.Matrix(inner_self._E) ** 2).trace() / 2)

    _solver_terms = (
        ("thickness", "sheet thickness L multiplying the in-plane stress"),
        ("bodyforce", "force per unit area; F0 is its negative"),
        ("basal_drag", "drag coefficient c on the sheet velocity (eta_a / h_a)"),
    )

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        velocityField: Optional[uw.discretisation.MeshVariable] = None,
        degree: Optional[int] = 2,
        verbose: Optional[bool] = False,
        DuDt: Union[SemiLagrangian_DDt, Lagrangian_DDt] = None,
        DFDt: Union[SemiLagrangian_DDt, Lagrangian_DDt] = None,
    ):
        super().__init__(mesh, velocityField, degree, verbose, DuDt=DuDt, DFDt=DFDt)

        self._P = _tangent_projector(mesh)
        self._thickness = expression(R"L", 1, "Sheet thickness")
        self._basal_drag = expression(R"c_b", 0, "Basal drag coefficient")
        self._bodyforce = expression(
            Rf"\mathbf{{f}}_0\left( {self.Unknowns.u.symbol} \right)",
            sympy.Matrix([[0] * self.mesh.cdim]),
            "Thin-sheet force per unit area: f_0(u)",
        )
        self._needs_function_rewire = True
        self._constitutive_model = None

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------

    @property
    def tangent_projector(self):
        r""":math:`\mathbf{P} = \mathbf{I} - \hat{\mathbf{n}}\hat{\mathbf{n}}^T`;
        the identity on a flat mesh."""
        return self._P

    # ------------------------------------------------------------------
    # parameters
    # ------------------------------------------------------------------

    @property
    def thickness(self):
        r"""Sheet thickness :math:`L` (a constant or an expression)."""
        return self._thickness

    @thickness.setter
    def thickness(self, value):
        self._needs_function_rewire = True
        self._thickness.sym = value

    @property
    def basal_drag(self):
        r"""Basal drag coefficient :math:`c` in :math:`c\,\mathbf{u}`; for a
        viscous substrate of viscosity :math:`\eta_a` and thickness
        :math:`h_a`, :math:`c = \eta_a / h_a`."""
        return self._basal_drag

    @basal_drag.setter
    def basal_drag(self, value):
        self._needs_function_rewire = True
        self._basal_drag.sym = value

    @property
    def bodyforce(self):
        r"""Force per unit area :math:`\mathbf{f}` on the right-hand side of
        the momentum balance."""
        return self._bodyforce

    @bodyforce.setter
    def bodyforce(self, value):
        self._needs_function_rewire = True
        if isinstance(value, uw.function.expressions.UWexpression):
            self._bodyforce.sym = value.sym
        else:
            self._bodyforce.sym = _nondimensional_row(value, self.mesh.cdim)

    # ------------------------------------------------------------------
    # kinematics and stress
    # ------------------------------------------------------------------

    @property
    def strainrate(self):
        r"""In-plane strain rate :math:`\dot{\boldsymbol\varepsilon}_s =
        \mathbf{P}\,\mathrm{sym}(\nabla_s \mathbf{u})\,\mathbf{P}`."""
        return sympy.Matrix(self.Unknowns.E)

    @property
    def stress_deviator(self):
        r"""In-plane deviatoric stress :math:`\boldsymbol{\tau}` from the
        constitutive model."""
        return sympy.Matrix(self.constitutive_model.flux)

    @property
    def stress(self):
        r"""In-plane sheet stress :math:`\boldsymbol{\sigma} = \boldsymbol{\tau}
        + \operatorname{tr}(\boldsymbol{\tau})\,\mathbf{P}` (plane-stress
        closure, :math:`p = -\operatorname{tr}\boldsymbol{\tau}`)."""
        tau = self.stress_deviator
        return tau + tau.trace() * self._P

    # ------------------------------------------------------------------
    # PETSc residual templates
    # ------------------------------------------------------------------

    F0 = Template(
        r"\mathbf{f}_0\left( \mathbf{u} \right)",
        lambda self: -self.bodyforce.sym + self.basal_drag.sym * self.u.sym,
        r"""Pointwise term: basal drag :math:`c\,\mathbf{u}` less the force per
        unit area :math:`\mathbf{f}`.""",
    )

    F1 = Template(
        r"\mathbf{F}_1\left( \mathbf{u} \right)",
        lambda self: self.thickness.sym * self.stress,
        r"""Flux term: the depth-integrated in-plane stress :math:`L\,
        \boldsymbol{\sigma}`.""",
    )
