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

import numpy as np
import sympy

import underworld3 as uw
import underworld3.timing as timing
from underworld3.coordinates import CoordinateSystemType
from underworld3.function import expression
from underworld3.systems import SNES_Vector
from underworld3.utilities._api_tools import Template

from .solvers import _ConstitutiveModelStateMixin, _StressHistoryMixin
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


def _facet_normal_field(mesh):
    r"""Unit normal of each (flat) cell of a manifold mesh as a cell-wise
    constant vector field, built once per mesh.

    The strain rate the assembler sees is the gradient along the flat facets,
    so the in-plane projector has to use the facet's own normal: with the
    analytic normal of the true surface a rigid rotation (a Killing field,
    zero strain on the sphere) keeps a spurious strain of order :math:`h`
    (measured: 0.7% of the velocity at h = 0.25). The sign of the normal does
    not enter :math:`\mathbf{n}\mathbf{n}^T`.
    """
    field = getattr(mesh, "_sheet_facet_normal", None)
    if field is None:
        field = uw.discretisation.MeshVariable(
            "sheet_facet_normal", mesh, mesh.cdim, degree=0, continuous=False)
        c_start, c_end = mesh.dm.getHeightStratum(0)
        normals = np.array([mesh.dm.computeCellGeometryFVM(c)[2] for c in range(c_start, c_end)])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        field.array[:, 0, :] = normals
        mesh._sheet_facet_normal = field
    return field


def _tangent_projector(mesh):
    r""":math:`\mathbf{P} = \mathbf{I} - \mathbf{n}\mathbf{n}^T` with the
    facet normal of the cell (the identity on a flat mesh), ``(cdim, cdim)``."""
    if mesh.dim == mesh.cdim:
        return sympy.eye(mesh.cdim)
    n = sympy.Matrix(_facet_normal_field(mesh).sym).reshape(1, mesh.cdim)
    return sympy.eye(mesh.cdim) - n.T * n


def _surface_spin(mesh, L):
    r"""Spin of a material element of a sheet on a curved surface,

    .. math::

        \mathbf{W} = \mathrm{skew}(\mathbf{P}\mathbf{L}\mathbf{P})
        + \dot{\mathbf{n}}\mathbf{n}^T - \mathbf{n}\dot{\mathbf{n}}^T,
        \qquad \dot{\mathbf{n}} = -\mathbf{L}^T\mathbf{n},

    with :math:`\mathbf{n}` the cell's facet normal, :math:`\mathbf{P}` its
    projector and :math:`\mathbf{L}` the surface velocity gradient the
    assembler supplies (projected on the derivative index only). The first term
    is the rotation within the surface; the second turns the normal, which for
    a tangential flow rotates at :math:`-\mathbf{L}^T\mathbf{n}` because
    :math:`\nabla_s(\mathbf{u}\cdot\mathbf{n}) = 0`. For a rigid rotation
    :math:`\mathbf{u} = \boldsymbol\omega\times\mathbf{X}` (exact in the P2
    space) this is the rotation tensor :math:`\boldsymbol\Omega` itself, so a
    Jaumann-rate stress is carried rigidly; :math:`\mathrm{skew}(\mathbf{L})`
    alone misses the turning of the normal."""
    n = sympy.Matrix(_facet_normal_field(mesh).sym).reshape(mesh.cdim, 1)
    P = sympy.eye(mesh.cdim) - n * n.T
    L = sympy.Matrix(L)
    PLP = P * L * P
    ndot = -L.T * n
    return (PLP - PLP.T) / 2 + ndot * n.T - n * ndot.T


class SNES_ThinSheet(_StressHistoryMixin, _ConstitutiveModelStateMixin, SNES_Vector):
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
                if new_u.mesh.dim < new_u.mesh.cdim:
                    inner_self._W = _surface_spin(new_u.mesh, inner_self._L)

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
        # On a manifold the velocity is carried in the embedding space; its
        # tangency u . n = 0 is a strong rotated constraint at every node (the
        # same per-node rotation as rotated free-slip and the split-node fault
        # pairs, so faults compose with it). Registering it routes solve()
        # through the rotated path.
        if mesh.dim < mesh.cdim:
            from underworld3.utilities.rotated_bc import SHEET_TANGENCY
            self._rotated_freeslip_bcs.append((SHEET_TANGENCY, _sheet_normal(mesh)))
        # The stress history is carried on the grid (the streamline-upwind
        # transport the Eulerian solvers use). On a surface mesh the
        # semi-Lagrangian trace-back leaves the faceted surface; next to a
        # split-node fault its point evaluation reads across the slit (measured:
        # a period-2-step slip-rate oscillation that the grid history removes,
        # at a sixth of the Newton work).
        self._stress_transport = "eulerian"
        # Last-seen effective order of the stress history (it ramps over the
        # opening steps); a change rewires the pointwise functions.
        self._prev_effective_order = None

    @timing.routine_timer_decorator
    def solve(
        self,
        zero_init_guess: bool = None,
        timestep: float = None,
        _force_setup: bool = False,
        verbose: bool = False,
        evalf: bool = False,
        divergence_retries: int = 0,
    ):
        """Solve the sheet momentum balance; with a viscoelastic constitutive
        model, carry the stress history across the step first.

        Parameters
        ----------
        timestep : float
            Required when the constitutive model carries a stress history: the
            elastic timestep of the Maxwell element.
        zero_init_guess, verbose, divergence_retries
            As for :meth:`SNES_Vector.solve`.
        evalf : bool, default False
            Numerical evaluation during the history carry.
        """
        if self.Unknowns.DFDt is None:
            super().solve(zero_init_guess, _force_setup=_force_setup, verbose=verbose,
                          divergence_retries=divergence_retries)
            return
        if timestep is None:
            raise ValueError(
                "timestep is required for a viscoelastic sheet: call "
                "sheet.solve(timestep=dt)")
        if (self.mesh.dim < self.mesh.cdim
                and getattr(self.constitutive_model, "_objective_rate", "none") == "upper_convected"):
            raise NotImplementedError(
                "the upper-convected rate needs the full velocity gradient of a "
                "material element, which a sheet on a curved surface does not yet "
                "supply; use objective_rate='jaumann' (the spin is supplied) or 'none'.")
        self._stress_history_prepare(timestep, _force_setup=_force_setup)
        if not self.is_setup:
            self._setup_pointwise_functions(verbose)
            self._setup_discretisation(verbose)
            self._setup_solver(verbose)
        self._stress_history_advance(timestep, verbose=verbose, evalf=evalf)
        super().solve(zero_init_guess, _force_setup=_force_setup, verbose=verbose,
                      divergence_retries=divergence_retries)
        self._stress_history_post_solve(timestep, verbose=verbose, evalf=evalf)
        self._keep_history_tangent()

    def _keep_history_tangent(self):
        r"""On a manifold, project every stored history level onto the tangent
        plane, :math:`\boldsymbol\sigma^* \leftarrow \mathbf{P}\boldsymbol\sigma^*\mathbf{P}`,
        with the analytic surface normal at each history node.

        The history is transported component by component in the embedding
        space, so a tensor carried along the curved surface keeps its Cartesian
        components while the tangent plane under it turns: it acquires normal
        components, which the plane-stress closure reads as a trace and so as a
        spurious pressure. Measured without this, an unfaulted Maxwell sheet
        driven by a divergence-free flow carried normal components up to 31% of
        the stress by t = 3. The projection removes the normal part left after
        the rotation.

        It is applied to the COMMITTED stress, after the solve, never between the
        transport and the solve. After the transport the history at a node still
        carries the components it had upstream, tangent to the upstream plane;
        the objective rotation inside the solve (the spin includes the turning of
        the normal) turns them into the node's own plane. Projecting first throws
        those components away before the rotation can use them: measured on a
        rigid quarter-turn of the sphere, a first-order error (4.0e-2 at 40 steps,
        1.9e-2 at 80) where the flat sheet carries the stress to 5e-4."""
        if self.mesh.dim == self.mesh.cdim:
            return
        for level in self.Unknowns.DFDt.psi_star:
            coords = level.coords
            rhat = coords / np.linalg.norm(coords, axis=1)[:, None]
            P = np.eye(self.mesh.cdim)[None] - rhat[:, :, None] * rhat[:, None, :]
            projected = P @ np.asarray(level.array) @ P
            # symmetric to round-off; a symmetric tensor variable stores one
            # value per off-diagonal pair and refuses two that differ
            level.array[...] = 0.5 * (projected + projected.transpose(0, 2, 1))

    @property
    def stress_transport(self) -> str:
        """How a viscoelastic stress history is carried (see
        :attr:`SNES_Stokes.stress_transport`). The default is the grid
        (``"eulerian"``, the streamline-upwind transport). On a manifold it is
        the only one: the semi-Lagrangian trace-back leaves the faceted surface.
        On a flat mesh the others may be chosen, but next to a split-node fault
        the trace-back reads across the slit."""
        return super().stress_transport

    @stress_transport.setter
    def stress_transport(self, value):
        if self.mesh.dim < self.mesh.cdim and str(value) != "eulerian":
            raise ValueError(
                f"stress_transport={value!r} is not available on a manifold "
                "mesh: the thin sheet carries its history on the grid "
                "('eulerian'); the semi-Lagrangian trace-back leaves the "
                "faceted surface.")
        _StressHistoryMixin.stress_transport.fset(self, value)

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------

    @property
    def tangent_projector(self):
        r""":math:`\mathbf{P} = \mathbf{I} - \mathbf{n}\mathbf{n}^T` with the
        cell's facet normal; the identity on a flat mesh. The tangency
        constraint at the nodes uses the analytic normal of the surface."""
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
