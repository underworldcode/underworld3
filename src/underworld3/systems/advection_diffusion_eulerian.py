r"""Advection-diffusion composed from a DDt transport manager.

The solver assembles the diffusive flux and the source on the mesh and takes
its transport (time derivative, advection, stabilisation) from the history
manager it holds. The default manager, :class:`~underworld3.systems.ddt.EulerianSUPG`,
makes it the fully implicit Eulerian scheme with SUPG stabilisation described
below; a :class:`~underworld3.systems.ddt.SemiLagrangian` manager makes it a
semi-Lagrangian scheme on the field history.

The scalar transport equation

.. math::
    \frac{\partial \phi}{\partial t} + \mathbf{u}\cdot\nabla\phi
        - \nabla\cdot(\kappa\nabla\phi) = f

discretised on the mesh with a linear multistep rule in time and a
streamline-upwind Petrov-Galerkin (SUPG) term in space. Every time level
is a mesh variable held by an :class:`~underworld3.systems.ddt.Eulerian`
history manager, so the scheme's order is a construction argument and the
timestep and multistep coefficients are runtime constants of the compiled
kernels: neither changes the generated code.

The companion of :class:`~underworld3.systems.solvers.SNES_AdvectionDiffusion`
(semi-Lagrangian). The Eulerian scheme is stable at any cell Courant number
and its accuracy is set by how far the transported feature moves in one
step; the semi-Lagrangian scheme's accuracy is set by how far a characteristic
turns in one step. See ``docs/developer/design/eulerian-supg-transport.md``.

The SUPG weak form, the Petrov-Galerkin test-function perturbation written
as a flux so that PETSc needs no modified test space, and its first
implementation on PetscDS are NengLu's (issue #657, branch ``levelset``);
this module keeps that formulation and its stabilisation parameter.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import sympy
from typing import Callable, Optional, Union

import underworld3 as uw
import underworld3.timing as timing
from underworld3.checkpoint.state import SnapshottableState
from underworld3.systems import SNES_Scalar
from underworld3.utilities._api_tools import Template
from underworld3.function import expression as public_expression
from underworld3.systems.ddt import _DDtBase, _as_row_vector
from underworld3.systems.ddt import Eulerian as Eulerian_DDt
from underworld3.systems.ddt import EulerianSUPG as EulerianSUPG_DDt
from underworld3.systems.solvers import (
    _advective_diffusive_dt,
    _dimensionalise_dt,
    _invalidate_solution_cache,
    _nondimensionalise_timestep,
)


@dataclass
class AdvDiffusionState(SnapshottableState):
    """Solver timestep/estimator metadata; transport history belongs to DuDt."""

    last_timestep: Optional[float] = None
    last_change_rate: Optional[float] = None
    order: int = 1
    theta: float = 0.5


def _check_supplied_manager(DuDt, order, theta):
    """A supplied history manager fixes the scheme: the arguments must agree with it."""
    if DuDt.order != order:
        raise ValueError(
            f"DuDt supplied is order {DuDt.order} but order={order} was asked for: a "
            "supplied manager fixes the scheme, pass the matching order.")
    if theta is not None and hasattr(DuDt, "theta") and float(DuDt.theta) != float(theta):
        raise ValueError(
            f"DuDt supplied has theta={DuDt.theta} but theta={theta} was asked for.")


class SNES_AdvectionDiffusion_Composed(SNES_Scalar):
    r"""Advection-diffusion solver composed from its DDt transport manager.

    With the default manager (:class:`~underworld3.systems.ddt.EulerianSUPG`):
    implicit in time, assembled on the mesh, SUPG in space.

    .. math::
        \frac{\partial \phi}{\partial t} + \mathbf{u}\cdot\nabla\phi
            - \nabla\cdot(\kappa\nabla\phi) = f

    A drop-in replacement for :class:`~underworld3.systems.solvers.SNES_AdvectionDiffusion`
    (``uw.systems.AdvDiffusionSLCN``): the constructor, ``order``, ``theta``,
    ``f``, ``V_fn``, ``constitutive_model``, ``delta_t``, ``estimate_dt`` and
    ``solve`` all keep the semi-Lagrangian solver's meaning, so a script changes
    the class name and nothing else::

        adv = uw.systems.AdvDiffusion(mesh, T, v.sym, order=1)   # was AdvDiffusionSLCN
        adv.constitutive_model = uw.constitutive_models.DiffusionModel
        adv.constitutive_model.Parameters.diffusivity = 1.0e-3
        adv.add_dirichlet_bc(0.0, "Left")
        adv.solve(timestep=dt)

    The arguments that only make sense for a trace-back
    (``restore_points_func``, ``monotone_mode``, ``old_frame_traceback``,
    ``DFDt``) are accepted and ignored with a warning.

    **Time schemes.** ``order`` and ``theta`` select the same schemes as for
    the semi-Lagrangian solver:

    ==========  =======  =====================================================
    ``order``   ``theta``  scheme
    ==========  =======  =====================================================
    1           0.5      Crank-Nicolson (default; the SLCN convention)
    1           1.0      backward Euler
    2           1.0      BDF2, all spatial terms at :math:`n+1` (the SL-BDF2 convention)
    3           1.0      BDF3
    ==========  =======  =====================================================

    ``order=2`` with ``theta=0.5`` is refused, as the semi-Lagrangian
    documentation says: a BDF stencil pairs with terms at :math:`n+1`, not
    with a centred flux. Every past time level is a mesh variable held by an
    :class:`~underworld3.systems.ddt.Eulerian` history manager, so gradients
    of past states are available in the kernels and both families come from
    one code path:

    backward differentiation (order :math:`N \ge 2`)

    .. math::
        \frac{1}{\Delta t}\sum_{k=0}^{N} c_k\,\phi^{n+1-k}
            + \mathbf{u}\cdot\nabla\phi^{n+1}
            - \nabla\cdot(\kappa\nabla\phi^{n+1}) = f

    the :math:`\theta` rule (order 1; Adams-Moulton of one step)

    .. math::
        \frac{\phi^{n+1}-\phi^{n}}{\Delta t}
            + \sum_{k=0}^{N} a_k\left[\mathbf{u}\cdot\nabla\phi^{n+1-k}
            - \nabla\cdot(\kappa\nabla\phi^{n+1-k})\right] = f

    The higher Adams-Moulton rules are assembled by the same code but are
    not offered: their bounded stability region blows up on an advection
    operator from about Courant 1 (see the design note). Both families ramp
    from first order over the opening steps unless a history is planted with
    ``solver.DuDt.set_initial_history``. A BDF3 request falls back to
    variable-step BDF2 whenever consecutive timesteps differ by more than 5%.

    **Which scheme.** Measured on a rotating Gaussian
    (``docs/developer/design/eulerian-supg-transport.md``): Crank-Nicolson is
    three to four times more accurate than BDF2 at the same timestep below
    Courant 2 on the feature scale, and rings once the feature is
    under-resolved in time; BDF2 is damped and stable at every Courant
    number; BDF3 is the most accurate scheme below Courant 1 when diffusion
    is present but grows slowly on pure advection; backward Euler carries 20
    to 40% error at any practical timestep.

    **Weak form.** With the strong residual of the chosen scheme
    :math:`R(\phi)` (time derivative, advection, source) the residual
    assembled through PETSc's pointwise interface is

    .. math::
        f_0 = R(\phi), \qquad
        \mathbf{f}_1 = \sum_k w_k\,\kappa\nabla\phi^{n+1-k}
            + \tau\,R(\phi)\,\mathbf{u},

    where :math:`w_k` are the weights of the spatial operator (:math:`w_0 = 1`
    for BDF, :math:`w_k = a_k` for Adams-Moulton). The SUPG contribution is
    the Petrov-Galerkin test-function perturbation
    :math:`\tau\,\mathbf{u}\cdot\nabla w` written as a flux against
    :math:`\nabla w`, so PETSc needs no modified test space. The strong
    residual carries no diffusion term because the pointwise kernels see
    first derivatives only; for linear elements that term vanishes
    identically, for higher orders it is the usual inconsistency of SUPG
    without a Laplacian reconstruction.

    **Stabilisation parameter.**

    .. math::
        \tau = \left[\left(\frac{2 c_0}{\Delta t}\right)^2
            + \left(\frac{2|\mathbf{u}|}{h}\right)^2
            + \left(\frac{4\kappa}{h^2}\right)^2\right]^{-1/2}

    with :math:`h` the local cell size (``mesh.cell_size()``) and
    :math:`c_0` the leading multistep coefficient. The three weights are
    runtime constants (``tau_weights``) and ``supg_weight`` scales the whole
    term, so a Galerkin baseline needs no rebuild.

    **What limits the timestep.** Nothing, for stability: the implicit
    scheme is stable at any cell Courant number, including on cells refined
    for a Stokes problem that the scalar does not need. Accuracy is set by
    how far the transported feature moves per step relative to its own
    width, as :math:`(\mathbf{u}\Delta t)^2` for the second-order schemes.
    :meth:`estimate_dt` therefore returns an accuracy-based step, the
    allowed change of the field per step as a fraction of its range, and
    only reports the cell-crossing time on request
    (``basis="resolution"``). Against the semi-Lagrangian solver: the
    semi-Lagrangian error is flat in the timestep but accumulates one
    interpolation per step, and its limit is the arc a characteristic turns
    per step; the Eulerian solve costs four to six times less per step in
    serial and needs no departure points in parallel.

    Parameters
    ----------
    mesh : Mesh
    u_Field : MeshVariable
        Continuous scalar field :math:`\phi`.
    V_fn : MeshVariable or sympy Matrix
        Advecting velocity, ``(1, dim)``.
    order : int, default 1
        Time-integration order, 1 to 3 (see the table above).
    theta : float, optional
        Crank-Nicolson blend at order 1: 0.5 (the default there) is
        Crank-Nicolson, 1.0 is backward Euler. Above order 1 the only
        consistent value is 1.0, which is taken when ``theta`` is not given
        and refused when 0.5 is asked for explicitly.
    verbose : bool, default False
    DuDt : DDt history manager, optional
        The transport plugin. By default an
        :class:`~underworld3.systems.ddt.EulerianSUPG` built from ``V_fn``,
        ``order`` and ``theta``. Any history manager that follows the DDt
        transport contract (``time_derivative``, ``advection``,
        ``stabilisation_flux``) can be supplied instead: a
        :class:`~underworld3.systems.ddt.SemiLagrangian` history turns this
        solver into a semi-Lagrangian scheme on the field history, with no
        assembled advection and no stabilisation. A supplied manager fixes
        ``order`` and ``theta``.
    restore_points_func, monotone_mode, old_frame_traceback, DFDt
        Semi-Lagrangian arguments, accepted for drop-in compatibility and
        ignored with a warning: there is no trace-back here.

    Notes
    -----
    The diffusivity is set through the constitutive model, as for every
    scalar solver; the solver starts with a
    :class:`~underworld3.constitutive_models.DiffusionModel` at
    :math:`\kappa = 0` (pure advection). The linear system is nonsymmetric,
    so the solver uses GMRES with an additive-Schwarz ILU preconditioner, the
    Krylov tolerance matched to the SNES tolerance so that a step is one
    Newton iteration. ``preconditioner = "fmg"`` hands the linear solve to
    geometric multigrid over the mesh's refinement hierarchy (a flexible GMRES
    outer solver, Galerkin coarse operators); measured, the Schwarz solve is
    cheaper at every Courant number to eight ranks, and multigrid is there for
    the rank count where a one-level method runs out of coarse space. Every
    option is overridable through ``petsc_options``.
    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        u_Field: uw.discretisation.MeshVariable,
        V_fn,
        order: int = 1,
        theta: Optional[float] = None,
        peclet_weight: float = 4.0,
        verbose: bool = False,
        DuDt: Optional[Eulerian_DDt] = None,
        DFDt=None,
        restore_points_func: Optional[Callable] = None,
        monotone_mode: Optional[str] = None,
        old_frame_traceback: bool = False,
    ):
        if not u_Field.continuous:
            raise ValueError(
                "u_Field must be a continuous MeshVariable: the SUPG weak form "
                "is continuous Galerkin."
            )
        ignored = [name for name, value in (
            ("restore_points_func", restore_points_func),
            ("monotone_mode", monotone_mode),
            ("old_frame_traceback", old_frame_traceback),
            ("DFDt", DFDt),
        ) if value]
        if ignored:
            warnings.warn(
                f"AdvDiffusion ignores {', '.join(ignored)}: these configure "
                "the semi-Lagrangian trace-back and the Eulerian scheme has none.",
                stacklevel=2,
            )
        requested_theta = theta
        order = int(order)
        if order not in (1, 2, 3):
            raise ValueError(f"order must be 1, 2 or 3, not {order}.")
        # theta means what it means for the semi-Lagrangian solver: the
        # Crank-Nicolson blend at order 1. Left unset, order 2 and 3 take the
        # only consistent value; set explicitly to 0.5 there, it is refused.
        theta = float(theta) if theta is not None else (0.5 if order == 1 else 1.0)
        # The multistep family follows the order: the Adams-Moulton (theta)
        # rule at order 1, backward differentiation above. Adams-Moulton at
        # orders 2 and 3 is assembled by the same code but is not offered:
        # its bounded stability region blows up on an advection operator
        # from about Courant 1 (design note, integrator study).
        if theta != 1.0 and order != 1:
            raise ValueError(
                "theta applies at order 1 only (0.5 is Crank-Nicolson, 1.0 is "
                "backward Euler); order 2 and 3 take theta=1.0, the same rule as "
                "the semi-Lagrangian solver (a BDF stencil pairs with terms at n+1, "
                "not with a centred flux)."
            )

        super().__init__(mesh, u_Field, u_Field.degree, verbose, DuDt=DuDt, DFDt=None)

        self.f = sympy.Matrix.zeros(1, 1)
        self._last_timestep = None
        self._last_change_rate = None

        # The transport plugin: the history manager owns the time scheme, the
        # advecting velocity, the assembled advection and the stabilisation.
        if DuDt is None:
            self.Unknowns.DuDt = EulerianSUPG_DDt(
                self.mesh,
                u_Field,
                V_fn,
                vtype=uw.VarType.SCALAR,
                degree=u_Field.degree,
                continuous=u_Field.continuous,
                order=order,
                theta=theta,
                varsymbol=u_Field.symbol,
                verbose=verbose,
                bcs=self.essential_bcs,
                smoothing=0.0,
                peclet_weight=peclet_weight,
            )
        else:
            if not isinstance(DuDt, _DDtBase):
                raise TypeError(f"DuDt must be a DDt history manager, not {type(DuDt).__name__}.")
            if sympy.Matrix(DuDt.psi_fn).shape != u_Field.sym.shape:
                raise ValueError("DuDt tracks a different unknown from u_Field.")
            _check_supplied_manager(DuDt, order, requested_theta)
            self.Unknowns.DuDt = DuDt
        self._theta = float(getattr(self.DuDt, "theta", theta))

        # Diffusivity lives on the constitutive model, as for every scalar
        # solver; kappa = 0 until the user sets it.
        self.constitutive_model = uw.constitutive_models.DiffusionModel
        self.constitutive_model.Parameters.diffusivity = 0.0

        # Linear solver: additive-Schwarz ILU by default, the managed multigrid
        # block on request (see ``preconditioner``). One Newton iteration per
        # step: the operator is linear in phi, so the Krylov tolerance must
        # reach the SNES tolerance or the SNES takes a second step, and a
        # second Jacobian assembly costs more than every linear solve of the
        # step (design note, "Preconditioner").
        self._set_linear_solver(multigrid=False)
        self.petsc_options["snes_rtol"] = 1.0e-8
        self.petsc_options["ksp_rtol"] = 1.0e-9
        self.petsc_options["snes_max_it"] = 20
        self._bind_transport_manager(self.DuDt)
        uw.get_default_model()._register_state_bearer(self)

    # ------------------------------------------------------------------
    # Linear solver
    # ------------------------------------------------------------------

    _SCHWARZ_OPTIONS = {
        "ksp_type": "gmres",
        "ksp_gmres_restart": 200,
        "pc_type": "asm",
        "sub_pc_type": "ilu",
        # RCM ordering improves the ILU fill on a convection-dominated operator.
        "sub_pc_factor_mat_ordering_type": "rcm",
    }

    def _set_linear_solver(self, multigrid: bool):
        """Own the linear solver (GMRES + additive-Schwarz ILU) or hand it to
        the managed multigrid block.

        Measured on the level-set advection step at 256^2 and 512^2 (design
        note, "Preconditioner"): with the Krylov tolerance matched to the
        SNES tolerance, additive Schwarz with ILU is the cheaper linear solve
        at every Courant number from 1/2 to 32, its iteration count does not
        change between one and eight ranks, and the geometric multigrid's
        cycle count grows with the Courant number nearly as fast as the
        Schwarz iteration count while each cycle costs about three Schwarz
        iterations. The linear solve is under a tenth of the step either way;
        assembly is the rest. Multigrid keeps its coarse space for a rank
        count where a one-level method runs out of one, which is what
        ``preconditioner = "fmg"`` is for.
        """
        from underworld3.utilities import multigrid_options

        opts = self.petsc_options
        bundle_keys = set()
        for bundle in (multigrid_options.gamg_bundle(),
                       multigrid_options.geometric_mg_bundle()):
            bundle_keys |= set(bundle.settings) | set(bundle.stale)
        if multigrid:
            # The managed block starts from the scalar solver's own keys
            # (GMRES + the GAMG bundle) and _apply_preconditioner_options
            # resolves the request against the mesh hierarchy at build time.
            self._pc_option_prefix = ""
            for key in self._SCHWARZ_OPTIONS:
                opts.delValue(key)
            self._push_managed_option("ksp_type", "gmres")
            for key, value in multigrid_options.gamg_bundle().settings.items():
                self._push_managed_option(key, value)
        else:
            self._pc_option_prefix = None
            for key in bundle_keys | {"ksp_type"}:
                opts.delValue(key)
                self._managed_pc_options.pop(self.petsc_options_prefix + key, None)
            for key, value in self._SCHWARZ_OPTIONS.items():
                opts[key] = value

    @property
    def preconditioner(self):
        """Linear preconditioner: ``"auto"`` (default), ``"fmg"`` or ``"gamg"``.

        ``"auto"`` is GMRES with an additive-Schwarz ILU preconditioner, the
        measured choice for this operator (see :meth:`_set_linear_solver`).
        ``"fmg"`` hands the block to the managed geometric-multigrid route:
        custom-P transfers over the mesh's refinement hierarchy or an adapt
        child's coarse tail, installed on the live PC at the first solve,
        under a flexible GMRES outer solver; without a hierarchy it warns and
        degrades to GAMG. ``"gamg"`` is algebraic multigrid. Setting the
        property rebuilds the solver at the next solve.
        """
        return self._preconditioner

    @preconditioner.setter
    def preconditioner(self, value):
        SNES_Scalar.preconditioner.fset(self, value)
        self._set_linear_solver(multigrid=self._preconditioner != "auto")

    def _object_viewer(self):
        from IPython.display import Latex, display

        super()._object_viewer()
        scheme = {("am", 1): f"Adams-Moulton order 1, theta = {self._theta}",
                  ("bdf", 1): "backward Euler"}.get(
            (self.integrator, self.order),
            f"{self.integrator.upper()} order {self.order}")
        display(Latex(r"$\quad\mathrm{u} = $ " + self.u.sym._repr_latex_()))
        display(Latex(r"$\quad\mathbf{v} = $ " + self.V_fn._repr_latex_()))
        display(Latex(r"$\quad\Delta t = $ " + self.delta_t._repr_latex_()))
        display(Latex(rf"$\quad$ time scheme: {scheme}"))

    # ------------------------------------------------------------------
    # Scheme description
    # ------------------------------------------------------------------

    def _bind_transport_manager(self, manager):
        bind_transport = getattr(manager, "_bind_transport_solver", None)
        if bind_transport is not None:
            bind_transport(self)

    @property
    def DuDt(self):
        """Transport manager bound to this solver's unknown."""
        return self.Unknowns.DuDt

    @DuDt.setter
    def DuDt(self, manager):
        if not isinstance(manager, _DDtBase):
            raise TypeError("DuDt must be a DDt transport manager.")
        if sympy.Matrix(manager.psi_fn).shape != self.u.sym.shape:
            raise ValueError("DuDt tracks a different unknown from u_Field.")
        self._bind_transport_manager(manager)
        self.Unknowns.DuDt = manager
        self._theta = float(getattr(manager, "theta", self._theta))
        self._last_timestep = manager._dt

    @property
    def integrator(self) -> str:
        """The multistep family in use: ``"am"`` (the theta rule) at order 1, ``"bdf"`` above."""
        return self.DuDt.integrator

    @property
    def order(self) -> int:
        """Requested order of the time integration."""
        return self.DuDt.order

    @property
    def theta(self) -> float:
        """Adams-Moulton blend at order 1 (1.0 backward Euler, 0.5 Crank-Nicolson).

        Settable after construction, as on the semi-Lagrangian solver: the
        blend is a runtime constant of the compiled kernels, refreshed from
        the history manager before every solve, so nothing is recompiled.
        """
        return self._theta

    @theta.setter
    def theta(self, value):
        value = float(value)
        if value != 1.0 and self.order != 1:
            raise ValueError(
                "theta applies at order 1 only (0.5 is Crank-Nicolson, 1.0 is "
                "backward Euler); order 2 and 3 take theta=1.0."
            )
        if not hasattr(self.DuDt, "theta"):
            raise AttributeError(f"{type(self.DuDt).__name__} has no theta to set.")
        self.DuDt.theta = value
        self._theta = value

    @property
    def delta_t(self):
        r"""The timestep :math:`\Delta t` as a UW expression.

        Set by :meth:`solve`, or assign it directly (a number or a quantity
        with time units) and call ``solve()`` without ``timestep``, as with
        the semi-Lagrangian solver. A new value updates a runtime constant of
        the compiled kernels; nothing is recompiled.
        """
        return self.DuDt.delta_t

    @delta_t.setter
    def delta_t(self, value):
        dt = float(_nondimensionalise_timestep(value))
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"timestep must be positive, not {dt}.")
        if dt != self._last_timestep:
            self._last_timestep = dt
        self.DuDt._dt = dt

    @property
    def V_fn(self):
        """Advecting velocity, ``(1, dim)`` (the history manager's)."""
        return self.DuDt.V_fn

    @V_fn.setter
    def V_fn(self, value):
        self.DuDt.V_fn = _as_row_vector(value, self.mesh.dim)
        self.is_setup = False

    @property
    def f(self):
        """Volumetric source term."""
        return self._f

    @f.setter
    def f(self, value):
        self._f = sympy.Matrix((value,))
        self._needs_function_rewire = True

    # The stabilisation knobs live on the history manager; these pass through.

    @property
    def peclet_weight(self) -> float:
        """The critical cell Péclet number of the weight (constructor choice; 0 = uniform)."""
        return self.DuDt.peclet_weight

    @property
    def supg_weight(self) -> float:
        """Scale of the SUPG term: 1 (default) or 0 for plain Galerkin. No rebuild."""
        return self.DuDt.supg_weight

    @supg_weight.setter
    def supg_weight(self, value):
        self.DuDt.supg_weight = value

    @property
    def tau_weights(self):
        r"""The weights :math:`(C_t, C_u, C_\kappa)` of the three terms in :math:`\tau`."""
        return self.DuDt.tau_weights

    @tau_weights.setter
    def tau_weights(self, values):
        self.DuDt.tau_weights = values

    # ------------------------------------------------------------------
    # The residual, composed from the history manager's contributions
    # ------------------------------------------------------------------

    def _diffusive_flux(self):
        r"""``(1, dim)`` flux :math:`\sum_k w_k\,\nabla\phi^{(k)}\cdot\kappa` from the constitutive tensor."""
        dim = self.mesh.dim
        c = self.constitutive_model.c
        total = sympy.zeros(1, dim)
        for w, phi in zip(self.DuDt.spatial_weights(), self.DuDt.states()):
            if w == 0:
                continue
            grad = self.mesh.vector.gradient(phi[0])
            total = total + w * (grad * c)
        return total

    def _strong_residual(self):
        """Time derivative, advection and source, as a ``(1, 1)`` matrix."""
        return self.DuDt.time_derivative() + self.DuDt.advection() - self._f

    def _scalar_diffusivity(self):
        kappa = self.constitutive_model.Parameters.diffusivity
        if isinstance(kappa, sympy.MatrixBase):
            raise ValueError(
                "The SUPG parameter needs a scalar diffusivity; anisotropic "
                "diffusion is not supported by this solver."
            )
        return kappa

    def _stabilisation_flux(self):
        if hasattr(self.DuDt, "diffusivity"):
            self.DuDt.diffusivity = self._scalar_diffusivity()
        return self.DuDt.stabilisation_flux(self._strong_residual())

    F0 = Template(
        r"f_0(\phi)",
        lambda self: self._strong_residual(),
        "Strong residual of the time scheme: time derivative, advection and source.",
    )
    F1 = Template(
        r"\mathbf{F}_1(\phi)",
        lambda self: self._diffusive_flux() + self._stabilisation_flux(),
        "Diffusive flux of the time scheme plus the SUPG flux tau R u.",
    )

    # ------------------------------------------------------------------
    # Timestep and solve
    # ------------------------------------------------------------------

    @timing.routine_timer_decorator
    def estimate_dt(self, fraction: float = 0.02, basis: Optional[str] = None,
                    direction_aware: bool = False, percentile: float = 0.0):
        r"""A timestep selected by the transport manager or implicit estimator.

        A rate-based manager supplies its stability estimate. The default
        implicit manager uses the accuracy estimate described below.

        The implicit scheme has no stability limit, so the cell-crossing time
        the semi-Lagrangian solver reports says nothing about how large a step
        this solver can take. What bounds the error is how much the field
        changes per step, and that is what the default estimate measures:

        .. math::
            \Delta t = f\,\frac{\max\phi - \min\phi}
                                {\max\left|\dot\phi\right|}

        with :math:`\dot\phi` the rate of change of the field. Before the
        first solve that rate is the advective one, :math:`|\mathbf{u}\cdot
        \nabla\phi|` at the mesh vertices; after a solve it is the rate the
        last step actually produced, :math:`|\phi^{n+1}-\phi^{n}|/\Delta t`,
        which includes diffusion and sources. The estimate is independent of
        the mesh, so a band of cells refined for another problem does not
        shrink it; it does shrink for a feature that is genuinely
        under-resolved, which is the honest answer.

        On the rotating Gaussian (``docs/developer/design/eulerian-supg-transport.md``)
        ``fraction=0.02`` gives Crank-Nicolson a round-trip error of a few
        tenths of a per cent after one revolution and BDF2 about 1.5%;
        ``fraction=0.07`` gives 2.5% and 9%.

        Parameters
        ----------
        fraction : float, default 0.02
            Allowed change of the field per step as a fraction of its range.
        basis : {None, "accuracy", "resolution", "stability"}
            None selects the manager default; "stability" applies to
            rate-based transport only.
            ``"resolution"`` returns the cell-crossing / diffusion time the
            semi-Lagrangian solver's ``estimate_dt`` returns, for scripts that
            size the step in Courant numbers.
        direction_aware, percentile
            Forwarded to the resolution estimate; ignored otherwise.

        Returns
        -------
        pint.Quantity or float
            With physical time units if a model with reference scales is
            active, otherwise nondimensional. ``inf`` if nothing changes.
        """
        from mpi4py import MPI

        self._bind_transport_manager(self.DuDt)
        estimate_transport = getattr(self.DuDt, "_estimate_transport_dt", None)
        if estimate_transport is not None:
            return estimate_transport(fraction=fraction, basis=basis,
                                      direction_aware=direction_aware, percentile=percentile)
        if basis is None:
            basis = "accuracy"
        if basis == "resolution":
            dt_estimate, dt_adv, dt_diff = _advective_diffusive_dt(
                self.constitutive_model.K, self.V_fn, self.mesh,
                direction_aware=direction_aware, percentile=percentile)
            self.dt_adv = dt_adv if not np.isinf(dt_adv) else 0.0
            self.dt_diff = dt_diff if not np.isinf(dt_diff) else 0.0
            if np.isinf(dt_estimate):
                return np.inf
            return _dimensionalise_dt(dt_estimate)
        if basis != "accuracy":
            raise ValueError(f"basis must be 'accuracy' or 'resolution', not {basis!r}.")

        comm = uw.mpi.comm
        values = np.asarray(self.u.array).reshape(-1)
        lo = comm.allreduce(float(values.min()) if values.size else np.inf, op=MPI.MIN)
        hi = comm.allreduce(float(values.max()) if values.size else -np.inf, op=MPI.MAX)
        field_range = hi - lo

        if self._last_change_rate is not None:
            rate = self._last_change_rate
        else:
            rate = self._advective_rate()
        self.dt_accuracy = fraction * field_range / rate if rate > 0.0 else np.inf
        if np.isinf(self.dt_accuracy) or field_range <= 0.0:
            return np.inf
        return _dimensionalise_dt(self.dt_accuracy)

    def _advective_rate(self):
        r"""Global maximum of :math:`|\mathbf{u}\cdot\nabla\phi|` at the mesh vertices.

        The gradient is the Clement recovery at the vertices (no point
        location, so it is safe on a mesh carrying many variables) and the
        velocity is evaluated at the same points.
        """
        from mpi4py import MPI
        from underworld3.function.gradient_evaluation import compute_clement_gradient_at_nodes

        coords = np.asarray(self.mesh.X.coords)
        n = coords.shape[0]
        if n:
            grad = np.asarray(compute_clement_gradient_at_nodes(self.u), dtype=float).reshape(n, -1)
            vel = uw.function.evaluate(self.V_fn, coords)
            vel = np.asarray(getattr(vel, "magnitude", vel), dtype=float).reshape(n, -1)
            local = float(np.abs((vel[:, :grad.shape[1]] * grad).sum(axis=1)).max())
        else:
            local = 0.0
        return uw.mpi.comm.allreduce(local, op=MPI.MAX)

    @property
    def state(self):
        """Snapshot the timestep and measured change used by estimate_dt."""
        return AdvDiffusionState(
            last_timestep=self._last_timestep,
            last_change_rate=self._last_change_rate,
            order=self.order, theta=self.theta,
        )

    @state.setter
    def state(self, state):
        if not isinstance(state, AdvDiffusionState):
            raise TypeError("AdvDiffusion state has the wrong type.")
        if state._schema_version != AdvDiffusionState._schema_version:
            raise ValueError("AdvDiffusion state schema changed since snapshot.")
        if state.order != self.order:
            raise ValueError("AdvDiffusion order changed since snapshot.")
        if hasattr(self.DuDt, "theta"):
            self.theta = state.theta
        self._last_timestep = None
        if state.last_timestep is not None:
            self.delta_t = state.last_timestep
        self._last_change_rate = state.last_change_rate

    def _prepare_transport_residual(self, verbose=False):
        """Build the solver-owned PDE residual and constrained scalar DM."""
        if not self.constitutive_model._solver_is_setup:
            self._needs_function_rewire = True
        self._build(verbose)
        self.is_setup = True
        self.constitutive_model._solver_is_setup = True

    def _compute_transport_residual(self, solution, residual):
        """Assemble at the current unknown with refreshed auxiliary fields."""
        solution.set(0.0)
        self.dm.localToGlobal(self.u.vec, solution, addv=False)
        residual.set(0.0)
        self.mesh.update_lvec()
        self.dm.setAuxiliaryVec(self.mesh.lvec, None)
        self._update_constants()
        self.snes.computeFunction(solution, residual)

    def solve(
        self,
        zero_init_guess: Optional[bool] = None,
        timestep=None,
        _force_setup: bool = False,
        _evalf: bool = False,
        verbose: bool = False,
        divergence_retries: int = 0,
    ):
        r"""Advance :math:`\phi` by one step.

        Same signature as the semi-Lagrangian solver. ``timestep`` sets
        :attr:`delta_t`; omit it to reuse the value already set. Changing it
        between calls updates a runtime constant of the compiled kernels;
        nothing is recompiled.
        """
        self._bind_transport_manager(self.DuDt)
        if timestep is not None:
            self.delta_t = timestep
        elif self.DuDt._dt is not None:
            self.delta_t = self.DuDt._dt
        elif self._last_timestep is None:
            raise ValueError(
                "solve() needs a timestep: pass timestep=<dt> or set solver.delta_t first."
            )
        dt = self._last_timestep

        if _force_setup:
            self._needs_function_rewire = True
        solve_transport = getattr(self.DuDt, "_solve_transport", None)
        if solve_transport is not None:
            solve_transport(self, dt, zero_init_guess=zero_init_guess,
                            verbose=verbose, divergence_retries=divergence_retries)
            return
        if not self.constitutive_model._solver_is_setup:
            self._needs_function_rewire = True
        # The base ``_build`` resolves the preconditioner choice against the
        # mesh hierarchy before the SNES reads its options. Running the three
        # setup stages directly here (the semi-Lagrangian solvers' pattern)
        # marks the solver set up, so ``_build`` returned early and the
        # geometric-multigrid request was silently inert (#683).
        self._build(verbose)

        self.DuDt.update_pre_solve(dt, verbose=verbose)
        before = np.array(self.u.data).reshape(-1)
        super().solve(zero_init_guess, _force_setup, divergence_retries=divergence_retries)
        _invalidate_solution_cache(self.u)
        # The realised rate of change of the field over this step feeds the
        # accuracy-based estimate_dt (from a copy of the unknown: the manager's
        # history may live on a swarm or carry units).
        from mpi4py import MPI
        change = np.abs(np.asarray(self.u.data).reshape(-1) - before)
        local = float(change.max()) if change.size else 0.0
        self._last_change_rate = uw.mpi.comm.allreduce(local, op=MPI.MAX) / dt
        self.DuDt.update_post_solve(dt, verbose=verbose)

        self.is_setup = True
        self.constitutive_model._solver_is_setup = True
