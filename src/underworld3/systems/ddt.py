r"""
Time derivative approximations for transient problems.

This module provides classes for computing time derivatives (D/Dt) using
various numerical schemes. These are used within solvers to discretize the
time dimension of PDEs.

All DDt classes share a common interface:

- ``update(dt, ...)`` — alias for ``update_pre_solve``
- ``update_pre_solve(dt, ...)`` — called before the PDE solve
- ``update_post_solve(dt, ...)`` — called after the PDE solve
- ``bdf(order)`` — backward differentiation formula (returns Δψ, divide by Δt for rate)
- ``adams_moulton_flux(order)`` — weighted flux for implicit integration

**Symbolic** -- Pure symbolic history, no mesh storage. Used for flux
tracking in SNES_Diffusion where the flux expression is a SymPy tree.

**Eulerian** -- Fixed-grid time derivative with optional grid-based
advection. When ``V_fn`` is provided, ``update_pre_solve`` applies an
explicit advection correction so that ``bdf()`` approximates the full
material derivative.

**SemiLagrangian** -- Characteristic-based D/Dt via departure points.
Traces backward along velocity field to sample upstream values.
Unconditionally stable for advection (no CFL constraint).

**Lagrangian** -- Full particle-following D/Dt. Creates and manages
its own swarm, advected during ``update_post_solve``.

**Lagrangian_Swarm** -- Specialized swarm-based Lagrangian using a
user-provided swarm. Swarm advection is the user's responsibility.

Notes
-----
The choice of time derivative scheme affects accuracy, stability, and
computational cost:

- **Eulerian**: Fixed grid, simple, best for weak-to-moderate advection
  or purely diffusive problems (e.g. Richards equation with no transport).
- **Eulerian (with V_fn)**: Adds explicit advection correction — subject
  to CFL but suitable when velocity is moderate.
- **Semi-Lagrangian**: Best for advection-dominated problems (high Péclet).
  Unconditionally stable but breaks down when v → 0.
- **Lagrangian**: Tracks material properties without numerical diffusion.
  Best for purely advected quantities (e.g. composition).

See Also
--------
underworld3.systems.solvers : PDE solvers using these time derivatives.
"""

import sympy
from sympy import sympify
import numpy as np

from dataclasses import dataclass, field
from typing import Any, Optional, Callable, Union

import underworld3 as uw
from underworld3 import VarType

import underworld3.timing as timing
from underworld3.utilities._api_tools import uw_object
from underworld3.checkpoint.state import SnapshottableState

from petsc4py import PETSc


# ----- Snapshot state dataclasses for DDt flavors -----
#
# Per the design note's "General serialisation contract" section, each
# DDt class exposes a derived State dataclass via ``.state``. The
# private ``_dt_history`` / ``_history_initialised`` / etc. remain the
# authoritative store; the dataclass is built on read and unpacked on
# write. See ``src/underworld3/checkpoint/state.py``.
#
# PR 3 retrofits the Symbolic class. PR 4 will extend the pattern to
# Eulerian, SemiLagrangian, Lagrangian, and Lagrangian_Swarm — each has
# the same dt_history / history_initialised / n_solves_completed / dt
# core plus a flavor-specific psi_star shape.


@dataclass
class _DDtCoreState(SnapshottableState):
    """Common evolution-tracking fields shared by every DDt flavor.

    Each concrete flavor extends this with its own psi_star
    representation (sympy expressions for Symbolic; mesh-variable
    names for Eulerian / SemiLagrangian; swarm-variable names for
    Lagrangian / Lagrangian_Swarm). The actual variable DOF / particle
    data lives in the mesh-variable or swarm-variable path of the
    snapshot — these State dataclasses carry only the metadata needed
    to re-bind on restore.
    """

    _schema_version: int = 1
    dt_history: list = field(default_factory=list)
    history_initialised: bool = False
    n_solves_completed: int = 0
    dt: Any = None


@dataclass
class DDtSymbolicState(_DDtCoreState):
    """Snapshot of a :class:`Symbolic` DDt instance's evolution state.

    ``Symbolic`` is the pure-symbolic flavor — ``psi_star`` history
    slots hold sympy expressions (immutable), captured by value.
    """

    psi_star: list = field(default_factory=list)


@dataclass
class DDtEulerianState(_DDtCoreState):
    """Snapshot of an :class:`Eulerian` DDt instance.

    ``psi_star`` is a list of :class:`MeshVariable` objects; their
    DOF arrays travel via the mesh-variable snapshot path. This State
    only records the variable names for restore-side verification
    that the binding still holds.
    """

    psi_star_var_names: list[str] = field(default_factory=list)


@dataclass
class DDtSemiLagrangianState(_DDtCoreState):
    """Snapshot of a :class:`SemiLagrangian` DDt instance.

    Like :class:`DDtEulerianState`, plus an optional ``forcing_star``
    variable (when ``with_forcing_history=True``) used by ETD-2
    integration of the Maxwell relaxation operator.
    """

    psi_star_var_names: list[str] = field(default_factory=list)
    forcing_star_var_name: Optional[str] = None
    with_forcing_history: bool = False


@dataclass
class DDtLagrangianState(_DDtCoreState):
    """Snapshot of a :class:`Lagrangian` DDt instance.

    ``psi_star`` is a list of :class:`SwarmVariable` objects on this
    DDt's internal swarm. Their data travels via the swarm-variable
    snapshot path.
    """

    psi_star_var_names: list[str] = field(default_factory=list)


@dataclass
class DDtLagrangianSwarmState(_DDtCoreState):
    """Snapshot of a :class:`Lagrangian_Swarm` DDt instance.

    Same shape as :class:`DDtLagrangianState`; the difference is
    operational (Lagrangian creates its own swarm, Lagrangian_Swarm
    uses a user-provided one) rather than state-shaped.
    """

    psi_star_var_names: list[str] = field(default_factory=list)


def _as_float(value):
    """Extract a plain float from various numeric types (Pint, UWQuantity, etc.)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "magnitude"):
        return float(value.magnitude)
    if hasattr(value, "value"):
        return float(value.value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bdf_coefficients(order, dt_current, dt_history):
    r"""Compute BDF coefficients, handling variable timesteps.

    For uniform timesteps the classical BDF-k coefficients are returned
    as exact ``sympy.Rational`` values.  When the timestep ratio differs
    from unity the variable-step BDF-2 formula is used (BDF-3 falls back
    to variable-step BDF-2 when the ratio exceeds 5 %).

    Parameters
    ----------
    order : int
        BDF order (1, 2, or 3).
    dt_current : float or None
        Current timestep :math:`\Delta t_n`.
    dt_history : list
        Previous timesteps ``[dt_{n-1}, dt_{n-2}, ...]``.

    Returns
    -------
    list of sympy.Basic
        Coefficients ``[c0, c1, c2, ...]`` such that

        .. math::

            \Delta\psi_{\mathrm{BDF}} = c_0\,\psi^{n+1}
                + c_1\,\psi^n + c_2\,\psi^{n-1} + \cdots

    Notes
    -----
    **BDF-1** (backward Euler) always returns ``[1, -1]``.

    **BDF-2** with variable step uses
    :math:`r = \Delta t_n / \Delta t_{n-1}`:

    .. math::

        c_0 = \frac{1+2r}{1+r}, \quad
        c_1 = -(1+r), \quad
        c_2 = \frac{r^2}{1+r}

    which reduces to :math:`[3/2,\;-2,\;1/2]` when :math:`r=1`.

    **BDF-3** uses the constant-step formula when all ratios are within
    5 % of unity; otherwise it falls back to variable-step BDF-2.
    """
    if order <= 1:
        return [sympy.Integer(1), sympy.Integer(-1)]

    dt_n = _as_float(dt_current)
    dt_nm1 = (
        _as_float(dt_history[0])
        if len(dt_history) > 0 and dt_history[0] is not None
        else None
    )

    if order == 2:
        if dt_n is not None and dt_nm1 is not None and dt_nm1 > 0:
            r = dt_n / dt_nm1
            if abs(r - 1.0) < 1e-12:
                # Exactly constant dt — use exact rationals
                return [
                    sympy.Rational(3, 2),
                    sympy.Integer(-2),
                    sympy.Rational(1, 2),
                ]
            else:
                # Variable dt
                return [
                    sympy.sympify((1 + 2 * r) / (1 + r)),
                    sympy.sympify(-(1 + r)),
                    sympy.sympify(r**2 / (1 + r)),
                ]
        else:
            # No usable history — constant-dt fallback
            return [
                sympy.Rational(3, 2),
                sympy.Integer(-2),
                sympy.Rational(1, 2),
            ]

    # order >= 3
    dt_nm2 = (
        _as_float(dt_history[1])
        if len(dt_history) > 1 and dt_history[1] is not None
        else None
    )
    if (
        dt_n is not None
        and dt_nm1 is not None
        and dt_nm2 is not None
        and dt_nm1 > 0
        and dt_nm2 > 0
    ):
        r1 = dt_n / dt_nm1
        r2 = dt_nm1 / dt_nm2
        if abs(r1 - 1.0) < 0.05 and abs(r2 - 1.0) < 0.05:
            # Approximately constant dt — standard BDF-3
            return [
                sympy.Rational(11, 6),
                sympy.Integer(-3),
                sympy.Rational(3, 2),
                sympy.Rational(-1, 3),
            ]
        else:
            # Variable dt — fall back to BDF-2 with variable coefficients
            return _bdf_coefficients(2, dt_current, dt_history)
    else:
        # Insufficient history — constant-dt BDF-3
        return [
            sympy.Rational(11, 6),
            sympy.Integer(-3),
            sympy.Rational(3, 2),
            sympy.Rational(-1, 3),
        ]


# ============================================================================
# BDF/AM Coefficient Expressions
# ============================================================================
#
# These helpers create UWexpression coefficient objects and build fixed-structure
# symbolic expressions for bdf() and adams_moulton_flux(). The coefficients are
# routed through PETSc's constants[] array by the JIT compiler, so changing
# effective_order or variable dt only requires PetscDSSetConstants() — no
# recompilation.
# ============================================================================

from underworld3.function.expressions import UWexpression as _UWexpression


def _create_coefficients(order, prefix, instance_id):
    """Create UWexpression objects for BDF or AM coefficients.

    Parameters
    ----------
    order : int
        Maximum order (number of history terms). Creates order+1 coefficients.
    prefix : str
        LaTeX prefix for display (e.g. "c^{BDF}" or "a^{AM}").
    instance_id : int
        Unique ID to disambiguate coefficients from different DDt instances.

    Returns
    -------
    list of UWexpression
        Coefficient expressions initialised to 0.0.
    """
    coeffs = []
    for i in range(order + 1):
        c = _UWexpression(
            rf"{prefix}_{{{i},{instance_id}}}",
            sym=0.0,
            description=f"{prefix} coefficient {i} (DDt instance {instance_id})",
            _unique_name_generation=True,
        )
        coeffs.append(c)
    return coeffs


def _update_bdf_values(coeffs, effective_order, dt, dt_history):
    """Update BDF coefficient UWexpression values for current state.

    Sets active coefficients from _bdf_coefficients() and zeroes the rest.
    """
    values = _bdf_coefficients(effective_order, dt, dt_history)
    for i, v in enumerate(values):
        coeffs[i].sym = float(v)
    for i in range(len(values), len(coeffs)):
        coeffs[i].sym = 0.0


def _update_am_values(coeffs, effective_order, theta=0.5):
    """Update Adams-Moulton coefficient UWexpression values for current state.

    AM coefficients for each order (constant-dt formulas):
    - Order 0: [1]
    - Order 1: [theta, 1-theta]
    - Order 2: [5/12, 8/12, -1/12]
    - Order 3: [9/24, 19/24, -5/24, 1/24]
    """
    if effective_order <= 0:
        values = [1.0]
    elif effective_order == 1:
        values = [float(theta), 1.0 - float(theta)]
    elif effective_order == 2:
        values = [5.0 / 12, 8.0 / 12, -1.0 / 12]
    elif effective_order >= 3:
        values = [9.0 / 24, 19.0 / 24, -5.0 / 24, 1.0 / 24]

    for i, v in enumerate(values):
        coeffs[i].sym = v
    for i in range(len(values), len(coeffs)):
        coeffs[i].sym = 0.0


def _create_exp_coefficients(instance_id):
    """Create UWexpression objects for the ETD-2 coefficients ``[α, φ]``.

    These are named to render as ``α_exp`` and ``φ_exp`` (with the
    DDt instance id appended) so they remain visually distinct from
    BDF/AM coefficients in symbolic output.
    """
    alpha = _UWexpression(
        rf"{{\alpha^{{\mathrm{{exp}}}}_{{[{instance_id}]}}}}",
        sym=0.0,
        description=f"Exp integrator α = exp(-Δt/τ) (DDt instance {instance_id})",
        _unique_name_generation=True,
    )
    phi = _UWexpression(
        rf"{{\varphi^{{\mathrm{{exp}}}}_{{[{instance_id}]}}}}",
        sym=0.0,
        description=f"Exp integrator φ = (1-α)/(Δt/τ) (DDt instance {instance_id})",
        _unique_name_generation=True,
    )
    return [alpha, phi]


def _update_exp_values(coeffs, dt, tau_eff):
    r"""Update exponential-integrator coefficient values for current state.

    Computes :math:`\alpha = e^{-\Delta t/\tau_\mathrm{eff}}` and
    :math:`\varphi = (1-\alpha)/(\Delta t/\tau_\mathrm{eff})` and stores
    them in ``coeffs[0]``, ``coeffs[1]`` respectively. The viscous limit
    (:math:`\Delta t/\tau \to \infty`) gives ``α=0, φ=0``; the elastic
    limit (:math:`\Delta t/\tau \to 0`) gives ``α=1, φ=1``.

    Parameters
    ----------
    coeffs : list of UWexpression
        Two-element list ``[α, φ]`` to update.
    dt : float or None
        Current timestep.
    tau_eff : float or None
        Maxwell relaxation time (η_eff/μ). When None or non-positive,
        defaults to the viscous limit.
    """
    dt_f = _as_float(dt)
    tau_f = _as_float(tau_eff)
    if dt_f is None or tau_f is None or tau_f <= 0.0 or dt_f <= 0.0:
        alpha, phi = 0.0, 0.0  # viscous limit
    else:
        x = dt_f / tau_f
        if x < 1e-12:
            alpha, phi = 1.0, 1.0  # elastic limit
        elif x > 50.0:
            alpha = 0.0
            phi = 1.0 / x  # well-defined small phi, exact for large x
        else:
            alpha = float(np.exp(-x))
            phi = (1.0 - alpha) / x
    coeffs[0].sym = alpha
    coeffs[1].sym = phi


def _build_weighted_sum(coeffs, psi_fn, psi_star_syms):
    """Build a fixed-structure weighted sum: c0*psi + c1*psi_star[0] + ...

    The symbolic structure includes all terms up to len(coeffs)-1.
    Inactive terms have coefficient=0 and vanish numerically.

    Parameters
    ----------
    coeffs : list of UWexpression
        Coefficient expressions (length = order + 1).
    psi_fn : sympy expression
        Current-time field expression.
    psi_star_syms : list of sympy expressions
        History term symbolic expressions (psi_star[i].sym or psi_star[i]).

    Returns
    -------
    sympy expression
        The weighted sum.
    """
    result = coeffs[0] * psi_fn
    for i in range(len(psi_star_syms)):
        if i + 1 < len(coeffs):
            result = result + coeffs[i + 1] * psi_star_syms[i]
    return result


class Symbolic(uw_object):
    r"""
    Symbolic history manager for time derivative approximations.

    Manages the update of a variable :math:`\psi` across timesteps. The history
    operator stores :math:`\psi` over several timesteps (given by ``order``) so
    that it can compute backward differentiation (BDF) or Adams-Moulton expressions.

    The history operator is defined as:

    .. math::

        \psi_p^{t-n\Delta t} &\leftarrow \psi_p^{t-(n-1)\Delta t} \\
        \psi_p^{t-(n-1)\Delta t} &\leftarrow \psi_p^{t-(n-2)\Delta t} \cdots \\
        \psi_p^{t-\Delta t} &\leftarrow \psi_p^{t}

    This is a purely symbolic history manager that operates on sympy expressions
    without mesh or swarm storage. It is useful for tracking symbolic expressions
    through time-stepping algorithms.

    Parameters
    ----------
    psi_fn : sympy.Basic
        The sympy expression to track. Can be scalar or matrix form.
    theta : float, optional
        Implicitness parameter for Adams-Moulton order 1 (default ``0.5``).
        Values: 0 = explicit, 1 = implicit, 0.5 = Crank-Nicolson.
    varsymbol : str, optional
        LaTeX symbol for display (default ``r"\\psi"``).
    verbose : bool, optional
        Enable verbose output (default ``False``).
    bcs : list, optional
        Boundary conditions (default ``[]``).
    order : int, optional
        Order of time integration (1-3) (default ``1``).
    smoothing : float, optional
        Smoothing parameter (default ``0.0``).

    Notes
    -----
    The ``Symbolic`` class is the base for understanding BDF and Adams-Moulton
    formulas without the complexity of mesh or swarm storage. It is primarily
    useful for:

    - Understanding time-stepping algorithm behavior
    - Debugging symbolic expressions in time-dependent problems
    - Prototyping before implementing with mesh/swarm storage

    For actual simulations, use ``Eulerian``, ``SemiLagrangian``, or
    ``Lagrangian`` which store history on computational meshes or swarms.

    See Also
    --------
    Eulerian : Mesh-based history with BDF time-stepping.
    SemiLagrangian : Nodal-swarm approach for advection-dominated problems.
    Lagrangian : Swarm-based material tracking.
    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        psi_fn: sympy.Basic,  # a sympy expression for ψ; can be scalar or matrix
        theta: Optional[float] = 0.5,
        varsymbol: Optional[str] = r"\psi",
        verbose: Optional[bool] = False,
        bcs=[],
        order: int = 1,
        smoothing: float = 0.0,
    ):
        super().__init__()
        self.theta = theta
        self.bcs = bcs
        self.verbose = verbose
        self.smoothing = smoothing
        self.order = order

        # History tracking: deferred initialization and effective order
        self._history_initialised = False
        self._n_solves_completed = 0
        self._dt = None  # current timestep (set by solver or update_pre_solve)
        self._dt_history = [None] * order  # previous timesteps for variable-dt BDF

        # Ensure psi_fn is a sympy Matrix.
        if not isinstance(psi_fn, sympy.Matrix):
            try:
                psi_fn = sympy.Matrix(psi_fn)
            except Exception:
                psi_fn = sympy.Matrix([[psi_fn]])
        self._psi_fn = psi_fn  # stored with its native shape
        self._shape = psi_fn.shape  # capture the shape

        # Set the display symbol for psi_fn and for the history variable.
        self._psi_fn_symbol = varsymbol  # e.g. "\psi"
        self._psi_star_symbol = varsymbol + r"^\ast"  # e.g. "\psi^\ast"

        # Create the history list: each element is a Matrix of shape _shape.
        self.psi_star = [sympy.zeros(*self._shape) for _ in range(order)]

        # BDF/AM coefficient UWexpressions — routed through PetscDS constants[]
        self._bdf_coeffs = _create_coefficients(order, r"c^{\mathrm{BDF}}", self.instance_number)
        self._am_coeffs = _create_coefficients(order, r"a^{\mathrm{AM}}", self.instance_number)
        # ETD-2 (exponential) coefficients [α, φ] for Maxwell-relaxation integration.
        # Treated as a peer to the BDF/AM coefficient sets; values are pushed via
        # PetscDSSetConstants every step in update_exp_coefficients().
        self._exp_coeffs = _create_exp_coefficients(self.instance_number)
        # Initialise to order-1 / viscous values
        _update_bdf_values(self._bdf_coeffs, 1, None, [])
        _update_am_values(self._am_coeffs, 1, self.theta)
        _update_exp_values(self._exp_coeffs, None, None)

        # Register with the active default model as a Snapshottable
        # state-bearer. Safe if no model is active.
        try:
            import underworld3 as _uw

            _uw.get_default_model()._register_state_bearer(self)
        except Exception:
            pass

        return

    # ----- Unitary snapshot / restore -----
    #
    # Option (B)-style adapter per the design note: state is a derived
    # dataclass that surfaces the mutable evolution-tracking attrs.
    # The private ``_dt_history`` / ``_history_initialised`` / etc.
    # remain the authoritative store; the State dataclass is built on
    # read and unpacked on write.
    #
    # See ``docs/developer/design/in_memory_checkpoint_design.md`` and
    # ``src/underworld3/checkpoint/state.py`` for the contract.

    @property
    def state(self) -> "DDtSymbolicState":
        """Return a snapshot-of-state dataclass for this DDt instance."""
        return DDtSymbolicState(
            dt_history=list(self._dt_history),
            history_initialised=bool(self._history_initialised),
            n_solves_completed=int(self._n_solves_completed),
            dt=self._dt,
            psi_star=list(self.psi_star),
        )

    @state.setter
    def state(self, s: "DDtSymbolicState") -> None:
        """Write a captured state back. Reconciles derived coefficients."""
        if s._schema_version != DDtSymbolicState._schema_version:
            raise ValueError(
                f"DDtSymbolicState schema version mismatch: snapshot "
                f"{s._schema_version} vs current "
                f"{DDtSymbolicState._schema_version}"
            )
        if len(s.dt_history) != len(self._dt_history):
            raise ValueError(
                f"dt_history length mismatch ({len(s.dt_history)} vs "
                f"{len(self._dt_history)}); order changed since snapshot?"
            )
        if len(s.psi_star) != len(self.psi_star):
            raise ValueError(
                f"psi_star length mismatch ({len(s.psi_star)} vs "
                f"{len(self.psi_star)}); order changed since snapshot?"
            )
        self._dt_history = list(s.dt_history)
        self._history_initialised = bool(s.history_initialised)
        self._n_solves_completed = int(s.n_solves_completed)
        self._dt = s.dt
        self.psi_star = list(s.psi_star)
        # Re-derive BDF/AM coefficients so downstream reads see values
        # consistent with the restored primary state without waiting
        # for the next update_pre_solve.
        _update_bdf_values(
            self._bdf_coeffs, self.effective_order, self._dt, self._dt_history
        )
        _update_am_values(self._am_coeffs, self.effective_order, self.theta)

    @property
    def psi_fn(self):
        r"""Current symbolic expression :math:`\psi` being tracked."""
        return self._psi_fn

    @psi_fn.setter
    def psi_fn(self, new_fn):
        """Set the tracked symbolic expression."""
        if not isinstance(new_fn, sympy.Matrix):
            try:
                new_fn = sympy.Matrix(new_fn)
            except Exception:
                new_fn = sympy.Matrix([[new_fn]])
        # Optionally, one could check for matching shape; here we update both.
        self._psi_fn = new_fn
        self._shape = new_fn.shape
        return

    def _object_viewer(self):
        from IPython.display import Latex, display

        # Display the primary variable
        display(Latex(rf"$\quad {self._psi_fn_symbol} = {sympy.latex(self._psi_fn)}$"))
        # Display the history variable using the different symbol.
        history_latex = ", ".join([sympy.latex(elem) for elem in self.psi_star])
        display(Latex(rf"$\quad {self._psi_star_symbol} = \left[{history_latex}\right]$"))

    @property
    def effective_order(self):
        """Current effective BDF order, accounting for history startup.

        For BDF order k, k distinct history values are needed. During
        startup, ``effective_order`` ramps from 1 to ``self.order`` as
        successive solves populate the history slots with distinct values.
        """
        # BDF-k requires k completed solves to have k distinct history values.
        # With 0 or 1 completed solves → order 1. Order 2 needs ≥2 solves.
        return min(self.order, max(1, self._n_solves_completed))

    def update_history_fn(self):
        r"""Copy current :math:`\psi` to the first history slot ``psi_star[0]``."""
        # Update the first history element with a copy of the current ψ.
        self.psi_star[0] = self.psi_fn.copy()

    def initialise_history(self):
        r"""Initialize all history slots to the current value of :math:`\psi`.

        Called automatically on the first ``update_pre_solve``. Can also
        be called manually after setting initial conditions.
        """
        self.update_history_fn()
        # Propagate the initial history to all history steps.
        for i in range(1, self.order):
            self.psi_star[i] = self.psi_star[0].copy()
        self._history_initialised = True
        return

    def initiate_history_fn(self):
        """Deprecated: use ``initialise_history`` instead."""
        self.initialise_history()

    def update(
        self,
        dt,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        """Update history (alias for ``update_pre_solve``)."""
        self.update_pre_solve(dt, evalf, verbose)
        return

    def update_pre_solve(
        self,
        dt,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        """Pre-solve update hook. Auto-initialises history on first call."""
        self._dt = dt

        if not self._history_initialised:
            self.initialise_history()

        # Update coefficient values for current effective_order and dt
        _update_bdf_values(self._bdf_coeffs, self.effective_order, self._dt, self._dt_history)
        _update_am_values(self._am_coeffs, self.effective_order, self.theta)

        return

    def update_post_solve(
        self,
        dt,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        r"""Shift history chain after solve: :math:`\psi^{*n} \leftarrow \psi^{*(n-1)}`."""
        self._dt = dt

        if verbose:
            print(f"Updating history for ψ = {self.psi_fn}", flush=True)

        # Record timestep history for variable-dt BDF
        for i in range(self.order - 1, 0, -1):
            self._dt_history[i] = self._dt_history[i - 1]
        self._dt_history[0] = dt

        # Shift history: copy each element down the chain.
        for i in range(self.order - 1, 0, -1):
            self.psi_star[i] = self.psi_star[i - 1].copy()
        self.update_history_fn()

        if self._n_solves_completed < self.order:
            self._n_solves_completed += 1

        return

    @property
    def bdf_coefficients(self):
        """Current BDF coefficients [c0, c1, ...] accounting for variable timesteps."""
        return _bdf_coefficients(self.effective_order, self._dt, self._dt_history)

    def bdf(self, order: Optional[int] = None):
        r"""Backward differentiation approximation of the time-derivative of ψ.

        Returns a fixed-structure symbolic expression using UWexpression
        coefficients. The coefficient values are updated each step in
        ``update_pre_solve`` — no JIT recompilation needed when the
        order ramps up or the timestep changes.

        Parameters
        ----------
        order : int, optional
            Ignored (kept for API compatibility). The effective order is
            controlled by the coefficient values.
        """
        return _build_weighted_sum(self._bdf_coeffs, self.psi_fn, self.psi_star)

    def adams_moulton_flux(self, order: Optional[int] = None):
        r"""Adams-Moulton flux approximation for implicit time integration.

        Returns a fixed-structure symbolic expression using UWexpression
        coefficients. Values are updated each step in ``update_pre_solve``.

        Parameters
        ----------
        order : int, optional
            Ignored (kept for API compatibility).
        """
        return _build_weighted_sum(self._am_coeffs, self.psi_fn, self.psi_star)

    def update_exp_coefficients(self, dt, tau_eff):
        r"""Update the ETD-2 (exponential) coefficient values for this step.

        Sets ``self._exp_coeffs[0].sym = α`` and ``self._exp_coeffs[1].sym = φ``
        from current ``dt`` and ``tau_eff`` (Maxwell relaxation time
        :math:`\tau = \eta_\mathrm{eff}/\mu`). Called by the constitutive
        model (which owns τ_eff) before each solve, peer to the BDF/AM
        coefficient updates that happen automatically in
        ``update_pre_solve``.
        """
        _update_exp_values(self._exp_coeffs, dt, tau_eff)


class Eulerian(uw_object):
    r"""
    Eulerian (mesh-based) history manager for time derivatives.

    Manages the update of a variable :math:`\psi` on the mesh across timesteps,
    storing history values on mesh variables for backward differentiation.

    .. math::

        \psi_p^{t-n\Delta t} &\leftarrow \psi_p^{t-(n-1)\Delta t} \\
        \psi_p^{t-(n-1)\Delta t} &\leftarrow \psi_p^{t-(n-2)\Delta t} \cdots \\
        \psi_p^{t-\Delta t} &\leftarrow \psi_p^{t}

    When ``V_fn`` is provided, the ``update_pre_solve`` method applies an
    explicit advection correction so that ``bdf()`` approximates the full
    material derivative :math:`D\psi/Dt = \partial\psi/\partial t + \mathbf{u}\cdot\nabla\psi`
    rather than the partial time derivative alone.

    .. note::
        The optional advection capability (V_fn parameter) is suitable for
        problems where the advection is weak or where a purely grid-based
        approach is desired (e.g., Richards equation with no transport).
        For advection-dominated problems, SemiLagrangian is more mature and
        generally preferred.

    Parameters
    ----------
    mesh : underworld3.discretisation.Mesh
        The computational mesh.
    psi_fn : MeshVariable or sympy.Basic
        The quantity to track. Can be a mesh variable or symbolic expression.
    vtype : VarType
        Variable type (SCALAR, VECTOR, etc.) for history storage.
    degree : int
        Polynomial degree for history mesh variables.
    continuous : bool
        Whether history variables are continuous across element boundaries.
    V_fn : sympy.Basic, optional
        Velocity field for grid-based advection correction.
        If None (default), computes pure ∂ψ/∂t. If set, computes
        D/Dt = ∂/∂t + u·∇ via operator splitting.
    evalf : bool, default=False
        If True, evaluate expressions numerically during updates.
    theta : float, default=0.5
        Time-stepping parameter for implicit/explicit blending.
        theta=0 is fully explicit, theta=1 is fully implicit.
    varsymbol : str, default=r"u"
        LaTeX symbol for display.
    verbose : bool, default=False
        Enable verbose output during updates.
    bcs : list, default=[]
        Boundary conditions to apply to projections.
    order : int, default=1
        Number of history timesteps to store (for multi-step methods).
    smoothing : float, default=0.0
        Smoothing parameter for projections.

    See Also
    --------
    SemiLagrangian : For advection-dominated problems with nodal swarm.
    Lagrangian : For full Lagrangian tracking on swarms.
    Symbolic : For purely symbolic history (no mesh storage).
    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        psi_fn: Union[
            uw.discretisation.MeshVariable, sympy.Basic
        ],  # sympy function or mesh variable
        vtype: uw.VarType,
        degree: int,
        continuous: bool,
        V_fn=None,
        evalf: Optional[bool] = False,
        theta: Optional[float] = 0.5,
        varsymbol: Optional[str] = r"u",
        verbose: Optional[bool] = False,
        bcs=[],
        order=1,
        smoothing=0.0,
    ):
        super().__init__()

        self.mesh = mesh
        self.V_fn = V_fn
        self.theta = theta
        self.bcs = bcs
        self.verbose = verbose
        self.degree = degree
        self.vtype = vtype
        self.continuous = continuous
        self.smoothing = smoothing
        self.evalf = evalf

        # History tracking: deferred initialization and effective order
        self._history_initialised = False
        self._n_solves_completed = 0

        # meshVariables are required for:
        #
        # u(t) - evaluation of u_fn at the current time
        # u*(t) - u_* evaluated from

        # psi is evaluated/stored at `order` timesteps. We can't
        # be sure if psi is a meshVariable or a function to be evaluated
        # psi_star is reaching back through each evaluation and has to be a
        # meshVariable (storage)

        if isinstance(psi_fn, uw.discretisation.MeshVariable):
            self._psi_fn = psi_fn.sym  ### get symbolic form of the meshvariable
            self._psi_meshVar = psi_fn
        else:
            self._psi_fn = psi_fn  ### already in symbolic form
            self._psi_meshVar = None

        self.order = order
        self._dt = None  # current timestep (set by solver or update_pre_solve)
        self._dt_history = [None] * order  # previous timesteps for variable-dt BDF

        psi_star = []
        self.psi_star = psi_star

        for i in range(order):
            self.psi_star.append(
                uw.discretisation.MeshVariable(
                    f"psi_star_Eulerian_{self.instance_number}_{i}",
                    self.mesh,
                    vtype=vtype,
                    degree=degree,
                    continuous=continuous,
                    varsymbol=rf"{varsymbol}^{{ {'*'*(i+1)} }}",
                )
            )

        # BDF/AM coefficient UWexpressions — routed through PetscDS constants[]
        self._bdf_coeffs = _create_coefficients(order, r"c^{\mathrm{BDF}}", self.instance_number)
        self._am_coeffs = _create_coefficients(order, r"a^{\mathrm{AM}}", self.instance_number)
        self._exp_coeffs = _create_exp_coefficients(self.instance_number)
        # Initialise to order-1 / viscous values
        _update_bdf_values(self._bdf_coeffs, 1, None, [])
        _update_am_values(self._am_coeffs, 1, self.theta)
        _update_exp_values(self._exp_coeffs, None, None)

        try:
            import underworld3 as _uw

            _uw.get_default_model()._register_state_bearer(self)
        except Exception:
            pass

        return

    @property
    def state(self) -> "DDtEulerianState":
        """Return a snapshot-of-state dataclass for this Eulerian DDt."""
        return DDtEulerianState(
            dt_history=list(self._dt_history),
            history_initialised=bool(self._history_initialised),
            n_solves_completed=int(self._n_solves_completed),
            dt=self._dt,
            psi_star_var_names=[ps.clean_name for ps in self.psi_star],
        )

    @state.setter
    def state(self, s: "DDtEulerianState") -> None:
        if s._schema_version != DDtEulerianState._schema_version:
            raise ValueError(
                f"DDtEulerianState schema version mismatch: snapshot "
                f"{s._schema_version} vs current "
                f"{DDtEulerianState._schema_version}"
            )
        if len(s.dt_history) != len(self._dt_history):
            raise ValueError(
                f"dt_history length mismatch ({len(s.dt_history)} vs "
                f"{len(self._dt_history)}); order changed since snapshot?"
            )
        current_names = [ps.clean_name for ps in self.psi_star]
        if s.psi_star_var_names and s.psi_star_var_names != current_names:
            raise ValueError(
                f"psi_star variable names changed since snapshot: "
                f"{s.psi_star_var_names} vs {current_names}"
            )
        self._dt_history = list(s.dt_history)
        self._history_initialised = bool(s.history_initialised)
        self._n_solves_completed = int(s.n_solves_completed)
        self._dt = s.dt
        _update_bdf_values(
            self._bdf_coeffs, self.effective_order, self._dt, self._dt_history
        )
        _update_am_values(self._am_coeffs, self.effective_order, self.theta)

    @property
    def psi_fn(self):
        r"""Current symbolic expression :math:`\psi` being tracked."""
        return self._psi_fn

    @psi_fn.setter
    def psi_fn(self, new_fn):
        """Set the tracked expression."""
        self._psi_fn = new_fn
        # self._psi_star_projection_solver.uw_function = self.psi_fn
        return

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        ## feedback on this instance
        # display(Latex(r"$\quad\psi = $ " + self.psi._repr_latex_()))
        # display(
        #     Latex(
        #         r"$\quad\Delta t_{\textrm{phys}} = $ "
        #         + sympy.sympify(self.dt_physical)._repr_latex_()
        #     )
        # )
        display(Latex(rf"$\quad$History steps = {self.order}"))

    def _setup_projections(self):
        """Initialize projection solvers for history updates."""
        ### using this to store terms that can't be evaluated (e.g. derivatives)
        # The projection operator for mapping derivative values to the mesh - needs to be different for each variable type, unfortunately ...
        if self.vtype == uw.VarType.SCALAR:
            self._psi_star_projection_solver = uw.systems.solvers.SNES_Projection(
                self.mesh, self.psi_star[0], verbose=False
            )
        elif self.vtype == uw.VarType.VECTOR:
            self._psi_star_projection_solver = uw.systems.solvers.SNES_Vector_Projection(
                self.mesh, self.psi_star[0], verbose=False
            )
        elif self.vtype == uw.VarType.SYM_TENSOR or self.vtype == uw.VarType.TENSOR:
            import math
            dim = self.mesh.dim
            if self.vtype == uw.VarType.SYM_TENSOR:
                Nc = math.comb(dim + 1, 2)  # 3 in 2D, 6 in 3D
                self._psi_star_indep_indices = [
                    (i, j) for i in range(dim) for j in range(i, dim)
                ]
            else:
                Nc = dim * dim
                self._psi_star_indep_indices = [
                    (i, j) for i in range(dim) for j in range(dim)
                ]

            self._psi_star_flat_var = uw.discretisation.MeshVariable(
                f"psi_star_flat_{self.instance_number}",
                self.mesh,
                (1, Nc),
                vtype=uw.VarType.MATRIX,
                degree=self.degree,
                continuous=self.continuous,
                varsymbol=r"{\psi^{*}_{\mathrm{flat}}}",
            )
            self._psi_star_projection_solver = uw.systems.solvers.SNES_MultiComponent_Projection(
                self.mesh,
                u_Field=self._psi_star_flat_var,
                n_components=Nc,
                degree=self.degree,
                verbose=False,
            )
            self._psi_star_use_multicomponent = True

        if getattr(self, '_psi_star_use_multicomponent', False):
            # Flatten tensor to (1, Nc) row for multicomponent solver
            import sympy
            indep = self._psi_star_indep_indices
            row = sympy.Matrix([[self.psi_fn[i, j] for (i, j) in indep]])
            self._psi_star_projection_solver.uw_function = row
        else:
            self._psi_star_projection_solver.uw_function = self.psi_fn
        self._psi_star_projection_solver.bcs = self.bcs
        self._psi_star_projection_solver.smoothing = self.smoothing

    @property
    def effective_order(self):
        """Current effective BDF order, accounting for history startup.

        For BDF order k, k distinct history values are needed. During
        startup, ``effective_order`` ramps from 1 to ``self.order`` as
        successive solves populate the history slots with distinct values.
        """
        # BDF-k requires k completed solves to have k distinct history values.
        # With 0 or 1 completed solves → order 1. Order 2 needs ≥2 solves.
        return min(self.order, max(1, self._n_solves_completed))

    def update_history_fn(self):
        r"""Copy current :math:`\psi` to ``psi_star[0]`` via evaluation or projection."""
        ### update first value in history chain
        ### avoids projecting if function can be evaluated
        try:
            try:
                self.psi_star[0].data[...] = self._psi_meshVar.data[...]
            except:
                self.psi_star[0].data[...] = uw.function.evaluate(
                    self.psi_fn,
                    self.psi_star[0].coords,
                    evalf=self.evalf,
                ).reshape(-1, max(self.psi_fn.shape))

        except:
            self._setup_projections()
            self._psi_star_projection_solver.solve()

    def initialise_history(self):
        r"""Initialize all history slots to the current value of :math:`\psi`.

        Called automatically on the first ``update_pre_solve``. Can also
        be called manually after setting initial conditions to ensure
        the history chain starts from the correct state.
        """
        self.update_history_fn()

        ### set up all history terms to the initial values
        for i in range(self.order - 1, 0, -1):
            self.psi_star[i].data[...] = self.psi_star[0].data[...]

        self._history_initialised = True
        return

    def set_initial_history(self, values, dt=None):
        r"""Plant history values for BDF restart or analytical IC.

        Bypasses the automatic ``effective_order`` ramp so the very
        first solve runs at the full BDF order rather than starting at
        BDF-1. Use this when you have known values at :math:`t` and
        past times — e.g. an analytical periodic solution, or a
        checkpointed history loaded from disk.

        Parameters
        ----------
        values : sequence of length ``self.order``
            ``values[k]`` is :math:`\psi` at :math:`t - k\,\Delta t`,
            i.e. ``values[0]`` is the current state. Each entry must
            be assignable to ``psi_star[k].array`` — either an array
            of matching shape, or a scalar that broadcasts.
        dt : float, optional
            Uniform timestep assumed between history slots. Required
            for ``order >= 2`` to seed correct multistep coefficients
            on the first solve. Ignored for ``order = 1``.
        """
        if len(values) != self.order:
            raise ValueError(
                f"set_initial_history requires {self.order} value(s) "
                f"(one per history slot, including the current state); "
                f"got {len(values)}."
            )
        for k, val in enumerate(values):
            self.psi_star[k].array[...] = val
        self._history_initialised = True
        self._n_solves_completed = self.order
        if dt is not None:
            self._dt_history = [float(dt)] * self.order
        elif self.order >= 2:
            import warnings
            warnings.warn(
                "set_initial_history called with order >= 2 but no "
                "dt — variable-dt BDF coefficients will be wrong on "
                "the first solve. Pass dt=<timestep> to suppress.",
                stacklevel=2,
            )
        return

    def initiate_history_fn(self):
        """Deprecated: use ``initialise_history`` instead."""
        self.initialise_history()

    def update(
        self,
        dt,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        """Update history (alias for ``update_pre_solve``)."""
        self.update_pre_solve(dt, evalf, verbose)
        return

    def update_pre_solve(
        self,
        dt,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        """Pre-solve: auto-initialise history and apply advection correction.

        On the first call, automatically initialises history from the
        current field values. If V_fn is set, also applies an explicit
        grid-based advection correction so that bdf() approximates the
        material derivative Dφ/Dt rather than ∂φ/∂t.
        """
        self._dt = dt

        if not self._history_initialised:
            self.initialise_history()

        # Update coefficient values for current effective_order and dt
        _update_bdf_values(self._bdf_coeffs, self.effective_order, self._dt, self._dt_history)
        _update_am_values(self._am_coeffs, self.effective_order, self.theta)

        if self.V_fn is not None and dt is not None:
            coords = self.psi_star[0].coords
            dim = self.mesh.dim
            X = self.mesh.X

            # Build u·∇φ symbolically for each component of psi_fn
            # psi_fn is a Matrix; V_fn is also a Matrix. For scalar
            # psi_fn the shape is (1,1); for vector it is (1,dim).
            psi = self.psi_fn
            V = self.V_fn
            ncomp = max(psi.shape)  # number of tracked components

            for c in range(ncomp):
                # ∂φ_c/∂x_i for each spatial dimension
                grad_c = sympy.Matrix([psi[c].diff(X[i]) for i in range(dim)])
                # u·∇φ_c = V_i * ∂φ_c/∂x_i
                advection_expr = sum(V[i] * grad_c[i] for i in range(dim))

                advection_vals = uw.function.evaluate(
                    advection_expr, coords, evalf=evalf,
                ).reshape(-1)

                self.psi_star[0].data[:, c] -= dt * advection_vals

        return

    def update_post_solve(
        self,
        dt,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        r"""Shift history chain after solve: :math:`\psi^{*n} \leftarrow \psi^{*(n-1)}`."""
        self._dt = dt

        if verbose and uw.mpi.rank == 0:
            print(f"Update {self.psi_fn}", flush=True)

        # Record timestep history for variable-dt BDF
        for i in range(self.order - 1, 0, -1):
            self._dt_history[i] = self._dt_history[i - 1]
        self._dt_history[0] = dt

        ### copy values down the chain
        for i in range(self.order - 1, 0, -1):
            self.psi_star[i].data[...] = self.psi_star[i - 1].data[...]

        ### update the history fn
        self.update_history_fn()

        if self._n_solves_completed < self.order:
            self._n_solves_completed += 1

        return

    @property
    def bdf_coefficients(self):
        """Current BDF coefficients [c0, c1, ...] accounting for variable timesteps."""
        return _bdf_coefficients(self.effective_order, self._dt, self._dt_history)

    def bdf(self, order=None):
        r"""Backward differentiation approximation of the time-derivative of :math:`\psi`.

        Returns a fixed-structure symbolic expression using UWexpression
        coefficients. Values are updated each step in ``update_pre_solve``.

        Parameters
        ----------
        order : int, optional
            Ignored (kept for API compatibility).
        """
        return _build_weighted_sum(self._bdf_coeffs, self.psi_fn, [ps.sym for ps in self.psi_star])

    def adams_moulton_flux(self, order=None):
        r"""Adams-Moulton flux approximation for implicit time integration.

        Returns a fixed-structure symbolic expression using UWexpression
        coefficients. Values are updated each step in ``update_pre_solve``.

        Parameters
        ----------
        order : int, optional
            Ignored (kept for API compatibility).
        """
        return _build_weighted_sum(self._am_coeffs, self.psi_fn, [ps.sym for ps in self.psi_star])

    def update_exp_coefficients(self, dt, tau_eff):
        r"""Update the ETD-2 (exponential) coefficient values for this step."""
        _update_exp_values(self._exp_coeffs, dt, tau_eff)


class SemiLagrangian(uw_object):
    r"""
    Semi-Lagrangian history manager using nodal swarm.

    Manages the semi-Lagrangian update of a mesh variable :math:`\psi`
    across timesteps. Uses a nodal swarm to track departure points and
    interpolate values back to the mesh.

    .. math::

        \psi_p^{t-n\Delta t} &\leftarrow \psi_p^{t-(n-1)\Delta t} \\
        \psi_p^{t-(n-1)\Delta t} &\leftarrow \psi_p^{t-(n-2)\Delta t} \cdots \\
        \psi_p^{t-\Delta t} &\leftarrow \psi_p^{t}

    The semi-Lagrangian method traces characteristics backward in time
    to find departure points, providing stable advection without CFL
    restrictions while maintaining accuracy.

    Parameters
    ----------
    mesh : underworld3.discretisation.Mesh
        The computational mesh.
    psi_fn : sympy.Function
        The quantity to advect (typically a mesh variable's symbolic form).
    V_fn : sympy.Function
        Velocity field for advection (e.g., ``stokes.u.sym``).
    vtype : VarType
        Variable type (SCALAR, VECTOR, SYM_TENSOR, etc.).
    degree : int
        Polynomial degree for mesh variable storage.
    continuous : bool
        Whether variables are continuous across element boundaries.
    swarm_degree : int, optional
        Polynomial degree for swarm interpolation. Defaults to ``degree``.
    swarm_continuous : bool, optional
        Continuity for swarm variables. Defaults to ``continuous``.
    varsymbol : str, optional
        LaTeX symbol for display.
    verbose : bool, default=False
        Enable verbose output during updates.
    bcs : list, default=[]
        Boundary conditions for projections.
    order : int, default=1
        Number of history timesteps (1 for first-order, 2 for second-order).
    smoothing : float, default=0.0
        Smoothing parameter for projections.
    preserve_moments : bool, default=False
        Use moment-preserving projection (experimental).
    with_forcing_history : bool, default=False
        When True, allocate an additional ``forcing_star`` MeshVariable
        (matching ``psi_star[0]``'s shape, vtype, degree, continuity) to
        store one history slot for the strain-rate forcing. Required by
        ETD-2 exponential integration of the Maxwell relaxation operator;
        ignored for BDF/AM. Populated each step via
        :meth:`update_forcing_history` (direct nodal evaluation of
        ``forcing_fn`` — typically the constitutive model's strain-rate
        symbol).

    Notes
    -----
    The semi-Lagrangian method is particularly useful for:

    - Advection-dominated problems (high Péclet number)
    - Problems where CFL stability is restrictive
    - Viscoelastic stress advection

    See Also
    --------
    Eulerian : For fixed-mesh time derivatives without advection.
    Lagrangian : For full particle-following Lagrangian tracking.
    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        psi_fn: sympy.Function,
        V_fn: sympy.Function,
        vtype: uw.VarType,
        degree: int,
        continuous: bool,
        swarm_degree: Optional[int] = None,
        swarm_continuous: Optional[bool] = None,
        varsymbol: Optional[str] = None,
        verbose: Optional[bool] = False,
        bcs=[],
        order=1,
        smoothing=0.0,
        preserve_moments=False,
        with_forcing_history: bool = False,
    ):
        super().__init__()

        self.mesh = mesh
        self.bcs = bcs
        self.verbose = verbose
        self.degree = degree
        self.continuous = continuous
        self._psi_fn = psi_fn
        self.V_fn = V_fn
        self.order = order
        self.preserve_moments = preserve_moments
        self.with_forcing_history = with_forcing_history

        # Forcing-history storage. Allocated only if requested. Populated
        # each step via update_forcing_history(forcing_fn) — used by ETD-2
        # exponential integration of the Maxwell relaxation operator to
        # supply the ε̇ⁿ history term in the constitutive flux.
        self.forcing_star = None
        self._forcing_fn = None  # set by the constitutive model
        self._forcing_vtype = None
        self._forcing_indep_indices = None

        # History tracking: deferred initialization and effective order
        self._history_initialised = False
        self._n_solves_completed = 0
        self._dt = None  # current timestep (set by solver or update_pre_solve)
        self._dt_history = [None] * order  # previous timesteps for variable-dt BDF

        # Source snapshot machinery (opt-in via enable_source_snapshot()).
        # Used when psi_fn references psi_star[0] itself (e.g. VE/VEP stress
        # history where flux = 2·viscosity·E_eff and E_eff contains psi_star[0]
        # via its history term). Without a snapshot the projection becomes
        # implicit in psi_star[0] and Min-mode at yield admits the wrong fixed
        # point. With snapshot, psi_star[0] symbols in the source are
        # substituted with a frozen snapshot variable that's refreshed each
        # step from psi_star[0]'s data array. The projection becomes a true
        # one-shot Galerkin projection.
        self._psi_snapshot_enabled = False
        self._psi_snapshot = None

        if swarm_degree is None:
            self.swarm_degree = degree
        else:
            self.swarm_degree = swarm_degree

        if swarm_continuous is None:
            self.swarm_continuous = continuous
        else:
            self.swarm_continuous = swarm_continuous

        if varsymbol is None:
            varsymbol = rf"u_{{ [{self.instance_number}] }}"

        # meshVariables are required for:
        #
        # u(t) - evaluation of u_fn at the current time
        # u*(t) - u_* evaluated from

        # psi is evaluated/stored at `order` timesteps. We can't
        # be sure if psi is a meshVariable or a function to be evaluated
        # but psi_star is reaching back through each evaluation and has to be a
        # meshVariable (storage)

        psi_star = []
        self.psi_star = psi_star

        # Propagate units from psi_fn to psi_star if the model supports units.
        # Internal psi_star variables should match the user's variable units when possible,
        # but if no reference quantities are set, use unitless variables to avoid strict mode errors.
        psi_units = uw.get_units(psi_fn)

        # Check if the model can handle units (has reference quantities set)
        model = uw.get_default_model()
        if psi_units is not None and not model.has_units():
            # Model doesn't have reference quantities - don't propagate units to internal vars
            psi_units = None

        for i in range(order):
            self.psi_star.append(
                uw.discretisation.MeshVariable(
                    f"psi_star_sl_{self.instance_number}_{i}",
                    self.mesh,
                    vtype=vtype,
                    degree=self.degree,
                    continuous=self.continuous,
                    varsymbol=rf"{{ {varsymbol}^{{ {'*'*(i+1)} }} }}",
                    units=psi_units,  # Inherit units from psi_fn (or None if model has no units)
                )
            )

        # Forcing-history slot (only allocated when ETD-2 / exponential
        # integration is engaged). Mirrors psi_star[0] in shape/vtype/
        # discretisation; populated each step in update_forcing_history()
        # via direct nodal evaluation of forcing_fn (typically the model's
        # strain-rate symbol).
        #
        # Units: deliberately ``units=None``. The forcing field is the
        # strain rate (1/time), distinct from psi_star's stress units
        # (Pa·s × ε̇ = Pa). We don't know strain-rate units at construction
        # time (forcing_fn is supplied later by the constitutive model).
        # ``update_forcing_history`` non-dimensionalises the evaluated
        # forcing before storing, matching the codebase convention that
        # variable storage holds non-dimensional values internally and
        # units are re-attached at the .data interface.
        self._forcing_vtype = vtype
        if with_forcing_history:
            self.forcing_star = uw.discretisation.MeshVariable(
                f"forcing_star_sl_{self.instance_number}",
                self.mesh,
                vtype=vtype,
                degree=self.degree,
                continuous=self.continuous,
                varsymbol=rf"{{ {varsymbol}_{{F}}^{{ * }} }}",
                units=None,
            )

        # BDF/AM/exp coefficient UWexpressions — routed through PetscDS constants[]
        self._bdf_coeffs = _create_coefficients(order, r"c^{\mathrm{BDF}}", self.instance_number)
        self._am_coeffs = _create_coefficients(order, r"a^{\mathrm{AM}}", self.instance_number)
        self._exp_coeffs = _create_exp_coefficients(self.instance_number)
        # Initialise to order-1 / viscous values
        _update_bdf_values(self._bdf_coeffs, 1, None, [])
        _update_am_values(self._am_coeffs, 1, 0.5)
        _update_exp_values(self._exp_coeffs, None, None)

        # Working variable that has a potentially different discretisation from psi_star
        # We project from this to psi_star and we use this variable to define the
        # advection sample points

        self._workVar = uw.discretisation.MeshVariable(
            f"W_{self.instance_number}_{i}",
            self.mesh,
            vtype=vtype,
            degree=self.swarm_degree,
            continuous=self.swarm_continuous,
            varsymbol=rf"{{ {varsymbol}^\nabla }}",
            units=psi_units,  # Inherit units from psi_fn
        )

        # We just need one swarm since this is inherently a sequential operation
        nswarm = uw.swarm.NodalPointSwarm(self._workVar, verbose)
        self._nswarm_psi = nswarm

        # The projection operator for mapping swarm values to the mesh - needs to be different for
        # each variable type, unfortunately ...

        if vtype == uw.VarType.SCALAR:
            self._psi_star_projection_solver = uw.systems.solvers.SNES_Projection(
                self.mesh, self.psi_star[0], verbose=False
            )
        elif vtype == uw.VarType.VECTOR:
            self._psi_star_projection_solver = uw.systems.solvers.SNES_Vector_Projection(
                self.mesh,
                self.psi_star[0],
                verbose=False,
            )

        elif vtype == uw.VarType.SYM_TENSOR or vtype == uw.VarType.TENSOR:
            import math
            dim = self.mesh.dim
            if vtype == uw.VarType.SYM_TENSOR:
                Nc = math.comb(dim + 1, 2)
                self._psi_star_indep_indices = [
                    (i, j) for i in range(dim) for j in range(i, dim)
                ]
            else:
                Nc = dim * dim
                self._psi_star_indep_indices = [
                    (i, j) for i in range(dim) for j in range(dim)
                ]

            self._psi_star_flat_var = uw.discretisation.MeshVariable(
                f"psi_star_flat_slcn_{self.instance_number}",
                self.mesh,
                (1, Nc),
                vtype=uw.VarType.MATRIX,
                degree=degree,
                continuous=continuous,
                varsymbol=r"{\psi^{*}_{\mathrm{flat}}}",
            )
            self._psi_star_projection_solver = uw.systems.solvers.SNES_MultiComponent_Projection(
                self.mesh,
                u_Field=self._psi_star_flat_var,
                n_components=Nc,
                degree=degree,
                verbose=False,
            )
            self._psi_star_use_multicomponent = True

        # We should find a way to add natural bcs here
        # (self.Unknowns.u carried as a symbol from solver to solver)

        if getattr(self, '_psi_star_use_multicomponent', False):
            import sympy
            indep = self._psi_star_indep_indices
            fn = self._workVar.sym
            row = sympy.Matrix([[fn[i, j] for (i, j) in indep]])
            self._psi_star_projection_solver.uw_function = row
        else:
            self._psi_star_projection_solver.uw_function = self._workVar.sym
        self._psi_star_projection_solver.bcs = bcs
        self._psi_star_projection_solver.smoothing = smoothing

        self._smoothing = smoothing

        self.I = uw.maths.Integral(mesh, None)

        try:
            import underworld3 as _uw

            _uw.get_default_model()._register_state_bearer(self)
        except Exception:
            pass

        return

    @property
    def state(self) -> "DDtSemiLagrangianState":
        return DDtSemiLagrangianState(
            dt_history=list(self._dt_history),
            history_initialised=bool(self._history_initialised),
            n_solves_completed=int(self._n_solves_completed),
            dt=self._dt,
            psi_star_var_names=[ps.clean_name for ps in self.psi_star],
            forcing_star_var_name=(
                self.forcing_star.clean_name
                if self.forcing_star is not None else None
            ),
            with_forcing_history=bool(self.with_forcing_history),
        )

    @state.setter
    def state(self, s: "DDtSemiLagrangianState") -> None:
        if s._schema_version != DDtSemiLagrangianState._schema_version:
            raise ValueError(
                f"DDtSemiLagrangianState schema version mismatch: snapshot "
                f"{s._schema_version} vs current "
                f"{DDtSemiLagrangianState._schema_version}"
            )
        if len(s.dt_history) != len(self._dt_history):
            raise ValueError(
                f"dt_history length mismatch ({len(s.dt_history)} vs "
                f"{len(self._dt_history)}); order changed since snapshot?"
            )
        current_names = [ps.clean_name for ps in self.psi_star]
        if s.psi_star_var_names and s.psi_star_var_names != current_names:
            raise ValueError(
                f"psi_star variable names changed since snapshot: "
                f"{s.psi_star_var_names} vs {current_names}"
            )
        if s.with_forcing_history != bool(self.with_forcing_history):
            raise ValueError(
                f"with_forcing_history flag differs between snapshot "
                f"({s.with_forcing_history}) and current "
                f"({self.with_forcing_history})"
            )
        self._dt_history = list(s.dt_history)
        self._history_initialised = bool(s.history_initialised)
        self._n_solves_completed = int(s.n_solves_completed)
        self._dt = s.dt
        _update_bdf_values(
            self._bdf_coeffs, self.effective_order, self._dt, self._dt_history
        )
        # SemiLagrangian's update_pre_solve uses theta=0.5 directly
        # (it doesn't take a theta argument in __init__), so the setter
        # matches that.
        _update_am_values(self._am_coeffs, self.effective_order, 0.5)

    @property
    def psi_fn(self):
        r"""Current symbolic expression :math:`\psi` being tracked."""
        return self._psi_fn

    @psi_fn.setter
    def psi_fn(self, new_fn):
        """Set the tracked expression and propagate to the projection's source.

        When :meth:`enable_source_snapshot` has been called, ``psi_star[0]``
        symbols in ``new_fn`` are transparently substituted with the
        snapshot variable's symbols before the source is pushed to the
        projection solver — so the projection becomes a true one-shot
        Galerkin projection regardless of whether ``new_fn`` references
        ``psi_star[0]``.
        """
        self._psi_fn = new_fn
        self._psi_star_projection_solver.uw_function = self._build_projection_source(new_fn)
        return

    def _build_projection_source(self, source_fn):
        """Construct the row matrix used as the projection's ``uw_function``.

        Applies snapshot substitution (psi_star[0] → snap) when enabled.
        Used by both ``psi_fn.setter`` and the ``initialise_history``
        fallback path so substitution semantics are consistent.
        """
        if getattr(self, '_psi_star_use_multicomponent', False):
            import sympy
            indep = self._psi_star_indep_indices
            row = sympy.Matrix([[source_fn[i, j] for (i, j) in indep]])
            if self._psi_snapshot_enabled and self._psi_snapshot is not None:
                ps0 = self.psi_star[0]
                psi_snapshot = self._psi_snapshot
                substitutions = {
                    ps0.sym[i, j]: psi_snapshot.sym[i, j]
                    for i in range(self.mesh.dim)
                    for j in range(self.mesh.dim)
                }
                row = row.subs(substitutions)
            return row
        else:
            # Scalar / vector path: psi_star[0] is a scalar/vector field. If
            # snapshot is needed for these vtypes, extend here similarly.
            return source_fn

    def enable_source_snapshot(self):
        """Enable snapshot substitution in the projection's source field.

        Call this once when the source expression (``psi_fn``) references
        ``psi_star[0]`` itself — without it the projection's residual
        ``(target − flux(psi_star[0]))·weight`` is implicit in the target
        because target and source share the same data field. With Min-mode
        plasticity at the yield kink, the implicit projection admits two
        fixed points (elastic and yield branches); under timestep change the
        iteration drifts to the elastic-branch fixed point and σ violates
        the yield surface.

        The snapshot is a separate mesh variable matching ``psi_star[0]``'s
        shape/vtype/degree. Each call to ``update_pre_solve`` copies
        ``psi_star[0].array → psi_snapshot.array``, freezing the source's
        input for the upcoming projection. Substitution makes the
        projection's compiled C code read from ``psi_snapshot.array``
        instead of ``psi_star[0].array`` — there's no recompile per step,
        just a memcpy.

        Idempotent: safe to call more than once.
        """
        if not getattr(self, '_psi_star_use_multicomponent', False):
            # Currently only wired for tensor projections (the case that
            # exposed the bug).  Scalar/vector extension is straightforward
            # if needed later.
            return

        if self._psi_snapshot is None:
            ps0 = self.psi_star[0]
            # NOTE: this currently registers a persistent MeshVariable in the
            # mesh DM, which is overkill for a transient buffer that's only
            # read by this DDt's projection.  A future improvement would be
            # a transient/scratch-variable mechanism (likely backed by
            # PETSc's auxiliary Vec machinery — already used elsewhere in
            # the codebase via DMSetAuxiliaryVec_UW) so the snapshot doesn't
            # accumulate in the DM across DDt creations.  See:
            # docs/developer/ai-notes/historical-notes.md for the
            # variable-deletion limitation context.
            self._psi_snapshot = uw.discretisation.MeshVariable(
                f"psi_snapshot_{self.instance_number}",
                self.mesh,
                ps0.shape,
                vtype=ps0.vtype,
                degree=ps0.degree,
                continuous=ps0.continuous,
            )
            # Initialise psi_snapshot's data to current psi_star[0]'s data
            # so the source evaluates consistently before the first refresh.
            self._psi_snapshot.data[...] = ps0.data[...]

        self._psi_snapshot_enabled = True

        # Re-run the psi_fn setter so the substitution is applied to the
        # currently-installed projection source.
        self.psi_fn = self._psi_fn

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        display(Latex(rf"$\quad$History steps = {self.order}"))

    @property
    def effective_order(self):
        """Current effective BDF order, accounting for history startup.

        For BDF order k, k distinct history values are needed. During
        startup, ``effective_order`` ramps from 1 to ``self.order`` as
        successive solves populate the history slots with distinct values.
        """
        # BDF-k requires k completed solves to have k distinct history values.
        # With 0 or 1 completed solves → order 1. Order 2 needs ≥2 solves.
        return min(self.order, max(1, self._n_solves_completed))

    def initialise_history(self):
        r"""Initialize all history slots to the current value of :math:`\psi`.

        Called automatically on the first ``update_pre_solve``. Can also
        be called manually after setting initial conditions.
        """
        # Evaluate psi_fn at psi_star node positions and store in psi_star[0]
        from underworld3.utilities.unit_aware_array import UnitAwareArray

        coords = self.psi_star[0].coords
        if hasattr(coords, "magnitude"):
            coords_nd = uw.non_dimensionalise(coords)
            if isinstance(coords_nd, UnitAwareArray):
                coords_nd = np.array(coords_nd)
            elif hasattr(coords_nd, 'magnitude'):
                coords_nd = coords_nd.magnitude
        else:
            coords_nd = coords

        try:
            eval_result = uw.function.evaluate(self.psi_fn, coords_nd)
            psi_units = self.psi_star[0].units
            if psi_units is not None and not isinstance(eval_result, UnitAwareArray):
                eval_result = UnitAwareArray(eval_result, units=psi_units)
            self.psi_star[0].array[...] = eval_result
        except Exception:
            # Fallback: project psi_fn onto psi_star[0] via the SNES projector.
            # Route through the shared builder so snapshot substitution
            # semantics are consistent.
            self._psi_star_projection_solver.uw_function = self._build_projection_source(self.psi_fn)
            self._psi_star_projection_solver.smoothing = 0.0
            self._psi_star_projection_solver.solve()
            if getattr(self, '_psi_star_use_multicomponent', False):
                # Fan out flat result to tensor psi_star[0]
                for k, (i, j) in enumerate(self._psi_star_indep_indices):
                    vals = self._psi_star_flat_var.array[:, 0, k]
                    self.psi_star[0].array[:, i, j] = vals
                    if i != j:
                        self.psi_star[0].array[:, j, i] = vals

        # Copy to all other history slots
        for i in range(1, self.order):
            self.psi_star[i].array[...] = self.psi_star[0].array[...]

        self._history_initialised = True
        return

    def set_initial_history(self, values, dt=None):
        r"""Plant history values for BDF restart or analytical IC.

        Bypasses the automatic ``effective_order`` ramp so the very
        first solve runs at the full BDF order rather than starting
        at BDF-1. Use this when you have known values at :math:`t`
        and past times — e.g. an analytical periodic solution, or a
        checkpointed history loaded from disk.

        Parameters
        ----------
        values : sequence of length ``self.order``
            ``values[k]`` is :math:`\psi` at :math:`t - k\,\Delta t`,
            i.e. ``values[0]`` is the current state. Each entry must
            be assignable to ``psi_star[k].array`` — either an array
            of matching shape, or a scalar that broadcasts.
        dt : float, optional
            Uniform timestep assumed between history slots. Required
            for ``order >= 2`` to seed correct multistep coefficients
            on the first solve. Ignored for ``order = 1``.
        """
        if len(values) != self.order:
            raise ValueError(
                f"set_initial_history requires {self.order} value(s) "
                f"(one per history slot, including the current state); "
                f"got {len(values)}."
            )
        for k, val in enumerate(values):
            self.psi_star[k].array[...] = val
        self._history_initialised = True
        self._n_solves_completed = self.order
        if dt is not None:
            self._dt_history = [float(dt)] * self.order
        elif self.order >= 2:
            import warnings
            warnings.warn(
                "set_initial_history called with order >= 2 but no "
                "dt — variable-dt BDF coefficients will be wrong on "
                "the first solve. Pass dt=<timestep> to suppress.",
                stacklevel=2,
            )
        return

    def initiate_history_fn(self):
        """Deprecated: use ``initialise_history`` instead."""
        self.initialise_history()

    def update(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
        dt_physical: Optional = None,
    ):
        """Update history (alias for ``update_pre_solve``)."""
        self.update_pre_solve(dt, evalf, verbose, dt_physical)
        return

    def update_post_solve(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
        dt_physical: Optional[float] = None,
    ):
        """Post-solve: record timestep and increment solve counter."""
        self._dt = dt

        # Record timestep history for variable-dt BDF
        for i in range(self.order - 1, 0, -1):
            self._dt_history[i] = self._dt_history[i - 1]
        self._dt_history[0] = dt

        if self._n_solves_completed < self.order:
            self._n_solves_completed += 1

        return

    def update_pre_solve(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
        dt_physical: Optional[float] = None,
        store_result: Optional[bool] = True,
    ):
        """Sample upstream values along characteristics before solve.

        On the first call, automatically initialises history from the
        current field values so that bdf() returns zero on the first step.

        Parameters
        ----------
        store_result : bool, optional
            If True (default), evaluate psi_fn at current positions and store
            in psi_star[0] before advection — the standard DDt behaviour.
            If False, skip this step and the history shift: only advect the
            existing psi_star levels upstream. Used by VE_Stokes where
            psi_star[0] already contains the projected actual stress from
            the previous solve.
        """

        self._dt = dt

        if not self._history_initialised:
            self.initialise_history()

        # Refresh the source-snapshot variable so the projection's source
        # field captures psi_star[0]'s state from BEFORE this step's solve.
        # Per-step memcpy keeps the snapshot machinery aligned with
        # psi_star[0] without recompiling the projection.  Routes through
        # ``.data`` rather than ``.array`` to skip unit conversion (both
        # variables already live in non-dimensional space) while keeping
        # the callback sync that pushes values into the underlying PETSc
        # local Vec.
        if self._psi_snapshot_enabled and self._psi_snapshot is not None:
            self._psi_snapshot.data[...] = self.psi_star[0].data[...]

        # Update coefficient values for current effective_order and dt
        _update_bdf_values(self._bdf_coeffs, self.effective_order, self._dt, self._dt_history)
        _update_am_values(self._am_coeffs, self.effective_order, 0.5)

        ## Progress from the oldest part of the history
        # 1. Copy the stored values down the chain in preparation for the next timestep
        #    The history term is the nodel value of psi_fn offset back along the characteristics
        #    according to the timestep.
        #    That is:
        #
        #      - psi_star[0] is the current value of psi_fn, sampled
        #        at the location of the nodes in their previous position at t-\Delta t
        #
        #      - psi_star[1] is the value of psi_star[0] from the previous timestep
        #        sampled at the location of the nodes at t - \Delta t. (note this is approximately
        #        equivalent to the value of psi_star[0] at t - 2\Delta t)
        #
        #      - psi_star[2] etc if required ...
        #
        #    First we copy the history, then we sample can sample upstream values

        if dt_physical is not None:
            phi = sympy.Min(1, dt / dt_physical)
        else:
            phi = sympy.sympify(1)

        if store_result:
            for i in range(self.order - 1, 0, -1):
                self.psi_star[i].array[...] = (
                    phi * self.psi_star[i - 1].array[...] + (1 - phi) * self.psi_star[i].array[...]
                )

        # 2. Compute the current value of psi_fn which we store in psi_star[0]
        #    Note the need to do a try/except to handle unsupported evaluations
        #    (e.g. of derivatives)
        #
        #    When store_result=False (e.g. VE stress history), skip this step —
        #    psi_star[0] already contains the projected actual stress from
        #    the previous solve and we want to advect *that*, not the flux.

        # CRITICAL FIX (2025-11-28): Handle coordinates correctly for unit-aware mode.
        # Previous bug: extracting .magnitude gives METERS (e.g., 1000000), but:
        # - mesh.get_closest_cells() expects [0-1] non-dimensional coords
        # - evaluate() assumes plain numpy is [0-1] non-dimensional
        # Solution: use uw.non_dimensionalise() for proper conversion, OR pass
        # unit-aware coords to evaluate() which handles conversion internally.
        from underworld3.utilities.unit_aware_array import UnitAwareArray

        psi_star_0_coords = self.psi_star[0].coords

        # For mesh internal operations, need non-dimensional [0-1] coordinates
        if hasattr(psi_star_0_coords, "magnitude"):
            # Unit-aware coords - need to non-dimensionalize (not just extract magnitude!)
            psi_star_0_coords_nd = uw.non_dimensionalise(psi_star_0_coords)
            # Extract to plain numpy for mesh operations
            if isinstance(psi_star_0_coords_nd, UnitAwareArray):
                psi_star_0_coords_nd = np.array(psi_star_0_coords_nd)
            elif hasattr(psi_star_0_coords_nd, 'magnitude'):
                psi_star_0_coords_nd = psi_star_0_coords_nd.magnitude
            else:
                psi_star_0_coords_nd = np.array(psi_star_0_coords_nd)
        else:
            # Plain numpy - assume already non-dimensional
            psi_star_0_coords_nd = psi_star_0_coords

        cellid = self.mesh.get_closest_cells(
            psi_star_0_coords_nd,
        )

        # Move slightly within the chosen cell to avoid edge effects
        centroid_coords = self.mesh._centroids[cellid]

        shift = 0.001
        node_coords_nd = (1.0 - shift) * psi_star_0_coords_nd[:, :] + shift * centroid_coords[
            :, :
        ]

        if store_result:
            try:
                # Use shifted ND coords to avoid quad mesh boundary issues
                # node_coords_nd is slightly shifted toward cell centroids
                # evaluate() treats plain numpy as ND [0-1] coordinates
                eval_result = uw.function.evaluate(
                    self.psi_fn,
                    node_coords_nd,
                    evalf=evalf,
                )
                # Wrap result with units if psi_star has units but eval didn't return UnitAwareArray
                psi_star_units = self.psi_star[0].units
                if psi_star_units is not None and not isinstance(eval_result, UnitAwareArray):
                    eval_result = UnitAwareArray(eval_result, units=psi_star_units)

                self.psi_star[0].array[...] = eval_result

            except Exception:
                # Fallback to projection solver for expressions that can't be directly evaluated
                # (e.g., containing derivatives)
                self._psi_star_projection_solver.uw_function = self.psi_fn
                self._psi_star_projection_solver.smoothing = 0.0
                self._psi_star_projection_solver.solve(verbose=verbose)

        # 3. Compute the upstream values from the psi_fn

        # We use the u_star variable as a working value here so we have to work backwards
        # so we don't over-write the history terms
        #

        # Convert dt to model units for numerical arithmetic
        # (after symbolic logic that may use dt with units)
        # Note: uw is already imported at module level (line 7)
        model = uw.get_default_model()

        # DIAGNOSTIC: Capture information about the unit system
        coords_template = self.psi_star[0].coords
        has_units = hasattr(coords_template, "magnitude") or hasattr(coords_template, "_magnitude")

        # Maintain unit system consistency: either keep everything with units or convert to non-dimensional
        if has_units:
            # Physical coordinate system with units
            # dt must be converted to base SI seconds so that dt * velocity(m/s) = distance(m)
            if hasattr(dt, "to"):  # It's a Pint quantity
                dt_for_calc = dt.to("second")  # Convert to seconds (still a quantity)
            else:
                # If dt is already a dimensionless number, treat it as seconds
                dt_for_calc = dt
        else:
            # Non-dimensional coordinate system - convert dt to non-dimensional
            # CRITICAL: Actually non-dimensionalize the timestep!
            if hasattr(dt, "magnitude") or hasattr(dt, "value"):
                # dt has units - non-dimensionalize it
                dt_nondim = uw.non_dimensionalise(dt, model)
                # Extract the dimensionless value
                if hasattr(dt_nondim, "magnitude"):
                    dt_for_calc = float(dt_nondim.magnitude)
                elif hasattr(dt_nondim, "value"):
                    dt_for_calc = float(dt_nondim.value)
                else:
                    dt_for_calc = float(dt_nondim)
            else:
                # Already dimensionless
                dt_for_calc = dt

        for i in range(self.order - 1, -1, -1):
            # 2nd order update along characteristics

            # Use shifted ND coords to avoid quad mesh boundary issues
            # node_coords_nd is slightly shifted toward cell centroids (lines 703-709)
            # evaluate() treats plain numpy as ND [0-1] coordinates
            v_result = uw.function.evaluate(
                self.V_fn,
                node_coords_nd,
            )

            # CRITICAL: Preserve UnitAwareArray through slicing
            # Slicing can sometimes return plain numpy views - need to preserve wrapper
            from underworld3.utilities.unit_aware_array import UnitAwareArray

            if isinstance(v_result, UnitAwareArray):
                # Slice and rewrap to preserve units
                v_at_node_pts = v_result[:, 0, :]
                if not isinstance(v_at_node_pts, UnitAwareArray):
                    # Slicing lost the wrapper - rewrap it
                    v_at_node_pts = UnitAwareArray(v_at_node_pts, units=v_result.units)
            else:
                v_at_node_pts = v_result[:, 0, :]

            # Non-dimensionalize velocities when working with dimensionless coordinates
            # This prevents dimensional mismatch: velocities in m/s mixed with coords in [0,1]
            # CRITICAL: evaluate now returns UnitAwareArray with units attached
            # Check if velocities already have units before trying to add them manually
            if not has_units:
                # Coordinates are dimensionless - need to non-dimensionalize velocities too
                if isinstance(v_at_node_pts, UnitAwareArray):
                    # Velocities already have units from evaluate - just non-dimensionalize
                    v_nondim = uw.non_dimensionalise(v_at_node_pts, model)
                    # Extract numpy array for dimensionless calculation
                    if isinstance(v_nondim, UnitAwareArray):
                        v_at_node_pts = np.array(v_nondim)
                    elif hasattr(v_nondim, "value"):
                        v_at_node_pts = v_nondim.value
                    else:
                        v_at_node_pts = v_nondim
                else:
                    # Velocities don't have units - try to add them manually (legacy path)
                    v_units = uw.get_units(self.V_fn)
                    if v_units and v_units != "dimensionless":
                        v_with_units = UnitAwareArray(v_at_node_pts, units=v_units)
                        v_nondim = uw.non_dimensionalise(v_with_units, model)
                        if isinstance(v_nondim, UnitAwareArray):
                            v_at_node_pts = np.array(v_nondim)
                        elif hasattr(v_nondim, "value"):
                            v_at_node_pts = v_nondim.value
                        else:
                            v_at_node_pts = v_nondim
            else:
                # Dimensional mode - ensure velocities have units
                # CRITICAL FIX (2025-11-27): Variable data is stored NON-DIMENSIONALLY.
                # We must DIMENSIONALIZE (not just wrap) the values before dimensional arithmetic.
                # Previous bug: wrapping 0.01 (ND) with cm/yr gave 0.01 cm/yr instead of 1 cm/yr.
                if not isinstance(v_at_node_pts, UnitAwareArray):
                    v_units = uw.get_units(self.V_fn)
                    if v_units and v_units != "dimensionless":
                        # Re-dimensionalize using the scaling system
                        if uw.is_nondimensional_scaling_active():
                            from underworld3.scaling import dimensionalise
                            # dimensionalise(nd_value, units) -> value * scale in those units
                            v_dimensional = dimensionalise(v_at_node_pts, v_units)
                            v_at_node_pts = UnitAwareArray(v_dimensional.magnitude, units=v_dimensional.units)
                        else:
                            # No scaling active - assume values are already dimensional
                            v_at_node_pts = UnitAwareArray(v_at_node_pts, units=v_units)

            # Get coordinates
            coords = self.psi_star[i].coords

            # CRITICAL: When working in dimensionless mode, extract coords to plain arrays
            # to match the dimensionless velocities (otherwise unit mismatch occurs)
            from underworld3.utilities.unit_aware_array import UnitAwareArray

            if not has_units and isinstance(coords, UnitAwareArray):
                # Extract to plain numpy for dimensionless arithmetic
                coords = np.array(coords)

            # CRITICAL (2025-11-27): Multiply velocity FIRST so UnitAwareArray.__mul__ handles it.
            # If we do `dt_for_calc * v_at_node_pts`, Pint handles it and loses UnitAwareArray units.
            mid_pt_coords = coords - v_at_node_pts * (0.5 * dt_for_calc)

            # Clamp midpoint coordinates to the domain boundary
            if self.mesh.return_coords_to_bounds is not None:
                mid_pt_coords = self.mesh.return_coords_to_bounds(mid_pt_coords)

            v_mid_result = uw.function.global_evaluate(
                self.V_fn,
                mid_pt_coords,
            )

            # CRITICAL: Preserve UnitAwareArray through slicing
            if isinstance(v_mid_result, UnitAwareArray):
                # Slice and rewrap to preserve units
                v_at_mid_pts = v_mid_result[:, 0, :]
                if not isinstance(v_at_mid_pts, UnitAwareArray):
                    # Slicing lost the wrapper - rewrap it
                    v_at_mid_pts = UnitAwareArray(v_at_mid_pts, units=v_mid_result.units)
            else:
                v_at_mid_pts = v_mid_result[:, 0, :]

            # Non-dimensionalize mid-point velocities when working with dimensionless coordinates
            # CRITICAL: global_evaluate now returns UnitAwareArray with units attached
            # Check if velocities already have units before trying to add them manually
            if not has_units:
                # Coordinates are dimensionless - need to non-dimensionalize velocities too
                if isinstance(v_at_mid_pts, UnitAwareArray):
                    # Velocities already have units from global_evaluate - just non-dimensionalize
                    v_nondim = uw.non_dimensionalise(v_at_mid_pts, model)
                    # Extract numpy array for dimensionless calculation
                    if isinstance(v_nondim, UnitAwareArray):
                        v_at_mid_pts = np.array(v_nondim)
                    elif hasattr(v_nondim, "value"):
                        v_at_mid_pts = v_nondim.value
                    else:
                        v_at_mid_pts = v_nondim
                else:
                    # Velocities don't have units - try to add them manually (legacy path)
                    v_units = uw.get_units(self.V_fn)
                    if v_units and v_units != "dimensionless":
                        v_with_units = UnitAwareArray(v_at_mid_pts, units=v_units)
                        v_nondim = uw.non_dimensionalise(v_with_units, model)
                        if isinstance(v_nondim, UnitAwareArray):
                            v_at_mid_pts = np.array(v_nondim)
                        elif hasattr(v_nondim, "value"):
                            v_at_mid_pts = v_nondim.value
                        else:
                            v_at_mid_pts = v_nondim
            else:
                # Dimensional mode - ensure velocities have units
                # CRITICAL: If V_fn doesn't have unit metadata, evaluate() returns plain numpy
                # We need to manually wrap it with units for dimensional arithmetic to work
                if not isinstance(v_at_mid_pts, UnitAwareArray):
                    v_units = uw.get_units(self.V_fn)
                    if v_units and v_units != "dimensionless":
                        # Wrap velocities with their proper units
                        v_at_mid_pts = UnitAwareArray(v_at_mid_pts, units=v_units)

            # Calculate upstream coordinates: current position - velocity * timestep
            end_pt_coords = coords - v_at_mid_pts * dt_for_calc

            # Clamp upstream coordinates to the domain boundary
            if self.mesh.return_coords_to_bounds is not None:
                end_pt_coords = self.mesh.return_coords_to_bounds(end_pt_coords)

            # Extract scalar from (1,1) Matrix for scalar variables
            # MeshVariable.sym returns Matrix([[value]]) for scalars
            expr_to_evaluate = self.psi_star[i].sym
            if hasattr(expr_to_evaluate, 'shape') and expr_to_evaluate.shape == (1, 1):
                expr_to_evaluate = expr_to_evaluate[0, 0]

            # Evaluate psi_star at upstream coordinates
            # global_evaluate now returns dimensional results (gateway fix 2025-11-28)
            value_at_end_points = uw.function.global_evaluate(
                expr_to_evaluate,
                end_pt_coords,
            )

            # CRITICAL FIX (2025-11-27): If psi_star has units, ensure the assigned
            # value also has units. global_evaluate may return plain arrays.
            psi_star_units = self.psi_star[i].units
            if psi_star_units is not None and not isinstance(value_at_end_points, UnitAwareArray):
                value_at_end_points = UnitAwareArray(value_at_end_points, units=psi_star_units)

            self.psi_star[i].array[...] = value_at_end_points

            # disable this for now - Compute moments before update
            if 0 and self.preserve_moments and self._workVar.num_components == 1:

                self.I.fn = self.psi_star[i].sym[0]
                Imean0 = self.I.evaluate()

                self.I.fn = (self.psi_star[i].sym[0] - Imean0) ** 2
                IL20 = np.sqrt(self.I.evaluate())


            # disable this for now - Restore moments after update
            if 0 and self.preserve_moments and self._workVar.num_components == 1:

                self.I.fn = self.psi_star[i].sym[0]
                Imean = self.I.evaluate()

                self.I.fn = (self.psi_star[i].sym[0] - Imean) ** 2
                IL2 = np.sqrt(self.I.evaluate())

                # TODO: DELETE remove swarm.access / data, replace with direct array assignment
                # with self.mesh.access(self.psi_star[i]):
                #     self.psi_star[i].data[...] += Imean0 - Imean

                self.psi_star[i].array[...] += Imean0 - Imean

                self.I.fn = (self.psi_star[i].sym[0] - Imean0) ** 2
                IL2 = np.sqrt(self.I.evaluate())

                # TODO: DELETE remove swarm.access / data, replace with direct array assignment
                # with self.mesh.access(self.psi_star[i]):
                #     self.psi_star[i].data[...] = (
                #         self.psi_star[i].data[...] - Imean0
                #     ) * IL20 / IL2 + Imean0

                self.psi_star[i].array[...] = (
                    self.psi_star[i].array[...] - Imean0
                ) * IL20 / IL2 + Imean0

        return

    @property
    def bdf_coefficients(self):
        """Current BDF coefficients [c0, c1, ...] accounting for variable timesteps."""
        return _bdf_coefficients(self.effective_order, self._dt, self._dt_history)

    def bdf(self, order=None):
        r"""Backward differentiation approximation of the time-derivative of :math:`\psi`.

        Returns a fixed-structure symbolic expression using UWexpression
        coefficients. Values are updated each step in ``update_pre_solve``.

        Parameters
        ----------
        order : int, optional
            Ignored (kept for API compatibility).
        """
        return _build_weighted_sum(self._bdf_coeffs, self.psi_fn, [ps.sym for ps in self.psi_star])

    def adams_moulton_flux(self, order=None):
        r"""Adams-Moulton flux approximation for implicit time integration.

        Returns a fixed-structure symbolic expression using UWexpression
        coefficients. Values are updated each step in ``update_pre_solve``.

        Parameters
        ----------
        order : int, optional
            Ignored (kept for API compatibility).
        """
        return _build_weighted_sum(self._am_coeffs, self.psi_fn, [ps.sym for ps in self.psi_star])

    def update_exp_coefficients(self, dt, tau_eff):
        r"""Update the scalar ETD-2 (exponential) coefficient UWexpressions.

        Sets ``self._exp_coeffs[0].sym = α = exp(-Δt/τ_eff)`` and
        ``self._exp_coeffs[1].sym = φ = (1-α)/(Δt/τ_eff)`` so the next solve
        uses the correct exponential coefficients via PetscDSSetConstants
        on the next ``_update_constants`` call.
        """
        _update_exp_values(self._exp_coeffs, dt, tau_eff)

    @property
    def _exp_alpha(self):
        """Convenience accessor for the ETD-2 ``α`` coefficient UWexpression."""
        return self._exp_coeffs[0]

    @property
    def _exp_phi(self):
        """Convenience accessor for the ETD-2 ``φ`` coefficient UWexpression."""
        return self._exp_coeffs[1]

    def update_forcing_history(self, forcing_fn=None, evalf=False, verbose=False):
        r"""Refresh ``forcing_star`` from ``forcing_fn`` via direct nodal evaluation.

        Used by ETD-2 exponential integration to store the current
        strain-rate field as :math:`\dot\varepsilon^{n}` for the next
        step's history term. Called by the constitutive model from the
        solver's post-solve hook. Direct nodal evaluation (rather than an
        L2 projection through SNES) is sufficient because the strain rate
        is :math:`\nabla\mathbf{u}`, well-defined at nodes, with no
        history-coupled term that would make a projection implicit.

        Unit handling
        -------------
        ``forcing_star`` is allocated with ``units=None`` (see
        ``__init__``). When the model is unit-aware, ``forcing_fn`` is a
        symbolic expression of the velocity field whose evaluation
        returns a ``UnitAwareArray`` carrying strain-rate units (1/time).
        We non-dimensionalise that result via the active scaling system
        before assigning to ``forcing_star.array``, which keeps the
        stored values consistent with the rest of the variable storage
        (codebase convention: variable storage is non-dimensional;
        units are re-attached at the ``.data`` interface). When the
        model is not unit-aware, the evaluation returns a plain ndarray
        and assignment is a straight numpy copy.

        Parameters
        ----------
        forcing_fn : sympy expression, optional
            Symbolic strain-rate field to evaluate at each node. If
            None, falls back to ``self._forcing_fn`` (set by the
            constitutive model at solve-attach time). No-op if neither
            is set or ``with_forcing_history=False``.
        evalf : bool, optional
            Forwarded to ``uw.function.evaluate`` (forces numerical
            evaluation when True).
        verbose : bool, optional
            Enable verbose output.
        """
        if not self.with_forcing_history or self.forcing_star is None:
            return
        if forcing_fn is None:
            forcing_fn = self._forcing_fn
        if forcing_fn is None:
            return  # constitutive model hasn't wired the forcing source yet

        from underworld3.utilities.unit_aware_array import UnitAwareArray

        coords = self.forcing_star.coords
        # Use non-dimensional coords for evaluate() (mirrors the psi_star
        # path in update_pre_solve)
        if hasattr(coords, "magnitude"):
            coords_nd = uw.non_dimensionalise(coords)
            if isinstance(coords_nd, UnitAwareArray):
                coords_nd = np.array(coords_nd)
            elif hasattr(coords_nd, 'magnitude'):
                coords_nd = coords_nd.magnitude
        else:
            coords_nd = coords

        def _eval_nd(component_expr):
            """Evaluate component at coords and non-dimensionalise to a
            plain 1-D float array suitable for nodal storage."""
            result = uw.function.evaluate(component_expr, coords_nd, evalf=evalf)
            # If the evaluation returned units (model is unit-aware),
            # non-dimensionalise before storing — keeps forcing_star's
            # internal storage non-dimensional like psi_star.
            if isinstance(result, UnitAwareArray) or hasattr(result, "magnitude"):
                result = uw.non_dimensionalise(result)
                if isinstance(result, UnitAwareArray):
                    result = np.array(result)
                elif hasattr(result, "magnitude"):
                    result = result.magnitude
            return np.asarray(result).flatten()

        vtype = self._forcing_vtype
        if vtype == uw.VarType.SYM_TENSOR or vtype == uw.VarType.TENSOR:
            dim = self.mesh.dim
            indep = (
                [(i, j) for i in range(dim) for j in range(i, dim)]
                if vtype == uw.VarType.SYM_TENSOR
                else [(i, j) for i in range(dim) for j in range(dim)]
            )
            new_arr = np.zeros_like(np.asarray(self.forcing_star.array))
            for (i, j) in indep:
                vals = _eval_nd(forcing_fn[i, j])
                new_arr[:, i, j] = vals
                if i != j:
                    new_arr[:, j, i] = vals
            self.forcing_star.array[...] = new_arr
        elif vtype == uw.VarType.VECTOR:
            dim = self.mesh.dim
            new_arr = np.zeros_like(np.asarray(self.forcing_star.array))
            for i in range(dim):
                new_arr[:, i] = _eval_nd(forcing_fn[i])
            self.forcing_star.array[...] = new_arr
        else:  # SCALAR
            self.forcing_star.array[:] = _eval_nd(forcing_fn)


## Consider Deprecating this one - it is the same as the Lagrangian_Swarm but
## sets up the swarm for itself. This does not have a practical use-case - the swarm version
## is slower, more cumbersome, and less stable / accurate. The only reason to use
## it is if there is an existing swarm that we can re-purpose.


class Lagrangian(uw_object):
    r"""
    Swarm-based Lagrangian history manager for material tracking.

    Manages the update of a Lagrangian variable :math:`\psi` on a swarm
    across timesteps. Creates and manages its own internal swarm for
    tracking material properties through the flow.

    .. math::

        \psi_p^{t-n\Delta t} \leftarrow \psi_p^{t-(n-1)\Delta t}

        \psi_p^{t-(n-1)\Delta t} \leftarrow \psi_p^{t-(n-2)\Delta t} \cdots

        \psi_p^{t-\Delta t} \leftarrow \psi_p^{t}

    The Lagrangian approach follows material points through the flow,
    avoiding numerical diffusion in advection. History values are stored
    on swarm variables with proxy mesh variables for solver integration.

    Parameters
    ----------
    mesh : underworld3.discretisation.Mesh
        The computational mesh.
    psi_fn : sympy.Function
        The quantity to track (e.g., stress tensor ``stokes.stress``).
    V_fn : sympy.Function
        Velocity field for particle advection.
    vtype : VarType
        Variable type (SCALAR, VECTOR, SYM_TENSOR, etc.).
    degree : int
        Polynomial degree for proxy mesh variables.
    continuous : bool
        Whether proxy variables are continuous across elements.
    varsymbol : str, default=r"u"
        LaTeX symbol for display.
    verbose : bool, default=False
        Enable verbose output during updates.
    bcs : list, default=[]
        Boundary conditions (currently unused for swarm variables).
    order : int, default=1
        Number of history timesteps to store.
    smoothing : float, default=0.0
        Smoothing parameter for projections.
    fill_param : int, default=3
        Fill parameter for swarm population density.

    Notes
    -----
    The Lagrangian method is ideal for:

    - Viscoelastic stress history tracking
    - Material property advection without diffusion
    - Tracking compositional fields

    The internal swarm is automatically advected during updates.

    See Also
    --------
    SemiLagrangian : Mesh-based semi-Lagrangian with departure points.
    Lagrangian_Swarm : For user-provided swarms.
    """

    instances = (
        0  # count how many of these there are in order to create unique private mesh variable ids
    )

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        psi_fn: sympy.Function,
        V_fn: sympy.Function,
        vtype: uw.VarType,
        degree: int,
        continuous: bool,
        varsymbol: Optional[str] = r"u",
        verbose: Optional[bool] = False,
        bcs=[],
        order=1,
        smoothing=0.0,
        fill_param=3,
    ):
        super().__init__()

        # create a new swarm to manage here
        dudt_swarm = uw.swarm.UWSwarm(mesh)

        self.mesh = mesh
        self.swarm = dudt_swarm
        self.psi_fn = psi_fn
        self.V_fn = V_fn
        self.verbose = verbose
        self.order = order

        # History tracking: deferred initialization and effective order
        self._history_initialised = False
        self._n_solves_completed = 0
        self._dt = None  # current timestep (set by solver or update_pre_solve)
        self._dt_history = [None] * order  # previous timesteps for variable-dt BDF

        psi_star = []
        self.psi_star = psi_star

        for i in range(order):
            self.psi_star.append(
                uw.swarm.SwarmVariable(
                    f"psi_star_sw_{self.instance_number}_{i}",
                    self.swarm,
                    vtype=vtype,
                    proxy_degree=degree,
                    proxy_continuous=continuous,
                    varsymbol=rf"{varsymbol}^{{ {'*'*(i+1)} }}",
                )
            )

        # BDF/AM coefficient UWexpressions — routed through PetscDS constants[]
        self._bdf_coeffs = _create_coefficients(order, r"c^{\mathrm{BDF}}", self.instance_number)
        self._am_coeffs = _create_coefficients(order, r"a^{\mathrm{AM}}", self.instance_number)
        # Initialise to order-1 values
        _update_bdf_values(self._bdf_coeffs, 1, None, [])
        _update_am_values(self._am_coeffs, 1, 0.5)

        dudt_swarm.populate(fill_param)

        try:
            import underworld3 as _uw

            _uw.get_default_model()._register_state_bearer(self)
        except Exception:
            pass

        return

    @property
    def state(self) -> "DDtLagrangianState":
        return DDtLagrangianState(
            dt_history=list(self._dt_history),
            history_initialised=bool(self._history_initialised),
            n_solves_completed=int(self._n_solves_completed),
            dt=self._dt,
            psi_star_var_names=[ps.clean_name for ps in self.psi_star],
        )

    @state.setter
    def state(self, s: "DDtLagrangianState") -> None:
        if s._schema_version != DDtLagrangianState._schema_version:
            raise ValueError(
                f"DDtLagrangianState schema version mismatch: snapshot "
                f"{s._schema_version} vs current "
                f"{DDtLagrangianState._schema_version}"
            )
        if len(s.dt_history) != len(self._dt_history):
            raise ValueError(
                f"dt_history length mismatch ({len(s.dt_history)} vs "
                f"{len(self._dt_history)}); order changed since snapshot?"
            )
        current_names = [ps.clean_name for ps in self.psi_star]
        if s.psi_star_var_names and s.psi_star_var_names != current_names:
            raise ValueError(
                f"psi_star variable names changed since snapshot: "
                f"{s.psi_star_var_names} vs {current_names}"
            )
        self._dt_history = list(s.dt_history)
        self._history_initialised = bool(s.history_initialised)
        self._n_solves_completed = int(s.n_solves_completed)
        self._dt = s.dt
        _update_bdf_values(
            self._bdf_coeffs, self.effective_order, self._dt, self._dt_history
        )
        _update_am_values(self._am_coeffs, self.effective_order, 0.5)

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        ## feedback on this instance
        display(Latex(r"$\quad\psi = $ " + self.psi._repr_latex_()))
        display(
            Latex(
                r"$\quad\Delta t_{\textrm{phys}} = $ "
                + sympy.sympify(self.dt_physical)._repr_latex_()
            )
        )
        display(Latex(rf"$\quad$History steps = {self.order}"))

    @property
    def effective_order(self):
        """Current effective BDF order, accounting for history startup."""
        # BDF-k requires k completed solves to have k distinct history values.
        # With 0 or 1 completed solves → order 1. Order 2 needs ≥2 solves.
        return min(self.order, max(1, self._n_solves_completed))

    def initialise_history(self):
        r"""Initialize all history slots to the current value of :math:`\psi`.

        Called automatically on the first ``update_pre_solve``. Can also
        be called manually after setting initial conditions.
        """
        psi_star_0 = self.psi_star[0]
        with self.swarm.access(psi_star_0):
            for i in range(psi_star_0.shape[0]):
                for j in range(psi_star_0.shape[1]):
                    updated_psi = uw.function.evaluate(
                        self.psi_fn[i, j],
                        self.swarm.data,
                    )
                    psi_star_0[i, j].data[:] = updated_psi

        # Copy to all other history slots
        for k in range(1, self.order):
            with self.swarm.access(self.psi_star[k]):
                self.psi_star[k].data[...] = psi_star_0.data[...]

        self._history_initialised = True
        return

    def initiate_history_fn(self):
        """Deprecated: use ``initialise_history`` instead."""
        self.initialise_history()

    ## Note: We may be able to eliminate this
    ## The SL updater and the Lag updater have
    ## different sequencing because of the way they
    ## update the history. It makes more sense for the
    ## full Lagrangian swarm to be updated after the solve
    ## and this means we have to grab the history values first.

    def update(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        """Update history (alias for ``update_post_solve``)."""
        self.update_post_solve(dt, evalf, verbose)
        return

    def update_pre_solve(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        """Pre-solve: auto-initialise history on first call."""
        self._dt = dt

        if not self._history_initialised:
            self.initialise_history()

        # Update coefficient values for current effective_order and dt
        _update_bdf_values(self._bdf_coeffs, self.effective_order, self._dt, self._dt_history)
        _update_am_values(self._am_coeffs, self.effective_order, 0.5)

        return

    def update_post_solve(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        """Shift history chain and advect swarm after solve."""
        self._dt = dt

        # Record timestep history for variable-dt BDF
        for i in range(self.order - 1, 0, -1):
            self._dt_history[i] = self._dt_history[i - 1]
        self._dt_history[0] = dt

        for h in range(self.order - 1):
            i = self.order - (h + 1)

            # copy the information down the chain
            if verbose:
                print(f"Lagrange order = {self.order}")
                print(f"Lagrange copying {i-1} to {i}")

            self.psi_star[i].array[...] = self.psi_star[i - 1].array[...]

        # Now update the swarm variable

        psi_star_0 = self.psi_star[0]
        with self.swarm.access(psi_star_0):
            for i in range(psi_star_0.shape[0]):
                for j in range(psi_star_0.shape[1]):
                    updated_psi = uw.function.evaluate(
                        self.psi_fn[i, j],
                        self.swarm.data,
                        evalf=evalf,
                    )
                    psi_star_0[i, j].data[:] = updated_psi

        # Now update the swarm locations

        self.swarm.advection(
            self.V_fn,
            delta_t=dt,
            restore_points_to_domain_func=self.mesh.return_coords_to_bounds,
        )

        if self._n_solves_completed < self.order:
            self._n_solves_completed += 1

    @property
    def bdf_coefficients(self):
        """Current BDF coefficients [c0, c1, ...] accounting for variable timesteps."""
        return _bdf_coefficients(self.effective_order, self._dt, self._dt_history)

    def bdf(self, order=None):
        r"""Backward differentiation approximation of the time-derivative of :math:`\psi`.

        Returns a fixed-structure symbolic expression using UWexpression
        coefficients. Values are updated each step in ``update_pre_solve``.

        Parameters
        ----------
        order : int, optional
            Ignored (kept for API compatibility).
        """
        return _build_weighted_sum(self._bdf_coeffs, self.psi_fn, [ps.sym for ps in self.psi_star])

    def adams_moulton_flux(self, order=None):
        r"""Adams-Moulton flux approximation for implicit time integration.

        Returns a fixed-structure symbolic expression using UWexpression
        coefficients. Values are updated each step in ``update_pre_solve``.

        Parameters
        ----------
        order : int, optional
            Ignored (kept for API compatibility).
        """
        return _build_weighted_sum(self._am_coeffs, self.psi_fn, [ps.sym for ps in self.psi_star])


class Lagrangian_Swarm(uw_object):
    r"""
    Swarm-based Lagrangian history manager (user-provided swarm).

    Manages the update of a Lagrangian variable :math:`\psi` on a user-supplied
    swarm across timesteps:

    .. math::

        \psi_p^{t-n\Delta t} \leftarrow \psi_p^{t-(n-1)\Delta t}

        \psi_p^{t-(n-1)\Delta t} \leftarrow \psi_p^{t-(n-2)\Delta t} \cdots

        \psi_p^{t-\Delta t} \leftarrow \psi_p^{t}

    Unlike ``Lagrangian``, this class uses a user-provided swarm rather than
    creating its own. The swarm should already be populated and configured
    for tracking material points.

    Parameters
    ----------
    swarm : underworld3.swarm.Swarm
        User-provided swarm for material point tracking.
    psi_fn : sympy.Function
        The quantity to track (e.g., stress tensor from a solver).
    vtype : underworld3.VarType
        Variable type (SCALAR, VECTOR, SYM_TENSOR, etc.).
    degree : int
        Interpolation degree for proxy mesh variables.
    continuous : bool
        Whether proxy mesh variables should be continuous.
    varsymbol : str, optional
        LaTeX symbol for display (default ``"u"``).
    verbose : bool, optional
        Enable verbose output (default ``False``).
    bcs : list, optional
        Boundary conditions (currently unused, default ``[]``).
    order : int, optional
        Order of time integration (1-3) (default ``1``).
    smoothing : float, optional
        Smoothing parameter (default ``0.0``).
    step_averaging : int, optional
        Number of steps for history averaging (default ``2``).

    Attributes
    ----------
    mesh : underworld3.discretisation.Mesh
        Reference to the computational mesh (from swarm).
    swarm : underworld3.swarm.Swarm
        The user-provided swarm.
    psi_fn : sympy.Function
        Symbolic expression for the tracked quantity.
    order : int
        Order of BDF integration.
    step_averaging : int
        Number of steps for averaging (affects BDF scaling).
    psi_star : list
        History values :math:`\psi^*, \psi^{**}, \ldots` as swarm variables.

    Notes
    -----
    Key differences from ``Lagrangian`` class:

    - Uses user-provided swarm (not internally created)
    - Swarm advection is NOT performed (user's responsibility)
    - Step averaging for smoothing history updates
    - Suitable when swarm is shared between multiple history managers

    The ``step_averaging`` parameter scales the BDF formula to account for
    updates that occur over multiple sub-steps within a timestep.

    See Also
    --------
    Lagrangian : Creates and manages its own swarm with advection.
    SemiLagrangian : Nodal-swarm approach for advection-dominated problems.
    Eulerian : Pure mesh-based history (no particle tracking).
    """

    instances = (
        0  # count how many of these there are in order to create unique private mesh variable ids
    )

    @timing.routine_timer_decorator
    def __init__(
        self,
        swarm: uw.swarm.Swarm,
        psi_fn: sympy.Function,
        vtype: uw.VarType,
        degree: int,
        continuous: bool,
        varsymbol: Optional[str] = r"u",
        verbose: Optional[bool] = False,
        bcs=[],
        order=1,
        smoothing=0.0,
        step_averaging=2,
    ):
        super().__init__()

        self.mesh = swarm.mesh
        self.swarm = swarm
        self.psi_fn = psi_fn
        self.verbose = verbose
        self.order = order
        self.step_averaging = step_averaging

        # History tracking: deferred initialization and effective order
        self._history_initialised = False
        self._n_solves_completed = 0
        self._dt = None  # current timestep (set by solver or update_pre_solve)
        self._dt_history = [None] * order  # previous timesteps for variable-dt BDF

        psi_star = []
        self.psi_star = psi_star

        for i in range(order):
            self.psi_star.append(
                uw.swarm.SwarmVariable(
                    f"psi_star_sw_{self.instance_number}_{i}",
                    self.swarm,
                    vtype=vtype,
                    proxy_degree=degree,
                    proxy_continuous=continuous,
                    varsymbol=rf"{varsymbol}^{{ {'*'*(i+1)} }}",
                )
            )

        # BDF/AM coefficient UWexpressions — routed through PetscDS constants[]
        self._bdf_coeffs = _create_coefficients(order, r"c^{\mathrm{BDF}}", self.instance_number)
        self._am_coeffs = _create_coefficients(order, r"a^{\mathrm{AM}}", self.instance_number)
        # Initialise to order-1 values
        _update_bdf_values(self._bdf_coeffs, 1, None, [])
        _update_am_values(self._am_coeffs, 1, 0.5)

        try:
            import underworld3 as _uw

            _uw.get_default_model()._register_state_bearer(self)
        except Exception:
            pass

        return

    @property
    def state(self) -> "DDtLagrangianSwarmState":
        return DDtLagrangianSwarmState(
            dt_history=list(self._dt_history),
            history_initialised=bool(self._history_initialised),
            n_solves_completed=int(self._n_solves_completed),
            dt=self._dt,
            psi_star_var_names=[ps.clean_name for ps in self.psi_star],
        )

    @state.setter
    def state(self, s: "DDtLagrangianSwarmState") -> None:
        if s._schema_version != DDtLagrangianSwarmState._schema_version:
            raise ValueError(
                f"DDtLagrangianSwarmState schema version mismatch: snapshot "
                f"{s._schema_version} vs current "
                f"{DDtLagrangianSwarmState._schema_version}"
            )
        if len(s.dt_history) != len(self._dt_history):
            raise ValueError(
                f"dt_history length mismatch ({len(s.dt_history)} vs "
                f"{len(self._dt_history)}); order changed since snapshot?"
            )
        current_names = [ps.clean_name for ps in self.psi_star]
        if s.psi_star_var_names and s.psi_star_var_names != current_names:
            raise ValueError(
                f"psi_star variable names changed since snapshot: "
                f"{s.psi_star_var_names} vs {current_names}"
            )
        self._dt_history = list(s.dt_history)
        self._history_initialised = bool(s.history_initialised)
        self._n_solves_completed = int(s.n_solves_completed)
        self._dt = s.dt
        _update_bdf_values(
            self._bdf_coeffs, self.effective_order, self._dt, self._dt_history
        )
        _update_am_values(self._am_coeffs, self.effective_order, 0.5)

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        ## feedback on this instance
        display(Latex(r"$\quad\psi = $ " + self.psi._repr_latex_()))
        display(
            Latex(
                r"$\quad\Delta t_{\textrm{phys}} = $ "
                + sympy.sympify(self.dt_physical)._repr_latex_()
            )
        )
        display(Latex(rf"$\quad$History steps = {self.order}"))

    @property
    def effective_order(self):
        """Current effective BDF order, accounting for history startup."""
        # BDF-k requires k completed solves to have k distinct history values.
        # With 0 or 1 completed solves → order 1. Order 2 needs ≥2 solves.
        return min(self.order, max(1, self._n_solves_completed))

    def initialise_history(self):
        r"""Initialize all history slots to the current value of :math:`\psi`.

        Called automatically on the first ``update_pre_solve``. Can also
        be called manually after setting initial conditions.
        """
        psi_star_0 = self.psi_star[0]
        with self.swarm.access(psi_star_0):
            for i in range(psi_star_0.shape[0]):
                for j in range(psi_star_0.shape[1]):
                    updated_psi = uw.function.evaluate(
                        self.psi_fn[i, j],
                        self.swarm.data,
                    )
                    psi_star_0[i, j].data[:] = updated_psi

        # Copy to all other history slots
        for k in range(1, self.order):
            with self.swarm.access(self.psi_star[k]):
                self.psi_star[k].data[...] = psi_star_0.data[...]

        self._history_initialised = True
        return

    def initiate_history_fn(self):
        """Deprecated: use ``initialise_history`` instead."""
        self.initialise_history()

    ## Note: We may be able to eliminate this
    ## The SL updater and the Lag updater have
    ## different sequencing because of the way they
    ## update the history. It makes more sense for the
    ## full Lagrangian swarm to be updated after the solve
    ## and this means we have to grab the history values first.

    def update(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        """Update history (alias for ``update_post_solve``)."""
        self.update_post_solve(dt, evalf, verbose)
        return

    def update_pre_solve(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        """Pre-solve: auto-initialise history on first call."""
        self._dt = dt

        if not self._history_initialised:
            self.initialise_history()

        # Update coefficient values for current effective_order and dt
        _update_bdf_values(self._bdf_coeffs, self.effective_order, self._dt, self._dt_history)
        _update_am_values(self._am_coeffs, self.effective_order, 0.5)

        return

    def update_post_solve(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        r"""Shift history chain and evaluate current :math:`\psi` on swarm."""
        self._dt = dt

        # Record timestep history for variable-dt BDF
        for i in range(self.order - 1, 0, -1):
            self._dt_history[i] = self._dt_history[i - 1]
        self._dt_history[0] = dt

        for h in range(self.order - 1):
            i = self.order - (h + 1)

            # copy the information down the chain
            if verbose:
                print(f"Lagrange swarm order = {self.order}", flush=True)
                print(
                    f"Mesh interpolant order = {self.psi_star[0]._meshVar.degree}",
                    flush=True,
                )
                print(f"Lagrange swarm copying {i-1} to {i}", flush=True)

            with self.swarm.access(self.psi_star[i]):
                self.psi_star[i].data[...] = self.psi_star[i - 1].data[...]

        phi = 1 / self.step_averaging

        psi_star_0 = self.psi_star[0]
        with self.swarm.access(psi_star_0):
            for i in range(psi_star_0.shape[0]):
                for j in range(psi_star_0.shape[1]):
                    updated_psi = uw.function.evaluate(
                        self.psi_fn[i, j],
                        self.swarm.data,
                        evalf=evalf,
                    )
                    psi_star_0[i, j].data[:] = (
                        phi * updated_psi + (1 - phi) * psi_star_0[i, j].data[:]
                    )

        if self._n_solves_completed < self.order:
            self._n_solves_completed += 1

        return

    @property
    def bdf_coefficients(self):
        """Current BDF coefficients [c0, c1, ...] accounting for variable timesteps."""
        return _bdf_coefficients(self.effective_order, self._dt, self._dt_history)

    def bdf(self, order=None):
        r"""Backward differentiation approximation of the time-derivative of :math:`\psi`.

        Returns a fixed-structure symbolic expression using UWexpression
        coefficients. Values are updated each step in ``update_pre_solve``.

        Parameters
        ----------
        order : int, optional
            Ignored (kept for API compatibility).
        """
        return _build_weighted_sum(self._bdf_coeffs, self.psi_fn, [ps.sym for ps in self.psi_star])

    def adams_moulton_flux(self, order=None):
        r"""Adams-Moulton flux approximation for implicit time integration.

        Returns a fixed-structure symbolic expression using UWexpression
        coefficients. Values are updated each step in ``update_pre_solve``.

        Parameters
        ----------
        order : int, optional
            Ignored (kept for API compatibility).
        """
        return _build_weighted_sum(self._am_coeffs, self.psi_fn, [ps.sym for ps in self.psi_star])
