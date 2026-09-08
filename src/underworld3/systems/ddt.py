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

import math
import warnings

import sympy
from sympy import sympify
import numpy as np

from dataclasses import dataclass, field
from typing import Any, Optional, Callable, Union

import underworld3 as uw
from underworld3 import VarType

import underworld3.timing as timing
from underworld3.utilities._api_tools import uw_object
from underworld3.utilities.unit_aware_array import UnitAwareArray
from underworld3.checkpoint.state import SnapshottableState
from underworld3.discretisation.remesh import RemeshPolicy, remap_var_set

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
    """A plain, NON-DIMENSIONAL float from a number, a Pint quantity or a UWQuantity.

    A quantity is scaled by the active model's reference scales (#701: taking
    its magnitude gave the kernels a dimensional timestep); without reference
    scales the magnitude is what non-dimensionalisation returns.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "magnitude") or hasattr(value, "dimensionality"):
        nd = uw.non_dimensionalise(value)
        if hasattr(nd, "magnitude"):
            return float(nd.magnitude)
        return float(nd)
    if hasattr(value, "value"):
        return float(value.value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_nondim_ndarray(value, units=None):
    """Reduce a possibly unit-carrying array to a plain non-dimensional ndarray.

    The semi-Lagrangian trace-back, the DM point-location and all variable
    storage operate in the mesh's NON-DIMENSIONAL coordinate/value space
    (UW3 issue #267): every array entering that arithmetic must be a plain
    ndarray of non-dimensional values. This is the single reduction used
    by every coordinate / velocity / forcing unwrap in this module.

    Parameters
    ----------
    value : array-like, UnitAwareArray, or Pint quantity
        The array to reduce. A plain ndarray with ``units=None`` is
        assumed to be non-dimensional already and is returned unchanged.
    units : pint.Unit or str, optional
        Units known out-of-band (e.g. from ``uw.get_units``) to attach
        to a plain array before non-dimensionalising. Ignored when
        ``value`` already carries units; ``None`` or ``"dimensionless"``
        means no attachment.

    Returns
    -------
    numpy.ndarray or original type
        Non-dimensional plain array (unit-carrying input), or ``value``
        unchanged (plain input with no ``units`` supplied).
    """
    carries_units = isinstance(value, UnitAwareArray) or hasattr(value, "magnitude")
    if not carries_units and units and str(units) != "dimensionless":
        value = UnitAwareArray(np.asarray(value), units=units)
        carries_units = True
    if not carries_units:
        return value
    # TODO(BUG): a raw pint.Quantity input reaches uw.non_dimensionalise(),
    # which crashes when reference scales are active (units.py protocol 5,
    # invalid `dimensionality=` kwarg). Pre-existing — the unified unwrap
    # sites had the same hasattr("magnitude") gate, and the DDt data paths
    # only ever supply UnitAwareArray/plain ndarray from uw.function
    # evaluate/global_evaluate. Fix belongs in units.py. See #328.
    nd = uw.non_dimensionalise(value)
    if isinstance(nd, UnitAwareArray):
        return np.array(nd)
    if hasattr(nd, "magnitude"):
        return nd.magnitude
    if hasattr(nd, "value"):
        return nd.value
    return np.asarray(nd)


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


class _DDtBase(uw_object):
    r"""Shared machinery for the DDt history-manager flavors.

    The five flavors (:class:`Symbolic`, :class:`Eulerian`,
    :class:`SemiLagrangian`, :class:`Lagrangian`,
    :class:`Lagrangian_Swarm`) share the same BDF/Adams-Moulton
    coefficient bookkeeping, effective-order startup ramp,
    fixed-structure ``bdf()`` / ``adams_moulton_flux()`` expressions,
    model registration and snapshot-restore validation. Each flavor owns
    its storage (sympy expressions, mesh variables or swarm variables in
    ``psi_star``) and its ``update_*`` sequencing.

    Deliberate per-flavor divergences are passed explicitly, never
    averaged away:

    - **AM theta**: Symbolic / Eulerian / SemiLagrangian carry a
      user-settable ``theta``; the swarm-based Lagrangian flavors have
      no theta parameter and always use the Crank-Nicolson value 0.5.
    - **History symbols**: Symbolic stores raw sympy matrices in
      ``psi_star``; the storage-backed flavors store variables and
      contribute ``.sym`` (see :meth:`_history_syms`).
    - **ETD-2 exp coefficients** exist only on the flavors used by the
      Maxwell / viscoelastic relaxation path (``with_exp=True``:
      Symbolic, Eulerian, SemiLagrangian).
    """

    def _init_history_tracking(self, order):
        """Deferred-initialisation and variable-dt bookkeeping attributes."""
        # The timestep as a runtime constant of the compiled kernels: every
        # flavour writes it through the ``_dt`` property, so a solver that
        # composes its residual from :meth:`time_derivative` never recompiles
        # when the step changes.
        self._delta_t = _UWexpression(
            rf"\Delta t_{{{self.instance_number}}}", 1.0, "DDt timestep",
            _unique_name_generation=True)
        # History tracking: deferred initialization and effective order
        self._history_initialised = False
        self._n_solves_completed = 0
        self._dt = None  # current timestep (set by solver or update_pre_solve)
        self._dt_history = [None] * order  # previous timesteps for variable-dt BDF

    @property
    def _dt(self):
        return self._dt_value

    @_dt.setter
    def _dt(self, value):
        self._dt_value = value
        if value is None:
            return
        try:
            dt = float(_as_float(value))
        except Exception:
            return
        if dt > 0.0:
            self._delta_t.sym = dt

    @property
    def delta_t(self):
        r"""The timestep :math:`\Delta t` as a UW expression (a runtime constant).

        Written by ``update_pre_solve`` and by a solver's ``delta_t`` setter;
        read by :meth:`time_derivative`.
        """
        return self._delta_t

    def _init_coefficient_expressions(self, order, theta, with_exp):
        """Create BDF/AM (and optionally ETD-2 exp) coefficient UWexpressions.

        The coefficients are routed through PetscDS constants[] (see the
        module-level coefficient helpers), so order ramp-up and variable
        dt only change values, never the compiled symbolic structure.
        All sets are initialised to the order-1 / viscous startup values.

        Parameters
        ----------
        order : int
            Maximum BDF/AM order (creates ``order + 1`` coefficients each).
        theta : float
            Adams-Moulton order-1 implicitness for the startup values —
            ``self.theta`` for flavors that expose it, 0.5
            (Crank-Nicolson) for the Lagrangian flavors that don't.
        with_exp : bool
            Also create the ETD-2 ``[α, φ]`` coefficients used by
            Maxwell-relaxation integration; values are pushed via
            PetscDSSetConstants every step in ``update_exp_coefficients``
            (Symbolic, Eulerian, SemiLagrangian only).
        """
        self._bdf_coeffs = _create_coefficients(order, r"c^{\mathrm{BDF}}", self.instance_number)
        self._am_coeffs = _create_coefficients(order, r"a^{\mathrm{AM}}", self.instance_number)
        if with_exp:
            self._exp_coeffs = _create_exp_coefficients(self.instance_number)
        # Initialise to order-1 / viscous values
        _update_bdf_values(self._bdf_coeffs, 1, None, [])
        _update_am_values(self._am_coeffs, 1, theta)
        if with_exp:
            _update_exp_values(self._exp_coeffs, None, None)

    def _register_with_default_model(self):
        """Register with the active default model as a snapshot state-bearer.

        Safe if no model is active.
        """
        try:
            uw.get_default_model()._register_state_bearer(self)
        except (ImportError, AttributeError):
            # Narrowed per Copilot review on #195: only swallow the
            # genuine bootstrap modes (uw attributes not yet wired during
            # underworld3 init, or older Model without the registry
            # method). Anything else propagates rather than silently
            # masking a registration bug — exactly the silent-state-
            # loss failure mode the design note warns against.
            pass

    # ----- Snapshot / restore helpers (see checkpoint/state.py) -----

    def _core_state_kwargs(self):
        """Common evolution-tracking fields for the State dataclasses."""
        return dict(
            dt_history=list(self._dt_history),
            history_initialised=bool(self._history_initialised),
            n_solves_completed=int(self._n_solves_completed),
            dt=self._dt,
        )

    def _validate_state_schema(self, s, state_cls):
        """Reject snapshots with a different schema or history depth."""
        if s._schema_version != state_cls._schema_version:
            raise ValueError(
                f"{state_cls.__name__} schema version mismatch: snapshot "
                f"{s._schema_version} vs current "
                f"{state_cls._schema_version}"
            )
        if len(s.dt_history) != len(self._dt_history):
            raise ValueError(
                f"dt_history length mismatch ({len(s.dt_history)} vs "
                f"{len(self._dt_history)}); order changed since snapshot?"
            )

    def _validate_psi_star_names(self, snapshot_names):
        """Verify the snapshot still binds to this instance's variables."""
        current_names = [ps.clean_name for ps in self.psi_star]
        if snapshot_names and snapshot_names != current_names:
            raise ValueError(
                f"psi_star variable names changed since snapshot: "
                f"{snapshot_names} vs {current_names}"
            )

    def _restore_core_state(self, s, am_theta):
        """Write the captured core state back and re-derive coefficients.

        Re-deriving the BDF/AM coefficient values means downstream reads
        see values consistent with the restored primary state without
        waiting for the next ``update_pre_solve``. ``am_theta`` carries
        the per-flavor Adams-Moulton implicitness (``self.theta`` where
        the flavor exposes it, 0.5 for the Lagrangian flavors).
        """
        self._dt_history = list(s.dt_history)
        self._history_initialised = bool(s.history_initialised)
        self._n_solves_completed = int(s.n_solves_completed)
        self._dt = s.dt
        _update_bdf_values(
            self._bdf_coeffs, self.effective_order, self._dt, self._dt_history
        )
        _update_am_values(self._am_coeffs, self.effective_order, am_theta)

    # ----- Order bookkeeping and the fixed-structure operators -----

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

    @property
    def bdf_coefficients(self):
        """Current BDF coefficients [c0, c1, ...] accounting for variable timesteps."""
        return _bdf_coefficients(self.effective_order, self._dt, self._dt_history)

    @property
    def bdf_coefficient_expressions(self):
        r"""The BDF coefficient symbols :math:`[c_0, c_1, \dots]` as UWexpressions.

        For a solver that assembles its own weighted sum of history terms
        (an Eulerian scheme applying the multistep rule to a spatial
        operator, say). The symbols are routed through PETSc's
        ``constants[]`` array, so their values follow ``effective_order``
        and the timestep without a recompile; ``bdf_coefficients`` gives
        the current values.
        """
        return list(self._bdf_coeffs)

    @property
    def am_coefficient_expressions(self):
        r"""The Adams-Moulton coefficient symbols :math:`[a_0, a_1, \dots]` as UWexpressions.

        :math:`a_0` weights the new state, :math:`a_k` the history slot
        ``psi_star[k-1]``. Same constants-routing as
        :attr:`bdf_coefficient_expressions`.
        """
        return list(self._am_coeffs)

    def _history_syms(self):
        """History terms as sympy expressions for the weighted sums.

        Storage-backed flavors contribute each history variable's
        ``.sym``; :class:`Symbolic` overrides this to return its raw
        sympy matrices.
        """
        return [ps.sym for ps in self.psi_star]

    # ----- velocity-expression snapshots (mid-time trace-back) -----
    def _V_matrix(self):
        """``V_fn`` as a sympy row matrix (a mesh variable contributes its symbol)."""
        V = self.V_fn
        if hasattr(V, "sym") and not isinstance(V, sympy.Basic):
            return sympy.Matrix(V.sym)
        return sympy.Matrix(V)

    def _velocity_degree(self):
        """Degree of the nodal velocity cache: the highest degree among the
        mesh variables in ``V_fn`` (2 for an analytic velocity)."""
        _, varfns, _ = uw.function.expressions.mesh_vars_in_expression(self._V_matrix())
        degs = [fn.meshvar().degree for fn in varfns]
        return max(degs) if degs else 2

    def _make_velocity_level(self, tag):
        """A cached velocity level: ``V_fn`` EVALUATED at the true nodes of a
        vector field (no nudge; the evaluator is exact at node coordinates on
        simplex, quad and annulus meshes). Caching by evaluation, rather than
        by substituting snapshots of the mesh variables into the expression,
        is what captures everything ``V_fn`` depends on at that time: the
        variables, constants that ramp, swarm proxies, the mesh geometry."""
        snap = uw.discretisation.MeshVariable(
            f"vcache_{tag}_{self.instance_number}", self.mesh, self.mesh.dim,
            degree=self._velocity_degree(), continuous=True,
            varsymbol=rf"{{ V^{{ ({tag}) }}_{{ [{self.instance_number}] }} }}",
            units=self._velocity_units(),   # same frame as V_fn, so 1.5 v - 0.5 v_prev is consistent
        )
        snap.remesh_policy = RemeshPolicy.CARRY
        snap._remesh_managed_by = self
        return {"var": snap, "expr": snap.sym}

    def _velocity_units(self):
        """Units of ``V_fn`` under an active units model, else None."""
        units = uw.get_units(self._V_matrix())
        if units is not None and not uw.get_default_model().has_units():
            units = None
        return units

    def _copy_velocity_level(self, dst, src=None):
        """``dst`` <- ``src`` (another level) or, with ``src=None``, ``V_fn``
        evaluated now at ``dst``'s nodes (reduced to the non-dimensional
        frame ``.data`` stores, issue #267)."""
        if src is None:
            vals = uw.function.evaluate(self._V_matrix(), np.asarray(dst["var"].coords_nd))
            vals = _to_nondim_ndarray(vals, units=self._velocity_units())
            dst["var"].data[...] = np.asarray(vals).reshape(-1, self.mesh.dim)
        else:
            dst["var"].data[...] = src["var"].data[...]

    def bdf(self, order: Optional[int] = None):
        r"""Backward differentiation approximation of the time-derivative of :math:`\psi`.

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
        return _build_weighted_sum(self._bdf_coeffs, self.psi_fn, self._history_syms())

    def adams_moulton_flux(self, order: Optional[int] = None):
        r"""Adams-Moulton flux approximation for implicit time integration.

        Returns a fixed-structure symbolic expression using UWexpression
        coefficients. Values are updated each step in ``update_pre_solve``.

        Parameters
        ----------
        order : int, optional
            Ignored (kept for API compatibility).
        """
        return _build_weighted_sum(self._am_coeffs, self.psi_fn, self._history_syms())

    def initiate_history_fn(self):
        """Deprecated: use ``initialise_history`` instead."""
        self.initialise_history()

    # ----- The transport contract -----
    #
    # A solver that owns an unknown composes its residual from these terms
    # and never asks which flavour it holds:
    #
    #     F0 = time_derivative() + advection() - f
    #     F1 = <the solver's own flux of the levels in spatial_weights()>
    #          + stabilisation_flux(R)
    #
    # The history flavours (Symbolic, Eulerian, SemiLagrangian, Lagrangian)
    # carry their transport in the history itself, so advection() and the
    # stabilisation flux are zero for them; EulerianSUPG assembles both.

    @property
    def integrator(self) -> str:
        """``"am"`` (the theta rule on the spatial terms) at order 1, ``"bdf"`` above."""
        return "am" if self.order == 1 else "bdf"

    def _unknown_shape(self):
        """Shape of the unknown as a matrix (``Symbolic`` stores ``_shape`` as data)."""
        psi = self.psi_fn
        return psi.shape if isinstance(psi, sympy.MatrixBase) else (1, 1)

    def states(self):
        r"""``[psi^{n+1}, psi^{n}, psi^{n-1}, ...]`` as matrices of the unknown's shape."""
        return [sympy.Matrix(self.psi_fn)] + [sympy.Matrix(h) for h in self._history_syms()]

    def spatial_weights(self):
        """Weight of a spatial operator at each level of :meth:`states`.

        ``[1, 0, ...]`` for the BDF family (every spatial term at n+1); the
        Adams-Moulton weights for the theta rule.
        """
        n = len(self.psi_star)
        if self.integrator == "bdf":
            return [sympy.Integer(1)] + [sympy.Integer(0)] * n
        return list(self.am_coefficient_expressions[: n + 1])

    def time_derivative(self):
        r"""The time derivative of the scheme, a matrix of the unknown's shape.

        ``(psi^{n+1} - psi^{n}) / dt`` for the theta rule, the BDF stencil over
        the history divided by ``dt`` above order 1, with ``dt`` the runtime
        constant :attr:`delta_t`.
        """
        if self.integrator == "am":
            new, old = self.states()[:2]
            return (new - old) / self._delta_t
        return sympy.Matrix(self.bdf()) / self._delta_t

    def advection(self):
        """The assembled advection term: zero for a history-carrying flavour."""
        return sympy.zeros(*self._unknown_shape())

    def stabilisation_flux(self, R):
        r"""The stabilisation flux for a strong residual ``R``: zero here.

        Shape ``(len(R), dim)``: one flux row per component of ``R``.
        """
        mesh = getattr(self, "mesh", None)
        if mesh is None:
            raise TypeError(f"{type(self).__name__} has no mesh: no flux shape to return.")
        return sympy.zeros(len(_as_matrix(R)), mesh.dim)


def _as_matrix(R):
    """A residual as a sympy Matrix: a bare scalar becomes ``(1, 1)``."""
    return R if isinstance(R, sympy.MatrixBase) else sympy.Matrix([[R]])


def _as_row_vector(V_fn, dim):
    """Coerce a velocity expression to a ``(1, dim)`` sympy row Matrix."""
    if isinstance(V_fn, uw.discretisation.MeshVariable):
        V_fn = V_fn.sym
    if isinstance(V_fn, sympy.MatrixBase):
        if V_fn.shape == (1, dim):
            return V_fn
        if V_fn.shape == (dim, 1):
            return V_fn.T
        raise ValueError(
            f"V_fn has shape {V_fn.shape} but the mesh is {dim}-D; expected a "
            f"(1, {dim}) row vector such as `v.sym` of a vector MeshVariable."
        )
    raise ValueError(
        f"V_fn must be a (1, {dim}) sympy Matrix or a vector MeshVariable, "
        f"not {type(V_fn).__name__}."
    )


class Symbolic(_DDtBase):
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
        Accepted for interface parity with the projection-backed flavors
        (``Eulerian`` / ``SemiLagrangian``); Symbolic has no projection
        solver, so this is stored but unused (default ``[]``).
    order : int, optional
        Order of time integration (1-3) (default ``1``).
    smoothing : float, optional
        Accepted for interface parity with the projection-backed flavors;
        stored but unused by Symbolic (default ``0.0``).

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
        # bcs / smoothing are interface-parity parameters (see the class
        # docstring): stored so callers can treat all DDt flavors alike,
        # never read by Symbolic itself. The evalf argument threaded
        # through the update methods is likewise ignored here (there is
        # no numerical evaluation of a purely symbolic history).
        self.bcs = bcs
        self.verbose = verbose
        self.smoothing = smoothing
        self.order = order

        self._init_history_tracking(order)

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

        self._init_coefficient_expressions(order, self.theta, with_exp=True)

        # Register with the active default model as a Snapshottable
        # state-bearer. Safe if no model is active.
        self._register_with_default_model()

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
            **self._core_state_kwargs(),
            psi_star=list(self.psi_star),
        )

    @state.setter
    def state(self, s: "DDtSymbolicState") -> None:
        """Write a captured state back. Reconciles derived coefficients."""
        self._validate_state_schema(s, DDtSymbolicState)
        if len(s.psi_star) != len(self.psi_star):
            raise ValueError(
                f"psi_star length mismatch ({len(s.psi_star)} vs "
                f"{len(self.psi_star)}); order changed since snapshot?"
            )
        self.psi_star = list(s.psi_star)
        self._restore_core_state(s, am_theta=self.theta)

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
        # Local import: IPython is an optional, notebook-only dependency.
        from IPython.display import Latex, display

        # Display the primary variable
        display(Latex(rf"$\quad {self._psi_fn_symbol} = {sympy.latex(self._psi_fn)}$"))
        # Display the history variable using the different symbol.
        history_latex = ", ".join([sympy.latex(elem) for elem in self.psi_star])
        display(Latex(rf"$\quad {self._psi_star_symbol} = \left[{history_latex}\right]$"))

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

    def _history_syms(self):
        """Symbolic stores raw sympy matrices in ``psi_star`` — return them as-is."""
        return list(self.psi_star)

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


class Eulerian(_DDtBase):
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
        num_components=None,
    ):
        super().__init__()

        self.mesh = mesh
        self.V_fn = V_fn
        # With a velocity, the plain Eulerian flavour applies it as an
        # explicit splitting correction of the history ("split");
        # EulerianSUPG assembles it in the solver's residual instead.
        self._advection_mode = "split"
        self.theta = theta
        self.bcs = bcs
        self.verbose = verbose
        self.degree = degree
        self.vtype = vtype
        self.continuous = continuous
        self.smoothing = smoothing
        self.evalf = evalf
        self.num_components = num_components

        self._init_history_tracking(order)

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

        psi_star = []
        self.psi_star = psi_star

        for i in range(order):
            self.psi_star.append(
                uw.discretisation.MeshVariable(
                    f"psi_star_Eulerian_{self.instance_number}_{i}",
                    self.mesh,
                    num_components,
                    vtype=vtype,
                    degree=degree,
                    continuous=continuous,
                    varsymbol=rf"{varsymbol}^{{ {'*'*(i+1)} }}",
                )
            )

        self._init_coefficient_expressions(order, self.theta, with_exp=True)

        # Register with the active default model as a Snapshottable
        # state-bearer. Safe if no model is active.
        self._register_with_default_model()

        return

    @property
    def state(self) -> "DDtEulerianState":
        """Return a snapshot-of-state dataclass for this Eulerian DDt."""
        return DDtEulerianState(
            **self._core_state_kwargs(),
            psi_star_var_names=[ps.clean_name for ps in self.psi_star],
        )

    @state.setter
    def state(self, s: "DDtEulerianState") -> None:
        self._validate_state_schema(s, DDtEulerianState)
        self._validate_psi_star_names(s.psi_star_var_names)
        self._restore_core_state(s, am_theta=self.theta)

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
        # Local import: IPython is an optional, notebook-only dependency.
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        ## feedback on this instance
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
            # Manifold meshes (dim < cdim): use the multi-component
            # projection to sidestep SNES_Vector's pre-manifold
            # mesh.dim/cdim entanglement. See sibling block in the
            # SLCN init for the rationale.
            if self.mesh.dim != self.mesh.cdim:
                self._psi_star_projection_solver = uw.systems.solvers.SNES_MultiComponent_Projection(
                    self.mesh,
                    u_Field=self.psi_star[0],
                    n_components=self.mesh.cdim,
                    verbose=False,
                )
            else:
                self._psi_star_projection_solver = uw.systems.solvers.SNES_Vector_Projection(
                    self.mesh, self.psi_star[0], verbose=False
                )
        elif self.vtype == uw.VarType.SYM_TENSOR or self.vtype == uw.VarType.TENSOR:
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
            indep = self._psi_star_indep_indices
            row = sympy.Matrix([[self.psi_fn[i, j] for (i, j) in indep]])
            self._psi_star_projection_solver.uw_function = row
        else:
            self._psi_star_projection_solver.uw_function = self.psi_fn
        self._psi_star_projection_solver.bcs = self.bcs
        self._psi_star_projection_solver.smoothing = self.smoothing

    def update_history_fn(self):
        r"""Copy current :math:`\psi` to ``psi_star[0]`` via evaluation or projection.

        Three routes, in order of preference: direct nodal copy when
        tracking a mesh variable with the same layout; pointwise
        evaluation of ``psi_fn``; an L2 projection for expressions that
        ``evaluate`` cannot handle (e.g. containing derivatives).
        """
        if self._psi_meshVar is not None:
            try:
                self.psi_star[0].data[...] = self._psi_meshVar.data[...]
                return
            except ValueError:
                # Sanctioned fallthrough: the tracked variable's nodal
                # layout differs from psi_star's (different degree /
                # continuity), so the direct copy cannot broadcast —
                # evaluate psi_fn at psi_star's own nodes instead.
                pass

        try:
            self.psi_star[0].data[...] = uw.function.evaluate(
                self.psi_fn,
                self.psi_star[0].coords,
                evalf=self.evalf,
            ).reshape(-1, max(self.psi_fn.shape))
        except Exception:
            # Sanctioned fallback: evaluate() cannot interpolate
            # expressions containing derivatives (e.g. flux terms) —
            # project them onto psi_star[0] instead.
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
            warnings.warn(
                "set_initial_history called with order >= 2 but no "
                "dt — variable-dt BDF coefficients will be wrong on "
                "the first solve. Pass dt=<timestep> to suppress.",
                stacklevel=2,
            )
        return

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

        if self.V_fn is not None and dt is not None and self._advection_mode == "split":
            self._apply_split_advection(dt, evalf)

        return

    def _apply_split_advection(self, dt, evalf=False):
        """Explicit operator-splitting correction: ``psi_star[0] -= dt (V . grad) psi``."""
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

    def update_exp_coefficients(self, dt, tau_eff):
        r"""Update the ETD-2 (exponential) coefficient values for this step."""
        _update_exp_values(self._exp_coeffs, dt, tau_eff)


class EulerianSUPG(Eulerian):
    r"""Eulerian history manager that assembles its transport: implicit advection with SUPG.

    The transport plugin of the Eulerian solvers. It holds the history of
    one unknown on the mesh, as :class:`Eulerian` does, and contributes the
    three terms a solver composes its residual from: the time derivative of
    the multistep scheme, the implicit advection
    :math:`\sum_k w_k\,(\mathbf{a}_k\cdot\nabla)\psi^{(k)}` applied
    component-wise to a scalar, a vector or a tensor unknown, and the
    streamline-upwind Petrov-Galerkin flux :math:`\tau\,R\otimes\mathbf{a}`
    of the solver's strong residual :math:`R`. The same solver takes a
    :class:`SemiLagrangian` history in its place: that flavour answers zero
    for the advection and the flux because its history is already traced
    back along the characteristics.

    ``V_fn`` is data: the velocity the transport uses at the new level. The
    nonlinearity of a self-advected unknown lives in what ``V_fn`` is (the
    unknown's own symbol for Newton, an extrapolated or Picard field for a
    linear step), and ``V_fn_history`` names the velocity at the stored
    levels when it is not ``V_fn`` (the stored velocity itself for momentum).

    The stabilisation parameter is

    .. math::
        \tau = \frac{w}{\sqrt{(C_t c_0/\Delta t)^2 + (C_u|\mathbf{a}|/h)^2 + (C_\kappa\kappa/h^2)^2}}

    (``tau_shape="inverse_sum"``) with :math:`h` the local cell size,
    :math:`c_0` the leading multistep coefficient, :math:`\kappa` the
    :attr:`diffusivity` the solver declares (the diffusivity of a scalar,
    :math:`\eta/\rho` for momentum, zero for a transported stress) and
    :math:`w` the product of ``supg_weight`` and the cell-Péclet weight
    :math:`Pe^2/(Pe^2 + Pe_c^2)`, :math:`Pe = |\mathbf{a}|h/2\kappa`, which
    switches the term off where diffusion dominates. ``"brooks_hughes"`` and
    ``"doubly_asymptotic"`` are the optimal 1-D shapes, each capped by the
    transient term. Every weight is a runtime constant of the kernels.

    Parameters
    ----------
    mesh, psi_fn, vtype, degree, continuous, varsymbol, verbose, bcs, smoothing
        As for :class:`Eulerian`; ``psi_fn`` is the unknown's MeshVariable.
    V_fn : MeshVariable or sympy row Matrix
        The advecting velocity, ``(1, dim)``.
    order : int, default 1
        1 is the theta rule (Crank-Nicolson at ``theta=0.5``), 2 and 3 BDF.
    theta : float, optional
        Crank-Nicolson blend at order 1 (0.5 default; 1.0 backward Euler).
        Orders 2 and 3 take ``theta=1.0`` and refuse anything else.
    diffusivity : expression, default 0
        What :math:`\tau` sees as the diffusive rate; a solver sets it from
        its constitutive model when it builds its flux.
    supg_weight, tau_weights, tau_shape, peclet_weight
        The stabilisation knobs described above.
    num_components : tuple, optional
        The history variable shape when ``vtype`` is ``MATRIX``.
    """

    _TAU_SHAPES = ("inverse_sum", "brooks_hughes", "doubly_asymptotic")

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        psi_fn,
        V_fn,
        vtype: uw.VarType,
        degree: int,
        continuous: bool,
        order: int = 1,
        theta: Optional[float] = None,
        varsymbol: Optional[str] = r"u",
        verbose: Optional[bool] = False,
        bcs=None,
        smoothing: float = 0.0,
        diffusivity=0,
        supg_weight: float = 1.0,
        tau_weights=(2.0, 2.0, 4.0),
        tau_shape: str = "inverse_sum",
        peclet_weight: float = 4.0,
        num_components=None,
    ):
        order = int(order)
        if order not in (1, 2, 3):
            raise ValueError(f"order must be 1, 2 or 3, not {order}.")
        theta = float(theta) if theta is not None else (0.5 if order == 1 else 1.0)
        if theta != 1.0 and order != 1:
            raise ValueError(
                "theta applies at order 1 only (0.5 is Crank-Nicolson, 1.0 is "
                "backward Euler); order 2 and 3 take theta=1.0 (a BDF stencil "
                "pairs with terms at n+1, not with a centred flux)."
            )
        if tau_shape not in self._TAU_SHAPES:
            raise ValueError(f"tau_shape must be one of {self._TAU_SHAPES}, got {tau_shape!r}")

        # A caller's list is kept BY REFERENCE on purpose: a solver passes its
        # live essential_bcs so conditions added later reach the projections.
        # Only the default gets a fresh list, never a shared one.
        super().__init__(
            mesh, psi_fn, vtype, degree, continuous, V_fn=None, theta=theta,
            varsymbol=varsymbol, verbose=verbose, bcs=[] if bcs is None else bcs,
            order=order, smoothing=smoothing, num_components=num_components,
        )
        self._advection_mode = "assembled"
        self._integrator = "am" if order == 1 else "bdf"
        self.V_fn = V_fn
        self.V_fn_history = None
        self.diffusivity = diffusivity
        self._tau_shape = str(tau_shape)
        self._peclet_weight = float(peclet_weight)

        # The stabilisation knobs are runtime constants.
        tag = self.instance_number
        unique = dict(_unique_name_generation=True)
        self._supg_weight = _UWexpression(
            rf"w^{{\mathrm{{SUPG}}}}_{{{tag}}}", 1.0, "SUPG term weight (0 = Galerkin)", **unique)
        self._tau_weights = [
            _UWexpression(rf"C^{{\tau}}_{{t,{tag}}}", 2.0, "tau transient weight", **unique),
            _UWexpression(rf"C^{{\tau}}_{{u,{tag}}}", 2.0, "tau advective weight", **unique),
            _UWexpression(rf"C^{{\tau}}_{{\kappa,{tag}}}", 4.0, "tau diffusive weight", **unique),
        ]
        self.supg_weight = supg_weight
        self.tau_weights = tau_weights

    # ----- data -----

    @property
    def V_fn(self):
        """The advecting velocity at the new level, ``(1, dim)``."""
        return self._V_fn

    @V_fn.setter
    def V_fn(self, value):
        self._V_fn = None if value is None else _as_row_vector(value, self.mesh.dim)

    @property
    def integrator(self) -> str:
        return self._integrator

    def advecting_velocity(self, level: int = 0):
        """The velocity carrying the unknown at ``states()[level]``."""
        if level == 0 or not self.V_fn_history:
            return self.V_fn
        return _as_row_vector(self.V_fn_history[level - 1], self.mesh.dim)

    @property
    def tau_shape(self) -> str:
        return self._tau_shape

    @property
    def peclet_weight(self) -> float:
        return self._peclet_weight

    @property
    def supg_weight(self) -> float:
        """Scale of the SUPG term: 1 (default) or 0 for plain Galerkin. No rebuild."""
        return float(self._supg_weight.sym)

    @supg_weight.setter
    def supg_weight(self, value):
        self._supg_weight.sym = float(value)

    @property
    def tau_weights(self):
        r"""The weights :math:`(C_t, C_u, C_\kappa)` of the three terms in :math:`\tau`."""
        return tuple(float(w.sym) for w in self._tau_weights)

    @tau_weights.setter
    def tau_weights(self, values):
        for w, v in zip(self._tau_weights, values):
            w.sym = float(v)

    # ----- the contract -----

    def _convective(self, a, psi):
        r"""``(a . grad) psi`` entry by entry, a matrix of ``psi``'s shape."""
        dim = self.mesh.dim
        grad = self.mesh.vector.gradient

        def entry(r, c):
            g = grad(psi[r, c])
            return sum(a[0, i] * g[0, i] for i in range(dim))

        return sympy.Matrix(*psi.shape, entry)

    def advection(self):
        r""":math:`\sum_k w_k\,(\mathbf{a}_k\cdot\nabla)\psi^{(k)}` over the levels of the scheme."""
        total = sympy.zeros(*self._unknown_shape())
        for k, (w, psi_k) in enumerate(zip(self.spatial_weights(), self.states())):
            if w == 0:
                continue
            total = total + w * self._convective(self.advecting_velocity(k), psi_k)
        return total

    def tau(self):
        r"""The stabilisation parameter :math:`\tau` (times the weights)."""
        dim = self.mesh.dim
        a = self.advecting_velocity(0)
        a_mag2 = sum(a[0, i] ** 2 for i in range(dim))
        h = self.mesh.cell_size()
        nu = self.diffusivity
        if self.integrator == "bdf":
            c0 = self.bdf_coefficient_expressions[0]
        else:
            c0 = sympy.Integer(1)
        ct, cu, cv = self._tau_weights
        transient = (ct * c0 / self._delta_t) ** 2
        weight = self._supg_weight
        if self._peclet_weight > 0.0:
            # Pe^2 / (Pe^2 + Pe_c^2) written without dividing by nu (1 for nu = 0).
            ah2 = a_mag2 * h ** 2
            weight = weight * ah2 / (ah2 + 4 * self._peclet_weight ** 2 * nu ** 2 + 1.0e-30)
        if self._tau_shape == "inverse_sum":
            advective = (cu * sympy.sqrt(a_mag2) / h) ** 2
            viscous = (cv * nu / h ** 2) ** 2
            return weight / sympy.sqrt(transient + advective + viscous + 1.0e-30)
        # The 1-D optimal shapes: tau = (h / 2|a|) xi(Pe), Pe = |a| h / (2 nu).
        a_mag = sympy.sqrt(a_mag2 + 1.0e-30)
        Pe = a_mag * h / (2 * nu + 1.0e-30)      # finite at zero diffusivity (the default)
        if self._tau_shape == "brooks_hughes":
            xi = 1 / sympy.tanh(Pe) - 1 / Pe      # coth is not C99: the printer would rewrite it through exp
        else:
            xi = sympy.Min(Pe / 3, 1)
        tau_steady = h / (2 * a_mag) * xi
        return weight / sympy.sqrt(transient + 1 / (tau_steady ** 2 + 1.0e-30))

    def stabilisation_flux(self, R):
        r"""The SUPG flux :math:`\tau\,R\otimes\mathbf{a}`, one row per component of ``R``.

        ``R`` is the solver's strong residual of the unknown's shape (first
        derivatives only). The result has shape ``(len(R), dim)``: for a
        scalar the row :math:`\tau R\mathbf{a}`, for a vector
        :math:`F_{ij} = \tau R_i a_j`.
        """
        R = _as_matrix(R)
        column = R.reshape(len(R), 1)
        return self.tau() * (column * self.advecting_velocity(0))

    def _object_viewer(self):
        from IPython.display import Latex, display

        super()._object_viewer()
        display(Latex(r"$\quad\mathbf{a} = $ " + self.V_fn._repr_latex_()))
        display(Latex(rf"$\quad$ integrator: {self.integrator}, tau shape: {self.tau_shape}"))


class SemiLagrangian(_DDtBase):
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
        Number of history timesteps and, for the time-derivative operator,
        the order of the BDF backward-difference stencil taken *along the
        characteristic*:

        - ``order = 1`` → ``[1, -1]``: single-step difference
          ``(ψ^{n+1} - ψ*)/Δt``. Paired with a trapezoidal (Crank-Nicolson,
          ``theta=0.5``) flux this is the standard **SLCN** scheme — second-
          order accurate even though the stencil and the departure point
          are first-order, because the trapezoidal-along-the-trajectory
          structure recovers the order (Spiegelman & Katz, 2006).
        - ``order = 2`` → ``[3/2, -2, 1/2]``: BDF2 stencil
          ``(3/2 ψ^{n+1} - 2 ψ* + 1/2 ψ**)/Δt``, using two departure
          points. This is the **SL-BDF2** scheme. BDF2 is a one-sided
          implicit method: it expects the *flux evaluated at* ``n+1``
          only, i.e. a Backward-Euler-centred flux (``theta=1.0``), **not**
          Crank-Nicolson. SL-BDF2 reaches the same second order as SLCN but
          avoids the spurious resonance/ringing CN can show on stiff modes
          (Bonaventura et al., 2021).

        .. important::
           BDF (time-derivative) and Adams-Moulton/θ (flux) are *distinct*
           multistep families and must be **paired consistently**: SLCN =
           ``order=1`` + ``theta=0.5``; SL-BDF2 = ``order=2`` + ``theta=1.0``.
           Mixing a BDF2 stencil with a Crank-Nicolson flux (``order=2`` +
           ``theta=0.5``) centres the two sides at different times and is
           **not** a consistent second-order scheme. In
           :class:`~underworld3.systems.solvers.SNES_AdvectionDiffusion` the
           advective ``DuDt`` carries this BDF ``order`` while the diffusive
           ``DFDt`` carries the θ-method flux — set them as a matched pair.
    smoothing : float, default=0.0
        Smoothing parameter for projections.
    preserve_moments : bool, default=False
        Not implemented. Passing ``True`` raises ``NotImplementedError``.
        (The moment-preserving projection this promised was never wired in;
        the parameter is retained so existing call signatures keep working.)
    with_forcing_history : bool, default=False
        When True, allocate an additional ``forcing_star`` MeshVariable
        (matching ``psi_star[0]``'s shape, vtype, degree, continuity) to
        store one history slot for the strain-rate forcing. Required by
        ETD-2 exponential integration of the Maxwell relaxation operator;
        ignored for BDF/AM. Populated each step via
        :meth:`update_forcing_history` (direct nodal evaluation of
        ``forcing_fn`` — typically the constitutive model's strain-rate
        symbol).
    theta : float, default=0.5
        Adams-Moulton θ for the implicit flux integrator at order 1.
        The order-1 AM coefficients are ``[θ, 1-θ]``:

        - ``θ = 0.5`` → Crank-Nicolson (trapezoidal, second-order
          accurate, A-stable). Default, matches legacy SLCN behaviour.
        - ``θ = 1.0`` → Backward Euler (L-stable, monotone for
          diffusion, first-order accurate). Use for stiff parabolic
          terms (under-resolved sharp gradients on deformed cells)
          where CN's lack of L-stability causes sign-flip ringing
          on stiff modes.

        Settable after construction as a property:
        ``adv_diff.DuDt.theta = 1.0``.

    Notes
    -----
    The semi-Lagrangian method is particularly useful for:

    - Advection-dominated problems (high Péclet number)
    - Problems where CFL stability is restrictive
    - Viscoelastic stress advection

    The time-derivative (BDF ``order``) and the diffusive flux integrator
    (Adams-Moulton ``theta``) are separate choices that must be paired
    consistently — see ``order`` and ``theta`` above and the discussion in
    ``docs/advanced/semi-lagrangian-time-integration.md``.

    References
    ----------
    Spiegelman, M., & Katz, R. F. (2006). A semi-Lagrangian Crank-Nicolson
    algorithm for the numerical solution of advection-diffusion problems.
    *Geochemistry, Geophysics, Geosystems*, 7(4).
    https://doi.org/10.1029/2005GC001073

    Bonaventura, L., Calzola, E., Carlini, E., & Ferretti, R. (2021).
    Second order fully semi-Lagrangian discretizations of
    advection-diffusion-reaction systems. *Journal of Scientific Computing*,
    88, 23. https://doi.org/10.1007/s10915-021-01518-8 — SL-BDF2 reaches
    second order while avoiding the spurious resonance of CN-type schemes.

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
        monotone_mode: Optional[str] = None,
        theta: float = 0.5,
        old_frame_traceback: bool = False,
        midtime_velocity: bool = True,
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
        # Mid-point velocity of the RK2 trace at the mid TIME (1.5 v^n -
        # 0.5 v^{n-1}); False reproduces the pre-2026-09 v^n-only trace.
        self.midtime_velocity = bool(midtime_velocity)
        if preserve_moments:
            raise NotImplementedError(
                "preserve_moments is not currently implemented"
            )
        self.preserve_moments = preserve_moments
        self.with_forcing_history = with_forcing_history
        # Monotonicity limiter for the SL trace-back result. Bound
        # the FE-interpolated upstream sample to the local data range
        # of psi_star at each trace-back point. Cures the FE Lagrange
        # overshoot pattern in cells with sharp gradients while
        # preserving FE accuracy elsewhere.
        #   None    → pure FE (legacy; can overshoot at non-nodal
        #             points in cells with sharp gradients)
        #   "clamp" → B.2: clip FE result to [nbr_min, nbr_max] of
        #             the k=dim+1 nearest psi_star DOFs
        #   "pick"  → B.1: keep FE if in nbr bounds, else re-evaluate
        #             via RBF (Shepard) at out-of-bounds DOFs
        # Settable after construction:
        #   ``adv_diff.DuDt.monotone_mode = "clamp"``
        self.monotone_mode = monotone_mode
        # Adams-Moulton θ for the implicit flux at order 1.
        # The order-1 AM coefficients are ``[θ, 1-θ]``:
        #   θ=0.5  → Crank-Nicolson (A-stable, 2nd order accuracy on
        #            flux; NOT L-stable — stiff modes get amplification
        #            factor → -1, can ring on under-resolved sharp
        #            gradients in deformed cells)
        #   θ=1.0  → Backward Euler (L-stable, monotone for diffusion;
        #            1st order accuracy on flux)
        # Default 0.5 preserves the legacy SLCN behaviour.
        # Settable after construction:
        #   ``adv_diff.DuDt.theta = 1.0``
        self.theta = float(theta)

        # Old-frame semi-Lagrangian reach-back (Stage 0 of the
        # lagged-clone design, docs/developer/design/
        # lagged-clone-sl-history.md). On a moving mesh the standard
        # ALE trace-back samples the CARRY'd history on the NEW
        # geometry and subtracts v_mesh = Δx/dt to compensate the node
        # motion. That fold is lossy at a disequilibrium free surface
        # (it re-interpolates the new mesh for the exact-by-construction
        # old nodal value, and leaves a spurious normal component) and
        # blows up high-Ra free-surface convection (~step 20, Ra=1e5).
        #
        # With ``old_frame_traceback=True`` the trace-back instead:
        #   * computes the departure foot from the PHYSICAL velocity
        #     only (no v_mesh — V_fn must be the physical velocity, NOT
        #     v − v_mesh), x_dep = x_new − dt·V(x_new − ½dt·V); and
        #   * samples ``psi_star`` on the mesh EPHEMERALLY restored to
        #     the previous-step (old) geometry, where the foot is always
        #     representable (the old domain covers the vacated layer).
        # Mesh motion is then exact (known old node positions) and only
        # the physical advection is approximate, sampled where it is
        # always interpolable. ``on_remesh`` stashes the old geometry;
        # ``update_pre_solve`` consumes it. Mesh-agnostic: works for a
        # free surface or interior-node (mmpde/OT) adaptation alike.
        #
        # Settable after construction:
        #   ``adv_diff.DuDt.old_frame_traceback = True``
        self.old_frame_traceback = bool(old_frame_traceback)
        # One-step stash of the previous-step (old) geometry, set by
        # ``on_remesh`` and consumed (cleared) by ``update_pre_solve``.
        self._oldframe_X = None

        # Forcing-history storage. Allocated only if requested. Populated
        # each step via update_forcing_history(forcing_fn) — used by ETD-2
        # exponential integration of the Maxwell relaxation operator to
        # supply the ε̇ⁿ history term in the constitutive flux.
        self.forcing_star = None
        self._forcing_fn = None  # set by the constitutive model
        self._forcing_vtype = None
        self._forcing_indep_indices = None

        self._init_history_tracking(order)

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
            # Phase-2: operator-managed history; see psi_star block below
            self.forcing_star.remesh_policy = RemeshPolicy.CARRY
            self.forcing_star._remesh_managed_by = self

        self._init_coefficient_expressions(order, self.theta, with_exp=True)

        # Working variable that has a potentially different discretisation
        # from psi_star (swarm_degree / swarm_continuous rather than
        # degree / continuous): we project from this to psi_star, and it
        # defines the advection sample points. Kept per-instance, hence
        # the instance-number suffix. (The name previously carried a
        # trailing loop index leaked from the psi_star loop — accidental,
        # not meaningful.)
        self._workVar = uw.discretisation.MeshVariable(
            f"psi_work_sl_{self.instance_number}",
            self.mesh,
            vtype=vtype,
            degree=self.swarm_degree,
            continuous=self.swarm_continuous,
            varsymbol=rf"{{ {varsymbol}^\nabla }}",
            units=psi_units,  # Inherit units from psi_fn
        )

        # Phase-2 remesh redesign: mark every DDt-owned mesh variable as
        # CARRY + operator-managed so the generic per-variable REMAP pass
        # in remesh_with_field_transfer SKIPS them — the on_remesh hook
        # below handles the whole stack coherently (CARRY for ALE, or
        # explicit REMAP for an opt-out adapt like OT's reset). This
        # avoids interpolation diffusion of the history each adapt —
        # critical for preserving the time-scheme order at order >= 2.
        for _v in self.psi_star:
            _v.remesh_policy = RemeshPolicy.CARRY
            _v._remesh_managed_by = self
        self._workVar.remesh_policy = RemeshPolicy.CARRY
        self._workVar._remesh_managed_by = self

        # Historically this allocated a NodalPointSwarm cache here, but
        # the actual trace-back path uses ``uw.function.global_evaluate``
        # on the upstream coords directly — the swarm was vestigial and
        # nothing reads ``_nswarm_psi`` anywhere in the codebase. Skip
        # the allocation; on manifold meshes it would fail anyway because
        # DMSwarm's built-in coord field is dim-sized while manifold
        # coords are cdim-sized.
        self._nswarm_psi = None

        # The projection operator for mapping swarm values to the mesh - needs to be different for
        # each variable type, unfortunately ...

        if vtype == uw.VarType.SCALAR:
            self._psi_star_projection_solver = uw.systems.solvers.SNES_Projection(
                self.mesh, self.psi_star[0], verbose=False
            )
        elif vtype == uw.VarType.VECTOR:
            # On manifold meshes (dim < cdim) SNES_Vector has
            # pre-manifold mesh.dim/cdim entanglement in its FE
            # attachment + Jacobian construction. The flux projection
            # has no cross-component coupling though, so the
            # block-diagonal SNES_MultiComponent_Projection (with
            # n_components = cdim) is mathematically the right tool
            # and sidesteps SNES_Vector entirely. Volume meshes
            # continue to use SNES_Vector_Projection.
            if self.mesh.dim != self.mesh.cdim:
                self._psi_star_projection_solver = uw.systems.solvers.SNES_MultiComponent_Projection(
                    self.mesh,
                    u_Field=self.psi_star[0],
                    n_components=self.mesh.cdim,
                    verbose=False,
                )
            else:
                self._psi_star_projection_solver = uw.systems.solvers.SNES_Vector_Projection(
                    self.mesh,
                    self.psi_star[0],
                    verbose=False,
                )

        elif vtype == uw.VarType.SYM_TENSOR or vtype == uw.VarType.TENSOR:
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
            # Phase-2: operator-managed history flattening view
            self._psi_star_flat_var.remesh_policy = RemeshPolicy.CARRY
            self._psi_star_flat_var._remesh_managed_by = self
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
            indep = self._psi_star_indep_indices
            fn = self._workVar.sym
            row = sympy.Matrix([[fn[i, j] for (i, j) in indep]])
            self._psi_star_projection_solver.uw_function = row
        else:
            self._psi_star_projection_solver.uw_function = self._workVar.sym
        self._psi_star_projection_solver.bcs = bcs
        self._psi_star_projection_solver.smoothing = smoothing

        self._smoothing = smoothing

        # Register with the active default model as a Snapshottable
        # state-bearer. Safe if no model is active.
        self._register_with_default_model()

        # Phase-2 remesh redesign: register the adapt-time hook.
        # ``on_remesh`` accumulates Δx into ``_pending_v_mesh_disp``
        # (initialised here); the next ``update_pre_solve`` consumes
        # it as a one-step ``v_mesh`` pulse in the SL trace-back so
        # the CARRY'd history reads at the right upstream node. See
        # docs/developer/design/REMESH_FIELD_TRANSFER_DESIGN.md.
        self._pending_v_mesh_disp = None
        # Per-DDt temporary holding v_mesh = Δx / dt for the trace-back
        # (created lazily on first ALE consumption — see
        # _activate_ale_for_traceback below).
        self._v_mesh_var = None
        try:
            self.mesh.register_remesh_hook(self)
        except AttributeError:
            # Sanctioned swallow: an older Mesh without the remesh-hook
            # registry — this DDt then simply runs without adapt-time ALE.
            pass

        return

    def on_remesh(self, ctx):
        """Adapt-time hook: ALE for the SL history stack.

        Two branches:

        * **Standard ALE (smooth adapt).** The SL-owned vars
          (``psi_star[i]``, ``forcing_star``, ``_workVar``, the
          flattening view ``_psi_star_flat_var``) are CARRY +
          operator-managed — the generic per-variable pass already
          skipped them, and we leave their ``.data`` untouched here.
          Accumulate ``ctx.total_disp`` onto
          ``self._pending_v_mesh_disp`` so the next
          :meth:`update_pre_solve` runs the SL trace-back along
          ``(V_fn − v_mesh)`` with ``v_mesh = Δx / dt`` — that
          subtraction is exactly what compensates for the arbitrary
          mesh motion when reading the CARRY'd history. One-step
          pulse: the next solve consumes Δx and clears it.

        * **Opt-out (discrete-jump adapts).** When the adapt is a
          discrete jump rather than a smooth displacement
          (``ctx.scratch.get("ale_opt_out")``), the linear
          ``Δx/dt → v_mesh`` interpretation breaks down. Fall back to
          Phase-1 REMAP for this DDt's managed vars: call
          :func:`~underworld3.discretisation.remesh.remap_var_set` with
          the pre-move snapshot in ``ctx.managed_snapshot``. The
          pending ``v_mesh`` is cleared because REMAP already brought
          the history onto the new positions.

        Accumulation across multiple adapts before one solve is
        linear: ``v_mesh_disp += ctx.total_disp``. The trace-back uses
        the SUM divided by the next ``dt``, which is the correct
        node-frame velocity for that step.
        """

        # Which DDt-owned vars do I own? Collect from the mesh.vars
        # registry by managed-by identity (matches the stamping in
        # __init__).
        owned = [v for v in self.mesh.vars.values()
                 if getattr(v, "_remesh_managed_by", None) is self]

        if ctx.scratch.get("ale_opt_out"):
            # REMAP fallback. ctx.managed_snapshot holds my vars'
            # pre-move .data (the helper snapshots all managed vars).
            # remap_var_set deforms back, restores, evaluates at new
            # DOF coords, deforms forward, writes — exactly Phase-1
            # behaviour for this DDt's stack.
            snap = {v: ctx.managed_snapshot[v]
                    for v in owned if v in ctx.managed_snapshot}
            remap_var_set(self.mesh, owned,
                          ctx.old_X, ctx.new_X, snap)
            # The ALE pulse / old-frame reach are meaningless on a
            # discrete reset; clear any pending state so the next solve
            # does a plain (current-mesh) trace-back.
            self._pending_v_mesh_disp = None
            self._oldframe_X = None
            return

        # Old-frame reach-back: don't build a v_mesh pulse at all.
        # Stash the geometry the CARRY'd history corresponds to (the
        # mesh as of the last solve) so the next ``update_pre_solve``
        # can sample ``psi_star`` on it. The history .data is unchanged
        # (CARRY), so across multiple adapts before one solve we keep
        # the EARLIEST old_X — the geometry the data actually belongs
        # to — rather than overwriting with each intermediate move.
        if self.old_frame_traceback:
            if self._oldframe_X is None:
                self._oldframe_X = np.asarray(ctx.old_X).copy()
            return

        # Standard ALE: leave CARRY'd .data alone, accumulate Δx for
        # the next trace-back to consume.
        disp = ctx.total_disp
        if getattr(self, "_pending_v_mesh_disp", None) is None:
            self._pending_v_mesh_disp = np.array(disp, copy=True)
        else:
            self._pending_v_mesh_disp = (
                self._pending_v_mesh_disp + disp)

    @property
    def state(self) -> "DDtSemiLagrangianState":
        return DDtSemiLagrangianState(
            **self._core_state_kwargs(),
            psi_star_var_names=[ps.clean_name for ps in self.psi_star],
            forcing_star_var_name=(
                self.forcing_star.clean_name
                if self.forcing_star is not None else None
            ),
            with_forcing_history=bool(self.with_forcing_history),
        )

    @state.setter
    def state(self, s: "DDtSemiLagrangianState") -> None:
        self._validate_state_schema(s, DDtSemiLagrangianState)
        self._validate_psi_star_names(s.psi_star_var_names)
        if s.with_forcing_history != bool(self.with_forcing_history):
            raise ValueError(
                f"with_forcing_history flag differs between snapshot "
                f"({s.with_forcing_history}) and current "
                f"({self.with_forcing_history})"
            )
        self._restore_core_state(s, am_theta=self.theta)

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
        # Local import: IPython is an optional, notebook-only dependency.
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        display(Latex(rf"$\quad$History steps = {self.order}"))

    def initialise_history(self):
        r"""Initialize all history slots to the current value of :math:`\psi`.

        Called automatically on the first ``update_pre_solve``. Can also
        be called manually after setting initial conditions.
        """
        # Evaluate psi_fn at psi_star node positions and store in psi_star[0]

        coords_nd = _to_nondim_ndarray(self.psi_star[0].coords)

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
            warnings.warn(
                "set_initial_history called with order >= 2 but no "
                "dt — variable-dt BDF coefficients will be wrong on "
                "the first solve. Pass dt=<timestep> to suppress.",
                stacklevel=2,
            )
        return

    def _activate_ale_for_traceback(self, dt_for_calc):
        """Populate ``self._v_mesh_var`` for the upcoming ALE trace-back.

        Called from :meth:`update_pre_solve` when
        ``self._pending_v_mesh_disp`` is set. Creates ``_v_mesh_var``
        on first use (vector MeshVariable, degree 1, continuous —
        smooth enough for the trace-back's mid-point and is the
        cheapest discretisation that still resolves a per-node mesh
        velocity), and writes ``data = Δx / dt`` so the SL trace-back
        can evaluate ``V_fn − v_mesh`` at any point on the mesh by
        sympy subtraction or post-evaluation numpy subtraction.

        The variable is REINIT-policy: its values are valid for the
        next trace-back only, and the next adapt repopulates them
        fresh. The generic remesh pass skips it.

        Returns ``True`` if ALE is active (caller should subtract
        v_mesh at each V_fn evaluation), ``False`` otherwise.
        """
        disp = self._pending_v_mesh_disp
        if disp is None:
            return False
        # Lazily create the v_mesh field. dim matches the mesh's
        # coordinate dimension (so it broadcasts cleanly against the
        # mesh-vector V_fn values).
        if self._v_mesh_var is None:
            vname = f"_v_mesh_sl_{self.instance_number}"
            self._v_mesh_var = uw.discretisation.MeshVariable(
                vname, self.mesh, self.mesh.cdim, degree=1,
                continuous=True,
                varsymbol=rf"{{v^{{\mathrm{{mesh}}}}_{{[{self.instance_number}]}} }}",
                remesh_policy=RemeshPolicy.REINIT,
            )
        # disp has shape (n_nodes, cdim) and lives in mesh-coord
        # space; dt_for_calc is in the matching time scaling, so the
        # ratio is the correct mesh velocity in the same unit system
        # V_fn evaluations land in. If dt_for_calc is a Pint quantity,
        # extract its magnitude — v_mesh_var.data is plain numpy.
        try:
            _dt_val = float(getattr(dt_for_calc, "magnitude", dt_for_calc))
        except (TypeError, ValueError):
            _dt_val = float(dt_for_calc)
        if _dt_val == 0.0:
            # No time has elapsed → no mesh velocity. Defensive.
            self._v_mesh_var.data[...] = 0.0
        else:
            # _v_mesh_var lives on the *new* mesh; disp was captured
            # against the same node ordering at adapt time. Direct
            # nodal write is correct (no interpolation needed).
            self._v_mesh_var.data[...] = np.asarray(disp) / _dt_val
        return True

    def _consume_ale_pulse(self):
        """Clear the one-step v_mesh pulse after the trace-back has used it.

        Called at the end of :meth:`update_pre_solve` so subsequent
        non-adapt steps see ``self._pending_v_mesh_disp is None`` and
        run a plain trace-back. The MeshVariable storage is left in
        place but its values become stale (REINIT policy on the var
        guarantees the generic remesh pass leaves it alone; the next
        :meth:`_activate_ale_for_traceback` rewrites .data fresh).
        """
        self._pending_v_mesh_disp = None

    def _record_psi_star_from_field_data(self):
        """Parallel-safe 'record current field into psi_star[0]'.

        The default record step evaluates ``psi_fn`` at its own node
        coordinates, which under MPI mis-locates on-vertex points at a
        process seam (first-pass ``get_closest_cells`` + FE extrapolation),
        seeding a spurious history value. When ``psi_fn`` is a single
        mesh-variable component living on this mesh with the same nodal
        layout as ``psi_star[0]``, "evaluate at own nodes" is exactly that
        variable's nodal data, so we copy it directly — no point location.

        Returns an array shaped like ``psi_star[0].array`` for that case, or
        ``None`` (caller falls back to ``evaluate``) for non-scalar or
        expression ``psi_fn`` (e.g. a flux with derivatives).
        """
        try:
            comps = list(self.psi_fn)  # sympy Matrix, row-major
            if len(comps) != 1:                  # scoped to scalar fields
                return None
            hit = uw.discretisation.meshVariable_lookup_by_symbol(
                self.mesh, comps[0])
            if hit is None:
                return None
            var, comp = hit
            vflat = np.asarray(var.array)
            vflat = vflat.reshape(vflat.shape[0], -1)
            out = np.array(np.asarray(self.psi_star[0].array))
            oflat = out.reshape(out.shape[0], -1)
            if vflat.shape[0] != oflat.shape[0] or oflat.shape[1] != 1:
                return None
            oflat[:, 0] = vflat[:, comp]
            return out
        except Exception:
            return None

    def _midtime_velocity_expr(self):
        r"""Velocity at :math:`t^{n+1/2}` for the mid-point stage of the
        trace-back: :math:`\tfrac32 v^n - \tfrac12 v^{n-1}` once a previous
        velocity has been recorded, else :math:`v^n`. :math:`v^{n-1}` is
        ``V_fn`` as evaluated at the previous step and cached at the nodes,
        so any expression (``-v``, ``v/2``, ``c(t) v``, ``v - v_mesh``) is
        carried as it was then."""
        if not getattr(self, "midtime_velocity", True):
            return None
        level = getattr(self, "_v_prev_level", None)
        if level is None or not getattr(self, "_v_prev_valid", False):
            return None
        return self._V_matrix() * sympy.Rational(3, 2) - level["expr"] * sympy.Rational(1, 2)

    def _record_velocity_history(self):
        """Cache ``V_fn`` evaluated at the true nodes as v^{n-1} for the
        next step. Evaluating at NUDGED nodes left a 0.001 h |grad v| bias
        that the extrapolation fed into every trace and moved the
        Blankenbach 1a wall Nusselt number by 0.9 %."""
        if getattr(self, "_v_prev_level", None) is None:
            self._v_prev_level = self._make_velocity_level("n-1")
            self._v_prev_valid = False
        self._copy_velocity_level(self._v_prev_level)
        self._v_prev_valid = True

    def _centroid_shifted_var_coords(self, var):
        """ND node coordinates of ``var`` nudged 0.1 % toward their cell
        centroids (see :meth:`_centroid_shifted_node_coords`)."""
        coords = np.asarray(var.coords_nd)
        cellid = self.mesh.get_closest_cells(coords).reshape(-1)
        cent = np.asarray(self.mesh._centroids)[cellid]
        return 0.999 * coords + 0.001 * cent

    def _velocity_nd_at(
        self,
        coords,
        use_global: bool = False,
        evalf: bool = False,
        subtract_v_mesh: bool = False,
        expr=None,
    ):
        r"""Evaluate the advecting velocity at ``coords``, reduced to ND space.

        Shared by the node-point and mid-point legs of the RK2
        characteristic trace-back in :meth:`update_pre_solve` — the two
        legs differ only in evaluator routing, which is what the
        parameters express. Returns a plain ``(N, dim)`` array of
        NON-DIMENSIONAL velocity values, ready for the trace-back
        arithmetic ``x_dep = x - v·dt`` in the mesh's ND coordinate
        space (issue #267).

        Parameters
        ----------
        coords : numpy.ndarray
            ND sample coordinates, shape ``(N, dim)``.
        use_global : bool, optional
            Node points are rank-local, so the node leg uses
            ``uw.function.evaluate``; mid-points may have left the
            local partition, so the mid-point leg routes through
            ``uw.function.global_evaluate`` (which also forwards
            ``evalf``).
        evalf : bool, optional
            Forwarded to ``global_evaluate`` only (matching the
            original per-leg call signatures).
        subtract_v_mesh : bool, optional
            Phase-2 ALE: subtract the mesh velocity ``v_mesh = Δx/dt``
            sampled from ``self._v_mesh_var`` at the same points, so the
            trace-back runs along ``V − v_mesh``. Done after evaluation
            (rather than symbolically as ``V_fn − v_mesh.sym``) so the
            subtraction inherits the same unit treatment as ``V_fn``.
        """
        fn = self._V_matrix() if expr is None else expr
        if use_global:
            v_result = uw.function.global_evaluate(fn, coords, evalf=evalf)
            if subtract_v_mesh:
                v_mesh = uw.function.global_evaluate(
                    self._v_mesh_var.sym, coords, evalf=evalf
                )
                v_result = v_result - v_mesh
        else:
            v_result = uw.function.evaluate(fn, coords)
            if subtract_v_mesh:
                v_mesh = uw.function.evaluate(self._v_mesh_var.sym, coords)
                v_result = v_result - v_mesh

        # Slicing can drop the UnitAwareArray wrapper — rewrap before the
        # ND reduction so the units are not silently lost.
        if isinstance(v_result, UnitAwareArray):
            v_at_pts = v_result[:, 0, :]
            if not isinstance(v_at_pts, UnitAwareArray):
                v_at_pts = UnitAwareArray(v_at_pts, units=v_result.units)
        else:
            v_at_pts = v_result[:, 0, :]

        # Non-dimensionalise to the DM/ND space: the trace-back arithmetic
        # and the subsequent point-location both work in ND coordinates.
        return _to_nondim_ndarray(v_at_pts, units=uw.get_units(self.V_fn))

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

    def _shift_history_with_blend(self, dt, dt_physical=None):
        r"""Shift the history chain one slot, with optional time-lag blending.

        The history term is the nodal value of :math:`\psi` offset back
        along the characteristics according to the timestep:

        - ``psi_star[0]`` is the current value of ``psi_fn``, sampled at
          the location of the nodes in their previous position at
          :math:`t - \Delta t`;
        - ``psi_star[1]`` is the value of ``psi_star[0]`` from the
          previous timestep sampled at the node locations at
          :math:`t - \Delta t` (approximately the value of
          ``psi_star[0]`` at :math:`t - 2\Delta t`);
        - ``psi_star[2]`` etc. if required.

        This method performs the copy-down-the-chain step,
        :math:`\psi^*_i \leftarrow \varphi\,\psi^*_{i-1} +
        (1-\varphi)\,\psi^*_i`, working from the oldest slot so nothing
        is overwritten early. With ``dt_physical`` given,
        :math:`\varphi = \min(1, \Delta t / \Delta t_{\mathrm{phys}})`
        under-relaxes the shift when the numerical step outpaces the
        physical relaxation time; otherwise :math:`\varphi = 1` (plain
        shift).
        """
        if dt_physical is not None:
            phi = sympy.Min(1, dt / dt_physical)
        else:
            phi = sympy.sympify(1)

        for i in range(self.order - 1, 0, -1):
            self.psi_star[i].array[...] = (
                phi * self.psi_star[i - 1].array[...] + (1 - phi) * self.psi_star[i].array[...]
            )

    def _centroid_shifted_node_coords(self):
        r"""ND node coordinates of ``psi_star[0]``, nudged toward cell centroids.

        Point-location and FE interpolation are ambiguous exactly on
        element edges/vertices (worst on quad meshes at the domain
        boundary), so the sample points are moved 0.1 % of the way toward
        the centroid of their owning cell: far enough to make cell
        ownership unambiguous, close enough not to bias the sampled
        values. Coordinates are plain non-dimensional arrays — never raw
        ``.magnitude``, which would be dimensional metres (see
        ``_to_nondim_ndarray`` and issue #267).
        """
        psi_star_0_coords_nd = _to_nondim_ndarray(self.psi_star[0].coords)

        cellid = self.mesh.get_closest_cells(
            psi_star_0_coords_nd,
        )
        centroid_coords = self.mesh._centroids[cellid]

        shift = 0.001
        return (1.0 - shift) * psi_star_0_coords_nd[:, :] + shift * centroid_coords[
            :, :
        ]

    def _record_current_field_into_history(
        self, node_coords_nd, evalf, verbose, oldframe_active
    ):
        r"""Record the current value of :math:`\psi` into ``psi_star[0]``.

        Three routes, in order of preference:

        1. direct nodal copy of the tracked field's data (parallel, or
           old-frame reach-back);
        2. pointwise evaluation of ``psi_fn`` at the centroid-shifted
           node coordinates (the validated serial path);
        3. an L2 projection for expressions that ``evaluate`` cannot
           handle (e.g. the NS viscous flux, which contains derivatives).
        """
        try:
            # Use shifted ND coords to avoid quad mesh boundary issues
            # node_coords_nd is slightly shifted toward cell centroids
            # evaluate() treats plain numpy as ND [0-1] coordinates.
            #
            # PARALLEL band-aid (parallel-singular-corruption, 2026-05):
            # this "record current field into psi_star" step samples psi_fn
            # at its OWN node coords. On-vertex sampling + first-pass
            # get_closest_cells mis-locates at a process seam under MPI,
            # recording a spurious history value that the implicit solve
            # then propagates (the seam spike in adaptive advection-
            # diffusion). When psi_fn is a single mesh-variable component on
            # this mesh (the SLCN adv-diff case), "evaluate at own nodes" ==
            # the field's nodal data, so under MPI copy it directly (exact,
            # no point location). Serial keeps the validated shifted-
            # evaluate path bit-identically; non-scalar / expression psi_fn
            # falls back to evaluate(). Proper fix (remap-on-adapt / ALE)
            # tracked separately.
            # Old-frame: record the history by a DIRECT nodal carry
            # of the field rather than re-evaluating psi_fn at the
            # (centroid-shifted) nodes of the DEFORMED mesh. The
            # re-evaluate injects boundary-layer interpolation error
            # that grows with mesh distortion and then rides the
            # old-geometry sample below — the exact value we want is
            # the carried nodal value (cf. the lagged-clone "store
            # primitives" principle). Reuses the parallel direct-copy
            # path, which returns None for non-scalar / expression
            # psi_fn (those fall back to evaluate).
            _direct = (self._record_psi_star_from_field_data()
                       if (uw.mpi.size > 1 or oldframe_active) else None)
            if _direct is not None:
                eval_result = _direct
            else:
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
            # (e.g., containing derivatives — true for the NS viscous flux every step).
            # Route via _build_projection_source so the (1, Nc) row-matrix flattening
            # required by SNES_MultiComponent_Projection is applied for tensor vtypes.
            # Without this, a (dim, dim) tensor function meets a (1, Nc) solver field
            # and SymPy raises "Matrix size mismatch: (1, Nc) + (dim, dim)" (issue #180).
            self._psi_star_projection_solver.uw_function = self._build_projection_source(
                self.psi_fn
            )
            self._psi_star_projection_solver.smoothing = 0.0
            self._psi_star_projection_solver.solve(verbose=verbose)

            # For tensor vtypes the projection writes into the flat (1, Nc) variable,
            # so we must fan it back out to psi_star[0] — otherwise subsequent
            # history operations read a stale tensor. Mirrors the same fan-out in
            # the projection fallback of initialise_history().
            if getattr(self, '_psi_star_use_multicomponent', False):
                for k, (i, j) in enumerate(self._psi_star_indep_indices):
                    vals = self._psi_star_flat_var.array[:, 0, k]
                    self.psi_star[0].array[:, i, j] = vals
                    if i != j:
                        self.psi_star[0].array[:, j, i] = vals

    def _nondim_timestep(self, dt):
        r"""Reduce ``dt`` to a plain non-dimensional model-time value.

        The semi-Lagrangian trace-back is performed ENTIRELY in the mesh's
        NON-DIMENSIONAL (DM) coordinate space: evaluate()/global_evaluate
        treat plain arrays as DM coords and the DM point-location uses DM
        values (0..L_model, NOT dimensional metres). So coords, velocity
        AND dt are all reduced to non-dimensional values, whether or not
        the model carries units. (Previously the has_units branch kept
        dimensional coords/velocity and left dt unitless -> a 'meter' vs
        'meter/second' subtraction crash and mislocation against the ND
        DM; UW3 issue #267.)
        """
        if hasattr(dt, "magnitude") or hasattr(dt, "value"):
            # dt carries units -> non-dimensionalise it
            dt_nondim = uw.non_dimensionalise(dt, uw.get_default_model())
            if hasattr(dt_nondim, "magnitude"):
                return float(dt_nondim.magnitude)
            elif hasattr(dt_nondim, "value"):
                return float(dt_nondim.value)
            else:
                return float(dt_nondim)
        else:
            # already non-dimensional model-time
            return dt

    def _trace_departure_points(
        self, i, node_coords_nd, dt_for_calc, evalf, subtract_v_mesh, oldframe_active
    ):
        r"""RK2 midpoint trace-back: departure points for history slot ``i``.

        Traces the characteristic backwards from each node,

        .. math::

            x_{\mathrm{mid}} = x - \tfrac{1}{2}\Delta t\, v(x), \qquad
            x_{\mathrm{dep}} = x - \Delta t\, v(x_{\mathrm{mid}}),

        entirely in the mesh's ND coordinate space. Midpoints are clamped
        to the domain; departure points are clamped unless the old-frame
        reach-back is active (the foot is then sampled on the OLD
        geometry, whose domain covers the layer a moving surface vacated
        — clamping to the new-mesh bounds would pull valid old-domain
        feet onto the boundary; the monotone limiter on the sample bounds
        any foot that falls outside the old mesh, matching the validated
        prototype, which omits this clamp).
        """
        # Use shifted ND coords to avoid quad mesh boundary issues
        # (node_coords_nd is slightly shifted toward cell centroids —
        # see _centroid_shifted_node_coords)
        v_at_node_pts = self._velocity_nd_at(
            node_coords_nd, subtract_v_mesh=subtract_v_mesh
        )

        # Departure point in the mesh's ND (DM) coordinate space. coords_nd is
        # the ND reduction of the (possibly dimensional) node coordinates —
        # identical to .coords for a non-units model, and the DM-space values
        # (0..L_model) when units are active, matching what global_evaluate /
        # the DM point-location expect. See #267.
        coords = np.asarray(self.psi_star[i].coords_nd)

        # CRITICAL (2025-11-27): Multiply velocity FIRST so UnitAwareArray.__mul__ handles it.
        # If we do `dt_for_calc * v_at_node_pts`, Pint handles it and loses UnitAwareArray units.
        mid_pt_coords = coords - v_at_node_pts * (0.5 * dt_for_calc)

        # Clamp midpoint coordinates to the domain boundary
        if self.mesh.return_coords_to_bounds is not None:
            mid_pt_coords = self.mesh.return_coords_to_bounds(mid_pt_coords)

        # Mid-point velocities may lie off-rank, so route through
        # global_evaluate (with evalf forwarded), unlike the on-node
        # evaluation above. The mid-point velocity is taken at the mid
        # TIME, t^{n+1/2}, by extrapolation from the two most recent
        # velocity fields, 1.5 v^n - 0.5 v^{n-1}; with v^n alone the
        # trace is only first order in an unsteady flow. On the first
        # step (no previous velocity) v^n is used.
        v_at_mid_pts = self._velocity_nd_at(
            mid_pt_coords,
            use_global=True,
            evalf=evalf,
            subtract_v_mesh=subtract_v_mesh,
            expr=self._midtime_velocity_expr(),
        )

        # Upstream (departure) coordinates: current position - velocity * timestep
        end_pt_coords = coords - v_at_mid_pts * dt_for_calc

        if (self.mesh.return_coords_to_bounds is not None
                and not oldframe_active):
            end_pt_coords = self.mesh.return_coords_to_bounds(end_pt_coords)

        return end_pt_coords

    def _sample_history_at_departure(
        self, i, end_pt_coords, evalf, monotone_mode, oldframe_active, oldframe_X
    ):
        r"""Sample ``psi_star[i]`` at its departure points and store back.

        The upstream sample is the semi-Lagrangian history value:
        :math:`\psi^*_i(x) \leftarrow \psi^*_i(x_{\mathrm{dep}})`.
        """
        # Extract scalar from (1,1) Matrix for scalar variables
        # MeshVariable.sym returns Matrix([[value]]) for scalars
        expr_to_evaluate = self.psi_star[i].sym
        if hasattr(expr_to_evaluate, 'shape') and expr_to_evaluate.shape == (1, 1):
            expr_to_evaluate = expr_to_evaluate[0, 0]

        # Evaluate psi_star at upstream coordinates
        # global_evaluate now returns dimensional results (gateway fix 2025-11-28)
        # When evalf=True, route through RBF (Shepard, bounded by
        # neighbour values) instead of FE shape functions. FE
        # Lagrange P3 can overshoot at non-nodal upstream points
        # in cells with sharp gradients — observed as the 'pepper'
        # DOF scatter that ignites catastrophic ringing on free-
        # surface convection at high Ra.
        #
        # The monotonicity limiter (B.1 "pick" / B.2 "clamp") that
        # bounds the FE/RBF trace-back to the local data range of
        # psi_star now lives in the evaluator as the `monotone`
        # option (uw.function.global_evaluate), so any resampling
        # path can request the same bounded result. monotone_mode is
        # None in the default trajectory → no-op (bit-identical).
        # Old-frame: sample psi_star on the mesh ephemerally
        # restored to the previous-step (old) geometry. The foot
        # (end_pt_coords) was computed in the current frame from the
        # physical velocity; the old mesh covers the old domain so
        # the foot is representable there with no extrapolation.
        # ``ephemeral_coords`` snapshots the current (new) geometry
        # and restores it on exit; ``_deform_mesh`` only rebuilds the
        # DS / DOF-coordinate caches, leaving every variable's nodal
        # .data untouched (de-risked: bit-identical round-trip), so
        # psi_star realises "the old field on the old geometry".
        if oldframe_active:
            with self.mesh.ephemeral_coords():
                self.mesh._deform_mesh(oldframe_X)
                value_at_end_points = uw.function.global_evaluate(
                    expr_to_evaluate,
                    end_pt_coords,
                    evalf=evalf,
                    monotone=monotone_mode,
                )
        else:
            value_at_end_points = uw.function.global_evaluate(
                expr_to_evaluate,
                end_pt_coords,
                evalf=evalf,
                monotone=monotone_mode,
            )

        # CRITICAL FIX (2025-11-27): If psi_star has units, ensure the assigned
        # value also has units. global_evaluate may return plain arrays.
        psi_star_units = self.psi_star[i].units
        if psi_star_units is not None and not isinstance(value_at_end_points, UnitAwareArray):
            value_at_end_points = UnitAwareArray(value_at_end_points, units=psi_star_units)

        self.psi_star[i].array[...] = value_at_end_points

        # TODO(DESIGN): a moment-preserving correction (restore mean and L2
        # moment of psi_star after the semi-Lagrangian update) was removed
        # here as dead code — see git history for the sketch if the
        # `preserve_moments` option is ever implemented.

    def update_pre_solve(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
        dt_physical: Optional[float] = None,
        store_result: Optional[bool] = True,
        monotone_mode: Optional[str] = "__instance__",
    ):
        """Sample upstream values along characteristics before solve.

        On the first call, automatically initialises history from the
        current field values so that bdf() returns zero on the first step.

        The method reads as its four phases: shift the history chain
        (:meth:`_shift_history_with_blend`), record the current field
        (:meth:`_record_current_field_into_history`), trace the
        characteristics back (:meth:`_trace_departure_points`), and
        sample the history at the departure points
        (:meth:`_sample_history_at_departure`).

        Parameters
        ----------
        store_result : bool, optional
            If True (default), evaluate psi_fn at current positions and store
            in psi_star[0] before advection — the standard DDt behaviour.
            If False, skip this step and the history shift: only advect the
            existing psi_star levels upstream. Used by VE_Stokes where
            psi_star[0] already contains the projected actual stress from
            the previous solve.
        monotone_mode : str or None, optional
            Override the instance ``self.monotone_mode`` for this call.
            Default ``"__instance__"`` (sentinel) means "use whatever is
            on the instance". Pass ``None``, ``"clamp"``, or ``"pick"``
            to force a particular mode for one call.
        """

        self._dt = dt

        # Resolve monotone_mode: explicit kwarg overrides instance attr.
        if monotone_mode == "__instance__":
            monotone_mode = getattr(self, "monotone_mode", None)

        if not self._history_initialised:
            self.initialise_history()

        # Old-frame reach-back (mutually exclusive with the ALE pulse:
        # ``on_remesh`` stashes ``_oldframe_X`` INSTEAD of a v_mesh disp,
        # so ``_ale_active`` is False below whenever this is True). When
        # active the foot is computed from the physical V (no v_mesh) and
        # ``psi_star`` is sampled on the mesh ephemerally restored to the
        # old geometry — see the ``global_evaluate`` block in the loop.
        # Computed up here because it also governs how psi_star[0] is
        # re-recorded (direct nodal carry, not a lossy re-evaluate on the
        # deformed mesh).
        _oldframe_active = (self.old_frame_traceback
                            and self._oldframe_X is not None)
        _oldframe_X = self._oldframe_X

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
        _update_am_values(self._am_coeffs, self.effective_order, self.theta)

        # 1. Shift the history chain down one slot (oldest first), with
        #    the optional dt_physical blend. Skipped when store_result is
        #    False (VE stress history: psi_star[0] is managed by the solve).
        if store_result:
            self._shift_history_with_blend(dt, dt_physical)

        # 2. Record the current value of psi_fn into psi_star[0]
        #    (direct copy / evaluate / projection — see the helper).
        #    When store_result=False (e.g. VE stress history), skip this —
        #    psi_star[0] already contains the projected actual stress from
        #    the previous solve and we want to advect *that*, not the flux.
        node_coords_nd = self._centroid_shifted_node_coords()

        if store_result:
            self._record_current_field_into_history(
                node_coords_nd, evalf, verbose, _oldframe_active
            )

        # 3. Trace the characteristics back and sample each history slot
        #    at its departure points. Work from the oldest slot backwards
        #    so we don't overwrite history terms we still need to sample.
        dt_for_calc = self._nondim_timestep(dt)

        # Phase-2 ALE: if an adapt stashed Δx, build v_mesh = Δx / dt as
        # a per-DDt MeshVariable now so the trace-back below can use
        # (V_fn − v_mesh) at any sample point. One-step pulse — cleared
        # at the end of this method.
        _ale_active = self._activate_ale_for_traceback(dt_for_calc)

        for i in range(self.order - 1, -1, -1):
            # Per-history-index ALE gating. The v_mesh correction
            # transforms a CARRY'd field's sampling from new-mesh to
            # old-mesh frame: psi_star_NEW(p) ≈ psi_star_OLD(p − Δx),
            # so to sample the OLD field at p_world we read the NEW
            # field at p_world + Δx, which is exactly what
            # X − (V − v_mesh)·dt achieves. This is only correct when
            # psi_star[i] still holds CARRY'd OLD data. When
            # ``store_result=True`` the i=0 slot is OVERWRITTEN by the
            # band-aid re-record above (psi_star[0] := current
            # psi_fn at new-mesh nodes), so for that slot the v_mesh
            # correction would double-shift — sample current T at
            # X + Δx instead of at X − V·dt. Gate it off for i=0 in
            # that case. (See REMESH_FIELD_TRANSFER_DESIGN.md §2a:
            # the order-1 re-record is the reason ALE buys nothing at
            # order=1 with theta=1 — the band-aid already did the
            # right thing.)
            _ale_this_iter = _ale_active and not (store_result and i == 0)

            end_pt_coords = self._trace_departure_points(
                i, node_coords_nd, dt_for_calc, evalf,
                _ale_this_iter, _oldframe_active,
            )
            self._sample_history_at_departure(
                i, end_pt_coords, evalf, monotone_mode,
                _oldframe_active, _oldframe_X,
            )

        # The velocity used this step becomes v^{n-1} for the next
        # step's mid-time extrapolation.
        if getattr(self, "midtime_velocity", True):
            self._record_velocity_history()

        # Phase-2 ALE: consume the one-step v_mesh pulse. Subsequent
        # non-adapt steps will see no pending displacement and run a
        # plain trace-back. If multiple adapts happen before the next
        # solve, on_remesh accumulates them additively and one
        # consumption clears the lot.
        if _ale_active:
            self._consume_ale_pulse()

        # Old-frame: consume the one-step old-geometry stash. The next
        # adapt re-stashes; a non-adapting step sees None and traces
        # back on the current mesh (old geom == new geom).
        if _oldframe_active:
            self._oldframe_X = None

        return

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

        # Use non-dimensional coords for evaluate() (mirrors the psi_star
        # path in update_pre_solve)
        coords_nd = _to_nondim_ndarray(self.forcing_star.coords)

        def _eval_nd(component_expr):
            """Evaluate component at coords and non-dimensionalise to a
            plain 1-D float array suitable for nodal storage."""
            result = uw.function.evaluate(component_expr, coords_nd, evalf=evalf)
            # If the evaluation returned units (model is unit-aware),
            # non-dimensionalise before storing — keeps forcing_star's
            # internal storage non-dimensional like psi_star.
            return np.asarray(_to_nondim_ndarray(result)).flatten()

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


class Lagrangian(_DDtBase):
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
        dudt_swarm = uw.swarm.Swarm(mesh)

        self.mesh = mesh
        self.swarm = dudt_swarm
        self.psi_fn = psi_fn
        self.V_fn = V_fn
        self.verbose = verbose
        self.order = order

        self._init_history_tracking(order)

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

        # No user-settable theta on the swarm-based Lagrangian flavors:
        # Crank-Nicolson (0.5) is their fixed Adams-Moulton value.
        self._init_coefficient_expressions(order, 0.5, with_exp=False)

        dudt_swarm.populate(fill_param)

        # Register with the active default model as a Snapshottable
        # state-bearer. Safe if no model is active.
        self._register_with_default_model()

        # Phase-1 remesh redesign: register the adapt-time hook on the
        # mesh. Lagrangian's psi_star history lives on a swarm, not on
        # the mesh — its transfer is the swarm's own particle migration
        # under the deformed cells, which is already correct.  Phase 1
        # hook is therefore a no-op (no mesh-side state to transfer);
        # Phase 2 may attach ALE-style annotation here in parallel with
        # SemiLagrangian.
        try:
            self.mesh.register_remesh_hook(self)
        except AttributeError:
            # Sanctioned swallow: an older Mesh without the remesh-hook
            # registry — this DDt then simply runs without adapt-time ALE.
            pass

        return

    def on_remesh(self, ctx):
        """Adapt-time hook (Phase 1 no-op).

        Lagrangian history is carried by the underlying swarm (each
        particle holds its own ``psi_star`` values), so there is no
        mesh-side state to transfer on an adapt — the swarm's particle
        positions stay put and the new cells re-claim them. Method is
        defined so the registration shim has a target.
        """
        del ctx  # Phase 1: explicitly unused
        return

    @property
    def state(self) -> "DDtLagrangianState":
        return DDtLagrangianState(
            **self._core_state_kwargs(),
            psi_star_var_names=[ps.clean_name for ps in self.psi_star],
        )

    @state.setter
    def state(self, s: "DDtLagrangianState") -> None:
        self._validate_state_schema(s, DDtLagrangianState)
        self._validate_psi_star_names(s.psi_star_var_names)
        # No theta parameter on this flavor — fixed Crank-Nicolson value.
        self._restore_core_state(s, am_theta=0.5)

    def _object_viewer(self):
        # Local import: IPython is an optional, notebook-only dependency.
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        ## feedback on this instance
        # Note: dt_physical is not tracked on the Lagrangian DDt classes,
        # so the viewer reports the expression and history depth only.
        display(Latex(r"$\quad\psi = $ " + sympy.sympify(self.psi_fn)._repr_latex_()))
        display(Latex(rf"$\quad$History steps = {self.order}"))

    def initialise_history(self):
        r"""Initialize all history slots to the current value of :math:`\psi`.

        Called automatically on the first ``update_pre_solve``. Can also
        be called manually after setting initial conditions.
        """
        psi_star_0 = self.psi_star[0]
        # Component-wise write through the canonical (N, components) storage.
        # Indexing the SwarmVariable itself (``psi_star_0[i, j]``) returns a
        # *symbolic* component with no ``.data`` — the modern component
        # address is ``.data[:, var._data_layout(i, j)]`` (audit SWARM-06).
        coords = np.asarray(self.swarm._particle_coordinates.data)
        for i in range(psi_star_0.shape[0]):
            for j in range(psi_star_0.shape[1]):
                updated_psi = uw.function.evaluate(
                    self.psi_fn[i, j],
                    coords,
                )
                psi_star_0.data[:, psi_star_0._data_layout(i, j)] = np.asarray(
                    updated_psi
                ).reshape(-1)

        # Copy to all other history slots
        for k in range(1, self.order):
            self.psi_star[k].data[...] = psi_star_0.data[...]

        self._history_initialised = True
        return

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

    def _proxy_values_at_particles(self, slot, coords, evalf):
        """The slot's proxy evaluated at the particles, shaped like ``slot.data``.

        A ``"cells"`` proxy is read through its own fitted polynomials (exact,
        no locator round trip); any other proxy through ``evaluate`` of the
        proxy mesh variable's symbol.
        """
        projector = getattr(slot, "_cell_projector", None)
        if projector is not None and getattr(slot, "_proxy_location", None) == "cells":
            slot._update_proxy_if_stale()
            vals = projector.interpolate(np.asarray(slot._meshVar.data), coords)
            return np.nan_to_num(vals)
        mv = slot._meshVar
        out = np.empty((coords.shape[0], slot.data.shape[1]))
        for i in range(slot.shape[0]):
            for j in range(slot.shape[1]):
                ij = slot._data_layout(i, j)
                out[:, ij] = np.asarray(
                    uw.function.evaluate(mv.sym[i, j], coords, evalf=evalf)
                ).reshape(-1)
        return out

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
        # Grab the current psi values at the (pre-advection) particle
        # positions via the canonical component storage (audit SWARM-06).
        coords = np.asarray(self.swarm._particle_coordinates.data)
        for i in range(psi_star_0.shape[0]):
            for j in range(psi_star_0.shape[1]):
                updated_psi = uw.function.evaluate(
                    self.psi_fn[i, j],
                    coords,
                    evalf=evalf,
                )
                psi_star_0.data[:, psi_star_0._data_layout(i, j)] = np.asarray(
                    updated_psi
                ).reshape(-1)

        # Now update the swarm locations

        self.swarm.advection(
            self.V_fn,
            delta_t=dt,
            restore_points_to_domain_func=self.mesh.return_coords_to_bounds,
        )

        if self._n_solves_completed < self.order:
            self._n_solves_completed += 1


class Lagrangian_Swarm(_DDtBase):
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
    proxy_location : {"nodes", "integration_points", "cells"}, optional
        Where each history slot's proxy lives; ``"integration_points"``
        reconstructs the particle history at the integration points and the
        weak form reads it there directly, with no nodal proxy and no basis
        interpolation (the Ellipsis / Underworld PIC-LIP mapping);
        ``"cells"`` fits a least-squares polynomial of degree ``degree`` per
        cell, exact for polynomial histories and integrated exactly by the
        default rule.
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
        proxy_location="nodes",
        particle_update="pic",
        residual_retention=1.0,
    ):
        super().__init__()

        self.mesh = swarm.mesh
        self.swarm = swarm
        self.psi_fn = psi_fn
        self.verbose = verbose
        self.order = order
        self.step_averaging = step_averaging
        if particle_update not in ("pic", "flip"):
            raise ValueError(f"particle_update must be 'pic' or 'flip', not {particle_update!r}")
        # "pic": after a solve every particle takes the mesh solution at its
        # position (blended over step_averaging steps), so sub-cell particle
        # detail is re-projected away each step. "flip": the particle keeps
        # its own value and adds the mesh INCREMENT, solution minus the proxy
        # the mesh saw, evaluated at the particle; the sub-cell residual
        # survives, scaled by residual_retention (1 = FLIP, 0 = PIC; set it
        # to exp(-kappa dt pi^2 / l^2) to let a diffusing residual decay).
        self.particle_update = particle_update
        self.residual_retention = residual_retention
        # "integration_points": each slot's proxy is an IntegrationPointVariable
        # reconstructed from the particles at the rule and read there directly
        # (the Ellipsis / Underworld PIC-LIP mapping); no nodal proxy.
        # "cells": each slot's proxy is a least-squares polynomial per cell
        # (discontinuous, degree `degree`), exact for polynomial histories
        # and integrated exactly by the default rule.
        self.proxy_location = proxy_location

        self._init_history_tracking(order)

        # Sample the history before the particles first move. Left to the
        # first update_pre_solve, the sampling happens AFTER the user's
        # swarm.advection() and the first step transports nothing (a
        # one-step lag, measured as 0.05 of displacement on the rotating
        # Gaussian, 2026-09-08). Weak reference: the swarm must not own us.
        import weakref

        _self = weakref.ref(self)

        def _initialise_before_first_move():
            mgr = _self()
            if mgr is not None and not mgr._history_initialised:
                mgr.initialise_history()

        hooks = getattr(swarm, "_pre_advection_hooks", None)
        if hooks is not None:
            hooks.append(_initialise_before_first_move)

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
                    proxy_location=proxy_location,
                    varsymbol=rf"{varsymbol}^{{ {'*'*(i+1)} }}",
                )
            )

        # No user-settable theta on the swarm-based Lagrangian flavors:
        # Crank-Nicolson (0.5) is their fixed Adams-Moulton value.
        self._init_coefficient_expressions(order, 0.5, with_exp=False)

        # Register with the active default model as a Snapshottable
        # state-bearer. Safe if no model is active.
        self._register_with_default_model()

        return

    @property
    def state(self) -> "DDtLagrangianSwarmState":
        return DDtLagrangianSwarmState(
            **self._core_state_kwargs(),
            psi_star_var_names=[ps.clean_name for ps in self.psi_star],
        )

    @state.setter
    def state(self, s: "DDtLagrangianSwarmState") -> None:
        self._validate_state_schema(s, DDtLagrangianSwarmState)
        self._validate_psi_star_names(s.psi_star_var_names)
        # No theta parameter on this flavor — fixed Crank-Nicolson value.
        self._restore_core_state(s, am_theta=0.5)

    def _object_viewer(self):
        # Local import: IPython is an optional, notebook-only dependency.
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        ## feedback on this instance
        # Note: dt_physical is not tracked on the Lagrangian DDt classes,
        # so the viewer reports the expression and history depth only.
        display(Latex(r"$\quad\psi = $ " + sympy.sympify(self.psi_fn)._repr_latex_()))
        display(Latex(rf"$\quad$History steps = {self.order}"))

    def initialise_history(self):
        r"""Initialize all history slots to the current value of :math:`\psi`.

        Called automatically on the first ``update_pre_solve``. Can also
        be called manually after setting initial conditions.
        """
        psi_star_0 = self.psi_star[0]
        # Component-wise write through the canonical (N, components) storage.
        # Indexing the SwarmVariable itself (``psi_star_0[i, j]``) returns a
        # *symbolic* component with no ``.data`` — the modern component
        # address is ``.data[:, var._data_layout(i, j)]`` (audit SWARM-06).
        coords = np.asarray(self.swarm._particle_coordinates.data)
        for i in range(psi_star_0.shape[0]):
            for j in range(psi_star_0.shape[1]):
                updated_psi = uw.function.evaluate(
                    self.psi_fn[i, j],
                    coords,
                )
                psi_star_0.data[:, psi_star_0._data_layout(i, j)] = np.asarray(
                    updated_psi
                ).reshape(-1)

        # Copy to all other history slots
        for k in range(1, self.order):
            self.psi_star[k].data[...] = psi_star_0.data[...]

        self._history_initialised = True
        return

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

    def _proxy_values_at_particles(self, slot, coords, evalf):
        """The slot's proxy evaluated at the particles, shaped like ``slot.data``.

        A ``"cells"`` proxy is read through its own fitted polynomials (exact,
        no locator round trip); any other proxy through ``evaluate`` of the
        proxy mesh variable's symbol.
        """
        projector = getattr(slot, "_cell_projector", None)
        if projector is not None and getattr(slot, "_proxy_location", None) == "cells":
            slot._update_proxy_if_stale()
            vals = projector.interpolate(np.asarray(slot._meshVar.data), coords)
            return np.nan_to_num(vals)
        mv = slot._meshVar
        out = np.empty((coords.shape[0], slot.data.shape[1]))
        for i in range(slot.shape[0]):
            for j in range(slot.shape[1]):
                ij = slot._data_layout(i, j)
                out[:, ij] = np.asarray(
                    uw.function.evaluate(mv.sym[i, j], coords, evalf=evalf)
                ).reshape(-1)
        return out

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

            self.psi_star[i].data[...] = self.psi_star[i - 1].data[...]

        phi = 1 / self.step_averaging

        psi_star_0 = self.psi_star[0]
        coords = np.asarray(self.swarm._particle_coordinates.data)
        if self.particle_update == "flip":
            # The proxy the mesh saw during this solve, at the particles: the
            # residual psi_p - proxy(x_p) is what the mesh never resolved.
            proxy_at_p = self._proxy_values_at_particles(psi_star_0, coords, evalf)
        # Blend the freshly-evaluated psi into slot 0 component-by-component
        # through the canonical (N, components) storage (audit SWARM-06).
        for i in range(psi_star_0.shape[0]):
            for j in range(psi_star_0.shape[1]):
                ij = psi_star_0._data_layout(i, j)
                updated_psi = np.asarray(
                    uw.function.evaluate(
                        self.psi_fn[i, j],
                        coords,
                        evalf=evalf,
                    )
                ).reshape(-1)
                if self.particle_update == "flip":
                    residual = np.asarray(psi_star_0.data[:, ij]) - proxy_at_p[:, ij]
                    psi_star_0.data[:, ij] = updated_psi + self.residual_retention * residual
                else:
                    psi_star_0.data[:, ij] = (
                        phi * updated_psi + (1 - phi) * psi_star_0.data[:, ij]
                    )

        if self._n_solves_completed < self.order:
            self._n_solves_completed += 1

        return



class IntegrationPointSemiLagrangian(_DDtBase):
    r"""Semi-Lagrangian history stored at the mesh integration points.

    The history slots ``psi_star[k]`` are
    :class:`~underworld3.discretisation.IntegrationPointVariable` objects, so
    the value the weak form sees at each integration point is the discrete
    solution from ``k+1`` steps ago evaluated **exactly** at the departure
    point of that integration point. There is no nodal history field and no
    second interpolation: only the FE solution's own error remains in the
    advected term. Compare :class:`SemiLagrangian`, which samples at the
    nodes, stores a nodal ``psi_star`` and lets the assembler interpolate it
    to the integration points.

    Because a delta field cannot be sampled off its points, the chain
    ``psi_star[k] <- psi_star[k-1]`` of :class:`SemiLagrangian` is replaced
    by nodal **snapshots** of the solution and of the velocity at the last
    ``order`` times. Slot ``k`` is filled by tracing ``k+1`` segments back
    from every integration point (segment ``j`` with the velocity at time
    ``n-j`` and that step's ``dt``) and evaluating the snapshot from time
    ``n-k`` at the foot. Every slot carries one evaluation error rather than
    one per generation.

    What is not here (yet): vector/tensor histories, units-aware velocity
    reduction, ALE / old-frame trace-back, forcing history, checkpoint state.
    Use :class:`SemiLagrangian` for those.

    Parameters
    ----------
    mesh, psi_fn, V_fn, degree, continuous, varsymbol, verbose, bcs, order, theta
        As for :class:`SemiLagrangian`. ``psi_fn`` may be a scalar
        ``MeshVariable`` (its nodal data is then copied into the snapshot
        rather than re-evaluated) or a scalar expression.
    ``V_fn`` may be any expression (``-v``, ``v/2``, ``c(t) v``); the
    velocity history caches it by evaluation at each time level.
    """

    def __init__(
        self,
        mesh,
        psi_fn,
        V_fn,
        vtype=VarType.SCALAR,
        degree: int = 1,
        continuous: bool = True,
        varsymbol: Optional[str] = None,
        verbose: bool = False,
        bcs=None,
        order: int = 1,
        theta: float = 0.5,
        monotone_mode: Optional[str] = None,
        **_unsupported,
    ):
        super().__init__()
        if vtype != VarType.SCALAR:
            raise NotImplementedError(
                "IntegrationPointSemiLagrangian: scalar histories only for now"
            )
        self.monotone_mode = monotone_mode
        self.mesh = mesh
        self.bcs = list(bcs) if bcs is not None else []   # per instance, never a shared default
        self.verbose = verbose
        self.degree = degree
        self.continuous = continuous
        self.order = order
        self.theta = float(theta)
        self.V_fn = V_fn

        if hasattr(psi_fn, "sym") and not isinstance(psi_fn, sympy.Basic):
            self._psi_meshVar = psi_fn
            self._psi_fn = psi_fn.sym
        else:
            self._psi_meshVar = None
            self._psi_fn = psi_fn if isinstance(psi_fn, sympy.Matrix) else sympy.Matrix([[psi_fn]])

        self._init_history_tracking(order)
        self._check_rule_oversampling(degree)

        if varsymbol is None:
            varsymbol = rf"u_{{ [{self.instance_number}] }}"
        inst = self.instance_number

        psi_units = uw.get_units(self._psi_fn)
        if psi_units is not None and not uw.get_default_model().has_units():
            psi_units = None
        self._psi_units = psi_units

        # History slots at the integration points (injected, never sampled).
        self.psi_star = [
            uw.discretisation.IntegrationPointVariable(
                f"psi_star_ip_{inst}_{k}", mesh,
                varsymbol=rf"{{ {varsymbol}^{{ {'*' * (k + 1)} }} }}",
                units=psi_units,
            )
            for k in range(order)
        ]
        # Nodal snapshots of the solution and velocity at times n, n-1, ...
        # (sampled at the departure points).
        self.psi_snap = [
            uw.discretisation.MeshVariable(
                f"psi_snap_ip_{inst}_{k}", mesh, 1, degree=degree, continuous=continuous,
                varsymbol=rf"{{ {varsymbol}^{{ (n-{k}) }} }}",
                units=psi_units,
            )
            for k in range(order)
        ]
        # At least two velocity levels: the current interval's mid-time
        # velocity is extrapolated from v^n and v^{n-1}. Each level caches
        # V_fn evaluated at the nodes at that time, so V_fn may be any
        # expression (variables, ramping constants, swarm proxies).
        self._n_v = max(order, 2)
        self.v_levels = [self._make_velocity_level(f"n-{k}") for k in range(self._n_v)]
        self._init_coefficient_expressions(order, self.theta, with_exp=False)

    def spatial_weights(self):
        """As the base class, except that at ``theta = 1`` the old-level
        weights, identically zero, are returned as literals. A runtime
        constant with value zero would leave ``0 * grad(psi*)`` in the weak
        form, and the slot has no gradient to differentiate (the JIT guard
        would refuse a dead term). This is what makes the history usable in
        the composed ``AdvDiffusion`` at ``order=1, theta=1``."""
        w = super().spatial_weights()
        if self.integrator == "am" and float(self.theta) == 1.0:
            return [sympy.Integer(1)] + [sympy.Integer(0)] * (len(w) - 1)
        return w

    def _check_rule_oversampling(self, degree):
        """Refuse a rule with no more points per cell than the history space
        has local dofs.

        The solve fits the sampled departure-point values to the continuous
        space by weighted least squares on the rule (the mass matrix is exact
        on the rule, so Galerkin with a sampled load *is* that fit). The fit
        contracts in the sampled norm only, and the shifted field's sampled
        norm can exceed its true norm (aliasing of the rule on grid-scale
        modes), so the pure-advection map is never strictly contractive.
        Measured one-step growth factors, P2 on triangles, Courant 0.25
        (power iteration): 6 points 1.03-1.14 (blows up), 9 points 1.005,
        12 points 1.0003; with physical diffusion at cell Peclet 100: 6
        points 1.01 (still unstable), 9 and 12 points 0.996 (stable). Nodal
        SLCN: 0.999. So: raise at <= 1x oversampling, warn below 2x.
        """
        PETSc.Options().setValue(f"ipsl_check_{self.instance_number}_petscspace_degree", degree)
        fe = PETSc.FE().createDefault(
            self.mesh.dim, 1, self.mesh.isSimplex, self.mesh.qdegree,
            f"ipsl_check_{self.instance_number}_", PETSc.COMM_SELF,
        )
        local_dofs = fe.getDimension()
        Nq = len(np.asarray(self.mesh.integration_rule.getData()[1]))
        if Nq <= local_dofs:
            need = self._qdegree_with_at_least(local_dofs + 1)
            want = self._qdegree_with_at_least(2 * local_dofs)
            raise RuntimeError(
                f"IntegrationPointSemiLagrangian: the mesh rule has {Nq} points per cell "
                f"but a degree-{degree} history has {local_dofs} local dofs; the "
                "least-squares fit is not oversampled and is unstable at small Courant "
                f"number. For this cell type and history degree the rule needs at least "
                f"qdegree={need} (more points than dofs); 2x oversampling, the verified "
                f"setting, is qdegree={want}."
            )
        if Nq < 2 * local_dofs:
            warnings.warn(
                f"IntegrationPointSemiLagrangian: {Nq} rule points per cell for "
                f"{local_dofs} local dofs is under 2x oversampling: weakly unstable under pure "
                "advection (growth ~1.005/step at 1.5x, Courant 0.25) and stable with physical "
                "diffusion at cell Peclet <= 100. 2x (qdegree 3 for P2 on triangles) is neutral.",
                stacklevel=3,
            )

    def _qdegree_with_at_least(self, npoints, qmax=12):
        """The smallest quadrature degree whose default rule on this mesh's
        cell type has at least ``npoints`` points per cell (None if none up
        to ``qmax``). Point counts come from PETSc's own rules."""
        for q in range(self.mesh.qdegree + 1, qmax + 1):
            fe = PETSc.FE().createDefault(
                self.mesh.dim, 1, self.mesh.isSimplex, q, f"ipsl_qscan_{q}_", PETSc.COMM_SELF,
            )
            if len(np.asarray(fe.getQuadrature().getData()[1])) >= npoints:
                return q
        return None

    # ------------------------------------------------------------------
    @property
    def psi_fn(self):
        r"""Current symbolic expression :math:`\psi` being tracked."""
        return self._psi_fn

    @psi_fn.setter
    def psi_fn(self, new_fn):
        self._psi_meshVar = None
        self._psi_fn = new_fn if isinstance(new_fn, sympy.Matrix) else sympy.Matrix([[new_fn]])

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display
        super()._object_viewer()
        display(Latex(r"$\quad\psi = $ " + self.psi_fn._repr_latex_()))
        display(Latex(r"$\quad\mathbf{v} = $ " + sympy.Matrix(self.V_fn)._repr_latex_()))
        display(Latex(rf"$\quad$History steps = {self.order} (at the integration points)"))

    # ------------------------------------------------------------------
    def _nudged_node_coords(self, var):
        """ND node coordinates of ``var`` moved 0.1 % toward their cell
        centroids so boundary nodes locate unambiguously (see
        :meth:`SemiLagrangian._centroid_shifted_node_coords`)."""
        coords = np.asarray(var.coords_nd)
        cellid = self.mesh.get_closest_cells(coords).reshape(-1)
        cent = np.asarray(self.mesh._centroids)[cellid]
        return 0.999 * coords + 0.001 * cent

    def _record_current(self):
        """Snapshot slot 0 <- the current solution and velocity."""
        ps = self.psi_snap[0]
        if self._psi_meshVar is not None and (
            self._psi_meshVar.degree == ps.degree
            and self._psi_meshVar.continuous == ps.continuous
        ):
            ps.data[...] = self._psi_meshVar.data[...]
        else:
            vals = uw.function.evaluate(self.psi_fn[0], self._nudged_node_coords(ps))
            ps.data[:, 0] = np.asarray(_to_nondim_ndarray(vals, units=self._psi_units)).reshape(-1)
        self._copy_velocity_level(self.v_levels[0])

    def _velocity_at(self, v_sym, coords, evalf):
        v = uw.function.global_evaluate(v_sym, coords, evalf=evalf)
        v = np.asarray(_to_nondim_ndarray(v, units=self._velocity_units()))
        if v.ndim == 3:
            v = v[:, 0, :]
        return v.reshape(coords.shape[0], self.mesh.dim)

    def _trace_segment(self, X, v_start_sym, v_mid_sym, dt, evalf):
        r"""One RK2 (midpoint) segment of the characteristic, backwards:
        ``x_mid = x - dt/2 v_start(x)``, ``x_dep = x - dt v_mid(x_mid)``,
        with ``v_mid`` the velocity at the segment's mid TIME."""
        clamp = self.mesh.return_coords_to_bounds
        v0 = self._velocity_at(v_start_sym, X, evalf)
        Xm = X - 0.5 * dt * v0
        if clamp is not None:
            Xm = clamp(Xm)
        vm = self._velocity_at(v_mid_sym, Xm, evalf)
        Xd = X - dt * vm
        if clamp is not None:
            Xd = clamp(Xd)
        return Xd

    def _segment_dt(self, j, dt):
        """Length of segment ``j`` (0 = the current step)."""
        if j == 0:
            return dt
        h = self._dt_history[j - 1]
        return dt if h is None else h

    def _fill_slots(self, dt, evalf):
        """Trace back from the integration points and sample the snapshots."""
        X0 = np.asarray(self.psi_star[0].coords_nd)
        X = X0.copy()
        half = sympy.Rational(1, 2)
        for k in range(self.order):
            # Segment k extends the trace from slot k-1's feet, so the
            # feet for slot k are those of slot k-1 traced one more step.
            # Segment k runs from t^{n+1-k} back to t^{n-k}. Its mid-time
            # velocity: for k=0 extrapolated, 1.5 v^n - 0.5 v^{n-1} (v^{n+1}
            # is not known yet); for k>=1 both ends are known, so the
            # average of v^{n+1-k} and v^{n-k}. The first stage, which
            # only places the mid-point, uses the velocity at the
            # segment's start time.
            V = [lvl["expr"] for lvl in self.v_levels]
            if k == 0:
                v_start = V[0]
                v_mid = V[0] * sympy.Rational(3, 2) - V[1] * half
            else:
                v_start = V[k - 1]
                v_mid = (V[k - 1] + V[k]) * half
            X = self._trace_segment(X, v_start, v_mid, self._segment_dt(k, dt), evalf)
            vals = uw.function.global_evaluate(
                self.psi_snap[k].sym[0], X, evalf=evalf, monotone=self.monotone_mode
            )
            self.psi_star[k].data[:, 0] = np.asarray(
                _to_nondim_ndarray(vals, units=self._psi_units)).reshape(-1)

    def initialise_history(self):
        """Start every snapshot and slot from the current field, so
        ``bdf()`` is zero on the first step."""
        self._record_current()
        for k in range(1, self.order):
            self.psi_snap[k].data[...] = self.psi_snap[0].data[...]
        for k in range(1, self._n_v):
            self._copy_velocity_level(self.v_levels[k], self.v_levels[0])
        X = np.asarray(self.psi_star[0].coords_nd)
        vals = uw.function.evaluate(self.psi_snap[0].sym[0], X)
        vals = np.asarray(_to_nondim_ndarray(vals, units=self._psi_units)).reshape(-1)
        for k in range(self.order):
            self.psi_star[k].data[:, 0] = vals
        self._history_initialised = True

    def update_pre_solve(self, dt, evalf=False, verbose=False, **_ignored):
        self._dt = dt
        if not self._history_initialised:
            self.initialise_history()
        _update_bdf_values(self._bdf_coeffs, self.effective_order, self._dt, self._dt_history)
        _update_am_values(self._am_coeffs, self.effective_order, self.theta)
        for k in range(self.order - 1, 0, -1):
            self.psi_snap[k].data[...] = self.psi_snap[k - 1].data[...]
        for k in range(self._n_v - 1, 0, -1):
            self._copy_velocity_level(self.v_levels[k], self.v_levels[k - 1])
        self._record_current()
        self._fill_slots(dt, evalf)

    def update(self, dt, evalf=False, verbose=False, **kwargs):
        self.update_pre_solve(dt, evalf=evalf, verbose=verbose, **kwargs)

    def update_post_solve(self, dt, evalf=False, verbose=False, **_ignored):
        self._dt = dt
        for i in range(self.order - 1, 0, -1):
            self._dt_history[i] = self._dt_history[i - 1]
        self._dt_history[0] = dt
        if self._n_solves_completed < self.order:
            self._n_solves_completed += 1
