"""Field transfer on mesh adaptation — adapt-op-owned snapshot/move/transfer.

Phase 1 of the remesh-field-transfer redesign (see
``docs/developer/design/REMESH_FIELD_TRANSFER_DESIGN.md``). Moves the
deform-back / ``global_evaluate`` / deform-forward dance out of harness
code and into the adapt operation, where it can transfer every
mesh-registered variable — including ones the user does not know about
(solver history, projection auxiliaries, RBF proxies, ...).

Three pieces:

* :class:`RemeshPolicy` — per-variable transfer semantics. Default
  ``REMAP`` (evaluate the old field at the new node positions);
  ``CARRY`` for genuinely Lagrangian fields whose DOF value belongs to a
  material point; ``REINIT`` for stateless work-vars that get recomputed
  from a source on next use.
* :class:`RemeshContext` — carries old and new coordinates, total
  displacement, dt, and a bound interpolator. Passed to operator
  ``on_remesh`` hooks (Phase 2 ALE uses this).
* :func:`remesh_with_field_transfer` — the helper used by the adapt
  entry points (``smooth_mesh_interior``, ``OT_adapt``,
  ``follow_metric``). Takes a closure ``do_move`` that runs the mover
  (calling :meth:`Mesh._deform_mesh` repeatedly); the helper handles
  snapshot + transfer + operator-hook dispatch.

Phase 1 keeps the per-step parallel-safe ``psi_star`` re-record band-aid
in :mod:`underworld3.systems.ddt` — it is orthogonal (a *per-step*
re-record, not the *adapt-time* transfer). Phase 2 will add ALE
semantics (CARRY history + a one-step ``v_mesh`` correction) via the
operator hook; that is deliberately out of scope here.
"""

from __future__ import annotations

import enum
import os
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

import underworld3 as uw

if TYPE_CHECKING:  # pragma: no cover
    from underworld3.discretisation.discretisation_mesh import Mesh


__all__ = [
    "RemeshPolicy",
    "RemeshContext",
    "remesh_with_field_transfer",
    "remap_var_set",
]


# Monotone-limiting mode for the REMAP transfer, read ONCE at import time
# from the ``REMESH_MONOTONE`` environment variable. On a freshly-adapted
# mesh the NEW boundary DOFs sit a sagitta OUTSIDE the OLD boundary cell
# (arc vs chord), so the old P2/P3 field, FE-evaluated there, overshoots
# wildly — in parallel the migrate lands those points in a containing cell
# on another rank and the overshoot is delivered as a "valid" (un-flagged)
# value. That is the parallel free-slip v.n "leak": a corrupt boundary T/V
# remap, NOT a BC bug. The default ``"clamp"`` bounds each resampled value
# to its k-NN source-nodal range; it is bit-identical to plain FE in smooth
# regions and parallel-safe (rank-local) — the same limiter as the
# SemiLagrangian trace-back fix. Set REMESH_MONOTONE to "off" (or "0",
# "none", "false", "") to disable, or to any other mode accepted by
# ``uw.function.global_evaluate(..., monotone=...)`` (e.g. "pick").
REMESH_MONOTONE = os.environ.get("REMESH_MONOTONE", "clamp")
if REMESH_MONOTONE.lower() in ("", "0", "off", "none", "false"):
    REMESH_MONOTONE = False


class RemeshPolicy(str, enum.Enum):
    """Per-variable transfer semantics on a mesh adapt.

    ``REMAP`` is the safe universal default — "the previous values are
    those at the original mesh points," correct to first order or better
    for any Eulerian quantity. Anything not classified stays here so a
    forgotten variable fails *safe* (transferred needlessly) rather than
    silently wrong (stale).
    """

    REMAP = "remap"
    """Evaluate the OLD field at the NEW node positions. Default.

    Applied once for all REMAP vars together by the adapt op (one
    deform-back → ``global_evaluate`` → deform-forward pair) — robust by
    construction because the queries are interior points of the old
    mesh, not on-vertex points of the new mesh.
    """

    CARRY = "carry"
    """Leave DOF values unchanged on the new node positions.

    For genuinely Lagrangian fields (nodes move *with* the material) or
    fields that an operator hook will handle coherently with its
    history. Plain CARRY is only correct under material-following
    motion; arbitrary mesh motion needs the ALE refinement (Phase 2).
    """

    REINIT = "reinit"
    """Mark stale; the value is recomputed from a source before next use.

    Strictly for stateless work-vars (gradient/Hessian projections, RBF
    proxies, projected boundary normals). Persistent state — solver
    history in particular — must NOT use REINIT: ``DuDt.psi_star[i]``
    for ``i ≥ 1`` are accumulated upstream of *earlier-step* velocity
    fields that are gone once Stokes overwrites ``V`` with the new
    step's velocity.
    """


@dataclass
class RemeshContext:
    """State passed to operator ``on_remesh`` hooks.

    Carries the geometric data an operator (typically a ``DuDt``) needs
    to update its own history coherently — old/new coordinates, total
    displacement across the mover sweep, ``dt``, and a scratch slot
    where an ALE-style operator stashes ``v_mesh`` for the next solve
    to consume. The framework has already handled the generic
    per-variable REMAP pass before hooks are fired.

    ``managed_snapshot`` holds pre-move ``.data`` for every
    operator-managed variable (vars with ``_remesh_managed_by`` set).
    The default CARRY path doesn't need it — ``.data`` is left
    untouched. A hook that needs to fall back to REMAP (e.g. on an OT
    reset, see ``ale_opt_out``) calls
    :func:`remap_var_set` with this dict and its own var list.

    ``scratch`` is a free-form dict where adapt ops or hooks publish
    flags / per-step state. Two keys are in use:

    * ``"ale_opt_out"`` — published by ``OT_adapt``
      (:mod:`underworld3.meshing._ot_adapt`, from inside ``do_move`` via
      ``mesh._remesh_pending_scratch``) on a reset adapt; consumed by
      the ``DuDt`` ``on_remesh`` hook (:mod:`underworld3.systems.ddt`),
      which falls back to REMAP for its managed history vars.
    * ``"v_mesh"`` — the Phase 2 ALE convention: an ALE-style operator
      hook stashes the mesh velocity here for the next solve to
      consume. No producer exists yet.
    """

    mesh: "Mesh"
    old_X: np.ndarray
    new_X: np.ndarray
    dt: Optional[float] = None
    scratch: dict = field(default_factory=dict)
    managed_snapshot: dict = field(default_factory=dict)

    @property
    def total_disp(self) -> np.ndarray:
        """Total node displacement across the whole mover sweep (new − old)."""
        return self.new_X - self.old_X


def _gather_transfer_vars(mesh: "Mesh") -> dict:
    """Snapshot registered mesh variables, partitioned by policy.

    Returns
    ``{"remap": [vars], "carry": [vars], "reinit": [vars], "managed": [vars]}``.
    The first three are vars the generic per-variable pass handles
    directly; "managed" collects vars whose ``_remesh_managed_by`` is
    set (an operator's ``on_remesh`` hook owns their transfer). Managed
    vars are excluded from the first three buckets even if their policy
    is REMAP/CARRY/REINIT — the operator is the single source of truth.
    """
    buckets = {"remap": [], "carry": [], "reinit": [], "managed": []}
    for var in list(mesh.vars.values()):
        if var is None:
            continue
        # Operator-managed vars are deferred to the registered hook so the
        # generic pass does not also transfer them. We still snapshot
        # their data (see _remesh_with_field_transfer_impl) so a hook
        # that needs to fall back to REMAP (e.g. on an OT reset adapt)
        # has the original values to work from.
        if getattr(var, "_remesh_managed_by", None) is not None:
            buckets["managed"].append(var)
            continue
        policy = getattr(var, "remesh_policy", RemeshPolicy.REMAP)
        # tolerate string values stored elsewhere
        if isinstance(policy, str):
            try:
                policy = RemeshPolicy(policy)
            except ValueError:
                policy = RemeshPolicy.REMAP
        if policy is RemeshPolicy.REMAP:
            buckets["remap"].append(var)
        elif policy is RemeshPolicy.CARRY:
            buckets["carry"].append(var)
        elif policy is RemeshPolicy.REINIT:
            buckets["reinit"].append(var)
        else:  # pragma: no cover — defensive
            buckets["remap"].append(var)
    return buckets


def _snapshot_var_data(vars_):
    """Copy the current ``.data`` array of each given var (defensive copy).

    Used for both the REMAP bucket and the operator-managed bucket in
    :func:`remesh_with_field_transfer`. Vars that cannot be snapshotted
    are omitted from the returned dict.
    """
    out = {}
    for var in vars_:
        try:
            arr = np.asarray(var.data)
            if arr.size == 0:
                continue
            out[var] = arr.copy()
        except Exception:
            # Sanctioned swallow: a var whose storage is unallocated or
            # size-0 on this rank (lazy allocation / empty local partition)
            # has nothing to snapshot. It is absent from the returned dict,
            # so the transfer pass never touches it (the write-back twin,
            # _write_var_data, skips the same vars).
            pass
    return out


def _write_var_data(var, values):
    """Overwrite ``var``'s DOF values in place, tolerating unwritable storage.

    Parameters
    ----------
    var : MeshVariable
        Target variable; its full ``.data`` buffer is overwritten.
    values : array_like or float
        Replacement values, broadcastable to ``var.data``'s shape
        (a scalar such as ``0.0`` zeros the whole buffer).
    """
    try:
        np.asarray(var.data)[...] = values
    except Exception:
        # Sanctioned swallow: a var whose storage is unallocated or size-0
        # on this rank (lazy allocation / empty local partition) has nowhere
        # to write. Skipping leaves the variable exactly as the snapshot
        # pass found it — the same vars are skipped by _snapshot_var_data,
        # so no transfer is silently half-applied.
        pass


def _remap_one_var(var, old_X, new_X, mesh):
    """Helper kept separate for diagnostics; not currently used.

    The actual transfer is done in ``remesh_with_field_transfer`` by
    moving the mesh once between the two coordinate states and calling
    ``global_evaluate`` per var, which is cheaper than per-var deform.
    """
    raise NotImplementedError("use remesh_with_field_transfer")


def _new_coord_cache(mesh, remap_vars):
    """Capture each REMAP var's DOF coordinates ON THE NEW MESH.

    Called *after* the mover has produced ``new_X`` — the mesh is at
    ``new_X``, so ``var.coords`` returns the correct DOF positions for
    that variable's (degree, continuous) basis. Returned dict is the
    target-point set for ``global_evaluate`` once the mesh is deformed
    back to the old state.
    """
    out = {}
    for var in remap_vars:
        try:
            out[var] = np.asarray(var.coords).copy()
        except Exception:
            out[var] = None
    return out


def remesh_with_field_transfer(
    mesh: "Mesh",
    do_move: Callable[[], None],
    *,
    dt: Optional[float] = None,
    extra_zero: Optional[list] = None,
    verbose: bool = False,
) -> bool:
    """Adapt-op contract: run a mover and transfer every registered var.

    ``do_move`` is a closure that performs the actual coordinate move —
    typically the body of one of the ``underworld3.meshing.smoothing``
    package movers (spring / MA / OT / anisotropic / MMPDE) or a
    ``follow_metric`` mover. It is expected to call
    :meth:`Mesh._deform_mesh` one or more times and leave the mesh
    sitting at the final adapted positions. ``do_move`` MUST NOT touch
    field ``.data`` — the helper owns transfer.

    The helper:

    1. Snapshots the current coordinates (``old_X``) and the current
       ``.data`` of every REMAP variable.
    2. Calls ``do_move()`` to update mesh coords (REMAP-vars'
       ``.data`` is unchanged because ``_deform_mesh`` does not touch
       it; only coordinate-keyed caches are invalidated).
    3. Captures the new DOF coordinates for every REMAP var
       (``var.coords`` on the new mesh).
    4. Performs ONE deform-back → ``global_evaluate`` → deform-forward
       pair for the entire REMAP set, writing the resampled values back
       into each var's ``.data``.
    5. Fires every registered ``on_remesh(ctx)`` hook (Phase 2 ALE
       operators use this; Phase 1 hooks are typically no-ops).
    6. Zeros ``extra_zero`` vars (caller's REINIT-with-zero list, e.g.
       ``[V, P]`` for a cold-restart of the flow solve).
    7. Marks REINIT vars stale (Phase 1: not strictly required because
       the only REINIT-stamped vars today are recomputed on first
       access; placeholder for future eager invalidation).

    Returns ``True`` if the mesh actually moved (``do_move`` reported
    geometry change); ``False`` if it short-circuited (caller can skip
    transfer too). Detection: compares ``mesh.X.coords`` before and
    after ``do_move``.

    Parallel: ``global_evaluate`` resolves off-rank target points
    correctly, so the transfer is partition-agnostic — the rank that
    owns a new-mesh DOF coordinate retrieves its value from whichever
    rank held that point on the old mesh. Local ``evaluate`` would
    leave stale values at the partition seams (the documented failure
    mode that motivated this design).
    """
    # Re-entrancy guard: composite adapt ops (OT_adapt) wrap the whole
    # reset+build+smooth pipeline once at the outer level; inner movers
    # (smooth_mesh_interior called from inside that pipeline) consult
    # this flag and skip their own wrap.
    if getattr(mesh, "_in_remesh_transfer", False):
        # Nested call: run the mover only. The OUTER wrapper owns
        # snapshot / transfer / hook dispatch, and it already set
        # mesh._remesh_pending_scratch, so an inner adapt op can still
        # publish flags there (e.g. OT_adapt marking a reset adapt).
        # The True return here is unconditional and meaningless — nested
        # callers cannot use it to tell whether the mesh actually moved;
        # only the outer call's return value carries that information.
        do_move()
        return True
    mesh._in_remesh_transfer = True
    # Per-adapt scratch dict surfaced to the closure so an adapt op
    # can publish flags (e.g. ``ale_opt_out``) before the operator
    # hooks fire. Drained into ctx.scratch after do_move returns.
    mesh._remesh_pending_scratch = {}
    try:
        return _remesh_with_field_transfer_impl(
            mesh, do_move, dt=dt,
            extra_zero=extra_zero, verbose=verbose)
    finally:
        mesh._in_remesh_transfer = False
        mesh._remesh_pending_scratch = None


def _remesh_with_field_transfer_impl(
    mesh, do_move, *, dt=None, extra_zero=None, verbose=False,
) -> bool:
    """Body of :func:`remesh_with_field_transfer`. Split so the
    re-entrancy guard wraps a single try/finally."""
    buckets = _gather_transfer_vars(mesh)
    remap_vars = buckets["remap"]
    reinit_vars = buckets["reinit"]
    managed_vars = buckets["managed"]

    old_X = np.asarray(mesh.X.coords).copy()
    old_data = _snapshot_var_data(remap_vars)
    # Snapshot managed vars too — needed when a hook opts out of ALE
    # for this adapt and falls back to REMAP (see RemeshContext docs).
    managed_snapshot = _snapshot_var_data(managed_vars)

    # Run the mover. It is allowed to call _deform_mesh many times; .data
    # is untouched by _deform_mesh, so REMAP snapshots stay valid.
    do_move()

    new_X = np.asarray(mesh.X.coords).copy()
    if np.array_equal(new_X, old_X):
        # Mover short-circuited (skip threshold, no metric change, ...).
        # No transfer to do, no hooks to fire — caller treats this as a
        # no-op adapt.
        return False

    # The one-shot REMAP dance for the generic per-variable pass.
    remap_var_set(mesh, remap_vars, old_X, new_X, old_data, verbose=verbose)

    # Operator hooks (Phase 2 ALE etc.). Currently fired even when
    # remap_vars is empty so a CARRY-only DDt still gets its v_mesh.
    # Drain the per-adapt scratch dict (set up by remesh_with_field_transfer)
    # — adapt ops publish here from inside do_move (e.g. OT sets
    # ``ale_opt_out`` so DDts fall back to REMAP for a reset event).
    pending_scratch = getattr(mesh, "_remesh_pending_scratch", None) or {}
    ctx = RemeshContext(mesh=mesh, old_X=old_X, new_X=new_X, dt=dt,
                        scratch=dict(pending_scratch),
                        managed_snapshot=managed_snapshot)
    for hook in list(_iter_active_hooks(mesh)):
        try:
            hook(ctx)
        except Exception as exc:
            # Best-effort hook contract: the generic REMAP pass above has
            # already secured every registered variable, so a failing
            # on_remesh hook can only lose an operator-private refinement
            # (e.g. an ALE history update degrades to the plain CARRY
            # values). One broken operator must not abort the transfer of
            # the remaining hooks' state mid-adapt, so dispatch continues;
            # the failure is reported only under verbose=True.
            # TODO(DESIGN): consider warning unconditionally — a silently
            # degraded ALE history is hard to diagnose downstream.
            if verbose:
                uw.pprint(
                    f"  remesh_with_field_transfer: hook raised: {exc}")

    # REINIT pass — Phase 1 is a marker only; the stamped vars (proxies,
    # projection aux, projected normals) recompute lazily on next use.
    # Kept as an explicit no-op so future eager invalidation has one
    # call site to change.
    for var in reinit_vars:
        _mark_reinit_stale(var)

    # Caller-supplied zero list (e.g. V, P after a topology-preserving
    # mover, when the flow solve wants a cold start). This is NOT the
    # REINIT policy — it is a user knob preserved from the old OT_adapt
    # API. REINIT is for *framework-stamped* stateless vars.
    if extra_zero:
        for var in extra_zero:
            _write_var_data(var, 0.0)

    return True


def _iter_active_hooks(mesh):
    """Yield live ``on_remesh(ctx)`` callbacks registered on the mesh.

    Hooks are stored as weakrefs to the registering operator; this
    iterator drops dead refs and yields a bound callable for each live
    one. Defined here (rather than as a Mesh method) so the helper
    module can be imported without importing the entire Mesh stack.
    """
    refs = getattr(mesh, "_remesh_hooks", None)
    if not refs:
        return
    live = []
    for ref in list(refs):
        op = ref() if isinstance(ref, weakref.ReferenceType) else ref
        if op is None:
            continue
        cb = getattr(op, "on_remesh", None)
        if cb is None:
            continue
        live.append((ref, cb))
    # Prune dead refs once per dispatch.
    if isinstance(refs, list):
        refs[:] = [ref for ref, _ in live]
    for _, cb in live:
        yield cb


def remap_var_set(mesh, vars_, old_X, new_X, old_data, *, verbose=False):
    """One-shot REMAP dance for a set of variables.

    Used by the generic per-variable pass in
    :func:`remesh_with_field_transfer`, and public so an operator's
    ``on_remesh`` hook can force-REMAP its CARRY-managed vars on an
    adapt that is ALE-incompatible (e.g. an OT_adapt reset, where the
    linear ``Δx/dt → v_mesh`` interpretation breaks down).

    Contract: on entry the mesh is at ``new_X`` and each var's
    ``.data`` may hold *either* the original snapshot value (CARRY:
    operator-managed vars that the generic pass skipped) or the
    not-yet-restored value (REMAP: the snapshot has not been written
    back). ``old_data`` is the snapshot keyed by var, captured *before*
    ``do_move`` ran. On exit the mesh is back at ``new_X`` and every
    var's ``.data`` holds the OLD field evaluated at the var's NEW DOF
    coordinates.

    The set may be empty (no-op). Operations are MPI-collective; every
    rank must call with the same set.
    """
    if not vars_:
        return
    # Capture target DOF coords on the NEW mesh first (we are sitting
    # on new_X right now). var.coords reads from the live mesh state.
    new_dof_coords = {}
    for var in vars_:
        try:
            new_dof_coords[var] = np.asarray(var.coords).copy()
        except Exception:
            new_dof_coords[var] = None

    # Deform back to the old coords, restore data so global_evaluate
    # sees the old field, evaluate at each var's new DOF coords,
    # then deform forward and write the resampled values back.
    mesh._deform_mesh(old_X)
    for var, data in old_data.items():
        _write_var_data(var, data)

    resampled = {}
    for var in vars_:
        target = new_dof_coords.get(var, None)
        if target is None or target.size == 0:
            continue
        try:
            # global_evaluate resolves off-rank targets via swarm migration.
            # REMESH_MONOTONE (module constant, see its comment) bounds the
            # boundary-overshoot failure mode of the plain FE evaluation.
            try:
                val = uw.function.global_evaluate(
                    var.sym, target, monotone=REMESH_MONOTONE)
            except (ValueError, NotImplementedError):
                # monotone needs a single-MeshVariable expr; composite /
                # unsupported vars fall back to plain FE (still transferred).
                val = uw.function.global_evaluate(var.sym, target)
        except Exception as exc:
            if verbose:
                uw.pprint(
                    f"  remap_var_set: skipping "
                    f"{getattr(var, 'name', var)!r} ({exc})")
            continue
        resampled[var] = np.asarray(val).reshape(
            np.asarray(var.data).shape)

    mesh._deform_mesh(new_X)
    for var, val in resampled.items():
        _write_var_data(var, val)



def _mark_reinit_stale(var):
    """Mark a REINIT variable stale.

    Phase 1: a hook for the future. The current REINIT-stamped vars
    (``_n_proj``, projection auxiliaries) recompute themselves on the
    next access path that owns them — there is nothing eager to do here
    that the lazy recomputation does not already cover. Kept as a
    single call site so future eager invalidation (e.g. zero ``.data``,
    or set a ``_needs_recompute`` flag the consumer checks) has one
    place to land.
    """
    flag = getattr(var, "_remesh_reinit_callback", None)
    if callable(flag):
        try:
            flag()
        except Exception:
            pass
