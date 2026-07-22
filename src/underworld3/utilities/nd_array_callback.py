"""
NDArray_With_Callback: A numpy ndarray subclass with modification callbacks.

This class is designed to help wrap underworld data that require us to
do parallel sync or PETSc object refreshing.

Key Features:
- Callbacks triggered when array data is modified
- Delayed callback execution for batch operations
- MPI synchronization in parallel contexts
- Global reduction operations (MPI-aware): global_max, global_min, global_sum,
  global_mean, global_size, global_norm, global_rms
- Weak reference ownership tracking

This is the base class for UnitAwareArray which adds unit preservation.
"""

import itertools
import numpy as np
import weakref
import logging
from typing import Callable, Any, Dict, List, Optional, Union
import threading

logger = logging.getLogger(__name__)

# Try to import underworld MPI - fall back gracefully if not available
try:
    import underworld3 as uw

    _has_uw_mpi = hasattr(uw, "mpi") and hasattr(uw.mpi, "barrier")
except ImportError:
    _has_uw_mpi = False
    uw = None


class DelayedCallbackManager:
    """
    Thread-local manager for deferred synchronisation across multiple
    NDArray_With_Callback instances.

    Writes made inside a delay context land in the arrays immediately; what
    is deferred is the *synchronisation* work the callbacks perform. Each
    delay level records which CANONICAL arrays were touched (dirty marking)
    rather than queueing per-write events — the flush at context exit then
    synchronises each touched variable exactly once, in the same order on
    every rank. Per-event queueing survives only for legacy untagged
    callbacks (plain ``add_callback``), whose replay is rank-local and must
    not contain collective operations.
    """

    def __init__(self):
        self._local = threading.local()

    def _get_state(self):
        """Get or create thread-local state."""
        if not hasattr(self._local, "delay_stack"):
            self._local.delay_stack = []
        return self._local

    def is_delaying(self):
        """Check if callbacks are currently being delayed."""
        state = self._get_state()
        return len(state.delay_stack) > 0

    def push_delay_context(self, context_info=None):
        """Enter a new delay context (one dirty-set per nesting level)."""
        state = self._get_state()
        state.delay_stack.append(
            {
                "context_info": context_info,
                "legacy_queue": [],
                "dirty_owners": {},
                "dirty_local": {},
                "dirty_collective": set(),
            }
        )

    def pop_delay_context(self):
        """Exit the current delay level and return its recorded state."""
        state = self._get_state()
        if not state.delay_stack:
            return None
        return state.delay_stack.pop()

    def add_delayed_callback(self, array, callback_func, change_info):
        """Queue a legacy per-event callback for rank-local replay at exit."""
        state = self._get_state()
        state.delay_stack[-1]["legacy_queue"].append(
            {
                "array": array,
                "callback": callback_func,
                "change_info": change_info.copy(),
            }
        )

    def mark_dirty(self, canonical):
        """Record that a canonical array was written in the current level.

        Owners carrying a ``_collective_flush_id`` (mesh variables — their
        PETSc pack is collective) go into the id set that is agreed across
        ranks at flush time. Everything else (swarm variables — their packs
        are rank-local; migration is separately rank-agreed) flushes
        rank-locally.
        """
        level = self._get_state().delay_stack[-1]
        owner = canonical.owner
        flush_id = getattr(owner, "_collective_flush_id", None)
        if flush_id is not None:
            level["dirty_collective"].add(flush_id)
        elif owner is not None and hasattr(owner, "_deferred_canonical_flush"):
            # Weak ref, re-resolved at flush: the owner's canonical may be
            # invalidated and rebuilt mid-context (swarm migration), and a
            # pinned array would flush stale pre-migration values.
            level["dirty_owners"].setdefault(id(owner), weakref.ref(owner))
        else:
            level["dirty_local"].setdefault(id(canonical), canonical)


# Global instance for managing delayed callbacks
_delayed_callback_manager = DelayedCallbackManager()


# --- Collective-flush registry -------------------------------------------
#
# Cross-rank agreement on WHICH variables to flush at a synchronised-update
# exit needs a key that is identical on every rank. Creation-order integer
# ids qualify because registered objects are created SPMD-collectively
# (mesh-variable construction performs collective DM operations, so the
# counter advances in lockstep). Variable NAMES do not qualify: temporary
# variables embed rank-local id() values in their names.

_collective_flush_ids = itertools.count()
_collective_flush_registry: Dict[int, "weakref.ref"] = {}


def register_collective_flush(obj):
    """Assign a creation-order id for the synchronised-update flush.

    ``obj`` must provide ``_deferred_canonical_flush()``, which every rank
    calls for every id in the agreed flush set.
    """
    flush_id = next(_collective_flush_ids)
    _collective_flush_registry[flush_id] = weakref.ref(obj)
    return flush_id


def _base_chain_resolves(array, canonical):
    """True if ``array`` IS ``canonical`` or a view whose base chain reaches it.

    Identity in the base chain, never ``np.may_share_memory``: that is False
    for any zero-size array, which would classify an empty rank's view as a
    copy and desynchronise the collective branch (#376).
    """
    if array is canonical:
        return True
    base = array.base
    while base is not None and base is not canonical:
        base = getattr(base, "base", None)
    return base is not None


def _deferred_flush_info(canonical):
    return {
        "operation": "deferred_flush",
        "indices": None,
        "old_value": None,
        "new_value": None,
        "array_shape": canonical.shape,
        "array_dtype": canonical.dtype,
        "data_has_changed": True,
    }


def fire_canonical_callbacks(canonical):
    """Fire each canonical-guarded callback once, with the canonical array.

    Reads the canonical array's LIVE callback list (derived views hold stale
    copies), so callbacks registered after a view was created still fire.
    """
    info = _deferred_flush_info(canonical)
    for callback in list(canonical._callbacks):
        if getattr(callback, "_is_canonical", False):
            callback(canonical, info)


def _flush_delay_level(level, aborted=False):
    """Flush one delay level: rank agreement first, then legacy replay,
    rank-local canonical flushes, and the collectively-agreed canonical
    flushes in creation order.

    The agreement allgather runs FIRST and unconditionally — empty sets
    and aborted ranks included — so every rank stays matched even when
    writes were rank-uneven or the context body raised on some ranks
    only. Any rank aborting makes every rank skip all flushing.
    """
    local_ids = [] if aborted else sorted(level["dirty_collective"])
    if _has_uw_mpi and uw.mpi.size > 1:
        gathered = uw.mpi.comm.allgather((bool(aborted), local_ids))
        if any(flag for flag, _ in gathered):
            return
        union = sorted(set().union(*(ids for _, ids in gathered)))
    else:
        if aborted:
            return
        union = local_ids

    for item in level["legacy_queue"]:
        item["callback"](item["array"], item["change_info"])

    # Rank-local canonical flushes re-resolve the LIVE canonical through
    # the owner where one exists: migration inside the context invalidates
    # and rebuilds swarm canonicals, and flushing a pinned pre-migration
    # array would resurrect stale values.
    for owner_ref in level["dirty_owners"].values():
        owner = owner_ref()
        if owner is not None:
            owner._deferred_canonical_flush()

    for canonical in level["dirty_local"].values():
        fire_canonical_callbacks(canonical)

    targets = {}
    for flush_id in union:
        ref = _collective_flush_registry.get(flush_id)
        targets[flush_id] = ref() if ref is not None else None
    missing = [flush_id for flush_id, obj in targets.items() if obj is None]
    if _has_uw_mpi and uw.mpi.size > 1:
        missing_anywhere = max(uw.mpi.comm.allgather(len(missing)))
    else:
        missing_anywhere = len(missing)
    if missing_anywhere:
        # Raise on EVERY rank: a one-rank raise inside the flush loop below
        # would leave the other ranks blocked in a collective.
        raise RuntimeError(
            "synchronised_array_update flush found dirty variables that no "
            f"longer exist (registration ids {missing or union}). A variable "
            "written inside the context was destroyed before context exit."
        )

    for flush_id in union:
        targets[flush_id]._deferred_canonical_flush()


class _DelayCallbacksContext:
    """Context manager behind ``delay_callback`` / ``synchronised_array_update``.

    Entering and exiting are collective when MPI is active: the entry
    barrier catches non-lockstep entry early, and the exit flush contains
    an allgather plus per-variable collective synchronisation.
    """

    def __init__(self, context_info):
        self.context_info = context_info

    def __enter__(self):
        if _has_uw_mpi and uw.mpi.size > 1:
            uw.mpi.barrier()
        _delayed_callback_manager.push_delay_context(self.context_info)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        level = _delayed_callback_manager.pop_delay_context()
        if level is not None:
            # BOTH paths run the flush's rank-agreement collective: if the
            # raising rank skipped it, the survivors' allgather would pair
            # with an unrelated collective elsewhere (round-1 review: a
            # CAUGHT rank-local exception deadlocked). Any rank aborting
            # makes every rank skip the flushes; values already landed in
            # the canonical arrays — only the deferred synchronisation is
            # dropped, symmetrically.
            _flush_delay_level(level, aborted=exc_type is not None)
        return False


class NDArray_With_Callback(np.ndarray):
    """A numpy ndarray subclass that triggers callbacks when array data is modified.

    This class maintains full numpy array compatibility while providing reactive
    programming capabilities for scientific computing applications.

    **Callback Function Signature**::

        def callback(array: NDArray_With_Callback, change_info: dict) -> None:
            pass

    The ``change_info`` dictionary contains:

    - ``operation`` (str): Operation name ('setitem', 'iadd', 'fill', etc.)
    - ``indices`` (tuple/slice/None): Location of change (for setitem operations)
    - ``old_value`` (None): Always None. Internal operations no longer snapshot
      prior values (no registered callback ever read them, and the copy was a
      full-array allocation per write); the key is retained for dict-shape
      compatibility.
    - ``new_value`` (array-like): New values being assigned
    - ``array_shape`` (tuple): Current shape of the array
    - ``array_dtype`` (np.dtype): Data type of the array

    **Features**:

    - **Multiple callbacks**: ``add_callback()``, ``remove_callback()``, ``clear_callbacks()``
    - **Enable/disable**: ``enable_callbacks()``, ``disable_callbacks()``
    - **Delayed execution**: ``delay_callback()``, ``delay_callbacks_global()``
    - **MPI synchronization**: Automatic barriers in parallel contexts
    - **Weak references**: Owner tracking without circular dependencies
    - **Global reductions**: MPI-aware ``global_max()``, ``global_min()``, ``global_sum()``, etc.

    **Global Reduction Operations (MPI-aware)**:

    - ``global_max(axis=None)``: Maximum value across all MPI ranks
    - ``global_min(axis=None)``: Minimum value across all MPI ranks
    - ``global_sum(axis=None)``: Sum of all values across all MPI ranks
    - ``global_mean(axis=None)``: True mean (global sum / global count)
    - ``global_size()``: Total number of elements across all ranks
    - ``global_norm(ord=2)``: 2-norm (Euclidean) across all ranks
    - ``global_rms()``: Root mean square across all ranks

    These methods use MPI collective operations (``allreduce``).
    All ranks must call these methods (they are collective operations).
    Subclasses like ``UnitAwareArray`` override these to preserve units.
    """

    def __new__(cls, input_array=None, owner=None, callback=None, disable_inplace_operators=False):
        """
        Create new NDArray_With_Callback instance.

        Parameters
        ----------
        input_array : array-like, optional
            Input data to create array from (defaults to empty array if None)
        owner : object, optional
            The object that owns this array (stored as weak reference)
        callback : callable, optional
            Initial callback function to register
        disable_inplace_operators : bool, optional
            If True, in-place operators (``+=``, ``-=``, ``*=``, ``/=``, etc.)
            will raise RuntimeError for parallel safety.
            Default is False for backward compatibility.
        """
        if input_array is None:
            input_array = []

        # Create the ndarray instance
        obj = np.asarray(input_array).view(cls)

        # Initialize callback system
        obj._callbacks = []
        obj._owner = weakref.ref(owner) if owner is not None else None
        obj._callback_enabled = True
        obj._disable_inplace_operators = disable_inplace_operators

        # Register initial callback if provided
        if callback is not None:
            obj._callbacks.append(callback)

        return obj

    def __array_finalize__(self, obj):
        """
        Called whenever the system allocates a new array from this template.
        """
        if obj is None:
            return

        # Copy callback information from parent array
        self._callbacks = getattr(obj, "_callbacks", []).copy()
        self._owner = getattr(obj, "_owner", None)
        self._callback_enabled = getattr(obj, "_callback_enabled", True)
        self._disable_inplace_operators = getattr(obj, "_disable_inplace_operators", False)

    # === numpy.ma (masked array) compatibility ===
    # These attributes are needed when numpy's masked array operations
    # interact with our array subclass.

    @property
    def _mask(self):
        """For numpy.ma compatibility - we have no mask."""
        return np.ma.nomask

    @_mask.setter
    def _mask(self, value):
        """For numpy.ma compatibility - ignore mask setting."""
        # We don't support masking, so ignore attempts to set a mask
        pass

    @property
    def mask(self):
        """Public mask property for numpy.ma compatibility.

        Matplotlib's quiver and other plotting functions access .mask directly.
        This aliases to _mask which returns np.ma.nomask (no masking).
        """
        return self._mask

    @mask.setter
    def mask(self, value):
        """Public mask setter for numpy.ma compatibility."""
        self._mask = value

    def filled(self, fill_value=None):
        """Return array with masked values filled.

        For numpy.ma compatibility. Since we have no mask, this just
        returns a copy of the data (as numpy array to avoid further
        masked array operations).

        Parameters
        ----------
        fill_value : scalar, optional
            Value used to fill masked entries. Ignored since we have no mask.

        Returns
        -------
        ndarray
            A copy of the data as a plain numpy array.
        """
        return np.asarray(self).copy()

    def _update_from(self, obj):
        """For numpy.ma compatibility - update from another array."""
        # This is used by masked array operations to update data
        if hasattr(obj, '__array__'):
            np.copyto(self, np.asarray(obj))
        elif obj is not None:
            np.copyto(self, obj)

    def __array_wrap__(self, result, context=None, return_scalar=False):
        """
        Called after numpy operations to wrap results back to our type.

        Parameters updated for NumPy 2.0 compatibility:
        - context: Information about the ufunc that produced the result (unused)
        - return_scalar: If True, return a scalar instead of 0-d array
        """
        if return_scalar or result.shape == ():
            # Scalar result, return as numpy scalar
            return result.item()

        # For in-place operations that return the same array, return self
        # Use numpy's view to avoid recursion
        try:
            self_as_ndarray = np.ndarray.view(self, np.ndarray)
            # NB: require a NON-None shared base for the base-equality branch.
            # A fresh ufunc result (e.g. scalar * arr) has base=None; if self
            # also owns its data (base=None) then `result.base is self.base`
            # is `None is None` == True, which under numpy 2.0's ufunc dispatch
            # wrongly returned `self` and dropped the operation (scalar * arr ==
            # arr). Only a genuine in-place/view result shares a non-None base.
            if result is self_as_ndarray or (
                self.base is not None
                and getattr(result, "base", None) is self.base
            ):
                return self
        except Exception:
            # If view comparison fails, fall back to simple check
            pass

        # For new array results, don't automatically wrap to our type
        # This prevents issues with operations that shouldn't preserve callbacks
        return np.asarray(result)

    def set_callback(self, callback: Callable):
        """
        Set a single callback function (replaces any existing callbacks).

        Parameters
        ----------
        callback : callable
            Function with signature: callback(array, change_info)
            - array: the NDArray_With_Callback instance
            - change_info: dict with operation details
        """
        self._callbacks = [callback] if callback is not None else []

    def add_callback(self, callback: Callable):
        """
        Add an additional callback function.

        Parameters
        ----------
        callback : callable
            Function to add to callback list
        """
        if callback is not None and callback not in self._callbacks:
            self._callbacks.append(callback)

    def add_canonical_callback(self, callback: Callable):
        """
        Register a callback that only ever fires for the canonical storage.

        ``self`` must be the canonical array at registration time. Derived
        arrays inherit the callback list via ``__array_finalize__``, so an
        unguarded callback also fires on views and temporary fancy-index
        copies. A copy's contents are partition-dependent, so a PETSc sync
        run from inside the callback executes its collectives on some ranks
        only — the #376 parallel hang. This wrapper applies the guard once,
        centrally:

        - write through a **view** of the canonical array: the data already
          landed in canonical storage, so the callback fires with the FULL
          canonical array;
        - write to a **copy**: skipped — numpy's fancy-index write-back
          re-fires the callback through the parent's ``__setitem__``, so
          nothing is lost;
        - view-vs-copy is decided by IDENTITY in numpy's base chain, never
          ``np.may_share_memory``, which is False for any zero-size array
          and would re-create the rank asymmetry on ranks whose local
          slice is empty.

        Known corner (from the #378 analysis): ``reshape``/``ravel`` of a
        NON-contiguous derived view produces a copy on non-empty ranks but
        a view on a zero-size rank, so that one pattern remains
        rank-asymmetric at the per-write level — locally
        indistinguishable. The ``uw.synchronised_array_update`` dirty-flag
        flush (#383) is the real fix: agreement happens per variable at
        context exit, not per write.

        Parameters
        ----------
        callback : callable
            Function with signature ``callback(array, change_info)``;
            ``array`` is always the canonical storage.
        """
        # weakref: the callback list lives ON the array, so a strong capture
        # of self inside the closure would be an uncollectable cycle
        canonical_ref = weakref.ref(self)

        def _canonical_dispatch(array, change_info):
            canonical = canonical_ref()
            if canonical is None:
                return
            if not _base_chain_resolves(array, canonical):
                return
            callback(canonical, change_info)

        _canonical_dispatch._is_canonical = True
        _canonical_dispatch._canonical_ref = canonical_ref
        _canonical_dispatch._wrapped = callback
        self.add_callback(_canonical_dispatch)

    def remove_callback(self, callback: Callable):
        """
        Remove a specific callback function.

        Accepts either the registered callable itself or the original
        function handed to :meth:`add_canonical_callback` (the list stores
        the guarding dispatch wrapper, not the original).

        Parameters
        ----------
        callback : callable
            Function to remove from callback list
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            return
        for registered in list(self._callbacks):
            if getattr(registered, "_wrapped", None) is callback:
                self._callbacks.remove(registered)

    def clear_callbacks(self):
        """Remove all registered callbacks."""
        self._callbacks.clear()

    def enable_callbacks(self):
        """Enable callback triggering."""
        self._callback_enabled = True

    def disable_callbacks(self):
        """Disable callback triggering (useful for batch operations)."""
        self._callback_enabled = False

    @property
    def owner(self):
        """Get the owner object (may be None if owner was garbage collected)."""
        return self._owner() if self._owner is not None else None

    def delay_callback(self, context_info=None):
        """
        Context manager to defer callback synchronisation until context exit.

        The delay context is global (thread-local), so it covers this array
        and any other arrays written inside it. Writes land immediately;
        each touched variable's canonical synchronisation runs once at exit,
        in the same order on every rank. Legacy untagged callbacks (plain
        ``add_callback``) keep per-event replay, which is rank-local.

        Parameters
        ----------
        context_info : str, optional
            Optional information about the context (for debugging)

        Example
        -------
        >>> with arr.delay_callback("batch update"):
        ...     arr[0] = 1
        ...     arr[1] = 2
        ...     arr[2] = 3
        # Deferred synchronisation runs here, once
        """

        return _DelayCallbacksContext(context_info)

    @staticmethod
    def delay_callbacks_global(context_info=None):
        """
        Create a delay context without a specific array instance.

        Same semantics as :meth:`delay_callback` — the context is global
        either way. ``uw.synchronised_array_update`` is the public wrapper.

        Example
        -------
        >>> with NDArray_With_Callback.delay_callbacks_global("field update"):
        ...     temperature.array[...] = new_T
        ...     material.array[...] = new_material
        # Each touched variable is synchronised exactly once, here
        """

        return _DelayCallbacksContext(context_info)

    def _trigger_callback(
        self, operation: str, indices=None, old_value=None, new_value=None, data_has_changed=True
    ):
        """
        Internal method to trigger all registered callbacks.

        Parameters
        ----------
        operation : str
            Name of the operation that triggered the callback
        indices : tuple or slice, optional
            Indices that were modified
        old_value : None
            Always None from internal operations (see class docstring);
            the parameter and dict key remain for compatibility
        new_value : array-like, optional
            New value(s) at the modified location
        data_has_changed : bool, optional
            Whether this operation may have changed the array data (default True)
        """
        if not self._callback_enabled or not self._callbacks:
            return

        change_info = {
            "operation": operation,
            "indices": indices,
            "old_value": old_value,
            "new_value": new_value,
            "array_shape": self.shape,
            "array_dtype": self.dtype,
            "data_has_changed": data_has_changed,
        }

        # Check if we're in a delay callback context
        if _delayed_callback_manager.is_delaying():
            for callback in self._callbacks:
                canonical_ref = getattr(callback, "_canonical_ref", None)
                if canonical_ref is None:
                    # Legacy untagged callback: per-event queue, replayed
                    # rank-locally at exit (must not contain collectives).
                    _delayed_callback_manager.add_delayed_callback(self, callback, change_info)
                    continue
                # Canonical-guarded callback: mark the variable dirty; it is
                # flushed ONCE at context exit, in the same order on every
                # rank. Copies are skipped — the parent write-back marks.
                if not data_has_changed:
                    continue
                canonical = canonical_ref()
                if canonical is None:
                    continue
                if not _base_chain_resolves(self, canonical):
                    continue
                _delayed_callback_manager.mark_dirty(canonical)
        else:
            # Execute callbacks immediately. Exceptions PROPAGATE: a swallowed
            # callback failure leaves PETSc out of sync with the canonical
            # array on this rank only — the silent desynchronisation that hid
            # the #376 parallel hang.
            for callback in self._callbacks.copy():  # Copy in case callbacks modify the list
                callback(self, change_info)

    def __setitem__(self, key, value):
        """Override setitem to trigger callbacks on assignment."""
        # Handle UnitAwareArray values by extracting magnitude
        # This allows: T.array[...] = uw.function.evaluate(...) where evaluate returns UnitAwareArray
        # Without this, numpy raises "only length-1 arrays can be converted to Python scalars"
        actual_value = value
        if hasattr(value, 'magnitude'):
            # UnitAwareArray or similar - extract the raw numeric data
            actual_value = value.magnitude

        # Perform the actual assignment
        super().__setitem__(key, actual_value)

        # Trigger callbacks
        self._trigger_callback("setitem", indices=key, new_value=value)

    def __iadd__(self, other):
        """In-place addition with callback."""
        if self._disable_inplace_operators:
            raise RuntimeError(
                "In-place addition (+=) is disabled for parallel safety. "
                "Use explicit assignment instead: arr = arr + other"
            )

        result = super().__iadd__(other)
        self._trigger_callback("iadd", new_value=other)
        return result

    def __isub__(self, other):
        """In-place subtraction with callback."""
        if self._disable_inplace_operators:
            raise RuntimeError(
                "In-place subtraction (-=) is disabled for parallel safety. "
                "Use explicit assignment instead: arr = arr - other"
            )

        result = super().__isub__(other)
        self._trigger_callback("isub", new_value=other)
        return result

    def __imul__(self, other):
        """In-place multiplication with callback."""
        if self._disable_inplace_operators:
            raise RuntimeError(
                "In-place multiplication (*=) is disabled for parallel safety. "
                "Use explicit assignment instead: arr = arr * other"
            )

        result = super().__imul__(other)
        self._trigger_callback("imul", new_value=other)
        return result

    def __itruediv__(self, other):
        """In-place true division with callback."""
        if self._disable_inplace_operators:
            raise RuntimeError(
                "In-place division (/=) is disabled for parallel safety. "
                "Use explicit assignment instead: arr = arr / other"
            )

        result = super().__itruediv__(other)
        self._trigger_callback("itruediv", new_value=other)
        return result

    def __ifloordiv__(self, other):
        """In-place floor division with callback."""
        if self._disable_inplace_operators:
            raise RuntimeError(
                "In-place floor division (//=) is disabled for parallel safety. "
                "Use explicit assignment instead: arr = arr // other"
            )

        result = super().__ifloordiv__(other)
        self._trigger_callback("ifloordiv", new_value=other)
        return result

    def __imod__(self, other):
        """In-place modulo with callback."""
        if self._disable_inplace_operators:
            raise RuntimeError(
                "In-place modulo (%=) is disabled for parallel safety. "
                "Use explicit assignment instead: arr = arr % other"
            )

        result = super().__imod__(other)
        self._trigger_callback("imod", new_value=other)
        return result

    def __ipow__(self, other):
        """In-place power with callback."""
        if self._disable_inplace_operators:
            raise RuntimeError(
                "In-place power (**=) is disabled for parallel safety. "
                "Use explicit assignment instead: arr = arr ** other"
            )

        result = super().__ipow__(other)
        self._trigger_callback("ipow", new_value=other)
        return result

    def __iand__(self, other):
        """In-place bitwise and with callback."""
        if self._disable_inplace_operators:
            raise RuntimeError(
                "In-place bitwise and (&=) is disabled for parallel safety. "
                "Use explicit assignment instead: arr = arr & other"
            )

        result = super().__iand__(other)
        self._trigger_callback("iand", new_value=other)
        return result

    def __ior__(self, other):
        """In-place bitwise or with callback."""
        if self._disable_inplace_operators:
            raise RuntimeError(
                "In-place bitwise or (|=) is disabled for parallel safety. "
                "Use explicit assignment instead: arr = arr | other"
            )

        result = super().__ior__(other)
        self._trigger_callback("ior", new_value=other)
        return result

    def __ixor__(self, other):
        """In-place bitwise xor with callback."""
        if self._disable_inplace_operators:
            raise RuntimeError(
                "In-place bitwise xor (^=) is disabled for parallel safety. "
                "Use explicit assignment instead: arr = arr ^ other"
            )

        result = super().__ixor__(other)
        self._trigger_callback("ixor", new_value=other)
        return result

    def __ilshift__(self, other):
        """In-place left shift with callback."""
        if self._disable_inplace_operators:
            raise RuntimeError(
                "In-place left shift (<<=) is disabled for parallel safety. "
                "Use explicit assignment instead: arr = arr << other"
            )

        result = super().__ilshift__(other)
        self._trigger_callback("ilshift", new_value=other)
        return result

    def __irshift__(self, other):
        """In-place right shift with callback."""
        if self._disable_inplace_operators:
            raise RuntimeError(
                "In-place right shift (>>=) is disabled for parallel safety. "
                "Use explicit assignment instead: arr = arr >> other"
            )

        result = super().__irshift__(other)
        self._trigger_callback("irshift", new_value=other)
        return result

    def fill(self, value):
        """Fill array with scalar value, triggering callback."""
        super().fill(value)
        self._trigger_callback("fill", new_value=value)

    def sort(self, axis=-1, kind=None, order=None):
        """Sort array in-place, triggering callback."""
        super().sort(axis=axis, kind=kind, order=order)
        self._trigger_callback("sort")

    def resize(self, new_shape, refcheck=True):
        """Resize array in-place, triggering callback."""
        super().resize(new_shape, refcheck=refcheck)
        self._trigger_callback("resize", new_value=new_shape)

    def copy(self, order="C"):
        """
        Return a copy of the array.

        The copy will have the same callbacks registered but will be independent.
        """
        result = super().copy(order=order).view(NDArray_With_Callback)
        result._callbacks = self._callbacks.copy()
        result._owner = self._owner
        result._callback_enabled = self._callback_enabled
        result._disable_inplace_operators = self._disable_inplace_operators
        return result

    def view(self, dtype=None, type=None):
        """
        Return a view of the array.

        Views share callbacks with the original array.
        """
        # Use numpy's ndarray.view directly to avoid recursion
        if type is None and dtype is None:
            # Simple view with same type and dtype
            result = np.ndarray.view(self, NDArray_With_Callback)
        elif type is None:
            # View with different dtype, then cast to our type
            temp_view = np.ndarray.view(self, dtype)
            result = np.ndarray.view(temp_view, NDArray_With_Callback)
        else:
            # Use specified type (may not be our type)
            result = np.ndarray.view(self, dtype, type)

        # Copy our attributes to the result if it's our type
        if isinstance(result, NDArray_With_Callback):
            result._callbacks = self._callbacks  # Share callbacks (not copy)
            result._owner = self._owner
            result._callback_enabled = self._callback_enabled
            result._disable_inplace_operators = self._disable_inplace_operators

        return result

    def sync_data(self, new_data):
        """
        Update array with new data, preserving callbacks and all metadata.

        This method efficiently handles both same-size and different-size data updates.
        For same-size updates, it uses efficient in-place copying. For different sizes,
        it creates a new array object but preserves all metadata and callbacks.

        Parameters
        ----------
        new_data : array-like
            New data to sync into this array. Can be different size/shape.

        Returns
        -------
        result : NDArray_With_Callback
            For same-size: returns self (same object)
            For different-size: returns new object with same metadata

        Notes
        -----
        - For same-size data: Uses efficient in-place copy (preserves object identity)
        - For different sizes: Creates new object but copies all callbacks/metadata
        - All callbacks, owner references, and settings are preserved
        - Triggers 'sync_data' callback after update

        Examples
        --------
        >>> arr = NDArray_With_Callback([1, 2, 3])
        >>> result = arr.sync_data([4, 5, 6])  # Same size: returns same object
        >>> assert result is arr
        >>> result = arr.sync_data([7, 8, 9, 10, 11])  # Different size: new object
        >>> assert result is not arr  # Different object
        >>> assert len(result._callbacks) == len(arr._callbacks)  # Same callbacks
        """
        new_array = np.asarray(new_data)

        if new_array.shape == self.shape and new_array.dtype == self.dtype:
            # Same size and dtype: ultra-efficient in-place copy
            np.copyto(self, new_array)

            # Trigger callback for the sync operation
            self._trigger_callback(
                "sync_data",
                new_value=new_array,
                indices=None,  # Full array update
                data_has_changed=False,  # Sync operation doesn't represent user data change
            )

            return self
        else:
            # Different size/dtype: create new object with same metadata
            # This is more reliable than trying to modify the existing array

            new_obj = type(self)(
                new_array,
                owner=self._owner() if self._owner is not None else None,
                disable_inplace_operators=self._disable_inplace_operators,
            )

            # Re-home callbacks onto the new object. Canonical-guarded
            # callbacks are bound (by weakref) to THIS array's identity —
            # copying their wrappers verbatim would leave callbacks that
            # never fire on the new object (every write would classify as
            # a foreign copy). Re-register their original functions against
            # the new canonical; plain callbacks copy across unchanged.
            for registered in self._callbacks:
                original = getattr(registered, "_wrapped", None)
                if original is not None:
                    new_obj.add_canonical_callback(original)
                else:
                    new_obj.add_callback(registered)
            new_obj._callback_enabled = self._callback_enabled

            # Trigger callback on the new object
            new_obj._trigger_callback(
                "sync_data",
                new_value=new_array,
                indices=None,
                data_has_changed=False,  # Sync operation doesn't represent user data change
            )

            return new_obj

    def __reduce__(self):
        """Support for pickling."""
        # Get the parent's reduce result
        pickled_state = super().__reduce__()

        # Add our custom attributes to the state
        new_state = pickled_state[2] + (
            self._callbacks,
            self._owner,
            self._callback_enabled,
            self._disable_inplace_operators,
        )

        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        """Support for unpickling."""
        # Split our custom attributes from the parent's state
        parent_state = state[:-4]
        self._callbacks, self._owner, self._callback_enabled, self._disable_inplace_operators = (
            state[-4:]
        )

        # Call parent's setstate
        super().__setstate__(parent_state)

    def __repr__(self):
        """String representation showing callback information."""
        base_repr = super().__repr__()
        callback_info = f", callbacks={len(self._callbacks)}"

        # Insert callback info before the closing parenthesis
        if base_repr.startswith("array(") and base_repr.endswith(")"):
            return base_repr[:-1] + callback_info + ")"
        else:
            return base_repr + callback_info

    # === GLOBAL REDUCTION OPERATIONS (MPI-aware) ===
    # These operations reduce across all MPI ranks.
    # Subclasses (like UnitAwareArray) can override to add unit preservation.

    def global_max(self, axis=None, out=None, keepdims=False):
        """
        Return maximum across all MPI ranks.

        For scalar results (axis=None), performs MPI reduction. For array results,
        performs component-wise maximum.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis along which to operate (default: None = reduce all dimensions)
        out : ndarray, optional
            Alternative output array
        keepdims : bool, optional
            Keep reduced dimensions as size 1 (default: False)

        Returns
        -------
        scalar or ndarray
            Global maximum value(s)
        """
        from mpi4py import MPI

        # Try to get underworld MPI comm, fall back to MPI.COMM_WORLD
        try:
            import underworld3 as uw
            comm = uw.mpi.comm
        except (ImportError, AttributeError):
            comm = MPI.COMM_WORLD

        # Handle empty arrays (use -inf as identity for max)
        if self.size == 0:
            if axis is None and not keepdims:
                local_max = -np.inf
            else:
                # Determine result shape for empty array
                if axis is None:
                    result_shape = tuple()
                elif keepdims:
                    result_shape = list(self.shape)
                    if isinstance(axis, int):
                        result_shape[axis] = 1
                    else:
                        for ax in axis:
                            result_shape[ax] = 1
                    result_shape = tuple(result_shape)
                else:
                    result_shape = tuple(
                        s for i, s in enumerate(self.shape)
                        if i not in (axis if isinstance(axis, tuple) else (axis,))
                    )
                local_max = np.full(result_shape, -np.inf)
        else:
            local_max = np.asarray(self).max(axis=axis, out=out, keepdims=keepdims)

        # Scalar result - perform MPI reduction
        if axis is None and not keepdims:
            return comm.allreduce(float(local_max), op=MPI.MAX)

        # Array result - component-wise reduction
        local_arr = np.asarray(local_max)

        if local_arr.ndim == 1:
            global_arr = np.array([
                comm.allreduce(float(local_arr[i]), op=MPI.MAX)
                for i in range(len(local_arr))
            ])
        else:
            global_arr = np.empty_like(local_arr)
            comm.Allreduce(local_arr, global_arr, op=MPI.MAX)

        return global_arr

    def global_min(self, axis=None, out=None, keepdims=False):
        """
        Return minimum across all MPI ranks.

        For scalar results (axis=None), performs MPI reduction. For array results,
        performs component-wise minimum.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis along which to operate (default: None = reduce all dimensions)
        out : ndarray, optional
            Alternative output array
        keepdims : bool, optional
            Keep reduced dimensions as size 1 (default: False)

        Returns
        -------
        scalar or ndarray
            Global minimum value(s)
        """
        from mpi4py import MPI

        try:
            import underworld3 as uw
            comm = uw.mpi.comm
        except (ImportError, AttributeError):
            comm = MPI.COMM_WORLD

        # Handle empty arrays (use +inf as identity for min)
        if self.size == 0:
            if axis is None and not keepdims:
                local_min = np.inf
            else:
                if axis is None:
                    result_shape = tuple()
                elif keepdims:
                    result_shape = list(self.shape)
                    if isinstance(axis, int):
                        result_shape[axis] = 1
                    else:
                        for ax in axis:
                            result_shape[ax] = 1
                    result_shape = tuple(result_shape)
                else:
                    result_shape = tuple(
                        s for i, s in enumerate(self.shape)
                        if i not in (axis if isinstance(axis, tuple) else (axis,))
                    )
                local_min = np.full(result_shape, np.inf)
        else:
            local_min = np.asarray(self).min(axis=axis, out=out, keepdims=keepdims)

        # Scalar result
        if axis is None and not keepdims:
            return comm.allreduce(float(local_min), op=MPI.MIN)

        # Array result
        local_arr = np.asarray(local_min)

        if local_arr.ndim == 1:
            global_arr = np.array([
                comm.allreduce(float(local_arr[i]), op=MPI.MIN)
                for i in range(len(local_arr))
            ])
        else:
            global_arr = np.empty_like(local_arr)
            comm.Allreduce(local_arr, global_arr, op=MPI.MIN)

        return global_arr

    def global_sum(self, axis=None, dtype=None, out=None, keepdims=False):
        """
        Return sum across all MPI ranks.

        For scalar results (axis=None), performs MPI reduction. For array results,
        performs component-wise sum.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis along which to operate (default: None = reduce all dimensions)
        dtype : data-type, optional
            Type of returned array
        out : ndarray, optional
            Alternative output array
        keepdims : bool, optional
            Keep reduced dimensions as size 1 (default: False)

        Returns
        -------
        scalar or ndarray
            Global sum value(s)
        """
        from mpi4py import MPI

        try:
            import underworld3 as uw
            comm = uw.mpi.comm
        except (ImportError, AttributeError):
            comm = MPI.COMM_WORLD

        local_sum = np.asarray(self).sum(axis=axis, dtype=dtype, out=out, keepdims=keepdims)

        # Scalar result
        if axis is None and not keepdims:
            return comm.allreduce(float(local_sum), op=MPI.SUM)

        # Array result
        local_arr = np.asarray(local_sum)

        if local_arr.ndim == 1:
            global_arr = np.array([
                comm.allreduce(float(local_arr[i]), op=MPI.SUM)
                for i in range(len(local_arr))
            ])
        else:
            global_arr = np.empty_like(local_arr)
            comm.Allreduce(local_arr, global_arr, op=MPI.SUM)

        return global_arr

    def global_mean(self, axis=None, dtype=None, out=None, keepdims=False):
        """
        Return mean across all MPI ranks.

        Computes the true global mean by summing all values across ranks and
        dividing by total count.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis along which to operate (default: None = reduce all dimensions)
        dtype : data-type, optional
            Type of returned array
        out : ndarray, optional
            Alternative output array
        keepdims : bool, optional
            Keep reduced dimensions as size 1 (default: False)

        Returns
        -------
        scalar or ndarray
            Global mean value(s)
        """
        from mpi4py import MPI

        try:
            import underworld3 as uw
            comm = uw.mpi.comm
        except (ImportError, AttributeError):
            comm = MPI.COMM_WORLD

        # Get local count
        if axis is None:
            local_count = self.size
        elif isinstance(axis, int):
            local_count = self.shape[axis]
        else:
            local_count = np.prod([self.shape[ax] for ax in axis])

        # Get global sum and count
        global_sum = self.global_sum(axis=axis, dtype=dtype, keepdims=keepdims)
        global_count = comm.allreduce(local_count, op=MPI.SUM)

        # Compute mean
        if axis is None and not keepdims:
            return float(global_sum) / global_count
        else:
            return np.asarray(global_sum) / global_count

    def global_size(self):
        """
        Return total number of elements across all MPI ranks.

        Useful for computing global statistics that require total element count.

        Returns
        -------
        int
            Total number of elements summed across all MPI ranks
        """
        from mpi4py import MPI

        try:
            import underworld3 as uw
            comm = uw.mpi.comm
        except (ImportError, AttributeError):
            comm = MPI.COMM_WORLD

        return comm.allreduce(self.size, op=MPI.SUM)

    def global_norm(self, ord=None):
        """
        Return 2-norm across all MPI ranks.

        Computes sqrt(sum of squares) across all ranks.

        Parameters
        ----------
        ord : {None, 2}, optional
            Order of the norm (only 2-norm supported, default: None = 2-norm)

        Returns
        -------
        float
            Global 2-norm value
        """
        from mpi4py import MPI

        try:
            import underworld3 as uw
            comm = uw.mpi.comm
        except (ImportError, AttributeError):
            comm = MPI.COMM_WORLD

        if ord is not None and ord != 2:
            raise NotImplementedError(
                f"global_norm() only supports ord=None or ord=2 (2-norm), got ord={ord}"
            )

        # Compute local sum of squares
        local_arr = np.asarray(self)
        local_sq_sum = np.sum(local_arr**2)

        # Global sum of squares
        global_sq_sum = comm.allreduce(float(local_sq_sum), op=MPI.SUM)

        return np.sqrt(global_sq_sum)

    def global_rms(self):
        """
        Return root mean square across all MPI ranks.

        Computes RMS = sqrt(sum of squares / total count) across all ranks.

        Returns
        -------
        float
            Global RMS value
        """
        norm = self.global_norm()
        size = self.global_size()
        return norm / np.sqrt(size)
