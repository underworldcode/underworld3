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
    Thread-local manager for delayed callbacks across multiple NDArray_With_Callback instances.

    This allows batch operations across multiple arrays to accumulate callbacks
    and trigger them all at once when the context exits.
    """

    def __init__(self):
        self._local = threading.local()

    def _get_state(self):
        """Get or create thread-local state."""
        if not hasattr(self._local, "delay_stack"):
            self._local.delay_stack = []
            self._local.delayed_callbacks = []
        return self._local

    def is_delaying(self):
        """Check if callbacks are currently being delayed."""
        state = self._get_state()
        return len(state.delay_stack) > 0

    def push_delay_context(self, context_info=None):
        """Enter a new delay context."""
        state = self._get_state()
        state.delay_stack.append(
            {
                "context_info": context_info,
                "callback_count": len(state.delayed_callbacks),
            }
        )

    def pop_delay_context(self):
        """Exit delay context and return callbacks accumulated in this context."""
        state = self._get_state()
        if not state.delay_stack:
            return []

        context = state.delay_stack.pop()
        start_idx = context["callback_count"]

        # Get callbacks from this context level
        context_callbacks = state.delayed_callbacks[start_idx:]

        # If we're exiting the outermost context, clear all callbacks
        if not state.delay_stack:
            state.delayed_callbacks.clear()
        else:
            # Remove only this context's callbacks (keep outer context callbacks)
            state.delayed_callbacks = state.delayed_callbacks[:start_idx]

        return context_callbacks

    def add_delayed_callback(self, array, callback_func, change_info):
        """Add a callback to the delayed execution queue."""
        state = self._get_state()
        state.delayed_callbacks.append(
            {
                "array": array,
                "callback": callback_func,
                "change_info": change_info.copy(),
            }
        )


# Global instance for managing delayed callbacks
_delayed_callback_manager = DelayedCallbackManager()


class NDArray_With_Callback(np.ndarray):
    """
    # NDArray_With_Callback

    A numpy ndarray subclass that triggers **callbacks** when array data is modified.
    This class maintains full numpy array compatibility while providing reactive programming
    capabilities for scientific computing applications.

    ## Mathematical Representation

    Given an array $\\mathbf{A} \\in \\mathbb{R}^{n \\times m}$, any modification operation
    $\\mathcal{O}(\\mathbf{A}) \\rightarrow \\mathbf{A}'$ will trigger registered callbacks:

    $$\\mathbf{A}' = \\mathcal{O}(\\mathbf{A}) \\implies \\text{callback}(\\mathbf{A}', \\text{change\\_info})$$

    Where $\\mathcal{O}$ represents operations like assignment, in-place arithmetic, or array methods.

    ## Callback Interface

    ### Function Signature
    ```python
    def callback(array: NDArray_With_Callback, change_info: dict) -> None:
        # Handle array modification
        pass
    ```

    ### change_info Dictionary

    | Key | Type | Description |
    |-----|------|-------------|
    | `operation` | `str` | Operation name ('setitem', 'iadd', 'fill', etc.) |
    | `indices` | `tuple/slice/None` | Location of change (for setitem operations) |
    | `old_value` | `array-like/None` | Previous values (when available) |
    | `new_value` | `array-like` | New values being assigned |
    | `array_shape` | `tuple` | Current shape of the array |
    | `array_dtype` | `np.dtype` | Data type of the array |

    ## Usage Examples

    ### Basic Reactive Array
    ```python
    def on_change(array, change_info):
        print(f"🔔 {change_info['operation']} at {change_info['indices']}")
        print(f"   New value: {change_info['new_value']}")

    # Create reactive array
    arr = NDArray_With_Callback([1, 2, 3])
    arr.set_callback(on_change)

    # Modifications trigger callbacks
    arr[0] = 99      # 🔔 setitem at 0, New value: 99
    arr += 10        # 🔔 iadd at None, New value: 10
    arr.fill(0)      # 🔔 fill at None, New value: 0
    ```

    ### Scientific Computing Integration
    ```python
    class Mesh:
        def __init__(self, coordinates):
            self._coords = coordinates

        @property
        def data(self):
            arr = NDArray_With_Callback(self._coords, owner=self)
            arr.set_callback(self._on_coordinates_changed)
            return arr

        def _on_coordinates_changed(self, array, info):
            # Invalidate cached computations
            self._jacobians = None
            self._mesh_quality = None
            # Trigger dependent updates
            self._update_connectivity()
    ```

    ### Delayed Callback Context
    ```python
    # Batch multiple operations
    with arr.delay_callback("batch update"):
        arr[0] = 1
        arr[1] = 2
        arr[2] = 3
    # All callbacks fire here (with MPI synchronization)

    # Global coordination across arrays
    with NDArray_With_Callback.delay_callbacks_global("mesh deformation"):
        mesh.data += displacement
        swarm.data += velocity * dt
    # Synchronized callback execution across all arrays
    ```

    ## Advanced Features

    - **Multiple callbacks**: `add_callback()`, `remove_callback()`, `clear_callbacks()`
    - **Enable/disable**: `enable_callbacks()`, `disable_callbacks()`
    - **Delayed execution**: `delay_callback()`, `delay_callbacks_global()`
    - **MPI synchronization**: Automatic barriers in parallel contexts
    - **Weak references**: Owner tracking without circular dependencies
    - **Error resilience**: Callback exceptions don't break array operations
    - **Global reductions**: MPI-aware `global_max()`, `global_min()`, `global_sum()`, etc.

    ## Global Reduction Operations (MPI-aware)

    These methods perform reduction operations across all MPI ranks, essential for
    parallel scientific computing where data is distributed across processes.

    | Method | Description |
    |--------|-------------|
    | `global_max(axis=None)` | Maximum value across all MPI ranks |
    | `global_min(axis=None)` | Minimum value across all MPI ranks |
    | `global_sum(axis=None)` | Sum of all values across all MPI ranks |
    | `global_mean(axis=None)` | True mean (global sum / global count) |
    | `global_size()` | Total number of elements across all ranks |
    | `global_norm(ord=2)` | 2-norm (Euclidean) across all ranks |
    | `global_rms()` | Root mean square across all ranks |

    ### Usage Example
    ```python
    # In parallel code, each rank has a portion of mesh coordinates
    coords = mesh.X.coords  # NDArray_With_Callback or subclass

    # Find global bounds (across all MPI ranks)
    x_min = coords[:, 0].global_min()  # True minimum across all ranks
    x_max = coords[:, 0].global_max()  # True maximum across all ranks

    # Compute global statistics
    mean_coord = coords.global_mean()  # True mean (not just local mean!)
    total_size = coords.global_size()  # Total elements across all ranks

    # Compute global norms
    rms_value = coords.global_rms()    # Root mean square
    l2_norm = coords.global_norm()     # Euclidean norm
    ```

    ### Important Notes

    - These methods use MPI collective operations (`allreduce`)
    - **All ranks must call these methods** (they are collective operations)
    - Subclasses like `UnitAwareArray` override these to preserve units

    ## Performance Notes

    - **Zero overhead** when callbacks disabled
    - **Minimal impact** on array operations (< 5% typical)
    - **Batch processing** via delayed contexts for optimal performance
    - **Thread-safe** delayed callback management
    - **Memory efficient** weak reference ownership tracking
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
            If True, in-place operators (+=, -=, *=, /=, etc.) will raise RuntimeError
            for parallel safety. Default is False for backward compatibility.
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

    def __array_wrap__(self, result):
        """
        Called after numpy operations to wrap results back to our type.
        """
        if result.shape == ():
            # Scalar result, return as numpy scalar
            return result.item()

        # For in-place operations that return the same array, return self
        # Use numpy's view to avoid recursion
        try:
            self_as_ndarray = np.ndarray.view(self, np.ndarray)
            if result is self_as_ndarray or (
                hasattr(result, "base") and hasattr(self, "base") and result.base is self.base
            ):
                return self
        except:
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

    def remove_callback(self, callback: Callable):
        """
        Remove a specific callback function.

        Parameters
        ----------
        callback : callable
            Function to remove from callback list
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)

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
        Context manager to delay callback execution until context exit.

        During the context, all callbacks from this array (and any other arrays
        using delay_callback) will be accumulated and executed when the outermost
        context exits.

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
        # All callbacks fire here at context exit
        """

        class DelayCallbackContext:
            def __init__(self, context_info):
                self.context_info = context_info

            def __enter__(self):
                # MPI barrier to ensure all processes enter delay context together
                if _has_uw_mpi:
                    try:
                        uw.mpi.barrier()
                    except Exception as e:
                        logger.warning(f"MPI barrier failed on delay context enter: {e}")

                _delayed_callback_manager.push_delay_context(self.context_info)
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                # Get callbacks accumulated during this context
                delayed_callbacks = _delayed_callback_manager.pop_delay_context()

                # MPI barrier to ensure all processes finish their delayed operations
                # before any process starts executing callbacks
                if _has_uw_mpi:
                    try:
                        uw.mpi.barrier()
                    except Exception as e:
                        logger.warning(f"MPI barrier failed before delayed callback execution: {e}")

                # Execute all delayed callbacks
                for callback_item in delayed_callbacks:
                    try:
                        callback_item["callback"](
                            callback_item["array"], callback_item["change_info"]
                        )
                    except Exception as e:
                        logger.warning(f"Delayed callback error: {e}")

                # MPI barrier to ensure all processes complete their callbacks
                # before any process exits the context
                if _has_uw_mpi:
                    try:
                        uw.mpi.barrier()
                    except Exception as e:
                        logger.warning(f"MPI barrier failed after delayed callback execution: {e}")

                # Don't suppress exceptions from the context
                return False

        return DelayCallbackContext(context_info)

    @staticmethod
    def delay_callbacks_global(context_info=None):
        """
        Static method to create a global delay context for all NDArray_With_Callback instances.

        This is useful when you don't have a specific array instance but want to delay
        callbacks from multiple arrays.

        Example
        -------
        >>> with NDArray_With_Callback.delay_callbacks_global("mesh update"):
        ...     mesh.data[0] = new_pos
        ...     swarm.data += displacement
        # All callbacks from all arrays fire here
        """

        class GlobalDelayCallbackContext:
            def __init__(self, context_info):
                self.context_info = context_info

            def __enter__(self):
                # MPI barrier to ensure all processes enter delay context together
                if _has_uw_mpi:
                    try:
                        uw.mpi.barrier()
                    except Exception as e:
                        logger.warning(f"MPI barrier failed on global delay context enter: {e}")

                _delayed_callback_manager.push_delay_context(self.context_info)
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                # Get callbacks accumulated during this context
                delayed_callbacks = _delayed_callback_manager.pop_delay_context()

                # MPI barrier to ensure all processes finish their delayed operations
                # before any process starts executing callbacks
                if _has_uw_mpi:
                    try:
                        uw.mpi.barrier()
                    except Exception as e:
                        logger.warning(
                            f"MPI barrier failed before global delayed callback execution: {e}"
                        )

                # Execute all delayed callbacks
                for callback_item in delayed_callbacks:
                    try:
                        callback_item["callback"](
                            callback_item["array"], callback_item["change_info"]
                        )
                    except Exception as e:
                        logger.warning(f"Delayed callback error: {e}")

                # MPI barrier to ensure all processes complete their callbacks
                # before any process exits the context
                if _has_uw_mpi:
                    try:
                        uw.mpi.barrier()
                    except Exception as e:
                        logger.warning(
                            f"MPI barrier failed after global delayed callback execution: {e}"
                        )

                return False

        return GlobalDelayCallbackContext(context_info)

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
        old_value : array-like, optional
            Previous value(s) at the modified location
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
            # Add callbacks to the delayed execution queue
            for callback in self._callbacks:
                _delayed_callback_manager.add_delayed_callback(self, callback, change_info)
        else:
            # Execute callbacks immediately
            for callback in self._callbacks.copy():  # Copy in case callbacks modify the list
                try:
                    callback(self, change_info)
                except Exception as e:
                    logger.warning(f"Callback error in {callback}: {e}")

    def __setitem__(self, key, value):
        """Override setitem to trigger callbacks on assignment."""
        if self._callback_enabled and self._callbacks:
            try:
                old_value = self[key].copy() if hasattr(self[key], "copy") else self[key]
            except (IndexError, ValueError):
                old_value = None
        else:
            old_value = None

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
        self._trigger_callback("setitem", indices=key, old_value=old_value, new_value=value)

    def __iadd__(self, other):
        """In-place addition with callback."""
        if self._disable_inplace_operators:
            raise RuntimeError(
                "In-place addition (+=) is disabled for parallel safety. "
                "Use explicit assignment instead: arr = arr + other"
            )

        if self._callback_enabled and self._callbacks:
            old_value = self.copy()
        else:
            old_value = None

        result = super().__iadd__(other)
        self._trigger_callback("iadd", old_value=old_value, new_value=other)
        return result

    def __isub__(self, other):
        """In-place subtraction with callback."""
        if self._disable_inplace_operators:
            raise RuntimeError(
                "In-place subtraction (-=) is disabled for parallel safety. "
                "Use explicit assignment instead: arr = arr - other"
            )

        if self._callback_enabled and self._callbacks:
            old_value = self.copy()
        else:
            old_value = None

        result = super().__isub__(other)
        self._trigger_callback("isub", old_value=old_value, new_value=other)
        return result

    def __imul__(self, other):
        """In-place multiplication with callback."""
        if self._disable_inplace_operators:
            raise RuntimeError(
                "In-place multiplication (*=) is disabled for parallel safety. "
                "Use explicit assignment instead: arr = arr * other"
            )

        if self._callback_enabled and self._callbacks:
            old_value = self.copy()
        else:
            old_value = None

        result = super().__imul__(other)
        self._trigger_callback("imul", old_value=old_value, new_value=other)
        return result

    def __itruediv__(self, other):
        """In-place true division with callback."""
        if self._disable_inplace_operators:
            raise RuntimeError(
                "In-place division (/=) is disabled for parallel safety. "
                "Use explicit assignment instead: arr = arr / other"
            )

        if self._callback_enabled and self._callbacks:
            old_value = self.copy()
        else:
            old_value = None

        result = super().__itruediv__(other)
        self._trigger_callback("itruediv", old_value=old_value, new_value=other)
        return result

    def __ifloordiv__(self, other):
        """In-place floor division with callback."""
        if self._disable_inplace_operators:
            raise RuntimeError(
                "In-place floor division (//=) is disabled for parallel safety. "
                "Use explicit assignment instead: arr = arr // other"
            )

        if self._callback_enabled and self._callbacks:
            old_value = self.copy()
        else:
            old_value = None

        result = super().__ifloordiv__(other)
        self._trigger_callback("ifloordiv", old_value=old_value, new_value=other)
        return result

    def __imod__(self, other):
        """In-place modulo with callback."""
        if self._disable_inplace_operators:
            raise RuntimeError(
                "In-place modulo (%=) is disabled for parallel safety. "
                "Use explicit assignment instead: arr = arr % other"
            )

        if self._callback_enabled and self._callbacks:
            old_value = self.copy()
        else:
            old_value = None

        result = super().__imod__(other)
        self._trigger_callback("imod", old_value=old_value, new_value=other)
        return result

    def __ipow__(self, other):
        """In-place power with callback."""
        if self._disable_inplace_operators:
            raise RuntimeError(
                "In-place power (**=) is disabled for parallel safety. "
                "Use explicit assignment instead: arr = arr ** other"
            )

        if self._callback_enabled and self._callbacks:
            old_value = self.copy()
        else:
            old_value = None

        result = super().__ipow__(other)
        self._trigger_callback("ipow", old_value=old_value, new_value=other)
        return result

    def __iand__(self, other):
        """In-place bitwise and with callback."""
        if self._disable_inplace_operators:
            raise RuntimeError(
                "In-place bitwise and (&=) is disabled for parallel safety. "
                "Use explicit assignment instead: arr = arr & other"
            )

        if self._callback_enabled and self._callbacks:
            old_value = self.copy()
        else:
            old_value = None

        result = super().__iand__(other)
        self._trigger_callback("iand", old_value=old_value, new_value=other)
        return result

    def __ior__(self, other):
        """In-place bitwise or with callback."""
        if self._disable_inplace_operators:
            raise RuntimeError(
                "In-place bitwise or (|=) is disabled for parallel safety. "
                "Use explicit assignment instead: arr = arr | other"
            )

        if self._callback_enabled and self._callbacks:
            old_value = self.copy()
        else:
            old_value = None

        result = super().__ior__(other)
        self._trigger_callback("ior", old_value=old_value, new_value=other)
        return result

    def __ixor__(self, other):
        """In-place bitwise xor with callback."""
        if self._disable_inplace_operators:
            raise RuntimeError(
                "In-place bitwise xor (^=) is disabled for parallel safety. "
                "Use explicit assignment instead: arr = arr ^ other"
            )

        if self._callback_enabled and self._callbacks:
            old_value = self.copy()
        else:
            old_value = None

        result = super().__ixor__(other)
        self._trigger_callback("ixor", old_value=old_value, new_value=other)
        return result

    def __ilshift__(self, other):
        """In-place left shift with callback."""
        if self._disable_inplace_operators:
            raise RuntimeError(
                "In-place left shift (<<=) is disabled for parallel safety. "
                "Use explicit assignment instead: arr = arr << other"
            )

        if self._callback_enabled and self._callbacks:
            old_value = self.copy()
        else:
            old_value = None

        result = super().__ilshift__(other)
        self._trigger_callback("ilshift", old_value=old_value, new_value=other)
        return result

    def __irshift__(self, other):
        """In-place right shift with callback."""
        if self._disable_inplace_operators:
            raise RuntimeError(
                "In-place right shift (>>=) is disabled for parallel safety. "
                "Use explicit assignment instead: arr = arr >> other"
            )

        if self._callback_enabled and self._callbacks:
            old_value = self.copy()
        else:
            old_value = None

        result = super().__irshift__(other)
        self._trigger_callback("irshift", old_value=old_value, new_value=other)
        return result

    def fill(self, value):
        """Fill array with scalar value, triggering callback."""
        if self._callback_enabled and self._callbacks:
            old_value = self.copy()
        else:
            old_value = None

        super().fill(value)
        self._trigger_callback("fill", old_value=old_value, new_value=value)

    def sort(self, axis=-1, kind=None, order=None):
        """Sort array in-place, triggering callback."""
        if self._callback_enabled and self._callbacks:
            old_value = self.copy()
        else:
            old_value = None

        super().sort(axis=axis, kind=kind, order=order)
        self._trigger_callback("sort", old_value=old_value)

    def resize(self, new_shape, refcheck=True):
        """Resize array in-place, triggering callback."""
        if self._callback_enabled and self._callbacks:
            old_value = self.copy()
            old_shape = self.shape
        else:
            old_value = None
            old_shape = None

        super().resize(new_shape, refcheck=refcheck)
        self._trigger_callback("resize", old_value=old_value, new_value=new_shape)

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

        # Store old info for callback
        if self._callback_enabled and self._callbacks:
            old_data = self.copy()
        else:
            old_data = None

        if new_array.shape == self.shape and new_array.dtype == self.dtype:
            # Same size and dtype: ultra-efficient in-place copy
            np.copyto(self, new_array)

            # Trigger callback for the sync operation
            self._trigger_callback(
                "sync_data",
                old_value=old_data,
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

            # Copy all callbacks and settings
            new_obj._callbacks = self._callbacks.copy()
            new_obj._callback_enabled = self._callback_enabled

            # Trigger callback on the new object
            new_obj._trigger_callback(
                "sync_data",
                old_value=old_data,
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
