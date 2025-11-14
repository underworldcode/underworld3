# underworld3/utilities/unit_aware_array.py
"""
Unit-Aware NDArray: Integration of units system with NDArray_With_Callback

This module provides UnitAwareArray, which extends NDArray_With_Callback with
comprehensive unit tracking and conversion capabilities.

Key Features:
- Automatic unit tracking for all array operations
- Unit compatibility checking for arithmetic operations
- Seamless integration with existing unit conversion utilities
- Preservation of callback functionality from NDArray_With_Callback
- Automatic conversion in mixed-unit operations when appropriate
"""

import numpy as np
from typing import Optional, Union, Any, Dict, List
from .nd_array_callback import NDArray_With_Callback
from ..function.unit_conversion import (
    has_units,
    detect_quantity_units,
    convert_quantity_units,
    add_units,
    convert_array_units,
)
# NOTE: get_units has been moved to units module
from ..units import get_units


class UnitAwareArray(NDArray_With_Callback):
    """
    A numpy ndarray subclass that combines callback functionality with unit awareness.

    This class extends NDArray_With_Callback to provide:
    - Automatic unit tracking and propagation
    - Unit compatibility checking for operations
    - Integration with UW3 unit conversion system
    - Preservation of all callback functionality

    Mathematical Representation:
    Given an array A with units [A], operations preserve dimensional consistency:
    - A [m] + B [m] → C [m]  ✓ (compatible units)
    - A [m] + B [s] → Error   ✗ (incompatible units)
    - A [m] * B [s] → C [m⋅s] ✓ (unit multiplication)
    - A [m] * 2     → C [m]   ✓ (scalar multiplication)

    Usage Examples:
    ```python
    # Create arrays with units
    length = UnitAwareArray([1, 2, 3], units="m")
    time = UnitAwareArray([0.1, 0.2, 0.3], units="s")

    # Operations preserve units
    velocity = length / time  # Result has units m/s

    # Unit checking prevents errors
    total = length + time  # Raises ValueError (incompatible units)

    # Automatic conversion when possible
    length_km = UnitAwareArray([1, 2, 3], units="km")
    total_length = length + length_km  # Converts km to m automatically

    # Callbacks still work
    def on_change(array, info):
        print(f"Array {array.units} changed: {info['operation']}")
    length.set_callback(on_change)
    length[0] = 5  # Triggers callback
    ```
    """

    def __new__(
        cls,
        input_array=None,
        units=None,
        owner=None,
        callback=None,
        unit_checking=True,
        auto_convert=True,
        **kwargs,
    ):
        """
        Create new UnitAwareArray instance.

        Parameters
        ----------
        input_array : array-like, optional
            Input data to create array from
        units : str or UWQuantity, optional
            Units for this array (e.g., "m", "m/s", "kg")
        owner : object, optional
            The object that owns this array (stored as weak reference)
        callback : callable, optional
            Initial callback function to register
        unit_checking : bool, optional
            If True, enforce unit compatibility in operations (default True)
        auto_convert : bool, optional
            If True, automatically convert compatible units (default True)
        **kwargs : dict
            Additional arguments passed to NDArray_With_Callback
        """
        # Create the NDArray_With_Callback instance
        obj = super().__new__(cls, input_array, owner, callback, **kwargs)

        # Initialize unit tracking
        obj._units = None
        obj._unit_checking = unit_checking
        obj._auto_convert = auto_convert
        obj._original_units = units  # Store original for reference

        # Set units if provided
        if units is not None:
            obj._set_units(units)
            # Initialize units backend for get_dimensionality() support
            from underworld3.utilities.units_mixin import PintBackend

            obj._units_backend = PintBackend()
        else:
            obj._units_backend = None

        return obj

    def __array_finalize__(self, obj):
        """Called whenever the system allocates a new array from this template."""
        # Call parent finalize first
        super().__array_finalize__(obj)

        if obj is None:
            return

        # Copy unit information from parent array
        self._units = getattr(obj, "_units", None)
        self._unit_checking = getattr(obj, "_unit_checking", True)
        self._auto_convert = getattr(obj, "_auto_convert", True)
        self._original_units = getattr(obj, "_original_units", None)
        self._units_backend = getattr(obj, "_units_backend", None)

    def _set_units(self, units):
        """
        Internal method to set units for this array.

        Parameters
        ----------
        units : str, Pint Unit, or UWQuantity
            Units specification
        """
        if isinstance(units, str):
            # Create a UWQuantity to validate units
            try:
                import underworld3 as uw

                unit_qty = uw.function.quantity(1.0, units)
                self._units = units
            except Exception as e:
                raise ValueError(f"Invalid units '{units}': {e}")
        elif has_units(units):
            # Extract units from UWQuantity or similar
            extracted_units = get_units(units)
            if extracted_units is not None:
                self._units = extracted_units
            else:
                # Fallback: get_units() returned None (e.g., for Pint Unit objects)
                # Convert to string representation
                self._units = str(units)
        else:
            self._units = str(units)

    @property
    def units(self):
        """Get the units of this array."""
        return self._units

    @property
    def has_units(self):
        """Check if this array has units."""
        return self._units is not None

    @property
    def dimensionality(self):
        """Get the dimensionality of this array."""
        if not self.has_units:
            return None
        if self._units_backend is None:
            return None
        quantity = self._units_backend.create_quantity(1.0, self._units)
        return self._units_backend.get_dimensionality(quantity)

    @property
    def unit_checking(self):
        """Check if unit compatibility checking is enabled."""
        return self._unit_checking

    @unit_checking.setter
    def unit_checking(self, value):
        """Enable/disable unit compatibility checking."""
        self._unit_checking = bool(value)

    @property
    def auto_convert(self):
        """Check if automatic unit conversion is enabled."""
        return self._auto_convert

    @auto_convert.setter
    def auto_convert(self, value):
        """Enable/disable automatic unit conversion."""
        self._auto_convert = bool(value)

    @property
    def magnitude(self):
        """
        Get the numerical values without units (like Pint's .magnitude).

        This returns a plain numpy array view of the data, stripping units.
        Useful when you need raw numerical values for dimensionless calculations.

        Returns
        -------
        np.ndarray
            Plain numpy array without unit tracking

        Examples
        --------
        >>> coords = mesh.X.coords  # UnitAwareArray with units="km"
        >>> x, y = coords[:, 0].magnitude, coords[:, 1].magnitude  # Plain arrays
        >>> temperature.array[:, 0, 0] = 300 + 2.6 * y  # Works - no units
        """
        # Use numpy's asarray to get a plain numpy array
        # This avoids our overridden view() method which preserves units
        return np.asarray(self)

    def to(self, target_units):
        """
        Convert this array to different units.

        Provides a unified interface matching Pint's `.to()` pattern.

        Parameters
        ----------
        target_units : str
            Target units to convert to

        Returns
        -------
        UnitAwareArray
            New array with converted values and target units

        Examples
        --------
        >>> coords = UnitAwareArray([1, 2, 3], units='km')
        >>> coords_m = coords.to('m')  # Convert to meters
        >>> print(coords_m)
        [1000. 2000. 3000.] [meter]
        """
        if not self.has_units:
            raise ValueError("Cannot convert units - array has no units")

        # Use Pint directly for more reliable conversion
        try:
            import underworld3 as uw

            # Get the unit registry from UW3's scaling module
            ureg = uw.scaling.units

            # Create Pint quantities using the registry
            source_quantity = ureg.Quantity(self.view(np.ndarray), self._units)
            target_quantity = source_quantity.to(target_units)

            # Create new UnitAwareArray with converted values
            return UnitAwareArray(
                target_quantity.magnitude,
                units=str(target_quantity.units),
                unit_checking=self._unit_checking,
                auto_convert=self._auto_convert,
            )

        except Exception as e:
            raise ValueError(f"Unit conversion failed: {e}")

    def _check_unit_compatibility(self, other, operation="operation"):
        """
        Check unit compatibility with another array or value.

        Parameters
        ----------
        other : array-like or scalar
            Other operand to check compatibility with
        operation : str
            Name of operation for error messages

        Returns
        -------
        tuple
            (compatible: bool, converted_other: array-like, result_units: str)
        """
        if not self._unit_checking:
            return True, other, self._units

        # Handle scalar values (dimensionless)
        if np.isscalar(other):
            if operation in ["add", "subtract"] and self.has_units:
                raise ValueError(
                    f"Cannot {operation} dimensionless scalar {other} "
                    f"to array with units '{self._units}'. "
                    f"Use array.to_units('dimensionless') or multiply by appropriate units."
                )
            return True, other, self._units

        # Initialize other_units to None
        other_units = None

        # Check if other has units
        if hasattr(other, "units"):
            # Could be UnitAwareArray, Pint Quantity, or UWQuantity
            other_units_obj = other.units
            # Convert to string if it's a Pint Unit object
            if hasattr(other_units_obj, "__str__") and not isinstance(other_units_obj, str):
                other_units = str(other_units_obj)
            else:
                other_units = other_units_obj

            # If it's a Pint Quantity or UWQuantity, extract the magnitude for operations
            if hasattr(other, "magnitude"):
                # For addition/subtraction, we need the actual value
                # For multiplication/division, this will be handled later
                if operation in ["add", "subtract"]:
                    # Extract scalar value for arithmetic
                    other = (
                        float(other.magnitude)
                        if np.isscalar(other.magnitude)
                        else np.asarray(other.magnitude)
                    )
        elif hasattr(other, "_units"):
            other_units = other._units
        elif has_units(other):
            other_units = get_units(other)

        # Both have no units - compatible
        if not self.has_units and other_units is None:
            return True, other, None

        # One has units, other doesn't - check operation type
        if self.has_units and other_units is None:
            if operation in ["add", "subtract"]:
                raise ValueError(
                    f"Cannot {operation} array with units '{self._units}' "
                    f"and dimensionless array. Convert to same unit system first."
                )
            return True, other, self._units

        if not self.has_units and other_units is not None:
            if operation in ["add", "subtract"]:
                raise ValueError(
                    f"Cannot {operation} dimensionless array "
                    f"and array with units '{other_units}'. Convert to same unit system first."
                )
            return True, other, other_units

        # Both have units - check compatibility
        if self._units == other_units:
            return True, other, self._units

        # Different units - try conversion if auto_convert enabled
        if self._auto_convert and operation in ["add", "subtract"]:
            try:
                # Try to convert other to self's units
                if hasattr(other, "to_units"):
                    converted_other = other.to_units(self._units)
                    return True, converted_other, self._units
                else:
                    # Convert using Pint directly for more reliable conversion
                    import underworld3 as uw

                    ureg = uw.scaling.units

                    # Create Pint quantities and convert
                    other_quantity = ureg.Quantity(np.asarray(other), other_units)
                    converted_quantity = other_quantity.to(self._units)

                    return True, converted_quantity.magnitude, self._units

            except Exception:
                # Conversion failed - incompatible units
                pass

        # Handle multiplication/division - use Pint for proper unit algebra
        if operation in ["multiply", "divide"]:
            try:
                import underworld3 as uw

                ureg = uw.scaling.units

                # Create Pint quantities to compute unit algebra
                self_qty = ureg.Quantity(1.0, self._units)

                # Handle if other is already a Pint Quantity
                if hasattr(other, "magnitude") and hasattr(other, "units"):
                    # Convert Pint Quantity to our units for value compatibility
                    try:
                        other_in_our_units = other.to(self._units)
                        other_qty = ureg.Quantity(1.0, self._units)
                        other_values = np.asarray(other_in_our_units.magnitude)
                    except:
                        # Can't convert - different dimensions, use as-is
                        other_qty = ureg.Quantity(1.0, other.units)
                        other_values = np.asarray(other.magnitude)
                else:
                    other_qty = ureg.Quantity(1.0, other_units)
                    other_values = np.asarray(other)

                # Perform the operation to get result units
                if operation == "multiply":
                    result_qty = self_qty * other_qty
                else:  # divide
                    result_qty = self_qty / other_qty

                # Extract the resulting units
                result_units_obj = result_qty.units

                # Check if dimensionless (both by string and by dimensionality)
                is_dimensionless = (
                    result_units_obj == ureg.dimensionless
                    or result_qty.dimensionality
                    == ureg.Quantity(1.0, "dimensionless").dimensionality
                )

                if is_dimensionless:
                    # Units cancel out - need to handle scale factor
                    # Get the magnitude which contains scale conversion factor
                    scale_factor = float(result_qty.magnitude)

                    # Convert the other array's values if scale factor != 1.0
                    if scale_factor != 1.0:
                        converted_other = other_values * scale_factor
                    else:
                        converted_other = other_values

                    # Return None for units (dimensionless)
                    return True, converted_other, None
                else:
                    # Units don't cancel - return string representation
                    result_units = str(result_units_obj)
                    return True, other_values, result_units

            except Exception as e:
                # Fallback to string concatenation if Pint fails
                if operation == "multiply":
                    result_units = f"({self._units})*({other_units})"
                else:  # divide
                    result_units = f"({self._units})/({other_units})"
                return True, other, result_units

        # Incompatible units for addition/subtraction
        raise ValueError(
            f"Cannot {operation} arrays with incompatible units: "
            f"'{self._units}' and '{other_units}'"
        )

    def _wrap_result(self, result, units="__unspecified__"):
        """
        Wrap operation result as UnitAwareArray with appropriate units.

        Parameters
        ----------
        result : array-like
            Result of operation
        units : str or None, optional
            Units for the result. If None, result is dimensionless (plain array).
            If not provided, defaults to self._units.

        Returns
        -------
        UnitAwareArray, ndarray, or scalar
            Wrapped result with units (or plain array if dimensionless)
        """
        if np.isscalar(result):
            # Scalar results don't need unit tracking
            return result

        # Determine final units
        if units == "__unspecified__":
            final_units = self._units
        else:
            final_units = units

        # If dimensionless (units explicitly set to None), return plain array
        if final_units is None:
            return np.asarray(result)

        # Preserve as UnitAwareArray with units
        return UnitAwareArray(
            result,
            units=final_units,
            unit_checking=self._unit_checking,
            auto_convert=self._auto_convert,
        )

    # === REDUCTION OPERATIONS ===
    # Override reduction methods to preserve units in scalar results

    def _wrap_scalar_result(self, value):
        """Wrap scalar result with units as UWQuantity."""
        if self.has_units:
            import underworld3 as uw

            return uw.function.quantity(float(value), self._units)
        return value

    def max(self, axis=None, out=None, keepdims=False, initial=None, where=True):
        """Return maximum with units preserved."""
        result = super().max(axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)
        if axis is None and not keepdims:
            # Scalar result - wrap with units
            return self._wrap_scalar_result(result)
        elif self.has_units:
            # Array result - wrap as UnitAwareArray
            return self._wrap_result(result, self._units)
        return result

    def min(self, axis=None, out=None, keepdims=False, initial=None, where=True):
        """Return minimum with units preserved."""
        result = super().min(axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)
        if axis is None and not keepdims:
            # Scalar result - wrap with units
            return self._wrap_scalar_result(result)
        elif self.has_units:
            # Array result - wrap as UnitAwareArray
            return self._wrap_result(result, self._units)
        return result

    def mean(self, axis=None, dtype=None, out=None, keepdims=False, where=True):
        """Return mean with units preserved."""
        result = super().mean(axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where)
        if axis is None and not keepdims:
            # Scalar result - wrap with units
            return self._wrap_scalar_result(result)
        elif self.has_units:
            # Array result - wrap as UnitAwareArray
            return self._wrap_result(result, self._units)
        return result

    def sum(self, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
        """Return sum with units preserved."""
        result = super().sum(
            axis=axis, dtype=dtype, out=out, keepdims=keepdims, initial=initial, where=where
        )
        if axis is None and not keepdims:
            # Scalar result - wrap with units
            return self._wrap_scalar_result(result)
        elif self.has_units:
            # Array result - wrap as UnitAwareArray
            return self._wrap_result(result, self._units)
        return result

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True):
        """Return standard deviation with units preserved."""
        if not self.has_units:
            # No units - use numpy's default
            return super().std(
                axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims, where=where
            )

        # Calculate std using unit-aware variance (avoid numpy's internal mean)
        variance = self.var(
            axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims, where=where
        )

        # Take square root of variance
        if axis is None and not keepdims:
            # Scalar result - extract magnitude, compute sqrt, re-wrap with units
            import underworld3 as uw

            if hasattr(variance, "magnitude"):
                std_value = np.sqrt(float(variance.magnitude))
            else:
                std_value = np.sqrt(float(variance))
            return uw.function.quantity(std_value, self._units)
        else:
            # Array result
            if hasattr(variance, "magnitude"):
                std_array = np.sqrt(np.asarray(variance.magnitude))
            else:
                std_array = np.sqrt(np.asarray(variance))
            return self._wrap_result(std_array, self._units)

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True):
        """Return variance with units squared."""
        if not self.has_units:
            # No units - use numpy's default
            return super().var(
                axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims, where=where
            )

        # Calculate variance manually using unit-aware mean to avoid numpy's internal mean
        # var = mean((x - mean(x))**2)

        # Get unit-aware mean
        arr_mean = self.mean(axis=axis, dtype=dtype, keepdims=True, where=where)

        # Compute deviations: (x - mean)
        # Extract magnitude from UWQuantity mean for subtraction
        if hasattr(arr_mean, "magnitude"):
            mean_value = (
                float(arr_mean.magnitude)
                if np.isscalar(arr_mean.magnitude)
                else np.asarray(arr_mean.magnitude)
            )
        else:
            mean_value = arr_mean

        # Get raw array values (without units) for arithmetic
        arr_values = np.asarray(self)

        # Subtract mean (values are in same units, so we can use plain subtraction on magnitudes)
        deviations = arr_values - mean_value

        # Square deviations (units become squared)
        squared_devs = deviations**2

        # Take mean of squared deviations
        if where is not True:
            # Handle where parameter if provided
            variance_value = np.mean(squared_devs, axis=axis, keepdims=keepdims, where=where)
        else:
            variance_value = np.mean(squared_devs, axis=axis, keepdims=keepdims)

        # Apply ddof correction
        if ddof != 0:
            if axis is None:
                n = self.size if where is True else np.count_nonzero(where)
            else:
                n = self.shape[axis] if where is True else np.count_nonzero(where, axis=axis)
            variance_value = variance_value * n / (n - ddof)

        # Wrap result with squared units
        var_units = f"({self._units})**2"
        if axis is None and not keepdims:
            # Scalar result
            import underworld3 as uw

            return uw.function.quantity(float(variance_value), var_units)
        else:
            # Array result
            return self._wrap_result(variance_value, var_units)

    # === GLOBAL REDUCTION OPERATIONS (MPI-aware) ===
    # These operations reduce across all MPI ranks

    def global_max(self, axis=None, out=None, keepdims=False):
        """
        Return maximum across all MPI ranks with units preserved.

        For scalar results (axis=None), performs MPI reduction. For array results,
        performs component-wise maximum. For tensors (ndim > 2), raises NotImplementedError.

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
        UWQuantity or ndarray
            Global maximum with units preserved

        Raises
        ------
        NotImplementedError
            If called on tensor data (ndim > 2)
        """
        import underworld3 as uw
        from mpi4py import MPI

        # Handle empty arrays (use -inf as identity for max)
        # IMPORTANT: Must preserve shape structure for MPI.Allreduce compatibility
        if self.size == 0:
            # Create appropriately shaped array of -inf
            if axis is None and not keepdims:
                local_max = -np.inf  # Scalar reduction
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
                        s
                        for i, s in enumerate(self.shape)
                        if i not in (axis if isinstance(axis, tuple) else (axis,))
                    )
                local_max = np.full(result_shape, -np.inf)
        else:
            local_max = self.max(axis=axis, out=out, keepdims=keepdims)

        # Check dimensionality for tensor rejection
        if axis is None and self.ndim > 2:
            raise NotImplementedError(
                f"global_max() not implemented for tensors (ndim={self.ndim}). "
                "Use global_max(axis=...) to reduce specific dimensions, or extract "
                "components individually."
            )

        # Scalar result - perform MPI reduction
        if axis is None and not keepdims:
            if self.has_units:
                # Extract magnitude for MPI, then re-wrap with units
                local_val = (
                    float(local_max.magnitude)
                    if hasattr(local_max, "magnitude")
                    else float(local_max)
                )
                global_val = uw.mpi.comm.allreduce(local_val, op=MPI.MAX)
                return uw.function.quantity(global_val, self._units)
            else:
                return uw.mpi.comm.allreduce(float(local_max), op=MPI.MAX)

        # Array result - component-wise reduction
        local_arr = np.asarray(
            local_max.magnitude if hasattr(local_max, "magnitude") else local_max
        )

        # For vectors, reduce component-wise
        if local_arr.ndim == 1:
            global_arr = np.array(
                [
                    uw.mpi.comm.allreduce(float(local_arr[i]), op=MPI.MAX)
                    for i in range(len(local_arr))
                ]
            )
        else:
            # For higher dimensional arrays, use allreduce directly (requires same shape on all ranks)
            global_arr = np.empty_like(local_arr)
            uw.mpi.comm.Allreduce(local_arr, global_arr, op=MPI.MAX)

        # Wrap result with units
        if self.has_units:
            return self._wrap_result(global_arr, self._units)
        return global_arr

    def global_min(self, axis=None, out=None, keepdims=False):
        """
        Return minimum across all MPI ranks with units preserved.

        For scalar results (axis=None), performs MPI reduction. For array results,
        performs component-wise minimum. For tensors (ndim > 2), raises NotImplementedError.

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
        UWQuantity or ndarray
            Global minimum with units preserved

        Raises
        ------
        NotImplementedError
            If called on tensor data (ndim > 2)
        """
        import underworld3 as uw
        from mpi4py import MPI

        # Handle empty arrays (use +inf as identity for min)
        # IMPORTANT: Must preserve shape structure for MPI.Allreduce compatibility
        if self.size == 0:
            # Create appropriately shaped array of +inf
            if axis is None and not keepdims:
                local_min = np.inf  # Scalar reduction
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
                        s
                        for i, s in enumerate(self.shape)
                        if i not in (axis if isinstance(axis, tuple) else (axis,))
                    )
                local_min = np.full(result_shape, np.inf)
        else:
            local_min = self.min(axis=axis, out=out, keepdims=keepdims)

        # Check dimensionality for tensor rejection
        if axis is None and self.ndim > 2:
            raise NotImplementedError(
                f"global_min() not implemented for tensors (ndim={self.ndim}). "
                "Use global_min(axis=...) to reduce specific dimensions, or extract "
                "components individually."
            )

        # Scalar result - perform MPI reduction
        if axis is None and not keepdims:
            if self.has_units:
                # Extract magnitude for MPI, then re-wrap with units
                local_val = (
                    float(local_min.magnitude)
                    if hasattr(local_min, "magnitude")
                    else float(local_min)
                )
                global_val = uw.mpi.comm.allreduce(local_val, op=MPI.MIN)
                return uw.function.quantity(global_val, self._units)
            else:
                return uw.mpi.comm.allreduce(float(local_min), op=MPI.MIN)

        # Array result - component-wise reduction
        local_arr = np.asarray(
            local_min.magnitude if hasattr(local_min, "magnitude") else local_min
        )

        # For vectors, reduce component-wise
        if local_arr.ndim == 1:
            global_arr = np.array(
                [
                    uw.mpi.comm.allreduce(float(local_arr[i]), op=MPI.MIN)
                    for i in range(len(local_arr))
                ]
            )
        else:
            # For higher dimensional arrays, use allreduce directly
            global_arr = np.empty_like(local_arr)
            uw.mpi.comm.Allreduce(local_arr, global_arr, op=MPI.MIN)

        # Wrap result with units
        if self.has_units:
            return self._wrap_result(global_arr, self._units)
        return global_arr

    def global_sum(self, axis=None, dtype=None, out=None, keepdims=False):
        """
        Return sum across all MPI ranks with units preserved.

        For scalar results (axis=None), performs MPI reduction. For array results,
        performs component-wise sum. For tensors (ndim > 2), raises NotImplementedError.

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
        UWQuantity or ndarray
            Global sum with units preserved

        Raises
        ------
        NotImplementedError
            If called on tensor data (ndim > 2)
        """
        import underworld3 as uw
        from mpi4py import MPI

        # Get local sum
        local_sum = self.sum(axis=axis, dtype=dtype, out=out, keepdims=keepdims)

        # Check dimensionality for tensor rejection
        if axis is None and self.ndim > 2:
            raise NotImplementedError(
                f"global_sum() not implemented for tensors (ndim={self.ndim}). "
                "Use global_sum(axis=...) to reduce specific dimensions, or extract "
                "components individually."
            )

        # Scalar result - perform MPI reduction
        if axis is None and not keepdims:
            if self.has_units:
                # Extract magnitude for MPI, then re-wrap with units
                local_val = (
                    float(local_sum.magnitude)
                    if hasattr(local_sum, "magnitude")
                    else float(local_sum)
                )
                global_val = uw.mpi.comm.allreduce(local_val, op=MPI.SUM)
                return uw.function.quantity(global_val, self._units)
            else:
                return uw.mpi.comm.allreduce(float(local_sum), op=MPI.SUM)

        # Array result - component-wise reduction
        local_arr = np.asarray(
            local_sum.magnitude if hasattr(local_sum, "magnitude") else local_sum
        )

        # For vectors, reduce component-wise
        if local_arr.ndim == 1:
            global_arr = np.array(
                [
                    uw.mpi.comm.allreduce(float(local_arr[i]), op=MPI.SUM)
                    for i in range(len(local_arr))
                ]
            )
        else:
            # For higher dimensional arrays, use allreduce directly
            global_arr = np.empty_like(local_arr)
            uw.mpi.comm.Allreduce(local_arr, global_arr, op=MPI.SUM)

        # Wrap result with units
        if self.has_units:
            return self._wrap_result(global_arr, self._units)
        return global_arr

    def global_mean(self, axis=None, dtype=None, out=None, keepdims=False):
        """
        Return mean across all MPI ranks with units preserved.

        Computes the true global mean by summing all values across ranks and
        dividing by total count. For tensors (ndim > 2), raises NotImplementedError.

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
        UWQuantity or ndarray
            Global mean with units preserved

        Raises
        ------
        NotImplementedError
            If called on tensor data (ndim > 2)
        """
        import underworld3 as uw
        from mpi4py import MPI

        # Check dimensionality for tensor rejection
        if axis is None and self.ndim > 2:
            raise NotImplementedError(
                f"global_mean() not implemented for tensors (ndim={self.ndim}). "
                "Use global_mean(axis=...) to reduce specific dimensions, or extract "
                "components individually."
            )

        # Get local sum and count
        local_sum = self.sum(axis=axis, dtype=dtype, keepdims=keepdims)

        if axis is None:
            local_count = self.size
        else:
            # Count elements along reduced axes
            if isinstance(axis, int):
                local_count = self.shape[axis]
            else:
                local_count = np.prod([self.shape[ax] for ax in axis])

        # Gather global sum and count
        global_sum = self.global_sum(axis=axis, dtype=dtype, keepdims=keepdims)
        global_count = uw.mpi.comm.allreduce(local_count, op=MPI.SUM)

        # Compute mean
        if axis is None and not keepdims:
            # Scalar result
            if self.has_units:
                mean_val = (
                    float(global_sum.magnitude) / global_count
                    if hasattr(global_sum, "magnitude")
                    else float(global_sum) / global_count
                )
                return uw.function.quantity(mean_val, self._units)
            else:
                return float(global_sum) / global_count
        else:
            # Array result
            global_arr = np.asarray(
                global_sum.magnitude if hasattr(global_sum, "magnitude") else global_sum
            )
            mean_arr = global_arr / global_count

            if self.has_units:
                return self._wrap_result(mean_arr, self._units)
            return mean_arr

    def global_var(self, axis=None, dtype=None, ddof=0, keepdims=False):
        """
        Return variance across all MPI ranks with units squared preserved.

        Uses parallel variance algorithm (Welford/Chan) for numerical stability.
        For tensors (ndim > 2), raises NotImplementedError.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis along which to operate (default: None = reduce all dimensions)
        dtype : data-type, optional
            Type of returned array
        ddof : int, optional
            Delta degrees of freedom (default: 0)
        keepdims : bool, optional
            Keep reduced dimensions as size 1 (default: False)

        Returns
        -------
        UWQuantity or ndarray
            Global variance with units squared

        Raises
        ------
        NotImplementedError
            If called on tensor data (ndim > 2)
        """
        import underworld3 as uw
        from mpi4py import MPI

        # Check dimensionality for tensor rejection
        if axis is None and self.ndim > 2:
            raise NotImplementedError(
                f"global_var() not implemented for tensors (ndim={self.ndim}). "
                "Use global_var(axis=...) to reduce specific dimensions, or extract "
                "components individually."
            )

        # Get local statistics
        local_mean = self.mean(axis=axis, dtype=dtype, keepdims=True)
        local_arr = np.asarray(self)

        # Extract magnitude for calculations
        if hasattr(local_mean, "magnitude"):
            mean_val = np.asarray(local_mean.magnitude)
        else:
            mean_val = np.asarray(local_mean)

        # Compute local sum of squared deviations
        deviations = local_arr - mean_val
        local_sq_dev = np.sum(deviations**2, axis=axis, keepdims=keepdims)
        local_sum = np.sum(local_arr, axis=axis, keepdims=keepdims)

        if axis is None:
            local_count = self.size
        else:
            if isinstance(axis, int):
                local_count = self.shape[axis]
            else:
                local_count = np.prod([self.shape[ax] for ax in axis])

        # Global reduce
        global_sq_dev = uw.mpi.comm.allreduce(
            float(local_sq_dev) if np.isscalar(local_sq_dev) else local_sq_dev, op=MPI.SUM
        )
        global_sum = uw.mpi.comm.allreduce(
            float(local_sum) if np.isscalar(local_sum) else local_sum, op=MPI.SUM
        )
        global_count = uw.mpi.comm.allreduce(local_count, op=MPI.SUM)

        # Compute global variance using parallel algorithm
        # var = (sum_sq_dev + sum^2/n_local - 2*sum*mean_local) / (n_global - ddof)
        # Simplified: var = sum_sq_dev / (n_global - ddof)
        # This assumes we're computing variance from scratch

        global_mean = global_sum / global_count

        # Better approach: use two-pass algorithm for numerical stability
        # We already have local squared deviations from local means
        # Need to correct for difference between local and global means

        # Correction term for difference between local and global means
        local_mean_arr = np.asarray(self.mean(axis=axis, keepdims=True))
        correction = local_count * (local_mean_arr - global_mean) ** 2

        global_correction = uw.mpi.comm.allreduce(
            float(correction) if np.isscalar(correction) else correction, op=MPI.SUM
        )

        # Total variance
        total_sq_dev = global_sq_dev + global_correction
        global_variance = total_sq_dev / (global_count - ddof)

        # Wrap result with squared units
        var_units = f"({self._units})**2" if self.has_units else None

        if axis is None and not keepdims:
            # Scalar result
            if self.has_units:
                # Use item() for numpy scalars to avoid deprecation warning
                variance_scalar = (
                    global_variance.item()
                    if hasattr(global_variance, "item")
                    else float(global_variance)
                )
                return uw.function.quantity(variance_scalar, var_units)
            return (
                global_variance.item()
                if hasattr(global_variance, "item")
                else float(global_variance)
            )
        else:
            # Array result
            if self.has_units:
                return self._wrap_result(global_variance, var_units)
            return global_variance

    def global_std(self, axis=None, dtype=None, ddof=0, keepdims=False):
        """
        Return standard deviation across all MPI ranks with units preserved.

        Computed as square root of global variance. For tensors (ndim > 2),
        raises NotImplementedError.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis along which to operate (default: None = reduce all dimensions)
        dtype : data-type, optional
            Type of returned array
        ddof : int, optional
            Delta degrees of freedom (default: 0)
        keepdims : bool, optional
            Keep reduced dimensions as size 1 (default: False)

        Returns
        -------
        UWQuantity or ndarray
            Global standard deviation with units preserved

        Raises
        ------
        NotImplementedError
            If called on tensor data (ndim > 2)
        """
        import underworld3 as uw

        # Get global variance
        global_variance = self.global_var(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)

        # Take square root
        if axis is None and not keepdims:
            # Scalar result
            if self.has_units:
                std_val = np.sqrt(
                    float(global_variance.magnitude)
                    if hasattr(global_variance, "magnitude")
                    else float(global_variance)
                )
                return uw.function.quantity(std_val, self._units)
            return np.sqrt(float(global_variance))
        else:
            # Array result
            var_arr = np.asarray(
                global_variance.magnitude
                if hasattr(global_variance, "magnitude")
                else global_variance
            )
            std_arr = np.sqrt(var_arr)

            if self.has_units:
                return self._wrap_result(std_arr, self._units)
            return std_arr

    def global_norm(self, ord=None):
        """
        Return norm across all MPI ranks.

        For scalars (ndim=1), computes sqrt(sum of squares). For vectors,
        computes vector norm. For tensors (ndim > 2), raises NotImplementedError.

        Parameters
        ----------
        ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
            Order of the norm (default: None = 2-norm)

        Returns
        -------
        UWQuantity or float
            Global norm with units preserved

        Raises
        ------
        NotImplementedError
            If called on tensor data (ndim > 2)
        """
        import underworld3 as uw
        from mpi4py import MPI

        # Check dimensionality for tensor rejection
        if self.ndim > 2:
            raise NotImplementedError(
                f"global_norm() not implemented for tensors (ndim={self.ndim}). "
                "Extract components individually or use global_norm() on slices."
            )

        # Default to 2-norm
        if ord is None or ord == 2:
            # Compute local sum of squares
            local_arr = np.asarray(self)
            local_sq_sum = np.sum(local_arr**2)

            # Global sum of squares
            global_sq_sum = uw.mpi.comm.allreduce(float(local_sq_sum), op=MPI.SUM)

            # Compute norm
            norm_val = np.sqrt(global_sq_sum)

            if self.has_units:
                return uw.function.quantity(norm_val, self._units)
            return norm_val
        else:
            raise NotImplementedError(
                f"global_norm() only supports ord=None or ord=2 (2-norm), got ord={ord}"
            )

    def global_size(self):
        """
        Return total number of elements across all MPI ranks.

        Useful for computing global statistics that require total element count,
        such as RMS or normalized quantities.

        Returns
        -------
        int
            Total number of elements summed across all MPI ranks

        Examples
        --------
        >>> coords = mesh.X.coords  # Shape: (N_local, 2)
        >>> total_points = coords.global_size()  # Sum of N_local across all ranks
        >>> rms = coords.global_norm() / np.sqrt(total_points)
        """
        import underworld3 as uw
        from mpi4py import MPI

        # Get local size
        local_size = self.size

        # Sum across all ranks
        global_total = uw.mpi.comm.allreduce(local_size, op=MPI.SUM)

        return global_total

    def global_rms(self):
        """
        Return root mean square across all MPI ranks with units preserved.

        Computes RMS = sqrt(sum of squares / total count) across all ranks.
        For tensors (ndim > 2), raises NotImplementedError.

        The RMS is computed as:
        RMS = global_norm() / sqrt(global_size())

        Returns
        -------
        UWQuantity or float
            Global RMS with units preserved

        Raises
        ------
        NotImplementedError
            If called on tensor data (ndim > 2)

        Examples
        --------
        >>> coords = mesh.X.coords  # UnitAwareArray with units="km"
        >>> rms_coord = coords.global_rms()  # Returns UWQuantity in km
        >>> print(f"RMS coordinate: {rms_coord}")
        """
        import underworld3 as uw

        # Check dimensionality for tensor rejection
        if self.ndim > 2:
            raise NotImplementedError(
                f"global_rms() not implemented for tensors (ndim={self.ndim}). "
                "Extract components individually or use global_rms() on slices."
            )

        # Get global norm and size
        norm = self.global_norm()
        size = self.global_size()

        # Compute RMS
        rms_val = float(norm.magnitude if hasattr(norm, "magnitude") else norm) / np.sqrt(size)

        # Return with units preserved
        if self.has_units:
            return uw.function.quantity(rms_val, self._units)
        return rms_val

    # Override arithmetic operations for unit checking
    def __add__(self, other):
        """Addition with unit compatibility checking."""
        compatible, converted_other, result_units = self._check_unit_compatibility(other, "add")

        if compatible:
            result = super().__add__(converted_other)
            return self._wrap_result(result, result_units)
        else:
            return NotImplemented

    def __radd__(self, other):
        """Right addition with unit compatibility checking."""
        compatible, converted_other, result_units = self._check_unit_compatibility(other, "add")

        if compatible:
            result = super().__radd__(converted_other)
            return self._wrap_result(result, result_units)
        else:
            return NotImplemented

    def __sub__(self, other):
        """Subtraction with unit compatibility checking."""
        compatible, converted_other, result_units = self._check_unit_compatibility(
            other, "subtract"
        )

        if compatible:
            result = super().__sub__(converted_other)
            return self._wrap_result(result, result_units)
        else:
            return NotImplemented

    def __rsub__(self, other):
        """Right subtraction with unit compatibility checking."""
        compatible, converted_other, result_units = self._check_unit_compatibility(
            other, "subtract"
        )

        if compatible:
            result = super().__rsub__(converted_other)
            return self._wrap_result(result, result_units)
        else:
            return NotImplemented

    def __mul__(self, other):
        """Multiplication with unit handling."""
        if self._unit_checking and not np.isscalar(other):
            compatible, converted_other, result_units = self._check_unit_compatibility(
                other, "multiply"
            )
        else:
            # Scalar multiplication preserves units
            converted_other = other
            result_units = self._units

        result = super().__mul__(converted_other)
        return self._wrap_result(result, result_units)

    def __rmul__(self, other):
        """Right multiplication with unit handling."""
        if self._unit_checking and not np.isscalar(other):
            compatible, converted_other, result_units = self._check_unit_compatibility(
                other, "multiply"
            )
        else:
            # Scalar multiplication preserves units
            converted_other = other
            result_units = self._units

        result = super().__rmul__(converted_other)
        return self._wrap_result(result, result_units)

    def __truediv__(self, other):
        """Division with unit handling."""
        if self._unit_checking and not np.isscalar(other):
            compatible, converted_other, result_units = self._check_unit_compatibility(
                other, "divide"
            )
        else:
            # Scalar division preserves units
            converted_other = other
            result_units = self._units

        result = super().__truediv__(converted_other)
        return self._wrap_result(result, result_units)

    def __rtruediv__(self, other):
        """Right division with unit handling."""
        if self._unit_checking and not np.isscalar(other):
            compatible, converted_other, result_units = self._check_unit_compatibility(
                other, "divide"
            )
        else:
            # Scalar division - units are inverted
            converted_other = other
            result_units = f"1/({self._units})" if self._units else None

        result = super().__rtruediv__(converted_other)
        return self._wrap_result(result, result_units)

    def __repr__(self):
        """String representation including units."""
        base_repr = super().__repr__()

        if self.has_units:
            # Insert units info before the closing parenthesis of dtype info
            if "dtype=" in base_repr:
                # Find dtype info and insert units before it
                dtype_pos = base_repr.rfind("dtype=")
                if dtype_pos > 0:
                    # Look for comma before dtype
                    comma_pos = base_repr.rfind(",", 0, dtype_pos)
                    if comma_pos > 0:
                        return (
                            base_repr[:comma_pos]
                            + f", units='{self._units}', "
                            + base_repr[comma_pos + 2 :]
                        )  # +2 to skip ", "
                    else:
                        # No comma found, insert at start of dtype
                        return (
                            base_repr[:dtype_pos]
                            + f"units='{self._units}', "
                            + base_repr[dtype_pos:]
                        )

            # Fallback - append units at end
            return base_repr.rstrip(")") + f", units='{self._units}')"

        return base_repr

    def __str__(self):
        """String representation for printing."""
        base_str = super().__str__()
        if self.has_units:
            return f"{base_str} [{self._units}]"
        return base_str

    def copy(self, order="C"):
        """Return a copy of the array with preserved units."""
        copied_array = super().copy(order=order)
        return UnitAwareArray(
            copied_array,
            units=self._units,
            unit_checking=self._unit_checking,
            auto_convert=self._auto_convert,
        )

    def astype(self, dtype, order="K", casting="unsafe", subok=True, copy=True):
        """Convert array type while preserving units."""
        converted_array = super().astype(dtype, order, casting, subok, copy)

        if subok and isinstance(converted_array, np.ndarray):
            return UnitAwareArray(
                converted_array,
                units=self._units,
                unit_checking=self._unit_checking,
                auto_convert=self._auto_convert,
            )

        return converted_array

    def view(self, dtype=None, type=None):
        """Return a view of the array with preserved units."""
        view_array = super().view(dtype, type)

        if type is None or type is UnitAwareArray:
            if isinstance(view_array, UnitAwareArray):
                # Units should already be copied via __array_finalize__
                return view_array
            else:
                # Create UnitAwareArray view
                return UnitAwareArray(
                    view_array,
                    units=self._units,
                    unit_checking=self._unit_checking,
                    auto_convert=self._auto_convert,
                )

        return view_array

    def reshape(self, *shape, order="C"):
        """Return a reshaped array with preserved units."""
        reshaped_array = super().reshape(*shape, order=order)
        return UnitAwareArray(
            reshaped_array,
            units=self._units,
            unit_checking=self._unit_checking,
            auto_convert=self._auto_convert,
        )

    def flatten(self, order="C"):
        """Return a flattened array with preserved units."""
        flattened_array = super().flatten(order)
        return UnitAwareArray(
            flattened_array,
            units=self._units,
            unit_checking=self._unit_checking,
            auto_convert=self._auto_convert,
        )

    def squeeze(self, axis=None):
        """Return a squeezed array with preserved units."""
        squeezed_array = super().squeeze(axis)
        return UnitAwareArray(
            squeezed_array,
            units=self._units,
            unit_checking=self._unit_checking,
            auto_convert=self._auto_convert,
        )

    def transpose(self, *axes):
        """Return a transposed array with preserved units."""
        transposed_array = super().transpose(*axes)
        return UnitAwareArray(
            transposed_array,
            units=self._units,
            unit_checking=self._unit_checking,
            auto_convert=self._auto_convert,
        )

    # === NUMPY FUNCTION INTEGRATION ===

    def __array_function__(self, func, types, args, kwargs):
        """
        Intercept numpy functions to preserve units.

        This method is part of NumPy's __array_function__ protocol (NumPy 1.17+).
        It allows UnitAwareArray to control how numpy functions behave when called
        with UnitAwareArray instances.

        Supported Functions:
        -------------------
        - np.cross(): Cross product with unit multiplication
        - np.dot(): Dot product with unit multiplication
        - np.concatenate(): Concatenation with unit compatibility checking
        - np.stack(), np.vstack(), np.hstack(): Stacking with unit compatibility

        Limitations:
        -----------
        - np.array() and np.asarray() do NOT use __array_function__ protocol.
          They use the lower-level __array__() method instead, which returns
          plain numpy arrays. This is by design in NumPy.

        - Scalar indexing (arr[0]) returns plain Python scalars without units.
          This is expected behavior for ndarray subclasses.

        - For internal calculations, use raw arrays (e.g., mesh._points) instead
          of unit-aware arrays to avoid unit propagation issues.

        Parameters
        ----------
        func : callable
            NumPy function being called
        types : list
            Types of all arguments
        args : tuple
            Positional arguments to func
        kwargs : dict
            Keyword arguments to func

        Returns
        -------
        result
            Result with appropriate unit handling, or NotImplemented if the
            function is not handled (falls back to default numpy behavior)
        """
        # Supported numpy functions with unit handling
        HANDLED_FUNCTIONS = {}

        def implements(numpy_function):
            """Register an __array_function__ implementation for numpy functions."""

            def decorator(func_impl):
                HANDLED_FUNCTIONS[numpy_function] = func_impl
                return func_impl

            return decorator

        # Register handlers for common numpy functions
        @implements(np.array)
        def array_impl(arr, *args, **kwargs):
            """Preserve units when creating arrays from UnitAwareArray."""
            # Get units from the source array
            if hasattr(arr, "_units"):
                result_units = arr._units
                unit_checking = arr._unit_checking
                auto_convert = arr._auto_convert
            else:
                result_units = None
                unit_checking = True
                auto_convert = True

            # Create the array using numpy's default behavior
            result = np.asarray(arr, *args, **kwargs)

            # Wrap with units if present
            if result_units is not None:
                return UnitAwareArray(
                    result,
                    units=result_units,
                    unit_checking=unit_checking,
                    auto_convert=auto_convert,
                )
            return result

        @implements(np.cross)
        def cross_impl(a, b, *args, **kwargs):
            """Handle cross product with unit multiplication."""
            # Extract units
            a_units = getattr(a, "_units", None)
            b_units = getattr(b, "_units", None)

            # Compute cross product using numpy's default behavior
            result = np.core.numeric.cross(np.asarray(a), np.asarray(b), *args, **kwargs)

            # Determine result units
            if a_units is not None and b_units is not None:
                # Unit multiplication for cross product
                result_units = f"({a_units})*({b_units})"
            elif a_units is not None:
                result_units = a_units
            elif b_units is not None:
                result_units = b_units
            else:
                result_units = None

            # Wrap with units if present
            if result_units is not None:
                return UnitAwareArray(
                    result,
                    units=result_units,
                    unit_checking=getattr(a, "_unit_checking", True),
                    auto_convert=getattr(a, "_auto_convert", True),
                )
            return result

        @implements(np.dot)
        def dot_impl(a, b, *args, **kwargs):
            """Handle dot product with unit multiplication."""
            # Extract units
            a_units = getattr(a, "_units", None)
            b_units = getattr(b, "_units", None)

            # Compute dot product
            result = np.core.multiarray.dot(np.asarray(a), np.asarray(b), *args, **kwargs)

            # Determine result units
            if a_units is not None and b_units is not None:
                # Unit multiplication for dot product
                result_units = f"({a_units})*({b_units})"
            elif a_units is not None:
                result_units = a_units
            elif b_units is not None:
                result_units = b_units
            else:
                result_units = None

            # Wrap with units if present and result is array
            if result_units is not None and not np.isscalar(result):
                return UnitAwareArray(
                    result,
                    units=result_units,
                    unit_checking=getattr(a, "_unit_checking", True),
                    auto_convert=getattr(a, "_auto_convert", True),
                )
            return result

        @implements(np.concatenate)
        def concatenate_impl(arrays, *args, **kwargs):
            """Concatenate arrays with unit compatibility checking."""
            # Check that all arrays have compatible units
            units_list = [getattr(arr, "_units", None) for arr in arrays]

            # Get first non-None units as reference
            ref_units = None
            for units in units_list:
                if units is not None:
                    ref_units = units
                    break

            # Check compatibility
            if ref_units is not None:
                for units in units_list:
                    if units is not None and units != ref_units:
                        raise ValueError(
                            f"Cannot concatenate arrays with incompatible units: "
                            f"'{ref_units}' and '{units}'"
                        )

            # Perform concatenation
            result = np.core.multiarray.concatenate(
                [np.asarray(arr) for arr in arrays], *args, **kwargs
            )

            # Wrap with units if present
            if ref_units is not None:
                return UnitAwareArray(
                    result,
                    units=ref_units,
                    unit_checking=getattr(arrays[0], "_unit_checking", True),
                    auto_convert=getattr(arrays[0], "_auto_convert", True),
                )
            return result

        @implements(np.stack)
        def stack_impl(arrays, *args, **kwargs):
            """Stack arrays with unit compatibility checking."""
            # Use same logic as concatenate
            return concatenate_impl(arrays, *args, **kwargs)

        @implements(np.vstack)
        def vstack_impl(arrays, *args, **kwargs):
            """Vertically stack arrays with unit compatibility checking."""
            return concatenate_impl(arrays, *args, **kwargs)

        @implements(np.hstack)
        def hstack_impl(arrays, *args, **kwargs):
            """Horizontally stack arrays with unit compatibility checking."""
            return concatenate_impl(arrays, *args, **kwargs)

        @implements(np.array_equal)
        def array_equal_impl(a1, a2, *args, **kwargs):
            """Compare arrays for equality, ignoring units."""
            # Convert to plain numpy arrays and compare
            return np.core.numeric.array_equal(np.asarray(a1), np.asarray(a2), *args, **kwargs)

        @implements(np.allclose)
        def allclose_impl(a, b, *args, **kwargs):
            """Check if arrays are close, ignoring units."""
            # Convert to plain numpy arrays and compare
            return np.core.numeric.allclose(np.asarray(a), np.asarray(b), *args, **kwargs)

        # Look up the handler
        if func not in HANDLED_FUNCTIONS:
            # Function not handled - use default numpy behavior
            return NotImplemented

        # Call the handler
        return HANDLED_FUNCTIONS[func](*args, **kwargs)


# Convenience functions for creating unit-aware arrays
def create_unit_aware_array(data, units=None, **kwargs):
    """
    Convenience function to create a UnitAwareArray.

    Parameters
    ----------
    data : array-like
        Input data
    units : str, optional
        Units for the array
    **kwargs : dict
        Additional arguments passed to UnitAwareArray

    Returns
    -------
    UnitAwareArray
        Array with unit tracking
    """
    return UnitAwareArray(data, units=units, **kwargs)


def zeros_with_units(shape, units, dtype=float, **kwargs):
    """
    Create a zero-filled UnitAwareArray with specified units.

    Parameters
    ----------
    shape : tuple
        Shape of the array
    units : str
        Units for the array
    dtype : data-type, optional
        Data type of the array
    **kwargs : dict
        Additional arguments passed to UnitAwareArray

    Returns
    -------
    UnitAwareArray
        Zero-filled array with units
    """
    data = np.zeros(shape, dtype=dtype)
    return UnitAwareArray(data, units=units, **kwargs)


def ones_with_units(shape, units, dtype=float, **kwargs):
    """
    Create a ones-filled UnitAwareArray with specified units.

    Parameters
    ----------
    shape : tuple
        Shape of the array
    units : str
        Units for the array
    dtype : data-type, optional
        Data type of the array
    **kwargs : dict
        Additional arguments passed to UnitAwareArray

    Returns
    -------
    UnitAwareArray
        Ones-filled array with units
    """
    data = np.ones(shape, dtype=dtype)
    return UnitAwareArray(data, units=units, **kwargs)


def full_with_units(shape, fill_value, units, dtype=None, **kwargs):
    """
    Create a UnitAwareArray filled with a specified value and units.

    Parameters
    ----------
    shape : tuple
        Shape of the array
    fill_value : scalar
        Fill value for the array
    units : str
        Units for the array
    dtype : data-type, optional
        Data type of the array
    **kwargs : dict
        Additional arguments passed to UnitAwareArray

    Returns
    -------
    UnitAwareArray
        Filled array with units
    """
    data = np.full(shape, fill_value, dtype=dtype)
    return UnitAwareArray(data, units=units, **kwargs)


# Test and demonstration function
def test_unit_aware_array():
    """Test the UnitAwareArray functionality."""
    print("Testing UnitAwareArray...")

    try:
        # Basic creation
        length = UnitAwareArray([1, 2, 3], units="m")
        time = UnitAwareArray([1, 2, 3], units="s")

        print(f"✓ Created length array: {length}")
        print(f"✓ Created time array: {time}")

        # Unit-preserving operations
        doubled_length = length * 2
        print(f"✓ Scalar multiplication: {doubled_length}")
        assert doubled_length.units == "m"

        # Unit compatibility checking
        try:
            total = length + time  # Should fail
            assert False, "Should have raised ValueError"
        except ValueError as e:
            print(f"✓ Unit compatibility checking: {e}")

        # Same units addition
        more_length = UnitAwareArray([4, 5, 6], units="m")
        total_length = length + more_length
        print(f"✓ Same units addition: {total_length}")
        assert total_length.units == "m"

        # Unit conversion
        length_mm = length.to_units("mm")
        print(f"✓ Unit conversion: {length_mm}")

        # Callback functionality preserved
        def on_change(array, info):
            print(f"📢 Callback: {info['operation']} on array with units {array.units}")

        length.set_callback(on_change)
        length[0] = 10  # Should trigger callback

        print("✓ All UnitAwareArray tests passed!")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_unit_aware_array()
