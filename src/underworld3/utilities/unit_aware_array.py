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
    get_units,
    detect_quantity_units,
    convert_quantity_units,
    add_units,
    convert_array_units,
)


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

    def __new__(cls, input_array=None, units=None, owner=None, callback=None,
                unit_checking=True, auto_convert=True, **kwargs):
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

        return obj

    def __array_finalize__(self, obj):
        """Called whenever the system allocates a new array from this template."""
        # Call parent finalize first
        super().__array_finalize__(obj)

        if obj is None:
            return

        # Copy unit information from parent array
        self._units = getattr(obj, '_units', None)
        self._unit_checking = getattr(obj, '_unit_checking', True)
        self._auto_convert = getattr(obj, '_auto_convert', True)
        self._original_units = getattr(obj, '_original_units', None)

    def _set_units(self, units):
        """
        Internal method to set units for this array.

        Parameters
        ----------
        units : str or UWQuantity
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
            self._units = get_units(units)
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

    def to_units(self, target_units):
        """
        Convert this array to different units.

        Parameters
        ----------
        target_units : str
            Target units to convert to

        Returns
        -------
        UnitAwareArray
            New array with converted values and target units
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
                auto_convert=self._auto_convert
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

        # Check if other has units
        if hasattr(other, 'units'):
            other_units = other.units
        elif hasattr(other, '_units'):
            other_units = other._units
        elif has_units(other):
            other_units = get_units(other)
        else:
            other_units = None

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
                if hasattr(other, 'to_units'):
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

        # Handle multiplication/division - combine units
        if operation in ["multiply", "divide"]:
            if operation == "multiply":
                # Unit multiplication - would need Pint integration for proper handling
                result_units = f"({self._units})*({other_units})"
            else:  # divide
                result_units = f"({self._units})/({other_units})"
            return True, other, result_units

        # Incompatible units for addition/subtraction
        raise ValueError(
            f"Cannot {operation} arrays with incompatible units: "
            f"'{self._units}' and '{other_units}'"
        )

    def _wrap_result(self, result, units=None):
        """
        Wrap operation result as UnitAwareArray with appropriate units.

        Parameters
        ----------
        result : array-like
            Result of operation
        units : str, optional
            Units for the result (defaults to self._units)

        Returns
        -------
        UnitAwareArray or scalar
            Wrapped result with units
        """
        if np.isscalar(result):
            # Scalar results don't need unit tracking
            return result

        # Preserve as UnitAwareArray with units
        return UnitAwareArray(
            result,
            units=units or self._units,
            unit_checking=self._unit_checking,
            auto_convert=self._auto_convert
        )

    # Override arithmetic operations for unit checking
    def __add__(self, other):
        """Addition with unit compatibility checking."""
        compatible, converted_other, result_units = self._check_unit_compatibility(
            other, "add"
        )

        if compatible:
            result = super().__add__(converted_other)
            return self._wrap_result(result, result_units)
        else:
            return NotImplemented

    def __radd__(self, other):
        """Right addition with unit compatibility checking."""
        compatible, converted_other, result_units = self._check_unit_compatibility(
            other, "add"
        )

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
                        return (base_repr[:comma_pos] +
                               f", units='{self._units}', " +
                               base_repr[comma_pos+2:])  # +2 to skip ", "
                    else:
                        # No comma found, insert at start of dtype
                        return (base_repr[:dtype_pos] +
                               f"units='{self._units}', " +
                               base_repr[dtype_pos:])

            # Fallback - append units at end
            return base_repr.rstrip(")") + f", units='{self._units}')"

        return base_repr

    def __str__(self):
        """String representation for printing."""
        base_str = super().__str__()
        if self.has_units:
            return f"{base_str} [{self._units}]"
        return base_str

    def copy(self, order='C'):
        """Return a copy of the array with preserved units."""
        copied_array = super().copy(order=order)
        return UnitAwareArray(
            copied_array,
            units=self._units,
            unit_checking=self._unit_checking,
            auto_convert=self._auto_convert
        )

    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        """Convert array type while preserving units."""
        converted_array = super().astype(dtype, order, casting, subok, copy)

        if subok and isinstance(converted_array, np.ndarray):
            return UnitAwareArray(
                converted_array,
                units=self._units,
                unit_checking=self._unit_checking,
                auto_convert=self._auto_convert
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
                    auto_convert=self._auto_convert
                )

        return view_array

    def reshape(self, *shape, order='C'):
        """Return a reshaped array with preserved units."""
        reshaped_array = super().reshape(*shape, order=order)
        return UnitAwareArray(
            reshaped_array,
            units=self._units,
            unit_checking=self._unit_checking,
            auto_convert=self._auto_convert
        )

    def flatten(self, order='C'):
        """Return a flattened array with preserved units."""
        flattened_array = super().flatten(order)
        return UnitAwareArray(
            flattened_array,
            units=self._units,
            unit_checking=self._unit_checking,
            auto_convert=self._auto_convert
        )

    def squeeze(self, axis=None):
        """Return a squeezed array with preserved units."""
        squeezed_array = super().squeeze(axis)
        return UnitAwareArray(
            squeezed_array,
            units=self._units,
            unit_checking=self._unit_checking,
            auto_convert=self._auto_convert
        )

    def transpose(self, *axes):
        """Return a transposed array with preserved units."""
        transposed_array = super().transpose(*axes)
        return UnitAwareArray(
            transposed_array,
            units=self._units,
            unit_checking=self._unit_checking,
            auto_convert=self._auto_convert
        )


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