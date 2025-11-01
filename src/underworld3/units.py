# underworld3/units.py
"""
High-level units and dimensional analysis utilities for Underworld3.

This module provides standalone functions for dimensional analysis that work with
arbitrary expressions, quantities, and unit-aware objects. These functions wrap
the underlying units backends (Pint/SymPy) to provide a consistent API without
exposing the backend implementation details to users.

Key functions:
- check_units_consistency() - Validate dimensional consistency of expressions
- get_dimensionality() - Extract dimensionality from quantities or expressions  
- non_dimensionalise() - Convert to non-dimensional values for solvers
- dimensionalise() - Add dimensions to non-dimensional values
- simplify_units() - Cancel and simplify unit expressions
- units_of() - Get the units of any quantity or expression
"""

import warnings
from typing import Any, Union, List, Optional, Dict
from abc import ABC, abstractmethod
import numpy as np


class UnitsError(Exception):
    """Exception raised for units-related errors."""

    pass


class DimensionalityError(UnitsError):
    """Exception raised for dimensional inconsistency errors."""

    pass


class NoUnitsError(UnitsError):
    """Exception raised when units are expected but not found."""

    pass


def _get_default_backend():
    """Get the default units backend."""
    try:
        from .utilities.units_mixin import PintBackend

        return PintBackend()
    except ImportError:
        raise ImportError("Pint is required for units support. Install with: pip install pint")


def _extract_units_info(obj):
    """
    Extract units information from various object types.

    Args:
        obj: Object that might have units (variable, quantity, expression, etc.)

    Returns:
        tuple: (has_units, units, backend) or (False, None, None)
    """
    # Check if it's a unit-aware variable
    if hasattr(obj, "has_units") and hasattr(obj, "units") and hasattr(obj, "_units_backend"):
        if obj.has_units:
            return True, obj.units, obj._units_backend
        else:
            return False, None, None

    # Check if it's a Pint quantity
    try:
        import pint

        if isinstance(obj, pint.Quantity):
            backend = _get_default_backend()
            return True, backend.get_units(obj), backend
    except ImportError:
        pass

    # Check if it's a SymPy expression containing unit-aware variables
    try:
        import sympy

        if isinstance(obj, sympy.Basic):
            # SymPy units backend removed - use Pint-native approach instead
            # First, try the improved get_units from function module which handles symbolic expressions
            try:
                from underworld3.function.unit_conversion import get_units as function_get_units

                units_result = function_get_units(obj)
                if units_result is not None:
                    backend = _get_default_backend()
                    return True, units_result, backend
            except Exception:
                pass  # Fall through to other methods

            # Second try: extract unit-aware variables from the expression
            units_from_variables = _extract_units_from_sympy_expression(obj)
            if units_from_variables is not None:
                return units_from_variables

            # Third try: for Matrix expressions, try the first element
            if isinstance(obj, sympy.Matrix) and obj.shape[0] > 0:
                first_element_units = _extract_units_info(obj[0])
                if first_element_units[0]:  # has_units
                    return first_element_units

            # Fourth try: check if the object contains physics units atoms
            try:
                import sympy.physics.units as units_module

                if hasattr(units_module, "dimensions"):
                    # Check if object contains any units
                    units_atoms = obj.atoms(sympy.physics.units.Quantity)
                    if units_atoms:
                        return True, obj, None  # Return the object itself as "units"
            except Exception:
                pass
    except ImportError:
        pass

    # No units found
    return False, None, None


def _extract_units_from_sympy_expression(expr):
    """
    Extract units from SymPy expressions containing unit-aware variables.

    This function analyzes mathematical expressions like 2*velocity to determine
    their units based on the unit-aware variables they contain.

    Args:
        expr: SymPy expression

    Returns:
        tuple: (has_units, units, backend) or None if no units found
    """
    try:
        # Import the function extraction utilities
        import underworld3.function.expressions as expr_utils

        # Extract all UW objects (function symbols) from the expression
        uw_objects = expr_utils.extract_expressions_and_functions(expr)

        # Get the default model to access registered variables
        import underworld3 as uw

        model = uw.get_default_model()

        # Map function symbols back to their variables
        unit_info_list = []
        for symbol in uw_objects:
            # Skip coordinate symbols
            if hasattr(symbol, "name") and symbol.name in ["N.x", "N.y", "N.z"]:
                continue

            # Find which variable this symbol belongs to
            for var_name, variable in model._variables.items():
                if hasattr(variable, "sym") and hasattr(variable, "has_units"):
                    # Check if this symbol matches any component of the variable
                    if hasattr(variable, "num_components"):
                        var_symbols = [variable.sym[i] for i in range(variable.num_components)]
                        if symbol in var_symbols and variable.has_units:
                            unit_info_list.append((variable.units, variable._units_backend))
                            break  # Found the variable, no need to continue

        if not unit_info_list:
            return None

        # Use the first unit-aware variable's units
        first_units, first_backend = unit_info_list[0]

        # Analyze the expression to determine result units
        result_units = _analyze_expression_units(expr, unit_info_list)

        return True, result_units, first_backend

    except Exception as e:
        # Debug: print the exception
        import warnings

        warnings.warn(f"Error in _extract_units_from_sympy_expression: {e}")
        return None


def _analyze_expression_units(expr, unit_info_list):
    """
    Analyze a SymPy expression to determine its resultant units.

    This implements basic dimensional analysis for mathematical operations:
    - Addition/subtraction: units must be the same, result has same units
    - Multiplication: units multiply
    - Division: units divide
    - Powers: units are raised to the power

    Args:
        expr: SymPy expression
        unit_info_list: List of (units, backend) tuples from variables in expression

    Returns:
        Units for the resulting expression
    """
    try:
        import sympy

        # For now, implement simple heuristics
        # TODO: Full dimensional analysis implementation

        # If expression is just multiplication by a constant, preserve units
        if isinstance(expr, sympy.Mul):
            # Check if it's a constant times a variable
            constants = []
            variables = []
            for arg in expr.args:
                if arg.is_number:
                    constants.append(arg)
                else:
                    variables.append(arg)

            if len(variables) == 1 and len(unit_info_list) == 1:
                # Simple case: constant * variable
                return unit_info_list[0][0]  # Return the variable's units

        # If expression is addition/subtraction, all terms must have same units
        if isinstance(expr, sympy.Add):
            if len(unit_info_list) >= 1:
                # All terms should have the same units for valid addition
                return unit_info_list[0][0]

        # Default: return the first variable's units
        if unit_info_list:
            return unit_info_list[0][0]

        # No units determined
        return None

    except Exception:
        # Fallback: return first variable's units if available
        if unit_info_list:
            return unit_info_list[0][0]
        return None


def check_units_consistency(*expressions) -> bool:
    """
    Check if multiple expressions have consistent units for addition/comparison.

    This function validates that all provided expressions have the same dimensionality,
    which is required for addition, subtraction, and comparison operations.

    Args:
        *expressions: Any number of expressions, quantities, or unit-aware objects

    Returns:
        bool: True if all expressions have consistent units, False otherwise

    Raises:
        DimensionalityError: If expressions have inconsistent units
        NoUnitsError: If some expressions have units and others don't

    Examples:
        >>> velocity1 = EnhancedMeshVariable("v1", mesh, 2, units="m/s")
        >>> velocity2 = EnhancedMeshVariable("v2", mesh, 2, units="km/h")
        >>> pressure = EnhancedMeshVariable("p", mesh, 1, units="Pa")

        >>> check_units_consistency(velocity1, velocity2)  # True - both velocities
        >>> check_units_consistency(velocity1, pressure)   # False - different dimensions
    """
    if len(expressions) < 2:
        return True

    # Extract units info from all expressions
    units_info = [_extract_units_info(expr) for expr in expressions]

    # Check if all have units or all don't have units
    has_units_flags = [info[0] for info in units_info]

    if not all(has_units_flags) and any(has_units_flags):
        raise NoUnitsError("Cannot mix expressions with units and without units")

    if not any(has_units_flags):
        # All are unitless - consistent
        return True

    # All have units - check dimensional consistency
    first_units = units_info[0][1]
    first_backend = units_info[0][2]

    for i, (has_units, units, backend) in enumerate(units_info[1:], 1):
        if not first_backend.check_dimensionality(
            first_backend.create_quantity(1.0, first_units), backend.create_quantity(1.0, units)
        ):
            raise DimensionalityError(
                f"Inconsistent units: expression 0 has {first_units}, "
                f"expression {i} has {units}"
            )

    return True


def get_dimensionality(expression) -> Optional[Any]:
    """
    Get the dimensionality of an expression or quantity.

    Args:
        expression: Expression, quantity, or unit-aware object

    Returns:
        Dimensionality representation (backend-specific) or None if no units

    Examples:
        >>> velocity = EnhancedMeshVariable("velocity", mesh, 2, units="m/s")
        >>> dims = get_dimensionality(velocity)
        >>> print(dims)  # [length] / [time]
    """
    has_units, units, backend = _extract_units_info(expression)

    if not has_units:
        return None

    quantity = backend.create_quantity(1.0, units)
    return backend.get_dimensionality(quantity)


def units_of(expression) -> Optional[Any]:
    """
    Get the units of an expression or quantity.

    Args:
        expression: Expression, quantity, or unit-aware object

    Returns:
        Units object or None if no units

    Examples:
        >>> velocity = EnhancedMeshVariable("velocity", mesh, 2, units="m/s")
        >>> units = units_of(velocity)
        >>> print(units)  # meter / second
    """
    has_units, units, backend = _extract_units_info(expression)
    return units if has_units else None


def non_dimensionalise(expression, model=None) -> Any:
    """
    Convert expression to non-dimensional form using model reference scales.

    This function uses dimensional analysis to compute appropriate scaling factors
    from the model's reference quantities, then divides the expression by those
    scales to produce dimensionless values. Dimensionality information is preserved
    to enable re-dimensionalization.

    Protocol-based approach works with:
    - MeshVariable/SwarmVariable (via .non_dimensional_value() method)
    - UWQuantity objects (extracts dimensionality, computes scale)
    - UnitAwareArray (extracts dimensionality from units)
    - Plain numbers (pass through unchanged)

    Args:
        expression: Expression, quantity, or unit-aware object to non-dimensionalise
        model: Model instance with reference quantities (uses default if None)

    Returns:
        Non-dimensional value(s) with preserved dimensionality metadata

    Raises:
        NoUnitsError: If expression has no units and model has reference quantities
        ValueError: If model has no reference quantities

    Examples:
        >>> # With variables (uses existing method)
        >>> velocity_var = MeshVariable("v", mesh, 2, units="m/s")
        >>> nondim_v = non_dimensionalise(velocity_var)

        >>> # With UWQuantity
        >>> velocity_qty = uw.quantity(5.0, "cm/year")
        >>> nondim_qty = non_dimensionalise(velocity_qty, model)
        >>> # Result is dimensionless but remembers it was velocity

        >>> # With plain number (no model reference quantities)
        >>> plain_value = 2.5
        >>> result = non_dimensionalise(plain_value)  # Returns 2.5
    """
    # Get model if not provided
    if model is None:
        import underworld3 as uw
        model = uw.get_default_model()

    # Protocol 1: Handle UWQuantity objects (check type first, not methods)
    from .function.quantities import UWQuantity
    if isinstance(expression, UWQuantity):
        # Extract dimensionality
        dimensionality = expression.dimensionality

        if not dimensionality:
            # Already dimensionless, just return the value
            return expression.value

        # Get appropriate scale from model
        if not hasattr(model, "_fundamental_scales"):
            # No reference quantities - return plain value
            return expression.value

        try:
            scale = model.get_scale_for_dimensionality(dimensionality)

            # Divide quantity by scale to get dimensionless value
            if hasattr(expression, '_pint_qty') and expression._pint_qty is not None:
                result_qty = expression._pint_qty / scale
                # Create dimensionless UWQuantity with preserved dimensionality
                nondim_value = float(result_qty.magnitude)
                return UWQuantity(nondim_value, units="dimensionless", dimensionality=dimensionality)
            else:
                # No Pint quantity available - just return value
                # (This case should be rare)
                return expression.value
        except ValueError as e:
            # Could not compute scale - might be missing fundamental dimension
            raise ValueError(f"Cannot non-dimensionalise: {e}")

    # Protocol 2: Handle UnitAwareArray objects (before generic method check)
    # UnitAwareArray doesn't have non_dimensional_value() method but should be supported
    try:
        from .utilities.unit_aware_array import UnitAwareArray
    except ImportError:
        UnitAwareArray = None  # Not available

    if UnitAwareArray is not None and isinstance(expression, UnitAwareArray):
        # Extract units from UnitAwareArray
        units_str = expression._units if hasattr(expression, '_units') else None

        if units_str is None or units_str == "dimensionless":
            # Already dimensionless - return as plain array
            return np.asarray(expression)

        # Get scale for this dimensionality
        if not hasattr(model, "_fundamental_scales"):
            # No reference quantities - return plain array
            return np.asarray(expression)

        try:
            # Create a quantity to get its dimensionality
            from .function.quantities import UWQuantity
            temp_qty = UWQuantity(1.0, units_str)
            dimensionality = temp_qty.dimensionality

            if not dimensionality:
                # Dimensionless
                return np.asarray(expression)

            # Get scale from model
            scale = model.get_scale_for_dimensionality(dimensionality)

            # Divide array by scale to get dimensionless values
            return np.asarray(expression) / float(scale.magnitude)

        except Exception as e:
            # Could not compute scale - return plain array
            import warnings
            warnings.warn(f"Could not non-dimensionalise UnitAwareArray: {e}", UserWarning)
            return np.asarray(expression)

    # Protocol 3: Handle objects with non_dimensional_value() method
    # This includes MeshVariable, SwarmVariable
    if hasattr(expression, 'non_dimensional_value') and callable(expression.non_dimensional_value):
        return expression.non_dimensional_value(model)

    # Protocol 4: Handle plain numbers (pass through)
    if isinstance(expression, (int, float, complex)):
        return expression

    # Protocol 5: Try to handle as Pint quantity directly
    try:
        import pint
        if isinstance(expression, pint.Quantity):
            # Extract dimensionality
            dimensionality = dict(expression.dimensionality)

            if not dimensionality:
                return expression.magnitude

            # Get scale from model
            if not hasattr(model, "_fundamental_scales"):
                return expression.magnitude

            try:
                scale = model.get_scale_for_dimensionality(dimensionality)
                result_qty = expression / scale
                # Return UWQuantity to preserve dimensionality
                return UWQuantity(float(result_qty.magnitude), units="dimensionless", dimensionality=dimensionality)
            except ValueError as e:
                raise ValueError(f"Cannot non-dimensionalise Pint quantity: {e}")
    except ImportError:
        pass

    # Protocol 6: Check for .non_dimensional_value() method (backward compatibility for Variables)
    # This comes last because UWQuantity also has this method but it doesn't work correctly for plain quantities
    if hasattr(expression, "non_dimensional_value") and hasattr(expression, "data"):
        # It's a MeshVariable or SwarmVariable with the method AND data attribute
        return expression.non_dimensional_value()

    # Could not non-dimensionalise
    raise TypeError(
        f"Cannot non-dimensionalise object of type {type(expression)}. "
        f"Must be MeshVariable, SwarmVariable, UWQuantity, UnitAwareArray, or plain number."
    )


def dimensionalise(expression, target_dimensionality=None, model=None) -> Any:
    """
    Restore dimensional form to non-dimensional values using model reference scales.

    This is the companion function to non_dimensionalise(). It multiplies dimensionless
    values by the appropriate reference scale to restore their dimensional form.

    The function can operate in two modes:
    1. **Auto mode**: Extract dimensionality from the expression itself (if preserved)
    2. **Explicit mode**: Use provided target_dimensionality

    Args:
        expression: Non-dimensional value (UWQuantity, UnitAwareArray, or plain number)
                   with preserved dimensionality metadata
        target_dimensionality: Optional dict specifying target dimensionality
                              (Pint format: e.g., {'[length]': 1, '[time]': -1} for velocity)
                              If None, uses dimensionality from expression
        model: Model instance with reference quantities (uses default if None)

    Returns:
        Dimensional quantity with appropriate units

    Raises:
        ValueError: If no dimensionality information available
        ValueError: If model has no reference quantities

    Examples:
        >>> # Auto mode - dimensionality preserved from non_dimensionalise()
        >>> velocity_qty = uw.quantity(5.0, "cm/year")
        >>> nondim_vel = non_dimensionalise(velocity_qty, model)
        >>> # nondim_vel remembers it was velocity
        >>> dimensional_vel = dimensionalise(nondim_vel, model=model)
        >>> # Result has appropriate units based on model scales

        >>> # Explicit mode - specify dimensionality
        >>> plain_value = 2.5  # dimensionless number
        >>> velocity_dimensionality = {'[length]': 1, '[time]': -1}
        >>> velocity = dimensionalise(plain_value, velocity_dimensionality, model)
        >>> # Result is 2.5 * (length_scale / time_scale)

        >>> # With arrays
        >>> nondim_array = UnitAwareArray([1.0, 2.0, 3.0],
        ...                               units="dimensionless",
        ...                               dimensionality={'[length]': 1})
        >>> dimensional_array = dimensionalise(nondim_array, model=model)
    """
    # Get model if not provided
    if model is None:
        import underworld3 as uw
        model = uw.get_default_model()

    # IDEMPOTENCY CHECK: If input already has dimensional units, return as-is
    # This prevents double-conversion (e.g., 2900 km → 2900000 km)
    from .function.quantities import UWQuantity
    from .function.unit_conversion import UnitAwareArray

    if isinstance(expression, UWQuantity):
        # Check if units are already dimensional (not "dimensionless")
        if hasattr(expression, '_units') and expression._units and expression._units != "dimensionless":
            return expression  # Already dimensional - idempotent return

    if isinstance(expression, UnitAwareArray):
        # Check if units are already dimensional
        if expression._units and expression._units != "dimensionless":
            return expression  # Already dimensional - idempotent return

    # Extract dimensionality from expression or use provided
    dimensionality = target_dimensionality

    # Try to extract from UWQuantity
    from .function.quantities import UWQuantity
    if isinstance(expression, UWQuantity) and dimensionality is None:
        dimensionality = expression.dimensionality

    # Try to extract from UnitAwareArray
    from .function.unit_conversion import UnitAwareArray
    if isinstance(expression, UnitAwareArray) and dimensionality is None:
        dimensionality = expression.dimensionality

    # Check if we have dimensionality
    if not dimensionality or dimensionality == {}:
        # No dimensionality - return as-is (already dimensional or truly dimensionless)
        if isinstance(expression, UWQuantity):
            return expression
        elif isinstance(expression, UnitAwareArray):
            return expression
        else:
            return expression

    # Get scale from model
    if not hasattr(model, "_fundamental_scales"):
        raise ValueError(
            "Cannot dimensionalise: model has no reference quantities. "
            "Use model.set_reference_quantities() first."
        )

    try:
        scale = model.get_scale_for_dimensionality(dimensionality)
    except ValueError as e:
        raise ValueError(f"Cannot compute scale for dimensionality {dimensionality}: {e}")

    # Apply scale based on expression type
    import numpy as np

    if isinstance(expression, UWQuantity):
        # Multiply dimensionless quantity by scale
        if hasattr(expression, '_pint_qty') and expression._pint_qty is not None:
            result_qty = expression._pint_qty * scale
            # Return UWQuantity with proper units
            return UWQuantity(
                float(result_qty.magnitude),
                units=str(result_qty.units),
                dimensionality=dimensionality
            )
        else:
            # Plain value - multiply by scale
            result_qty = expression.value * scale
            return UWQuantity(
                float(result_qty.magnitude),
                units=str(result_qty.units),
                dimensionality=dimensionality
            )

    elif isinstance(expression, UnitAwareArray):
        # Multiply array by scale
        result_qty = expression.view(np.ndarray) * scale
        # Return UnitAwareArray with proper units
        return UnitAwareArray(
            result_qty.magnitude,
            units=str(result_qty.units),
            dimensionality=dimensionality
        )

    elif isinstance(expression, (int, float, complex, np.ndarray)):
        # Plain number or array - multiply by scale
        result_qty = expression * scale
        if isinstance(expression, np.ndarray):
            # Return UnitAwareArray
            return UnitAwareArray(
                result_qty.magnitude,
                units=str(result_qty.units),
                dimensionality=dimensionality
            )
        else:
            # Return UWQuantity
            return UWQuantity(
                float(result_qty.magnitude),
                units=str(result_qty.units),
                dimensionality=dimensionality
            )

    else:
        raise TypeError(
            f"Cannot dimensionalise object of type {type(expression)}. "
            f"Must be UWQuantity, UnitAwareArray, or plain number/array."
        )


def simplify_units(expression) -> Any:
    """
    Simplify and cancel units in an expression.

    This function performs dimensional analysis to simplify unit expressions,
    canceling common factors and reducing to fundamental units.

    Args:
        expression: Expression with units to simplify

    Returns:
        Expression with simplified units

    Examples:
        >>> # Force per area = pressure
        >>> force_per_area = force / area  # N/m²
        >>> simplified = simplify_units(force_per_area)  # Pa
    """
    has_units, units, backend = _extract_units_info(expression)

    if not has_units:
        return expression

    # This would need backend-specific implementation
    # For now, return the original expression
    warnings.warn("Unit simplification not fully implemented")
    return expression


def create_quantity(value, units: Union[str, Any], backend: Optional[str] = None) -> Any:
    """
    Create a dimensional quantity from a value and units.

    Args:
        value: Numeric value or array
        units: Units specification (string or units object)
        backend: Backend to use ('pint' or 'sympy'), defaults to 'pint'

    Returns:
        Dimensional quantity

    Examples:
        >>> velocity_qty = create_quantity([1.0, 2.0], "m/s")
        >>> pressure_qty = create_quantity(101325, "Pa")
    """
    if backend is None:
        units_backend = _get_default_backend()
    else:
        if backend.lower() == "pint":
            from .utilities.units_mixin import PintBackend

            units_backend = PintBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}. Only 'pint' is supported.")

    return units_backend.create_quantity(value, units)


def convert_units(quantity, target_units: Union[str, Any]) -> Any:
    """
    Convert quantity to different units.

    Args:
        quantity: Quantity to convert
        target_units: Target units for conversion

    Returns:
        Quantity converted to target units

    Raises:
        DimensionalityError: If units are not compatible for conversion
        NoUnitsError: If quantity has no units

    Examples:
        >>> velocity_ms = create_quantity(10, "m/s")
        >>> velocity_kmh = convert_units(velocity_ms, "km/h")
        >>> print(velocity_kmh)  # 36.0 kilometer / hour
    """
    has_units, units, backend = _extract_units_info(quantity)

    if not has_units:
        raise NoUnitsError("Cannot convert quantity without units")

    # Check compatibility
    target_quantity = backend.create_quantity(1.0, target_units)
    if not backend.check_dimensionality(quantity, target_quantity):
        raise DimensionalityError(f"Cannot convert {units} to {target_units}")

    # This would need backend-specific conversion implementation
    warnings.warn("Unit conversion not fully implemented")
    return quantity


def get_scaling_coefficients() -> Dict[str, Any]:
    """
    Get the current scaling coefficients used for non-dimensionalisation.

    Returns:
        Dictionary of scaling coefficients for fundamental dimensions

    Examples:
        >>> coeffs = get_scaling_coefficients()
        >>> print(coeffs['[length]'])  # 1.0 meter
        >>> print(coeffs['[time]'])    # 1.0 year
    """
    # Use the existing scaling module
    import underworld3.scaling as scaling

    return scaling.get_coefficients()


def set_scaling_coefficients(coefficients: Dict[str, Any]) -> None:
    """
    Set custom scaling coefficients for non-dimensionalisation.

    Args:
        coefficients: Dictionary mapping dimension names to scaling quantities

    Examples:
        >>> coeffs = get_scaling_coefficients()
        >>> coeffs['[length]'] = create_quantity(1000, "km")  # Geological scale
        >>> coeffs['[time]'] = create_quantity(1e6, "year")   # Geological time
        >>> set_scaling_coefficients(coeffs)
    """
    # This would need to update the scaling module's global coefficients
    warnings.warn("Setting custom scaling coefficients not implemented")


# Convenience functions for common operations
def is_dimensionless(expression) -> bool:
    """Check if expression is dimensionless."""
    return get_dimensionality(expression) is None


def has_units(expression) -> bool:
    """Check if expression has units."""
    return _extract_units_info(expression)[0]


def same_units(expr1, expr2) -> bool:
    """Check if two expressions have the same units."""
    try:
        return check_units_consistency(expr1, expr2)
    except (DimensionalityError, NoUnitsError):
        return False


def is_velocity(expression) -> bool:
    """Check if expression has velocity dimensions [length]/[time]."""
    dims = get_dimensionality(expression)
    if dims is None:
        return False
    # This would need backend-specific dimensionality checking
    # For now, check string representation
    return "[length]" in str(dims) and "[time]" in str(dims)


def is_pressure(expression) -> bool:
    """Check if expression has pressure dimensions [mass]/([length]⋅[time]²)."""
    dims = get_dimensionality(expression)
    if dims is None:
        return False
    # This would need backend-specific dimensionality checking
    return "[mass]" in str(dims) and "[length]" in str(dims) and "[time]" in str(dims)


# High-level validation functions
def validate_expression_units(expression, expected_units: Union[str, Any]) -> bool:
    """
    Validate that an expression has the expected units.

    Args:
        expression: Expression to validate
        expected_units: Expected units (string or units object)

    Returns:
        True if units match, False otherwise

    Raises:
        NoUnitsError: If expression has no units but units are expected
    """
    has_units_flag, actual_units, backend = _extract_units_info(expression)

    if not has_units_flag:
        raise NoUnitsError("Expression has no units but units were expected")

    expected_quantity = backend.create_quantity(1.0, expected_units)
    actual_quantity = backend.create_quantity(1.0, actual_units)

    return backend.check_dimensionality(actual_quantity, expected_quantity)


def assert_dimensionality(
    value,
    expected_dimensionality: str,
    value_name: str = "value",
    allow_dimensionless: bool = True,
    strict: bool = False
) -> None:
    """
    Assert that a value has the expected dimensionality.

    This is a general type-safety gatekeeper that validates physical dimensionality
    at various points in the code. Complements get_dimensionality() by providing
    enforcement rather than just inspection.

    Args:
        value: The value to validate (quantity, expression, variable, array, etc.)
        expected_dimensionality: Expected dimensionality as a string
            - Specific dimensionality: "[length]", "[length]/[time]", "[mass]*[length]/[time]**2"
            - Dimensionless: "dimensionless" or ""
        value_name: Name of the value being validated (for error messages)
        allow_dimensionless: If True, accept dimensionless values even when dimensional
            expected (default: True, as dimensionless is valid for solver operations)
        strict: If True, raise error on dimensionless when dimensional expected
            (default: False, overrides allow_dimensionless)

    Raises:
        DimensionalityError: If dimensionality doesn't match expected
        NoUnitsError: If strict=True and value is dimensionless when dimensional expected

    Examples:
        >>> # Validate coordinates have length dimensionality
        >>> coords = mesh.X.coords
        >>> assert_dimensionality(coords, "[length]", "coordinates")

        >>> # Validate velocity has correct dimensionality
        >>> velocity = uw.discretisation.MeshVariable("v", mesh, 2, units="m/s")
        >>> assert_dimensionality(velocity, "[length]/[time]", "velocity")

        >>> # Validate pressure
        >>> pressure = uw.quantity(1e5, "Pa")
        >>> assert_dimensionality(pressure, "[mass]/([length]*[time]**2)", "pressure")

        >>> # Accept dimensionless when dimensional expected (default)
        >>> dimensionless_coords = np.array([[0, 1], [1, 1]])
        >>> assert_dimensionality(dimensionless_coords, "[length]", "coords")  # OK

        >>> # Strict mode: reject dimensionless when dimensional expected
        >>> assert_dimensionality(
        ...     dimensionless_coords, "[length]", "coords", strict=True
        ... )  # Raises NoUnitsError
    """
    # Check if value has units
    has_units_flag, actual_units, backend = _extract_units_info(value)

    # Handle dimensionless values
    if not has_units_flag:
        # Dimensionless value encountered
        if expected_dimensionality in ("dimensionless", ""):
            # Expected dimensionless, got dimensionless - OK
            return

        # Expected dimensional, got dimensionless
        if strict:
            # Strict mode: reject dimensionless
            raise NoUnitsError(
                f"{value_name} is dimensionless but expected dimensionality '{expected_dimensionality}'. "
                f"In strict mode, dimensionless values are not accepted when dimensional expected."
            )
        elif allow_dimensionless:
            # Permissive mode: accept dimensionless (valid for solver operations)
            return
        else:
            # allow_dimensionless=False: reject
            raise NoUnitsError(
                f"{value_name} is dimensionless but expected dimensionality '{expected_dimensionality}'."
            )

    # Value has units - check dimensionality
    if expected_dimensionality in ("dimensionless", ""):
        # Expected dimensionless, got dimensional - error
        raise DimensionalityError(
            f"{value_name} has units '{actual_units}' but expected dimensionless value."
        )

    # Create reference quantity for expected dimensionality
    try:
        # Map common dimensionality strings to example units
        dimensionality_to_units = {
            "[length]": "meter",
            "[time]": "second",
            "[mass]": "kilogram",
            "[temperature]": "kelvin",
            "[length]/[time]": "meter/second",
            "[mass]/([length]*[time]**2)": "pascal",  # pressure
            "[mass]*[length]**2/[time]**2": "joule",  # energy
            "[mass]*[length]**2/[time]**3": "watt",   # power
        }

        # Try to get example units for this dimensionality
        example_units = dimensionality_to_units.get(expected_dimensionality, None)

        if example_units is None:
            # Try to parse the dimensionality string directly as units
            # This handles custom cases like "[mass]*[length]/[time]**2"
            # We'll just use the backend's create_quantity which should handle it
            warnings.warn(
                f"Unknown dimensionality pattern '{expected_dimensionality}', attempting direct parse",
                UserWarning
            )
            example_units = expected_dimensionality.replace("[", "").replace("]", "")

        expected_quantity = backend.create_quantity(1.0, example_units)
        actual_quantity = backend.create_quantity(1.0, actual_units)

        # Check dimensionality match
        if not backend.check_dimensionality(actual_quantity, expected_quantity):
            raise DimensionalityError(
                f"{value_name} has dimensionality '{actual_units}' but expected '{expected_dimensionality}'. "
                f"Got units '{actual_units}' which don't match the expected physical dimension."
            )

    except Exception as e:
        if isinstance(e, (DimensionalityError, NoUnitsError)):
            raise
        # If we can't check dimensionality, warn but don't fail
        warnings.warn(
            f"Could not validate dimensionality for {value_name} with units '{actual_units}': {e}",
            UserWarning
        )


def validate_coordinates_dimensionality(coords) -> None:
    """
    Validate that coordinates have length dimensionality [L].

    This function checks that the provided coordinates are either dimensionless
    (which is valid for solver operations) or have length dimensionality. It will
    raise an error if coordinates have the wrong dimensionality (e.g., time,
    temperature, velocity).

    Args:
        coords: Coordinate array to validate

    Raises:
        DimensionalityError: If coordinates have units but not length dimensionality

    Examples:
        >>> # Valid: dimensionless coords (for solvers)
        >>> validate_coordinates_dimensionality(np.array([[0, 1], [1, 1]]))

        >>> # Valid: coords with length units
        >>> coords = uw.function.UnitAwareArray([[0, 1000], [1000, 1000]], units="meter")
        >>> validate_coordinates_dimensionality(coords)

        >>> # Invalid: coords with time units (would raise error)
        >>> time_coords = uw.quantity(5.0, "second")
        >>> validate_coordinates_dimensionality(time_coords)  # Raises DimensionalityError
    """
    # Check if coords have units
    has_units_flag, actual_units, backend = _extract_units_info(coords)

    if not has_units_flag:
        # Dimensionless is valid - solver space uses dimensionless coords
        return

    # Check that units have length dimensionality [L]
    try:
        length_ref = backend.create_quantity(1.0, "meter")
        coord_quantity = backend.create_quantity(1.0, actual_units)

        if not backend.check_dimensionality(coord_quantity, length_ref):
            raise DimensionalityError(
                f"Coordinates must have length dimensionality [L], but got '{actual_units}'. "
                f"Coordinates should be positions in space (e.g., meters, kilometers), "
                f"not other physical quantities like time, temperature, or velocity."
            )
    except Exception as e:
        if isinstance(e, DimensionalityError):
            raise
        # If we can't check dimensionality, warn but don't fail
        warnings.warn(
            f"Could not validate coordinate dimensionality for units '{actual_units}': {e}",
            UserWarning
        )


def enforce_units_consistency(*expressions) -> None:
    """
    Enforce units consistency, raising an error if inconsistent.

    Args:
        *expressions: Expressions that must have consistent units

    Raises:
        DimensionalityError: If units are inconsistent
        NoUnitsError: If some have units and others don't
    """
    check_units_consistency(*expressions)  # This already raises appropriate errors


# Note: derivative_units() function has been removed (2025-10-16)
# Natural Pint arithmetic now works directly:
#   gradT_units = temperature.units / mesh.units
# This returns a pint.Unit object that can be used directly
