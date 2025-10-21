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
    if hasattr(obj, 'has_units') and hasattr(obj, 'units') and hasattr(obj, '_units_backend'):
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
            # Still extract units from variables within SymPy expressions

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
                if hasattr(units_module, 'dimensions'):
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
            if hasattr(symbol, 'name') and symbol.name in ['N.x', 'N.y', 'N.z']:
                continue

            # Find which variable this symbol belongs to
            for var_name, variable in model._variables.items():
                if hasattr(variable, 'sym') and hasattr(variable, 'has_units'):
                    # Check if this symbol matches any component of the variable
                    if hasattr(variable, 'num_components'):
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
            first_backend.create_quantity(1.0, first_units),
            backend.create_quantity(1.0, units)
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


def non_dimensionalise(expression, scaling_system: Optional[Dict] = None) -> Any:
    """
    Convert expression to non-dimensional form for solvers.
    
    This function extracts non-dimensional values suitable for PETSc solvers
    by applying the current scaling system.
    
    Args:
        expression: Expression, quantity, or unit-aware object to non-dimensionalise
        scaling_system: Optional custom scaling coefficients
        
    Returns:
        Non-dimensional value(s)
        
    Raises:
        NoUnitsError: If expression has no units to scale
        
    Examples:
        >>> velocity = EnhancedMeshVariable("velocity", mesh, 2, units="m/s")
        >>> # Set some data...
        >>> nondim_velocity = non_dimensionalise(velocity)
        >>> # Use nondim_velocity in solver
    """
    has_units, units, backend = _extract_units_info(expression)
    
    if not has_units:
        raise NoUnitsError("Cannot non-dimensionalise expression without units")
    
    # If it's a unit-aware variable, use its non_dimensional_value method
    if hasattr(expression, 'non_dimensional_value'):
        return expression.non_dimensional_value()
    
    # Otherwise, create a quantity and non-dimensionalise
    if hasattr(expression, 'data'):
        # Variable-like object with data
        quantity = backend.create_quantity(expression.data, units)
    else:
        # Assume it's already a quantity
        quantity = expression
    
    return backend.non_dimensionalise(quantity)


def dimensionalise(value, target_units: Union[str, Any], 
                  scaling_system: Optional[Dict] = None) -> Any:
    """
    Add dimensions to non-dimensional values.
    
    This function converts non-dimensional solver results back to dimensional
    quantities with specified units.
    
    Args:
        value: Non-dimensional value(s) from solver
        target_units: Units to apply (string or units object)
        scaling_system: Optional custom scaling coefficients
        
    Returns:
        Dimensional quantity with target units
        
    Examples:
        >>> # Get non-dimensional result from solver
        >>> nondim_result = solve_system()
        >>> # Convert back to dimensional
        >>> velocity = dimensionalise(nondim_result, "m/s")
    """
    backend = _get_default_backend()
    return backend.dimensionalise(value, target_units)


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


def create_quantity(value, units: Union[str, Any], 
                   backend: Optional[str] = None) -> Any:
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
        if backend.lower() == 'pint':
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