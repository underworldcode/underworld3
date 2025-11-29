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
    # PRIORITY 0: Check for SymPy expressions first (including derivatives)
    # This must come before unit-aware object check because UWexpression wraps
    # SymPy derivatives and we need to detect the derivative, not the wrapper's units
    try:
        import sympy

        # Check if object has a SymPy expression inside (UWexpression, variables, etc.)
        # TRANSPARENT CONTAINER PRINCIPLE (2025-11-26): Check .sym property (UWexpression,
        # variables) or raw SymPy. The ._expr attribute was part of the deleted
        # UnitAwareExpression class and is no longer used.
        sympy_expr = None
        if hasattr(obj, 'sym'):
            sympy_expr = obj.sym
        elif isinstance(obj, sympy.Basic):
            sympy_expr = obj

        if sympy_expr is not None and isinstance(sympy_expr, sympy.Basic):
            # Check for derivatives FIRST before anything else
            if hasattr(sympy_expr, 'diffindex'):
                # This is a derivative! Compute derivative units: var_units / coord_units
                try:
                    # Get the function being differentiated
                    if hasattr(sympy_expr, 'func'):
                        var_units_info = _extract_units_info(sympy_expr.func)
                        var_units = var_units_info[1] if var_units_info[0] else None

                        # Get the coordinate being differentiated with respect to
                        if hasattr(sympy_expr, 'args') and len(sympy_expr.args) > sympy_expr.diffindex:
                            coord = sympy_expr.args[sympy_expr.diffindex]
                            coord_units_info = _extract_units_info(coord)
                            coord_units = coord_units_info[1] if coord_units_info[0] else None

                            # Compute derivative units
                            if var_units and coord_units:
                                backend = _get_default_backend()
                                # Parse as Pint units and compute derivative
                                var_qty = backend.create_quantity(1.0, var_units)
                                coord_qty = backend.create_quantity(1.0, coord_units)
                                derivative_qty = var_qty / coord_qty
                                derivative_units = derivative_qty.units  # Keep as Pint Unit object
                                return True, derivative_units, backend
                            elif var_units:
                                # No coord units - just return var units
                                backend = _get_default_backend()
                                return True, var_units, backend
                except Exception:
                    # Fall through to other checks if derivative units computation fails
                    pass
    except ImportError:
        pass

    # PRIORITY 1: Check for UWexpression (has has_units and units but NOT _units_backend)
    # TRANSPARENT CONTAINER PRINCIPLE (2025-11-26): UWexpression is a container that
    # derives units from its contents. It doesn't store units itself when wrapping
    # a composite SymPy expression, so we must check .sym to discover units.
    if hasattr(obj, "has_units") and hasattr(obj, "units") and hasattr(obj, "sym"):
        # First check if the object directly reports units (atomic case)
        if obj.has_units and obj.units is not None:
            backend = _get_default_backend()
            return True, obj.units, backend

        # For composite expressions, check the .sym property
        # This handles: th1 = uw.expression("th1", x - xx0 - velocity*t_now)
        # where th1.has_units=False but th1.sym contains unit-aware atoms
        try:
            import sympy
            if hasattr(obj, 'sym') and isinstance(obj.sym, sympy.Basic) and not isinstance(obj.sym, sympy.Number):
                # Recursively check the wrapped expression for units
                sym_units_info = _extract_units_info(obj.sym)
                if sym_units_info[0]:  # If .sym has units, return them
                    return sym_units_info
        except Exception:
            pass
        return False, None, None

    # PRIORITY 2: Check if it's a unit-aware object with full protocol (variables, quantities)
    # All unit-aware objects implement the protocol: has_units, units, _units_backend
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

        # FIRST: Check for Matrix objects before checking sympy.Basic
        # Matrix doesn't inherit from Basic in the expected way, so check it first
        if isinstance(obj, sympy.Matrix) and obj.shape[0] > 0:
            first_element_units = _extract_units_info(obj[0])
            if first_element_units[0]:  # has_units
                return first_element_units

        if isinstance(obj, sympy.Basic):
            # PRIORITY 1: Check for derivatives (diffindex attribute)
            # This must come first because derivatives need special units computation
            if hasattr(obj, 'diffindex'):
                # This is a derivative! Compute derivative units: var_units / coord_units
                try:
                    # Get the function being differentiated
                    if hasattr(obj, 'func'):
                        var_units_info = _extract_units_info(obj.func)
                        var_units = var_units_info[1] if var_units_info[0] else None

                        # Get the coordinate being differentiated with respect to
                        if hasattr(obj, 'args') and len(obj.args) > obj.diffindex:
                            coord = obj.args[obj.diffindex]
                            coord_units_info = _extract_units_info(coord)
                            coord_units = coord_units_info[1] if coord_units_info[0] else None

                            # Compute derivative units
                            if var_units and coord_units:
                                backend = _get_default_backend()
                                # Parse as Pint units and compute derivative
                                var_qty = backend.create_quantity(1.0, var_units)
                                coord_qty = backend.create_quantity(1.0, coord_units)
                                derivative_qty = var_qty / coord_qty
                                derivative_units = derivative_qty.units  # Keep as Pint Unit object
                                return True, derivative_units, backend
                            elif var_units:
                                # No coord units - just return var units
                                backend = _get_default_backend()
                                return True, var_units, backend
                except Exception:
                    # Fall through to other checks if derivative units computation fails
                    pass

            # PRIORITY 2: Check for patched _units attribute (coordinate units from patch_coordinate_units)
            # This handles mesh.X[0], mesh.N.x, etc. which have _units directly attached
            if hasattr(obj, '_units') and obj._units is not None:
                backend = _get_default_backend()
                return True, obj._units, backend

            # SymPy units backend removed - use Pint-native approach instead
            # Third try: use compute_expression_units which handles symbolic expressions with dimensional analysis
            from underworld3.function.unit_conversion import compute_expression_units

            units_result = compute_expression_units(obj)
            if units_result is not None:
                backend = _get_default_backend()
                # compute_expression_units returns pint.Unit objects - keep as Pint object
                return True, units_result, backend

            # Third try: extract unit-aware variables from the expression
            units_from_variables = _extract_units_from_sympy_expression(obj)
            if units_from_variables is not None:
                return units_from_variables

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
                        # Also check if symbol's function class matches variable function class
                        # This handles component access like v[0] which creates a derived symbol
                        symbol_matches = symbol in var_symbols
                        if not symbol_matches and hasattr(symbol, 'func') and hasattr(variable.sym[0], 'func'):
                            # Check if the function classes match (e.g., both are instances of same UW function)
                            symbol_matches = type(symbol.func) == type(variable.sym[0].func)

                        if symbol_matches and variable.has_units:
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

        # If expression is multiplication, combine units using Pint
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

            elif len(unit_info_list) > 1:
                # Multiple variables: multiply their units together using Pint
                from underworld3.scaling import units as ureg
                result_unit = ureg("dimensionless")  # Start with dimensionless
                for units_obj, backend in unit_info_list:
                    # Handle both strings and Pint Unit objects
                    if isinstance(units_obj, str):
                        units_qty = ureg(units_obj)
                    else:
                        units_qty = 1.0 * units_obj  # Convert Pint Unit to Quantity
                    result_unit = result_unit * units_qty
                return result_unit.units  # Keep as Pint Unit object

        # If expression is division, divide units using Pint
        if isinstance(expr, sympy.Pow):
            # Check for division (negative power)
            if len(expr.args) == 2 and expr.args[1] == -1:
                # This is a division: expr.args[0] ** -1
                if len(unit_info_list) >= 1:
                    # Return reciprocal units
                    from underworld3.scaling import units as ureg
                    units_obj = unit_info_list[0][0]
                    # Handle both strings and Pint Unit objects
                    if isinstance(units_obj, str):
                        units_qty = ureg(units_obj)
                    else:
                        units_qty = 1.0 * units_obj  # Convert Pint Unit to Quantity
                    result_unit = units_qty ** -1
                    return result_unit.units  # Keep as Pint Unit object
            elif len(expr.args) == 2 and len(unit_info_list) >= 1:
                # General power: raise units to the power
                from underworld3.scaling import units as ureg
                units_obj = unit_info_list[0][0]
                power = float(expr.args[1])
                # Handle both strings and Pint Unit objects
                if isinstance(units_obj, str):
                    units_qty = ureg(units_obj)
                else:
                    units_qty = 1.0 * units_obj  # Convert Pint Unit to Quantity
                result_unit = units_qty ** power
                return result_unit.units  # Keep as Pint Unit object

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


def get_units(expression, simplify: bool = False) -> Optional[Any]:
    """
    Get the units of an expression or quantity.

    This is the unified, public API for extracting units from any object type
    (variables, quantities, SymPy expressions, etc.). It replaces the previous
    `units_of()` function and the internal `function.unit_conversion.get_units()`
    for a clean, single API surface.

    IMPORTANT: This function ALWAYS returns Pint Unit objects (or None), never strings.
    We accept strings for INPUT (user convenience), but always return Pint objects.

    Args:
        expression: Expression, quantity, or unit-aware object
        simplify: If True, convert mixed units to simplified base SI units.
                  For example, `megayear ** 0.5 * meter / second ** 0.5`
                  simplifies to just `meter`. Default: False.

    Returns:
        Pint Unit object or None if no units

    Examples:
        >>> velocity = uw.discretisation.MeshVariable("velocity", mesh, 2, units="m/s")
        >>> units = uw.get_units(velocity)
        >>> print(units)  # <Unit('meter / second')>

        >>> # Mixed units from composite expressions
        >>> th2 = uw.expression("th2", ((2 * kappa * t_now))**0.5)
        >>> uw.get_units(th2)  # megayear ** 0.5 * meter / second ** 0.5
        >>> uw.get_units(th2, simplify=True)  # meter (simplified!)
    """
    has_units, units, backend = _extract_units_info(expression)

    if not has_units or units is None:
        return None

    # Optionally simplify mixed units to base SI units
    if simplify:
        try:
            # Create a quantity with value 1 and these units, then simplify
            qty = 1 * units
            simplified = qty.to_base_units()
            return simplified.units
        except Exception:
            # If simplification fails, return original units
            pass

    return units


# Backward compatibility alias - deprecated, use get_units() instead
units_of = get_units


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
                # CRITICAL FIX: Convert to base SI units BEFORE dividing by scale
                # The scale is in SI units (e.g., m/s), so the input must also be in SI
                # to ensure proper unit cancellation. Without this, km/yr and cm/yr both
                # give the same dimensionless value (wrong!).
                qty_si = expression._pint_qty.to_base_units()
                result_qty = qty_si / scale

                # After division, should be dimensionless (or very close)
                # Check that units actually cancelled
                if result_qty.dimensionality:
                    # Units didn't cancel - something is wrong
                    raise ValueError(
                        f"Units did not cancel during non-dimensionalisation. "
                        f"Input: {expression._pint_qty}, Scale: {scale}, "
                        f"Result: {result_qty} (expected dimensionless)"
                    )

                # Create dimensionless UWQuantity with preserved dimensionality
                # Handle both scalar and array magnitudes
                mag = result_qty.magnitude
                if hasattr(mag, '__len__') and not isinstance(mag, str):
                    # Array magnitude - return as plain array (or UnitAwareArray for consistency)
                    try:
                        from .utilities.unit_aware_array import UnitAwareArray
                        nondim_array = UnitAwareArray(np.asarray(mag), units="dimensionless")
                        nondim_array._dimensionality = dimensionality
                        return nondim_array
                    except ImportError:
                        return np.asarray(mag)
                else:
                    # Scalar magnitude
                    nondim_value = float(mag)
                    result = UWQuantity(nondim_value, units="dimensionless")
                    result._dimensionality = dimensionality  # Store for unit tracking
                    return result
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
        # IDEMPOTENCY CHECK: Return early if already non-dimensional
        # Non-dimensional arrays have units=None or units='dimensionless'
        units_obj = expression._units if hasattr(expression, '_units') else None

        # Check if already dimensionless using Pint
        if units_obj is None:
            return expression

        # Use Pint to check if dimensionless
        from .scaling import units as ureg
        if hasattr(units_obj, 'dimensionality'):
            # Pint Unit object - check dimensionality directly
            if units_obj.dimensionality == ureg.dimensionless.dimensionality:
                return expression

        # Get scale for this dimensionality
        if not hasattr(model, "_fundamental_scales"):
            # No reference quantities - return plain array
            return np.asarray(expression)

        try:
            # Get dimensionality directly from Pint Unit object
            # POLICY: units_obj should always be a Pint Unit, never a string
            if not hasattr(units_obj, 'dimensionality'):
                raise ValueError(f"Units object has no dimensionality: {type(units_obj)}")

            # Extract dimensionality as dict for model.get_scale_for_dimensionality()
            dimensionality = dict(units_obj.dimensionality)

            if not dimensionality:
                # Dimensionless - return as plain array
                return np.asarray(expression)

            # Get scale from model
            scale = model.get_scale_for_dimensionality(dimensionality)

            # CRITICAL FIX (2025-11-27): Convert array to SI units BEFORE dividing by scale.
            # The scale is computed in SI units (e.g., m/s), so the input must also be
            # in SI for proper unit cancellation. Without this conversion:
            # - Array in cm/yr with values [1.0] would give 1.0 / 31709791.983765 = 3.15e-8
            # - But 1 cm/yr = 3.17e-10 m/s, so correct result = 3.17e-10 / (scale_m/s)
            #
            # Create Pint Quantity with array values and units, convert to SI, then divide
            qty_with_units = np.asarray(expression) * units_obj  # Pint Quantity
            qty_si = qty_with_units.to_base_units()              # Convert to SI base units
            result_qty = qty_si / scale                          # Divide by scale (units cancel)

            # Extract the dimensionless magnitude
            nondim_array = np.asarray(result_qty.magnitude)

            # Create new UnitAwareArray with 'dimensionless' units to make future
            # calls idempotent (subsequent non_dimensionalise calls will recognize
            # the array as already non-dimensional). IMPORTANT: Preserve the original
            # dimensionality metadata so results can be re-dimensionalized later
            nondim_ua = UnitAwareArray(nondim_array, units="dimensionless")

            # Copy the dimensionality metadata from the original array's units
            # This allows subsequent non_dimensionalise() calls to be truly idempotent
            # while preserving information needed for dimensionalization
            nondim_ua._dimensionality = dimensionality

            return nondim_ua

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

    # Protocol 7: Objects with .data or ._compute_nondimensional_value()
    # TRANSPARENT CONTAINER PRINCIPLE (2025-11-27): Let objects handle their own
    # non-dimensionalization. If an object has .data, delegate to it.
    if hasattr(expression, 'data') and hasattr(expression, 'has_units'):
        # Object knows how to non-dimensionalize itself - delegate!
        return expression.data

    # Could not non-dimensionalise
    raise TypeError(
        f"Cannot non-dimensionalise object of type {type(expression)}. "
        f"Must be MeshVariable, SwarmVariable, UWQuantity, UnitAwareArray, or plain number."
    )


def show_nondimensional_form(expression, model=None):
    """
    Display the non-dimensionalized form of a complex expression.

    This function unwraps the expression (expanding UW wrappers to SymPy),
    applies non-dimensional scaling, and returns the result for inspection.
    Useful for seeing what will actually be compiled into C code when
    scaling is active.

    Args:
        expression: Any SymPy expression, UW expression, or variable
        model: Model instance with reference quantities (uses default if None)

    Returns:
        SymPy expression with non-dimensional scaling applied

    Examples:
        >>> # See what the Stokes flux looks like non-dimensionally
        >>> flux = stokes.constitutive_model.flux
        >>> nondim_flux = uw.show_nondimensional_form(flux)
        >>> print(nondim_flux)  # Should show values close to 1.0, not 1e21

        >>> # Check a parameter
        >>> viscosity = model.Parameters.shear_viscosity_0
        >>> print(uw.show_nondimensional_form(viscosity))  # Should be ~1.0
    """
    import underworld3 as uw
    from .function.expressions import _unwrap_for_compilation

    # Get model
    if model is None:
        model = uw.get_default_model()

    # Check if scaling is active
    if not uw._is_scaling_active():
        raise ValueError(
            "Non-dimensional scaling is not active. "
            "Set reference quantities first: model.set_reference_quantities(...)"
        )

    # Unwrap the expression (this will apply non-dimensional scaling to parameters)
    unwrapped = _unwrap_for_compilation(expression, keep_constants=False, return_self=False)

    return unwrapped


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
    from .utilities.unit_aware_array import UnitAwareArray

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
    from .utilities.unit_aware_array import UnitAwareArray
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
                units=result_qty.units,  # ✅ Pass Pint Unit, not string
                dimensionality=dimensionality
            )
        else:
            # Plain value - multiply by scale
            result_qty = expression.value * scale
            return UWQuantity(
                float(result_qty.magnitude),
                units=result_qty.units,  # ✅ Pass Pint Unit, not string
                dimensionality=dimensionality
            )

    elif isinstance(expression, UnitAwareArray):
        # Multiply array by scale
        result_qty = expression.view(np.ndarray) * scale
        # Return UnitAwareArray with proper units
        # NOTE: UnitAwareArray doesn't store dimensionality, only units
        return UnitAwareArray(
            result_qty.magnitude,
            units=result_qty.units  # ✅ Pass Pint Unit, not string
        )

    elif isinstance(expression, (int, float, complex, np.ndarray)):
        # Plain number or array - multiply by scale
        result_qty = expression * scale
        if isinstance(expression, np.ndarray):
            # Return UnitAwareArray
            # NOTE: UnitAwareArray doesn't store dimensionality, only units
            return UnitAwareArray(
                result_qty.magnitude,
                units=result_qty.units  # ✅ Pass Pint Unit, not string
            )
        else:
            # Return UWQuantity
            return UWQuantity(
                float(result_qty.magnitude),
                units=result_qty.units,  # ✅ Pass Pint Unit, not string
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
