"""
Helper functions for non-dimensionalization and scaling coefficient derivation.

This module provides functions to automatically derive scaling coefficients
from reference quantities using dimensional analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Any


def derive_scaling_coefficients(model):
    """
    Automatically derive scaling coefficients for all variables and parameters
    based on the model's reference quantities.

    Args:
        model: The Model object with reference quantities set

    This function performs dimensional analysis to find the appropriate
    combination of reference quantities that produces the correct
    dimensionality for each variable/parameter.
    """
    if not hasattr(model, "_reference_quantities") or not model._reference_quantities:
        return  # No reference quantities to work with

    # Get fundamental scales from reference quantities
    if not hasattr(model, "_fundamental_scales"):
        return  # No fundamental scales derived yet

    fundamental_scales = model._fundamental_scales

    # Process all registered variables
    for name, var in model._variables.items():
        if hasattr(var, "dimensionality") and var.dimensionality:
            # Try to derive scaling coefficient from dimensionality
            scale = _derive_scale_for_variable(var, fundamental_scales, model._reference_quantities)
            if scale is not None and hasattr(var, "set_reference_scale"):
                var.set_reference_scale(scale)


def _derive_scale_for_variable(var, fundamental_scales, reference_quantities):
    """
    Derive the scaling coefficient for a variable based on its dimensionality.

    Args:
        var: Variable with dimensionality property
        fundamental_scales: Dictionary of fundamental scales (length, time, mass, etc.)
        reference_quantities: Dictionary of reference quantities

    Returns:
        Scaling coefficient or None if unable to derive
    """
    # First check if any reference quantity matches the variable's dimensionality exactly
    var_dim = var.dimensionality
    for name, qty in reference_quantities.items():
        if hasattr(qty, "dimensionality"):
            if str(qty.dimensionality) == str(var_dim):
                # Direct match found - convert to base units first
                try:
                    base_qty = qty.to_base_units()
                    if hasattr(base_qty, "magnitude"):
                        return float(base_qty.magnitude)
                    elif hasattr(base_qty, "value"):
                        return float(base_qty.value)
                except:
                    # Fallback if conversion fails
                    if hasattr(qty, "magnitude"):
                        return float(qty.magnitude)
                    elif hasattr(qty, "value"):
                        return float(qty.value)

    # Try to construct from fundamental scales using Pint dimensionality objects
    scale = _construct_from_fundamental_scales(var_dim, fundamental_scales)
    if scale is not None:
        return scale

    # For common variable types, use heuristics
    var_name = var.name.lower() if hasattr(var, "name") else ""
    scale = _heuristic_scaling(var_name, var_dim, fundamental_scales, reference_quantities)

    return scale


def _construct_from_fundamental_scales(var_dimensionality, fundamental_scales):
    """
    Try to construct a scaling coefficient from fundamental scales using Pint dimensionality objects.

    Args:
        var_dimensionality: Pint dimensionality object (not string!)
        fundamental_scales: Dict with 'length', 'time', 'mass', 'temperature' Pint Quantity scales

    Returns:
        Scaling coefficient or None
    """
    if not fundamental_scales:
        return None

    # Get Pint units registry for dimensionality comparisons
    try:
        from ..scaling import units as ureg
    except ImportError:
        import pint

        ureg = pint.UnitRegistry()

    # Define dimensionality objects for common physical quantities
    # These are the canonical Pint dimensionality objects
    dim_checks = [
        # (dimensionality, formula to compute scale)
        (
            (ureg.meter / ureg.second).dimensionality,
            lambda s: s.get("length") / s.get("time") if "length" in s and "time" in s else None,
        ),
        (
            (ureg.meter**2 / ureg.second).dimensionality,
            lambda s: (
                s.get("length") ** 2 / s.get("time") if "length" in s and "time" in s else None
            ),
        ),
        (
            (ureg.pascal).dimensionality,  # Pressure: mass / (length * time^2)
            lambda s: (
                s.get("mass") / (s.get("length") * s.get("time") ** 2)
                if all(k in s for k in ["mass", "length", "time"])
                else None
            ),
        ),
        (
            (ureg.pascal * ureg.second).dimensionality,  # Viscosity: mass / (length * time)
            lambda s: (
                s.get("mass") / (s.get("length") * s.get("time"))
                if all(k in s for k in ["mass", "length", "time"])
                else None
            ),
        ),
        (
            ureg.kelvin.dimensionality,
            lambda s: s.get("temperature") if "temperature" in s else None,
        ),
        (ureg.meter.dimensionality, lambda s: s.get("length") if "length" in s else None),
        (ureg.second.dimensionality, lambda s: s.get("time") if "time" in s else None),
        (ureg.kilogram.dimensionality, lambda s: s.get("mass") if "mass" in s else None),
    ]

    # Check each known dimensionality pattern
    for target_dim, formula in dim_checks:
        if var_dimensionality == target_dim:
            try:
                result = formula(fundamental_scales)
                if result is None:
                    continue
                # Extract magnitude from Pint Quantity if needed
                if hasattr(result, "magnitude"):
                    return float(result.magnitude)
                else:
                    return float(result)
            except:
                continue

    return None


def get_required_reference_quantities(units_str, reference_quantities=None):
    """
    Determine which reference quantities are required for a given unit string.

    Uses pure dimensional analysis - NOT hardcoded parameter names.
    Checks if the given unit's dimensionality can be derived from available
    reference quantities through dimensional analysis alone.

    Args:
        units_str: Unit string (e.g., 'Pa', 'm/s', 'K')
        reference_quantities: Optional dict of reference quantities to check against

    Returns:
        tuple: (can_be_derived, descriptive message)
               can_be_derived=True if dimensionality can be derived from available quantities
               message contains explanation of what is available or missing
    """
    try:
        from ..scaling import units as ureg
    except ImportError:
        import pint

        ureg = pint.UnitRegistry()

    # Parse the target units
    try:
        qty = ureg(units_str)
        target_dim = qty.dimensionality
    except:
        return (True, "")  # If we can't parse, don't validate

    # If no reference quantities provided, we can't determine if derivable
    if reference_quantities is None:
        return (False, f"No reference quantities provided to derive units '{units_str}'")

    if not reference_quantities:
        return (False, f"Reference quantities not set for units '{units_str}'")

    # Extract dimensionalities from all reference quantities
    ref_dims = {}
    for name, ref_qty in reference_quantities.items():
        try:
            if hasattr(ref_qty, "dimensionality"):
                ref_dims[name] = ref_qty.dimensionality
        except:
            pass

    # Check if target dimensionality exactly matches any reference quantity
    for name, ref_dim in ref_dims.items():
        if target_dim == ref_dim:
            return (True, f"Dimensionality '{units_str}' matches reference quantity '{name}'")

    # Check if target dimensionality can be derived from reference quantities
    # by analyzing the fundamental scales that can be extracted
    can_derive, explanation = _check_derivability_from_dimensions(target_dim, ref_dims, ureg)

    return (can_derive, explanation)


def _check_derivability_from_dimensions(target_dim, ref_dimensions, ureg):
    """
    Check if a target dimensionality can be derived from reference quantity dimensions.

    Uses pure dimensional analysis to determine if the combination of available
    reference quantities can produce the target dimensionality.

    Args:
        target_dim: Target Pint dimensionality object
        ref_dimensions: Dict mapping reference quantity names to their dimensionalities
        ureg: Pint UnitRegistry

    Returns:
        tuple: (can_derive, explanation_message)
    """
    import itertools

    # Get the fundamental dimensions from reference quantities
    # We need to determine [L], [M], [T], [θ]
    L = M = T = theta = None

    # Track velocity and viscosity for compound derivations
    has_velocity = False
    has_viscosity = False

    # Try to extract fundamental scales from reference quantities
    for name, dim in ref_dimensions.items():
        # Check for pure length
        if dim == ureg.meter.dimensionality:
            L = (name, "direct")

        # Check for pure time
        elif dim == ureg.second.dimensionality:
            T = (name, "direct")

        # Check for pure mass
        elif dim == ureg.kilogram.dimensionality:
            M = (name, "direct")

        # Check for pure temperature
        elif dim == ureg.kelvin.dimensionality:
            theta = (name, "direct")

        # Check for velocity: [L]/[T]
        elif dim == (ureg.meter / ureg.second).dimensionality:
            has_velocity = True

        # Check for viscosity: [M]/([L][T])
        elif dim == (ureg.pascal * ureg.second).dimensionality:
            has_viscosity = True

    # Now try to compute the target dimensionality from [L], [M], [T], [θ]

    # Case 1: We have all three fundamental scales directly
    if L is not None and T is not None and M is not None:
        # We have all fundamental scales - can derive almost anything
        return (True, f"Dimensionality can be derived from all fundamental scales [L], [M], [T]")

    # Case 2: Derive [T] from [L] and velocity [L]/[T]
    if L is not None and has_velocity:
        T_derived = True
    else:
        T_derived = T is not None

    # Case 3: Derive [M] from viscosity [M]/[L]/[T] when we have [L] and [T]
    if has_viscosity and L is not None and T_derived:
        M_derived = True
    else:
        M_derived = M is not None

    # Case 4: Check if we can now derive the target dimensionality
    if L is not None and T_derived and M_derived:
        # We have all fundamental scales (at least derivable)
        return (True, f"Dimensionality can be derived from available reference quantities")

    # Case 5: Specific patterns for velocity
    velocity_dim = (ureg.meter / ureg.second).dimensionality
    if target_dim == velocity_dim:
        if L is not None and T_derived:
            return (
                True,
                "Velocity scale can be derived from length and velocity reference quantities",
            )
        else:
            return (
                False,
                f"Velocity scale [L]/[T] requires length and either time or velocity references",
            )

    # Case 6: Specific pattern for pressure [M]/[L]/[T²]
    pressure_dim = ureg.pascal.dimensionality
    if target_dim == pressure_dim:
        if L is not None and T_derived and M_derived:
            return (True, "Pressure scale can be derived from available reference quantities")
        else:
            missing = []
            if not L:
                missing.append("[L]")
            if not T_derived:
                missing.append(
                    "[T] (try providing domain length with velocity, or direct time scale)"
                )
            if not M_derived:
                missing.append("[M] (try providing viscosity with length/time scales)")
            return (False, f"Pressure scale [M]/[L]/[T²] missing: {', '.join(missing)}")

    # Case 7: Generic success for anything else with all scales
    if L is not None and T_derived and M_derived:
        return (True, "Dimensionality can be derived through dimensional analysis")

    # Default: cannot derive
    return (True, "Dimensionality checking incomplete - using default scaling")


def validate_variable_reference_quantities(var_name, units_str, model):
    """
    Validate that a variable's units can be properly scaled by reference quantities.

    Uses pure dimensional analysis (NOT hardcoded parameter names) to determine
    if the variable's dimensionality can be derived from available reference quantities.

    Args:
        var_name: Name of the variable
        units_str: Unit string for the variable
        model: Model object to check for reference quantities

    Returns:
        tuple: (is_valid, warning_message)
               is_valid=True if units can be scaled properly
               warning_message contains details if validation fails
    """
    if units_str is None:
        return (True, "")

    # Check if model has any reference quantities set
    has_ref_quantities = (
        hasattr(model, "_reference_quantities")
        and model._reference_quantities
        and len(model._reference_quantities) > 0
    )

    if not has_ref_quantities:
        return (
            False,
            f"Variable '{var_name}' has units '{units_str}' but no reference quantities are set.\n"
            f"Call model.set_reference_quantities() before creating variables with units.",
        )

    # Use dimensional analysis to check if units can be derived
    can_derive, message = get_required_reference_quantities(units_str, model._reference_quantities)

    if can_derive:
        # All good - dimensionality can be derived
        return (True, "")
    else:
        # Cannot derive the needed dimensionality
        return (
            False,
            f"Variable '{var_name}' with units '{units_str}' cannot be properly scaled.\n"
            f"  Issue: {message}\n"
            f"  Available reference quantities: {list(model._reference_quantities.keys())}\n"
            f"  Ensure you provide enough reference quantities to define all fundamental scales [L], [M], [T], [θ].",
        )

    return (True, "")


def _heuristic_scaling(var_name, dimensionality, fundamental_scales, reference_quantities):
    """
    Use heuristics based on variable name and type to determine scaling.

    Args:
        var_name: Name of the variable (lowercase)
        dimensionality: Dimensionality string
        fundamental_scales: Dictionary of fundamental scales
        reference_quantities: Dictionary of reference quantities

    Returns:
        Scaling coefficient or None
    """
    # Velocity-like variables
    if any(v in var_name for v in ["velocity", "vel", "flow"]):
        # Look for velocity reference
        for name, qty in reference_quantities.items():
            if "velocity" in name.lower() or "vel" in name.lower():
                if hasattr(qty, "magnitude"):
                    return float(qty.magnitude)
        # Fallback to L/T
        if "length" in fundamental_scales and "time" in fundamental_scales:
            return fundamental_scales["length"] / fundamental_scales["time"]

    # Pressure-like variables
    if any(p in var_name for p in ["pressure", "stress", "sigma", "tau"]):
        # Look for pressure/stress reference
        for name, qty in reference_quantities.items():
            if any(p in name.lower() for p in ["pressure", "stress", "viscosity"]):
                if "viscosity" in name.lower():
                    # Pressure scale from viscosity: eta * v / L
                    if all(k in fundamental_scales for k in ["length", "time", "mass"]):
                        L = fundamental_scales["length"]
                        T = fundamental_scales["time"]
                        M = fundamental_scales["mass"]
                        # Stress ~ viscosity * strain_rate ~ (M/(L*T)) * (1/T)
                        return M / (L * T**2)
        # Fallback
        if all(k in fundamental_scales for k in ["mass", "length", "time"]):
            return fundamental_scales["mass"] / (
                fundamental_scales["length"] * fundamental_scales["time"] ** 2
            )

    # Temperature-like variables
    if any(t in var_name for t in ["temperature", "temp", "thermal"]):
        if "temperature" in fundamental_scales:
            return fundamental_scales["temperature"]
        # Look for temperature reference
        for name, qty in reference_quantities.items():
            if "temperature" in name.lower() or "temp" in name.lower():
                if hasattr(qty, "magnitude"):
                    return float(qty.magnitude)

    return None


def apply_nondimensional_scaling(expr, model):
    """
    Apply non-dimensional scaling to a SymPy expression.

    This replaces all variables and parameters with their non-dimensional
    equivalents by dividing by their scaling coefficients.

    Args:
        expr: SymPy expression
        model: Model object with scaling coefficients set

    Returns:
        Scaled SymPy expression
    """
    import sympy

    substitutions = {}

    # Find all function symbols (variables)
    for func_symbol in expr.atoms(sympy.Function):
        # Try to find corresponding variable in model
        for var_name, var in model._variables.items():
            if hasattr(var, "sym"):
                # Check if this function matches the variable's symbolic form
                if str(func_symbol.func) == str(var.sym.func):
                    if hasattr(var, "scaling_coefficient") and var.scaling_coefficient != 1.0:
                        # Create non-dimensional symbol
                        nd_func = sympy.Function(f"{func_symbol.func}_star")
                        # Replace with scaled version
                        substitutions[func_symbol] = func_symbol / var.scaling_coefficient
                    break

    # Find all regular symbols (parameters)
    for symbol in expr.atoms(sympy.Symbol):
        # Check if this is a tracked parameter
        # This would need model to track parameters separately
        # For now, we'll look for common parameter patterns
        if hasattr(model, "_constitutive_models"):
            for const_model in model._constitutive_models:
                if hasattr(const_model, "parameters"):
                    for param_name, param in const_model.parameters.items():
                        if str(symbol) == str(param.sym) if hasattr(param, "sym") else str(param):
                            if (
                                hasattr(param, "scaling_coefficient")
                                and param.scaling_coefficient != 1.0
                            ):
                                substitutions[symbol] = symbol / param.scaling_coefficient

    return expr.subs(substitutions) if substitutions else expr
