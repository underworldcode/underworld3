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
    if not hasattr(model, '_reference_quantities') or not model._reference_quantities:
        return  # No reference quantities to work with

    # Get fundamental scales from reference quantities
    if not hasattr(model, '_fundamental_scales'):
        return  # No fundamental scales derived yet

    fundamental_scales = model._fundamental_scales

    # Process all registered variables
    for name, var in model._variables.items():
        if hasattr(var, 'dimensionality') and var.dimensionality:
            # Try to derive scaling coefficient from dimensionality
            scale = _derive_scale_for_variable(var, fundamental_scales, model._reference_quantities)
            if scale is not None and hasattr(var, 'set_reference_scale'):
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
        if hasattr(qty, 'dimensionality'):
            if str(qty.dimensionality) == str(var_dim):
                # Direct match found - convert to base units first
                try:
                    base_qty = qty.to_base_units()
                    if hasattr(base_qty, 'magnitude'):
                        return float(base_qty.magnitude)
                    elif hasattr(base_qty, 'value'):
                        return float(base_qty.value)
                except:
                    # Fallback if conversion fails
                    if hasattr(qty, 'magnitude'):
                        return float(qty.magnitude)
                    elif hasattr(qty, 'value'):
                        return float(qty.value)

    # Try to construct from fundamental scales using Pint dimensionality objects
    scale = _construct_from_fundamental_scales(var_dim, fundamental_scales)
    if scale is not None:
        return scale

    # For common variable types, use heuristics
    var_name = var.name.lower() if hasattr(var, 'name') else ''
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
        ((ureg.meter / ureg.second).dimensionality,
         lambda s: s.get('length') / s.get('time') if 'length' in s and 'time' in s else None),

        ((ureg.meter**2 / ureg.second).dimensionality,
         lambda s: s.get('length')**2 / s.get('time') if 'length' in s and 'time' in s else None),

        ((ureg.pascal).dimensionality,  # Pressure: mass / (length * time^2)
         lambda s: s.get('mass') / (s.get('length') * s.get('time')**2) if all(k in s for k in ['mass', 'length', 'time']) else None),

        ((ureg.pascal * ureg.second).dimensionality,  # Viscosity: mass / (length * time)
         lambda s: s.get('mass') / (s.get('length') * s.get('time')) if all(k in s for k in ['mass', 'length', 'time']) else None),

        (ureg.kelvin.dimensionality,
         lambda s: s.get('temperature') if 'temperature' in s else None),

        (ureg.meter.dimensionality,
         lambda s: s.get('length') if 'length' in s else None),

        (ureg.second.dimensionality,
         lambda s: s.get('time') if 'time' in s else None),

        (ureg.kilogram.dimensionality,
         lambda s: s.get('mass') if 'mass' in s else None),
    ]

    # Check each known dimensionality pattern
    for target_dim, formula in dim_checks:
        if var_dimensionality == target_dim:
            try:
                result = formula(fundamental_scales)
                if result is None:
                    continue
                # Extract magnitude from Pint Quantity if needed
                if hasattr(result, 'magnitude'):
                    return float(result.magnitude)
                else:
                    return float(result)
            except:
                continue

    return None


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
    if any(v in var_name for v in ['velocity', 'vel', 'flow']):
        # Look for velocity reference
        for name, qty in reference_quantities.items():
            if 'velocity' in name.lower() or 'vel' in name.lower():
                if hasattr(qty, 'magnitude'):
                    return float(qty.magnitude)
        # Fallback to L/T
        if 'length' in fundamental_scales and 'time' in fundamental_scales:
            return fundamental_scales['length'] / fundamental_scales['time']

    # Pressure-like variables
    if any(p in var_name for p in ['pressure', 'stress', 'sigma', 'tau']):
        # Look for pressure/stress reference
        for name, qty in reference_quantities.items():
            if any(p in name.lower() for p in ['pressure', 'stress', 'viscosity']):
                if 'viscosity' in name.lower():
                    # Pressure scale from viscosity: eta * v / L
                    if all(k in fundamental_scales for k in ['length', 'time', 'mass']):
                        L = fundamental_scales['length']
                        T = fundamental_scales['time']
                        M = fundamental_scales['mass']
                        # Stress ~ viscosity * strain_rate ~ (M/(L*T)) * (1/T)
                        return M / (L * T**2)
        # Fallback
        if all(k in fundamental_scales for k in ['mass', 'length', 'time']):
            return fundamental_scales['mass'] / (fundamental_scales['length'] * fundamental_scales['time']**2)

    # Temperature-like variables
    if any(t in var_name for t in ['temperature', 'temp', 'thermal']):
        if 'temperature' in fundamental_scales:
            return fundamental_scales['temperature']
        # Look for temperature reference
        for name, qty in reference_quantities.items():
            if 'temperature' in name.lower() or 'temp' in name.lower():
                if hasattr(qty, 'magnitude'):
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
            if hasattr(var, 'sym'):
                # Check if this function matches the variable's symbolic form
                if str(func_symbol.func) == str(var.sym.func):
                    if hasattr(var, 'scaling_coefficient') and var.scaling_coefficient != 1.0:
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
        if hasattr(model, '_constitutive_models'):
            for const_model in model._constitutive_models:
                if hasattr(const_model, 'parameters'):
                    for param_name, param in const_model.parameters.items():
                        if str(symbol) == str(param.sym) if hasattr(param, 'sym') else str(param):
                            if hasattr(param, 'scaling_coefficient') and param.scaling_coefficient != 1.0:
                                substitutions[symbol] = symbol / param.scaling_coefficient

    return expr.subs(substitutions) if substitutions else expr