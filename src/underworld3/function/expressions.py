"""
UWexpression - Lazy-evaluation expression wrapper for Underworld3

This module provides:
- UWexpression: A SymPy Symbol that wraps values for lazy evaluation
- Helper functions for unwrapping expressions for JIT compilation and display

Design Principles (Simplified Architecture 2025-11, updated 2025-12):
1. UWexpression is a SymPy Symbol that wraps something (lazy evaluation)
2. Units are DISCOVERED from the wrapped thing, not tracked separately
3. Arithmetic returns pure SymPy expressions - delegate to Symbol
4. Unit conversion (.to(), .to_base_units(), etc.) delegates to uw.* base functions
   - This follows the DRY principle: conversion logic in ONE place (units.py)
   - UWexpression.to() simply calls uw.convert_units(self, target)
"""

import sympy
import numpy as np
from sympy import Symbol, simplify, Number
import underworld3 as uw
from underworld3.utilities._api_tools import uw_object
from underworld3.utilities.mathematical_mixin import MathematicalMixin
from underworld3.discretisation import MeshVariable
from .quantities import UWQuantity


# ============================================================================
# Helper Functions for Unit Operations
# ============================================================================

def simplify_units(units):
    """
    Simplify combined units to human-readable form.

    For example: megayear * centimeter / year → kilometer (with proper scaling)

    This uses Pint's to_compact() to choose human-friendly unit prefixes.
    """
    if units is None:
        return None
    try:
        # Create a quantity with value 1 and these units, then simplify
        qty = 1 * units
        simplified = qty.to_compact()
        return simplified.units
    except Exception:
        # If simplification fails, return original units
        return units


# ============================================================================
# Unified Expression Unwrapping System (2025-12)
# ============================================================================
#
# This unified system replaces the previous multiple paths for unwrapping:
# - _substitute_all_once / _unwrap_expressions (JIT compilation)
# - _expand_dimensional_once / _expand_dimensional (user display)
# - unwrap_for_evaluate (evaluation/lambdify)
#
# Key design:
# - Single tree-walking logic with mode parameter
# - Mode controls value extraction: 'nondimensional', 'dimensional', 'symbolic'
# - Uses free_symbols instead of atoms() for more reliable iteration
# ============================================================================


def _unwrap_atom(atom, mode='nondimensional'):
    """
    Extract the value from a single atom based on mode.

    Args:
        atom: UWexpression, UWQuantity, UWCoordinate, or other symbol
        mode: 'nondimensional' - use .data for ND values (JIT/evaluate)
              'dimensional' - use .value for display
              'symbolic' - use .sym for symbolic substitution

    Returns:
        The unwrapped value (float, sympy.Number, or sympy expression)
    """
    import underworld3
    from underworld3.coordinates import UWCoordinate

    # UWCoordinate: always unwrap to BaseScalar (placeholder for evaluation)
    if isinstance(atom, UWCoordinate):
        return atom.sym

    # UWexpression
    if isinstance(atom, UWexpression):
        if mode == 'dimensional':
            # User display: show dimensional value
            return atom.value if atom.value is not None else atom.sym
        elif mode == 'nondimensional':
            # JIT/evaluate: non-dimensionalize if scaling active
            if atom.has_units and underworld3._is_scaling_active():
                try:
                    return float(atom.data)  # ND value
                except Exception:
                    pass
            # Recursively unwrap to get inner expression
            return atom.sym
        else:  # symbolic
            return atom.sym

    # UWQuantity (not wrapped in UWexpression)
    if isinstance(atom, UWQuantity):
        if mode == 'nondimensional':
            import underworld3
            if underworld3._is_scaling_active() and atom.has_units:
                try:
                    return float(atom.data)
                except Exception:
                    pass
            return atom.value if hasattr(atom, 'value') else atom
        elif mode == 'dimensional':
            return atom.value if hasattr(atom, 'value') else atom
        else:
            return atom

    # Not a UW type - return unchanged
    return atom


def _unwrap_expression_once(expr, mode='nondimensional'):
    """
    Single substitution pass over all UW atoms in an expression.

    Uses free_symbols for reliable iteration (avoids issues with atoms()).

    Args:
        expr: SymPy expression possibly containing UW atoms
        mode: See _unwrap_atom

    Returns:
        Expression with UW atoms substituted
    """
    from underworld3.coordinates import UWCoordinate

    # Handle non-expression types directly
    if isinstance(expr, UWQuantity) and not isinstance(expr, UWexpression):
        return _unwrap_atom(expr, mode)

    if isinstance(expr, UWCoordinate):
        return _unwrap_atom(expr, mode)

    if isinstance(expr, UWexpression):
        # Unwrap the UWexpression itself first
        inner = _unwrap_atom(expr, mode)
        if inner is not expr:
            return inner

    if not hasattr(expr, 'free_symbols'):
        return expr

    # Build substitution dict for all UW atoms
    substitutions = {}
    for sym in expr.free_symbols:
        if isinstance(sym, (UWexpression, UWQuantity, UWCoordinate)):
            replacement = _unwrap_atom(sym, mode)
            if replacement is not sym:
                substitutions[sym] = replacement

    if substitutions:
        return expr.subs(substitutions)
    return expr


def unwrap_expression(expr, mode='nondimensional', depth=None):
    """
    Unified unwrapping of UW expressions.

    This is the single entry point for all expression unwrapping needs.

    Args:
        expr: Expression to unwrap (UWexpression, SymPy expr, or value)
        mode: 'nondimensional' - for JIT compilation and evaluation (uses .data)
              'dimensional' - for user display (uses .value)
              'symbolic' - just expand .sym structure
        depth: Maximum expansion depth (None = complete expansion)

    Returns:
        Pure SymPy expression with all UW atoms expanded
    """
    # Extract sym if needed
    if hasattr(expr, 'sym') and isinstance(expr, UWexpression):
        working = expr
    elif isinstance(expr, sympy.Basic):
        working = expr
    elif isinstance(expr, UWQuantity):
        return _unwrap_atom(expr, mode)
    else:
        return sympy.sympify(expr)

    # Fixed-point iteration (or depth-limited)
    if depth is None:
        result = working
        result_next = _unwrap_expression_once(result, mode)
        iteration = 0
        max_iterations = 100  # Safety limit
        while result is not result_next and iteration < max_iterations:
            result = result_next
            result_next = _unwrap_expression_once(result, mode)
            iteration += 1
        return result
    else:
        result = working
        for _ in range(depth):
            result_next = _unwrap_expression_once(result, mode)
            if result is result_next:
                break
            result = result_next
        return result


# ============================================================================
# Legacy API - Preserved for backward compatibility
# ============================================================================

def is_constant_expr(fn):
    """
    Check if expression has no mesh variable dependencies.

    An expression is "constant" in the UW sense if it doesn't depend on
    mesh coordinates or mesh variables.
    """
    deps = extract_expressions_and_functions(fn)
    return not bool(deps)


def extract_expressions(fn):
    """Extract all UWexpression atoms from a SymPy expression."""
    import underworld3

    if isinstance(fn, underworld3.function.expression):
        fn = fn.sym

    if not hasattr(fn, 'atoms'):
        return set()

    atoms = fn.atoms(sympy.Symbol)

    # exhaustion criterion
    if atoms == fn.atoms():
        return set()

    for atom in atoms:
        if isinstance(atom, underworld3.function.expression):
            sub_atomic = extract_expressions(atom)
            atoms = atoms.union(sub_atomic)

    return atoms


def extract_expressions_and_functions(fn):
    """Extract all UWexpression, Function, and BaseScalar atoms."""
    import underworld3

    if isinstance(fn, underworld3.function.expression):
        fn = fn.sym

    # Handle UWQuantity objects - they don't have atoms() method
    if isinstance(fn, underworld3.function.UWQuantity):
        return set()

    if not hasattr(fn, 'atoms'):
        return set()

    atoms = fn.atoms(sympy.Symbol, sympy.Function, sympy.vector.scalar.BaseScalar)

    # exhaustion criterion
    if atoms == fn.atoms():
        return atoms

    for atom in atoms:
        if isinstance(atom, underworld3.function.expression):
            sub_atomic = extract_expressions_and_functions(atom)
            atoms = atoms.union(sub_atomic)

    return atoms


def _unwrap_expressions(fn, keep_constants=True, return_self=True):
    """
    Main unwrapping logic for JIT compilation.

    DEPRECATED: Use unwrap_expression(fn, mode='nondimensional') instead.
    This function is preserved for backward compatibility.
    """
    # Use unified implementation
    return unwrap_expression(fn, mode='nondimensional')


def _unwrap_for_compilation(fn, keep_constants=True, return_self=True):
    """
    INTERNAL ONLY: Unwrap UW expressions to pure SymPy for JIT compilation.

    DEPRECATED: Use unwrap_expression(fn, mode='nondimensional') instead.
    """
    # Handle UWDerivativeExpression specially
    if isinstance(fn, UWDerivativeExpression):
        result = fn.doit()
    elif isinstance(fn, sympy.Matrix):
        f = lambda x: unwrap_expression(x, mode='nondimensional')
        result = fn.applyfunc(f)
    else:
        result = unwrap_expression(fn, mode='nondimensional')

    return result


# Alias for internal use
substitute = _unwrap_for_compilation


def expand(expr, depth=None, simplify_result=False):
    """
    Expand UW expression to reveal SymPy structure for inspection.

    This function recursively expands nested UW expressions to reveal their
    underlying SymPy representation. It's designed for user inspection and
    debugging - use dimensional values (not scaled).

    Args:
        expr: UW expression to expand
        depth (int, optional): Maximum expansion depth. None = full expansion
        simplify_result (bool): If True, apply SymPy simplification

    Returns:
        Pure SymPy expression with all UW wrappers removed (dimensional values)
    """
    expanded = unwrap_expression(expr, mode='dimensional', depth=depth)

    if simplify_result:
        expanded = sympy.simplify(expanded)

    return expanded


def unwrap(fn, depth=None, keep_constants=True, return_self=True):
    """
    Expand UW expression to reveal SymPy structure.

    Args:
        fn: Expression to unwrap
        depth: Maximum expansion depth (None = complete)
        keep_constants: If False, use nondimensional mode (for JIT)
        return_self: If False, use nondimensional mode (for JIT)

    Returns:
        Unwrapped SymPy expression
    """
    if not keep_constants or not return_self:
        # JIT compilation path - use nondimensional mode
        return unwrap_expression(fn, mode='nondimensional', depth=depth)

    # Display path - use dimensional mode
    return unwrap_expression(fn, mode='dimensional', depth=depth)


def unwrap_for_evaluate(expr, scaling_active=None):
    """
    Unwrap expression for evaluate/lambdify path with proper unit handling.

    Type-based dispatch (2025-12 UWCoordinate design):
    - UWCoordinate: unwrap to BaseScalar (placeholder, NO scaling)
    - UWexpression with units: nondimensionalize via .data
    - UWexpression without units: recursively unwrap .sym
    - UWQuantity: nondimensionalize via .data
    - BaseScalar/MeshVariable.sym: pass through unchanged

    Returns:
        tuple: (unwrapped_expr, result_dimensionality)
    """
    import underworld3 as uw
    from underworld3.units import get_units, get_dimensionality
    from underworld3.coordinates import UWCoordinate

    # Step 1: Get expression dimensionality
    # IMPORTANT: For UWexpression, try the wrapper FIRST because it stores units
    # via ._value_with_units. The raw .sym expression has no unit metadata.
    if isinstance(expr, UWexpression):
        # Try wrapper first (has unit info from arithmetic)
        result_units = get_units(expr)
        # If wrapper has no units, try the .sym expression (may contain unit-aware variables)
        if result_units is None and isinstance(expr.sym, sympy.Expr) and not isinstance(expr.sym, sympy.Number):
            result_units = get_units(expr.sym)
    else:
        result_units = get_units(expr)

    if result_units is not None:
        try:
            if hasattr(result_units, 'dimensionality'):
                result_dimensionality = result_units.dimensionality
            else:
                result_dimensionality = get_dimensionality(result_units)
        except Exception:
            result_dimensionality = None
    else:
        result_dimensionality = None

    # Determine if we should non-dimensionalize
    if scaling_active is None:
        scaling_active = uw.is_nondimensional_scaling_active()

    model = uw.get_default_model()
    should_scale = scaling_active and model.has_units()

    # Step 2: Handle UWCoordinate directly (placeholder - no scaling)
    if isinstance(expr, UWCoordinate):
        # Coordinates are placeholders that take input values at evaluation
        # Unwrap to BaseScalar for JIT compatibility, don't scale
        return expr.sym, result_dimensionality

    # Step 3: Extract and process expression
    if isinstance(expr, UWexpression):
        if isinstance(expr.sym, sympy.Expr) and not isinstance(expr.sym, sympy.Number):
            # Wraps a symbolic expression
            sym_expr = expr.sym
        elif expr.has_units:
            # Wraps a quantity with units
            if should_scale:
                return sympy.sympify(float(expr.data)), result_dimensionality
            else:
                return sympy.sympify(float(expr.value)), result_dimensionality
        else:
            return sympy.sympify(expr.value), result_dimensionality

    elif isinstance(expr, UWQuantity):
        if should_scale:
            return sympy.sympify(expr.data), result_dimensionality
        return sympy.sympify(expr.value), result_dimensionality

    elif hasattr(expr, 'sym'):
        sym_expr = expr.sym
    else:
        sym_expr = expr

    # Step 4: Process composite expressions - TYPE-BASED DISPATCH
    if isinstance(sym_expr, sympy.Expr):
        substitutions = {}
        for sym in sym_expr.free_symbols:
            # UWCoordinate: unwrap to BaseScalar, NO scaling
            if isinstance(sym, UWCoordinate):
                substitutions[sym] = sym.sym  # The BaseScalar

            # UWexpression: nondimensionalize based on whether it has units
            elif isinstance(sym, UWexpression):
                if sym.has_units:
                    if should_scale:
                        substitutions[sym] = float(sym.data)
                    else:
                        substitutions[sym] = float(sym.value)
                else:
                    # RECURSIVE UNWRAP: When a UWexpression has no units but wraps
                    # a composite expression, recursively unwrap to nondimensionalize
                    # any UWexpressions inside it.
                    inner_unwrapped, _ = unwrap_for_evaluate(sym.sym, scaling_active=should_scale)
                    substitutions[sym] = inner_unwrapped

            # UWQuantity (not wrapped in UWexpression): nondimensionalize
            elif isinstance(sym, UWQuantity):
                if should_scale:
                    substitutions[sym] = float(sym.data)
                else:
                    substitutions[sym] = float(sym.value)

            # BaseScalar, MeshVariable.sym, etc.: pass through unchanged

        if substitutions:
            sym_expr = sym_expr.subs(substitutions)

    return sym_expr, result_dimensionality


def substitute_expr(fn, sub_expr, keep_constants=True, return_self=True):
    """Substitute a specific expression throughout."""
    expr = fn
    expr_s = _substitute_one_expr(expr, sub_expr, keep_constants)

    while expr is not expr_s:
        expr = expr_s
        expr_s = _substitute_one_expr(expr, sub_expr, keep_constants)
    return expr


# ============================================================================
# UWexpression Class - Simplified (no UWQuantity inheritance)
# ============================================================================

class UWexpression(MathematicalMixin, uw_object, Symbol):
    """
    A SymPy Symbol that wraps a value for lazy evaluation.

    UWexpression is a named symbolic placeholder. When used in SymPy expressions,
    it acts as a Symbol. At evaluation time, the wrapped value is substituted.

    Key Design (Simplified 2025-11):
    - Inherits from Symbol for SymPy compatibility
    - Does NOT inherit from UWQuantity (expressions don't have units themselves)
    - Units are discovered from the wrapped thing when needed
    - Arithmetic returns pure SymPy expressions

    Parameters
    ----------
    name : str
        LaTeX-style name for display (e.g., r"\\alpha", r"\\rho_0")
    sym : any, optional
        The wrapped value. Can be:
        - A number
        - A UWQuantity (carries units)
        - Another UWexpression (nested lazy evaluation)
        - A SymPy expression
    description : str, optional
        Human-readable description

    Examples
    --------
    >>> alpha = uw.expression(r"\\alpha", uw.quantity(3e-5, "1/K"))
    >>> rho0 = uw.expression(r"\\rho_0", uw.quantity(3300, "kg/m^3"))
    >>>
    >>> # Symbolic multiplication
    >>> product = rho0 * alpha  # Returns SymPy Mul
    >>>
    >>> # Wrap the product for lazy evaluation
    >>> combo = uw.expression(r"\\rho_0 \\alpha", product)
    """

    _expr_count = 0
    _expr_names = {}
    _ephemeral_expr_names = {}

    def __new__(
        cls,
        name,
        *args,
        _unique_name_generation=False,
        **kwargs,
    ):
        import warnings
        import weakref

        instance_no = UWexpression._expr_count

        # If the expression already exists, return it
        if name in UWexpression._expr_names.keys() and _unique_name_generation == False:
            return UWexpression._expr_names[name]

        # Check both dicts
        name_exists_persistent = name in UWexpression._expr_names
        name_exists_ephemeral = name in UWexpression._ephemeral_expr_names

        if (name_exists_persistent or name_exists_ephemeral) and _unique_name_generation == True:
            invisible = rf"\hspace{{ {instance_no/10000}pt }}"
            unique_name = f"{{ {name} {invisible} }}"
        else:
            unique_name = name

        obj = Symbol.__new__(cls, unique_name)
        obj._instance_no = instance_no
        obj._unique_name = unique_name
        obj._given_name = name
        obj._is_ephemeral = _unique_name_generation

        if _unique_name_generation:
            def cleanup_callback(ref):
                if unique_name in UWexpression._ephemeral_expr_names:
                    del UWexpression._ephemeral_expr_names[unique_name]
            try:
                UWexpression._ephemeral_expr_names[unique_name] = weakref.ref(obj, cleanup_callback)
            except TypeError:
                UWexpression._expr_names[unique_name] = obj
        else:
            UWexpression._expr_names[unique_name] = obj

        UWexpression._expr_count += 1

        return obj

    def __init__(
        self,
        name,
        sym=None,
        description="No description provided",
        value=None,  # Legacy parameter
        units=None,  # Units for wrapping the value
        **kwargs,
    ):
        # Handle legacy 'value' parameter
        if value is not None and sym is None:
            import warnings
            warnings.warn(
                message=f"DEPRECATION: Use 'sym' attribute instead of 'value': {value}"
            )
            sym = value

        if value is not None and sym is not None:
            raise ValueError("Both 'sym' and 'value' provided - use only one")

        # If units are provided and sym is a plain numeric value, wrap it in UWQuantity
        # Don't wrap if sym is already a UWQuantity, UWexpression, or SymPy expression
        if units is not None and not isinstance(sym, UWQuantity):
            # Only wrap plain numeric values - not expressions or other complex types
            if isinstance(sym, (int, float, np.integer, np.floating)):
                sym = UWQuantity(sym, units)
            elif sym is not None and not isinstance(sym, (sympy.Basic, UWexpression)):
                # Try to wrap other numeric-like things (e.g., numpy scalars)
                try:
                    sym = UWQuantity(float(sym), units)
                except (TypeError, ValueError):
                    # Can't convert to float - ignore units parameter
                    pass

        # TRANSPARENT CONTAINER PRINCIPLE (2025-11-27):
        # Store the wrapped object directly - don't extract or decompose it.
        # The container provides access to what's inside, never "owns" metadata.
        self._wrapped = sym

        # _sym stores the wrapped object directly (not extracted parts)
        # This allows unwrap() to see UWQuantity and handle it correctly
        if isinstance(sym, UWQuantity):
            self._sym = sym  # Keep the full UWQuantity!
        elif isinstance(sym, (sympy.Basic, sympy.matrices.MatrixBase)):
            self._sym = sym
        else:
            self._sym = sympy.sympify(sym) if sym is not None else None

        # Metadata
        self.symbol = self._given_name
        self._description = description

        # UW object tracking
        self._uw_id = uw_object._obj_count
        uw_object._obj_count += 1

    # =========================================================================
    # Core Properties
    # =========================================================================

    @property
    def sym(self):
        """Get the symbolic/numeric value."""
        return self._sym

    @sym.setter
    def sym(self, new_value):
        """Update the wrapped value."""
        # TRANSPARENT CONTAINER PRINCIPLE: Store the object directly
        self._wrapped = new_value

        if isinstance(new_value, UWQuantity):
            self._sym = new_value  # Keep the full UWQuantity!
        elif isinstance(new_value, (sympy.Basic, sympy.matrices.MatrixBase)):
            self._sym = new_value
        else:
            self._sym = sympy.sympify(new_value) if new_value is not None else None

    @property
    def value(self):
        """Get the dimensional value of the wrapped thing."""
        # TRANSPARENT CONTAINER: Always derive from _sym (the wrapped object)
        if hasattr(self._sym, 'value'):
            return self._sym.value
        return self._sym

    @property
    def data(self):
        """Get the non-dimensional value for computation."""
        return self._compute_nondimensional_value()

    def _compute_nondimensional_value(self):
        """
        Internal: compute the non-dimensional value from the wrapped object.

        This is the machinery that .data uses. Named explicitly to be self-documenting.
        """
        # TRANSPARENT CONTAINER: Derive from _sym (the wrapped object)
        if hasattr(self._sym, 'data'):
            return self._sym.data  # Delegate to wrapped object's .data
        elif hasattr(self._sym, 'value'):
            return self._sym.value  # Fallback to dimensional value
        return self._sym

    @property
    def units(self):
        """Get units from the wrapped thing (if it has units)."""
        # TRANSPARENT CONTAINER: Always derive from _sym
        if hasattr(self._sym, 'units'):
            return self._sym.units
        return None

    @property
    def has_units(self):
        """Check if the wrapped thing has units."""
        return self.units is not None

    @property
    def dimensionality(self):
        """Get dimensionality from the wrapped thing."""
        # TRANSPARENT CONTAINER: Always derive from _sym
        if hasattr(self._sym, 'dimensionality'):
            return self._sym.dimensionality
        return {}

    @property
    def expression(self):
        """Get the unwrapped expression."""
        return unwrap(self)

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, value):
        self._description = value

    @property
    def expression_number(self):
        """Unique number of the expression instance."""
        return self._instance_no

    # =========================================================================
    # Quantity accessor (for when you need to do quantity arithmetic)
    # =========================================================================

    @property
    def quantity(self):
        """
        Get the wrapped quantity for numeric arithmetic with units.

        Returns the underlying UWQuantity if one was provided, or creates
        one from the value.
        """
        # TRANSPARENT CONTAINER: Derive from _sym (the wrapped object)
        if isinstance(self._sym, UWQuantity):
            return self._sym
        else:
            return UWQuantity(self.value, units=None)

    # =========================================================================
    # Unit Conversion Methods - Delegate to uw.* base functions
    # =========================================================================

    def to(self, target_units: str) -> 'UWexpression':
        """
        Convert to different units.

        Delegates to uw.convert_units() for the actual conversion.

        Parameters
        ----------
        target_units : str
            Target units (e.g., "m/s", "km", "degC")

        Returns
        -------
        UWexpression
            New expression with converted value and units

        Examples
        --------
        >>> radius = uw.expression("r", uw.quantity(6370, "km"))
        >>> radius_m = radius.to("m")
        >>> print(radius_m.value)  # 6370000.0
        """
        import underworld3 as uw
        return uw.convert_units(self, target_units)

    def to_base_units(self) -> 'UWexpression':
        """
        Convert to SI base units.

        Delegates to uw.to_base_units() for the actual conversion.

        Returns
        -------
        UWexpression
            New expression with value in SI base units

        Examples
        --------
        >>> velocity = uw.expression("v", uw.quantity(100, "km/h"))
        >>> velocity_si = velocity.to_base_units()
        >>> print(velocity_si.value)  # 27.78 (m/s)
        """
        import underworld3 as uw
        return uw.to_base_units(self)

    def to_reduced_units(self) -> 'UWexpression':
        """
        Simplify units by canceling common factors.

        Delegates to uw.to_reduced_units() for the actual simplification.

        Returns
        -------
        UWexpression
            New expression with simplified units
        """
        import underworld3 as uw
        return uw.to_reduced_units(self)

    def to_compact(self) -> 'UWexpression':
        """
        Convert to most human-readable unit representation.

        Automatically chooses unit prefixes (kilo, mega, micro, etc.)
        to make the number more readable.

        Delegates to uw.to_compact() for the actual conversion.

        Returns
        -------
        UWexpression
            New expression with compact units

        Examples
        --------
        >>> length = uw.expression("L", uw.quantity(0.001, "km"))
        >>> length_compact = length.to_compact()
        >>> print(length_compact)  # 1.0 [meter]
        """
        import underworld3 as uw
        return uw.to_compact(self)

    # =========================================================================
    # SymPy Compatibility
    # =========================================================================

    def _sympy_(self):
        """SymPy protocol - return self (we ARE a Symbol)."""
        return self

    def _sympify_(self):
        """SymPy sympify protocol - return self."""
        return self

    def atoms(self, *types):
        """Use Symbol's atoms() method."""
        return Symbol.atoms(self, *types)

    def __bool__(self):
        """Always True for boolean contexts."""
        return True

    def __hash__(self):
        """Delegate to Symbol's hash."""
        return Symbol.__hash__(self)

    def __eq__(self, other):
        """Delegate to Symbol's equality (symbolic identity)."""
        return Symbol.__eq__(self, other)

    def __ne__(self, other):
        """Delegate to Symbol's inequality."""
        return Symbol.__ne__(self, other)

    @property
    def is_number(self):
        """UWexpression is a Symbol, not a number."""
        return False

    @property
    def is_comparable(self):
        """Delegate to wrapped expression."""
        if self._sym is not None and hasattr(self._sym, 'is_comparable'):
            return self._sym.is_comparable
        return True

    @property
    def is_extended_real(self):
        """Delegate to wrapped expression."""
        if self._sym is not None and hasattr(self._sym, 'is_extended_real'):
            return self._sym.is_extended_real
        return None

    @property
    def is_positive(self):
        """Delegate to wrapped expression."""
        if self._sym is not None and hasattr(self._sym, 'is_positive'):
            return self._sym.is_positive
        return None

    @property
    def is_negative(self):
        """Delegate to wrapped expression."""
        if self._sym is not None and hasattr(self._sym, 'is_negative'):
            return self._sym.is_negative
        return None

    @property
    def is_zero(self):
        """Delegate to wrapped expression."""
        if self._sym is not None and hasattr(self._sym, 'is_zero'):
            return self._sym.is_zero
        return None

    @property
    def is_finite(self):
        """Delegate to wrapped expression."""
        if self._sym is not None and hasattr(self._sym, 'is_finite'):
            return self._sym.is_finite
        return None

    def is_constant(self, *wrt, **flags):
        """SymPy-compatible is_constant - delegate to Symbol."""
        return Symbol.is_constant(self, *wrt, **flags)

    def is_uw_constant(self):
        """UW-specific: does this have no mesh variable dependencies?"""
        return is_constant_expr(self)

    def constant(self):
        """Deprecated - use is_uw_constant()."""
        return is_constant_expr(self)

    def diff(self, *args, **kwargs):
        """Differentiation - delegate to Symbol."""
        return Symbol.diff(self, *args, **kwargs)

    # =========================================================================
    # Arithmetic - Return UWexpression to preserve units through operations
    # =========================================================================

    def __mul__(self, other):
        """Multiplication - return UWexpression to preserve units."""
        # Handle matrix cases
        if hasattr(self, '_sym') and isinstance(self._sym, sympy.MatrixBase):
            return self._sym.__mul__(other)
        if isinstance(other, (sympy.MatrixBase, sympy.matrices.expressions.MatrixExpr)):
            # Use applyfunc to multiply each element by self (as Symbol).
            # This preserves unit tracking: result is Matrix([x * self, ...])
            # where get_units() can find units for both self AND matrix elements.
            # DON'T use self._sym * other - that loses matrix element units!
            return other.applyfunc(lambda x: x * self)

        # Handle UWQuantity - preserve units
        if isinstance(other, UWQuantity):
            return other.__rmul__(self)

        # Handle UWexpression - preserve LAZY evaluation by returning SymPy product
        # The symbolic product (self * other) preserves references to both expressions
        # as atoms in the SymPy Mul object. Units are derived on demand by get_units()
        # which traverses the expression tree and finds the UWexpression atoms.
        #
        # DESIGN: We don't store units on intermediate products because:
        # 1. SymPy Mul objects are immutable - can't attach attributes
        # 2. Storing units creates sync issues if operands change
        # 3. get_units() already has infrastructure to derive units from expression trees
        if isinstance(other, UWexpression):
            # Return raw SymPy product - units derived on demand via get_units()
            return Symbol.__mul__(self, other)

        # Scalar multiplication - preserve self's units
        if isinstance(other, (int, float)):
            if self.units is not None:
                result_value = self.value * other
                return UWexpression(
                    f"({self.name}*{other})",
                    UWQuantity(result_value, self.units),
                    _unique_name_generation=True
                )

        # Default to SymPy multiplication
        return Symbol.__mul__(self, other)

    def __rmul__(self, other):
        """Right multiplication - handle UWQuantity to preserve units."""
        # Handle UWQuantity - preserve units
        if isinstance(other, UWQuantity):
            return other.__mul__(self)

        # Scalar multiplication - preserve self's units
        if isinstance(other, (int, float)):
            if self.units is not None:
                result_value = other * self.value
                return UWexpression(
                    f"({other}*{self.name})",
                    UWQuantity(result_value, self.units),
                    _unique_name_generation=True
                )

        return Symbol.__rmul__(self, other)

    def __truediv__(self, other):
        """Division - return UWexpression to preserve units."""
        # Handle UWQuantity - use full Pint arithmetic
        if isinstance(other, UWQuantity):
            self_units = self.units
            other_units = other.units
            if self_units is not None and other_units is not None:
                from ..scaling import units as ureg
                # Use FULL Pint quantity arithmetic
                self_pint = self.value * self_units
                other_pint = other.value * other_units
                result_pint = (self_pint / other_pint).to_compact()
                return UWexpression(
                    f"({self.name}/{other})",
                    UWQuantity(result_pint.magnitude, str(result_pint.units)),
                    _unique_name_generation=True
                )
            elif self_units is not None:
                from ..scaling import units as ureg
                self_pint = self.value * self_units
                result_pint = (self_pint / other.value).to_compact()
                return UWexpression(
                    f"({self.name}/{other})",
                    UWQuantity(result_pint.magnitude, str(result_pint.units)),
                    _unique_name_generation=True
                )
            elif other_units is not None:
                from ..scaling import units as ureg
                other_pint = other.value * other_units
                result_pint = (self.value / other_pint).to_compact()
                return UWexpression(
                    f"({self.name}/{other})",
                    UWQuantity(result_pint.magnitude, str(result_pint.units)),
                    _unique_name_generation=True
                )

        # Handle UWexpression - preserve LAZY evaluation by returning SymPy quotient
        # Same design as __mul__: return raw SymPy quotient, derive units on demand
        if isinstance(other, UWexpression):
            return Symbol.__truediv__(self, other)

        # Scalar division - preserve self's units
        if isinstance(other, (int, float)):
            if self.units is not None:
                result_value = self.value / other
                return UWexpression(
                    f"({self.name}/{other})",
                    UWQuantity(result_value, self.units),
                    _unique_name_generation=True
                )

        return Symbol.__truediv__(self, other)

    def __rtruediv__(self, other):
        """Right division - handle UWQuantity to preserve units."""
        if isinstance(other, UWQuantity):
            return other.__truediv__(self)

        # Scalar / expression - units become inverted
        if isinstance(other, (int, float)):
            self_units = self.units
            # Only handle if units is a Pint unit object (not None or string)
            if self_units is not None and hasattr(self_units, 'dimensionality'):
                from ..scaling import units as ureg
                combined_units = (1 / self_units).units
                result_value = other / self.value
                return UWexpression(
                    f"({other}/{self.name})",
                    UWQuantity(result_value, combined_units),
                    _unique_name_generation=True
                )

        return Symbol.__rtruediv__(self, other)

    def __add__(self, other):
        """Addition - handle UWQuantity and UWexpression with unit conversion."""
        from .quantities import UWQuantity

        # Handle UWexpression + UWexpression
        if isinstance(other, UWexpression):
            # TRANSPARENT CONTAINER: If self.sym is UWQuantity, use proper Pint arithmetic
            if isinstance(self._sym, UWQuantity) and isinstance(other._sym, UWQuantity):
                # Both contain UWQuantity - use proper unit-aware addition
                # Pint will handle 10cm + 1m = 110cm automatically
                result_qty = self._sym + other._sym
                return UWexpression(
                    f"({self.name}+{other.name})",
                    result_qty,  # Pass full UWQuantity - Transparent Container stores it
                    _unique_name_generation=True,
                )
            elif isinstance(self._sym, UWQuantity):
                # self has units, other doesn't - convert other to UWQuantity
                other_qty = UWQuantity(other.value, self.units) if self.units else other.value
                if isinstance(other_qty, UWQuantity):
                    result_qty = self._sym + other_qty
                    return UWexpression(
                        f"({self.name}+{other.name})",
                        result_qty,
                        _unique_name_generation=True,
                    )
                else:
                    result_value = self.value + other.value
                    return UWexpression(
                        f"({self.name}+{other.name})",
                        result_value,
                        _unique_name_generation=True,
                        units=self.units
                    )
            else:
                # Neither has UWQuantity in _sym - use simple value arithmetic
                result_value = self.value + other.value
                return UWexpression(
                    f"({self.name}+{other.name})",
                    result_value,
                    _unique_name_generation=True,
                    units=self.units or other.units
                )

        # Handle UWexpression + UWQuantity
        if isinstance(other, UWQuantity):
            # TRANSPARENT CONTAINER: If self.sym is UWQuantity, use proper Pint arithmetic
            if isinstance(self._sym, UWQuantity):
                # Both are UWQuantity - Pint handles unit conversion
                result_qty = self._sym + other
                return UWexpression(
                    f"({self.name}+qty)",
                    result_qty,
                    _unique_name_generation=True,
                )
            else:
                # self.sym is not UWQuantity - convert and add
                if self.units is not None:
                    self_qty = UWQuantity(self.value, self.units)
                    result_qty = self_qty + other
                    return UWexpression(
                        f"({self.name}+qty)",
                        result_qty,
                        _unique_name_generation=True,
                    )
                else:
                    # No units on self - just add values
                    result_value = self.value + other.value
                    return UWexpression(
                        f"({self.name}+qty)",
                        result_value,
                        _unique_name_generation=True,
                        units=other.units
                    )
        return Symbol.__add__(self, other)

    def __radd__(self, other):
        """Right addition - handle UWQuantity specially."""
        from .quantities import UWQuantity
        if isinstance(other, UWQuantity):
            return self.__add__(other)  # Addition is commutative
        return Symbol.__radd__(self, other)

    def __sub__(self, other):
        """Subtraction - LAZY EVALUATION pattern.

        When subtracting UWexpressions, we preserve both symbols in the tree
        rather than doing eager arithmetic. This allows unwrap_for_evaluate()
        to substitute the correct nondimensional values later.

        The key insight is that if one operand is a coordinate (no units) and
        the other has units, we CANNOT do the subtraction eagerly because:
        - Coordinates are already in ND form (from mesh scaling)
        - Unit-bearing quantities need to be nondimensionalized by .data

        By keeping both symbols, unwrap_for_evaluate can process each one
        correctly according to its type.
        """
        from .quantities import UWQuantity

        # Handle UWexpression - UWexpression: LAZY EVALUATION
        # Keep both symbols in the tree - don't do eager arithmetic
        if isinstance(other, UWexpression):
            # Delegate to SymPy Symbol subtraction - preserves both symbols
            return Symbol.__sub__(self, other)

        # Handle UWexpression - UWQuantity: Wrap in UWexpression first (LAZY)
        if isinstance(other, UWQuantity):
            # Wrap the UWQuantity in a UWexpression to preserve it as a symbol
            # This allows unwrap_for_evaluate to find and nondimensionalize it
            if other._pint_qty is not None:
                latex_name = f"{other._pint_qty:~L}"
            else:
                latex_name = str(other.value)
            wrapped_other = UWexpression(
                latex_name,
                other,  # Store the full UWQuantity - Transparent Container
                _unique_name_generation=True
            )
            return Symbol.__sub__(self, wrapped_other)

        return Symbol.__sub__(self, other)

    def __rsub__(self, other):
        """Right subtraction - handle UWQuantity specially."""
        from .quantities import UWQuantity
        if isinstance(other, UWQuantity):
            # UWQuantity - UWexpression → UWexpression
            # Convert self to other's units (other is the "base" unit here)
            self_units = self.units
            other_units = other.units

            if self_units is not None and other_units is not None:
                # Convert self's value to other's units for correct subtraction
                # other - self: result is in other's units
                try:
                    from ..scaling import units as ureg
                    self_in_other_units = (self.value * self_units).to(other_units).magnitude
                except Exception:
                    self_in_other_units = self.value
                result_value = other.value - self_in_other_units
                result_units = other_units
            else:
                # Use self.value (not self.sym) for arithmetic
                result_value = other.value - self.value
                result_units = self_units or other_units

            return UWexpression(
                f"(qty-{self.name})",
                result_value,
                _unique_name_generation=True,
                units=result_units
            )
        return Symbol.__rsub__(self, other)

    def __pow__(self, other):
        """Power - delegate to Symbol."""
        return Symbol.__pow__(self, other)

    def __rpow__(self, other):
        """Right power - delegate to Symbol."""
        return Symbol.__rpow__(self, other)

    def __neg__(self):
        """Negation - delegate to Symbol."""
        return Symbol.__neg__(self)

    # =========================================================================
    # Display
    # =========================================================================

    def __repr__(self):
        """
        User-friendly representation showing value with units.

        For expressions with units, shows: value [units]
        For expressions with symbolic content, shows: name = symbolic_expr
        For named expressions with simple values, shows: name = value [units]
        """
        units = self.units
        value = self.value

        # Check if this is a "named" expression (user-defined name vs auto-generated)
        is_named = (
            hasattr(self, '_given_name') and
            self._given_name is not None and
            not self._given_name.startswith('(')  # Auto-generated names start with (
        )

        # Format the value part
        if units is not None:
            # Has units - show value with units
            value_str = f"{value} [{units}]"
        elif self._sym is not None and isinstance(self._sym, sympy.Basic) and not self._sym.is_number:
            # Symbolic expression (not just a number)
            value_str = str(self._sym)
        else:
            # Plain numeric value
            value_str = str(value) if value is not None else str(self.name)

        # For named expressions, show "name = value"
        if is_named and self._given_name != value_str:
            return f"{self._given_name} = {value_str}"
        else:
            return value_str

    def __str__(self):
        """String representation showing value with units if available."""
        units = self.units
        if units is not None:
            return f"{self.value} [{units}]"
        elif self._sym is not None:
            return str(self._sym)
        return str(self.name)

    def _repr_latex_(self):
        """
        LaTeX representation for Jupyter notebooks.

        Jupyter prioritizes _repr_latex_ over __repr__, so we override
        SymPy's default to show units.
        """
        units = self.units
        value = self.value

        # Check if this is a "named" expression (user-defined name vs auto-generated)
        is_named = (
            hasattr(self, '_given_name') and
            self._given_name is not None and
            not self._given_name.startswith('(')
        )

        # Format value for LaTeX
        if isinstance(value, float):
            # Use scientific notation for very small/large numbers
            if value != 0 and (abs(value) < 0.01 or abs(value) >= 10000):
                value_latex = f"{value:.2e}".replace('e', r' \times 10^{') + '}'
            else:
                value_latex = str(value)
        else:
            value_latex = str(value)

        # Format units for LaTeX (Pint units have LaTeX-compatible format)
        if units is not None:
            units_str = str(units).replace('**', '^').replace('*', r' \cdot ')
            value_with_units = f"{value_latex} \\; \\mathrm{{{units_str}}}"
        else:
            value_with_units = value_latex

        # For named expressions, show name = value [units]
        if is_named:
            # Use the LaTeX name if provided (e.g., r"\alpha")
            name_latex = self._given_name
            # Clean up for LaTeX if needed
            if not name_latex.startswith('\\'):
                name_latex = f"\\mathrm{{{name_latex}}}"
            return f"${name_latex} = {value_with_units}$"
        else:
            return f"${value_with_units}$"

    def _repr_html_(self):
        """
        HTML representation for Jupyter notebooks (fallback if LaTeX not available).
        """
        units = self.units
        value = self.value

        is_named = (
            hasattr(self, '_given_name') and
            self._given_name is not None and
            not self._given_name.startswith('(')
        )

        if units is not None:
            value_str = f"{value} [{units}]"
        else:
            value_str = str(value)

        if is_named:
            return f"<b>{self._given_name}</b> = {value_str}"
        else:
            return value_str

    def _repr_png_(self):
        """
        Disable PNG rendering to ensure _repr_latex_ is used.

        SymPy's init_printing() may enable PNG rendering which bypasses
        our custom _repr_latex_. By returning None, we force Jupyter to
        fall back to text/latex format.
        """
        return None

    def _repr_svg_(self):
        """Disable SVG rendering to ensure _repr_latex_ is used."""
        return None

    def _repr_mimebundle_(self, **kwargs):
        """
        MIME bundle for Jupyter display - highest priority representation.

        This method has ABSOLUTE HIGHEST PRIORITY in Jupyter's display system.
        It overrides ANY type-based formatters (including SymPy's init_printing()).

        Why this is needed:
        - SymPy's init_printing() registers formatters for sympy.Basic types
        - UWexpression inherits from sympy.Symbol (a sympy.Basic subclass)
        - Without this, SymPy's formatter renders UWexpression as raw symbols
        - _repr_mimebundle_ cannot be overridden by type formatters

        Returns dict of MIME type → content for display.
        """
        # Get our custom LaTeX representation
        latex = self._repr_latex_()

        # Also provide plain text fallback
        text = repr(self)

        return {
            'text/latex': latex,
            'text/plain': text,
        }

    def _ipython_display_(self):
        """
        IPython/Jupyter display hook - ABSOLUTE highest priority.

        This method OVERRIDES MathematicalMixin._ipython_display_ to show
        our custom representation with units instead of raw SymPy symbols.

        Why this override is needed:
        - MathematicalMixin._ipython_display_ calls display(Math(latex(sym)))
        - This shows only the symbol name without units
        - We want to show value + units for UWexpressions
        """
        try:
            from IPython.display import display, Latex

            # Use our custom LaTeX representation with units
            latex_str = self._repr_latex_()
            display(Latex(latex_str))
        except ImportError:
            # IPython not available - silent fallback
            pass


# ============================================================================
# UWDerivativeExpression - Placeholder for derivative expressions
# ============================================================================

class UWDerivativeExpression(UWexpression):
    """
    Expression representing a derivative that can be evaluated lazily.

    This is a placeholder - the full implementation should be in the old file
    if needed.
    """

    def __init__(self, expr, *args, **kwargs):
        super().__init__("derivative", sym=expr, **kwargs)
        self._expr = expr
        self._args = args

    def doit(self):
        """Evaluate the derivative."""
        result = self._expr
        for arg in self._args:
            result = result.diff(arg)
        return result


# ============================================================================
# Helper function for finding mesh variables in expressions
# ============================================================================

def mesh_vars_in_expression(expr):
    """
    Find all mesh variables used in an expression and verify they use the same mesh.

    Returns:
        tuple: (mesh, set of UnderworldAppliedFunction objects)
    """
    varfns = set()

    def unpack_var_fns(exp):
        if isinstance(exp, uw.function._function.UnderworldAppliedFunctionDeriv):
            raise RuntimeError(
                "Derivative functions are not handled in evaluations, "
                "a projection should be used first to create a mesh Variable."
            )

        isUW = isinstance(exp, uw.function._function.UnderworldAppliedFunction)
        isMatrix = isinstance(exp, sympy.Matrix)

        if isUW:
            varfns.add(exp)
            if exp.args != exp.meshvar().mesh.r:
                raise RuntimeError(
                    f"Mesh Variable functions can only be evaluated as functions of '{exp.meshvar().mesh.r}'.\n"
                    f"However, mesh variable '{exp.meshvar().name}' appears to take the argument {exp.args}."
                )
        elif isMatrix:
            for sub_exp in exp:
                if isinstance(sub_exp, uw.function._function.UnderworldAppliedFunction):
                    varfns.add(sub_exp)
                else:
                    for arg in sub_exp.args:
                        unpack_var_fns(arg)
        else:
            for arg in exp.args:
                unpack_var_fns(arg)

        return

    unpack_var_fns(expr)

    # Check the same mesh is used for all mesh variables
    mesh = None
    for varfn in varfns:
        if mesh is None:
            mesh = varfn.meshvar().mesh
        else:
            if mesh != varfn.meshvar().mesh:
                raise RuntimeError(
                    "In this expression there are functions defined on different meshes. "
                    "This is not supported"
                )

    return mesh, varfns


# ============================================================================
# Backward Compatibility Aliases
# ============================================================================

expression = UWexpression
