import sympy
from sympy import Symbol, simplify, Number
import underworld3 as uw
from underworld3.utilities._api_tools import uw_object
from underworld3.utilities.mathematical_mixin import MathematicalMixin
from underworld3.discretisation import MeshVariable
from .quantities import UWQuantity


def _substitute_all_once(fn, keep_constants=True, return_self=True):
    import underworld3

    # Handle UWQuantity objects directly
    if isinstance(fn, underworld3.function.UWQuantity):
        import os
        debug = os.environ.get('UW_DEBUG_UNWRAP', False)
        if debug:
            print(f"[_substitute_all_once] UWQuantity detected: {fn}")
            print(f"  has_units: {fn.has_units}")
            print(f"  scaling_active: {underworld3._is_scaling_active()}")
        # If scaling is active and quantity has units, non-dimensionalize it
        if underworld3._is_scaling_active() and fn.has_units:
            try:
                nondim = underworld3.non_dimensionalise(fn)
                if debug:
                    print(f"  non_dimensionalised: {nondim}")
                if hasattr(nondim, '_sym'):
                    result = nondim._sym
                elif hasattr(nondim, 'value'):
                    # Convert to SymPy so downstream code can use .atoms(), .subs(), etc.
                    result = sympy.sympify(nondim.value)
                if debug:
                    print(f"  returning: {result}")
                return result
            except Exception as e:
                if debug:
                    print(f"  ERROR: {e}")
                pass
        # Otherwise return the symbolic/numeric value
        if hasattr(fn, '_sym'):
            return fn._sym
        elif hasattr(fn, 'value'):
            # Convert to SymPy for consistency
            return sympy.sympify(fn.value)
        else:
            return fn

    if keep_constants and return_self and is_constant_expr(fn):
        if isinstance(fn, UWexpression):
            return fn.sym
        else:
            return fn

    if isinstance(fn, UWexpression):
        expr = fn.sym
    else:
        expr = fn

    for atom in extract_expressions_and_functions(fn):
        if isinstance(atom, UWexpression):
            if keep_constants and is_constant_expr(atom):
                continue
            else:
                # Check if scaling is active - if yes, use non-dimensional value
                if underworld3._is_scaling_active() and hasattr(atom, '_pint_qty') and atom._pint_qty is not None:
                    # Atom has units - compute non-dimensional value
                    try:
                        nondim_atom = underworld3.non_dimensionalise(atom)
                        # non_dimensionalise returns UWQuantity with dimensionless value
                        if hasattr(nondim_atom, 'value'):
                            expr = expr.subs(atom, nondim_atom.value)
                        else:
                            # Fallback to .sym if non-dimensionalisation didn't work
                            expr = expr.subs(atom, atom.sym)
                    except:
                        # If non-dimensionalisation fails, use dimensional value
                        expr = expr.subs(atom, atom.sym)
                else:
                    # No scaling or no units - use dimensional value
                    expr = expr.subs(atom, atom.sym)

    return expr


def _substitute_one_expr(fn, sub_expr, keep_constants=True, return_self=True):
    expr = fn

    if keep_constants and return_self and is_constant_expr(fn):
        if isinstance(fn, UWexpression):
            return fn.sym
        else:
            return fn

    for atom in fn.atoms():
        if atom is sub_expr:
            if keep_constants and isinstance(atom.sym, (float, int, Number)):
                continue
            else:
                expr = expr.subs(atom, atom.sym)

    return expr


# Not sure the best name for this
def substitute(fn, keep_constants=True, return_self=True):
    """Alias for _unwrap_for_compilation() - used internally for substitution."""
    return _unwrap_for_compilation(fn, keep_constants, return_self)


def _unwrap_expressions(fn, keep_constants=True, return_self=True):
    expr = fn
    expr_s = _substitute_all_once(expr, keep_constants, return_self)

    while expr is not expr_s:
        expr = expr_s
        expr_s = _substitute_all_once(expr, keep_constants, return_self)

    return expr


def _unwrap_for_compilation(fn, keep_constants=True, return_self=True):
    """
    INTERNAL ONLY: Unwrap UW expressions to pure SymPy for JIT compilation.

    This function recursively flattens nested UW expressions, applies non-dimensional
    scaling transformations, and strips ALL metadata including units. It is designed
    exclusively for pre-compilation use in solvers.

    DO NOT USE THIS IN USER-FACING CODE. Use `expand()` instead for user inspection.

    Args:
        fn: Expression to unwrap
        keep_constants: Whether to preserve constants
        return_self: Whether to return self for constant expressions

    Returns:
        Pure SymPy expression with scale factors applied if scaling context is active
        (all UW wrappers and metadata stripped for compilation)
    """
    import underworld3 as uw

    # Handle UWQuantity specially - non-dimensionalize if scaling is active
    if isinstance(fn, UWQuantity):
        import os
        debug = os.environ.get('UW_DEBUG_UNWRAP', False)
        if debug:
            print(f"[_unwrap_for_compilation] UWQuantity: {fn}")
            print(f"  scaling_active: {uw._is_scaling_active()}")
            print(f"  has_units: {fn.has_units}")
        if uw._is_scaling_active() and fn.has_units:
            nondim = uw.non_dimensionalise(fn)
            if debug:
                print(f"  nondim: {nondim}")
            if hasattr(nondim, 'value'):
                result = sympy.sympify(nondim.value)
                if debug:
                    print(f"  returning value: {result}")
                return result
            elif hasattr(nondim, '_sym'):
                if debug:
                    print(f"  returning _sym: {nondim._sym}")
                return nondim._sym
        # Otherwise just return the value
        if hasattr(fn, '_sym'):
            return fn._sym
        elif hasattr(fn, 'value'):
            result = sympy.sympify(fn.value)
            if debug:
                print(f"  returning plain value: {result}")
            return result
        return fn

    # Handle UWDerivativeExpression specially - evaluate it first
    if isinstance(fn, UWDerivativeExpression):
        result = fn.doit()
    elif isinstance(fn, sympy.Matrix):
        f = lambda x: _unwrap_expressions(
            x, keep_constants=keep_constants, return_self=return_self
        )
        result = fn.applyfunc(f)
    else:
        result = _unwrap_expressions(
            fn, keep_constants=keep_constants, return_self=return_self
        )

    # Apply scaling if context is active
    if uw._is_scaling_active():
        result = _apply_scaling_to_unwrapped(result)

    return result


def unwrap(fn, keep_constants=True, return_self=True):
    """
    DEPRECATED: Use `expand()` for user-facing expansion or call internal
    `_unwrap_for_compilation()` directly in solver code.

    This function is kept for backward compatibility but will be removed in a future version.
    """
    import warnings
    warnings.warn(
        "unwrap() is deprecated and will be removed. "
        "Use expand() for user inspection or _unwrap_for_compilation() for solver code.",
        DeprecationWarning,
        stacklevel=2
    )
    return _unwrap_for_compilation(fn, keep_constants, return_self)


def expand(expr):
    """
    Recursively expand UW expression for user inspection while preserving units.

    This function is the user-facing counterpart to _unwrap_for_compilation().
    It recursively expands nested UW expressions to reveal their SymPy structure
    while preserving unit metadata that would be stripped by compilation unwrapping.

    Unlike _unwrap_for_compilation(), this function:
    - Does NOT apply non-dimensional scaling
    - DOES preserve unit information
    - Returns a unit-aware expression wrapper suitable for display

    Args:
        expr: UW expression to expand (UWexpression, MeshVariable, etc.)

    Returns:
        Expanded expression with units preserved (same type as input when possible,
        or plain SymPy if no units)

    Example:
        >>> T = uw.discretisation.MeshVariable("T", mesh, 1, units="kelvin")
        >>> Ra = uw.discretisation.MeshVariable("Ra", mesh, 1)
        >>> buoyancy = Ra * T
        >>> expanded = uw.expand(buoyancy)  # Shows Ra(x,y) * T(x,y) with units preserved
    """
    import sympy

    # Extract units before expansion
    units = getattr(expr, 'units', None)

    # Get the SymPy expression
    if hasattr(expr, 'sym'):
        sym_expr = expr.sym
    elif isinstance(expr, sympy.Basic):
        sym_expr = expr
    else:
        # Try sympify
        sym_expr = sympy.sympify(expr)

    # Recursively expand .sym attributes (but don't apply scaling)
    expanded_sym = _unwrap_expressions(sym_expr, keep_constants=True, return_self=True)

    # If we have units, try to return a unit-aware wrapper
    if units is not None:
        try:
            # Try to use the hierarchical unit-aware architecture if available
            from underworld3.expression import LazyExpression
            return LazyExpression(expanded_sym, units)
        except ImportError:
            # Fallback: just return SymPy with a note
            pass

    # Return pure SymPy (no units to preserve)
    return expanded_sym


def _apply_scaling_to_unwrapped(expr):
    """
    Apply non-dimensional scaling to an unwrapped SymPy expression.

    This function finds all variable symbols in the expression and scales them
    by dividing by their scaling_coefficient (reference scale).

    Non-dimensionalization: T(x,y) → T(x,y) / T_ref

    Args:
        expr: SymPy expression (potentially with UW variable symbols)

    Returns:
        SymPy expression with non-dimensional scaling applied
    """
    import underworld3 as uw
    import sympy

    # SURGICAL FIX (2025-11-14): Disable variable scaling during JIT compilation
    # PETSc stores variables in non-dimensional form already, so variable symbols
    # like p(x,y) return ND values. Scaling them again creates double-ND bug.
    # Only constants (UWQuantity) need ND conversion, which happens earlier in
    # _unwrap_for_compilation() at lines 142-171.
    #
    # See: User bug report showing pressure coefficient 0.000315576 instead of 1.0
    # Test: Auto-ND vs manual-ND should produce identical expressions for PETSc
    return expr

    try:
        # Get the model registry to find variables
        model = uw.get_default_model()
        substitutions = {}

        # Find all function symbols in the expression
        # These represent UW variables like T(x,y), v_0(x,y), etc.
        if hasattr(expr, 'atoms'):
            function_symbols = expr.atoms(sympy.Function)
        else:
            function_symbols = set()

        # For each variable in the model, check if it has scaling
        for var_name, variable in model._variables.items():
            # Check for scaling_coefficient (from dimensionality_mixin)
            if not hasattr(variable, 'scaling_coefficient'):
                continue

            coeff = variable.scaling_coefficient

            # Only scale if coefficient != 1.0 (actually has scaling)
            if coeff == 1.0:
                continue

            # Get the variable's symbol
            if hasattr(variable, '_base_var'):
                # For enhanced variables, get the base variable's symbol
                var_sym = variable._base_var.sym
            else:
                var_sym = getattr(variable, 'sym', None)

            if var_sym is None:
                continue

            # Get all function symbols from the variable's symbol
            if hasattr(var_sym, 'atoms'):
                var_function_symbols = var_sym.atoms(sympy.Function)
            else:
                continue

            # Find matching symbols and create substitutions
            # Substitution: T(x,y) → T(x,y) / T_ref (non-dimensionalize)
            for func_symbol in function_symbols:
                for var_func_symbol in var_function_symbols:
                    if str(func_symbol) == str(var_func_symbol):
                        # DIVIDE by scaling coefficient for non-dimensionalization
                        substitutions[func_symbol] = func_symbol / coeff

        # Apply all substitutions
        if substitutions:
            return expr.subs(substitutions)
        else:
            return expr

    except Exception as e:
        import warnings
        warnings.warn(f"Could not apply non-dimensional scaling to expression: {e}")
        return expr


def unwrap_for_evaluate(expr, scaling_active=None):
    """
    Unwrap expression for evaluate/lambdify path with proper unit handling.

    This is specifically designed for the evaluate pathway where:
    - MeshVariables reference PETSc data (already non-dimensional)
    - Constants (UWQuantity) need non-dimensionalization if scaling active
    - Expression is passed to lambdify, not JIT compiled

    Process:
    1. Compute expression dimensionality (for later re-dimensionalization)
    2. Flatten expression to unit-aware atoms
    3. Non-dimensionalize constants (NOT variables - they're already ND in PETSc)
    4. Return flattened expression and dimensionality

    Args:
        expr: Expression to unwrap (any UW expression type)
        scaling_active: Override ND scaling check (default: use global state)

    Returns:
        tuple: (unwrapped_expr, result_dimensionality)
            - unwrapped_expr: SymPy expression ready for lambdify
            - result_dimensionality: Dict for re-dimensionalization (or None)

    Example:
        >>> # Constant: gets non-dimensionalized
        >>> expr = uw.quantity(0.0001, 'cm/yr')
        >>> unwrapped, dims = unwrap_for_evaluate(expr)

        >>> # Variable: left as-is (PETSc has ND data)
        >>> expr = velocity[0]
        >>> unwrapped, dims = unwrap_for_evaluate(expr)

        >>> # Mixed: constants scaled, variables left alone
        >>> expr = 2 * velocity[0] + uw.quantity(0.0001, 'cm/yr')
        >>> unwrapped, dims = unwrap_for_evaluate(expr)
    """
    import underworld3 as uw
    import sympy
    from .quantities import UWQuantity
    from underworld3.units import get_units, get_dimensionality

    # Step 1: Get expression dimensionality for later re-dimensionalization
    result_units = get_units(expr)
    if result_units is not None:
        try:
            result_dimensionality = get_dimensionality(expr)
        except:
            result_dimensionality = None
    else:
        result_dimensionality = None

    # Determine if we should non-dimensionalize
    if scaling_active is None:
        scaling_active = uw.is_nondimensional_scaling_active()

    model = uw.get_default_model()
    should_scale = scaling_active and model.has_units()

    # Step 2 & 3: Extract SymPy expression and process atoms
    # First, get the SymPy core from various wrapper types
    from underworld3.expression.unit_aware_expression import UnitAwareExpression

    if isinstance(expr, UWQuantity):
        # UWQuantity: needs special handling for non-dimensionalization
        if should_scale:
            # Non-dimensionalize the constant
            nondim_qty = uw.non_dimensionalise(expr)
            if hasattr(nondim_qty, 'value'):
                return sympy.sympify(nondim_qty.value), result_dimensionality
            elif hasattr(nondim_qty, '_sym'):
                return nondim_qty._sym, result_dimensionality
        # No scaling: just return the value
        if hasattr(expr, 'value'):
            return sympy.sympify(expr.value), result_dimensionality
        elif hasattr(expr, '_sym'):
            return expr._sym, result_dimensionality

    elif isinstance(expr, UnitAwareExpression):
        # UnitAwareExpression: get underlying SymPy
        sym_expr = expr._sym if hasattr(expr, '_sym') else expr
    elif hasattr(expr, 'sym'):
        # MeshVariable or similar: get .sym
        sym_expr = expr.sym
    else:
        # Already SymPy or plain
        sym_expr = expr

    # Step 3: Process the expression - find and non-dimensionalize constants
    # but leave variable function symbols untouched
    if should_scale and hasattr(sym_expr, 'atoms'):
        # Find all UWQuantity atoms in the expression
        # We need to substitute them with their non-dimensional values
        substitutions = {}

        # Check for UWexpression atoms that wrap constants
        for atom in sym_expr.atoms():
            # Look for UWexpression wrapper atoms
            if hasattr(atom, '_pint_qty') and atom._pint_qty is not None:
                # This is a UW constant with units - non-dimensionalize it
                try:
                    nondim_atom = uw.non_dimensionalise(atom)
                    if hasattr(nondim_atom, 'value'):
                        substitutions[atom] = nondim_atom.value
                except:
                    pass  # Skip if non-dimensionalization fails

        # Apply substitutions
        if substitutions:
            sym_expr = sym_expr.subs(substitutions)

    # Step 4: Return unwrapped expression and dimensionality
    # Note: Variable symbols like T(x,y) are left as-is
    # They'll be interpolated from PETSc (which stores ND values)
    return sym_expr, result_dimensionality


def substitute_expr(fn, sub_expr, keep_constants=True, return_self=True):
    expr = fn
    expr_s = _substitute_one_expr(expr, sub_expr, keep_constants)

    while expr is not expr_s:
        expr = expr_s
        expr_s = _substitute_one_expr(expr, sub_expr, keep_constants)
    return expr


def is_constant_expr(fn):

    deps = extract_expressions_and_functions(fn)

    # bool(deps) -> True if not the empty set
    if bool(deps):
        return False
    else:
        return True


def extract_expressions(fn):
    import underworld3

    if isinstance(fn, underworld3.function.expression):
        fn = fn.sym

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

    import underworld3

    if isinstance(fn, underworld3.function.expression):
        fn = fn.sym

    # Handle UWQuantity objects - they don't have atoms() method
    if isinstance(fn, underworld3.function.UWQuantity):
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


class UWexpression(MathematicalMixin, UWQuantity, uw_object, Symbol):
    """
    underworld `expressions` are sympy symbols with attached
                numeric/expression values that are substituted into an underworld function
                before evaluation. In sympy expressions, the symbol form is shown.

    ```{python}
        alpha = UWexpression(
                        r'\\alpha',
                        sym=3.0e-5,
                        description="thermal expansivity"
                            )
        print(alpha.sym)
        print(alpha.description)
    ```

    """

    _expr_count = 0
    _expr_names = {}
    _ephemeral_expr_names = {}  # Weak references to ephemeral expressions

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

        ## if the expression already exists, update it and return (natural Python behavior)

        if name in UWexpression._expr_names.keys() and _unique_name_generation == False:
            # Preserve object identity, update internal state (lazy evaluation pattern)
            # This is Pythonic: like updating an attribute rather than creating a new object
            existing = UWexpression._expr_names[name]

            # Update sym value if provided in kwargs (will be set in __init__)
            # The __init__ will handle updating the sym value, we just return existing object
            return existing

        # Check both persistent and ephemeral dicts for existing expressions
        name_exists_persistent = name in UWexpression._expr_names
        name_exists_ephemeral = name in UWexpression._ephemeral_expr_names

        if (name_exists_persistent or name_exists_ephemeral) and _unique_name_generation == True:
            # Make hspace 100x smaller (nearly invisible for ephemeral expressions)
            invisible = rf"\hspace{{ {instance_no/10000}pt }}"
            unique_name = f"{{ {name} {invisible} }}"
        else:
            unique_name = name

        obj = Symbol.__new__(cls, unique_name)
        obj._instance_no = instance_no
        obj._unique_name = unique_name
        obj._given_name = name
        obj._is_ephemeral = _unique_name_generation

        # Store ephemeral expressions with weak references for garbage collection
        if _unique_name_generation:
            # Use weakref.ref with callback to clean up the dict entry
            def cleanup_callback(ref):
                # Remove from ephemeral dict when garbage collected
                if unique_name in UWexpression._ephemeral_expr_names:
                    del UWexpression._ephemeral_expr_names[unique_name]

            try:
                UWexpression._ephemeral_expr_names[unique_name] = weakref.ref(obj, cleanup_callback)
            except TypeError:
                # If weak references aren't supported (shouldn't happen for Symbols),
                # fall back to strong reference
                UWexpression._expr_names[unique_name] = obj
        else:
            # Persistent expressions use strong references
            UWexpression._expr_names[unique_name] = obj

        UWexpression._expr_count += 1

        return obj

    def __init__(
        self,
        name,
        sym=None,
        description="No description provided",
        value=None,
        units=None,
        **kwargs,
    ):
        # Handle legacy 'value' parameter
        if value is not None and sym is None:
            import warnings
            warnings.warn(
                message=f"DEPRECATION warning, don't use 'value' attribute for expression: {value}, please use 'sym' attribute"
            )
            sym = value

        if value is not None and sym is not None:
            raise ValueError(
                "Both 'sym' and 'value' attributes are provided, please use one"
            )

        # Determine the value and units for UWQuantity initialization
        # Handle UWQuantity as sym parameter (the beautiful symmetry!)
        if isinstance(sym, UWQuantity):
            if units is not None:
                # Convert quantity to match expression's target units
                if sym.has_units:
                    converted_qty = sym.to(units)
                    sym_value = converted_qty.value
                    final_units = units
                else:
                    # Dimensionless quantity, just use the value
                    sym_value = sym.value
                    final_units = units
            else:
                # Use quantity's units directly
                sym_value = sym.value
                final_units = sym.units if sym.has_units else None
        else:
            # Traditional initialization - use provided sym and units
            sym_value = sym
            final_units = units

        # Initialize UWQuantity parent class with extracted value and units
        UWQuantity.__init__(self, value=sym_value if sym_value is not None else 0, units=final_units)

        # UWexpression-specific attributes
        self.symbol = self._given_name
        self.sym = sym_value  # Accept anything, sympify is opinionated
        self.description = description

        # this is not being honoured by sympy Symbol so do it by hand
        self._uw_id = uw_object._obj_count
        uw_object._obj_count += 1

        return

    def __repr__(self):
        """
        Override MathematicalMixin.__repr__ to return Symbol representation.

        This is critical for SymPy's internal sympify operations to work correctly.
        When SymPy performs operations like z/r, it internally calls sympify in strict mode,
        which needs to be able to parse the repr() output. Since UWexpression inherits
        from Symbol, we return the symbol name instead of the symbolic expression.
        """
        return str(self.name)

    def _sympy_(self):
        """
        Return the Symbol itself for deferred evaluation.

        Note: Uses _sympy_() protocol (not _sympify_()) for SymPy 1.14+ compatibility.
        This is required for proper symbolic algebra in strict mode (matrix operations).

        CRITICAL CHANGE (2025-10-28): Changed from returning self._sym to returning self.
        This is required for proper symbolic algebra with expressions.

        Why return self:
        - Preserves UWexpression as Symbol in SymPy expression trees
        - Enables symbolic multiplication: alpha * kappa → \alpha*\kappa
        - Prevents premature evaluation to numeric values
        - Works with Symbol's natural arithmetic operators

        Previous implementation returned self._sym which broke:
        - Expression multiplication (Ra * T failed with TypeError)
        - Symbolic preservation (alpha.sym returned 0.00003 instead of \alpha)
        """
        return self  # NOT self._sym!

    def __bool__(self):
        """
        Override boolean evaluation to prevent __len__ calls.

        UWexpression objects should always evaluate to True for boolean
        contexts, just like regular SymPy Symbol objects. This prevents
        SymPy from calling __len__ during boolean evaluation.
        """
        return True

    def __hash__(self):
        """
        Make UWexpression hashable by delegating to Symbol's hash.

        Required for SymPy's caching and algebraic operations. Since UWexpression
        inherits from multiple classes (some of which may define __eq__), Python
        requires explicit __hash__ to make the class hashable.
        """
        return Symbol.__hash__(self)

    # ===================================================================
    # Delegate SymPy assumption properties to wrapped expression
    # This is CRITICAL for lazy evaluation with Min, Max, Piecewise, etc.
    # ===================================================================

    @property
    def is_comparable(self):
        """Delegate comparability check to wrapped expression."""
        if self._sym is not None and hasattr(self._sym, 'is_comparable'):
            return self._sym.is_comparable
        return True  # Default to comparable

    @property
    def is_number(self):
        """Delegate number check to wrapped expression."""
        if self._sym is not None and hasattr(self._sym, 'is_number'):
            return self._sym.is_number
        return False  # Symbol default

    @property
    def is_extended_real(self):
        """Delegate extended_real check to wrapped expression."""
        if self._sym is not None and hasattr(self._sym, 'is_extended_real'):
            return self._sym.is_extended_real
        return None  # Unknown

    @property
    def is_positive(self):
        """Delegate positivity check to wrapped expression."""
        if self._sym is not None and hasattr(self._sym, 'is_positive'):
            return self._sym.is_positive
        return None  # Unknown

    @property
    def is_negative(self):
        """Delegate negativity check to wrapped expression."""
        if self._sym is not None and hasattr(self._sym, 'is_negative'):
            return self._sym.is_negative
        return None  # Unknown

    @property
    def is_zero(self):
        """Delegate zero check to wrapped expression."""
        if self._sym is not None and hasattr(self._sym, 'is_zero'):
            return self._sym.is_zero
        return None  # Unknown

    @property
    def is_finite(self):
        """Delegate finite check to wrapped expression."""
        if self._sym is not None and hasattr(self._sym, 'is_finite'):
            return self._sym.is_finite
        return None  # Unknown

    @property
    def is_infinite(self):
        """Delegate infinite check to wrapped expression."""
        if self._sym is not None and hasattr(self._sym, 'is_infinite'):
            return self._sym.is_infinite
        return None  # Unknown

    # ===================================================================
    # Arithmetic method overrides - delegate to Symbol
    # CRITICAL: Bypasses UWQuantity's __mul__ in MRO
    # ===================================================================
    # These overrides ensure that when a UWexpression (which inherits from
    # Symbol) participates in arithmetic operations, we use Symbol's arithmetic
    # instead of UWQuantity's. This prevents unhashable type errors and enables
    # symbolic algebra.

    def __mul__(self, other):
        """Multiply - delegate to Symbol to preserve symbolic expressions."""
        # Special case: If self.sym is a Matrix, use matrix multiplication
        if hasattr(self, '_sym') and isinstance(self._sym, sympy.MatrixBase):
            return self._sym.__mul__(other)

        # Special case: If multiplying by a Matrix, scalar * Matrix element-wise multiplication
        # Convert self to its symbolic value first so SymPy's matrix multiplication works
        # Note: Check for both MatrixBase and MatrixExpr to catch transpose, slices, etc.
        if isinstance(other, (sympy.MatrixBase, sympy.matrices.expressions.MatrixExpr)):
            # For UWexpression * Matrix, use self.sym if available, otherwise self (as Symbol)
            if hasattr(self, '_sym') and self._sym is not None:
                return self._sym * other  # Let SymPy handle it
            else:
                # Fallback to Symbol behavior
                return Symbol.__mul__(self, other)

        return Symbol.__mul__(self, other)

    def __rmul__(self, other):
        """Right multiply - delegate to Symbol."""
        return Symbol.__rmul__(self, other)

    def __truediv__(self, other):
        """Divide - delegate to Symbol."""
        return Symbol.__truediv__(self, other)

    def __rtruediv__(self, other):
        """Right divide - delegate to Symbol."""
        return Symbol.__rtruediv__(self, other)

    def __add__(self, other):
        """Add - delegate to Symbol."""
        return Symbol.__add__(self, other)

    def __radd__(self, other):
        """Right add - delegate to Symbol."""
        return Symbol.__radd__(self, other)

    def __sub__(self, other):
        """Subtract - delegate to Symbol."""
        return Symbol.__sub__(self, other)

    def __rsub__(self, other):
        """Right subtract - delegate to Symbol."""
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

    def copy(self, other):
        if not isinstance(other, UWexpression):
            raise ValueError
        else:
            # Note: sympy symbols are uniquely defined by name and so
            # the uw expressions based on symbols cannot be renamed: only the
            # value can be changed. As a result, copy is just an assignment to
            # self.sym and should be deprecated.

            # Update the symbolic value
            # self.symbol = other.symbol # Can't change this
            self._sym = other._sym
            # self.description = other.description # Shouldn't change this

            # Copy unit metadata if present
            if hasattr(other, '_pint_qty'):
                self._pint_qty = other._pint_qty
            if hasattr(other, '_has_pint_qty'):
                self._has_pint_qty = other._has_pint_qty
            if hasattr(other, '_custom_units'):
                self._custom_units = other._custom_units
            if hasattr(other, '_has_custom_units'):
                self._has_custom_units = other._has_custom_units
            if hasattr(other, '_symbolic_with_units'):
                self._symbolic_with_units = other._symbolic_with_units
            if hasattr(other, '_dimensionality'):
                self._dimensionality = other._dimensionality
            if hasattr(other, '_model_registry'):
                self._model_registry = other._model_registry
            if hasattr(other, '_model_instance'):
                self._model_instance = other._model_instance

        return

    # Matches sympy
    def is_constant(self):
        return is_constant_expr(self)

    # deprecate
    def constant(self):
        return is_constant_expr(self)

    @property
    def expression_number(self):
        """Unique number of the expression instance"""
        return self._expr_count

    @property
    def sym(self):
        return self._sym

    @sym.setter
    def sym(self, new_value):
        if isinstance(new_value, (sympy.Basic, sympy.matrices.MatrixBase)):
            self._sym = new_value
        else:
            self._sym = sympy.sympify(new_value)
        return

    # TODO: DEPRECATION REMOVED
    # The value attribute is inherited from UWQuantity base class
    # Old deprecated value property removed to avoid MRO conflicts

    @property
    def expression(self):
        return unwrap(self)

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, new_description):
        self._description = new_description
        return

    def set_display_name(self, new_latex_name):
        """
        Change the LaTeX display name while preserving the unique SymPy identity.
        
        This allows changing how the expression appears in LaTeX output and string 
        representations without affecting the underlying SymPy symbol identity, which
        must remain unique for proper symbolic computation.
        
        Parameters:
        -----------
        new_latex_name : str
            The new LaTeX name for display purposes (e.g., r"\\eta_0")
            
        Example:
        --------
        >>> viscosity = uw.function.expression(r"\\eta", 1.0, "Viscosity")
        >>> viscosity.set_display_name(r"\\eta_0")  # Now displays as η₀
        """
        self.symbol = new_latex_name
        # Note: We don't change _given_name, _unique_name, or the SymPy Symbol identity
        # This preserves uniqueness while allowing display customization
        return


    def sub_all(self, keep_constants=True):
        return substitute(self, keep_constants=keep_constants)

    def sub_expr(self, expr, keep_constants=True):
        self_s = substitute_expr(self, expr, keep_constants=keep_constants)

        return self_s
    
    def diff(self, *symbols, **kwargs):
        """
        Override diff to handle wrapped expressions properly.
        
        When differentiating a UWexpression, we need to differentiate
        the wrapped symbolic value (.sym), not the expression symbol itself.
        
        This enables natural derivative syntax:
            rho = UWexpression(r'\rho', sym=1000*(1 + 0.01*x))
            drho_dx = rho.diff(x)  # Evaluated derivative
            drho_dx_deferred = rho.diff(x, evaluate=False)  # Deferred derivative
        
        Args:
            *symbols: Variables to differentiate with respect to
            **kwargs: Additional options including:
                - evaluate (bool): If False, return a deferred derivative object
                - Other SymPy diff assumptions
            
        Returns:
            The derivative of the wrapped expression (evaluated or deferred)
        """
        # Check for evaluate flag
        evaluate = kwargs.pop('evaluate', True)
        
        if not evaluate:
            # Return deferred derivative for lazy evaluation
            if len(symbols) != 1:
                raise NotImplementedError("Deferred derivatives only support single variables currently")
            
            diff_variable = symbols[0]
            latex_expr = sympy.latex(self)
            latex_diff_variable = sympy.latex(diff_variable)
            latex = (
                r"\partial \left[" + latex_expr + r"\right] / \partial " + latex_diff_variable
            )
            
            return UWDerivativeExpression(latex, self, diff_variable)
        
        # Evaluated derivative (original implementation)
        if self._sym is not None:
            # Differentiate the wrapped symbolic value
            result = sympy.diff(self._sym, *symbols, **kwargs)
            
            # If the result contains nested UWexpressions, unwrap them
            for atom in result.atoms():
                if isinstance(atom, UWexpression) and atom._sym is not None:
                    result = result.subs(atom, atom.sym)
            
            return result
        else:
            # If no wrapped value, behave like a regular Symbol
            return super().diff(*symbols, **kwargs)

    def dependencies(self, keep_constants=True):
        return extract_expressions(self)

    def all_dependencies(self, keep_constants=True):
        return extract_expressions_and_functions(self)

    def _ipython_display_(self):
        from IPython.display import Latex, Markdown, display

        display(Markdown("$" + self.symbol + "$"))


    def _repr_latex_(self):
        # print("Customised !")
        return rf"$\displaystyle {str(self.symbol)}$"

    def _object_viewer(self, description=True, level=1):
        from IPython.display import Latex, Markdown, display
        import sympy

        level = max(1, level)

        if isinstance(self.sym, (sympy.Basic, sympy.matrices.MatrixBase)):
            latex = self.sym._repr_latex_()
        else:
            latex = sympy.sympify(self.sym)._repr_latex_()

        ## feedback on this instance
        if sympy.sympify(self.sym) is not None:
            display(
                Latex(
                    r"$" + r"\quad" * level + "$" + self._repr_latex_() + "$=$" + latex
                ),
            )
            if description == True:
                display(
                    Markdown(
                        r"$"
                        + r"\quad" * level
                        + "$"
                        + f"**Description:**  {self.description}"
                    ),
                )

        try:
            atoms = self.sym.atoms()
            for atom in atoms:
                if atom is not self.sym:
                    try:
                        atom._object_viewer(description=False, level=level + 1)
                    except AttributeError:
                        pass
        except:
            pass

        return



class UWDerivativeExpression(UWexpression):
    """
    underworld `expressions` are sympy symbols with attached
    numeric/expression values that are substituted into an underworld function
    before evaluation.

    derivative expressions are unevaluated / symbolic derivatives that remain
    symbolic until they need to be evaluated.

    Note - this class would usually be automatically generated by asking for the
    derivative of an expression with `evaluate=False`

    ```{python}
        alpha = UWDerivativeExpression(
                        r'\\alpha',
                        expr=uw_expression,
                        diff_expr=diff_expression,
                        description=rf"\\partial{expr.description}/\\partial{diff_expr.description}"
                            )
        print(alpha.sym)
        print(alpha.description)
    ```

    """

    def __new__(cls, name, *args, **kwargs):
        """
        Create ephemeral derivative expressions with automatic unique naming.

        Derivative expressions are typically created on-the-fly (e.g., temperature.diff(y))
        and used immediately in assignments. They should not trigger uniqueness warnings.
        """
        # Force unique name generation for derivative expressions (ephemeral/anonymous)
        return UWexpression.__new__(cls, name, *args, _unique_name_generation=True, **kwargs)

    def __init__(
        self,
        name,
        expr,
        diff_variable,
        description="derivative of expression provided",
    ):

        self.symbol = self._given_name

        self._sym = expr  # Accept anything, sympify is overly opinionated if we try to `sympify`
        self._diff_variable = diff_variable
        self.description = description

        # this is not being honoured by sympy Symbol so do it by hand
        self._uw_id = uw_object._obj_count
        uw_object._obj_count += 1

        return

    def doit(self):
        """Evaluate the deferred derivative"""
        return uw.function.derivative(self._sym, self.diff_variable)

    @property
    def sym(self):
        """Return the evaluated derivative for the sym property"""
        try:
            return self._sym.sym.diff(self._diff_variable)
        except:
            return self._sym.diff(self._diff_variable)

    @property
    def expr(self):
        """The expression being differentiated"""
        return self._sym

    @property
    def diff_variable(self):
        """The variable with respect to which we're differentiating"""
        return self._diff_variable

    @diff_variable.setter
    def diff_variable(self, value):
        self._diff_variable = value
    
    def diff(self, *symbols, **kwargs):
        """
        Enable chained derivatives on deferred derivative objects.
        
        This allows natural syntax for higher-order derivatives:
            d2f_dx2 = f.diff(x, evaluate=False).diff(x)
            d2f_dxdy = f.diff(x, evaluate=False).diff(y, evaluate=False)
        
        Args:
            *symbols: Variables to differentiate with respect to
            **kwargs: Additional options including evaluate flag
            
        Returns:
            A new derivative expression (evaluated or deferred)
        """
        evaluate = kwargs.pop('evaluate', True)
        
        if not evaluate:
            # Create a nested deferred derivative
            if len(symbols) != 1:
                raise NotImplementedError("Deferred derivatives only support single variables currently")
            
            diff_variable = symbols[0]
            latex_expr = sympy.latex(self)
            latex_diff_variable = sympy.latex(diff_variable)
            latex = (
                r"\partial \left[" + latex_expr + r"\right] / \partial " + latex_diff_variable
            )
            
            # Create a new deferred derivative of this deferred derivative
            return UWDerivativeExpression(latex, self, diff_variable)
        else:
            # Evaluate this derivative first, then differentiate the result
            evaluated = self.doit()
            return sympy.diff(evaluated, *symbols, **kwargs)
    

    # TODO: DEPRECATION REMOVED
    # The value attribute is inherited from UWQuantity base class (via UWexpression)
    # Old deprecated value property removed to avoid MRO conflicts


def mesh_vars_in_expression(
    expr,
):

    varfns = set()

    def unpack_var_fns(exp):

        if isinstance(exp, uw.function._function.UnderworldAppliedFunctionDeriv):
            raise RuntimeError(
                "Derivative functions are not handled in evaluations, a projection should be used first to create a mesh Variable."
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
                    # Recursively search for more functions
                    for arg in sub_exp.args:
                        unpack_var_fns(arg)

        else:
            # Recursively search for more functions
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
                    "In this expression there are functions defined on different meshes. This is not supported"
                )

    return mesh, varfns
