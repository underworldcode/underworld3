"""
Hierarchical Unit-Aware Expression Architecture for Underworld3

This module implements a clean separation of concerns for unit-aware symbolic expressions:
1. Pure SymPy computation (no units)
2. Unit metadata tracking (Pint units)
3. Mathematical operations (with unit updates)
4. Lazy evaluation (deferred computation)
5. Domain objects (user-facing API)

The key insight is that units and computation are kept separate but synchronized,
allowing SymPy to remain pure while still preserving dimensional information.
"""

import sympy
import numpy as np
from typing import Optional, Any, Union, Callable
from underworld3.scaling import units as ureg
from underworld3.function.quantities import quantity
from underworld3 import get_default_model
from underworld3.function import fn_unwrap as unwrap


# ==============================================================================
# Layer 1: Pure SymPy Core (just computation, no units)
# ==============================================================================
# We use standard SymPy - nothing special here


# ==============================================================================
# Layer 2: Unit-Aware Wrapper (tracks units alongside SymPy expression)
# ==============================================================================

class UnitAwareExpression:
    """
    Base class that wraps a SymPy expression with unit metadata.

    Key principle: Keep SymPy and units separate but synchronized.
    This ensures SymPy operations remain pure while units flow through naturally.
    """

    def __init__(self, expr: sympy.Basic, units: Optional[ureg.Unit] = None):
        """
        Initialize with a SymPy expression and optional units.

        Parameters
        ----------
        expr : sympy.Basic
            Pure SymPy expression for computation
        units : pint.Unit, optional
            Unit metadata (None means dimensionless)
        """
        self._expr = expr
        self._units = units

    @property
    def sym(self):
        """Access pure SymPy expression for computation."""
        return self._expr

    @property
    def units(self):
        """
        Get units for this expression.

        This enables proper unit handling for compound expressions like:
        - temperature / velocity[0] → kelvin * second / meter
        - temperature**2 → kelvin**2
        - velocity.dot(velocity) → meter**2 / second**2

        If units were explicitly provided at construction time (from Pint arithmetic),
        those are trusted. Otherwise, units are computed from the SymPy expression structure.

        Returns
        -------
        pint.Unit or None
            Pint Unit object (never string). None for dimensionless quantities.
        """
        # If we have explicitly provided units (from Pint arithmetic), trust them
        # This is critical for expressions like (UWQuantity * UWexpression) where
        # the SymPy structure doesn't preserve full unit information from both operands
        if self._units is not None:
            # IMPORTANT: Always return Pint Unit objects, never strings
            # This follows the architecture principle: "Accept strings for user convenience,
            # but ALWAYS store and return Pint objects internally"
            if hasattr(self._units, 'dimensionality'):
                # It's already a pint.Unit - return directly
                return self._units
            # It's a string - parse to Pint Unit
            return ureg.parse_expression(self._units) if isinstance(self._units, str) else self._units

        # Otherwise, compute units from the SymPy expression
        from underworld3.function.unit_conversion import compute_expression_units
        computed_units = compute_expression_units(self._expr)

        # Always return Pint Unit, never string
        if computed_units is not None:
            if hasattr(computed_units, 'dimensionality'):
                # Already a Pint Unit
                return computed_units
            # It's a string - parse to Pint Unit
            return ureg.parse_expression(computed_units) if isinstance(computed_units, str) else computed_units

        # No units found - dimensionless
        return None

    @property
    def has_units(self):
        """Check if this expression has units (for protocol compatibility)."""
        return self._units is not None

    @property
    def _units_backend(self):
        """Get the units backend (for protocol compatibility with get_units)."""
        # Import here to avoid circular dependency
        from underworld3.units import _get_default_backend
        return _get_default_backend()

    @property
    def dimensionality(self):
        """Get the dimensionality of this expression."""
        if not self.has_units:
            return None
        if self._units_backend is None:
            return None
        quantity = self._units_backend.create_quantity(1.0, self._units)
        return self._units_backend.get_dimensionality(quantity)

    def __repr__(self):
        # Use .units property (returns Pint Unit) and convert to string for display
        units = self.units
        unit_str = f" [{units}]" if units is not None else " [dimensionless]"
        return f"{self.__class__.__name__}({self._expr}{unit_str})"

    # =========================================================================
    # Mathematical Operations with Unit Preservation
    # =========================================================================

    def __mul__(self, other):
        """Multiplication preserves and combines units."""
        # Extract SymPy expression and units from other
        if isinstance(other, UnitAwareExpression):
            other_expr = other._expr
            other_units = other._units
        elif isinstance(other, (int, float, complex)):
            other_expr = sympy.sympify(other)
            other_units = None
        else:
            # Try to extract .sym if available (for compatibility)
            other_expr = getattr(other, 'sym', other)
            other_units = getattr(other, 'units', None)

        # Multiply SymPy expressions
        new_expr = self._expr * other_expr

        # Combine units using Pint
        if self._units and other_units:
            new_units = self._units * other_units
        elif self._units:
            new_units = self._units
        elif other_units:
            new_units = other_units
        else:
            new_units = None

        # Return new UnitAwareExpression with combined result
        return self.__class__(new_expr, new_units)

    def __rmul__(self, other):
        """Right multiplication."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Division updates units appropriately."""
        # Similar pattern to multiplication
        if isinstance(other, UnitAwareExpression):
            other_expr = other._expr
            other_units = other._units
        elif isinstance(other, (int, float, complex)):
            other_expr = sympy.sympify(other)
            other_units = None
        else:
            other_expr = getattr(other, 'sym', other)
            other_units = getattr(other, 'units', None)

        # Divide SymPy expressions
        new_expr = self._expr / other_expr

        # Divide units
        if self._units and other_units:
            new_units = self._units / other_units
        elif self._units:
            new_units = self._units
        elif other_units:
            new_units = ureg.dimensionless / other_units
        else:
            new_units = None

        return self.__class__(new_expr, new_units)

    def __rtruediv__(self, other):
        """Right division."""
        if isinstance(other, (int, float, complex)):
            other_expr = sympy.sympify(other)
            other_units = None
        else:
            other_expr = getattr(other, 'sym', other)
            other_units = getattr(other, 'units', None)

        new_expr = other_expr / self._expr

        if other_units and self._units:
            new_units = other_units / self._units
        elif other_units:
            new_units = other_units
        elif self._units:
            new_units = ureg.dimensionless / self._units
        else:
            new_units = None

        return self.__class__(new_expr, new_units)

    def __add__(self, other):
        """Addition requires compatible units."""
        # Import here to avoid circular dependency
        from underworld3.function.quantities import UWQuantity

        if isinstance(other, UnitAwareExpression):
            # Check unit compatibility using Pint's dimensional analysis
            if self._units and other._units:
                try:
                    # Create dummy Pint quantities to check compatibility
                    self_pint = 1.0 * self._units
                    other_pint = 1.0 * other._units

                    # Try to convert - this will raise if incompatible
                    _ = other_pint.to(self._units)

                    # Units are compatible - addition preserves left operand units
                    new_expr = self._expr + other._expr
                    return self.__class__(new_expr, self._units)
                except Exception as e:
                    raise ValueError(
                        f"Cannot add {other._units} and {self._units}: "
                        f"incompatible dimensions. {e}"
                    )
            new_expr = self._expr + other._expr
            return self.__class__(new_expr, self._units or other._units)
        elif isinstance(other, UWQuantity):
            # Handle UWQuantity operands - convert to sympy value and check units
            other_units = other.units if hasattr(other, 'units') else None

            if self._units and other_units:
                try:
                    # Check dimensional compatibility using Pint
                    self_pint = 1.0 * self._units
                    other_pint = 1.0 * other_units
                    _ = other_pint.to(self._units)  # Check if conversion is possible

                    # Convert other to sympy value and add
                    other_value = sympy.sympify(float(other.value))
                    new_expr = self._expr + other_value
                    return self.__class__(new_expr, self._units)  # Preserve left operand units
                except Exception as e:
                    raise ValueError(
                        f"Cannot add {other_units} and {self._units}: "
                        f"incompatible dimensions. {e}"
                    )
            # If no units or only one has units, just add
            other_value = sympy.sympify(float(other.value))
            new_expr = self._expr + other_value
            return self.__class__(new_expr, self._units or other_units)
        elif isinstance(other, (int, float)) and other == 0:
            # Allow adding zero without units
            return self
        else:
            raise TypeError(f"Cannot add {type(self).__name__} and {type(other).__name__}")

    def __radd__(self, other):
        """Right addition - preserve left operand's units (other + self)."""
        if isinstance(other, UnitAwareExpression):
            # When other + self, 'other' is left operand so its units should be preserved
            if self._units and other._units:
                try:
                    # Create dummy Pint quantities to check compatibility
                    self_pint = 1.0 * self._units
                    other_pint = 1.0 * other._units

                    # Try to convert - this will raise if incompatible
                    _ = self_pint.to(other._units)

                    # Units are compatible - preserve other's units (left operand)
                    new_expr = self._expr + other._expr
                    return self.__class__(new_expr, other._units)
                except Exception as e:
                    raise ValueError(
                        f"Cannot add {other._units} and {self._units}: "
                        f"incompatible dimensions. {e}"
                    )
            new_expr = other._expr + self._expr
            return self.__class__(new_expr, other._units or self._units)
        elif isinstance(other, (int, float)) and other == 0:
            # 0 + self = self
            return self
        else:
            raise TypeError(f"Cannot add {type(other).__name__} and {type(self).__name__}")

    def __sub__(self, other):
        """Subtraction requires compatible units - preserves left operand units."""
        # Import here to avoid circular dependency
        from underworld3.function.quantities import UWQuantity

        if isinstance(other, UnitAwareExpression):
            if self._units and other._units:
                try:
                    # Create dummy Pint quantities to check compatibility
                    self_pint = 1.0 * self._units
                    other_pint = 1.0 * other._units

                    # Try to convert - this will raise if incompatible
                    _ = other_pint.to(self._units)

                    # Units are compatible - subtraction preserves left operand units
                    new_expr = self._expr - other._expr
                    return self.__class__(new_expr, self._units)
                except Exception as e:
                    raise ValueError(
                        f"Cannot subtract {other._units} from {self._units}: "
                        f"incompatible dimensions. {e}"
                    )
            new_expr = self._expr - other._expr
            return self.__class__(new_expr, self._units or other._units)
        elif isinstance(other, UWQuantity):
            # Handle UWQuantity operands - convert to sympy value and check units
            other_units = other.units if hasattr(other, 'units') else None

            if self._units and other_units:
                try:
                    # Check dimensional compatibility using Pint
                    self_pint = 1.0 * self._units
                    other_pint = 1.0 * other_units
                    _ = other_pint.to(self._units)  # Check if conversion is possible

                    # Convert other to sympy value and subtract
                    other_value = sympy.sympify(float(other.value))
                    new_expr = self._expr - other_value
                    return self.__class__(new_expr, self._units)  # Preserve left operand units
                except Exception as e:
                    raise ValueError(
                        f"Cannot subtract {other_units} from {self._units}: "
                        f"incompatible dimensions. {e}"
                    )
            # If no units or only one has units, just subtract
            other_value = sympy.sympify(float(other.value))
            new_expr = self._expr - other_value
            return self.__class__(new_expr, self._units or other_units)
        else:
            raise TypeError(f"Cannot subtract {type(other).__name__} from {type(self).__name__}")

    def __rsub__(self, other):
        """Right subtraction - preserve left operand's units (other - self)."""
        if isinstance(other, UnitAwareExpression):
            # When other - self, 'other' is left operand so its units should be preserved
            if self._units and other._units:
                try:
                    # Create dummy Pint quantities to check compatibility
                    self_pint = 1.0 * self._units
                    other_pint = 1.0 * other._units

                    # Try to convert - this will raise if incompatible
                    _ = self_pint.to(other._units)

                    # Units are compatible - preserve other's units (left operand)
                    new_expr = other._expr - self._expr
                    return self.__class__(new_expr, other._units)
                except Exception as e:
                    raise ValueError(
                        f"Cannot subtract {self._units} from {other._units}: "
                        f"incompatible dimensions. {e}"
                    )
            new_expr = other._expr - self._expr
            return self.__class__(new_expr, other._units or self._units)
        elif isinstance(other, (int, float)) and other == 0:
            # 0 - self = -self
            return self.__class__(-self._expr, self._units)
        else:
            raise TypeError(f"Cannot subtract {type(self).__name__} from {type(other).__name__}")

    def __pow__(self, power):
        """Exponentiation updates units."""
        if isinstance(power, (int, float)):
            new_expr = self._expr ** power
            if self._units:
                new_units = self._units ** power
            else:
                new_units = None
            return self.__class__(new_expr, new_units)
        else:
            raise TypeError(f"Cannot raise {type(self).__name__} to power of {type(power).__name__}")

    def __neg__(self):
        """Negation preserves units."""
        return self.__class__(-self._expr, self._units)

    # =========================================================================
    # Unit Conversion
    # =========================================================================

    def to(self, target_units: str) -> 'UnitAwareExpression':
        """
        Convert to different units, returning a new symbolic expression with scaling wrapper.

        For symbolic expressions (not yet evaluated), this returns a NEW expression
        with the appropriate scaling factor/offset applied. The original expression
        remains unchanged.

        Parameters
        ----------
        target_units : str
            Target units to convert to (e.g., 'km/s', 'degC')

        Returns
        -------
        UnitAwareExpression
            New expression with scaling wrapper and target units

        Examples
        --------
        >>> # Simple scaling (no offset)
        >>> velocity_ms = velocity[0]  # Has units 'm/s'
        >>> velocity_kms = velocity_ms.to('km/s')  # Returns velocity[0] * 0.001

        >>> # Offset conversion (temperature)
        >>> temp_kelvin = temperature  # Has units 'K'
        >>> temp_celsius = temp_kelvin.to('degC')  # Returns temperature - 273.15

        Notes
        -----
        - Symbolic conversion preserves lazy evaluation
        - Only compatible units can be converted (e.g., can't convert 'm/s' to 'kelvin')
        - Uses Pint for dimensional analysis and conversion factor computation
        """
        # IMPORTANT: Use the computed .units property, not self._units!
        # For compound expressions, self._units might be stale but .units
        # is computed lazily from the expression tree
        computed_units = self.units  # This calls the lazy @property

        if not computed_units:
            raise ValueError(f"Cannot convert expression without units. Expression: {self._expr}")

        # Convert current units to Pint unit if it's a string
        if isinstance(computed_units, str):
            current_pint = ureg(computed_units)
        elif hasattr(computed_units, 'dimensionality'):
            current_pint = computed_units
        else:
            # Try to create Pint unit from whatever we have
            current_pint = ureg(str(computed_units))

        # Parse target units - handle both strings and Pint Unit objects
        if isinstance(target_units, str):
            target_pint = ureg(target_units)
        elif hasattr(target_units, 'dimensionality'):
            # Already a Pint Unit object
            target_pint = target_units
        else:
            # Try to convert to Pint Unit
            target_pint = ureg(str(target_units))

        # Check dimensionality compatibility
        if current_pint.dimensionality != target_pint.dimensionality:
            raise ValueError(
                f"Cannot convert from {self._units} to {target_units}: "
                f"incompatible dimensionalities"
            )

        # Create quantities to compute conversion
        from_qty = 1.0 * current_pint
        to_qty = from_qty.to(target_pint)

        # Check if this is an offset unit (like Celsius/Fahrenheit)
        # For offset units: new = old * factor + offset
        # For regular units: new = old * factor
        try:
            # Try zero conversion to detect offset
            zero_from = 0.0 * current_pint
            zero_to = zero_from.to(target_pint)
            offset = zero_to.magnitude
            has_offset = abs(offset) > 1e-10
        except:
            # If offset detection fails, assume no offset
            has_offset = False
            offset = 0.0

        # Compute conversion factor
        factor = to_qty.magnitude

        # Create new symbolic expression with scaling wrapper
        # Handle both scalar and matrix expressions
        import sympy

        if has_offset:
            # Offset conversion: expr * factor + offset
            # For matrices, we need to use elementwise operations
            if isinstance(self._expr, sympy.MatrixBase):
                # Matrix: apply operation element-wise
                new_expr = self._expr * factor + sympy.ones(*self._expr.shape) * offset
            else:
                # Scalar: direct operation
                new_expr = self._expr * factor + offset
        else:
            # Simple scaling: expr * factor
            if abs(factor - 1.0) > 1e-10:  # Only apply scaling if factor != 1
                new_expr = self._expr * factor
            else:
                new_expr = self._expr

        # Return new UnitAwareExpression with target units
        return self.__class__(new_expr, target_pint)

    def to_base_units(self) -> 'UnitAwareExpression':
        """
        Convert to SI base units (meter, second, kilogram, etc.).

        For composite expressions containing UWexpression symbols, this method
        changes ONLY the display units, not the expression tree. This is necessary
        because embedded conversion factors would be double-applied during
        nondimensional evaluation cycles.

        For simple expressions without symbols, the conversion factor is applied.

        Returns
        -------
        UnitAwareExpression
            New expression with base SI units

        Examples
        --------
        >>> # Simple expression - applies conversion
        >>> velocity_kms = uw.expression("v", 5, units="km/hour")
        >>> velocity_ms = velocity_kms.to_base_units()
        >>> # Returns: v * 0.2777... [meter / second]

        >>> # Composite expression - only changes display units
        >>> sqrt_expr = ((kappa * t_now))**0.5  # megayear^0.5 * meter / second^0.5
        >>> sqrt_m = sqrt_expr.to_base_units()  # meter (display only)
        >>> # Evaluation uses original expression tree - no double-application
        """
        if not self.units:
            raise ValueError("Cannot convert expression without units to base units")

        # Create dummy Pint Quantity to compute conversion
        current_qty = 1.0 * self.units
        base_qty = current_qty.to_base_units()

        # Extract scaling factor and new units
        factor = base_qty.magnitude
        new_units = base_qty.units

        # Check if expression contains UWexpression symbols
        import sympy
        from underworld3.function.expressions import UWexpression
        uwexpr_atoms = list(self._expr.atoms(UWexpression))

        if uwexpr_atoms:
            # Composite expression with UWexpression symbols
            # DO NOT apply conversion factor - would be double-applied during evaluation
            # Only change display units for unit simplification
            import warnings
            warnings.warn(
                f"to_base_units() on composite expression with symbols: "
                f"changing display units only ('{self.units}' → '{new_units}'). "
                f"This is a unit simplification, not an actual conversion. "
                f"Use to_compact() if you want automatic readable units instead.",
                UserWarning
            )
            new_expr = self._expr
        else:
            # Simple expression - safe to apply conversion factor
            if abs(factor - 1.0) > 1e-10:
                new_expr = self._expr * factor
            else:
                new_expr = self._expr

        return self.__class__(new_expr, new_units)

    def to_compact(self) -> 'UnitAwareExpression':
        """
        Convert to compact representation with best automatic units.

        Uses Pint's to_compact() to select the most readable unit representation.

        Returns
        -------
        UnitAwareExpression
            New expression with automatically selected compact units

        Examples
        --------
        >>> distance_mm = uw.expression("d", 1e6, units="mm")
        >>> distance_km = distance_mm.to_compact()
        >>> # Returns: d * 0.001 [kilometer]
        """
        if not self.units:
            raise ValueError("Cannot compact expression without units")

        try:
            # Create dummy Pint Quantity to compute conversion
            current_qty = 1.0 * self.units
            compact_qty = current_qty.to_compact()

            # Extract scaling factor and new units
            factor = compact_qty.magnitude
            new_units = compact_qty.units

            # Apply scaling to symbolic expression
            import sympy
            if abs(factor - 1.0) > 1e-10:
                new_expr = self._expr * factor
            else:
                new_expr = self._expr

            return self.__class__(new_expr, new_units)
        except AttributeError:
            raise AttributeError(
                "to_compact() requires Pint >= 0.17. "
                "Upgrade with: pip install --upgrade pint"
            )

    def to_reduced_units(self) -> 'UnitAwareExpression':
        """
        Simplify unit expressions by canceling common factors.

        For composite expressions containing UWexpression symbols, this method
        changes ONLY the display units, not the expression tree. This is necessary
        because embedded conversion factors would be double-applied during
        nondimensional evaluation cycles.

        For simple expressions without symbols, the conversion factor is applied.

        Returns
        -------
        UnitAwareExpression
            New expression with simplified units

        Examples
        --------
        >>> # Simple expression - applies conversion
        >>> expr = velocity * time  # cm/year * Myr
        >>> simplified = expr.to_reduced_units()
        >>> # Returns: expr * 1e6 [centimeter]

        >>> # Composite expression - only simplifies display units
        >>> sqrt_expr = ((kappa * t_now))**0.5  # megayear^0.5 * meter / second^0.5
        >>> sqrt_simplified = sqrt_expr.to_reduced_units()  # meter (display only)
        """
        if not self.units:
            # Already dimensionless
            return self

        # Create dummy Pint Quantity to compute conversion
        current_qty = 1.0 * self.units
        reduced_qty = current_qty.to_reduced_units()

        # Extract scaling factor and new units
        factor = reduced_qty.magnitude
        new_units = reduced_qty.units

        # Check if expression contains UWexpression symbols
        import sympy
        from underworld3.function.expressions import UWexpression
        uwexpr_atoms = list(self._expr.atoms(UWexpression))

        if uwexpr_atoms:
            # Composite expression with UWexpression symbols
            # DO NOT apply conversion factor - would be double-applied during evaluation
            # Only change display units for unit simplification
            import warnings
            warnings.warn(
                f"to_reduced_units() on composite expression with symbols: "
                f"changing display units only ('{self.units}' → '{new_units}'). "
                f"This is a unit simplification, not an actual conversion.",
                UserWarning
            )
            new_expr = self._expr
        else:
            # Simple expression - safe to apply conversion factor
            if abs(factor - 1.0) > 1e-10:
                new_expr = new_expr * factor
            else:
                new_expr = self._expr

        return self.__class__(new_expr, new_units)

    def to_nice_units(self) -> 'UnitAwareExpression':
        """
        Convert to 'nice' representation using automatic compact units.

        Alias for to_compact() - finds the most readable unit representation.

        Returns
        -------
        UnitAwareExpression
            New expression with nice, readable units
        """
        return self.to_compact()

    # For SymPy compatibility
    def _sympy_(self):
        """Allow sympify to extract the SymPy expression.

        Note: The correct protocol method is _sympy_() not _sympify_().
        SymPy checks for this method when converting objects to SymPy expressions,
        including in strict mode (used by matrix operations).
        """
        return self._expr

    @property
    def args(self):
        """SymPy compatibility - expose args of underlying expression."""
        return self._expr.args

    # =========================================================================
    # Mathematical Operations (promote to MathematicalExpression)
    # =========================================================================

    def diff(self, var):
        """
        Differentiate with respect to a variable, updating units.

        This method promotes the UnitAwareExpression to a MathematicalExpression
        which has full calculus support.

        Parameters
        ----------
        var : symbol or UnitAwareExpression
            Variable to differentiate with respect to

        Returns
        -------
        MathematicalExpression
            Result of differentiation with updated units
        """
        # Promote to MathematicalExpression and differentiate
        math_expr = MathematicalExpression(self._expr, self._units)
        return math_expr.diff(var)


# ==============================================================================
# Layer 3: Mathematical Expression (adds calculus operations)
# ==============================================================================

class MathematicalExpression(UnitAwareExpression):
    """
    Extends UnitAwareExpression with mathematical operations like
    differentiation and integration that update units appropriately.
    """

    def diff(self, var):
        """
        Differentiate with respect to a variable, updating units.

        d/dx of a quantity with units [U] where x has units [X]
        results in units [U]/[X]
        """
        # Extract variable's SymPy symbol and units
        if isinstance(var, UnitAwareExpression):
            var_sym = var._expr
            var_units = var._units
        else:
            var_sym = var
            var_units = None

        # Differentiate the SymPy expression
        diff_expr = self._expr.diff(var_sym)

        # Update units: original_units / var_units
        if self._units and var_units:
            new_units = self._units / var_units
        elif self._units:
            new_units = self._units  # Differentiating w.r.t dimensionless
        else:
            new_units = None

        return MathematicalExpression(diff_expr, new_units)

    def integrate(self, var):
        """
        Integrate with respect to a variable, updating units.

        ∫ dx of a quantity with units [U] where x has units [X]
        results in units [U]*[X]
        """
        # Extract variable's SymPy symbol and units
        if isinstance(var, UnitAwareExpression):
            var_sym = var._expr
            var_units = var._units
        else:
            var_sym = var
            var_units = None

        # Integrate the SymPy expression
        int_expr = sympy.integrate(self._expr, var_sym)

        # Update units: original_units * var_units
        if self._units and var_units:
            new_units = self._units * var_units
        elif self._units:
            new_units = self._units
        else:
            new_units = None

        return MathematicalExpression(int_expr, new_units)

    def expand(self):
        """Expand the expression (preserves units)."""
        return MathematicalExpression(self._expr.expand(), self._units)

    def simplify(self):
        """Simplify the expression (preserves units)."""
        return MathematicalExpression(self._expr.simplify(), self._units)

    def subs(self, substitutions):
        """Substitute variables (preserves units)."""
        if isinstance(substitutions, dict):
            # Handle dictionary of substitutions
            new_subs = {}
            for key, value in substitutions.items():
                if isinstance(key, UnitAwareExpression):
                    key = key._expr
                if isinstance(value, UnitAwareExpression):
                    value = value._expr
                new_subs[key] = value
            new_expr = self._expr.subs(new_subs)
        else:
            # Handle single substitution
            old, new = substitutions
            if isinstance(old, UnitAwareExpression):
                old = old._expr
            if isinstance(new, UnitAwareExpression):
                new = new._expr
            new_expr = self._expr.subs(old, new)

        return MathematicalExpression(new_expr, self._units)


# ==============================================================================
# Layer 4: Lazy Expression (deferred evaluation)
# ==============================================================================

class LazyExpression(MathematicalExpression):
    """
    Adds lazy evaluation - expression is not evaluated until explicitly requested.
    This preserves the lazy evaluation requirement.
    """

    def __init__(self, expr, units=None, evaluator=None):
        super().__init__(expr, units)
        self._evaluator = evaluator  # Function to evaluate when needed
        self._cached_result = None

    def evaluate(self, coords=None, **kwargs):
        """
        Evaluate the expression with given parameters.
        Returns a unit-aware result.
        """
        if self._evaluator:
            # Use custom evaluator
            raw_result = self._evaluator(self._expr, coords=coords, **kwargs)
        else:
            # Default evaluation using SymPy's lambdify
            from sympy import lambdify

            # Handle coordinates if provided
            if coords is not None:
                # Use underworld's evaluate function
                import underworld3 as uw
                raw_result = uw.function.evaluate(unwrap(self._expr), coords)
            else:
                # Extract symbols from expression
                symbols = list(self._expr.free_symbols)
                if not symbols:
                    # Constant expression
                    raw_result = float(self._expr)
                else:
                    # Create evaluator
                    func = lambdify(symbols, self._expr, 'numpy')
                    # Get values for symbols
                    values = [kwargs.get(str(s), 0) for s in symbols]
                    raw_result = func(*values)

        # Wrap result with units if present
        if self._units:
            # Check if we need to dimensionalize
            model = get_default_model()
            if model and model.has_units():
                # Get dimensionality from units
                if hasattr(self._units, 'dimensionality'):
                    dimensionality = dict(self._units.dimensionality)
                else:
                    # Try creating quantity to get dimensionality
                    temp_qty = 1.0 * self._units
                    dimensionality = dict(temp_qty.dimensionality)

                # Dimensionalize the result
                import underworld3 as uw
                return uw.dimensionalise(raw_result, target_dimensionality=dimensionality, model=model)
            else:
                # No model scaling - return with units directly
                return quantity(raw_result, self._units)
        else:
            return raw_result

    def min(self):
        """Find minimum value (with proper dimensionalization)."""
        if hasattr(self._expr, 'min'):
            # Direct min method
            raw_min = self._expr.min()
        else:
            # Need to evaluate to get min
            result = self.evaluate()
            if hasattr(result, 'magnitude'):
                return result  # Already has units
            raw_min = np.min(result)

        # Apply units and dimensionalization
        if self._units:
            model = get_default_model()
            if model and model.has_units():
                # Get dimensionality
                if hasattr(self._units, 'dimensionality'):
                    dimensionality = dict(self._units.dimensionality)
                else:
                    temp_qty = 1.0 * self._units
                    dimensionality = dict(temp_qty.dimensionality)

                # Dimensionalize the result
                import underworld3 as uw
                return uw.dimensionalise(raw_min, target_dimensionality=dimensionality, model=model)
            else:
                return quantity(raw_min, self._units)
        return raw_min

    def max(self):
        """Find maximum value (with proper dimensionalization)."""
        if hasattr(self._expr, 'max'):
            # Direct max method
            raw_max = self._expr.max()
        else:
            # Need to evaluate to get max
            result = self.evaluate()
            if hasattr(result, 'magnitude'):
                return result  # Already has units
            raw_max = np.max(result)

        # Apply units and dimensionalization
        if self._units:
            model = get_default_model()
            if model and model.has_units():
                # Get dimensionality
                if hasattr(self._units, 'dimensionality'):
                    dimensionality = dict(self._units.dimensionality)
                else:
                    temp_qty = 1.0 * self._units
                    dimensionality = dict(temp_qty.dimensionality)

                # Dimensionalize the result
                import underworld3 as uw
                return uw.dimensionalise(raw_max, target_dimensionality=dimensionality, model=model)
            else:
                return quantity(raw_max, self._units)
        return raw_max


# ==============================================================================
# Helper Functions
# ==============================================================================

def create_unit_aware(expr, units=None):
    """
    Factory function to create appropriate unit-aware expression.

    Parameters
    ----------
    expr : Any
        Expression to wrap (can be SymPy, numeric, or already unit-aware)
    units : str or pint.Unit, optional
        Units for the expression

    Returns
    -------
    LazyExpression
        Unit-aware expression with full functionality
    """
    # Convert string units to Pint units
    if isinstance(units, str):
        units = ureg(units)

    # Handle different input types
    if isinstance(expr, (UnitAwareExpression, MathematicalExpression, LazyExpression)):
        # Already unit-aware, update units if provided
        if units is not None:
            return LazyExpression(expr._expr, units)
        return expr
    elif isinstance(expr, sympy.Basic):
        # SymPy expression
        return LazyExpression(expr, units)
    elif isinstance(expr, (int, float, complex, np.ndarray)):
        # Numeric value - convert to SymPy
        sym_expr = sympy.sympify(expr) if not isinstance(expr, np.ndarray) else sympy.Matrix(expr)
        return LazyExpression(sym_expr, units)
    else:
        # Try to extract sym property
        if hasattr(expr, 'sym'):
            return LazyExpression(expr.sym, units or getattr(expr, 'units', None))
        else:
            # Last resort - sympify
            return LazyExpression(sympy.sympify(expr), units)


# Export main classes and functions
__all__ = [
    'UnitAwareExpression',
    'MathematicalExpression',
    'LazyExpression',
    'create_unit_aware'
]