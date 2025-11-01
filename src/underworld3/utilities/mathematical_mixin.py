# underworld3/utilities/mathematical_mixin.py
"""
Fixed Mathematical Mixin for Underworld3 Variables

This version addresses inconsistencies identified through comprehensive testing.
"""

import sympy
import inspect
from typing import Any


class MathematicalMixin:
    """
    Fixed Mathematical Mixin with consistent error handling and operation support.

    Key improvements:
    - Better error handling for unsupported operations
    - Intelligent method delegation with signature handling
    - Proper validation of sym property
    - Consistent behavior across different variable types
    """

    def _validate_sym(self):
        """Validate that sym property is available and valid."""
        try:
            # Try to access sym property - this should not trigger __getattr__ recursion
            # since __getattr__ now guards against accessing private methods
            sym = self.sym
        except AttributeError:
            # Genuinely missing sym property
            raise AttributeError(f"{type(self).__name__} object has no 'sym' property")
        except RecursionError:
            # Recursion loop in sym property access
            raise AttributeError(
                f"{type(self).__name__} object has recursive 'sym' property access"
            )
        except Exception as e:
            # For other exceptions (like RuntimeError), preserve the original exception type
            # This allows tests to check for specific exception types
            raise e

        if sym is None:
            raise ValueError(
                f"{type(self).__name__}.sym is None - variable not properly initialized"
            )
        return sym

    def _sympify_(self):
        """SymPy protocol: Tell SymPy to use the symbolic form."""
        return self._validate_sym()

    def __getitem__(self, index):
        """Component access with proper bounds checking."""
        sym = self._validate_sym()

        # Check for scalar variables (no indexing allowed)
        if not hasattr(sym, "shape") or not hasattr(sym, "__getitem__"):
            raise TypeError(f"'{type(self).__name__}' object (scalar) is not subscriptable")

        # For matrices, validate index bounds
        if hasattr(sym, "shape"):
            if isinstance(index, int):
                if index < 0:
                    raise IndexError("Negative indexing not supported for mathematical variables")

                # For matrices, check against the total number of components
                # Row vectors have shape (1, n) - use cols, column vectors have shape (n, 1) - use rows
                rows, cols = sym.shape
                if rows == 1 and cols > 1:
                    # Row vector - check against columns
                    if index >= cols:
                        raise IndexError(
                            f"Index {index} out of bounds for variable with {cols} components"
                        )
                elif cols == 1 and rows > 1:
                    # Column vector - check against rows
                    if index >= rows:
                        raise IndexError(
                            f"Index {index} out of bounds for variable with {rows} components"
                        )
                else:
                    # General matrix or scalar - delegate to SymPy's error handling
                    pass

        return sym[index]

    def __iter__(self):
        """Allow iteration for matrix-type UWexpressions.

        SymPy's simplification code tries to iterate over Symbol subclasses,
        treating them as atomic elements. For Matrix types, we delegate to
        SymPy's iteration. For scalar Symbols, we raise TypeError to indicate
        they are atomic and should not be iterated.

        This prevents: TypeError: 'UWexpression' object (scalar) is not subscriptable
        when SymPy's cancel() → factor_terms() tries to factorize expressions.
        """
        sym = self._validate_sym()

        # Only Matrix-like objects should be iterable
        # Check if it's actually a SymPy Matrix type
        if isinstance(sym, sympy.MatrixBase):
            # Matrix/vector - delegate to SymPy's iteration
            return iter(sym)
        else:
            # Scalar Symbol - raise TypeError to indicate it's atomic
            # This tells SymPy not to iterate over it
            raise TypeError(f"iteration over a 0-d {type(self).__name__}")

    def __len__(self):
        """Length method for SymPy sequence compatibility."""
        sym = self._validate_sym()

        # For matrices, return the length (number of elements)
        if hasattr(sym, "__len__"):
            return len(sym)
        else:
            # For scalars or objects without len, raise TypeError
            raise TypeError(f"object of type '{type(self).__name__}' has no len()")

    def __repr__(self):
        """String representation returns the symbolic form."""
        try:
            sym = self._validate_sym()
            return repr(sym)
        except (AttributeError, ValueError):
            # Fallback for broken variables
            return f"<{type(self).__name__} with invalid sym>"

    def _repr_latex_(self):
        """Jupyter notebook LaTeX representation."""
        try:
            sym = self._validate_sym()
            from sympy import latex

            return f"$${latex(sym)}$$"
        except (AttributeError, ValueError, ImportError):
            return f"$${type(self).__name__}$$"

    def _ipython_display_(self):
        """IPython/Jupyter display hook."""
        try:
            from IPython.display import display, Math
            from sympy import latex

            sym = self._validate_sym()
            display(Math(latex(sym)))
        except ImportError:
            # IPython not available - silent fallback
            pass
        except (AttributeError, ValueError):
            # Broken sym - show type name
            try:
                from IPython.display import display

                display(f"{type(self).__name__} (invalid)")
            except ImportError:
                pass

    def diff(self, *args, **kwargs):
        """
        Direct differentiation with automatic unit wrapping.

        Returns a unit-aware expression if the original variable has units,
        plain SymPy expression otherwise.

        For Matrix/vector results, creates a custom wrapper that provides
        unit-aware indexing, so derivative[0] returns a unit-aware object.

        Examples:
            temperature.diff(y)[0]  # Returns unit-aware dT/dy with .units and .to()
            velocity[0].diff(x)     # Returns unit-aware dVx/dx with unit methods
            velocity.diff(x)        # Returns UnitAwareDerivativeMatrix with unit-aware elements
        """
        sym = self._validate_sym()
        result = sym.diff(*args, **kwargs)

        # Try to wrap the result with units
        try:
            import underworld3 as uw

            if hasattr(uw, "with_units"):
                # Check if result is a Matrix
                if hasattr(result, "shape"):
                    # Return a special Matrix wrapper that provides unit-aware indexing
                    return UnitAwareDerivativeMatrix(result, uw.with_units)
                else:
                    # Scalar result - wrap directly
                    return uw.with_units(result)
        except (ImportError, AttributeError):
            pass

        # Fallback: return plain SymPy result
        return result

    # Arithmetic operations with better error handling
    def __add__(self, other):
        """Addition with scalar broadcasting and error handling."""
        sym = self._validate_sym()

        # KEY FIX: Don't substitute .sym for MathematicalMixin objects
        # This preserves symbolic expressions for lazy evaluation
        if hasattr(other, "_sympify_") and not isinstance(other, MathematicalMixin):
            other = other._sympify_()

        try:
            # Try normal SymPy addition first
            return sym + other
        except (TypeError, ValueError):
            # If that fails, try scalar broadcasting
            try:
                other_sym = sympy.sympify(other)
                if other_sym.is_number and hasattr(sym, "shape"):
                    # Broadcast scalar to matrix shape
                    broadcasted = other_sym * sympy.ones(*sym.shape)
                    return sym + broadcasted
            except:
                pass
            # If broadcasting doesn't work, re-raise with helpful message
            raise TypeError(
                f"Cannot add {type(self).__name__} and {type(other).__name__}. "
                f"Check dimensional compatibility."
            )

    def __radd__(self, other):
        """Right addition with scalar broadcasting."""
        return self.__add__(other)

    def __sub__(self, other):
        """Subtraction with scalar broadcasting and error handling."""
        sym = self._validate_sym()

        # KEY FIX: Don't substitute .sym for MathematicalMixin objects
        # This preserves symbolic expressions for lazy evaluation
        if hasattr(other, "_sympify_") and not isinstance(other, MathematicalMixin):
            other = other._sympify_()

        try:
            return sym - other
        except (TypeError, ValueError):
            try:
                other_sym = sympy.sympify(other)
                if other_sym.is_number and hasattr(sym, "shape"):
                    broadcasted = other_sym * sympy.ones(*sym.shape)
                    return sym - broadcasted
            except:
                pass
            raise TypeError(
                f"Cannot subtract {type(other).__name__} from {type(self).__name__}. "
                f"Check dimensional compatibility."
            )

    def __rsub__(self, other):
        """Right subtraction with scalar broadcasting."""
        sym = self._validate_sym()

        # KEY FIX: Don't substitute .sym for MathematicalMixin objects
        # This preserves symbolic expressions for lazy evaluation
        if hasattr(other, "_sympify_") and not isinstance(other, MathematicalMixin):
            other = other._sympify_()

        try:
            return other - sym
        except (TypeError, ValueError):
            try:
                other_sym = sympy.sympify(other)
                if other_sym.is_number and hasattr(sym, "shape"):
                    broadcasted = other_sym * sympy.ones(*sym.shape)
                    return broadcasted - sym
            except:
                pass
            raise TypeError(
                f"Cannot subtract {type(self).__name__} from {type(other).__name__}. "
                f"Check dimensional compatibility."
            )

    def __mul__(self, other):
        """Multiplication with error handling."""
        sym = self._validate_sym()

        # KEY FIX: Don't substitute .sym for MathematicalMixin objects
        # This preserves symbolic expressions for lazy evaluation
        if hasattr(other, "_sympify_") and not isinstance(other, MathematicalMixin):
            other = other._sympify_()

        try:
            return sym * other
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Cannot multiply {type(self).__name__} and {type(other).__name__}: {e}"
            )

    def __rmul__(self, other):
        """Right multiplication with error handling."""
        sym = self._validate_sym()

        # Call _sympify_() to get SymPy representation that can multiply with matrices
        # Skip for MathematicalMixin objects (MeshVariable, SwarmVariable) to preserve lazy evaluation
        # UWexpression no longer inherits from MathematicalMixin, so it will be sympified

        if hasattr(other, "_sympify_"):
            # Allow sympification for non-MathematicalMixin objects (includes UWexpression)
            if not isinstance(other, MathematicalMixin):
                other = other._sympify_()

        # SymPy naturally handles scalar * Matrix correctly:
        # - scalar * 1x1 Matrix → 1x1 Matrix (scalar result)
        # - scalar * Nx1 Matrix → Nx1 Matrix (vector result)
        # UWexpression._sympify_() returns _sym (pure SymPy Float/Symbol/Mul/etc), enabling multiplication

        try:
            return other * sym
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Cannot multiply {type(other).__name__} and {type(self).__name__}: {e}"
            )

    def __truediv__(self, other):
        """Division with error handling."""
        sym = self._validate_sym()

        # KEY FIX: Don't substitute .sym for MathematicalMixin objects
        # This preserves symbolic expressions for lazy evaluation
        if hasattr(other, "_sympify_") and not isinstance(other, MathematicalMixin):
            other = other._sympify_()

        try:
            return sym / other
        except (TypeError, ValueError) as e:
            raise TypeError(f"Cannot divide {type(self).__name__} by {type(other).__name__}: {e}")

    def __rtruediv__(self, other):
        """Right division with error handling."""
        sym = self._validate_sym()

        # KEY FIX: Don't substitute .sym for MathematicalMixin objects
        # This preserves symbolic expressions for lazy evaluation
        if hasattr(other, "_sympify_") and not isinstance(other, MathematicalMixin):
            other = other._sympify_()

        try:
            return other / sym
        except TypeError:
            # Division by matrix/vector often not supported
            raise TypeError(
                f"Division {type(other).__name__} / {type(self).__name__} not supported. "
                f"Consider element-wise operations if appropriate."
            )

    def __pow__(self, other):
        """Power with error handling."""
        sym = self._validate_sym()

        # KEY FIX: Don't substitute .sym for MathematicalMixin objects
        # This preserves symbolic expressions for lazy evaluation
        if hasattr(other, "_sympify_") and not isinstance(other, MathematicalMixin):
            other = other._sympify_()

        try:
            return sym**other
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Cannot raise {type(self).__name__} to power {type(other).__name__}: {e}"
            )

    def __rpow__(self, other):
        """Right power with error handling."""
        sym = self._validate_sym()

        # KEY FIX: Don't substitute .sym for MathematicalMixin objects
        # This preserves symbolic expressions for lazy evaluation
        if hasattr(other, "_sympify_") and not isinstance(other, MathematicalMixin):
            other = other._sympify_()

        try:
            return other**sym
        except TypeError:
            # Matrix/vector exponents often not mathematically meaningful
            raise TypeError(
                f"Exponentiation {type(other).__name__} ** {type(self).__name__} not supported. "
                f"Matrix/vector exponents are not generally well-defined."
            )

    def __neg__(self):
        """Negation."""
        sym = self._validate_sym()
        try:
            return -sym
        except (TypeError, ValueError) as e:
            raise TypeError(f"Cannot negate {type(self).__name__}: {e}")

    def __pos__(self):
        """Positive (identity operation)."""
        sym = self._validate_sym()
        try:
            return +sym
        except TypeError:
            # Some SymPy objects don't support unary +, just return the symbol
            return sym

    def norm(self, norm_type=None):
        """
        Compute the norm of the variable.

        This method intelligently delegates to either SymPy or PETSc:
        - If called without arguments: Uses SymPy Matrix norm (for mathematical expressions)
        - If called with arguments: Uses PETSc vector norm (for computational operations)

        Parameters
        ----------
        norm_type : int, optional
            PETSc norm type (0=NORM_1, 2=NORM_2, 3=NORM_INFINITY)
            If None, uses SymPy Matrix norm (defaults to 2-norm)

        Returns
        -------
        sympy.Expr or float/tuple
            If norm_type is None: SymPy expression for the norm
            If norm_type is provided: PETSc norm value (float or tuple for multi-component)

        Examples
        --------
        >>> vel.norm()  # Mathematical: returns SymPy sqrt(v_x^2 + v_y^2)
        >>> vel.norm(2)  # Computational: returns PETSc L2 norm value
        """
        if norm_type is None:
            # Mathematical usage: delegate to SymPy Matrix.norm()
            sym = self._validate_sym()
            return sym.norm()  # SymPy norm defaults to 2-norm (Euclidean)
        else:
            # Computational usage: delegate to PETSc norm (via super())
            # This calls the parent class method (e.g., _BaseMeshVariable.norm())
            return super().norm(norm_type)

    def sym_repr(self):
        """
        Mathematical representation of the variable.

        Shows the symbolic form that would be used in mathematical
        expressions and JIT compilation.

        Returns:
            String representation of the symbolic form

        Example:
            velocity = MeshVariable("velocity", mesh, 2)
            velocity         # Shows computational view
            velocity.sym_repr()  # Shows: "Matrix([[V_0(x, y, z)], [V_1(x, y, z)]])"
        """
        return str(self.sym)

    def __getattr__(self, name):
        """Enhanced method delegation with signature handling."""
        # Prevent recursion if _validate_sym is being accessed
        if name == "_validate_sym" or name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        try:
            sym = self._validate_sym()
        except (AttributeError, ValueError, RecursionError):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}' "
                f"(sym property is invalid)"
            )

        if hasattr(sym, name):
            attr = getattr(sym, name)

            if callable(attr):

                def method_wrapper(*args, **kwargs):
                    # KEY FIX: Don't substitute .sym for MathematicalMixin objects
                    # This preserves symbolic expressions for lazy evaluation
                    converted_args = []
                    for arg in args:
                        if hasattr(arg, "_sympify_") and not isinstance(arg, MathematicalMixin):
                            converted_args.append(arg._sympify_())
                        else:
                            converted_args.append(arg)

                    converted_kwargs = {}
                    for key, value in kwargs.items():
                        if hasattr(value, "_sympify_") and not isinstance(value, MathematicalMixin):
                            converted_kwargs[key] = value._sympify_()
                        else:
                            converted_kwargs[key] = value

                    try:
                        result = attr(*converted_args, **converted_kwargs)
                        # Handle in-place methods that return None
                        if result is None:
                            # Method likely modified the object in-place, return the modified sym
                            return sym
                        return result
                    except TypeError as e:
                        # Provide helpful error message with signature info
                        try:
                            sig = inspect.signature(attr)
                            raise TypeError(
                                f"Method {name}() failed: {e}. " f"Expected signature: {sig}"
                            ) from e
                        except (ValueError, TypeError):
                            # If signature inspection fails, give basic error
                            raise TypeError(
                                f"Method {name}() failed: {e}. "
                                f"Check arguments and method documentation."
                            ) from e

                return method_wrapper
            else:
                # It's a property, return its value directly
                return attr

        # If attribute doesn't exist on sym, raise appropriate error
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'. "
            f"Available attributes: {[attr for attr in dir(sym) if not attr.startswith('_')]}"
        )


class UnitAwareDerivativeMatrix:
    """
    Wrapper for SymPy Matrix derivatives that provides unit-aware indexing.

    When you index into this matrix (e.g., result[0]), it automatically wraps
    the element in a unit-aware object using uw.with_units().

    When used in arithmetic expressions without indexing, it automatically unwraps
    to the underlying SymPy Matrix via the _sympify_() protocol.
    """

    def __init__(self, sympy_matrix, with_units_func):
        self._matrix = sympy_matrix
        self._with_units = with_units_func

    def _sympify_(self):
        """SymPy protocol: Return the underlying SymPy Matrix for arithmetic operations."""
        return self._matrix

    def __getitem__(self, index):
        """Index into matrix and wrap result with units."""
        element = self._matrix[index]
        return self._with_units(element)

    def __repr__(self):
        return repr(self._matrix)

    def _repr_latex_(self):
        """Jupyter LaTeX representation."""
        if hasattr(self._matrix, "_repr_latex_"):
            return self._matrix._repr_latex_()
        from sympy import latex

        return f"$${latex(self._matrix)}$$"

    @property
    def shape(self):
        """Pass through shape attribute."""
        return self._matrix.shape

    def __len__(self):
        """Return length of matrix for SymPy compatibility."""
        return len(self._matrix)

    def __getattr__(self, name):
        """Delegate other attributes to the underlying matrix."""
        return getattr(self._matrix, name)


# Convenience function for testing with better error handling
def test_mathematical_mixin_fixed():
    """Test the fixed mathematical mixin implementation."""

    class MockVariable(MathematicalMixin):
        def __init__(self, name, sym_value):
            self.name = name
            self.sym = sym_value

    print("Testing Fixed MathematicalMixin...")

    # Create test symbols
    x, y = sympy.symbols("x y")

    # Test scalar variable
    scalar_sym = sympy.Function("T")(x, y)
    scalar_var = MockVariable("temperature", scalar_sym)

    # Test vector variable
    vector_sym = sympy.Matrix([sympy.Function("V_0")(x, y), sympy.Function("V_1")(x, y)])
    vector_var = MockVariable("velocity", vector_sym)

    print("✓ Basic arithmetic operations")
    assert (2 * scalar_var).equals(2 * scalar_sym)
    assert (vector_var * 3).equals(vector_sym * 3)

    print("✓ Scalar broadcasting")
    result = vector_var + 1
    expected = vector_sym + sympy.ones(*vector_sym.shape)
    assert result.equals(expected)

    print("✓ Component access")
    assert vector_var[0].equals(vector_sym[0])

    print("✓ Error handling for unsupported operations")
    try:
        result = 2 / vector_var  # Should raise clear error
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "not supported" in str(e)

    print("✓ Method delegation with defaults")
    norm_result = vector_var.norm()  # Should work with default arguments
    expected_norm = vector_sym.norm(2)
    assert norm_result.equals(expected_norm)

    print("✓ Bounds checking")
    try:
        vector_var[-1]  # Should raise IndexError
        assert False, "Should have raised IndexError"
    except IndexError as e:
        assert "Negative indexing not supported" in str(e)

    print("✓ All fixed MathematicalMixin tests passed!")
    return True


if __name__ == "__main__":
    test_mathematical_mixin_fixed()
