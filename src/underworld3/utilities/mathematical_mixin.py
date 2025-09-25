# underworld3/function/mathematical_mixin.py
"""
Mathematical Mixin for Underworld3 Variables

This module provides a mixin class that enables variables to work directly
in mathematical expressions without requiring explicit .sym access.
"""

import sympy
from typing import Any


class MathematicalMixin:
    """
    Mixin class that makes variables work directly in mathematical expressions.
    
    Key principle: Objects behave like their symbolic form in mathematical contexts
    while preserving computational data storage and access.
    
    This mixin enables:
    - Direct arithmetic: v1 = -1 * v2 (instead of v1 = -1 * v2.sym)
    - Component access: v[0] (instead of v.sym[0])
    - Mathematical display in notebooks
    - JIT compilation compatibility
    
    Usage:
        class MeshVariable(MathematicalMixin, _MeshVariable):
            pass
            
        # Now works naturally:
        velocity = MeshVariable("velocity", mesh, 2)
        momentum = density * velocity  # Direct arithmetic
        v_x = velocity[0]              # Component access
    """
    
    def _sympify_(self):
        """
        SymPy protocol: Tell SymPy to use the symbolic form.
        
        This is the key method that enables direct arithmetic operations.
        When SymPy encounters this object in an expression, it automatically 
        calls this method to get the SymPy representation.
        
        Returns:
            The symbolic form (self.sym) which is a SymPy Function or Matrix
            
        Example:
            velocity = MeshVariable("velocity", mesh, 2)
            momentum = density * velocity  # Calls velocity._sympify_()
            # Result: density * velocity.sym (pure SymPy expression)
            
        JIT Compatibility:
            The returned SymPy object contains the same Function atoms that
            the JIT system expects, preserving all compilation functionality.
        """
        return self.sym
    
    def __getitem__(self, index):
        """
        Component access for vector/tensor fields.
        
        Enables: velocity[0] instead of velocity.sym[0]
        
        Args:
            index: Component index (int)
            
        Returns:
            Pure SymPy expression (SymPy Function) representing the component
            
        Raises:
            IndexError: If accessing components of scalar field or index out of range
            
        Example:
            velocity = MeshVariable("velocity", mesh, 2)
            v_x = velocity[0]  # Returns velocity.sym[0]
            expr = 1 + velocity[0]  # Works in arithmetic
        """
        return self.sym[index]
    
    def __repr__(self):
        """
        String representation returns the symbolic form.
        
        When variables are typed out in notebooks, users expect to see
        the mathematical symbol, not the computational object details.
        
        Returns:
            String representation of the symbolic form (self.sym)
        """
        return repr(self.sym)
    
    def _repr_latex_(self):
        """
        Jupyter notebook LaTeX representation.
        
        Returns the LaTeX representation of the symbol for pretty display
        in Jupyter notebooks.
        
        Returns:
            LaTeX string representation of the symbolic form
        """
        from sympy import latex
        return f"$${latex(self.sym)}$$"
    
    def _ipython_display_(self):
        """
        IPython/Jupyter display hook.
        
        Override the default display behavior to show the mathematical symbol
        instead of the object details view.
        
        This is what gets called when a variable is evaluated in a notebook cell.
        """
        from IPython.display import display, Math
        from sympy import latex
        
        # Display the LaTeX representation of the symbol
        display(Math(latex(self.sym)))
    
    
    def diff(self, *args, **kwargs):
        """
        Direct differentiation.
        
        Convenience method that operates on the symbolic form directly.
        
        Args:
            *args: Arguments passed to SymPy diff method
            **kwargs: Keyword arguments passed to SymPy diff method
            
        Returns:
            Pure SymPy expression suitable for JIT compilation
            
        Example:
            velocity = MeshVariable("velocity", mesh, 2)
            x, y = mesh.X
            dv_dx = velocity.diff(x)  # Equivalent to velocity.sym.diff(x)
        """
        return self.sym.diff(*args, **kwargs)
    
    # Arithmetic operations with smart scalar broadcasting
    def __add__(self, other):
        """
        Addition with scalar broadcasting: var + other
        
        Enables natural addition with scalars:
        - velocity + 2  # Broadcasts scalar to matrix shape
        - temperature + 273.15  # Add offset to all components
        
        For non-scalar addition, falls back to SymPy's normal behavior.
        """
        try:
            # Try normal SymPy addition first
            return self.sym + other
        except (TypeError, ValueError):
            # If that fails, try scalar broadcasting
            other_sym = sympy.sympify(other)
            if other_sym.is_number and hasattr(self.sym, 'shape'):
                # Broadcast scalar to matrix shape
                broadcasted = other_sym * sympy.ones(*self.sym.shape)
                return self.sym + broadcasted
            # If broadcasting doesn't apply, re-raise the original error
            return self.sym + other
    
    def __radd__(self, other):
        """
        Right addition with scalar broadcasting: other + var
        
        Since addition is commutative, delegate to __add__.
        """
        return self.__add__(other)
    
    def __sub__(self, other):
        """
        Subtraction with scalar broadcasting: var - other
        
        Enables natural subtraction with scalars:
        - pressure - 101325  # Subtract atmospheric pressure
        - temperature - 273.15  # Convert to Celsius
        """
        try:
            # Try normal SymPy subtraction first
            return self.sym - other
        except (TypeError, ValueError):
            # If that fails, try scalar broadcasting
            other_sym = sympy.sympify(other)
            if other_sym.is_number and hasattr(self.sym, 'shape'):
                # Broadcast scalar to matrix shape
                broadcasted = other_sym * sympy.ones(*self.sym.shape)
                return self.sym - broadcasted
            # If broadcasting doesn't apply, re-raise the original error
            return self.sym - other
    
    def __rsub__(self, other):
        """
        Right subtraction with scalar broadcasting: other - var
        
        Enables expressions like: 100 - temperature
        """
        try:
            # Try normal SymPy subtraction first
            return other - self.sym
        except (TypeError, ValueError):
            # If that fails, try scalar broadcasting
            other_sym = sympy.sympify(other)
            if other_sym.is_number and hasattr(self.sym, 'shape'):
                # Broadcast scalar to matrix shape
                broadcasted = other_sym * sympy.ones(*self.sym.shape)
                return broadcasted - self.sym
            # If broadcasting doesn't apply, re-raise the original error
            return other - self.sym
    
    def __mul__(self, other):
        """Multiplication: var * other"""
        return self.sym * other
    
    def __rmul__(self, other):
        """Right multiplication: other * var"""
        return other * self.sym
    
    def __truediv__(self, other):
        """Division: var / other"""
        return self.sym / other
    
    def __rtruediv__(self, other):
        """Right division: other / var"""
        return other / self.sym
    
    def __pow__(self, other):
        """Power: var ** other"""
        return self.sym ** other
    
    def __rpow__(self, other):
        """Right power: other ** var"""
        return other ** self.sym
    
    def __neg__(self):
        """Negation: -var"""
        return -self.sym
    
    def __pos__(self):
        """Positive: +var"""
        return +self.sym
    
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
        """
        Delegate any missing attributes to the symbolic form with smart argument conversion.
        
        This enables full SymPy Matrix functionality without having to
        implement every method individually. Methods like .dot(), .T, 
        .norm(), .cross(), etc. are automatically available.
        
        For callable methods, arguments that are MathematicalMixin objects
        are automatically converted to their .sym form.
        
        Args:
            name: Attribute name being accessed
            
        Returns:
            The attribute from self.sym, or a wrapper for callable methods
            
        Example:
            velocity = MeshVariable("velocity", mesh, 2)
            velocity.T             # Returns velocity.sym.T (transpose)
            velocity.dot(velocity) # Works! Converts to velocity.sym.dot(velocity.sym)
            velocity.norm()        # Returns velocity.sym.norm()
        """
        # Only delegate if the attribute exists on the symbolic form
        if hasattr(self.sym, name):
            attr = getattr(self.sym, name)
            
            # If it's a callable method, wrap it to handle MathematicalMixin arguments
            if callable(attr):
                def method_wrapper(*args, **kwargs):
                    # Convert any MathematicalMixin arguments to their .sym form
                    converted_args = []
                    for arg in args:
                        if hasattr(arg, '_sympify_'):
                            # This is a MathematicalMixin object, use its symbolic form
                            converted_args.append(arg.sym)
                        else:
                            converted_args.append(arg)
                    
                    converted_kwargs = {}
                    for key, value in kwargs.items():
                        if hasattr(value, '_sympify_'):
                            converted_kwargs[key] = value.sym
                        else:
                            converted_kwargs[key] = value
                    
                    # Call the original method with converted arguments
                    return attr(*converted_args, **converted_kwargs)
                
                return method_wrapper
            else:
                # If it's a property, return its value directly
                return attr
        
        # If attribute doesn't exist on sym, let Python raise its normal AttributeError
        # by re-raising the attribute lookup on self (which will fail naturally)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


# Convenience function for testing
def test_mathematical_mixin():
    """
    Test function to verify MathematicalMixin functionality.
    
    This function can be used to test the mixin with mock objects
    that have a .sym property.
    """
    
    class MockVariable(MathematicalMixin):
        """Mock variable for testing"""
        def __init__(self, name, sym_value):
            self.name = name
            self.sym = sym_value
        
        def __repr__(self):
            # Call MathematicalMixin's __repr__
            return super().__repr__()
    
    # Create test symbols
    x, y = sympy.symbols('x y')
    # Use a Matrix for testing matrix methods
    test_sym = sympy.Matrix([sympy.Function('V_0')(x, y), sympy.Function('V_1')(x, y)])
    
    # Create mock variable
    mock_var = MockVariable("test", test_sym)
    
    # Test _sympify_
    assert mock_var._sympify_() == test_sym
    
    # Test arithmetic operations
    result_mul = 2 * mock_var
    expected_mul = 2 * test_sym
    assert result_mul.equals(expected_mul)
    
    result_rmul = mock_var * 3
    expected_rmul = test_sym * 3
    assert result_rmul.equals(expected_rmul)
    
    # Test scalar broadcasting for addition
    result_add = mock_var + 1
    expected_add = test_sym + sympy.ones(*test_sym.shape)
    assert result_add.equals(expected_add)
    
    # Test right addition broadcasting
    result_radd = 2 + mock_var  
    expected_radd = test_sym + 2 * sympy.ones(*test_sym.shape)
    assert result_radd.equals(expected_radd)
    
    # Test scalar broadcasting for subtraction
    result_sub = mock_var - 3
    expected_sub = test_sym - 3 * sympy.ones(*test_sym.shape)
    assert result_sub.equals(expected_sub)
    
    # Test right subtraction broadcasting  
    result_rsub = 5 - mock_var
    expected_rsub = 5 * sympy.ones(*test_sym.shape) - test_sym
    assert result_rsub.equals(expected_rsub)
    
    result_neg = -mock_var
    expected_neg = -test_sym
    assert result_neg.equals(expected_neg)
    
    # Test delegated matrix methods via __getattr__
    # Test transpose
    result_T = mock_var.T
    expected_T = test_sym.T
    assert result_T.equals(expected_T)
    
    # Test norm (should work for Matrix)
    result_norm = mock_var.norm()
    expected_norm = test_sym.norm()
    assert result_norm.equals(expected_norm)
    
    # Test dot product with another matrix
    other_matrix = sympy.Matrix([1, 2])  # Compatible for dot product
    result_dot = mock_var.dot(other_matrix)
    expected_dot = test_sym.dot(other_matrix)
    assert result_dot.equals(expected_dot)
    
    # Test dot product with another MathematicalMixin object (the key test!)
    mock_var2 = MockVariable("test2", test_sym)
    result_dot_self = mock_var.dot(mock_var2)
    expected_dot_self = test_sym.dot(test_sym)
    assert result_dot_self.equals(expected_dot_self)
    
    print("MathematicalMixin tests passed!")
    print("✓ Basic arithmetic operations")
    print("✓ Scalar broadcasting for addition and subtraction")  
    print("✓ Matrix methods via __getattr__ delegation")
    print("✓ Smart argument conversion for methods (T.dot(T) works!)")
    
    return True


if __name__ == "__main__":
    # Run tests if module is executed directly
    test_mathematical_mixin()