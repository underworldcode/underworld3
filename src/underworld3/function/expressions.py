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
        return fn._sympify_()

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
    return unwrap(fn, keep_constants, return_self)


def _unwrap_expressions(fn, keep_constants=True, return_self=True):
    expr = fn
    expr_s = _substitute_all_once(expr, keep_constants, return_self)

    while expr is not expr_s:
        expr = expr_s
        expr_s = _substitute_all_once(expr, keep_constants, return_self)

    return expr


def unwrap(fn, keep_constants=True, return_self=True):
    """
    Unwrap UW expressions to pure SymPy expressions for compilation.

    Args:
        fn: Expression to unwrap
        keep_constants: Whether to preserve constants
        return_self: Whether to return self for constant expressions

    Returns:
        Pure SymPy expression with scale factors applied if scaling context is active
    """
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
    import underworld3 as uw
    if uw._is_scaling_active():
        result = _apply_scaling_to_unwrapped(result)

    return result


def _apply_scaling_to_unwrapped(expr):
    """
    Apply scale factors to an unwrapped SymPy expression.

    This function finds all variable symbols in the expression and replaces them
    with scaled versions by looking up their variables in the model registry.

    Args:
        expr: SymPy expression (potentially with UW variable symbols)

    Returns:
        SymPy expression with scale factors applied to variables with units
    """
    import underworld3 as uw
    import sympy

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


        # For each variable in the model, check if its symbols appear in the expression
        for var_name, variable in model._variables.items():
            if (hasattr(variable, 'has_units') and hasattr(variable, 'scale_factor') and
                variable.has_units and variable.scale_factor is not None):


                # Get the variable's symbol and find matching function symbols
                if hasattr(variable, '_base_var'):
                    # For enhanced variables, get the base variable's symbol
                    var_sym = variable._base_var.sym
                else:
                    var_sym = getattr(variable, 'sym', None)

                if var_sym is not None:
                    # Get all function symbols from the variable's symbol
                    if hasattr(var_sym, 'atoms'):
                        var_function_symbols = var_sym.atoms(sympy.Function)
                    else:
                        var_function_symbols = set()


                    # Find matching symbols and create substitutions
                    for func_symbol in function_symbols:
                        for var_func_symbol in var_function_symbols:
                            if str(func_symbol) == str(var_func_symbol):
                                substitutions[func_symbol] = func_symbol * variable.scale_factor


        # Apply all substitutions
        if substitutions:
            return expr.subs(substitutions)
        else:
            return expr

    except Exception as e:
        import warnings
        warnings.warn(f"Could not apply scaling to unwrapped expression: {e}")
        return expr




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

    def __new__(
        cls,
        name,
        *args,
        _unique_name_generation=False,
        **kwargs,
    ):

        import warnings

        instance_no = UWexpression._expr_count

        ## if the expression already exists, do not replace it (but return the existing object instead)

        if name in UWexpression._expr_names.keys() and _unique_name_generation == False:
            warnings.warn(
                message=f"EXPRESSIONS {name}: Each expression should have a unique name - new expression was not generated",
            )
            return UWexpression._expr_names[name]

        if name in UWexpression._expr_names and _unique_name_generation == True:
            invisible = rf"\hspace{{ {instance_no/100}pt }}"
            unique_name = f"{{ {name} {invisible} }}"
        else:
            unique_name = name

        obj = Symbol.__new__(cls, unique_name)
        obj._instance_no = instance_no
        obj._unique_name = unique_name
        obj._given_name = name

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

        # Initialize UnitAwareMixin attributes directly to avoid MRO conflicts
        from underworld3.utilities.units_mixin import UnitAwareMixin
        UnitAwareMixin.__init__(self)

        # Handle UWQuantity as sym parameter (the beautiful symmetry!)
        if isinstance(sym, UWQuantity):
            if units is not None:
                # Convert quantity to match expression's target units
                if sym.has_units:
                    converted_qty = sym.to(units)
                    quantity_value = converted_qty.value
                    self.set_units(units)  # Set units directly on UnitAwareMixin
                else:
                    # Dimensionless quantity, just use the value
                    quantity_value = sym.value
            else:
                # Use quantity's units directly
                if sym.has_units:
                    quantity_value = sym.value
                    self.set_units(sym.units)  # Set units directly on UnitAwareMixin
                else:
                    quantity_value = sym.value
            # Use the converted value for the expression's symbolic representation
            sym_value = quantity_value
        else:
            # Traditional initialization - set units if provided
            if units is not None:
                self.set_units(units)
            sym_value = sym

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

    def _sympify_(self):
        """
        Override the _sympify_ method to return the pure SymPy representation.

        This prevents infinite recursion in UWQuantity.atoms() method by returning
        self._sym (pure SymPy object without UWQuantity.atoms method) instead of
        self (UWexpression with inherited UWQuantity.atoms method).

        The recursion occurs when:
        1. UWQuantity.atoms() calls self._sympify_()
        2. UWexpression._sympify_() returns self
        3. UWQuantity.atoms() calls atoms() on the result
        4. Since result is still UWexpression, infinite recursion ensues

        Fix: Return self._sym (pure SymPy) to break the recursion chain.
        """
        return self._sym

    def __bool__(self):
        """
        Override boolean evaluation to prevent __len__ calls.

        UWexpression objects should always evaluate to True for boolean
        contexts, just like regular SymPy Symbol objects. This prevents
        SymPy from calling __len__ during boolean evaluation.
        """
        return True

    def copy(self, other):
        if not isinstance(other, UWexpression):
            raise ValueError
        else:
            # Note: sympy symbols are uniquely defined by name and so
            # the uw expressions based on symbols cannot be renamed: only the
            # value can be changed. As a result, copy is just an assignment to
            # self.sym and should be deprecated.

            # self.symbol = other.symbol # Can't change this
            self._sym = other._sym
            # self.description = other.description # Shouldn't change this

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
