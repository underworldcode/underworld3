import sympy
from sympy import Symbol, simplify, Number
from underworld3.utilities._api_tools import uw_object
from underworld3.discretisation import _MeshVariable


def _substitute_all_once(fn, keep_constants=True, return_self=True):

    if keep_constants and return_self and is_constant_expr(fn):
        if isinstance(fn, UWexpression):
            return fn.sym
        else:
            return fn

    if isinstance(fn, UWexpression):
        expr = fn.sym
    else:
        expr = fn

    for atom in extract_expressions(fn):

        if isinstance(atom, UWexpression):
            if keep_constants and isinstance(atom.sym, (float, int, Number)):
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
        expr_s = _substitute_all_once(expr, keep_constants)

    return expr


def unwrap(fn, keep_constants=True, return_self=True):
    if isinstance(fn, sympy.Matrix):
        f = lambda x: _unwrap_expressions(
            x, keep_constants=keep_constants, return_self=return_self
        )
        return fn.applyfunc(f)
    else:
        return _unwrap_expressions(
            fn, keep_constants=keep_constants, return_self=return_self
        )


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

    atoms = fn.atoms(sympy.Symbol, sympy.Function, sympy.vector.scalar.BaseScalar)

    # exhaustion criterion
    if atoms == fn.atoms():
        return atoms

    for atom in atoms:
        if isinstance(atom, underworld3.function.expression):
            sub_atomic = extract_expressions_and_functions(atom)
            atoms = atoms.union(sub_atomic)

    return atoms


class UWexpression(uw_object, Symbol):
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

    def __new__(
        cls,
        name,
        *args,
        **kwargs,
    ):

        instance_no = UWexpression._expr_count

        invisible = rf"\hspace{{ {instance_no/100}pt }}"
        unique_name = f"{{ {name} {invisible} }}"
        obj = Symbol.__new__(cls, unique_name)
        obj._instance_no = instance_no
        obj._unique_name = unique_name

        UWexpression._expr_count += 1

        return obj

    def __init__(
        self, name, sym=None, description="No description provided", value=None
    ):
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

        self.symbol = self._unique_name
        self.sym = sympy.sympify(sym)
        self.description = description

        # this is not being honoured by sympy Symbol so do it by hand
        self._uw_id = uw_object._obj_count
        uw_object._obj_count += 1

        return

    def copy(self, other):
        if not isinstance(other, UWexpression):
            raise ValueError
        else:
            self.symbol = other.symbol
            self._sym = other._sym
            self.description = other.description

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
        self._sym = sympy.sympify(new_value)
        return

    # TODO: DEPRECATION
    # The value attribute is no longer needed in the offical release of Underworld3
    @property
    def value(self):
        import warnings

        warnings.warn(
            message=f"DEPRECATION warning, don't use 'value' attribute for expression: {self}, please use 'sym' attribute"
        )
        return self._sym

    @value.setter
    def value(self, new_value):
        import warnings

        warnings.warn(
            message=f"DEPRECATION warning, don't use 'value' attribute for expression: {new_value}, please use 'sym' attribute"
        )
        self._sym = sympy.sympify(new_value)
        return

    @property
    def expression(self):
        return self.sub_all()

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, new_description):
        self._description = new_description
        return

    def unwrap(self, keep_constants=True, return_self=True):
        return unwrap(self, keep_constants=keep_constants)

    # # This should be the thing that gets called by sympy when we ask for self.diff()
    # def fdiff(self, variable, evaluate=True):
    #     if evaluate:
    #         return sympy.diff(self.unwrap(keep_constants=False, return_self=False), variable, evaluate=True)
    #     else:
    #         if isinstance(self, (sympy.Matrix, sympy.ImmutableMatrix)):
    #             f = lambda x: sympy.diff(
    #                     x, variable, evaluate=False
    #                 )
    #             return self.applyfunc(f)
    #         else:
    #             return sympy.diff(self, variable, evaluate=False)

    def sub_all(self, keep_constants=True):
        return substitute(self, keep_constants=keep_constants)

    def sub_expr(self, expr, keep_constants=True):
        self_s = substitute_expr(self, expr, keep_constants=keep_constants)

        return self_s

    def dependencies(self, keep_constants=True):
        return extract_expressions(self)

    def all_dependencies(self, keep_constants=True):
        return extract_expressions_and_functions(self)

    def _ipython_display_(self):
        from IPython.display import Latex, Markdown, display

        display(Markdown("$" + self.symbol + "$"))

    def _object_viewer(self, description=True, level=1):
        from IPython.display import Latex, Markdown, display
        import sympy

        level = max(1, level)

        ## feedback on this instance
        if sympy.sympify(self.sym) is not None:
            display(
                Latex(
                    r"$"
                    + r"\quad" * level
                    + "$"
                    + self._repr_latex_()
                    + "$=$"
                    + sympy.sympify(self.sym)._repr_latex_()
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
