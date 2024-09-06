import sympy
from sympy import Symbol, simplify, Number
from underworld3.utilities._api_tools import uw_object


def _substitute_all_once(fn, keep_constants=True, return_self=True):

    if keep_constants and return_self and is_constant_expr(fn):
        if isinstance(fn, UWexpression):
            return fn.sym
        else:
            return fn

    expr = fn
    for atom in fn.atoms():
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


def substitute(fn, keep_constants=True, return_self=True):
    expr = fn
    expr_s = _substitute_all_once(expr, keep_constants, return_self)
    while expr is not expr_s:
        expr = expr_s
        expr_s = _substitute_all_once(expr, keep_constants)

    return expr


def substitute_expr(fn, sub_expr, keep_constants=True, return_self=True):
    expr = fn
    expr_s = _substitute_one_expr(expr, sub_expr, keep_constants)

    while expr is not expr_s:
        expr = expr_s
        expr_s = _substitute_one_expr(expr, sub_expr, keep_constants)
    return expr


def is_constant_expr(fn):

    deps = extract_expressions(fn)

    for dep in deps:
        if not isinstance(dep.sym, (float, int, sympy.Number)):
            return False

        return True


def extract_expressions(fn):

    subbed_expr = substitute(fn, keep_constants=True, return_self=False)
    return subbed_expr.atoms(sympy.Symbol)


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

        invisible = r"\,\!" * instance_no
        unique_name = f"{{ {{ {invisible} }} {name} }}"
        obj = Symbol.__new__(cls, unique_name)
        obj._instance_no = instance_no

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

        invisible = r"\,\!" * self._instance_no
        self.symbol = f"{{ {{ {invisible} }} {name} }}"
        self.sym = sympy.sympify(sym)
        self.description = description

        # this is not being honoured by sympy Symbol
        #
        self._uw_id = uw_object._obj_count
        uw_object._obj_count += 1

        return

    def copy(self, other):
        if not isinstance(other, UWexpression):
            raise ValueError
        else:
            self.symbol = other.symbol
            self.sym = other.sym
            self.description = other.description

        return

    def constant(self):

        deps = self.dependencies()

        for dep in deps:
            if not isinstance(dep.sym, (float, int, sympy.Number)):
                return False

        return True

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

    def sub_all(self, keep_constants=True):
        self_s = substitute(self, keep_constants=keep_constants)

        return self_s

    def sub_expr(self, expr, keep_constants=True):
        self_s = substitute_expr(self, expr, keep_constants=keep_constants)

        return self_s

    def dependencies(self, keep_constants=True):
        subbed_expr = substitute(self.sym, keep_constants=keep_constants)
        return subbed_expr.atoms(sympy.Symbol)

    def all_dependencies(self, keep_constants=True):
        subbed_expr = substitute(self.sym, keep_constants=keep_constants)
        return subbed_expr.atoms(sympy.Symbol, sympy.Function)

    # def _substitute_all_once(fn, keep_constants=True):
    #     expr = fn
    #     for atom in fn.atoms():
    #         if isinstance(atom, UWexpression):
    #             if keep_constants and isinstance(atom.value, (float, int, Number)):
    #                 continue
    #             else:
    #                 expr = expr.subs(atom, atom.value)

    #     return expr

    # def substitute(fn, keep_constants=True):
    #     expr = fn
    #     expr_s = _substitute_all_once(expr, keep_constants)
    #     while expr is not expr_s:
    #         expr = expr_s
    #         expr_s = cls._substitute_all_once(expr, keep_constants)
    #     return expr

    # def _substitute_one_expr(fn, sub_expr, keep_constants=True):
    #     expr = fn
    #     for atom in fn.atoms():
    #         if atom is sub_expr:
    #             if keep_constants and isinstance(atom.value, (float, int)):
    #                 continue
    #             else:
    #                 expr = expr.subs(atom, atom.value)

    #     return expr

    # def substitute_expr(fn, sub_expr, keep_constants=True):
    #     expr = fn
    #     expr_s = _substitute_one_expr(expr, sub_expr, keep_constants)

    #     while expr is not expr_s:
    #         expr = expr_s
    #         expr_s = cls._substitute_one_expr(expr, sub_expr, keep_constants)
    #     return expr

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
