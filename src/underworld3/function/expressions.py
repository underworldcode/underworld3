import sympy
from sympy import Symbol, simplify, Number
from underworld3.utilities._api_tools import uw_object


def _substitute_all_once(fn, keep_constants=True, return_self=True):

    if keep_constants and return_self and is_constant_expr(fn):
        if isinstance(fn, UWexpression):
            return fn.value
        else:
            return fn

    expr = fn
    for atom in fn.atoms():
        if isinstance(atom, UWexpression):
            if keep_constants and isinstance(atom.value, (float, int, Number)):
                continue
            else:
                expr = expr.subs(atom, atom.value)

    return expr


def _substitute_one_expr(fn, sub_expr, keep_constants=True, return_self=True):
    expr = fn

    if keep_constants and return_self and is_constant_expr(fn):
        if isinstance(fn, UWexpression):
            return fn.value
        else:
            return fn

    for atom in fn.atoms():
        if atom is sub_expr:
            if keep_constants and isinstance(atom.value, (float, int, Number)):
                continue
            else:
                expr = expr.subs(atom, atom.value)

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
        if not isinstance(dep.value, (float, int, sympy.Number)):
            return False

        return True


def extract_expressions(fn):

    subbed_expr = substitute(fn, keep_constants=True, return_self=False)
    return subbed_expr.atoms(sympy.Symbol)


class UWexpression(Symbol, uw_object):
    """
    underworld `expressions` are sympy symbols with attached
                numeric/expression values that are substituted into an underworld function
                before evaluation. In sympy expressions, the symbol form is shown.

    ```{python}
        alpha = UWexpression(
                        r'\\alpha',
                        value=3.0e-5,
                        description="thermal expansivity"
                            )
        print(alpha.value)
        print(alpha.description)
    ```

    """

    def __new__(cls, name, value, description="No description provided"):

        obj = Symbol.__new__(cls, name)
        obj.value = sympy.sympify(value)
        obj.symbol = name
        return obj

    def __init__(self, name, value, description="No description provided"):

        self._value = sympy.sympify(value)
        self._description = description

        return

    def copy(self, other):
        if not isinstance(other, UWexpression):
            raise ValueError
        else:
            self.symbol = other.symbol
            self.value = other.value
            self.description = other.description

        return

    def constant(self):

        deps = self.dependencies()

        for dep in deps:
            if not isinstance(dep.value, (float, int, sympy.Number)):
                return False

        return True

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = sympy.sympify(new_value)
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
        subbed_expr = substitute(self.value, keep_constants=keep_constants)
        return subbed_expr.atoms(sympy.Symbol)

    def all_dependencies(self, keep_constants=True):
        subbed_expr = substitute(self.value, keep_constants=keep_constants)
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

    def _object_viewer(self, description=True, level=1):
        from IPython.display import Latex, Markdown, display
        import sympy

        level = max(1, level)

        ## feedback on this instance
        if sympy.sympify(self.value) is not None:
            display(
                Latex(
                    r"$"
                    + "\quad" * level
                    + "$"
                    + self._repr_latex_()
                    + "$=$"
                    + sympy.sympify(self.value)._repr_latex_()
                ),
            )
            if description == True:
                display(
                    Markdown(
                        r"$"
                        + "\quad" * level
                        + "$"
                        + f"**Description:**  {self.description}"
                    ),
                )

        try:
            atoms = self.value.atoms()
            for atom in atoms:
                if atom is not self.value:
                    try:
                        atom._object_viewer(description=False, level=level + 1)
                    except AttributeError:
                        pass
        except:
            pass

        return
