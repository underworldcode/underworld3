from sympy import Symbol, simplify
from underworld3.utilities._api_tools import uw_object


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
        obj.value = value
        obj.symbol = name
        return obj

    def __init__(self, name, value, description="No description provided"):

        self._value = value
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

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value
        return

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, new_description):
        self._description = new_description
        return

    def sub_all(self, keep_constants=True):
        cls = self.__class__
        self_s = cls.substitute(self, keep_constants=keep_constants)

        return self_s

    def sub_expr(self, expr, keep_constants=True):
        cls = self.__class__
        self_s = cls._substitute_one_expr(self, expr, keep_constants=keep_constants)

        return self_s

    @classmethod
    def _substitute_all_once(cls, fn, keep_constants=True):
        expr = fn
        for atom in fn.atoms():
            if isinstance(atom, cls):
                if keep_constants and isinstance(atom.value, (float, int)):
                    continue
                else:
                    expr = expr.subs(atom, atom.value)

        return expr

    @classmethod
    def substitute(cls, fn, keep_constants=True):
        expr = fn
        expr_s = cls._substitute_all_once(expr, keep_constants)
        while expr is not expr_s:
            expr = expr_s
            expr_s = cls._substitute_all_once(expr, keep_constants)
        return expr

    @classmethod
    def _substitute_one_expr(cls, fn, sub_expr, keep_constants=True):
        expr = fn
        for atom in fn.atoms():
            if atom is sub_expr:
                if keep_constants and isinstance(atom.value, (float, int)):
                    continue
                else:
                    expr = expr.subs(atom, atom.value)

        return expr

    @classmethod
    def substitute_expr(cls, fn, sub_expr, keep_constants=True):
        expr = fn
        expr_s = cls._substitute_one_expr(expr, sub_expr, keep_constants)

        while expr is not expr_s:
            expr = expr_s
            expr_s = cls._substitute_one_expr(expr, sub_expr, keep_constants)
        return expr

    def _object_viewer(self, description=True, level=1):
        from IPython.display import Latex, Markdown, display
        import sympy

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


# class UWConstant_expression(UWexpression):
