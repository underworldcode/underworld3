from sympy import Symbol, simplify
from underworld3.utilities._api_tools import uw_object


class uw_constant(Symbol, uw_object):
    """
    underworld `constants` are sympy symbols with attached
                numeric values that are substituted into an underworld function
                before evaluation. In sympy expressions, the symbol form is shown.

    ```{python}
    alpha = uw_constant(
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
        return obj

    def __init__(self, name, value, description="No description provided"):

        self._value = value
        self._description = description

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

    @classmethod
    def subsitute(cls, fn):
        expr = fn
        for atom in fn.atoms():
            if isinstance(atom, cls):
                expr = expr.subs(atom, atom.value)

        return expr

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        information = f"""
 > ${self}$: {self.value} - {self.description}
        """

        display(Markdown(information))

        return
