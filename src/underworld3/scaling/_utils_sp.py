from itertools import chain
from collections import OrderedDict

from sympy.physics.units import convert_to, Quantity, Dimension
from sympy.physics.units.systems.si import SI

def unit(expr):
    """Remove all non-unit symbols from a SymPy expression."""
    return expr.subs({x: 1 for x in expr.args if not x.has(Quantity)})

def expr_dimension(expr):
    """
    Computes the physical dimension of a SymPy units expression.
    Works for composite expressions (products, powers, quotients).
    """
    if hasattr(expr, 'dimension'):
        return expr.dimension
    if expr.is_Pow:
        base_dim = expr_dimension(expr.base)
        return base_dim ** expr.exp
    if expr.is_Mul:
        dim = Dimension(1)
        for arg in expr.args:
            dim *= expr_dimension(arg)
        return dim
    if expr.is_Number:
        return Dimension(1)
    raise ValueError(f"Cannot determine dimension for {expr}")

def ensure_lower(maybe_str):
    """dict keys can be any hashable object - only call lower if str"""
    return maybe_str.lower() if isinstance(maybe_str, str) else maybe_str

def ensure_to_base_units(val):
    return convert_to(val, SI._base_units)

_RaiseKeyError = object()  # singleton for no-default behavior

class TransformedDict(dict):
    __slots__ = ()

    @staticmethod
    def _process_args(mapping=(), **kwargs):
        if hasattr(mapping, 'items'):
            mapping = mapping.items()
        return ((ensure_lower(k), ensure_to_base_units(v)) for k, v in chain(mapping, kwargs.items()))

    def __init__(self, mapping=(), **kwargs):
        super().__init__(self._process_args(mapping, **kwargs))

    def __getitem__(self, k):
        return super().__getitem__(ensure_lower(k))

    def __setitem__(self, k, v):
        return super().__setitem__(ensure_lower(k), ensure_to_base_units(v))

    def __delitem__(self, k):
        return super().__delitem__(ensure_lower(k))

    def get(self, k, default=None):
        return super().get(ensure_lower(k), default)

    def setdefault(self, k, default=None):
        return super().setdefault(ensure_lower(k), default)

    def pop(self, k, v=_RaiseKeyError):
        if v is _RaiseKeyError:
            return super().pop(ensure_lower(k))
        return super().pop(ensure_lower(k), v)

    def update(self, mapping=(), **kwargs):
        super().update(self._process_args(mapping, **kwargs))

    def __contains__(self, k):
        return super().__contains__(ensure_lower(k))

    def copy(self):
        return type(self)(self)

    @classmethod
    def fromkeys(cls, keys, v=None):
        return super().fromkeys((ensure_lower(k) for k in keys), v)

    def _repr_html_(self):
        attributes = OrderedDict()
        for key in ["[mass]", "[length]", "[temperature]", "[time]", "[substance]"]:
            attributes[key] = self.get(key, "")
        header = (
            "<table style='border-collapse:collapse;'>"
            "<tr><th style='padding:4px 8px;border:1px solid #ccc;'>Dimension</th>"
            "<th style='padding:4px 8px;border:1px solid #ccc;'>Value</th></tr>"
        )
        footer = "</table>"
        html = ""
        for key, val in attributes.items():
            # Format value: try to split number and unit for better display
            if hasattr(val, 'args') and len(val.args) == 2:
                mag, unit = val.args
                try:
                    mag_str = f"{float(mag):.3e}"
                except Exception:
                    mag_str = str(mag)
                val_str = f"{mag_str} {unit}"
            else:
                val_str = str(val)
            html += (
                f"<tr>"
                f"<td style='padding:4px 8px;border:1px solid #ccc;'>{key}</td>"
                f"<td style='padding:4px 8px;border:1px solid #ccc;'>{val_str}</td>"
                f"</tr>"
            )
        return header + html + footer
