"""Units of a nested named expression cost one visit per level (#1080).

``UWexpression.units`` asked ``hasattr(self._sym, 'units')`` and then read
``self._sym.units``: two evaluations of the inner expression's property per
level, so ``uw.get_units`` doubled with each level of nesting (1.0 s at depth
20). ``dimensionality``, ``value``, ``data``, ``is_extended_real`` and
``is_finite`` had the same pattern (9.7 s at depth 20 for ``is_extended_real``).
The test counts the properties' calls rather than timing them.
"""
import pytest

import underworld3 as uw
from underworld3.function.expressions import UWexpression


@pytest.mark.level_1
@pytest.mark.tier_b
def test_units_of_a_nested_expression_visit_each_level_once(monkeypatch):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.5)
    f = uw.discretisation.MeshVariable("f_1080", mesh, 1, degree=1)
    depth = 16
    e = f.sym[0]
    for k in range(depth):
        e = uw.expression(rf"n_{{1080,{k}}}", e, "nested")

    calls = {"units": 0, "dimensionality": 0, "value": 0, "data": 0,
             "is_extended_real": 0, "is_finite": 0}
    for name in calls:
        prop = getattr(UWexpression, name)

        def counted(self, _prop=prop, _name=name):
            calls[_name] += 1
            return _prop.fget(self)

        monkeypatch.setattr(UWexpression, name, property(counted))

    for name in calls:
        getattr(e, name)
    for name, n in calls.items():
        assert n <= 2 * depth, (name, calls)
