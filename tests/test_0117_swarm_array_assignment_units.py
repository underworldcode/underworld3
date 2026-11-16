"""Assigning to ``SwarmVariable.array`` converts units, as ``MeshVariable.array``
does (#993).

The setter wrote the value's magnitude straight into ``.data``, which is
non-dimensional, so a unit-aware 500 K, or a plain array of 500, stored 500
where 0.5 is right (reference temperature 1000 K). It now goes through the
view's ``__setitem__``: a unit-aware value is converted and a plain array is
refused as ambiguous, as for a MeshVariable.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.unit_aware_array import UnitAwareArray

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

_SHAPES = {
    "s": dict(size=1, vtype=uw.VarType.SCALAR),
    "v": dict(size=2, vtype=uw.VarType.VECTOR),
    "m": dict(size=(2, 3), vtype=uw.VarType.MATRIX),
    "t": dict(size=(2, 2), vtype=uw.VarType.TENSOR),
    "y": dict(size=(2, 2), vtype=uw.VarType.SYM_TENSOR),
}


@pytest.fixture
def swarm_vars():
    uw.get_default_model().set_reference_quantities(
        length=uw.quantity(1000, "km"), time=uw.quantity(1, "Myr"),
        mass=uw.quantity(1e24, "kg"), temperature=uw.quantity(1000, "K"),
    )
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.25, regular=True, qdegree=2)
    sw = uw.swarm.Swarm(mesh)
    vs = {k: uw.swarm.SwarmVariable(f"{k}_993", sw, kw["size"], vtype=kw["vtype"],
                                   units="K", _proxy=False)
          for k, kw in _SHAPES.items()}
    sw.populate(fill_param=1)
    yield vs
    uw.reset_default_model()


def test_a_plain_array_is_refused_as_for_a_mesh_variable(swarm_vars):
    """A plain array is ambiguous (kelvin, or non-dimensional?), and MeshVariable
    refuses it; it used to be stored as non-dimensional, 1000 times too large."""
    v = swarm_vars["s"]
    v.data[:] = 0.5
    with pytest.raises(ValueError, match="plain array"):
        v.array = np.asarray(v.array)
    assert np.asarray(v.data) == pytest.approx(0.5)


@pytest.mark.parametrize("key", list(_SHAPES))
def test_a_unit_aware_value_is_converted(swarm_vars, key):
    v = swarm_vars[key]
    v.data[:] = 0.0
    v.array = UnitAwareArray(np.full(np.asarray(v.array).shape, 500.0), units="K")
    assert np.asarray(v.data) == pytest.approx(0.5)
