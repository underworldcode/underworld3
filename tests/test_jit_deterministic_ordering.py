import pytest


pytestmark = pytest.mark.level_1


class _FakeField:
    def __init__(self, name, field_id):
        self.name = name
        self.clean_name = name
        self.field_id = field_id

    def __str__(self):
        return self.name


class _FakeSymbol:
    def __init__(self, name, ccode):
        self.name = name
        self._ccodestr = ccode

    def __str__(self):
        return self.name


def test_stable_sorted_prefers_field_id_for_mesh_variables():
    from underworld3.utilities._jitextension import _stable_sorted

    pressure = _FakeField("Pressure", 1)
    velocity = _FakeField("Velocity", 0)

    assert _stable_sorted([pressure, velocity]) == [velocity, pressure]


def test_stable_sorted_is_independent_of_input_container_order():
    from underworld3.utilities._jitextension import _stable_sorted

    symbols = {
        _FakeSymbol("gamma", "petsc_x[2]"),
        _FakeSymbol("alpha", "petsc_x[0]"),
        _FakeSymbol("beta", "petsc_x[1]"),
    }

    assert [symbol.name for symbol in _stable_sorted(symbols)] == [
        "alpha",
        "beta",
        "gamma",
    ]
