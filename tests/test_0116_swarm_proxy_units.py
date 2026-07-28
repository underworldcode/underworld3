"""Swarm proxy variables must refresh when the model carries units.

Regression test for issue #426.

The swarm kd-tree is built from `Swarm._particle_coordinates.data`, which is
always **non-dimensional**. The proxy refresh used to query it with
`MeshVariable.coords`, which **dimensionalises** as soon as the model has
reference quantities set. `KDTree._convert_coords_to_tree_units` then raised

    ValueError: KD-tree was built with dimensionless coordinates, but query
    coordinates have units 'meter'. Convert to dimensionless first.

so proxied swarm variables did not work at all under an active units model —
including `IndexSwarmVariable` material level sets. The fix is `.coords_nd`.

It failed loudly rather than silently corrupting, which is why it survived: no
test set reference quantities and then touched a proxy.
"""

import numpy as np
import pytest

import underworld3 as uw
from underworld3.meshing import UnstructuredSimplexBox

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


@pytest.fixture
def units_model():
    """A model with reference quantities active, torn down afterwards."""
    orchestration_model = uw.get_default_model()
    orchestration_model.set_reference_quantities(
        length=uw.quantity(1000.0, "km"),
        viscosity=uw.quantity(1e21, "Pa*s"),
    )
    yield orchestration_model
    uw.reset_default_model()


@pytest.fixture
def mesh(units_model):
    return UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8.0
    )


def test_proxy_refresh_under_active_units(mesh):
    """The plain SwarmVariable proxy path."""
    swarm = uw.swarm.Swarm(mesh)
    var = swarm.add_variable(name="rho", size=1, proxy_degree=1, units="kg/m^3")
    swarm.populate(fill_param=3)

    var.data[:, 0] = 3300.0
    var._update_proxy_if_stale()

    values = np.asarray(var._meshVar.data[:, 0])
    assert np.allclose(values, 3300.0, rtol=1e-10), (
        "a constant field did not survive the proxy refresh under units"
    )

    del swarm


def test_index_swarm_proxy_refresh_under_active_units(mesh):
    """The IndexSwarmVariable level-set path, which has its own kd-tree code."""
    swarm = uw.swarm.Swarm(mesh)
    material = uw.swarm.IndexSwarmVariable("mat", swarm, indices=2, proxy_degree=1)
    swarm.populate(fill_param=3)

    coords = swarm._particle_coordinates.data
    material.data[:, 0] = np.where(coords[:, 0] < 0.5, 0, 1)
    material._update_proxy_variables()

    level_sets = [np.asarray(v.data[:, 0]) for v in material._meshLevelSetVars]

    # Partition of unity: the level sets share a denominator by construction,
    # and the multi-material constitutive path divides by their sum.
    total = sum(level_sets)
    assert np.allclose(total, 1.0, atol=1e-8), (
        f"level sets do not sum to one (range {total.min():.3e}..{total.max():.3e})"
    )
    for i, ls in enumerate(level_sets):
        assert ls.min() >= -1e-12 and ls.max() <= 1.0 + 1e-12, (
            f"level set {i} left [0, 1]: {ls.min():.3e}..{ls.max():.3e}"
        )

    del swarm


def test_proxy_refresh_still_works_without_units():
    """The dimensionless path must be unaffected by the fix."""
    uw.reset_default_model()
    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8.0
    )
    swarm = uw.swarm.Swarm(mesh)
    var = swarm.add_variable(name="f", size=1, proxy_degree=1)
    swarm.populate(fill_param=3)

    var.data[:, 0] = 2.5
    var._update_proxy_if_stale()

    assert np.allclose(np.asarray(var._meshVar.data[:, 0]), 2.5, rtol=1e-10)

    del swarm
    del mesh


# ---------------------------------------------------------------------------
# The proxy advertises the swarm variable's units (issue #439).
#
# The proxy MeshVariable is what `var.sym` resolves to, so if it carries no
# units then reading a proxied variable through the symbolic path silently
# returns the NON-DIMENSIONAL number as though it were the answer. Measured
# before the fix, with density=3300 kg/m^3 as the reference and a stored
# value of 1.0:
#
#     swarm.array 3300.0   proxy.array 1.0        <- disagree
#     evaluate(rho.sym) -> ndarray 1.0, no units
#
# Stored data is non-dimensional on both sides either way; only what the
# proxy declares changes.
# ---------------------------------------------------------------------------
@pytest.fixture
def dimensional_model():
    orchestration_model = uw.get_default_model()
    orchestration_model.set_reference_quantities(
        length=uw.quantity(1000.0, "km"),
        viscosity=uw.quantity(1e21, "Pa*s"),
        density=uw.quantity(3300.0, "kg/m^3"),
    )
    yield orchestration_model
    uw.reset_default_model()


@pytest.fixture
def dimensional_mesh(dimensional_model):
    return UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8.0
    )


def test_proxy_inherits_the_swarm_variable_units(dimensional_mesh):
    swarm = uw.swarm.Swarm(dimensional_mesh)
    var = swarm.add_variable(name="rho", size=1, proxy_degree=1, units="kg/m^3")
    swarm.populate(fill_param=3)

    assert var._meshVar.units == var.units, (
        f"proxy advertises {var._meshVar.units}, variable advertises {var.units}"
    )

    del swarm


def test_proxy_and_swarm_dimensional_views_agree(dimensional_mesh):
    """`.array` dimensionalises; both sides must reach the same number."""
    swarm = uw.swarm.Swarm(dimensional_mesh)
    var = swarm.add_variable(name="rho", size=1, proxy_degree=1, units="kg/m^3")
    swarm.populate(fill_param=3)

    var.data[:, 0] = 1.0                      # non-dimensional 1.0 == 3300 kg/m^3
    var._update_proxy_if_stale()

    swarm_value = np.asarray(var.array).ravel()[0]
    proxy_value = np.asarray(var._meshVar.array).ravel()[0]

    assert np.isclose(swarm_value, 3300.0, rtol=1e-10)
    assert np.isclose(proxy_value, swarm_value, rtol=1e-10), (
        f"swarm .array {swarm_value:.6g} but proxy .array {proxy_value:.6g}"
    )
    # Storage stays non-dimensional on both sides.
    assert np.isclose(np.asarray(var._meshVar.data[0, 0]), 1.0, rtol=1e-10)

    del swarm


def test_evaluating_a_proxied_symbol_returns_dimensional_values(dimensional_mesh):
    """The silent-wrong-value case: evaluate used to return the raw 1.0."""
    from underworld3.utilities.unit_aware_array import UnitAwareArray

    swarm = uw.swarm.Swarm(dimensional_mesh)
    var = swarm.add_variable(name="rho", size=1, proxy_degree=1, units="kg/m^3")
    swarm.populate(fill_param=3)

    var.data[:, 0] = 1.0
    values = uw.function.evaluate(var.sym[0], np.array([[0.5, 0.5], [0.25, 0.75]]))

    assert isinstance(values, UnitAwareArray), (
        f"evaluate returned {type(values).__name__}; a units-carrying proxy "
        "must not present non-dimensional numbers as the answer"
    )
    assert str(values.units) == "kilogram / meter ** 3"
    assert np.allclose(np.asarray(values).ravel(), 3300.0, rtol=1e-8)

    del swarm


def test_variable_without_units_keeps_a_dimensionless_proxy(dimensional_mesh):
    """Only variables that declare units get a units-carrying proxy."""
    swarm = uw.swarm.Swarm(dimensional_mesh)
    var = swarm.add_variable(name="f", size=1, proxy_degree=1)
    swarm.populate(fill_param=3)

    assert var._meshVar.units is None

    var.data[:, 0] = 1.0
    var._update_proxy_if_stale()
    assert np.isclose(np.asarray(var._meshVar.array).ravel()[0], 1.0, rtol=1e-10)

    del swarm
