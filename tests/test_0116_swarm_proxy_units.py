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
