"""Swarm proxy variables transfer a linear field without smearing it.

A proxy mesh variable is refreshed from the particles by local RBF
interpolation. That transfer used to be inverse-distance (Shepard) weighting,
which reproduces constants but not gradients; the proxy default is now
``order=1``, which reproduces both.

Why a *linear* field is the right probe: it lies exactly inside both the P1
and the P2 proxy space, so the finite element discretisation contributes
nothing at all to the measured error. Everything left is particle -> node
transfer error, which makes the assertion below unambiguous.

Measured on this configuration when the default was changed (2026-07-27):
inverse distance gave a relative max error of ~5e-3 on the linear field and
~9e-3 on a quadratic one; the linear-exact scheme gives round-off and ~1e-4
respectively.
"""

import numpy as np
import pytest

import underworld3 as uw
from underworld3.meshing import UnstructuredSimplexBox

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _linear(coords):
    return 0.5 + coords @ np.arange(1, coords.shape[1] + 1, dtype=float)


def _quadratic(coords):
    return 0.5 + (coords ** 2).sum(axis=1)


@pytest.fixture
def proxied_swarm():
    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8.0
    )
    swarm = uw.swarm.Swarm(mesh)
    var = swarm.add_variable(name="f", size=1, proxy_degree=1)
    swarm.populate(fill_param=4)
    yield swarm, var
    del swarm
    del mesh


def _proxy_error(var, field, **refresh):
    """Refresh the proxy from an analytic field and return its relative error."""
    proxy = var._meshVar
    particle_coords = var.swarm._particle_coordinates.data.copy()
    node_coords = np.asarray(proxy.coords)

    var.data[:, 0] = field(particle_coords)
    var._rbf_to_meshVar(proxy, **refresh)

    expected = field(node_coords)
    return np.abs(np.asarray(proxy.data[:, 0]) - expected).max() / np.abs(expected).max()


def test_proxy_default_reproduces_a_linear_field(proxied_swarm):
    """The shipped default must be linear-exact — no arguments passed."""
    _, var = proxied_swarm
    error = _proxy_error(var, _linear)
    assert error < 1.0e-12, (
        f"proxy of an exactly linear field carries a relative error of "
        f"{error:.3e}; the default transfer is not linear-exact"
    )


def test_inverse_distance_smears_the_same_linear_field(proxied_swarm):
    """The control arm: without order=1 the error is real and much larger.

    This is what stops the test above passing for the wrong reason (e.g. a
    tolerance that any scheme would meet on this mesh).
    """
    _, var = proxied_swarm
    exact = _proxy_error(var, _linear, order=1)
    shepard = _proxy_error(var, _linear, order=0, nnn=3)

    assert shepard > 1.0e-4, (
        f"inverse distance is expected to smear a linear field, got {shepard:.3e}"
    )
    assert shepard > 1.0e6 * max(exact, 1.0e-16)


def test_proxy_default_improves_a_quadratic_field(proxied_swarm):
    """Linear exactness is not a trick that only helps linear fields."""
    _, var = proxied_swarm
    exact = _proxy_error(var, _quadratic, order=1)
    shepard = _proxy_error(var, _quadratic, order=0, nnn=3)

    assert exact < shepard / 10.0, (
        f"order=1 gave {exact:.3e} vs inverse distance {shepard:.3e} on a "
        "quadratic field; expected at least a 10x improvement"
    )


def test_proxy_still_reproduces_a_constant_exactly(proxied_swarm):
    """Constants were already exact under inverse distance — do not regress."""
    _, var = proxied_swarm
    error = _proxy_error(var, lambda coords: np.full(coords.shape[0], 3.25))
    assert error < 1.0e-12, f"constant field carries error {error:.3e}"


def test_proxy_monotone_keeps_the_linear_field_exact(proxied_swarm):
    """Turning the limiter on must not cost the proxy its exactness.

    Proxy nodes routinely sit outside the convex hull of their own particle
    stencil — not only at the domain boundary — so a limiter that clipped
    against the stencil's raw min/max would fire on the linear part. This one
    limits only the non-affine correction.
    """
    _, var = proxied_swarm
    limited = _proxy_error(var, _linear, monotone=True)
    assert limited < 1.0e-12, (
        f"proxy with the limiter on carries {limited:.3e} on an exactly linear "
        "field; the limiter is clipping the linear reconstruction"
    )
