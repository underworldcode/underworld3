"""Regression tests for the retired ``swarm.points`` writable stack (#379).

``swarm.points`` kept a private writable wrapper whose per-write callback
ran collective particle migration — ranks writing unevenly deadlocked in
parallel — and raised a bare rank-local ``RuntimeError`` on any size
mismatch. Reading it could also trigger a collective (a forced migration
after mesh changes), so a read performed on some ranks only could hang.

It now receives the same treatment as ``mesh.points`` (BF-18): the
deprecated read returns a detached, read-only snapshot with no side
effects; the setter refuses loudly and points at ``swarm.coords`` and
``swarm._particle_coordinates.data``.
"""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.fixture()
def swarm():
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
    )
    swarm = uw.swarm.Swarm(mesh=mesh)
    swarm.populate(fill_param=2)
    return swarm


def test_points_read_is_deprecated_snapshot(swarm):
    with pytest.warns(DeprecationWarning, match="swarm.points is deprecated"):
        pts = swarm.points
    assert pts.shape == (swarm.local_size, swarm.mesh.dim)
    assert not pts.flags.writeable


def test_points_snapshot_write_raises(swarm):
    with pytest.warns(DeprecationWarning):
        pts = swarm.points
    with pytest.raises(ValueError, match="read-only"):
        pts[0, 0] = 99.0


def test_points_setter_raises_with_guidance(swarm):
    replacement = np.asarray(swarm._particle_coordinates.data)
    with pytest.raises(AttributeError, match="swarm.coords"):
        swarm.points = replacement + 0.01


def test_points_refusal_leaves_coordinates_intact(swarm):
    before = np.asarray(swarm._particle_coordinates.data).copy()
    with pytest.raises(AttributeError):
        swarm.points = before + 0.5
    assert np.array_equal(np.asarray(swarm._particle_coordinates.data), before)


def test_coords_is_the_supported_path(swarm):
    """The sanctioned interfaces carry the write swarm.points used to:
    full-array via the coords setter, masked via _particle_coordinates.data
    under migration_control()."""
    shift = 1.0e-3
    original = np.asarray(swarm._particle_coordinates.data).copy()

    swarm.coords = original + shift
    moved = np.asarray(swarm._particle_coordinates.data)
    assert np.allclose(moved, original + shift)

    # The masked idiom migration_control() documents
    coords = swarm._particle_coordinates.data
    mask = np.zeros(coords.shape[0], dtype=bool)
    mask[: coords.shape[0] // 2] = True
    with swarm.migration_control():
        coords[mask] -= shift
    after = np.asarray(swarm._particle_coordinates.data)
    assert np.allclose(after[mask], original[mask])
    assert np.allclose(after[~mask], original[~mask] + shift)
