"""Regression tests for the removed ``mesh.points`` setter (BF-18).

The deprecated setter ended with ``self._coords = model_coords``, rebinding
the coordinate store to a plain ndarray and silently discarding the
``NDArray_With_Callback`` wrapper — so the deform callback never fired and
PETSc coordinates were never updated. Writes looked accepted but changed
nothing downstream (2026-07 audit, BF-18 / READ-43). The setter now raises
``AttributeError`` pointing at :meth:`Mesh.deform`.
"""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _make_mesh():
    return uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
    )


def test_points_setter_raises_with_guidance():
    mesh = _make_mesh()
    shifted = np.asarray(mesh._coords) + 0.1
    with pytest.raises(AttributeError, match="mesh.deform"):
        mesh.points = shifted


def test_points_setter_leaves_state_intact():
    """The refused write must not touch the coordinate store or PETSc."""
    mesh = _make_mesh()
    wrapper_type = type(mesh._coords)
    petsc_before = mesh.dm.getCoordinatesLocal().array.copy()
    coords_before = np.array(mesh._coords)

    with pytest.raises(AttributeError):
        mesh.points = np.asarray(mesh._coords) + 0.1

    assert type(mesh._coords) is wrapper_type  # wrapper not discarded
    assert np.array_equal(np.array(mesh._coords), coords_before)
    assert np.array_equal(mesh.dm.getCoordinatesLocal().array, petsc_before)


def test_deform_is_the_supported_write_path():
    """mesh.deform() propagates the same write the setter used to swallow."""
    mesh = _make_mesh()
    new_coords = np.array(mesh._coords) + 0.1
    mesh.deform(new_coords)
    assert np.allclose(np.array(mesh._coords), new_coords)
    petsc_coords = mesh.dm.getCoordinatesLocal().array.reshape(-1, mesh.dim)
    assert abs(petsc_coords.min() - 0.1) < 1e-12
