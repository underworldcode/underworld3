"""The segmented generators' native-coordinate modes refuse, with the reason.

The native coordinate systems (``SPHERE_SURFACE_NATIVE``, ``SPHERICAL_NATIVE``)
were removed from ``CoordinateSystemType`` in ee29df2a (2025-10-10), with the
vector calculus they needed. ``SegmentedSphericalSurface2D`` (a longitude-latitude
surface) and ``SegmentedSphericalShell`` / ``SegmentedSphericalBall`` with
``coordinatesNative=True`` have failed since: on development the surface with a
PETSc periodicity error ("Periodic length 0. > max cell size 0. in dimension 1")
and the other two with "'CoordinateSystemType' has no attribute
'SPHERICAL_NATIVE'". They now refuse before any work, naming the removal and the
alternative; the Cartesian shell still builds.
"""
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


@pytest.mark.parametrize("make, alternative", [
    (lambda: uw.meshing.SegmentedSphericalSurface2D(cellSize=0.2, numSegments=3),
     "SphericalManifold"),
    (lambda: uw.meshing.SegmentedSphericalShell(cellSize=0.3, numSegments=3,
                                                coordinatesNative=True),
     "coordinatesNative=False"),
    (lambda: uw.meshing.SegmentedSphericalBall(cellSize=0.3, numSegments=3,
                                               coordinatesNative=True),
     "coordinatesNative=False"),
], ids=["surface 2D", "shell, native", "ball, native"])
def test_a_native_coordinate_generator_refuses_with_the_reason(make, alternative):
    with pytest.raises(NotImplementedError, match="ee29df2a") as refusal:
        make()
    assert alternative in str(refusal.value)


def test_the_cartesian_shell_still_builds():
    mesh = uw.meshing.SegmentedSphericalShell(cellSize=0.3, numSegments=3)
    assert mesh.dm.getHeightStratum(0)[1] > 0
