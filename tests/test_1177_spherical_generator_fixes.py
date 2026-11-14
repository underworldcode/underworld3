"""Two small defects of the mesh generators (#949, items 2 and 5)."""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1]


@pytest.mark.tier_b
def test_a_coarse_shell_with_an_internal_boundary_builds():
    """The shell volume is identified by its radius, not to within half a cell.

    With radii 1/2, 3/4 and 1 and a cell size of 1/2, half a cell is the whole gap
    between the internal and outer spheres: both matched the outer radius and the
    constructor raised "Could not identify the spherical-shell volume".
    """
    mesh = uw.meshing.SphericalShellInternalBoundary(
        radiusOuter=1.0, radiusInternal=0.75, radiusInner=0.5, cellSize=0.5, qdegree=2)
    for region in (mesh.regions.Inner, mesh.regions.Outer):
        assert mesh.dm.getLabel(region.name).getStratumSize(region.value) > 0
    for wall, radius in (("Lower", 0.5), ("Internal", 0.75), ("Upper", 1.0)):
        surface = mesh.bounding_surfaces[wall]
        assert surface.radius == pytest.approx(radius, rel=1.0e-12)


@pytest.mark.tier_b
def test_a_mask_from_a_label_is_one_on_the_labelled_vertices():
    """``meshVariable_mask_from_label`` raised NameError: MeshVariable."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25)
    mask = mesh.meshVariable_mask_from_label("Top", mesh.boundaries.Top.value)
    on_top = np.abs(np.asarray(mask.coords)[:, 1] - 1.0) < 1.0e-12
    values = np.asarray(mask.array).reshape(-1)
    assert on_top.any()
    assert np.all(values[on_top] == 1.0) and np.all(values[~on_top] == 0.0)
