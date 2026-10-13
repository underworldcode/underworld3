import pytest

# All tests in this module are quick core tests
pytestmark = pytest.mark.level_1
# These tests just check if all the meshes can be built / returned but no validation
# about whether they can be used.


def test_create_usb_2d_mesh():
    from underworld3.meshing import UnstructuredSimplexBox

    mesh = UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 2.0), cellSize=1.0 / 8.0)

    return


def test_create_usb_2d_r_mesh():
    from underworld3.meshing import UnstructuredSimplexBox

    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 2.0), cellSize=1.0 / 8.0, regular=True
    )

    return


def test_create_usb_3d_mesh():
    from underworld3.meshing import UnstructuredSimplexBox

    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 2.0), cellSize=1.0 / 8.0
    )

    return


def test_create_usb_3d_r_mesh():
    from underworld3.meshing import UnstructuredSimplexBox

    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 2.0), cellSize=1.0 / 8.0, regular=True
    )

    return


def test_create_sqb_2d_mesh():
    from underworld3.meshing import StructuredQuadBox

    mesh = StructuredQuadBox(elementRes=(16, 16), degree=1, qdegree=1)

    return


def test_create_sqb_3d_mesh():
    from underworld3.meshing import StructuredQuadBox

    mesh = StructuredQuadBox(elementRes=(8, 16, 4), degree=1, qdegree=2)

    return


def test_create_cs_hex_mesh():
    from underworld3.meshing import CubedSphere

    mesh = CubedSphere(
        radiusOuter=1.0, radiusInner=0.5, numElements=5, degree=1, qdegree=2, simplex=False
    )

    return


def test_create_cs_simplex_mesh():
    from underworld3.meshing import CubedSphere

    mesh = CubedSphere(
        radiusOuter=1.0, radiusInner=0.5, numElements=5, degree=1, qdegree=2, simplex=False
    )

    return


def test_create_ss_mesh():
    from underworld3.meshing import SphericalShell

    mesh = SphericalShell(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2, degree=1, qdegree=2)

    return


def test_create_solid_s_mesh():
    from underworld3.meshing import SphericalShell

    mesh = SphericalShell(radiusOuter=1.0, radiusInner=0.0, cellSize=0.2, degree=1, qdegree=2)

    return


def test_create_ann_us_mesh():
    from underworld3.meshing import Annulus

    mesh = Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.1, degree=1, qdegree=2)

    return


def test_create_solid_ann_us_mesh():
    from underworld3.meshing import Annulus

    mesh = Annulus(radiusOuter=1.0, radiusInner=0.0, cellSize=0.1, degree=1, qdegree=2)

    return


def test_create_solid_sqdIB_2d_mesh():
    from underworld3.meshing import BoxInternalBoundary

    mesh = BoxInternalBoundary(
        elementRes=(8, 8),
        zelementRes=(4, 4),
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        zintCoord=0.5,
        degree=1,
        qdegree=2,
    )

    return


def test_create_solid_sqdIB_3d_mesh():
    from underworld3.meshing import BoxInternalBoundary

    mesh = BoxInternalBoundary(
        elementRes=(4, 4, 4),
        zelementRes=(2, 2),
        minCoords=(0.0, 0.0, 0.0),
        maxCoords=(1.0, 1.0, 1.0),
        zintCoord=0.5,
        degree=1,
        qdegree=2,
    )

    return


def test_create_solid_usbIB_2d_mesh():
    from underworld3.meshing import BoxInternalBoundary

    mesh = BoxInternalBoundary(
        cellSize=1 / 4,
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        zintCoord=0.5,
        degree=1,
        qdegree=2,
        simplex=True,
    )

    return


def test_create_solid_usbIB_3d_mesh():
    from underworld3.meshing import BoxInternalBoundary

    mesh = BoxInternalBoundary(
        cellSize=1 / 4,
        minCoords=(0.0, 0.0, 0.0),
        maxCoords=(1.0, 1.0, 1.0),
        zintCoord=0.5,
        degree=1,
        qdegree=2,
        simplex=True,
    )

    return


@pytest.mark.tier_a
def test_no_null_boundary_label_is_manufactured():
    """A mesh does not label every vertex with the ``Null_Boundary`` sentinel.

    ``Null_Boundary`` (666) used to be created on every mesh, marking the whole
    depth-0 stratum. It marked no facet, so it integrated nothing, and its only
    consumer was a fake natural BC the solver manufactured — since removed,
    with no change to any answer at any rank count.

    Asserting its ABSENCE is what protects the fix that motivated the removal:
    a pass that looks for material interfaces by reading labelled *points* saw
    every vertex of every UW3 mesh labelled, and silently declined all of them.
    """
    from underworld3.meshing import StructuredQuadBox

    mesh = StructuredQuadBox(elementRes=(4, 4))
    dm = mesh.dm

    # Asserted against the DM's own list of label NAMES. `getLabel` on a name
    # the DM does not have returns a non-None DMLabel wrapper with a null
    # handle, so `is not None` reports every absent label as present; the
    # codebase's `if label:` idiom exists for exactly this reason.
    names = [dm.getLabelName(i) for i in range(dm.getNumLabels())]
    assert "Null_Boundary" not in names, names
    assert not dm.getLabel("Null_Boundary")
    assert "Null_Boundary" not in [b.name for b in mesh.boundaries]

    # The boundary that DOES mean "the whole outside" is still there, and it
    # is a different animal: exterior FACETS, so it integrates.
    all_bd = mesh.dm.getLabel("All_Boundaries")
    assert all_bd is not None
    fS, fE = mesh.dm.getHeightStratum(1)
    pts = all_bd.getStratumIS(1001).getIndices()
    assert len(pts) > 0
    assert ((pts >= fS) & (pts < fE)).all(), "All_Boundaries must mark facets"
