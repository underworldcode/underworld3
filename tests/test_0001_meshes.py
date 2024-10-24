# These tests just check if all the meshes can be built / returned but no validation
# about whether they can be used.


def test_create_usb_2d_mesh():
    from underworld3.meshing import UnstructuredSimplexBox

    mesh = UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 2.0), cellSize=1.0 / 8.0)

    return


def test_create_usb_2d_r_mesh():
    from underworld3.meshing import UnstructuredSimplexBox

    mesh = UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 2.0), cellSize=1.0 / 8.0, regular=True)

    return


def test_create_usb_3d_mesh():
    from underworld3.meshing import UnstructuredSimplexBox

    mesh = UnstructuredSimplexBox(minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 2.0), cellSize=1.0 / 8.0)

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

    mesh = CubedSphere(radiusOuter=1.0, radiusInner=0.5, numElements=5, degree=1, qdegree=2, simplex=False)

    return


def test_create_cs_simplex_mesh():
    from underworld3.meshing import CubedSphere

    mesh = CubedSphere(radiusOuter=1.0, radiusInner=0.5, numElements=5, degree=1, qdegree=2, simplex=False)

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
    from underworld3.meshing import StructuredQuadBoxInternalBoundary2D

    mesh = StructuredQuadBoxInternalBoundary2D(elementRes=(8,8),yelementRes=(4,4),minCoords=(0.0,0.0),maxCoords=(1.0, 1.0),yintCoord=0.5,degree=1,qdegree=2)


    return
    
def test_create_solid_sqdIB_3d_mesh():
    from underworld3.meshing import StructuredQuadBoxInternalBoundary3D

    mesh = StructuredQuadBoxInternalBoundary3D(elementRes=(4,4,4),zelementRes=(2,2),minCoords=(0.0,0.0,0.0),maxCoords=(1.0, 1.0, 1.0),zintCoord=0.5,degree=1,qdegree=2)

    return
