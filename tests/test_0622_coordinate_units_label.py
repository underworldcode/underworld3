"""``mesh.X.units`` names the units ``mesh.X.coords`` carries (#979).

With a 1000 km reference length, ``mesh.X.coords`` holds metres (SI, from the
length scale) and says so, but ``mesh.X.units`` reported ``kilometer``: anything
that labelled the coordinates with ``X.units`` was 1000 times off. ``mesh.units``
stays the unit output is written in (XDMF, plots), here kilometres.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


@pytest.fixture
def units_model():
    uw.get_default_model().set_reference_quantities(
        length=uw.quantity(1000, "km"), time=uw.quantity(1, "Myr"),
        mass=uw.quantity(1e24, "kg"), temperature=uw.quantity(1000, "K"),
    )
    yield
    uw.reset_default_model()


def test_the_coordinate_label_matches_the_coordinates(units_model):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25)
    coords = mesh.X.coords
    assert uw.units(str(mesh.X.units)) == uw.units(str(coords.units))
    # Labelling the magnitudes with X.units gives the true extent, 1000 km.
    extent = uw.quantity(float(np.asarray(coords).max()), str(mesh.X.units)).to("km")
    assert extent.magnitude == pytest.approx(1000.0)


def test_mesh_units_stay_the_output_unit(units_model):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25)
    assert uw.units(str(mesh.units)) == uw.units("km")


def test_without_units_there_is_no_label():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25)
    assert mesh.X.units is None
