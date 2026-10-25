"""The PyVista conversions work without trame.

``visualisation.initialise`` asks for the trame Jupyter backend. Where trame is
not installed (a headless container, a cluster node) pyvista's theme setter
raises ImportError, and every conversion -- ``mesh_to_pv_mesh`` and with it
every render -- failed although no notebook was involved. The fallback is the
static backend, which pyvista always has.
"""
import pytest

import underworld3 as uw
import underworld3.visualisation as vis
from underworld3.visualisation.visualisation import initialise

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

pv = pytest.importorskip("pyvista")


@pytest.fixture
def trame_missing(monkeypatch):
    """The theme's backend setter raises ImportError for trame, as without it."""
    theme_class = type(pv.global_theme)
    real = theme_class.jupyter_backend

    def setter(self, value):
        if value == "trame":
            raise ImportError("trame is not installed")
        real.fset(self, value)

    monkeypatch.setattr(theme_class, "jupyter_backend", property(real.fget, setter))


def test_initialise_falls_back_to_static_without_trame(trame_missing):
    initialise(None)
    assert pv.global_theme.jupyter_backend == "static"


def test_mesh_to_pv_mesh_without_trame(trame_missing):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25)
    pvmesh = vis.mesh_to_pv_mesh(mesh)
    start, end = mesh.dm.getHeightStratum(0)
    assert pvmesh.n_cells == end - start
