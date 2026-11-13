"""``plot_scalar`` and ``plot_vector`` work with their default colormap (#957).

Both defaulted to ``cmap=""``, which pyvista rejects ("Invalid colormap ''"),
so every call that did not name a colormap raised. The default is now ``None``,
pyvista's own default.
"""
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]

pv = pytest.importorskip("pyvista")
vis = pytest.importorskip("underworld3.visualisation")


@pytest.fixture
def off_screen(monkeypatch):
    monkeypatch.setattr(pv, "OFF_SCREEN", True)
    # show() would open a window; the colormap is resolved in add_mesh, before it.
    monkeypatch.setattr(pv.Plotter, "show", lambda self, *a, **k: None)


@pytest.fixture(scope="module")
def mesh():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25
    )


def test_plot_scalar_default_colormap(off_screen, mesh):
    T = uw.discretisation.MeshVariable("T_957", mesh, 1, degree=1)
    T.data[:, 0] = 1.0
    vis.plot_scalar(mesh, T.sym[0])


def test_plot_vector_default_colormap(off_screen, mesh):
    v = uw.discretisation.MeshVariable("v_957", mesh, 2, degree=1)
    v.data[:, 0] = 1.0
    vis.plot_vector(mesh, v)


def test_a_named_colormap_still_reaches_pyvista(off_screen, mesh):
    T = uw.discretisation.MeshVariable("T2_957", mesh, 1, degree=1)
    T.data[:, 0] = 1.0
    with pytest.raises(ValueError, match="colormap"):
        vis.plot_scalar(mesh, T.sym[0], cmap="no-such-colormap-957")
