"""``plot_scalar``, ``plot_vector`` and ``save_colorbar`` run with their default colormap (#957).

Their colormap default was the empty string, which pyvista 0.46 rejects with
``ValueError: Invalid colormap ''`` and Matplotlib with "'' is not a valid value
for cmap", so none of them ran without a colormap. The default is now None,
which leaves the choice to pyvista's theme or Matplotlib's default.

The plots are not rendered: a ``pyvista.Plotter`` on a headless runner can end
the process (see ``tests/test_0800_optional_modules.py``). The plot tests put a
recorder in place of the Plotter and assert what ``add_mesh`` is handed; the
colour bar is written through Matplotlib's Agg backend.
"""

import pytest

import underworld3 as uw

pv = pytest.importorskip("pyvista")
vis = pytest.importorskip("underworld3.visualisation")

pytestmark = [pytest.mark.level_1]


class _RecordingPlotter:
    """Stands in for ``pyvista.Plotter``: keeps what ``add_mesh`` is given."""

    calls = []

    def __init__(self, *args, **kwargs):
        self.camera = self

    def add_mesh(self, *args, **kwargs):
        _RecordingPlotter.calls.append(kwargs)

    def _nothing(self, *args, **kwargs):
        return None

    show = add_text = add_arrows = screenshot = zoom = _nothing


@pytest.fixture
def recorder(monkeypatch):
    _RecordingPlotter.calls = []
    monkeypatch.setattr(pv, "Plotter", _RecordingPlotter)
    return _RecordingPlotter.calls


def _mesh():
    return uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25)


@pytest.mark.tier_b
def test_plot_scalar_leaves_the_colormap_to_pyvista(recorder):
    mesh = _mesh()
    T = uw.discretisation.MeshVariable("T_cmap", mesh, 1, degree=1)
    vis.plot_scalar(mesh, T.sym[0], scalar_name="T")
    assert [call["cmap"] for call in recorder] == [None]


@pytest.mark.tier_b
def test_plot_vector_leaves_the_colormap_to_pyvista(recorder):
    mesh = _mesh()
    V = uw.discretisation.MeshVariable("V_cmap", mesh, 2, degree=1)
    V.array[:, 0, 0] = 1.0
    vis.plot_vector(mesh, V, vector_name="V")
    assert [call["cmap"] for call in recorder] == [None]


@pytest.mark.tier_b
def test_save_colorbar_with_its_default_colormap(tmp_path):
    pytest.importorskip("matplotlib")
    vis.visualisation.save_colorbar(vmin=0.0, vmax=1.0, output_path=f"{tmp_path}/", fname="cb")
    assert (tmp_path / "cb_cbvert.png").is_file()
