"""``plot_scalar``, ``plot_vector`` and ``save_colorbar`` run with their default colormap (#957).

Their colormap default was the empty string, which pyvista 0.46 rejects with
``ValueError: Invalid colormap ''`` and Matplotlib with "'' is not a valid value
for cmap", so none of them ran without a colormap. The default is now None,
which leaves the choice to pyvista's theme or Matplotlib's default. The plots
are drawn off screen; each test asserts that the call returns, and the colour
bar that its file is written.
"""

import pytest

import underworld3 as uw

pv = pytest.importorskip("pyvista")
vis = pytest.importorskip("underworld3.visualisation")

pytestmark = [pytest.mark.level_1]


def _mesh():
    return uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25)


@pytest.mark.tier_b
def test_plot_scalar_with_its_default_colormap():
    mesh = _mesh()
    T = uw.discretisation.MeshVariable("T_cmap", mesh, 1, degree=1)
    assert vis.plot_scalar(mesh, T.sym[0], scalar_name="T") is None


@pytest.mark.tier_b
def test_plot_vector_with_its_default_colormap():
    mesh = _mesh()
    V = uw.discretisation.MeshVariable("V_cmap", mesh, 2, degree=1)
    V.array[:, 0, 0] = 1.0
    assert vis.plot_vector(mesh, V, vector_name="V") is None


@pytest.mark.tier_b
def test_save_colorbar_with_its_default_colormap(tmp_path):
    pytest.importorskip("matplotlib")
    vis.visualisation.save_colorbar(vmin=0.0, vmax=1.0, output_path=f"{tmp_path}/", fname="cb")
    assert (tmp_path / "cb_cbvert.png").is_file()
