"""Native higher-order rendering keeps stretched and high-degree cells (#1083).

The native path (#1057) orders the several nodes of an edge, face or cell
interior by matching them to target positions. The targets used to be the
equispaced lattice, compared in physical space against the smallest physical
spacing. But PETSc's Lagrange nodes are not equispaced from degree 3 on, and
the smallest physical spacing shrinks with a cell's aspect ratio. So the
match refused exactly affine cells: P4 and Q3 at aspect 100, P5 and Q4 at
aspect 10, and P8 at aspect 1 (H1's measurements). The caller then fell back
to a Delaunay of the nodes, silently.

The targets are now PETSc's own reference nodes, and the comparison is made
in each cell's local coordinates, at the reference cell's scale. Each case
here must render natively and pass every check of #1057's test: sub-cell
count, DOF order, positive signed areas, conforming boundary, and per-cell
fill with an identical size-ratio pattern. A refused match now warns.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import underworld3 as uw

pytest.importorskip("pyvista")
vis = pytest.importorskip("underworld3.visualisation")

from test_1057_native_vis_higher_order import _check  # noqa: E402

pytestmark = [pytest.mark.level_2]


def _stretched(simplex, aspect, cell_size=0.25):
    """A unit box squashed by ``aspect`` in y: every cell is an affine image of
    its unstretched self, ``aspect`` times thinner."""
    if simplex:
        mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                                 cellSize=cell_size)
    else:
        mesh = uw.meshing.StructuredQuadBox(elementRes=(3, 3))
    if aspect != 1:
        x = np.asarray(mesh.X.coords).copy()
        x[:, 1] /= aspect
        mesh.deform(x)
    return mesh


@pytest.mark.tier_b
@pytest.mark.parametrize("simplex, degree, aspect", [
    (True, 4, 100), (True, 5, 10), (False, 3, 100), (False, 4, 10),
])
def test_stretched_cells_render_natively(simplex, degree, aspect):
    _check(_stretched(simplex, aspect), degree, exact_volume=1.0 / aspect)


@pytest.mark.tier_b
def test_degree_8_renders_natively():
    _check(_stretched(True, 1, cell_size=0.5), 8, exact_volume=1.0)


@pytest.mark.tier_b
def test_refused_match_warns():
    # Nodes scattered by most of a spacing cannot be matched: the native path
    # returns None, and the public entry point says why it fell back.
    from underworld3.visualisation.visualisation import _lagrange_subcell_connectivity

    mesh = _stretched(True, 1)
    T = uw.discretisation.MeshVariable("T_scatter", mesh, 1, degree=3)
    X = np.asarray(T.coords_nd)
    rng = np.random.default_rng(1083)
    moved = X + 0.09 * rng.uniform(-1, 1, X.shape)
    scattered = SimpleNamespace(mesh=mesh, degree=3, continuous=True, name="T_scatter",
                                coords_nd=moved, coords=moved)
    assert _lagrange_subcell_connectivity(scattered) is None
    with pytest.warns(UserWarning, match="Delaunay"):
        vis.meshVariable_to_pv_mesh_object(scattered)
