r"""``evaluate`` is exact near edges and vertices for a field the element reproduces (#1115).

The cell locator accepted a point in a cell it was within a face tolerance of
(1e-2 of the cell's control-point spacing), so a point just inside one cell
could be answered in the neighbour that shares the face. The evaluator then
clamped the reference coordinates onto that face: an error of about
:math:`|\nabla f|\,\delta` for a point a distance :math:`\delta` from it, even
for a field the element represents exactly. On a 1/16 simplex box it was 4e-6
at :math:`\delta` = 1e-6 and 4.4e-5 at 1e-5; quads were exact.

The baseline is exactness: P1 holds a linear field and P2 a quadratic one, so
the only error left is rounding.
"""
import numpy as np
import pytest
import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]

DISTANCES = (1e-8, 1e-6, 1e-5, 1e-4, 1e-3)


def _field(dim, degree):
    if degree == 1:
        return lambda X: 2 * X[:, 0] + 3 * X[:, 1] - (X[:, 2] if dim == 3 else 0)
    return lambda X: (2 * X[:, 0] + 3 * X[:, 1] + X[:, 0] ** 2 - X[:, 0] * X[:, 1]
                      + 0.5 * X[:, 1] ** 2 + (X[:, 2] ** 2 - X[:, 0] * X[:, 2] if dim == 3 else 0))


def _mesh(dim, kind):
    if kind == "quad":
        return uw.meshing.StructuredQuadBox(elementRes=(8,) * dim, minCoords=(0.0,) * dim,
                                            maxCoords=(1.0,) * dim, qdegree=3)
    return uw.meshing.UnstructuredSimplexBox(minCoords=(0.0,) * dim, maxCoords=(1.0,) * dim,
                                             cellSize=1 / 8 if dim == 2 else 0.25, qdegree=3)


def _vertices_and_edge_midpoints(mesh):
    """Interior mesh vertices, and the midpoints of the cells' edges, from the DM."""
    dm, dim = mesh.dm, mesh.dim
    vs, ve = dm.getDepthStratum(0)
    xyz = dm.getCoordinatesLocal().array.reshape(-1, dim)
    edges = set()
    cs, ce = dm.getHeightStratum(0)
    for c in range(cs, ce):
        cv = [p - vs for p in dm.getTransitiveClosure(c)[0] if vs <= p < ve]
        edges.update((min(a, b), max(a, b)) for i, a in enumerate(cv) for b in cv[i + 1:])
    keep = lambda X: X[(X.min(axis=1) > 1e-2) & (X.max(axis=1) < 1 - 1e-2)]
    midpoints = np.array([(xyz[a] + xyz[b]) / 2 for a, b in edges])
    return keep(xyz), keep(midpoints)


def _max_error(mesh, T, f, points):
    got = np.asarray(uw.function.evaluate(T.sym[0], np.ascontiguousarray(points))).reshape(-1)
    return float(np.abs(got - f(points)).max())


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("kind", ["simplex", "quad"])
def test_an_exact_field_is_exact_near_vertices_and_edges(dim, degree, kind):
    mesh = _mesh(dim, kind)
    T = uw.discretisation.MeshVariable(f"T_{dim}{degree}{kind}", mesh, 1, degree=degree)
    f = _field(dim, degree)
    T.array[:, 0, 0] = f(np.asarray(T.coords))
    vertices, midpoints = _vertices_and_edge_midpoints(mesh)
    rng = np.random.default_rng(0)

    # exactly on vertices and edges, then a distance delta off them, in a
    # random direction
    errors = {"on": max(_max_error(mesh, T, f, vertices), _max_error(mesh, T, f, midpoints))}
    for delta in DISTANCES:
        for name, base in (("vertex", vertices), ("edge", midpoints)):
            d = rng.normal(size=base.shape)
            d /= np.linalg.norm(d, axis=1)[:, None]
            errors[f"{name} + {delta:.0e}"] = _max_error(mesh, T, f, base + delta * d)

    worst = max(errors.values())
    assert worst < 1e-12, {k: f"{v:.1e}" for k, v in errors.items() if v >= 1e-12}
