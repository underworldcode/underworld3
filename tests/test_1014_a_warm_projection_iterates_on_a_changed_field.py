"""A warm-started projection whose field has changed iterates (#1014's reference).

Since #1014 a stationary solve is held to ``rtol * max(||F(x0)||, ||F(0)||)``. For
a projection ``||F(0)||`` is the whole field, so a warm re-solve whose change
stayed inside ``rtol`` of the field took no iteration and returned the previous
answer. The library re-projects every step of a time loop (stress histories,
semi-Lagrangian histories, the free surface's and Darcy's velocities), and a
history that stores a projection then drifts: the forward stress history of
test_1061 went from 6.4e-4 to 8.0e-3, from step 161 on. Projections now measure
their increment: a warm re-solve reduces the change by ``rtol``.

Each projection here reads a source variable, which changes by 1e-5 of itself
between two solves, inside the default tolerance (1e-3) of the whole field. From a
converged start, the warm solve at the default tolerance must iterate and carry
the change: within 1e-2 of it against a converged projection of the new field.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]

_count = [0]


def _projection(kind, mesh, source):
    _count[0] += 1
    n = {"scalar": 1, "vector": 2, "multicomponent": 3}[kind]
    if kind == "multicomponent":
        target = uw.discretisation.MeshVariable(f"P1014_{_count[0]}", mesh, (1, n),
                                                vtype=uw.VarType.MATRIX, degree=2)
    else:
        target = uw.discretisation.MeshVariable(f"P1014_{_count[0]}", mesh, n, degree=2)
    if kind == "scalar":
        projection = uw.systems.Projection(mesh, target)
        projection.uw_function = source.sym[0]
    elif kind == "vector":
        projection = uw.systems.Vector_Projection(mesh, target)
        projection.uw_function = sympy.Matrix([[source.sym[0], 2 * source.sym[0]]])
    else:
        projection = uw.systems.MultiComponent_Projection(mesh, target)
        projection.uw_function = sympy.Matrix(
            [[source.sym[0], 2 * source.sym[0], -source.sym[0]]])
    return projection, target


@pytest.mark.parametrize("kind", ["scalar", "vector", "multicomponent"])
def test_a_warm_projection_of_a_changed_field_iterates(kind):
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.1)
    _count[0] += 1
    source = uw.discretisation.MeshVariable(f"S1014_{_count[0]}", mesh, 1, degree=2)
    X = np.asarray(source.coords)
    source.array[:, 0, 0] = 2 + np.sin(3 * X[:, 0]) * np.cos(2 * X[:, 1])

    warm, target = _projection(kind, mesh, source)
    warm.tolerance = 1.0e-10           # start from the converged projection
    warm.solve()
    before = np.asarray(target.array).copy()
    warm.tolerance = 1.0e-3            # the default, for the warm re-solve

    source.array[:, 0, 0] = np.asarray(source.array[:, 0, 0]) * (1 + 1.0e-5)
    warm.solve()
    assert warm.snes.getIterationNumber() >= 1, "the warm re-solve took no iteration"

    reference, reference_target = _projection(kind, mesh, source)
    reference.tolerance = 1.0e-10
    reference.solve()
    change = np.abs(np.asarray(reference_target.array) - before).max()
    error = np.abs(np.asarray(target.array) - np.asarray(reference_target.array)).max()
    # The change is about 2e-5. Skipped, the warm answer is off by all of it; solved
    # to rtol of its own residual, by about 1e-3 of it.
    assert error < 1.0e-2 * change, (error, change)
