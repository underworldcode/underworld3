"""A node an essential wall constrains contributes zero to a rotated wall's sigma_nn
in its constrained components (#936).

``boundary_normal_traction`` and ``boundary_normal_traction_integral`` read the
reaction at every node of the rotated wall, including a corner where it meets an
essential wall. They copied the reaction into a pooled local vector with
``globalToLocal``, which does not write constrained DOFs, so at that corner they read
whatever the pool last held: on this fixture the Dirichlet velocity (0, 0.05), and
after the pool was filled with 12345, sigma_nn on Top ran from -3.7e5 to 4.6e4
against -1.1 to 1.3. The local vector is zeroed first now, so such a node contributes
zero.

The integral skips a node whose first component is constrained, so it read the pool
only at a node with that component free and another pinned: here the Top-Left corner
when the Left wall pins v_y alone.

- sigma_nn on Top, and its integral against 1 + x^2, are the same whatever the DM's
  pooled local vectors hold: as the solve left them, filled with 12345, or with -777.
  The Left wall pins both components, or v_y alone. (The weight is not zero at the
  corner x = 0, where the constrained node sits, and not constant, which the mean
  removal would cancel.)
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


@pytest.fixture(scope="module", params=[(0.0, 0.05), (None, 0.05)],
                ids=["left-pins-both", "left-pins-vy"])
def solved(request):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, qdegree=2)
    tag = "b" if request.param[0] is not None else "y"
    v = uw.discretisation.MeshVariable("V1141" + tag, mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P1141" + tag, mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, _y = mesh.X
    stokes.bodyforce = sympy.Matrix([[0.0, sympy.cos(sympy.pi * x)]])
    # the Left wall pins both components, or v_y alone, at its corners with Top and
    # Bottom
    stokes.add_dirichlet_bc(request.param, "Left")
    stokes.add_rotated_freeslip_bc(0, "Top", normal=(0.6, 0.8))
    stokes.add_rotated_freeslip_bc(0, "Bottom", normal=(0.0, -1.0))
    stokes.add_rotated_freeslip_bc(0, "Right", normal=(1.0, 0.0))
    stokes.solve()
    return stokes


def _fill_pool(dm, value, n=8):
    """Fill the DM's pooled local vectors with ``value``."""
    vecs = [dm.getLocalVec() for _ in range(n)]
    for w in vecs:
        w.set(value)
    for w in vecs:
        dm.restoreLocalVec(w)


def test_sigma_nn_does_not_read_the_pool(solved):
    _xs, as_left = solved.boundary_normal_traction("Top")
    results = [np.asarray(as_left)]
    for value in (12345.0, -777.0):
        _fill_pool(solved.dm, value)
        _xs, sig = solved.boundary_normal_traction("Top")
        results.append(np.asarray(sig))
    assert np.all(np.isfinite(results[0]))
    spans = [(float(r.min()), float(r.max())) for r in results]
    assert all(np.array_equal(results[0], r) for r in results[1:]), (
        f"sigma_nn on Top depends on the pooled vectors' contents: ranges "
        f"{spans} with the pool as the solve left it, at 12345 and at -777 (#936)")


def test_the_sigma_nn_integral_does_not_read_the_pool(solved):
    x, _y = solved.mesh.X
    weight = 1 + x**2
    values = [solved.boundary_normal_traction_integral("Top", weight)]
    for fill in (12345.0, -777.0):
        _fill_pool(solved.dm, fill)
        values.append(solved.boundary_normal_traction_integral("Top", weight))
    assert np.isfinite(values[0])
    assert values[1] == values[0] and values[2] == values[0], (
        f"the integral of sigma_nn (1 + x^2) over Top depends on the pooled vectors' "
        f"contents: {values} (#936)")
