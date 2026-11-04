"""A component an essential wall constrains contributes zero to a rotated wall's
sigma_nn, and the rest of the node's load is kept (#936).

``boundary_normal_traction`` and ``boundary_normal_traction_integral`` read the
reaction at every node of the rotated wall, including a corner where it meets an
essential wall. They copied the reaction into a pooled local vector with
``globalToLocal``, which does not write constrained DOFs, so at that corner they read
whatever the pool last held: on this fixture the Dirichlet velocity, and after the
pool was filled with 12345, sigma_nn on Top ran from -3.7e5 to 4.6e4, against -0.63 to
1.15 with the constrained components read as zero (Louis's ruling, 2026-11-04). The
integral also took a node's owner from its FIRST component, so a corner whose v_x is
pinned and v_y free was dropped on every rank with the load on its free component.

The Left wall pins both components, v_y alone, or v_x alone:
- sigma_nn on Top is the same whatever the DM's pooled local vectors hold: as the solve
  left them, filled with 12345, or with -777; so is its integral against 1 + x^2 where
  a component is free at the corner (the weight is not zero at x = 0, where the corner
  sits, and not constant, which the mean removal would cancel);
- the integral of sigma_nn with no mean removed is the sum over the wall's nodes of
  -n.r, r the reaction read from the global vector with constrained components zero.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
import underworld3.utilities.rotated_bc as rbc

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

TOP_NORMAL = (0.6, 0.8)
LEFT = {"left-pins-both": (0.0, 0.05), "left-pins-vy": (None, 0.05),
        "left-pins-vx": (0.0, None)}


@pytest.fixture(scope="module")
def solved(request):
    left = LEFT[request.param]
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, qdegree=2)
    tag = request.param.split("-")[-1]
    v = uw.discretisation.MeshVariable("V1141" + tag, mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P1141" + tag, mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, _y = mesh.X
    stokes.bodyforce = sympy.Matrix([[0.0, sympy.cos(sympy.pi * x)]])
    # the Left wall pins one or both components at its corners with Top and Bottom
    stokes.add_dirichlet_bc(left, "Left")
    stokes.add_rotated_freeslip_bc(0, "Top", normal=TOP_NORMAL)
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


@pytest.mark.parametrize("solved", list(LEFT), indirect=True)
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


@pytest.mark.parametrize("solved", ["left-pins-vy", "left-pins-vx"], indirect=True)
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


@pytest.mark.parametrize("solved", list(LEFT), indirect=True)
def test_the_wall_load_is_the_reaction_with_constrained_components_zero(solved):
    dm = solved.dm
    lsec = dm.getLocalSection()
    l2g = dm.getLGMap()
    rc = solved._rotated_freeslip_info["reaction"]
    owned = rc.getArray(readonly=True)
    rstart, _rend = rc.getOwnershipRange()
    expected = 0.0
    for q, nrm in rbc._boundary_velocity_nodes(solved, "Top", normal=TOP_NORMAL):
        lo = lsec.getFieldOffset(q, 0)
        rows = l2g.apply(list(range(lo, lo + 2)))
        r = np.array([owned[g - rstart] if g >= 0 else 0.0 for g in rows])
        expected -= float(np.dot(nrm, r))
    got = solved.boundary_normal_traction_integral("Top", 1, remove_mean=False)
    assert abs(got - expected) <= 1.0e-12 * max(1.0, abs(expected)), (
        f"the integral of sigma_nn over Top is {got!r}; the wall's nodes carry "
        f"{expected!r} with the constrained components zero (#936)")
