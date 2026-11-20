"""An uncached rotated solve measures its first residual without the rotation
gauge, as a cached re-solve does (#884).

The rigid-rotation modes of a free-slip annulus are a null space of the rotated
operator, and the loop measures every residual with their component projected
out: the solve cannot reduce it. On an uncached solve the modes were collected
only after the first assembly, so iteration 0's residual, r0, the reference and
a warm start's rest-state residual kept that component, and a first solve and
its cached re-solve stopped against different references. Measured on
development (annulus, cell size 0.2, body force (-y, x), a rigid rotation):
r0 = 0.1213 uncached against 0.0703 cached.

Both now measure 0.0703, cold and warm.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _annulus(name):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2, qdegree=3)
    v = uw.discretisation.MeshVariable(f"V{name}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P{name}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, y = mesh.X
    stokes.bodyforce = sympy.Matrix([[-y, x]])
    for wall in ("Lower", "Upper"):
        stokes.add_rotated_freeslip_bc(0, wall)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-6
    return v, stokes


def test_a_first_solve_and_its_cached_re_solve_share_their_reference():
    _v, stokes = _annulus("884c")
    stokes.solve(zero_init_guess=True)
    first = stokes._rotated_freeslip_info
    stokes.solve(zero_init_guess=True)
    second = stokes._rotated_freeslip_info
    assert not first["workspace_reused"] and second["workspace_reused"]
    assert first["rnorm0"] == pytest.approx(second["rnorm0"], rel=1e-12)
    assert first["ref"] == pytest.approx(second["ref"], rel=1e-12)
    assert first["rnorm0"] == pytest.approx(0.0703, rel=1e-3), first["rnorm0"]


def test_an_uncached_warm_start_measures_its_rest_state_in_the_quotient():
    # the rest-state residual of a warm start is measured before the first
    # assembly too; a fresh solver started from a field is uncached
    v, cold = _annulus("884w")
    cold.solve(zero_init_guess=True)
    start = np.array(v.array) * 0.5
    refs = []
    for k in range(2):
        v.array[...] = start
        cold.solve(zero_init_guess=False)
        refs.append(cold._rotated_freeslip_info)
    v2, fresh = _annulus("884f")
    v2.array[...] = start
    fresh.solve(zero_init_guess=False)
    info = fresh._rotated_freeslip_info
    assert not info["workspace_reused"]
    assert info["fnorm_rest"] == pytest.approx(refs[-1]["fnorm_rest"], rel=1e-12)
    assert info["ref"] == pytest.approx(refs[-1]["ref"], rel=1e-12)
