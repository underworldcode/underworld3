"""API contract of the Eulerian SUPG Navier-Stokes solver.

Structural checks that run in seconds: the export, argument validation, the
scheme assembled from the velocity history, the advecting-velocity switch and
the Picard passes, and the Stokes limit.

Run: pixi run python -m pytest tests/test_1056_navier_stokes_supg_api.py -v
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.fixture(scope="module")
def mesh():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2, qdegree=3)


def _cavity(mesh, tag, **kwargs):
    """Lid-driven cavity: no-slip walls, a unit lid, unit viscosity."""
    v = uw.discretisation.MeshVariable(f"U_{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P_{tag}", mesh, 1, degree=1)
    ns = uw.systems.NavierStokesSUPG(mesh, v, p, **kwargs)
    ns.constitutive_model = uw.constitutive_models.ViscousFlowModel
    ns.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    for b in ("Left", "Right", "Bottom"):
        ns.add_dirichlet_bc((0.0, 0.0), b)
    ns.add_dirichlet_bc((1.0, 0.0), "Top")
    return ns, v, p


def test_exported_and_constructs_with_the_scalar_solver_rules(mesh):
    ns, _v, _p = _cavity(mesh, "a", rho=1.0)
    assert type(ns).__name__ == "SNES_NavierStokes_SUPG"
    assert ns.integrator == "am" and ns.order == 1 and ns.theta == 0.5
    assert isinstance(ns.DuDt, uw.systems.ddt.Eulerian) and ns.DuDt.V_fn is None
    assert ns.DFDt is None
    assert _cavity(mesh, "b", order=2)[0].integrator == "bdf"
    with pytest.raises(ValueError, match="theta applies"):
        _cavity(mesh, "c", order=2, theta=0.5)
    with pytest.raises(ValueError, match="stress history"):
        _cavity(mesh, "d", DFDt=object())
    with pytest.raises(ValueError, match="advection must be"):
        _cavity(mesh, "e", advection="upwind")


def test_a_step_runs_and_the_scheme_is_one_linear_solve(mesh):
    ns, v, _p = _cavity(mesh, "s", rho=1.0)
    ns.solve(timestep=0.05)
    assert ns.snes.getIterationNumber() == 1
    assert np.isfinite(np.asarray(v.array)).all()
    assert ns.picard_count == 0
    ns.solve(timestep=0.05, picard_iterations=3)
    assert 1 <= ns.picard_count <= 3


def test_stokes_limit_reproduces_the_stokes_solver(mesh):
    """With rho -> 0 the momentum equation is the Stokes equation."""
    ns, v, p = _cavity(mesh, "z", rho=0.0)
    ns.supg_weight = 0.0
    ns.tolerance = 1.0e-7
    ns.solve(timestep=1.0)
    vs = uw.discretisation.MeshVariable("U_stokes", mesh, 2, degree=2)
    ps = uw.discretisation.MeshVariable("P_stokes", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, vs, ps)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    for b in ("Left", "Right", "Bottom"):
        stokes.add_dirichlet_bc((0.0, 0.0), b)
    stokes.add_dirichlet_bc((1.0, 0.0), "Top")
    stokes.tolerance = ns.tolerance
    stokes.solve()
    a, b = np.asarray(v.array).reshape(-1), np.asarray(vs.array).reshape(-1)
    assert np.abs(a - b).max() < 1e-5 * np.abs(b).max()


def test_timestep_is_a_runtime_constant_and_theta_is_settable(mesh):
    ns, _v, _p = _cavity(mesh, "t", rho=1.0)
    ns.solve(timestep=0.05)
    key = ns._current_jit_cache_key
    ns.solve(timestep=0.02)
    assert ns._current_jit_cache_key == key
    ns.theta = 1.0
    ns.solve(timestep=0.02)
    assert ns.theta == 1.0 and ns.DuDt.theta == 1.0
    assert ns._current_jit_cache_key == key
