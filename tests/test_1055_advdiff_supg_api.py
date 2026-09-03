"""API contract of the Eulerian SUPG advection-diffusion solver.

Structural checks that run in seconds: the export, argument validation, the
scheme assembled from the history manager, and the rule that a change of
timestep is a change of a runtime constant, never a recompile.

Run: pixi run python -m pytest tests/test_1055_advdiff_supg_api.py -v
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.fixture(scope="module")
def mesh():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0), cellSize=0.25, qdegree=3)


def _solver(mesh, tag, **kwargs):
    x, y = mesh.X
    T = uw.discretisation.MeshVariable(f"T_{tag}", mesh, 1, degree=2)
    T.array[:, 0, 0] = uw.function.evaluate(
        sympy.exp(-((x - 0.5) ** 2 + y ** 2) / 0.03), T.coords).reshape(-1)
    adv = uw.systems.AdvDiffusionSUPG(mesh, T, sympy.Matrix([[-y, x]]), **kwargs)
    for b in ("Left", "Right", "Top", "Bottom"):
        adv.add_dirichlet_bc(0.0, b)
    return adv, T


def test_exported_and_constructs_with_the_slcn_defaults(mesh):
    adv, _T = _solver(mesh, "a")
    assert type(adv).__name__ == "SNES_AdvectionDiffusion_SUPG"
    # order 1, theta 0.5: Crank-Nicolson, the semi-Lagrangian solver's default
    assert adv.integrator == "am" and adv.order == 1 and adv.theta == 0.5
    assert isinstance(adv.DuDt, uw.systems.ddt.Eulerian)
    assert adv.DuDt.V_fn is None, "advection is implicit, not a history correction"


def test_slcn_order_theta_pairs_select_the_documented_schemes(mesh):
    assert _solver(mesh, "p1", order=1, theta=1.0)[0].integrator == "am"     # backward Euler
    assert _solver(mesh, "p2", order=2, theta=1.0)[0].integrator == "bdf"    # SL-BDF2's counterpart
    assert _solver(mesh, "p3", order=2)[0].integrator == "bdf"               # theta 0.5 only bites at order 1
    with pytest.raises(ValueError, match="theta applies"):
        _solver(mesh, "p4", order=2, theta=0.5)


def test_semi_lagrangian_only_arguments_are_ignored_with_a_warning(mesh):
    with pytest.warns(UserWarning, match="monotone_mode, old_frame_traceback"):
        adv, _T = _solver(mesh, "q", monotone_mode="clamp", old_frame_traceback=True)
    adv.solve(timestep=0.01)


def test_solve_takes_the_slcn_signature_and_delta_t(mesh):
    adv, T = _solver(mesh, "s")
    adv.solve(False, 0.01)                      # positional, as SLCN allows
    adv.delta_t = 0.02                          # set once ...
    adv.solve()                                 # ... and reuse
    assert float(adv.delta_t.sym) == 0.02
    assert np.isfinite(np.asarray(T.array)).all()


@pytest.mark.parametrize("tag, kwargs, message", [
    ("v0", dict(order=4), "order must be"),
    ("v1", dict(order=0), "order must be"),
    ("v2", dict(order=2, theta=0.5), "theta applies"),
    ("v3", dict(order=3, theta=0.5), "theta applies"),
])
def test_scheme_arguments_are_validated(mesh, tag, kwargs, message):
    with pytest.raises(ValueError, match=message):
        _solver(mesh, tag, **kwargs)


def test_timestep_is_required(mesh):
    adv, _T = _solver(mesh, "b")
    with pytest.raises(ValueError, match="needs a timestep"):
        adv.solve()


def test_bdf_diffusive_flux_is_the_constitutive_flux(mesh):
    """For the BDF family the assembled diffusive flux is exactly the
    constitutive model's own flux of the new state; no history enters it."""
    adv, _T = _solver(mesh, "c", order=2)
    adv.constitutive_model.Parameters.diffusivity = 0.7
    difference = adv._diffusive_flux() - adv.constitutive_model.flux.T
    assert all(sympy.simplify(e) == 0 for e in difference)


def test_multistep_weights_reach_every_stored_time_level(mesh):
    # The theta rule at higher order is assembled by the same code; it is not
    # offered publicly (unstable for advection), so the family is switched
    # on the instance here to cover the weighted-sum path.
    adv, _T = _solver(mesh, "d", order=2)
    adv._integrator = "am"
    weights = adv._spatial_weights()
    assert len(weights) == 3
    states = adv._states()
    assert len(states) == 3
    # every history state appears (through its derivatives) in the advection operator
    names = {str(atom.func) for atom in adv._advection().atoms(sympy.Function)}
    for s in states[1:]:
        assert any(str(s.func) in n for n in names), (s, names)


def test_timestep_change_is_a_constant_update_not_a_recompile(mesh):
    adv, _T = _solver(mesh, "e", order=2)
    adv.solve(timestep=0.01)
    key = adv._current_jit_cache_key
    names = [getattr(c, "name", str(c)) for c in adv.constants_manifest]
    assert any(r"\Delta t" in n for n in names), names
    assert any("BDF" in n for n in names), names
    adv.solve(timestep=0.013)
    assert adv._current_jit_cache_key == key
    assert float(adv.delta_t.sym) == 0.013


def test_timestep_change_reaches_the_kernels(mesh):
    """A solver stepped 0.01 then 0.02 gives the same field as a fresh solver
    stepped 0.02 from the same state: the constant is really updated."""
    adv1, T1 = _solver(mesh, "f1")
    adv1.solve(timestep=0.01)
    state = np.array(T1.array)
    adv1.solve(timestep=0.02)

    adv2, T2 = _solver(mesh, "f2")
    T2.array[...] = state
    adv2.DuDt.initialise_history()
    adv2.solve(timestep=0.02)
    # to the linear-solver tolerance (measured 2e-11 against a 2e-2 control)
    assert np.allclose(np.asarray(T1.array), np.asarray(T2.array), rtol=0, atol=1e-8)

    # negative control: a different timestep gives a visibly different field
    adv3, T3 = _solver(mesh, "f3")
    T3.array[...] = state
    adv3.DuDt.initialise_history()
    adv3.solve(timestep=0.01)
    assert np.abs(np.asarray(T2.array) - np.asarray(T3.array)).max() > 1e-3


def test_order_ramps_from_one_unless_history_is_planted(mesh):
    adv, T = _solver(mesh, "g", order=2)
    adv.solve(timestep=0.01)
    assert adv.DuDt.effective_order == 1
    adv.solve(timestep=0.01)
    assert adv.DuDt.effective_order == 2

    adv2, T2 = _solver(mesh, "h", order=2)
    adv2.DuDt.set_initial_history([np.array(T2.array), np.array(T2.array)], dt=0.01)
    adv2.solve(timestep=0.01)
    assert adv2.DuDt.effective_order == 2


def test_galerkin_baseline_needs_no_rebuild(mesh):
    adv, _T = _solver(mesh, "i")
    adv.solve(timestep=0.01)
    key = adv._current_jit_cache_key
    adv.supg_weight = 0.0
    adv.solve(timestep=0.01)
    assert adv._current_jit_cache_key == key
    assert adv.supg_weight == 0.0


def test_solves_on_an_adapt_child_with_its_own_preconditioner():
    """An adapt child carries a mesh-owned multigrid hierarchy that the
    solver base installs opportunistically. This solver owns its (additive
    Schwarz) preconditioner, so the pickup must be skipped: installing a
    PCMG hierarchy on a non-MG preconditioner segfaulted inside PETSc."""
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0), cellSize=0.25, qdegree=3,
        refinement=1)
    x, y = base.X

    def metric(pts):
        h = np.where(np.abs(pts[:, 0]) < 0.1, 0.03, 0.125)
        return 1.0 / h ** 2

    child = base.adapt(metric, max_levels=2)
    xc, yc = child.X
    T = uw.discretisation.MeshVariable("T_child", child, 1, degree=2)
    T.array[:, 0, 0] = uw.function.evaluate(
        sympy.exp(-((xc - 0.5) ** 2 + yc ** 2) / 0.03), T.coords).reshape(-1)
    adv = uw.systems.AdvDiffusionSUPG(child, T, sympy.Matrix([[-yc, xc]]))
    for b in ("Left", "Right", "Top", "Bottom"):
        adv.add_dirichlet_bc(0.0, b)
    adv.solve(timestep=0.02)
    assert adv.snes.getKSP().getPC().getType() == "asm"
    assert adv._custom_mg is None
    data = np.asarray(T.array[:, 0, 0])
    assert np.isfinite(data).all() and 0.9 < data.max() < 1.01


def test_estimate_dt_is_accuracy_based_and_resolution_on_request(mesh):
    """The default estimate follows the field, not the mesh; the resolution
    basis reproduces the semi-Lagrangian solver's cell-crossing time."""
    adv, T = _solver(mesh, "t")
    dt_acc = float(adv.estimate_dt())
    dt_res = float(adv.estimate_dt(basis="resolution"))
    assert np.isfinite(dt_acc) and dt_acc > 0 and np.isfinite(dt_res) and dt_res > 0
    # a tighter fraction is a proportionally smaller step
    assert float(adv.estimate_dt(fraction=0.01)) == pytest.approx(0.5 * dt_acc)

    x, y = mesh.X
    T2 = uw.discretisation.MeshVariable("T_t2", mesh, 1, degree=2)
    slcn = uw.systems.AdvDiffusionSLCN(mesh, T2, sympy.Matrix([[-y, x]]))
    slcn.constitutive_model = uw.constitutive_models.DiffusionModel
    slcn.constitutive_model.Parameters.diffusivity = 0.0
    assert dt_res == pytest.approx(float(slcn.estimate_dt()), rel=1e-12)

    # after a step the estimate uses the realised rate of change
    adv.solve(timestep=dt_acc)
    assert adv._last_change_rate > 0
    dt_after = float(adv.estimate_dt())
    assert np.isfinite(dt_after) and 0.2 * dt_acc < dt_after < 5 * dt_acc

    with pytest.raises(ValueError, match="basis must be"):
        adv.estimate_dt(basis="courant")
