"""The two JIT routes, and the switch between them (#823, tier 2).

``"graph"`` (the default) compiles each named quantity once, as one C temporary;
``"expanded"`` is the JIT before tier 2, which expands every named quantity into one
expression. The expanded route is kept as a fallback and a reference: if a model
misbehaves on one route and not the other, the JIT is at fault; if on both, the model
is. These tests hold the switch to its contract and the two routes to the same
answers, so that the fallback cannot rot unnoticed.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


@pytest.fixture(autouse=True)
def default_route(monkeypatch):
    monkeypatch.delenv("UW_JIT_ROUTE", raising=False)
    yield
    uw.use_jit_route(None)


def test_the_route_setting(monkeypatch):
    from underworld3.utilities._jitextension import resolve_jit_route

    assert uw.jit_route() == "graph"
    monkeypatch.setenv("UW_JIT_ROUTE", "expanded")
    assert uw.jit_route() == "expanded"
    uw.use_jit_route("graph")                    # the call wins over the environment
    assert uw.jit_route() == "graph"
    assert resolve_jit_route("expanded") == "expanded"   # an explicit route wins
    uw.use_jit_route(None)
    assert uw.jit_route() == "expanded"
    with pytest.raises(ValueError, match="JIT route"):
        uw.use_jit_route("tree")
    monkeypatch.setenv("UW_JIT_ROUTE", "nonsense")
    with pytest.raises(ValueError, match="JIT route"):
        uw.jit_route()


def _viscoplastic_box(name, tangent):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25)
    v = uw.discretisation.MeshVariable("V" + name, mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P" + name, mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    P = stokes.constitutive_model.Parameters
    P.shear_viscosity_0 = 1.0
    P.yield_stress = (uw.expression(r"C_{" + name + "}", 0.3)
                      + uw.expression(r"\mu_{" + name + "}", 0.2) * p.sym[0])
    P.yield_stress_min = uw.expression(r"\tau_{" + name + "}", 0.01)
    stokes.add_dirichlet_bc((1.0, 0.0), "Top")
    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
    stokes.bodyforce = sympy.Matrix([0, -1])
    stokes.consistent_jacobian = tangent
    stokes.tolerance = 1.0e-9
    stokes.petsc_options["snes_max_it"] = 200     # Picard converges linearly here
    return stokes, v, p


@pytest.mark.parametrize("tangent", [True, False])
def test_both_routes_solve_a_viscoplastic_box_alike(tangent):
    """The same solve, Newton or Picard, on each route: the same nonlinear and linear
    iteration counts and the same solution, to round-off."""
    uw.reset_default_model()
    results = {}
    for route in ("graph", "expanded"):
        stokes, v, p = _viscoplastic_box(f"0026{route[0]}{int(tangent)}", tangent)
        stokes.jit_route = route
        stokes.solve(zero_init_guess=True)
        report = stokes.solve_report
        assert stokes.snes.getConvergedReason() > 0, (route, report.reason_str)
        results[route] = (report.nl_its, report.ksp_its,
                          np.array(v.array), np.array(p.array))
    g, e = results["graph"], results["expanded"]
    assert g[:2] == e[:2], (g[:2], e[:2])
    assert np.max(np.abs(g[2] - e[2])) <= 1.0e-9 * np.max(np.abs(e[2]))
    assert np.max(np.abs(g[3] - e[3])) <= 1.0e-9 * np.max(np.abs(e[3]))


def test_switching_one_solver_rebuilds_it_and_keeps_its_answer(monkeypatch):
    """A solver's jit_route overrides the process default; changing it rebuilds that
    solver's kernels at the next solve (a new module) and nothing else."""
    import underworld3.utilities._jitextension as jx

    headers = []
    generate = jx.generate_c_source

    def keep(*args, **kwargs):
        modname, codeguys, diag = generate(*args, **kwargs)
        headers.append(dict(codeguys)["cy_ext.h"])
        return modname, codeguys, diag

    monkeypatch.setattr(jx, "generate_c_source", keep)
    uw.reset_default_model()
    stokes, v, p = _viscoplastic_box("0026s", True)
    stokes.solve(zero_init_guess=True)
    first_key, first_v = stokes._current_jit_cache_key, np.array(v.array)
    assert "uwt_" in headers[-1]                 # the default is the graph

    stokes.jit_route = "expanded"
    stokes.solve(zero_init_guess=True)
    assert stokes._current_jit_cache_key != first_key
    assert "uwt_" not in headers[-1]             # one expanded expression per output
    assert np.max(np.abs(np.array(v.array) - first_v)) <= 1.0e-9 * np.max(np.abs(first_v))

    stokes.jit_route = None                      # back to the default
    stokes.solve(zero_init_guess=True)
    assert stokes._current_jit_cache_key == first_key
    with pytest.raises(ValueError, match="JIT route"):
        stokes.jit_route = "tree"


def test_the_process_default_reaches_new_solvers():
    uw.reset_default_model()
    uw.use_jit_route("expanded")
    stokes, v, p = _viscoplastic_box("0026d", True)
    assert stokes.jit_route is None and stokes._jit_route_in_use() == "expanded"
    stokes.solve(zero_init_guess=True)
    assert stokes.snes.getConvergedReason() > 0
