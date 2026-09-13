"""Every recorded operator says whether it admits a discrete adjoint.

The verdict is written when the operator runs, not when someone asks for a
gradient, so a run says where its adjoint breaks while it runs — instead of
that being discovered three hours into an inversion.

The verdicts are STRUCTURAL: about the operator as configured, not about
whether a driver exists yet. What they encode:

  * an implicit step is a residual: Jacobian transpose for the state,
    symbolic derivative for a parameter — supported;
  * a rotated constraint solves on a rotated operator in its own Krylov loop
    with no transpose path — refused;
  * a solve that did not converge is linearised about a state it never
    reached — refused, after the fact;
  * a semi-Lagrangian history's interpolation at the departure points is not
    materialised as an operator — refused, naming what is missing;
  * a swarm step is adjointable exactly when the particle set is fixed across
    it — checked by counting.

``transcript_adjoint_segments`` reads the verdicts back as the partition they
imply: strong-constraint within a segment, weak-constraint across a refusal.
"""

import warnings

import numpy as np
import pytest
import sympy

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _fresh_model():
    import underworld3 as uw

    uw.reset_default_model()
    return uw, uw.get_default_model()


@pytest.fixture(scope="module")
def mesh():
    import underworld3 as uw

    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8, qdegree=3
    )


def _poisson(uw, mesh, name):
    T = uw.discretisation.MeshVariable(name, mesh, 1, degree=2)
    solver = uw.systems.Poisson(mesh, u_Field=T)
    solver.constitutive_model = uw.constitutive_models.DiffusionModel
    solver.constitutive_model.Parameters.diffusivity = 1.0
    solver.f = 1.0
    solver.add_dirichlet_bc(0.0, "Top")
    solver.add_dirichlet_bc(0.0, "Bottom")
    solver.petsc_options.delValue("ksp_monitor")
    return solver


def _stokes(uw, mesh, tag):
    V = uw.discretisation.MeshVariable(f"V_{tag}", mesh, 2, degree=2)
    P = uw.discretisation.MeshVariable(f"P_{tag}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=V, pressureField=P)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = sympy.Matrix([0.0, -1.0])
    stokes.petsc_options.delValue("ksp_monitor")
    return stokes, V


def _adjoint_of(model, kind):
    events = [e for e in model.transcript[0].events if e["kind"] == kind]
    assert events, f"no {kind} event was recorded"
    return events[-1]["adjoint"]


# ---------------------------------------------------------------------------
# solves
# ---------------------------------------------------------------------------


def test_a_plain_implicit_solve_is_supported(mesh):
    uw, model = _fresh_model()
    solver = _poisson(uw, mesh, "T_adj_ok")
    model.tracker.time, model.tracker.step = 0.0, 0
    with model.step(0.1):
        solver.solve()
    verdict = _adjoint_of(model, "solve")
    assert verdict["supported"] is True
    assert "Jacobian transpose" in verdict["reason"]


def test_a_rotated_constraint_refuses_and_says_why(mesh):
    """The rotated solve runs its own Krylov loop on a rotated operator; there
    is no transpose path through it. The transcript must say so rather than
    let the solve pass as an ordinary residual."""
    uw, model = _fresh_model()
    stokes, _ = _stokes(uw, mesh, "rot")
    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
    stokes.add_dirichlet_bc((0.0, sympy.oo), "Left")
    stokes.add_dirichlet_bc((0.0, sympy.oo), "Right")
    stokes.add_rotated_freeslip_bc(0.0, "Top")
    model.tracker.time, model.tracker.step = 0.0, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with model.step(0.1):
            stokes.solve()
    verdict = _adjoint_of(model, "solve")
    assert verdict["supported"] is False
    assert "rotated" in verdict["reason"]


def test_an_unconverged_solve_is_refused_after_the_fact(mesh):
    """The structural verdict is written before the solve. A solve that then
    diverged is linearised about a state it never reached — the outcome must
    override the verdict."""
    uw, model = _fresh_model()
    solver = _poisson(uw, mesh, "T_adj_div")
    solver.petsc_options["ksp_max_it"] = 1
    solver.petsc_options["ksp_rtol"] = 1.0e-30
    solver.petsc_options["snes_max_it"] = 1
    model.tracker.time, model.tracker.step = 0.0, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with model.step(0.1):
            solver.solve()
    verdict = _adjoint_of(model, "solve")
    assert verdict["supported"] is False
    assert "did not converge" in verdict["reason"]


# ---------------------------------------------------------------------------
# histories
# ---------------------------------------------------------------------------


def _advdiff(uw, mesh, tag, order=1):
    T = uw.discretisation.MeshVariable(f"T_{tag}", mesh, 1, degree=2)
    V = uw.discretisation.MeshVariable(f"V_{tag}", mesh, 2, degree=2)
    x, y = mesh.X
    V.array[:, 0, :] = np.asarray(
        uw.function.evaluate(sympy.Matrix([[-(y - 0.5), (x - 0.5)]]), V.coords)
    ).reshape(-1, 2)
    solver = uw.systems.AdvDiffusion(mesh, u_Field=T, V_fn=V.sym, order=order)
    solver.constitutive_model = uw.constitutive_models.DiffusionModel
    solver.constitutive_model.Parameters.diffusivity = 1.0e-3
    solver.petsc_options.delValue("ksp_monitor")
    return solver


def test_an_eulerian_history_is_supported(mesh):
    """An implicit step IS a residual; the SUPG adjoint that passed its Taylor
    test at 1.00000 is exactly this case."""
    uw, model = _fresh_model()
    solver = _advdiff(uw, mesh, "eul")
    model.tracker.time, model.tracker.step = 0.0, 0
    with model.step(0.01):
        solver.solve(timestep=0.01)
    verdict = _adjoint_of(model, "history_shift")
    assert verdict["supported"] is True
    assert "owning solver" in verdict["reason"]


def test_a_semi_lagrangian_history_refuses_and_names_what_is_missing():
    uw, model = _fresh_model()
    # Its own mesh: the semi-Lagrangian trace-back fails point location on
    # the module's shared mesh after the Eulerian test has run on it under
    # pytest, though the same sequence passes as a script. Not chased here.
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8, qdegree=3
    )
    T = uw.discretisation.MeshVariable("T_sl", mesh, 1, degree=2)
    V = uw.discretisation.MeshVariable("V_sl", mesh, 2, degree=2)
    x, y = mesh.X
    V.array[:, 0, :] = np.asarray(
        uw.function.evaluate(sympy.Matrix([[-(y - 0.5), (x - 0.5)]]), V.coords)
    ).reshape(-1, 2)
    solver = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=V.sym)
    solver.constitutive_model = uw.constitutive_models.DiffusionModel
    solver.constitutive_model.Parameters.diffusivity = 1.0e-3
    solver.petsc_options.delValue("ksp_monitor")
    model.tracker.time, model.tracker.step = 0.0, 0
    with model.step(0.01):
        solver.solve(timestep=0.01)
    verdict = _adjoint_of(model, "history_shift")
    assert verdict["supported"] is False
    assert "departure" in verdict["reason"]
    assert "not materialised" in verdict["reason"]


def test_every_history_scheme_declares_a_verdict():
    """The base refuses by naming the class, so a scheme added without a
    verdict shows up as undeclared rather than passing as either."""
    from underworld3.systems import ddt

    base = ddt._DDtBase._adjoint_support
    silent = []
    for name in dir(ddt):
        cls = getattr(ddt, name)
        if (isinstance(cls, type) and issubclass(cls, ddt._DDtBase)
                and cls is not ddt._DDtBase
                and cls._adjoint_support is base):
            silent.append(name)
    assert silent == [], f"these history schemes declare no adjoint verdict: {silent}"


# ---------------------------------------------------------------------------
# swarms
# ---------------------------------------------------------------------------


def _swarm_in_flow(uw, mesh, V_fn_matrix):
    swarm = uw.swarm.Swarm(mesh)
    swarm.populate(fill_param=2)
    return swarm


def test_a_swarm_step_on_a_fixed_particle_set_is_supported(mesh):
    """A rotating flow keeps every particle inside the box."""
    uw, model = _fresh_model()
    x, y = mesh.X
    V = sympy.Matrix([[-(y - 0.5), (x - 0.5)]])
    swarm = _swarm_in_flow(uw, mesh, V)
    model.tracker.time, model.tracker.step = 0.0, 0
    with model.step(0.05):
        swarm.advection(V, 0.05)
    event = [e for e in model.transcript[0].events if e["kind"] == "swarm_advect"][-1]
    assert event["n_before"] == event["n_after"] > 0
    assert event["adjoint"]["supported"] is True
    assert "fixed particle set" in event["adjoint"]["reason"]


def test_a_swarm_step_that_loses_particles_refuses_with_the_count():
    """Particles leave the domain, the box's own return-to-bounds is switched
    off, so the migrate deletes them: the state changed dimension, and no
    linear map can be transposed across that.

    On its own mesh: switching the return-to-bounds off is a change to the
    mesh, and the module's shared one is used by the semi-Lagrangian test."""
    uw, model = _fresh_model()
    own = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8, qdegree=3
    )
    V = sympy.Matrix([[10.0, 0.0]])            # everything exits to the right
    swarm = _swarm_in_flow(uw, own, V)
    own.return_coords_to_bounds = None
    model.tracker.time, model.tracker.step = 0.0, 0
    with model.step(0.5):
        swarm.advection(V, 0.5)
    event = [e for e in model.transcript[0].events if e["kind"] == "swarm_advect"][-1]
    assert event["n_after"] < event["n_before"], event
    assert event["adjoint"]["supported"] is False
    assert f"{event['n_before']} -> {event['n_after']}" in event["adjoint"]["reason"]


# ---------------------------------------------------------------------------
# the partition
# ---------------------------------------------------------------------------


def _step(index, events, completed=True):
    return {"kind": "step", "index": index, "label": None, "t0": index * 0.1,
            "t1": (index + 1) * 0.1, "dt": 0.1, "completed": completed,
            "restorable": True, "wall": 0.1, "events": events}


def _solve(name, supported, reason="r"):
    return {"kind": "solve", "name": name, "part": f"{name}#1",
            "adjoint": {"supported": supported, "reason": reason}}


def test_segments_partition_the_window_at_the_refusals():
    import underworld3 as uw

    steps = [_step(i, [_solve("Stokes(v)", True)]) for i in range(6)]
    steps[3]["events"] = [_solve("Stokes(v)", False, "the particle set changed")]
    runs = [{"run": {"kind": "run", "model": "t"}, "steps": steps, "notes": []}]

    segments = uw.transcript_adjoint_segments(runs)
    assert [(s["first"], s["last"], s["supported"]) for s in segments] == [
        (0, 2, True), (3, 3, False), (4, 5, True)
    ]
    assert segments[1]["refusals"] == [("Stokes(v)", "the particle set changed")]


def test_segments_leave_abandoned_steps_out_and_flag_undeclared_verdicts():
    import underworld3 as uw

    steps = [
        _step(0, [_solve("Stokes(v)", True)]),
        _step(1, [_solve("Stokes(v)", True)], completed=False),
        _step(1, [{"kind": "solve", "name": "Old(v)", "part": "Old(v)#1"}]),
    ]
    runs = [{"run": {"kind": "run", "model": "t"}, "steps": steps, "notes": []}]
    segments = uw.transcript_adjoint_segments(runs)
    assert [s["steps"] for s in segments] == [1, 1]
    assert segments[1]["supported"] is False
    assert "without an adjoint verdict" in segments[1]["refusals"][0][1]


# ---------------------------------------------------------------------------
# the text transcript
# ---------------------------------------------------------------------------


def test_the_text_transcript_notes_a_refusal_once_per_change(tmp_path):
    """A semi-Lagrangian run refuses identically every step. The note is
    written when the set of refusals CHANGES — on the first refusing step, and
    again when it clears — not three hundred times."""
    uw, model = _fresh_model()
    model.transcript_file = tmp_path / "t.log"
    model.transcript_format = "text"
    model.tracker.time, model.tracker.step = 0.0, 0

    def refusing():
        model._record_step_event(
            "history_shift", "SemiLagrangian(T)", dt=0.1,
            part="SemiLagrangian#9",
            adjoint={"supported": False, "reason": "not materialised"})

    for n in range(6):
        with model.step(0.1):
            if n in (1, 2, 4):
                refusing()

    lines = (tmp_path / "t.log").read_text().splitlines()
    refused = [l for l in lines if "no adjoint through" in l]
    cleared = [l for l in lines if "admits one again" in l]
    assert len(refused) == 2, refused          # steps 1 and 4, not 1, 2 and 4
    assert len(cleared) == 2, cleared          # steps 3 and 5
    # a clean run says nothing: step 0 is one line, with no note under it
    assert "admits one" not in lines[lines.index(next(l for l in lines if l.strip().startswith("0 "))) + 1]
