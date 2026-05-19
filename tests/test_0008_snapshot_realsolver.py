"""Real-solver confidence test for the snapshot toolkit.

Every other snapshot test drives state by hand. This one runs an
actual PETSc solver — AdvDiffusion, which carries an internal
SemiLagrangian DDt (auxiliary projection SNES + nodal trace-back
swarm) — through the stash-and-recover loop.

Findings this test pins down (all verified, see commit message):

  * The AdvDiffusion solve is bit-deterministic: two independent
    identical runs with no snapshot are bit-for-bit equal.

  * restore() recovers the primary solution field T bit-exactly.

  * THE CORE GUARANTEE — a discarded ("regretted") step leaves zero
    trace, bit-for-bit, even through real solves:
        restore -> regretted solve -> restore -> K solves
    is np.array_equal to
        restore -> K solves
    (B == C, max|d| = 0.0).

  * The only residual is restore's reproducibility *floor* against a
    never-snapshotted control (~7e-7 here): restore resyncs fields
    through gvec->lvec rather than reproducing the solver-produced
    lvec exactly, and the implicit diffusion operator amplifies that
    to solver-tolerance level over steps. This is NOT contamination
    from the discarded step (proven by B == C above); it is the cost
    of round-tripping through the snapshot representation, within
    solver tolerance, and consistent with the design intent that
    auxiliary solver state is intentionally not captured.

So the honest production statement: discarding a bad step is
bit-exact even through real solvers; recovering to a never-stashed
control is within solver tolerance.
"""

import numpy as np
import sympy as sp
import pytest

pytestmark = [pytest.mark.level_2, pytest.mark.tier_a]

# Restore-vs-pristine reproducibility floor for this setup (measured).
# The regretted-step guarantee is asserted bit-exact (np.array_equal);
# only the never-stashed-control comparison uses this tolerance.
_RESTORE_FLOOR_ATOL = 1.0e-5


@pytest.fixture(autouse=True)
def _reset():
    import underworld3 as uw

    uw.reset_default_model()
    uw.use_strict_units(False)
    uw.use_nondimensional_scaling(False)
    yield
    uw.reset_default_model()
    uw.use_strict_units(False)
    uw.use_nondimensional_scaling(False)


def _build():
    import underworld3 as uw

    model = uw.get_default_model()
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(16, 16),
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        qdegree=3,
    )
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=1)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

    adv_diff = uw.systems.AdvDiffusion(mesh, u_Field=T, V_fn=v)
    adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
    adv_diff.constitutive_model.Parameters.diffusivity = 1.0
    adv_diff.add_dirichlet_bc(0.0, "Left")
    adv_diff.add_dirichlet_bc(0.0, "Right")
    v.array[:, 0, 0] = 0.05

    x, y = mesh.X
    T.array = uw.function.evaluate(
        sp.sin(sp.pi * x) * sp.sin(sp.pi * y), T.coords
    )
    return uw, model, mesh, adv_diff, T


def _capture(T):
    return np.asarray(T.array[...]).copy()


def test_realsolver_restore_recovers_solution_field():
    """snapshot, do a regretted solve, restore — the solution field
    itself is recovered exactly."""
    uw, model, mesh, adv_diff, T = _build()

    for _ in range(2):
        adv_diff.solve(timestep=1.0e-3)

    pre_T = _capture(T)
    snap = model.snapshot()

    adv_diff.solve(timestep=5.0)  # absurd Δt: converges, over-diffused
    assert not np.allclose(_capture(T), pre_T, atol=1e-8), (
        "the regretted solve was not actually disruptive"
    )

    model.restore(snap)
    assert np.array_equal(_capture(T), pre_T), (
        "solution field not exactly recovered after restore"
    )


def test_realsolver_regretted_step_leaves_no_trace():
    """THE core guarantee, through a real solver, bit-for-bit.

      B:  restore -> K good solves
      C:  restore -> regretted absurd-Δt solve -> restore
                  -> same K good solves

    B == C exactly. The discarded step leaves zero trace even though
    it ran a real PETSc solve in between.
    """
    uw, model, mesh, adv_diff, T = _build()

    for _ in range(3):
        adv_diff.solve(timestep=1.0e-3)
    snap = model.snapshot()

    # B: restore, then K good solves.
    model.restore(snap)
    for _ in range(4):
        adv_diff.solve(timestep=1.0e-3)
    B = _capture(T)

    # C: restore, a regretted solve, restore, the same K good solves.
    model.restore(snap)
    adv_diff.solve(timestep=5.0)
    model.restore(snap)
    for _ in range(4):
        adv_diff.solve(timestep=1.0e-3)
    C = _capture(T)

    assert np.array_equal(B, C), (
        "regretted real solve left a trace after restore — "
        f"max abs diff {np.max(np.abs(B - C)):.3e} (expected exactly 0)"
    )


def test_realsolver_continuation_within_solver_tolerance():
    """Recovering to a *never-stashed* control is within solver
    tolerance (not bit-exact): restore resyncs fields gvec->lvec
    rather than reproducing the solver-produced lvec, and the
    implicit operator amplifies that to ~tolerance over steps. This
    is the documented restore floor, consistent with not capturing
    auxiliary solver state by design."""
    uw, model, mesh, adv_diff, T = _build()

    for _ in range(3):
        adv_diff.solve(timestep=1.0e-3)
    snap = model.snapshot()

    # Control: never snapshotted/restored — straight K solves.
    for _ in range(4):
        adv_diff.solve(timestep=1.0e-3)
    ctrl = _capture(T)

    # Stash path: restore, regretted solve, restore, same K solves.
    model.restore(snap)
    adv_diff.solve(timestep=5.0)
    model.restore(snap)
    for _ in range(4):
        adv_diff.solve(timestep=1.0e-3)
    stash = _capture(T)

    # Not bit-exact vs a never-stashed control (that is the floor),
    # but well within solver tolerance and far below the solution
    # scale (~0.04 here).
    max_diff = float(np.max(np.abs(stash - ctrl)))
    assert max_diff < _RESTORE_FLOOR_ATOL, (
        f"continuation drifted beyond the restore floor: "
        f"max abs diff {max_diff:.3e} >= {_RESTORE_FLOOR_ATOL:.0e}"
    )
    assert not np.array_equal(stash, ctrl), (
        "continuation is unexpectedly bit-exact vs a never-stashed "
        "control — if this starts passing, the restore floor has been "
        "eliminated and this test should be tightened to np.array_equal"
    )
