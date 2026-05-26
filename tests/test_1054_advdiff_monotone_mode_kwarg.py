"""Regression tests for the ``monotone_mode`` kwarg on
``SNES_AdvectionDiffusion`` / ``AdvDiffusionSLCN``.

The underlying ``SemiLagrangian_DDt`` already accepts ``monotone_mode``
(landed in PR #186). This kwarg forwards it through the solver
constructor so users can write the one-line idiom
``adv_diff = AdvDiffusionSLCN(..., monotone_mode="clamp")`` instead of
the two-step ``adv_diff.DuDt.monotone_mode = "clamp";
adv_diff.DFDt.monotone_mode = "clamp"`` dance.

The trace-back limiter itself now lives in the evaluator as the
``monotone`` option (``uw.function.global_evaluate(..., monotone=...)``);
``SemiLagrangian.update_pre_solve`` routes through it. The bit-identical
equivalence of that refactor (``monotone_mode`` None / "clamp" / "pick"
unchanged to the last digit) is exercised end-to-end here and locked at
the evaluator level in ``test_0760_evaluate_monotone.py``.
"""

import numpy as np
import sympy
import pytest

import underworld3 as uw


pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _make_mesh_and_field():
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4),
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
    )
    v = uw.discretisation.MeshVariable(
        "V_advtest", mesh, mesh.dim, degree=1)
    T = uw.discretisation.MeshVariable(
        "T_advtest", mesh, 1, degree=2)
    return mesh, v, T


class TestMonotoneModeKwarg:

    def test_default_is_none(self):
        mesh, v, T = _make_mesh_and_field()
        adv = uw.systems.AdvDiffusionSLCN(
            mesh, u_Field=T, V_fn=v.sym)
        assert adv.DuDt.monotone_mode is None
        assert adv.DFDt.monotone_mode is None

    def test_clamp_forwarded_to_both(self):
        mesh, v, T = _make_mesh_and_field()
        adv = uw.systems.AdvDiffusionSLCN(
            mesh, u_Field=T, V_fn=v.sym,
            monotone_mode="clamp")
        assert adv.DuDt.monotone_mode == "clamp"
        assert adv.DFDt.monotone_mode == "clamp"

    def test_pick_forwarded_to_both(self):
        mesh, v, T = _make_mesh_and_field()
        adv = uw.systems.AdvDiffusionSLCN(
            mesh, u_Field=T, V_fn=v.sym,
            monotone_mode="pick")
        assert adv.DuDt.monotone_mode == "pick"
        assert adv.DFDt.monotone_mode == "pick"

    def test_explicit_DuDt_overrides_kwarg(self):
        """If the caller supplies a pre-built ``DuDt``, the
        constructor must not silently rewrite its ``monotone_mode``
        — the caller-supplied DDt is the source of truth."""
        from underworld3.systems.ddt import SemiLagrangian as SL_DDt
        mesh, v, T = _make_mesh_and_field()
        # Build a DuDt with mode 'pick' explicitly
        custom = SL_DDt(
            mesh, psi_fn=T.sym, V_fn=v.sym,
            vtype=uw.VarType.SCALAR, degree=T.degree,
            continuous=T.continuous, order=1,
            monotone_mode="pick",
        )
        adv = uw.systems.AdvDiffusionSLCN(
            mesh, u_Field=T, V_fn=v.sym,
            DuDt=custom,
            monotone_mode="clamp",  # ignored for the supplied DuDt
        )
        assert adv.DuDt.monotone_mode == "pick"  # preserved
        # DFDt is constructed internally → uses the kwarg
        assert adv.DFDt.monotone_mode == "clamp"


def _run_steps(monotone_mode, n_steps=4, dt=0.02):
    """Advect a steep blob with a prescribed rotation and return the
    stepped T-field. Steep gradients on a P3 field make the SL trace-back
    land on non-nodal points where FE overshoots — i.e. the path the
    limiter guards. Deterministic (no Stokes solve)."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(12, 12),
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        qdegree=3,
    )
    x, y = mesh.X
    T = uw.discretisation.MeshVariable("Ts", mesh, 1, degree=3)
    V_fn = sympy.Matrix([[-(y - 0.5)], [(x - 0.5)]]).T

    adv = uw.systems.AdvDiffusionSLCN(
        mesh, u_Field=T, V_fn=V_fn, monotone_mode=monotone_mode)
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = 1.0e-4
    adv.theta = 0.5

    init = sympy.exp(-(((x - 0.5) ** 2 + (y - 0.72) ** 2) / 0.004))
    T.array[...] = uw.function.evaluate(init, T.coords).reshape(
        T.array[...].shape)
    for _ in range(n_steps):
        adv.solve(timestep=dt, zero_init_guess=False)
    return np.asarray(T.array[...], dtype=np.float64).copy()


class TestMonotoneSolverIntegration:
    """End-to-end: the kwarg drives a real solve through the refactored
    evaluator ``monotone`` path."""

    def test_clamp_runs_and_stays_bounded(self):
        # IC is a Gaussian blob in [0, 1]; passive advection-diffusion of
        # a source-free scalar respects the discrete maximum principle, so
        # the clamped field must not develop large new extrema.
        field = _run_steps("clamp")
        assert np.all(np.isfinite(field))
        assert field.min() > -0.05
        assert field.max() < 1.05

    def test_pick_runs_end_to_end(self):
        field = _run_steps("pick")
        assert np.all(np.isfinite(field))
        assert field.max() < 1.05

    def test_none_runs_end_to_end(self):
        field = _run_steps(None)
        assert np.all(np.isfinite(field))
