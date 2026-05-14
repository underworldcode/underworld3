"""Regression tests for the ``monotone_mode`` kwarg on
``SNES_AdvectionDiffusion`` / ``AdvDiffusionSLCN``.

The underlying ``SemiLagrangian_DDt`` already accepts ``monotone_mode``
(landed in PR #186). This kwarg forwards it through the solver
constructor so users can write the one-line idiom
``adv_diff = AdvDiffusionSLCN(..., monotone_mode="clamp")`` instead of
the two-step ``adv_diff.DuDt.monotone_mode = "clamp";
adv_diff.DFDt.monotone_mode = "clamp"`` dance.
"""

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
