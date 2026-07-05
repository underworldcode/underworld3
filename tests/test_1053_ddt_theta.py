"""Regression tests for SemiLagrangian DDt ``theta`` parameter.

``theta`` controls the Adams-Moulton flux integrator's coefficient at
order 1: AM coefficients are ``[θ, 1-θ]``. Default 0.5 is Crank-
Nicolson (legacy SLCN); 1.0 is Backward Euler.

These tests verify the constructor parameter is plumbed through to the
AM coefficient values and that mutation after construction is picked
up on the next update.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.systems import ddt as ddt_module

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _make_sl(theta=0.5, order=1):
    """Build a tiny SemiLagrangian DDt for a scalar field."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4),
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
    )
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=1)
    psi = sympy.zeros(1, 1)
    return ddt_module.SemiLagrangian(
        mesh,
        psi_fn=psi,
        V_fn=v.sym,
        vtype=uw.VarType.SCALAR,
        degree=2,
        continuous=False,
        order=order,
        theta=theta,
    )


class TestTheta:

    def test_default_theta_is_cn(self):
        d = _make_sl()
        assert d.theta == 0.5
        # __init__ ran _update_am_values(self._am_coeffs, 1, theta=0.5)
        assert float(d._am_coeffs[0].sym) == pytest.approx(0.5)
        assert float(d._am_coeffs[1].sym) == pytest.approx(0.5)

    def test_theta_one_is_backward_euler(self):
        d = _make_sl(theta=1.0)
        assert d.theta == 1.0
        # Order-1 AM coefficients with θ=1.0: [1.0, 0.0]
        assert float(d._am_coeffs[0].sym) == pytest.approx(1.0)
        assert float(d._am_coeffs[1].sym) == pytest.approx(0.0)

    def test_theta_zero_is_forward_euler(self):
        d = _make_sl(theta=0.0)
        assert d.theta == 0.0
        # AM coefficients with θ=0.0: [0.0, 1.0]
        assert float(d._am_coeffs[0].sym) == pytest.approx(0.0)
        assert float(d._am_coeffs[1].sym) == pytest.approx(1.0)

    def test_theta_attribute_settable_after_construction(self):
        d = _make_sl(theta=0.5)
        # Mutate the instance attribute
        d.theta = 1.0
        # Re-run the per-step AM-coefficient update path manually
        # (replicates what update_pre_solve does on each step).
        ddt_module._update_am_values(
            d._am_coeffs, d.effective_order, d.theta)
        assert float(d._am_coeffs[0].sym) == pytest.approx(1.0)
        assert float(d._am_coeffs[1].sym) == pytest.approx(0.0)

    def test_theta_default_unchanged_when_omitted(self):
        """Backwards compatibility: omitting theta keeps legacy CN."""
        mesh = uw.meshing.StructuredQuadBox(
            elementRes=(4, 4),
            minCoords=(0.0, 0.0),
            maxCoords=(1.0, 1.0),
        )
        v = uw.discretisation.MeshVariable(
            "U_legacy", mesh, mesh.dim, degree=1)
        psi = sympy.zeros(1, 1)
        d = ddt_module.SemiLagrangian(
            mesh, psi_fn=psi, V_fn=v.sym,
            vtype=uw.VarType.SCALAR, degree=2, continuous=False,
            order=1)
        assert d.theta == 0.5
        assert float(d._am_coeffs[0].sym) == pytest.approx(0.5)
        assert float(d._am_coeffs[1].sym) == pytest.approx(0.5)

    def test_state_restore_preserves_theta(self):
        """Regression (2026-07 audit, D9a / READ-45): restoring a state
        snapshot re-derives the AM coefficients — it must use the
        instance's theta, not a hard-coded Crank-Nicolson 0.5."""
        d = _make_sl(theta=1.0)
        snapshot = d.state
        d.state = snapshot
        # Backward-Euler coefficients [1.0, 0.0] must survive the restore.
        assert float(d._am_coeffs[0].sym) == pytest.approx(1.0)
        assert float(d._am_coeffs[1].sym) == pytest.approx(0.0)
