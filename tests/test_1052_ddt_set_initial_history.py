"""Regression tests for SemiLagrangian / Eulerian DDt.set_initial_history.

The method is the supported entry point for planting BDF history at the
start of a run — used both for analytical-IC benchmarks (no startup
transient) and for checkpoint/restart (resume the multistep history
without an order ramp).

We instantiate a SemiLagrangian DDt directly (no Stokes solver) and
check the bookkeeping after set_initial_history.
"""

import warnings
import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.systems import ddt as ddt_module

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _make_semilagrangian(order):
    """Build a tiny SemiLagrangian DDt for SYM_TENSOR fields."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
    )
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=1)
    psi = sympy.zeros(2, 2)  # zero stress placeholder
    return ddt_module.SemiLagrangian(
        mesh,
        psi_fn=psi,
        V_fn=v.sym,
        vtype=uw.VarType.SYM_TENSOR,
        degree=2,
        continuous=False,
        order=order,
    )


class TestSetInitialHistory:

    def test_order1_marks_initialised(self):
        d = _make_semilagrangian(order=1)
        n_nodes = d.psi_star[0].array.shape[0]
        arr = np.zeros((n_nodes, 2, 2))
        arr[:, 0, 1] = arr[:, 1, 0] = 0.4
        d.set_initial_history([arr])
        assert d._history_initialised is True
        assert d._n_solves_completed == 1
        assert np.allclose(d.psi_star[0].array[:, 0, 1], 0.4)

    def test_order2_planted_history_and_dt(self):
        d = _make_semilagrangian(order=2)
        n_nodes = d.psi_star[0].array.shape[0]
        a = np.zeros((n_nodes, 2, 2)); a[:, 0, 1] = a[:, 1, 0] = 0.5
        b = np.zeros((n_nodes, 2, 2)); b[:, 0, 1] = b[:, 1, 0] = 0.4
        d.set_initial_history([a, b], dt=0.05)
        assert d._history_initialised is True
        assert d._n_solves_completed == 2
        assert d._dt_history == [0.05, 0.05]
        assert np.allclose(d.psi_star[0].array[:, 0, 1], 0.5)
        assert np.allclose(d.psi_star[1].array[:, 0, 1], 0.4)

    def test_scalar_broadcast(self):
        """Scalar values broadcast to whole field."""
        d = _make_semilagrangian(order=1)
        d.set_initial_history([0.7])
        assert np.allclose(d.psi_star[0].array, 0.7)

    def test_wrong_length_raises(self):
        d = _make_semilagrangian(order=2)
        with pytest.raises(ValueError, match="requires 2 value"):
            d.set_initial_history([0.0])

    def test_order2_no_dt_warns(self):
        d = _make_semilagrangian(order=2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            d.set_initial_history([0.0, 0.0])
            assert any("variable-dt" in str(rec.message) for rec in w)

    def test_order1_no_dt_silent(self):
        d = _make_semilagrangian(order=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            d.set_initial_history([0.0])
            assert not any("variable-dt" in str(rec.message) for rec in w)
