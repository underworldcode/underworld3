"""estimate_dt must return a SCALAR, not a zero-dimensional array.

Every estimate_dt implementation funnels its nondimensional result through
``np.squeeze``. That collapses a numpy scalar to a numpy scalar but promotes a
plain Python float to a 0-d ``ndarray`` -- which is neither of the documented
return types, and which poisons the arithmetic downstream: ``solve(timestep=dt)``
compares the estimate against ``self.delta_t`` (a UWexpression), and
ndarray-vs-sympy comparison raises ``TypeError: Could not convert object to
sequence`` instead of deferring to sympy.

The contract used to hold only by accident -- ``get_min_radius`` happened to
return a numpy scalar, so the squeeze was a no-op. When it started returning a
plain float (the #405 empty-rank work), the transient Darcy solve broke. These
tests pin the contract itself so it cannot depend on a caller's numeric type
again.
"""

import numpy as np
import pytest
import sympy as sp

import underworld3 as uw
from underworld3.systems.solvers import _as_scalar

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def test_as_scalar_collapses_zero_dim_arrays_only():
    """0-d arrays become scalars; everything else is passed through."""
    collapsed = _as_scalar(np.squeeze(0.5))
    assert not isinstance(collapsed, np.ndarray), (
        "a 0-d array must not survive as an array")
    assert collapsed == 0.5

    # Real arrays and plain scalars are untouched.
    arr = np.array([1.0, 2.0])
    assert _as_scalar(arr) is arr
    assert _as_scalar(0.5) == 0.5
    assert _as_scalar(np.float64(0.5)) == 0.5


def test_estimate_dt_returns_a_usable_scalar():
    """The value estimate_dt returns must be accepted by solve(timestep=...).

    Regression: a 0-d ndarray reached ``timestep != self.delta_t`` and raised
    TypeError against the UWexpression, so the solver could not be stepped with
    its own timestep estimate.
    """
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 8))
    h = uw.discretisation.MeshVariable("h1007", mesh, 1, degree=2)
    v = uw.discretisation.MeshVariable("v1007", mesh, mesh.dim, degree=1)

    darcy = uw.systems.TransientDarcy(mesh, h, v, order=1, theta=0.5)
    darcy.constitutive_model = uw.constitutive_models.DarcyFlowModel
    darcy.constitutive_model.Parameters.permeability = 1.0
    darcy.constitutive_model.Parameters.s = sp.Matrix([0, 0]).T
    darcy.storage = 1.0
    darcy.f = 0.0

    dt = darcy.estimate_dt()

    assert not (isinstance(dt, np.ndarray) and dt.ndim == 0), (
        f"estimate_dt returned a 0-d array ({dt!r}); callers compare this "
        "against a UWexpression and numpy raises rather than deferring")
    assert np.isfinite(float(dt)) and float(dt) > 0.0

    # The comparison that actually broke, exercised directly.
    assert (dt != darcy.delta_t) is not NotImplemented
