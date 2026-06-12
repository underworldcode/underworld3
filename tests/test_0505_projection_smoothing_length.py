"""Locks the `smoothing_length` API on the four Projection classes.

The projection solvers form
    u  −  ∇·(α ∇u)  =  ũ,
i.e. a screened-Poisson smoother whose Green's function decays as
exp(−r/L) with L = √α. The L-valued, unit-aware accessor
`smoothing_length` is a convenience view on `smoothing = α`; this test
locks the round-trip and the Pint-unit handling.
"""
import numpy as np
import pytest

import underworld3 as uw
import underworld3.systems as sys


@pytest.fixture(scope="module")
def mesh():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=0.2, qdegree=2,
    )


@pytest.mark.tier_a
@pytest.mark.level_1
def test_scalar_smoothing_length_roundtrip(mesh):
    """L=0.05  ⇒  α=L²=0.0025; reading back gives L (unit-aware)."""
    V = uw.discretisation.MeshVariable(
        "V_sl", mesh, vtype=uw.VarType.SCALAR, degree=2, continuous=True)
    proj = sys.Projection(mesh, V)
    proj.smoothing_length = 0.05
    assert float(proj.smoothing) == pytest.approx(0.0025)
    L = proj.smoothing_length
    # In an active scaling context this is a Pint Quantity in metres;
    # without one it's a plain float — both should round-trip.
    L_num = getattr(L, "magnitude", L)
    assert float(L_num) == pytest.approx(0.05)


@pytest.mark.tier_a
@pytest.mark.level_1
def test_vector_smoothing_length_roundtrip(mesh):
    W = uw.discretisation.MeshVariable(
        "W_sl", mesh, vtype=uw.VarType.VECTOR, degree=2, continuous=True)
    proj = sys.Vector_Projection(mesh, W)
    proj.smoothing_length = 0.1
    assert float(proj.smoothing) == pytest.approx(0.01)
    L_num = getattr(proj.smoothing_length, "magnitude", proj.smoothing_length)
    assert float(L_num) == pytest.approx(0.1)


@pytest.mark.tier_a
@pytest.mark.level_1
def test_tensor_smoothing_length_roundtrip(mesh):
    T = uw.discretisation.MeshVariable(
        "T_sl", mesh, vtype=uw.VarType.SYM_TENSOR, degree=2, continuous=True)
    Tw = uw.discretisation.MeshVariable(
        "Tw_sl", mesh, vtype=uw.VarType.SCALAR, degree=2, continuous=True)
    proj = sys.Tensor_Projection(mesh, T, Tw)
    proj.smoothing_length = 0.2
    assert float(proj.smoothing) == pytest.approx(0.04)
    L_num = getattr(proj.smoothing_length, "magnitude", proj.smoothing_length)
    assert float(L_num) == pytest.approx(0.2)


@pytest.mark.tier_a
@pytest.mark.level_1
def test_multicomponent_smoothing_length_roundtrip(mesh):
    proj = sys.solvers.SNES_MultiComponent_Projection(
        mesh, n_components=3)
    proj.smoothing_length = 0.15
    assert float(proj.smoothing) == pytest.approx(0.0225)
    L_num = getattr(proj.smoothing_length, "magnitude", proj.smoothing_length)
    assert float(L_num) == pytest.approx(0.15)


@pytest.mark.tier_a
@pytest.mark.level_1
def test_smoothing_setter_keeps_alpha_view(mesh):
    """Setting `smoothing` (α) makes `smoothing_length` = √α."""
    V = uw.discretisation.MeshVariable(
        "V_smooth", mesh, vtype=uw.VarType.SCALAR, degree=2, continuous=True)
    proj = sys.Projection(mesh, V)
    proj.smoothing = 0.16
    assert float(proj.smoothing) == pytest.approx(0.16)
    L_num = getattr(proj.smoothing_length, "magnitude", proj.smoothing_length)
    assert float(L_num) == pytest.approx(0.4)  # √0.16


@pytest.mark.tier_a
@pytest.mark.level_1
def test_smoothing_length_pint_quantity_roundtrip(mesh):
    """Pint Quantity input round-trips back as a Pint Quantity in metres.

    Setting `smoothing_length = 0.05 * meter` under an active scaling
    context with a 1000 m reference length stores 5e-5 (non-dim) and
    α = (5e-5)² = 2.5e-9. Reading back returns a Pint Quantity that
    re-dimensionalises to 0.05 m.
    """
    model = uw.Model()
    model.set_reference_quantities(
        length=uw.quantity(1000, "m"),
        nondimensional_scaling=True,
    )
    uw.use_nondimensional_scaling(True)
    try:
        V = uw.discretisation.MeshVariable(
            "V_pint", mesh, vtype=uw.VarType.SCALAR, degree=2, continuous=True)
        proj = sys.Projection(mesh, V)
        proj.smoothing_length = uw.quantity(0.05, "m")

        # Non-dim store: 0.05 m / 1000 m = 5e-5; α = 2.5e-9.
        assert float(proj.smoothing) == pytest.approx(2.5e-9)

        # Pint-quantity round-trip.
        L_out = proj.smoothing_length
        assert hasattr(L_out, "magnitude"), \
            f"Pint input should round-trip as a Pint Quantity, got {type(L_out)}"
        assert float(L_out.to("m").magnitude) == pytest.approx(0.05)
    finally:
        uw.use_nondimensional_scaling(False)


@pytest.mark.tier_a
@pytest.mark.level_1
def test_smoothing_length_negative_raises(mesh):
    """Negative `smoothing_length` is ill-posed and must raise ValueError."""
    V = uw.discretisation.MeshVariable(
        "V_neg", mesh, vtype=uw.VarType.SCALAR, degree=2, continuous=True)
    proj = sys.Projection(mesh, V)
    with pytest.raises(ValueError, match="smoothing_length must be"):
        proj.smoothing_length = -0.1


@pytest.mark.tier_a
@pytest.mark.level_1
def test_smoothing_length_smoothes_a_step(mesh):
    """End-to-end: smoothing a step function at scale L attenuates the
    short-wavelength content. We check that the projected field's
    peak-to-peak range shrinks as L grows, which is the screened-Poisson
    low-pass behaviour the docstring promises."""
    import sympy
    x, y = mesh.X
    V = uw.discretisation.MeshVariable(
        "V_filter", mesh, vtype=uw.VarType.SCALAR, degree=2, continuous=True)
    # Sharp step in x at x = 0.5
    target = sympy.Piecewise((1.0, x < 0.5), (0.0, True))
    ranges = {}
    for L in (0.0, 0.05, 0.2):
        proj = sys.Projection(mesh, V)
        proj.uw_function = target
        proj.smoothing_length = L
        proj.solve()
        ranges[L] = float(V.data.max() - V.data.min())
    # Pure L2 projection should saturate the step (≈ 1.0)
    assert ranges[0.0] > 0.95
    # Larger L should produce a smoother (smaller-range) reconstruction
    assert ranges[0.2] < ranges[0.05] < ranges[0.0]
