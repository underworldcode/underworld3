"""Default-on solver difficulty reporting + bounded/resumable difficulty solves.

Covers the general base-solver capability (on ``SolverBaseClass``, so every SNES-based
solver gets it):

1. ``test_solve_report_default_on`` — every ``solve()`` leaves a ``SolveReport`` (no opt-in),
   with sensible iteration counts / residuals, and the property is read-only.
2. ``test_solve_report_scalar`` — the same report is populated for a scalar (Poisson) solver,
   proving it lives on the base class, not Stokes only.
3. ``test_estimate_difficulty_bounded`` — an iteration-capped probe stops at the cap
   (``DIVERGED_MAX_IT``), is flagged ``bounded``, and reports the effort spent.
4. ``test_resume_is_lossless_and_same_termination`` — a chunked start/stop/restart chain
   reproduces the uninterrupted solution AND terminates at the same ‖F‖ (anchored to the
   original ‖F0‖, not the tighter restart residual).
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.systems.solve_report import SolveReport, reason_string, contraction


def _linear_stokes(cell=0.25):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=cell
    )
    v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True)
    st = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    st.constitutive_model = uw.constitutive_models.ViscousFlowModel
    st.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, y = mesh.X
    st.bodyforce = sympy.Matrix([0, sympy.sin(np.pi * x) * sympy.sin(np.pi * y)])
    for w in ("Bottom", "Top"):
        st.add_dirichlet_bc((sympy.oo, 0.0), w)
    for w in ("Left", "Right"):
        st.add_dirichlet_bc((0.0, sympy.oo), w)
    return st, v


def _nonlinear_stokes(cell=0.2):
    """Shear-thinning viscosity -> the default (Picard) tangent takes several iterations."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=cell
    )
    v = uw.discretisation.MeshVariable("Un", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("Pn", mesh, 1, degree=1, continuous=True)
    st = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    st.constitutive_model = uw.constitutive_models.ViscousFlowModel
    edot = st.Unknowns.Einv2 + 1.0e-8
    st.constitutive_model.Parameters.shear_viscosity_0 = (1.0e-2 + edot) ** (-0.4)
    st.tolerance = 1.0e-6
    x, y = mesh.X
    st.bodyforce = sympy.Matrix([0, 2.0 * sympy.sin(np.pi * x) * sympy.sin(np.pi * y)])
    for w in ("Bottom", "Top"):
        st.add_dirichlet_bc((sympy.oo, 0.0), w)
    for w in ("Left", "Right"):
        st.add_dirichlet_bc((0.0, sympy.oo), w)
    return st, v


@pytest.mark.level_1
@pytest.mark.tier_a
def test_solve_report_default_on():
    st, _ = _linear_stokes()
    assert st.solve_report is None            # nothing before the first solve
    st.solve()
    r = st.solve_report
    assert isinstance(r, SolveReport)
    assert r.converged and r.reason > 0
    assert r.nl_its >= 1 and r.ksp_its >= 1
    assert r.fnorm == r.fnorm                 # finite (not NaN)
    assert len(r.history) >= 1
    if r.fnorm0:
        assert 0.0 < r.reduction <= 1.0 + 1e-12
    assert len(st.solve_history) == 1         # recorded in the trail
    # read-only
    with pytest.raises(AttributeError):
        st.solve_report = 1
    with pytest.raises(AttributeError):
        st.solve_history = 1


@pytest.mark.level_1
@pytest.mark.tier_a
def test_solve_report_scalar():
    """Report is populated for a scalar solver too (it's on SolverBaseClass)."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.25
    )
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 1.0
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc(1.0, "Top")
    poisson.solve()
    r = poisson.solve_report
    assert isinstance(r, SolveReport) and r.converged


@pytest.mark.level_1
@pytest.mark.tier_a
def test_estimate_difficulty_bounded():
    st, _ = _nonlinear_stokes()
    rep = st.estimate_difficulty(max_nl_its=2, warm=False)
    assert rep.bounded
    assert rep.reason == -5                    # DIVERGED_MAX_IT — the intended stop
    assert rep.nl_its == 2                     # exactly the cap
    assert not rep.converged                   # truncated, not converged
    assert rep is st.solve_report


@pytest.mark.level_1
@pytest.mark.tier_a
def test_resume_is_lossless_and_same_termination():
    # reference: one uninterrupted solve
    ref, vref = _nonlinear_stokes()
    ref.solve(zero_init_guess=True)
    u_ref = vref.data.copy()
    f_ref = ref.solve_report.fnorm
    assert ref.solve_report.nl_its >= 3        # genuinely nonlinear

    # chunked: bounded probe then resume to completion (large cap)
    st, v = _nonlinear_stokes()
    rep1 = st.estimate_difficulty(max_nl_its=2, warm=False)
    assert rep1.bounded and rep1.reason == -5
    rep2 = st.estimate_difficulty(max_nl_its=100, warm=True)
    assert rep2.converged

    # lossless: warm-continued solution matches the uninterrupted reference
    rel = np.linalg.norm(v.data - u_ref) / max(np.linalg.norm(u_ref), 1e-30)
    assert rel < 1e-6, f"resume not lossless (rel-err {rel:.2e})"

    # same termination point (A5): the chunked chain converges at tolerance*||F0||,
    # NOT the tighter tolerance*||F_restart||
    assert abs(rep2.fnorm - f_ref) / f_ref < 1e-3, (
        f"chunked termination {rep2.fnorm:.3e} != uninterrupted {f_ref:.3e}"
    )


@pytest.mark.level_1
@pytest.mark.tier_a
def test_solve_report_helpers():
    assert reason_string(2) == "CONVERGED_FNORM_RELATIVE"
    assert reason_string(-5) == "DIVERGED_MAX_IT"
    assert reason_string(999).startswith("UNKNOWN")
    assert contraction([50.0, 1e-3, 1e-6, 1e-9]) is not None
    assert contraction([50.0]) is None        # < 2 points -> undefined
