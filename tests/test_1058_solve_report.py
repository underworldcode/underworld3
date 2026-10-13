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
    # 2 is CONVERGED_FNORM_ABS in PETSc. This line previously asserted
    # CONVERGED_FNORM_RELATIVE — it pinned the table to itself rather than to the enum,
    # which is how the off-by-one survived. See test_snes_reason_table_matches_petsc.
    assert reason_string(2) == "CONVERGED_FNORM_ABS"
    assert reason_string(-5) == "DIVERGED_MAX_IT"
    assert reason_string(999).startswith("UNKNOWN")
    assert contraction([50.0, 1e-3, 1e-6, 1e-9]) is not None
    assert contraction([50.0]) is None        # < 2 points -> undefined


# --------------------------------------------------------------------------- #
#  Rotated free-slip paths (solve outside self.snes -> report from the rotated
#  result dicts; regression for the review findings on PR #377)
# --------------------------------------------------------------------------- #

def _rotated_stokes(nonlinear=False):
    """Box Stokes with rotated strong free-slip on all four walls."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.25
    )
    v = uw.discretisation.MeshVariable("Ur", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("Pr", mesh, 1, degree=1, continuous=True)
    st = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    st.constitutive_model = uw.constitutive_models.ViscousFlowModel
    if nonlinear:
        edot = st.Unknowns.Einv2 + 1.0e-8
        st.constitutive_model.Parameters.shear_viscosity_0 = (1.0e-2 + edot) ** (-0.4)
    else:
        st.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    st.tolerance = 1.0e-6
    x, y = mesh.X
    st.bodyforce = sympy.Matrix([0, sympy.sin(np.pi * x) * sympy.sin(np.pi * y)])
    for w in ("Bottom", "Top", "Left", "Right"):
        st.add_rotated_freeslip_bc(0, w)
    st.petsc_use_pressure_nullspace = True
    return st, v


@pytest.mark.level_1
@pytest.mark.tier_a
def test_rotated_linear_solve_report():
    """The linear rotated path leaves a populated report: KSP-namespace reason,
    a real residual norm (not NaN), and one entry in the history trail."""
    st, _ = _rotated_stokes(nonlinear=False)
    st.solve()
    r = st.solve_report
    assert isinstance(r, SolveReport)
    assert r.converged and r.ksp_its >= 1
    assert r.nl_its == 1                       # one outer solve on the linear path
    assert r.fnorm == r.fnorm                  # finite: rnorm is in the result dict
    assert r.reason_str.startswith("KSP_")     # KSP namespace, not the SNES table
    assert len(st.solve_history) == 1


@pytest.mark.level_1
@pytest.mark.tier_a
def test_rotated_nonlinear_solve_report():
    """The manual nonlinear rotated loop reports its OUTER verdict and effort:
    nl_its from nonlinear_iterations, ksp_its summed over the per-Newton list."""
    st, _ = _rotated_stokes(nonlinear=True)
    st.solve()
    info = st._rotated_freeslip_info
    r = st.solve_report
    assert isinstance(r, SolveReport), "nonlinear rotated solve left no report"
    assert r.converged == info["converged"]
    assert r.nl_its == info["nonlinear_iterations"] and r.nl_its >= 2
    assert r.ksp_its == sum(info["ksp_its"])
    assert r.fnorm == r.fnorm and r.fnorm0 is not None and r.fnorm < r.fnorm0
    assert r.reason_str.startswith("KSP_")
    assert len(st.solve_history) == 1


@pytest.mark.level_1
@pytest.mark.tier_a
def test_estimate_difficulty_rejects_rotated_and_bad_kwargs():
    st, _ = _rotated_stokes(nonlinear=False)
    with pytest.raises(NotImplementedError):
        st.estimate_difficulty(max_nl_its=2)
    st2, _ = _nonlinear_stokes()
    with pytest.raises(TypeError):
        st2.estimate_difficulty(max_nl_its=2, zero_init_guess=True)
    with pytest.raises(TypeError):
        st2.estimate_difficulty(max_nl_its=2, divergence_retries=1)


@pytest.mark.level_1
@pytest.mark.tier_a
def test_warm_probe_on_converged_state_does_not_poison_anchor():
    """A warm estimate_difficulty on an already-converged state must not arm an
    anchor at tolerance*||F_converged|| (an unreachable target that would make the
    next probe grind and mislabel its exit)."""
    st, _ = _nonlinear_stokes()
    st.solve(zero_init_guess=True)
    rep = st.estimate_difficulty(max_nl_its=10, warm=True)
    assert rep.converged
    assert st._resume_abs_target is None       # converged probe leaves no armed anchor
    rep2 = st.estimate_difficulty(max_nl_its=10, warm=True)
    assert rep2.converged and rep2.nl_its <= 1  # no grinding on a converged state
    assert st._resume_abs_target is None


@pytest.mark.level_1
@pytest.mark.tier_a
def test_ksp_reason_table_matches_petsc():
    """The KSP reason table is hand-written (this module must import without petsc4py
    at build-doc time) — pin it to the real enum so a PETSc renumbering is caught."""
    from petsc4py import PETSc
    from underworld3.systems.solve_report import KSP_REASON_STRINGS, ksp_reason_string

    enum_names = {}
    for name, value in vars(PETSc.KSP.ConvergedReason).items():
        if isinstance(value, int) and not name.startswith("_"):
            enum_names.setdefault(value, name)
    for code, label in KSP_REASON_STRINGS.items():
        if code == 0:
            continue                           # petsc4py spells 0 CONVERGED_ITERATING
        assert label == f"KSP_{enum_names[code]}", (code, label, enum_names[code])
    assert ksp_reason_string(-3) == "KSP_DIVERGED_MAX_IT"
    assert ksp_reason_string(999).startswith("KSP_UNKNOWN")


@pytest.mark.level_1
@pytest.mark.tier_a
def test_snes_reason_table_matches_petsc():
    """The SNES table is hand-written for the same reason as the KSP one, and unlike the
    KSP one it was never pinned to the enum — so it drifted. Every positive code was
    shifted by one: a solve that stopped on the STEP norm (the weakest criterion, and
    what a stalled viscoplastic solve reports) was labelled CONVERGED_ITS, while a
    genuine residual convergence was labelled CONVERGED_SNORM_RELATIVE. Reading a
    difficulty report is how continuation drivers decide whether a station is reachable,
    so the labels have to be the real ones."""
    from petsc4py import PETSc
    from underworld3.systems.solve_report import REASON_STRINGS, reason_string

    # A code can carry more than one petsc4py spelling (0 is both CONVERGED_ITERATING
    # and ITERATING), so collect every alias: picking one via setdefault would pin the
    # test to the order vars() happens to yield, which is not a petsc4py guarantee.
    enum_names = {}
    for name, value in vars(PETSc.SNES.ConvergedReason).items():
        if isinstance(value, int) and not name.startswith("_"):
            enum_names.setdefault(value, set()).add(name)

    for code, label in REASON_STRINGS.items():
        assert code in enum_names, f"{code} ({label}) is not a PETSc SNES reason at all"
        assert label in enum_names[code], (code, label, sorted(enum_names[code]))

    # Every reason PETSc can return must be nameable — an UNKNOWN_n in a report is a
    # gap in the table, and the ones that went missing were real diverged states.
    for code in enum_names:
        assert code in REASON_STRINGS, (
            f"PETSc reason {code} ({sorted(enum_names[code])}) unmapped")

    assert reason_string(4) == "CONVERGED_SNORM_RELATIVE"
    assert reason_string(5) == "CONVERGED_ITS"
    assert reason_string(999).startswith("UNKNOWN")

    # The solver carries a SECOND copy of this table (code -> (NAME, explanation)) so
    # its diagnostics can add a one-line gloss. It had the identical off-by-one, and
    # fixing only one copy would leave the two disagreeing — so pin both to the enum
    # and to each other.
    from underworld3.systems import Stokes

    solver_table = Stokes._convergence_reasons
    for code, (label, _explanation) in solver_table.items():
        assert code in enum_names, f"{code} ({label}) is not a PETSc SNES reason at all"
        assert label in enum_names[code], (code, label, sorted(enum_names[code]))
    for code in enum_names:
        assert code in solver_table, (
            f"PETSc reason {code} ({sorted(enum_names[code])}) unmapped in the solver")
    assert {c: n for c, (n, _) in solver_table.items()} == REASON_STRINGS
