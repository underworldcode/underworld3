"""Rotated strong free-slip BC (add_rotated_freeslip_bc) through the real solver API.

Increment 1 validation: on an axis-aligned box, rotated free-slip on all four walls
must reproduce the native essential free-slip solve, enforce zero wall-normal flow,
and (annulus) give per-node radial free-slip with machine-zero leakage.
"""
import numpy as np
import pytest
import sympy
import underworld3 as uw
from underworld3.function import analytic as A
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _wrap(dm, m0):
    return uw.discretisation.Mesh(
        dm.clone(), simplex=True,
        coordinate_system_type=m0.CoordinateSystem.coordinate_type,
        qdegree=3, boundaries=m0.boundaries)


def _solcx_essential(mesh, sol):
    v = uw.discretisation.MeshVariable("vE", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("pE", mesh, 1, degree=1, continuous=False)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    s.bodyforce = sol.fn_bodyforce
    s.penalty = 0.0
    s.tolerance = 1e-9
    s.add_essential_bc((sympy.oo, 0.0), "Top")
    s.add_essential_bc((sympy.oo, 0.0), "Bottom")
    s.add_essential_bc((0.0, sympy.oo), "Left")
    s.add_essential_bc((0.0, sympy.oo), "Right")
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    s.solve()
    return v


def test_rotated_freeslip_box_reproduces_essential():
    """Box: rotated free-slip on all 4 axis-aligned walls == native essential free-slip."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(16, 16), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
    sol = A.SolCx(mesh, eta_A=1.0, eta_B=1.0e3, x_c=0.5, n=1)
    vE = _solcx_essential(mesh, sol)

    v = uw.discretisation.MeshVariable("vR", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("pR", mesh, 1, degree=1, continuous=False)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    s.bodyforce = sol.fn_bodyforce
    s.penalty = 0.0
    s.tolerance = 1e-9
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    s.solve()

    rel = np.linalg.norm(v.data - vE.data) / np.linalg.norm(vE.data)
    assert rel < 1e-4, f"rotated free-slip differs from essential by {rel:.2e}"
    # exact analytic accuracy
    assert sol.velocity_error(v) < 1e-3


@pytest.mark.level_2
def test_rotated_freeslip_spherical_shell_3d():
    """3D spherical shell, free-slip inner+outer (the Zhong #248 configuration):
    all THREE rigid rotations must be recognised as nullspace modes and projected
    out, the self-contained Schur solve must converge in a bounded iteration
    count (the 1/mu-mass Schur preconditioner — ~30 its with selfp), and the
    radial leak must be machine-zero on both boundaries."""
    RI, RO = 0.55, 1.0
    mesh = uw.meshing.SphericalShell(radiusInner=RI, radiusOuter=RO,
                                     cellSize=0.25, qdegree=2)
    x, y, z = mesh.X
    r = sympy.sqrt(x**2 + y**2 + z**2)
    v = uw.discretisation.MeshVariable("Vs", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("Ps", mesh, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    # Y_20-like radial internal load, zero on the boundaries
    ylm = (3 * (z / r) ** 2 - 1) / 2
    g = (r - RI) * (RO - r) * 20.0
    s.bodyforce = ylm * g / r * sympy.Matrix([[x, y, z]])
    nhat = sympy.Matrix([[x / r, y / r, z / r]])
    s.add_rotated_freeslip_bc(0, "Lower", normal=nhat)
    s.add_rotated_freeslip_bc(0, "Upper", normal=nhat)
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    s.tolerance = 1e-7
    s.solve()

    info = s._rotated_freeslip_info
    assert info["ksp_reason"] > 0, f"rotated KSP diverged: {info['ksp_reason']}"
    assert info["ksp_its"] <= 25, (
        f"Schur iteration blow-out: {info['ksp_its']} outer its "
        "(1/mu-mass Schur preconditioning regressed?)")
    assert info["rotation_gauge_removed"], "3D rotation gauge not detected/removed"

    vc = v.coords
    rr = np.linalg.norm(vc, axis=1)
    rhat = vc / rr[:, None]
    vr = np.einsum("ij,ij->i", v.data, rhat)
    vmax = np.linalg.norm(v.data, axis=1).max() + 1e-30
    for lab, mask in [("inner", np.abs(rr - RI) < 1e-3), ("outer", np.abs(rr - RO) < 1e-3)]:
        leak = np.abs(vr[mask]).max() / vmax
        assert leak < 1e-10, f"{lab} radial leakage {leak:.2e} not machine-zero"
    # all three rigid-rotation gauges removed (nodal check, serial test)
    for k, t in enumerate([
            np.column_stack([np.zeros(len(vc)), -vc[:, 2], vc[:, 1]]),
            np.column_stack([vc[:, 2], np.zeros(len(vc)), -vc[:, 0]]),
            np.column_stack([-vc[:, 1], vc[:, 0], np.zeros(len(vc))])]):
        rotfrac = abs(np.sum(v.data * t)) / (np.linalg.norm(t) * np.linalg.norm(v.data) + 1e-30)
        assert rotfrac < 1e-8, f"rotation mode {k} gauge {rotfrac:.2e} not removed"


def test_rotated_freeslip_annulus_zero_leakage():
    """Annulus: per-node radial free-slip on both arcs → machine-zero v_r leakage,
    with the analytic radial normal."""
    RI, RO = 0.5, 1.0
    mesh = uw.meshing.Annulus(radiusInner=RI, radiusOuter=RO, cellSize=0.1, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x**2 + y**2)
    th = sympy.atan2(y, x)
    v = uw.discretisation.MeshVariable("Va", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("Pa", mesh, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    s.bodyforce = sympy.Matrix([[x / r * sympy.cos(4 * th) * (r - RI) * (RO - r) * 40.0,
                                 y / r * sympy.cos(4 * th) * (r - RI) * (RO - r) * 40.0]])
    nhat = sympy.Matrix([[x / r, y / r]])
    s.add_rotated_freeslip_bc(0, "Lower", normal=nhat)
    s.add_rotated_freeslip_bc(0, "Upper", normal=nhat)
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    s.solve()

    vc = v.coords
    rr = np.hypot(vc[:, 0], vc[:, 1])
    rhat = vc / rr[:, None]
    vr = np.einsum("ij,ij->i", v.data, rhat)
    vmax = np.linalg.norm(v.data, axis=1).max() + 1e-30
    for lab, mask in [("inner", np.abs(rr - RI) < 1e-4), ("outer", np.abs(rr - RO) < 1e-4)]:
        leak = np.abs(vr[mask]).max() / vmax
        assert leak < 1e-10, f"{lab} radial leakage {leak:.2e} not machine-zero"
    # rigid-rotation gauge removed
    t = np.column_stack([-vc[:, 1], vc[:, 0]])
    rotfrac = abs(np.sum(v.data * t) / np.sum(t * t)) * np.sqrt(np.sum(t * t)) / (np.linalg.norm(v.data) + 1e-30)
    assert rotfrac < 1e-8, f"rotation gauge {rotfrac:.2e} not removed"


def test_rotated_freeslip_geometric_fmg_velocity_block():
    """The rotated free-slip velocity block is driven by GEOMETRIC FMG on a custom
    prolongation (set_custom_fmg) — no direct solve — and still reproduces the
    essential free-slip solution, with the wall-normal velocity exact."""
    # a small 2-level nested hierarchy (one refinement) keeps the cost down while
    # still driving the velocity block by geometric FMG on the custom prolongation.
    m0 = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.2, regular=True, qdegree=3)
    coarse = [_wrap(m0.dm, m0)]
    fine = _wrap(m0.dm.refine(), m0)
    sol = A.SolCx(fine, eta_A=1.0, eta_B=1.0e3, x_c=0.5, n=1)

    vE = _solcx_essential(fine, sol)

    v = uw.discretisation.MeshVariable("vF", fine, fine.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("pF", fine, 1, degree=1, continuous=False)
    s = uw.systems.Stokes(fine, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    s.saddle_preconditioner = 1.0 / sol.fn_viscosity
    s.bodyforce = sol.fn_bodyforce
    s.tolerance = 1e-9
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    custom_mg.set_custom_fmg(s, coarse, builder="barycentric", field_id=0)
    s.solve()

    # geometric MG on the velocity block converged, and matches essential
    assert s._rotated_freeslip_info["ksp_reason"] > 0
    rel = np.linalg.norm(v.data - vE.data) / np.linalg.norm(vE.data)
    assert rel < 5e-3, f"FMG rotated free-slip differs from essential by {rel:.2e}"


def test_rotated_freeslip_boundary_normal_traction_solcx():
    """sigma_nn recovered by boundary_normal_traction on the top boundary reproduces
    the exact SolCx analytic sigma_yy (mean-removed) to a few percent. Exercises the
    Cartesian-reaction + n_hat projection (corner-correct) + consistent P2 line mass.

    Run at a modest resolution: SolCx breakage is catastrophic (corr collapses toward
    0, relL2 blows up to O(1)), not gradual, so a coarse mesh still catches it — the
    thresholds carry margin over the res-24 values (corr 0.998, relL2 0.056)."""
    res = 24
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(res, res), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
    sol = A.SolCx(mesh, eta_A=1.0, eta_B=1.0e3, x_c=0.5, n=1)
    v = uw.discretisation.MeshVariable("vT", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("pT", mesh, 1, degree=1, continuous=False)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    s.bodyforce = sol.fn_bodyforce
    s.penalty = 0.0
    s.tolerance = 1e-9
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    s.solve()

    xs, sig = s.boundary_normal_traction("Top")
    top = np.asarray(xs)
    syy = np.asarray(sol.evaluate_stress(top))[:, 1]
    syy = syy - syy.mean()
    sig = np.asarray(sig)
    corr = np.dot(sig, syy) / (np.linalg.norm(sig) * np.linalg.norm(syy))
    sig = sig if corr >= 0 else -sig                      # sigma = -R sign convention
    relL2 = np.linalg.norm(sig - syy) / np.linalg.norm(syy)
    assert abs(corr) > 0.97, f"sigma_nn shape corr {corr:.3f} too low"
    assert relL2 < 0.10, f"sigma_nn relL2 vs analytic sigma_yy {relL2:.3f} too large"


def test_rotated_freeslip_sigma_nn_lumped_no_overshoot():
    """The default (lumped) sigma_nn de-smear is MONOTONE at the SolCx viscosity jump:
    its total variation matches the analytic (no Gibbs overshoot), whereas the
    consistent-mass de-smear adds spurious variation. This is the property that matters
    for driving a free surface (an overshoot injects a spurious surface-velocity pulse)."""
    res = 24
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(res, res), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
    sol = A.SolCx(mesh, eta_A=1.0, eta_B=1.0e3, x_c=0.5, n=1)
    v = uw.discretisation.MeshVariable("vL", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("pL", mesh, 1, degree=1, continuous=False)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    s.bodyforce = sol.fn_bodyforce
    s.penalty = 0.0
    s.tolerance = 1e-9
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    s.solve()

    def total_variation(xs, sig):
        o = np.argsort(np.asarray(xs)[:, 0])
        return float(np.sum(np.abs(np.diff(np.asarray(sig)[o]))))

    xs, sig_l = s.boundary_normal_traction("Top", mass="lumped")     # default
    _, sig_c = s.boundary_normal_traction("Top", mass="consistent")
    syy = np.asarray(sol.evaluate_stress(np.asarray(xs)))[:, 1]
    tv_ref = total_variation(xs, syy - syy.mean())
    tv_l = total_variation(xs, sig_l)
    tv_c = total_variation(xs, sig_c)
    # lumped adds essentially no spurious variation over the analytic; consistent does
    assert tv_l < 1.1 * tv_ref, f"lumped TV {tv_l:.3f} exceeds analytic {tv_ref:.3f}"
    assert tv_l < tv_c, f"lumped TV {tv_l:.3f} not smoother than consistent {tv_c:.3f}"


def _powerlaw_stokes(mesh, prefix, amp=2.0, nexp=3.0, cj=None):
    """A genuinely NONLINEAR Stokes: power-law viscosity eta = eps_II^(1/n - 1)
    (smooth, so Newton/Picard iterates robustly), driven by a horizontally-varying
    vertical body force so there is real shear. ``cj`` sets ``consistent_jacobian``
    (None=default frozen/Picard, True=consistent Newton, "continuation"=staged)."""
    x, y = mesh.X
    v = uw.discretisation.MeshVariable(prefix + "v", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable(prefix + "p", mesh, 1, degree=1, continuous=False)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    g = sympy.Matrix([[v.sym[0].diff(x), v.sym[0].diff(y)],
                      [v.sym[1].diff(x), v.sym[1].diff(y)]])
    e = 0.5 * (g + g.T)
    eII = sympy.sqrt(0.5 * (e[0, 0] ** 2 + e[1, 1] ** 2) + e[0, 1] ** 2 + 1.0e-12)
    s.constitutive_model.Parameters.shear_viscosity_0 = eII ** (1.0 / nexp - 1.0)
    s.bodyforce = sympy.Matrix([[0.0, -amp * sympy.cos(sympy.pi * x)]])
    s.penalty = 0.0
    s.tolerance = 1e-7
    s.petsc_use_pressure_nullspace = True
    if cj is not None:
        s.consistent_jacobian = cj
    return s, v, p


def _powerlaw_essential(mesh, prefix, cj=None):
    s, v, p = _powerlaw_stokes(mesh, prefix, cj=cj)
    s.add_essential_bc((sympy.oo, 0.0), "Top")
    s.add_essential_bc((sympy.oo, 0.0), "Bottom")
    s.add_essential_bc((0.0, sympy.oo), "Left")
    s.add_essential_bc((0.0, sympy.oo), "Right")
    s.solve()
    return v, s.snes.getIterationNumber()


@pytest.fixture(scope="module")
def plaw_box_ref():
    """A small power-law box + its native ESSENTIAL nonlinear free-slip solution,
    solved ONCE (Newton tangent) and shared across the rotated-vs-essential box
    tests. The converged solution is tangent-independent, so this one reference
    serves the Picard / Newton / continuation tests. Returns (mesh, vE_data,
    essential_newton_iters)."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
    vE, its = _powerlaw_essential(mesh, "refE", cj=True)
    return mesh, np.copy(vE.data), its


def test_rotated_freeslip_nonlinear_matches_essential(plaw_box_ref):
    """Default (frozen/Picard) tangent through the rotated path: genuinely iterates
    and converges to the native essential nonlinear free-slip answer (both impose
    v_n=0 — identical discrete problem), with machine-zero wall-normal flow on every
    wall. Exercises the rotated constraint INSIDE the nonlinear iteration."""
    mesh, vE, _ = plaw_box_ref
    s, vR, pR = _powerlaw_stokes(mesh, "nlP")           # default (Picard) tangent
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.solve()
    info = s._rotated_freeslip_info
    assert info["nonlinear_iterations"] > 1, "rotated solve did not genuinely iterate"
    # the reported count is the number of Newton increments solved — one linear
    # solve per increment (guards the loop-index off-by-one in the result dict)
    assert info["nonlinear_iterations"] == len(info["ksp_its"])
    assert np.linalg.norm(vR.data - vE) / np.linalg.norm(vE) < 1e-3
    vc = vR.coords
    for lab, msk, comp in [("Top", np.abs(vc[:, 1] - 1) < 1e-6, 1),
                           ("Bottom", np.abs(vc[:, 1]) < 1e-6, 1),
                           ("Left", np.abs(vc[:, 0]) < 1e-6, 0),
                           ("Right", np.abs(vc[:, 0] - 1) < 1e-6, 0)]:
        leak = np.abs(vR.data[msk, comp]).max() if msk.any() else 0.0
        assert leak < 1e-10, f"{lab} wall-normal velocity {leak:.2e} not machine-zero"


def test_rotated_freeslip_newton_tangent(plaw_box_ref):
    """consistent_jacobian=True is genuine Newton through the rotated path: it
    converges in about the same small iteration count as the native essential Newton
    solve (NOT the ~4-5x larger Picard count) and to the same answer at (near) machine
    precision — the rotated constraint does not degrade the consistent tangent."""
    mesh, vE, ess_its = plaw_box_ref
    s, vR, pR = _powerlaw_stokes(mesh, "ntR", cj=True)
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.solve()
    its = s._rotated_freeslip_info["nonlinear_iterations"]
    assert its <= 2 * ess_its + 2, (
        f"rotated Newton took {its} iters vs essential Newton {ess_its} "
        f"— tangent likely not the consistent one")
    assert its < 20, f"rotated Newton not converging at Newton rate ({its} iters)"
    assert np.linalg.norm(vR.data - vE) / np.linalg.norm(vE) < 1e-6


def test_rotated_freeslip_continuation_tangent(plaw_box_ref):
    """consistent_jacobian='continuation' works through the rotated path: a staged
    Picard→Newton solve (α=0 to the loose newton_switch_rtol, then α=1) that switches
    tangents and converges to the essential Newton answer. picard=N extends the α=0
    phase (so the wrapper's picard forwarding and the staging are both exercised)."""
    mesh, vE, _ = plaw_box_ref
    s, vR, pR = _powerlaw_stokes(mesh, "ctR", cj="continuation")
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.solve()
    info = s._rotated_freeslip_info
    assert info["continuation_switched"], "continuation never switched Picard→Newton"
    assert 1 < info["nonlinear_iterations"] < 36, "continuation not staging as expected"
    assert np.linalg.norm(vR.data - vE) / np.linalg.norm(vE) < 1e-6

    # picard=N holds the α=0 (Picard) phase for >= N iterations before switching
    s2, _, _ = _powerlaw_stokes(mesh, "ctR2", cj="continuation")
    for wall in ("Top", "Bottom", "Left", "Right"):
        s2.add_rotated_freeslip_bc(0, wall)
    s2.solve(picard=25)
    assert s2._rotated_freeslip_info["nonlinear_iterations"] >= 25, (
        "picard did not extend the continuation Picard phase")


def test_rotated_freeslip_picard_newton_unsupported_raises(plaw_box_ref):
    """picard>0 with the pure consistent-Newton tangent has no frozen warmup tangent
    to form, so the rotated path raises a clear NotImplementedError pointing to
    'continuation' rather than silently ignoring it. Also confirms the SNES_Stokes
    wrapper forwards picard to the plain-Stokes solve at all."""
    mesh, _, _ = plaw_box_ref
    s, v, p = _powerlaw_stokes(mesh, "prN", cj=True)
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    with pytest.raises(NotImplementedError, match="continuation"):
        s.solve(picard=3)


def test_rotated_freeslip_nonlinear_warm_start(plaw_box_ref):
    """Warm-start (zero_init_guess=False) through the nonlinear rotated path: a 2-step
    'time loop' re-solving from the previous converged state stays correct and
    converges in few iterations (the step-norm exit avoids chasing machine noise)."""
    mesh, vE, _ = plaw_box_ref
    s, vR, pR = _powerlaw_stokes(mesh, "wsR", cj=True)     # Newton (fast)
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.solve()                              # cold
    s.solve(zero_init_guess=False)         # warm (step 2)
    assert np.linalg.norm(vR.data - vE) / np.linalg.norm(vE) < 1e-6
    # warm-start from an already-converged state converges in few iterations
    assert s._rotated_freeslip_info["nonlinear_iterations"] <= 3


def test_rotated_freeslip_nonlinear_geometric_fmg():
    """The NONLINEAR rotated free-slip velocity block is driven by geometric FMG on
    the custom prolongation (set_custom_fmg) — the rotated prolongation is built once
    and reused across Newton iterations — and still converges to the essential
    nonlinear free-slip solution."""
    # a small 2-level nested hierarchy (one refinement) + the Newton tangent keep the
    # cost down while still exercising the rotated custom-FMG prolongation reuse.
    m0 = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.4, regular=True, qdegree=3)
    coarse = [_wrap(m0.dm, m0)]
    fine = _wrap(m0.dm.refine(), m0)
    vE, _ = _powerlaw_essential(fine, "nlfE", cj=True)

    s, vR, pR = _powerlaw_stokes(fine, "nlfR", cj=True)
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    custom_mg.set_custom_fmg(s, coarse, builder="barycentric", field_id=0)
    s.solve()

    info = s._rotated_freeslip_info
    assert info["nonlinear_iterations"] > 1
    assert info["ksp_reason"] > 0                # geometric MG increment converged
    rel = np.linalg.norm(vR.data - vE.data) / np.linalg.norm(vE.data)
    assert rel < 5e-3, f"nonlinear FMG rotated free-slip differs from essential by {rel:.2e}"


def test_rotated_freeslip_nonlinear_annulus_zero_leakage():
    """Genuinely-rotated frame under nonlinear iteration: a power-law annulus with
    per-node radial free-slip on both arcs genuinely iterates, gives machine-zero
    radial leakage, and the rigid-rotation gauge is removed."""
    RI, RO = 0.5, 1.0
    mesh = uw.meshing.Annulus(radiusInner=RI, radiusOuter=RO, cellSize=0.3, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x**2 + y**2)
    th = sympy.atan2(y, x)
    v = uw.discretisation.MeshVariable("Vnla", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("Pnla", mesh, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    g = sympy.Matrix([[v.sym[0].diff(x), v.sym[0].diff(y)],
                      [v.sym[1].diff(x), v.sym[1].diff(y)]])
    e = 0.5 * (g + g.T)
    eII = sympy.sqrt(0.5 * (e[0, 0] ** 2 + e[1, 1] ** 2) + e[0, 1] ** 2 + 1.0e-12)
    s.constitutive_model.Parameters.shear_viscosity_0 = eII ** (1.0 / 3.0 - 1.0)
    s.bodyforce = sympy.Matrix([[x / r * sympy.cos(4 * th) * (r - RI) * (RO - r) * 40.0,
                                 y / r * sympy.cos(4 * th) * (r - RI) * (RO - r) * 40.0]])
    nhat = sympy.Matrix([[x / r, y / r]])
    s.consistent_jacobian = True                 # Newton tangent (few iterations)
    s.add_rotated_freeslip_bc(0, "Lower", normal=nhat)
    s.add_rotated_freeslip_bc(0, "Upper", normal=nhat)
    s.petsc_use_pressure_nullspace = True
    s.tolerance = 1e-7
    s.solve()

    info = s._rotated_freeslip_info
    assert info["nonlinear_iterations"] > 1, "annulus rotated solve did not genuinely iterate"
    assert info["rotation_gauge_removed"]
    vc = v.coords
    rr = np.hypot(vc[:, 0], vc[:, 1])
    rhat = vc / rr[:, None]
    vr = np.einsum("ij,ij->i", v.data, rhat)
    vmax = np.linalg.norm(v.data, axis=1).max() + 1e-30
    for lab, mask in [("inner", np.abs(rr - RI) < 1e-4), ("outer", np.abs(rr - RO) < 1e-4)]:
        leak = np.abs(vr[mask]).max() / vmax
        assert leak < 1e-10, f"{lab} radial leakage {leak:.2e} not machine-zero"


def test_rotated_freeslip_dynamic_topography_field():
    """dynamic_topography writes h = -(sigma_nn - mean)/(rho g) onto a scalar surface
    MeshVariable (the free-surface hand-off): the field at the top vertices reproduces
    the analytic SolCx topography, and the field is usable symbolically (BdIntegral)."""
    res = 24
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(res, res), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
    sol = A.SolCx(mesh, eta_A=1.0, eta_B=1.0e3, x_c=0.5, n=1)
    v = uw.discretisation.MeshVariable("vD", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("pD", mesh, 1, degree=1, continuous=False)
    hf = uw.discretisation.MeshVariable("hD", mesh, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    s.bodyforce = sol.fn_bodyforce
    s.penalty = 0.0
    s.tolerance = 1e-9
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    s.solve()

    ret = s.dynamic_topography("Top", hf, buoyancy_scale=1.0)
    assert ret is hf
    # field at the top vertices reproduces the analytic topography (mean-removed)
    hc = hf.coords
    top = np.abs(hc[:, 1] - 1.0) < 1e-6
    tt = np.asarray(sol.topography_top(hc[top])); tt = tt - tt.mean()
    hv = hf.data[top, 0]
    corr = np.dot(hv, tt) / (np.linalg.norm(hv) * np.linalg.norm(tt))
    hv = hv if corr >= 0 else -hv
    relL2 = np.linalg.norm(hv - tt) / np.linalg.norm(tt)
    assert abs(corr) > 0.97, f"topography field corr {corr:.3f} too low"
    assert relL2 < 0.12, f"topography field relL2 {relL2:.3f} too large"
    # symbolically usable (the free-surface integrator reads it via BdIntegral)
    bdl2 = float(np.sqrt(uw.maths.BdIntegral(
        mesh=mesh, fn=hf.sym[0] ** 2, boundary="Top").evaluate()))
    assert bdl2 > 0.0


# --- Prescribed non-zero wall-normal datum (u.n = ũ_n) ---------------------------
# The rotated constraint imposes u.n = datum strongly: datum=0 is pure free-slip (the
# held lid); a non-zero datum is the "consistent" material-surface velocity. Both share
# the same rotated matrix and differ only in the constraint RHS, so datum=0 must be
# BIT-IDENTICAL to plain free-slip. (Set via solver._rotated_freeslip_datum until the
# add_rotated_freeslip_bc datum argument lands — underworldcode/underworld3 tracking.)

def _annulus_datum_solve(mode, tag):
    RI, RO = 0.5, 1.0
    mesh = uw.meshing.Annulus(radiusInner=RI, radiusOuter=RO, cellSize=0.1, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x**2 + y**2)
    nhat = sympy.Matrix([[x / r, y / r]])
    v = uw.discretisation.MeshVariable("Vd" + tag, mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("Pd" + tag, mesh, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    blob = sympy.exp(-(((x - 0.75) ** 2 + y ** 2) / 0.05))
    s.bodyforce = sympy.Matrix([[50.0 * blob * x / r, 50.0 * blob * y / r]])
    s.add_essential_bc((0.0, 0.0), "Lower")          # no-slip inner (pins rotation gauge)
    s.add_rotated_freeslip_bc(0, "Upper", normal=nhat)
    if mode == "zero":
        s._rotated_freeslip_datum = {"Upper": 0.0}
    elif mode == "cos":
        s._rotated_freeslip_datum = {"Upper": x / r}   # u.n = cos(theta), mean-zero => ∮u.n=0
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    s.tolerance = 1e-9
    s.solve()
    return mesh, v


def test_rotated_freeslip_datum_zero_matches_freeslip():
    """datum={'Upper': 0} (the prescribed-datum path with a zero datum) reproduces plain
    rotated free-slip to iterative round-off — the held / free-slip case is untouched.
    (A zero datum takes the same free-slip RHS path; the residual is the iterative
    solver's run-to-run round-off, ~1e-15, not a code-path difference.)"""
    _, v0 = _annulus_datum_solve("plain", "0")
    _, vz = _annulus_datum_solve("zero", "z")
    d = np.abs(vz.data - v0.data).max()
    assert d < 1e-10, f"zero datum perturbs the free-slip solution by {d:.2e} (> round-off)"


def test_rotated_freeslip_prescribed_normal_datum():
    """A prescribed non-zero wall-normal datum u.n = cos(theta) is imposed to the SAME
    (machine) precision as u.n = 0 at the constrained surface nodes."""
    RO = 1.0
    mesh, v = _annulus_datum_solve("cos", "c")
    vc = v.coords
    rr = np.hypot(vc[:, 0], vc[:, 1])
    outer = np.abs(rr - RO) < 2.0e-2                    # outer velocity nodes (incl. edge mids)
    rhat = vc[outer] / rr[outer, None]
    vn = np.einsum("ij,ij->i", v.data[outer], rhat)
    target = vc[outer, 0] / rr[outer]                   # cos(theta) at the same node coords
    err = np.abs(vn - target).max()
    assert err < 1e-8, f"prescribed u.n=cos(theta) not imposed: max nodal error {err:.2e}"
    assert vn.max() > 0.9 and vn.min() < -0.9, "prescribed normal velocity magnitude wrong"
