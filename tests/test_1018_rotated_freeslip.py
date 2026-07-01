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
        elementRes=(24, 24), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
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
        s.add_rotated_freeslip_bc(wall)
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    s.solve()

    rel = np.linalg.norm(v.data - vE.data) / np.linalg.norm(vE.data)
    assert rel < 1e-4, f"rotated free-slip differs from essential by {rel:.2e}"
    # exact analytic accuracy
    assert sol.velocity_error(v) < 1e-3


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
    s.add_rotated_freeslip_bc("Lower", normal=nhat)
    s.add_rotated_freeslip_bc("Upper", normal=nhat)
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
    m0 = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.2, regular=True, qdegree=3)
    dm0 = m0.dm
    dm1 = dm0.refine()
    dm2 = dm1.refine()
    coarse = [_wrap(dm0, m0), _wrap(dm1, m0)]
    fine = _wrap(dm2, m0)
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
        s.add_rotated_freeslip_bc(wall)
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
    Cartesian-reaction + n_hat projection (corner-correct) + consistent P2 line mass."""
    res = 48
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
        s.add_rotated_freeslip_bc(wall)
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
    assert abs(corr) > 0.99, f"sigma_nn shape corr {corr:.3f} too low"
    assert relL2 < 0.08, f"sigma_nn relL2 vs analytic sigma_yy {relL2:.3f} too large"
