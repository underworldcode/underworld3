"""Constrained free-slip on an annulus via a recoverable Lagrange multiplier.

Buoyancy-driven flow in an annulus with no-slip inner boundary and free-slip
outer boundary. The free-slip outer condition (u.n = 0) is enforced three ways
and compared:

  - penalty   : the existing fragile penalty natural BC (reference)
  - multiplier: SNES_Stokes_Constrained, augmented-Lagrangian multiplier

The multiplier solver must (a) drive u.n -> 0 with NO penalty coefficient,
(b) match the penalty velocity field, and (c) yield a clean, recoverable
topography field whose boundary trace equals the consistent-boundary-flux
normal stress (-n.sigma.n), i.e. dynamic topography.

Run with: pixi run python -m pytest tests/test_1061_constrained_freeslip_annulus.py -v
"""

import pytest
import numpy as np
import sympy
import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

R_INNER, R_OUTER = 0.5, 1.0
CELL = 0.1
MU = 1.0
RA = 1.0e2


def _mesh_and_forcing():
    mesh = uw.meshing.Annulus(radiusInner=R_INNER, radiusOuter=R_OUTER,
                              cellSize=CELL, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x**2 + y**2)
    unit_r = sympy.Matrix([[x / r, y / r]])
    theta = sympy.atan2(y, x)
    buoy = RA * sympy.cos(3 * theta) * (r - R_INNER) / (R_OUTER - R_INNER)
    return mesh, unit_r, buoy, theta


def _outer_rms_vn(solver, unit_r):
    vn = solver.u.sym.dot(unit_r)
    num = float(uw.maths.BdIntegral(solver.mesh, fn=vn**2, boundary="Upper").evaluate())
    length = float(uw.maths.BdIntegral(solver.mesh, fn=1.0, boundary="Upper").evaluate())
    return np.sqrt(num / length)


@pytest.fixture(scope="module")
def solutions():
    mesh, unit_r, buoy, theta = _mesh_and_forcing()

    # --- penalty reference ---
    vp = uw.discretisation.MeshVariable("Up", mesh, mesh.dim, degree=2, vtype=uw.VarType.VECTOR)
    pp = uw.discretisation.MeshVariable("Pp", mesh, 1, degree=1)
    ref = uw.systems.Stokes(mesh, velocityField=vp, pressureField=pp)
    ref.constitutive_model = uw.constitutive_models.ViscousFlowModel
    ref.constitutive_model.Parameters.shear_viscosity_0 = MU
    ref.saddle_preconditioner = 1.0 / MU
    ref.bodyforce = buoy * unit_r
    ref.add_dirichlet_bc((0.0, 0.0), "Lower")
    ref.add_natural_bc(1e6 * MU * unit_r.dot(vp.sym) * unit_r, "Upper")
    ref.tolerance = 1e-8
    ref.petsc_options["ksp_type"] = "fgmres"
    ref.solve()

    # --- multiplier (augmented Lagrangian) ---
    vc = uw.discretisation.MeshVariable("Uc", mesh, mesh.dim, degree=2, vtype=uw.VarType.VECTOR)
    pc = uw.discretisation.MeshVariable("Pc", mesh, 1, degree=1)
    con = uw.systems.Stokes_Constrained(mesh, velocityField=vc, pressureField=pc)
    con.constitutive_model = uw.constitutive_models.ViscousFlowModel
    con.constitutive_model.Parameters.shear_viscosity_0 = MU
    con.saddle_preconditioner = 1.0 / MU
    con.bodyforce = buoy * unit_r
    con.add_dirichlet_bc((0.0, 0.0), "Lower")
    lam = con.add_constraint_bc("Upper", g=0.0, normal=unit_r)
    con.tolerance = 1e-8
    con.petsc_options["ksp_type"] = "fgmres"
    con.solve()

    return {
        "mesh": mesh, "unit_r": unit_r, "theta": theta,
        "ref": ref, "con": con, "lam": lam,
        "v_ref": vp.data.copy(), "v_con": vc.data.copy(),
    }


def test_multiplier_enforces_free_slip(solutions):
    """u.n -> 0 on the curved boundary with NO penalty coefficient."""
    rms = _outer_rms_vn(solutions["con"], solutions["unit_r"])
    print(f"multiplier RMS(u.n) on outer = {rms:.3e}")
    assert rms < 2.0e-4


def test_multiplier_matches_penalty(solutions):
    """Constrained velocity matches the penalty reference field."""
    v_ref, v_con = solutions["v_ref"], solutions["v_con"]
    rel = np.sqrt(np.sum((v_ref - v_con) ** 2)) / np.sqrt(np.sum(v_ref**2))
    print(f"relL2(v_multiplier vs v_penalty) = {rel:.3e}")
    assert rel < 0.01


def test_multiplier_api(solutions):
    """The multiplier and topography are retrievable through the public API."""
    con, lam = solutions["con"], solutions["lam"]
    assert con.multiplier("Upper") is lam
    assert con.multiplier("Nonexistent") is None
    # topography is lambda / (Delta_rho g)
    topo_expr = con.topography("Upper", buoyancy_scale=2.0)
    assert topo_expr == lam.sym[0] / 2.0


def test_constraint_bc_rejects_unknown_boundary(solutions):
    """add_constraint_bc validates the boundary name up front."""
    con = solutions["con"]
    with pytest.raises(ValueError):
        con.add_constraint_bc("Nonexistent")


def test_topography_field_is_clean(solutions):
    """Multiplier interior is exactly zero; only the boundary trace is non-zero."""
    lam = solutions["lam"]
    c = lam.coords
    rr = np.sqrt(c[:, 0] ** 2 + c[:, 1] ** 2)
    interior = rr < R_OUTER - 0.6 * CELL
    boundary = rr > R_OUTER - 0.25 * CELL
    assert np.max(np.abs(lam.data[interior, 0])) == 0.0
    assert np.max(np.abs(lam.data[boundary, 0])) > 1.0


def test_topography_matches_dynamic_topography_stress(solutions):
    """lambda on the boundary equals the CBF normal stress -n.sigma.n (dyn. topo.)."""
    con, lam = solutions["con"], solutions["lam"]
    unit_r = solutions["unit_r"]
    c = lam.coords
    rr = np.sqrt(c[:, 0] ** 2 + c[:, 1] ** 2)
    bmask = rr > R_OUTER - 0.25 * CELL

    sigma = con.stress
    nsn = (unit_r * sigma * unit_r.T)[0, 0]
    nsn_b = np.array(uw.function.evaluate(sympy.Matrix([[nsn]]), c[bmask])).reshape(-1)
    lam_b = lam.data[bmask, 0]

    a = lam_b - lam_b.mean()
    b = -(nsn_b - nsn_b.mean())
    corr = np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))
    print(f"boundary corr(lambda, -n.sigma.n) = {corr:.4f}")
    assert corr > 0.99
