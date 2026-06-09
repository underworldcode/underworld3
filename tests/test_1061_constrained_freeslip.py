"""Constrained free-slip: a Lagrange multiplier INSIDE the saddle point.

SNES_Stokes_Constrained enforces u.n = g on a boundary via a multiplier h
that lives in the coupled system (grouped u | [p,h] Schur split), in ONE solve.
The converged h is the boundary normal traction = dynamic topography.

Two regimes:
  (A) box, open top  -> no pressure nullspace -> direct lu; axis-aligned walls
      compared against a Dirichlet free-slip reference (exact).
  (B) annulus, enclosed, curved boundary -> pressure nullspace + grouped Schur
      via DEFAULT solver options (no per-script PETSc options); compared against
      the penalty reference, with topography recovery h ~ -n.sigma.n.

Run: pixi run python -m pytest tests/test_1061_constrained_freeslip.py -v
"""

import pytest
import numpy as np
import sympy
import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

MU = 1.0


def _direct(s):
    s.petsc_options["snes_type"] = "ksponly"
    s.petsc_options["pc_type"] = "lu"
    s.petsc_options["pc_factor_mat_solver_type"] = "mumps"
    s.petsc_options["pc_use_amat"] = None
    s.petsc_options["ksp_type"] = "preonly"


# --------------------------------------------------------------------------- #
# (A) open-top box: nullspace-free, direct lu, axis-aligned multiplier walls
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def box():
    def forcing(m):
        xx, yy = m.X
        return sympy.Matrix([0.0, sympy.sin(sympy.pi * xx) * sympy.cos(sympy.pi * yy)])

    m0 = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.15, qdegree=3)
    ref = uw.systems.Stokes(m0)
    ref.constitutive_model = uw.constitutive_models.ViscousFlowModel
    ref.constitutive_model.Parameters.shear_viscosity_0 = MU
    ref.bodyforce = forcing(m0)
    ref.add_dirichlet_bc((0.0, 0.0), "Bottom")
    ref.add_dirichlet_bc((0.0, None), "Left")    # Dirichlet free-slip (exact)
    ref.add_dirichlet_bc((0.0, None), "Right")
    _direct(ref)
    ref.solve()

    mb = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.15, qdegree=3)
    blk = uw.systems.Stokes_Constrained(mb)
    blk.constitutive_model = uw.constitutive_models.ViscousFlowModel
    blk.constitutive_model.Parameters.shear_viscosity_0 = MU
    blk.bodyforce = forcing(mb)
    blk.add_dirichlet_bc((0.0, 0.0), "Bottom")
    hL = blk.add_constraint_bc("Left", g=0.0, normal=sympy.Matrix([[-1.0, 0.0]]))
    hR = blk.add_constraint_bc("Right", g=0.0, normal=sympy.Matrix([[1.0, 0.0]]))
    _direct(blk)
    blk.solve()
    return dict(ref=ref, blk=blk, hL=hL, hR=hR)


def test_box_constraint_enforced(box):
    """u.n -> 0 on the multiplier walls (parameter-free, one solve)."""
    blk = box["blk"]
    c = blk.u.coords
    wall = (c[:, 0] < 1e-6) | (c[:, 0] > 1 - 1e-6)
    rms = np.sqrt(np.mean(blk.u.data[wall, 0] ** 2))
    assert rms < 1e-7


def test_box_matches_dirichlet_reference(box):
    """Block velocity matches the exact Dirichlet free-slip reference."""
    rel = np.sqrt(np.sum((box["ref"].u.data - box["blk"].u.data) ** 2)) / np.sqrt(
        np.sum(box["ref"].u.data ** 2)
    )
    assert rel < 1e-3


def test_box_topography_is_normal_traction(box):
    """h on the wall correlates with -n.sigma.n (the dynamic-topography stress)."""
    blk, hR = box["blk"], box["hR"]
    sxx = blk.stress[0, 0]
    ch = hR.coords
    onb = ch[:, 0] > 1 - 1e-6
    nsn = np.array(uw.function.evaluate(sxx, ch[onb])).reshape(-1)
    corr = np.corrcoef(hR.data[onb, 0], -nsn)[0, 1]
    assert corr > 0.99


def test_multiplier_and_topography_api(box):
    blk, hL = box["blk"], box["hL"]
    assert blk.multiplier("Left") is hL
    assert blk.multiplier("Nonexistent") is None
    assert blk.topography("Left", buoyancy_scale=2.0) == hL.sym[0] / 2.0


def test_rejects_unknown_boundary(box):
    with pytest.raises(ValueError):
        box["blk"].add_constraint_bc("Nonexistent")


# --------------------------------------------------------------------------- #
# (B) enclosed annulus, curved boundary, DEFAULT solver options ("just works")
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def annulus():
    R_I, R_O, CELL, RA = 0.5, 1.0, 0.1, 1.0e2
    mesh = uw.meshing.Annulus(radiusInner=R_I, radiusOuter=R_O, cellSize=CELL, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x**2 + y**2)
    unit_r = sympy.Matrix([[x / r, y / r]])
    th = sympy.atan2(y, x)
    buoy = RA * sympy.cos(3 * th) * (r - R_I) / (R_O - R_I)

    ref = uw.systems.Stokes(mesh)
    ref.constitutive_model = uw.constitutive_models.ViscousFlowModel
    ref.constitutive_model.Parameters.shear_viscosity_0 = MU
    ref.saddle_preconditioner = 1.0 / MU
    ref.bodyforce = buoy * unit_r
    ref.add_dirichlet_bc((0.0, 0.0), "Lower")
    ref.add_natural_bc(1e6 * MU * unit_r.dot(ref.u.sym) * unit_r, "Upper")
    ref.tolerance = 1e-8
    ref.petsc_options["ksp_type"] = "fgmres"
    ref.solve()

    blk = uw.systems.Stokes_Constrained(mesh)
    blk.constitutive_model = uw.constitutive_models.ViscousFlowModel
    blk.constitutive_model.Parameters.shear_viscosity_0 = MU
    blk.saddle_preconditioner = 1.0 / MU
    blk.bodyforce = buoy * unit_r
    blk.add_dirichlet_bc((0.0, 0.0), "Lower")
    hb = blk.add_constraint_bc("Upper", g=0.0, normal=unit_r)
    # Enclosed -> pressure/multiplier gauge; DEFAULT grouped-Schur solver config.
    blk.petsc_use_pressure_nullspace = True
    blk.tolerance = 1e-8
    blk.solve()
    return dict(mesh=mesh, unit_r=unit_r, R_O=R_O, CELL=CELL, ref=ref, blk=blk, hb=hb)


def test_annulus_two_way_schur(annulus):
    """Default config collapses the 3-field DM to a 2-way velocity|[p,h] split."""
    pc = annulus["blk"].snes.getKSP().getPC()
    assert len(pc.getFieldSplitSubKSP()) == 2


def test_annulus_constraint_enforced(annulus):
    blk, unit_r = annulus["blk"], annulus["unit_r"]
    vn = blk.u.sym.dot(unit_r)
    num = float(uw.maths.BdIntegral(blk.mesh, fn=vn**2, boundary="Upper").evaluate())
    length = float(uw.maths.BdIntegral(blk.mesh, fn=1.0, boundary="Upper").evaluate())
    assert np.sqrt(num / length) < 2.0e-3


def test_annulus_matches_penalty(annulus):
    pts = annulus["ref"].u.coords
    vref = np.array(uw.function.evaluate(annulus["ref"].u.sym, pts))
    vblk = np.array(uw.function.evaluate(annulus["blk"].u.sym, pts))
    rel = np.sqrt(np.sum((vref - vblk) ** 2)) / np.sqrt(np.sum(vref**2))
    assert rel < 0.02


def test_annulus_topography_recovery(annulus):
    """h|_Gamma correlates with -n.sigma.n: the multiplier IS dynamic topography."""
    blk, unit_r, hb = annulus["blk"], annulus["unit_r"], annulus["hb"]
    R_O, CELL = annulus["R_O"], annulus["CELL"]
    srr = uw.discretisation.MeshVariable("srr_t1062", blk.mesh, 1, degree=2)
    proj = uw.systems.Projection(blk.mesh, srr)
    proj.uw_function = (unit_r * blk.stress * unit_r.T)[0, 0]
    proj.solve()
    ch = hb.coords
    onb = np.sqrt(ch[:, 0] ** 2 + ch[:, 1] ** 2) > R_O - 0.01
    corr = np.corrcoef(hb.data[onb, 0], -srr.data[onb, 0])[0, 1]
    assert corr > 0.99


# --------------------------------------------------------------------------- #
# (C) variable viscosity (SolCx-style): the AL is viscosity-weighted, so a
#     large contrast is the real stress test. Block must match Dirichlet ref.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("contrast", [1.0e2, 1.0e6])
def test_box_variable_viscosity_matches_dirichlet(contrast):
    def visc(m):
        xx, yy = m.X
        return sympy.Piecewise((1.0, xx < 0.5), (contrast, True))

    def forcing(m):
        xx, yy = m.X
        return sympy.Matrix([0.0, sympy.sin(sympy.pi * xx) * sympy.cos(sympy.pi * yy)])

    m0 = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.1, qdegree=3)
    ref = uw.systems.Stokes(m0)
    ref.constitutive_model = uw.constitutive_models.ViscousFlowModel
    ref.constitutive_model.Parameters.shear_viscosity_0 = visc(m0)
    ref.bodyforce = forcing(m0)
    ref.add_dirichlet_bc((0.0, 0.0), "Bottom")
    ref.add_dirichlet_bc((0.0, None), "Left")
    ref.add_dirichlet_bc((0.0, None), "Right")
    _direct(ref)
    ref.solve()

    mb = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.1, qdegree=3)
    blk = uw.systems.Stokes_Constrained(mb)
    blk.constitutive_model = uw.constitutive_models.ViscousFlowModel
    blk.constitutive_model.Parameters.shear_viscosity_0 = visc(mb)
    blk.bodyforce = forcing(mb)
    blk.add_dirichlet_bc((0.0, 0.0), "Bottom")
    blk.add_constraint_bc("Left", g=0.0, normal=sympy.Matrix([[-1.0, 0.0]]))
    blk.add_constraint_bc("Right", g=0.0, normal=sympy.Matrix([[1.0, 0.0]]))
    _direct(blk)
    blk.solve()

    rel = np.sqrt(np.sum((ref.u.data - blk.u.data) ** 2)) / np.sqrt(np.sum(ref.u.data ** 2))
    assert rel < 1e-4


# --------------------------------------------------------------------------- #
# (D) constraint Schur preconditioner under strong lateral viscosity contrast.
#     The bare `a11` (mass) Schur PC walls when the boundary viscosity contrast is
#     strong (the augmentation r carries all the conditioning, which also corrupts
#     the recovered lambda = topography). The constrained solver defaults to `selfp`,
#     which builds the true constraint Schur -C diag(A)^-1 C^T from the actual
#     operator blocks: it cracks the wall, makes the augmentation optional, and keeps
#     lambda clean at small r. Uniform mu still converges (the well-conditioned case).
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def contrast_annulus():
    R_I, R_O, CELL, RA, MU_HI = 0.5, 1.0, 0.1, 1.0e2, 1.0e3
    mesh = uw.meshing.Annulus(radiusInner=R_I, radiusOuter=R_O, cellSize=CELL, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x**2 + y**2)
    unit_r = sympy.Matrix([[x / r, y / r]])
    th = sympy.atan2(y, x)
    mu = MU * (MU_HI / MU) ** ((1 + sympy.cos(th)) / 2)   # smooth lateral ramp
    buoy = RA * sympy.cos(3 * th) * (r - R_I) / (R_O - R_I)

    blk = uw.systems.Stokes_Constrained(mesh)
    blk.constitutive_model = uw.constitutive_models.ViscousFlowModel
    blk.constitutive_model.Parameters.shear_viscosity_0 = mu
    blk.saddle_preconditioner = 1.0 / mu
    blk.bodyforce = buoy * unit_r
    blk.add_dirichlet_bc((0.0, 0.0), "Lower")
    # reduced augmentation (1e2, vs the 1e4 default): only viable with a Schur PC
    # that conditions the constraint block — here the default `selfp`.
    hb = blk.add_constraint_bc("Upper", g=0.0, normal=unit_r, augmentation_base=1.0e2)
    blk.petsc_use_pressure_nullspace = True
    blk.tolerance = 1e-8
    blk.petsc_options["snes_linesearch_type"] = "basic"
    blk.solve()
    return dict(mesh=mesh, unit_r=unit_r, R_O=R_O, blk=blk, hb=hb)


def test_contrast_converges(contrast_annulus):
    """With the default selfp Schur PC, reduced augmentation still converges."""
    blk = contrast_annulus["blk"]
    assert blk.snes.getConvergedReason() > 0


def test_contrast_constraint_enforced(contrast_annulus):
    blk, unit_r, mesh = contrast_annulus["blk"], contrast_annulus["unit_r"], contrast_annulus["mesh"]
    vn = blk.u.sym.dot(unit_r)
    num = float(uw.maths.BdIntegral(mesh, fn=vn**2, boundary="Upper").evaluate())
    length = float(uw.maths.BdIntegral(mesh, fn=1.0, boundary="Upper").evaluate())
    assert np.sqrt(num / length) < 1.0e-3


def test_contrast_topography_recovery(contrast_annulus):
    """At reduced augmentation the multiplier recovers -n.sigma.n (clean topography)."""
    blk, unit_r, hb = contrast_annulus["blk"], contrast_annulus["unit_r"], contrast_annulus["hb"]
    R_O = contrast_annulus["R_O"]
    srr = uw.discretisation.MeshVariable("srr_contrast", blk.mesh, 1, degree=2)
    proj = uw.systems.Projection(blk.mesh, srr)
    proj.uw_function = (unit_r * blk.stress * unit_r.T)[0, 0]
    proj.solve()
    ch = hb.coords
    onb = np.sqrt(ch[:, 0] ** 2 + ch[:, 1] ** 2) > R_O - 0.01
    corr = np.corrcoef(hb.data[onb, 0], -srr.data[onb, 0])[0, 1]
    assert corr > 0.99


def test_selfp_is_default_schur_pc(annulus):
    """The constrained solver defaults to the selfp Schur preconditioner."""
    blk = annulus["blk"]
    assert str(blk.petsc_options.getAll().get("pc_fieldsplit_schur_precondition")) == "selfp"
    # the experimental hand-scaled 1/mu Pmat term stays opt-in (off)
    assert blk.multiplier_schur_pc is False
