"""A manifold thin sheet takes ``solve(picard=N)`` and an untransported history (#876, #887).

A sheet on a sphere always solves on the rotated path. ``SNES_Vector.solve``
handed that path ``picard=0``, and ``ThinSheet.solve`` took no ``picard``:
``sheet.solve(picard=2)`` was a TypeError, so the ``solve(picard=N)`` recipe of the
plasticity guides could not reach a nonlinear sheet. And a manifold sheet refused
``stress_transport="none"``, which stress-transport.md recommends for a tidal cycle.

- ``picard=3`` stages at least three Picard iterations, then Newton, and converges
  to the answer of the plain solve; with ``newton_switch_rtol=1`` the switch comes
  at exactly the third;
- the GAMG-fallback advice on a refined mesh names the call that works, with
  ``field_id=0`` for Stokes and without it for a sheet, and following it builds
  the multigrid (#887);
- an explicit picard stages on a fault contact, for a sheet and for Stokes, where
  it raised; a cold picard=0 solve there compiles no blended Jacobian;
- ``stress_transport="none"`` is taken on a manifold and agrees with the grid
  history on a slow flow; the flavours that move points off the surface are
  still refused.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _sphere():
    return uw.meshing.SphericalManifold(radius=1.0, cellSize=0.3, degree=1, qdegree=5)


def _toroidal(sphere):
    x, y, z = sphere.X
    return sympy.Matrix([[2 * y * z, -2 * x * z, 0]])


def _nonlinear_sheet(sphere):
    sheet = uw.systems.ThinSheet(sphere)
    sheet.constitutive_model = uw.constitutive_models.ViscousFlowModel
    u = sheet.u.sym
    sheet.constitutive_model.Parameters.shear_viscosity_0 = 1 + 50 * u.dot(u)
    sheet.basal_drag = 1
    sheet.bodyforce = list(_toroidal(sphere))
    sheet.tolerance = 1.0e-8
    return sheet


def test_picard_stages_a_picard_phase_on_a_manifold_sheet():
    sphere = _sphere()
    plain = _nonlinear_sheet(sphere)
    plain.solve()
    u_plain = np.asarray(plain.u.array).copy()

    staged = _nonlinear_sheet(sphere)
    # development: TypeError, unexpected keyword argument 'picard'
    staged.solve(picard=3)
    info = staged._rotated_freeslip_info
    assert info["continuation_switched"], "no Picard phase: picard did not reach the rotated loop"
    assert info["nonlinear_iterations"] >= 3, info["nonlinear_iterations"]
    assert info["converged"]
    u_staged = np.asarray(staged.u.array)
    assert np.linalg.norm(u_staged - u_plain) < 1.0e-6 * np.linalg.norm(u_plain)

    exact = _nonlinear_sheet(sphere)
    exact.newton_switch_rtol = 1.0          # switch as soon as picard iterations have run
    exact.solve(picard=3)
    info = exact._rotated_freeslip_info
    assert info["continuation_switch_iteration"] == 3, info["continuation_switch_iteration"]
    assert info["converged"]


def _follow_the_advice(capfd, solver, mesh):
    from underworld3.utilities import custom_mg
    solver.solve()
    out = capfd.readouterr().out
    assert "falls back to GAMG" in out, out
    field = "field_id=0" in out
    call = ("custom_mg.set_custom_fmg(solver, mesh.multigrid_levels[:-1]"
            + (", field_id=0)" if field else ")"))
    assert call in out, out
    custom_mg.set_custom_fmg(solver, mesh.multigrid_levels[:-1],
                             **({"field_id": 0} if field else {}))
    return field


def test_the_gamg_advice_on_a_refined_stokes_mesh_works(capfd):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.4,
                              refinement=1, qdegree=3)
    x, y = mesh.X
    def stokes(tag):
        v = uw.discretisation.MeshVariable(f"V1159{tag}", mesh, 2, degree=2)
        p = uw.discretisation.MeshVariable(f"P1159{tag}", mesh, 1, degree=1)
        s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
        s.constitutive_model = uw.constitutive_models.ViscousFlowModel
        s.bodyforce = sympy.Matrix([[x * y, x ** 2 - y ** 2]])
        s.add_rotated_freeslip_bc(0, "Lower")
        s.add_rotated_freeslip_bc(0, "Upper")
        s.petsc_use_pressure_nullspace = True
        return s
    s = stokes("a")
    # development's advice left out field_id=0: CollectiveBuildError
    assert _follow_the_advice(capfd, s, mesh)
    s.solve()
    assert s._rotated_freeslip_info["converged"]
    assert s._rotated_freeslip_info["velocity_pc_type"] != "gamg"
    assert "falls back to GAMG" not in capfd.readouterr().out


def test_the_gamg_advice_on_a_refined_sheet_works(capfd):
    sphere = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.5, degree=1, qdegree=5,
                                          refinement=1)
    sheet = _nonlinear_sheet(sphere)
    assert not _follow_the_advice(capfd, sheet, sphere)
    sheet.solve()
    assert sheet._rotated_freeslip_info["converged"]
    assert sheet._rotated_freeslip_info["velocity_pc_type"] != "gamg"


def test_a_manifold_sheet_takes_an_untransported_history():
    # stress-transport.md recommends "none" for a tidal or seismic cycle; on a
    # manifold it was refused with the trace-back flavours (#887). A slow flow
    # moves far less than a cell a step, so the untransported history agrees with
    # the grid-carried one, and both stay tangent to the sphere.
    sphere = _sphere()
    x, y, z = sphere.X
    taus = {}
    for transport in ("eulerian", "none"):
        sheet = uw.systems.ThinSheet(sphere)
        sheet.stress_transport = transport
        assert sheet.stress_transport == transport
        sheet.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
            sheet.Unknowns, order=1, integrator="etd")
        sheet.constitutive_model.Parameters.shear_viscosity_0 = 1
        sheet.constitutive_model.Parameters.shear_modulus = 1
        sheet.basal_drag = 4
        sheet.bodyforce = list(4.0e-3 * sympy.Matrix([[2 * y * z, -2 * x * z, 0]]))
        sheet.tolerance = 1.0e-9
        for _ in range(4):
            sheet.solve(timestep=0.1, zero_init_guess=False)
        taus[transport] = np.asarray(sheet.tau.array).copy()
        c = sheet.tau.coords
        rhat = c / np.linalg.norm(c, axis=1)[:, None]
        normal = np.linalg.norm(np.einsum("nij,nj->ni", taus[transport], rhat), axis=1)
        assert normal.max() < 1.0e-12 * np.abs(taus[transport]).max()
    scale = np.abs(taus["eulerian"]).max()
    assert scale > 0.0
    # they differ (1.1e-4 of the stress, measured), so "none" did take effect
    difference = np.abs(taus["none"] - taus["eulerian"]).max()
    assert 0.0 < difference < 1.0e-3 * scale
    with pytest.raises(ValueError, match="manifold"):
        uw.systems.ThinSheet(sphere).stress_transport = "semi_lagrangian"


def _fault_sheet():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.1, qdegree=3)
    split = mesh.add_fault(("Flt", np.array([[0.3, 0.45], [0.7, 0.55]])))
    x, y = split.X
    s = uw.systems.ThinSheet(split)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    u = s.u.sym
    s.constitutive_model.Parameters.shear_viscosity_0 = 1 + 50 * u.dot(u)
    s.tolerance = 1.0e-8
    for wall in ("Left", "Right", "Top", "Bottom"):
        s.add_essential_bc([y, 0], wall)                 # remote shear
    s.add_fault_bc(0, "Flt")                             # frictionless
    return s


def test_picard_stages_on_a_fault_contact_sheet():
    # a fault contact runs the rotated loop too; picard=2 raised NotImplementedError
    # there because solve() did not compile the blended Jacobian for it (hyperion's
    # review of #876, F4)
    plain = _fault_sheet()
    plain.solve()
    assert plain._rotated_freeslip_info["converged"]
    # the loop takes no automatic Picard step, so a cold solve compiles no blend
    assert not plain._picard_blend
    u_plain = np.asarray(plain.u.array).copy()
    staged = _fault_sheet()
    staged.solve(picard=2)
    info = staged._rotated_freeslip_info
    assert info["continuation_switched"], "no Picard phase on the fault-contact loop"
    assert info["converged"]
    u_staged = np.asarray(staged.u.array)
    # both stop at tolerance 1e-8 of the residual: 1.5e-8 apart (wombat)
    assert np.linalg.norm(u_staged - u_plain) < 1.0e-7 * np.linalg.norm(u_plain)


def test_picard_stages_on_a_fault_contact_stokes_solve():
    # the same staging on Stokes, whose fault contacts raised for picard > 0 too
    def build():
        base = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                                 cellSize=0.1, refinement=1, qdegree=3)
        mesh = base.add_fault([("Flt", np.array([[0.30, 0.45], [0.70, 0.55]]))])
        _x, y = mesh.X
        s = uw.systems.Stokes(mesh)
        s.constitutive_model = uw.constitutive_models.ViscousFlowModel
        u = s.Unknowns.u.sym
        s.constitutive_model.Parameters.shear_viscosity_0 = 1 + 50 * u.dot(u)
        s.petsc_use_pressure_nullspace = True
        for wall in ("Top", "Bottom", "Left", "Right"):
            s.add_dirichlet_bc((y - 0.5, 0.0), wall)
        s.add_fault_bc(0, "Flt")
        s.tolerance = 1.0e-8
        return s
    plain = build()
    plain.solve()
    assert plain._rotated_freeslip_info["converged"] and not plain._picard_blend
    u_plain = np.asarray(plain.u.array).copy()
    staged = build()
    staged.solve(picard=2)
    info = staged._rotated_freeslip_info
    assert info["continuation_switched"] and info["converged"], info
    # 1.4e-10 apart (wombat)
    assert np.linalg.norm(np.asarray(staged.u.array) - u_plain) < 1.0e-7 * np.linalg.norm(u_plain)
