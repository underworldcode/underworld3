"""ThinSheet follow-ups from the review of the thin-sheet work (#875, #879).

* #875: L and c_b went into F0/F1 as their numbers, so describe() and the
  transcript showed numbers, and a dimensional thickness skipped the unwrap
  path. The rows from add_basal_drag / add_basal_traction were declared
  nowhere. They now enter as the expressions, and basal_coupling is a term.
* #879: the history projection after a viscoelastic step was not in the
  transcript, and the facet-normal field was filled once and went stale after
  a deform.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2]


def _box_sheet():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.25)
    sheet = uw.systems.ThinSheet(mesh)
    sheet.constitutive_model = uw.constitutive_models.ViscousFlowModel
    sheet.constitutive_model.Parameters.shear_viscosity_0 = 1
    sheet.tolerance = 1.0e-10
    return mesh, sheet


@pytest.mark.tier_a
def test_thickness_and_drag_enter_as_expressions_and_the_basal_rows_are_a_term():
    mesh, sheet = _box_sheet()
    sheet.thickness = 2
    sheet.basal_drag = 4
    sheet.bodyforce = [1, 0]
    # uniform u: no strain, so c u = L f, u = L f / c
    sheet.solve()
    assert np.allclose(sheet.u.array[:, 0, :], [0.5, 0.0], atol=1e-8)
    assert sympy.Matrix(sheet.F1.sym).has(sheet.thickness)
    assert sympy.Matrix(sheet.F0.sym).has(sheet.thickness)
    assert sympy.Matrix(sheet.F0.sym).has(sheet.basal_drag)
    sheet.thickness = 3
    sheet.solve()
    assert np.allclose(sheet.u.array[:, 0, :], [0.75, 0.0], atol=1e-8)

    names = [t["name"] for t in sheet._declared_terms()]
    assert "basal_coupling" in names
    sheet.add_basal_traction([0.5, 0])
    assert sympy.Matrix(sheet.basal_coupling) == sympy.Matrix([[-0.5, 0]])


def _sphere_sheet(eta):
    sphere = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.3, degree=1, qdegree=5)
    x, y, z = sphere.X
    sheet = uw.systems.ThinSheet(sphere)
    sheet.constitutive_model = uw.constitutive_models.ViscousFlowModel
    sheet.constitutive_model.Parameters.shear_viscosity_0 = eta(sheet)
    sheet.tolerance = 1.0e-9
    sheet.basal_drag = 1
    sheet.bodyforce = list(sympy.Matrix([[2 * y * z, -2 * x * z, 0]]))
    return sphere, sheet


def _triangle_normals(mesh):
    """Unit normal of each cell from its own three vertices, ``(b - a) x (c - a)``:
    independent of the DM's cell geometry, which the code under test uses. The
    sign is the cell's orientation, so compare up to sign."""
    dm = mesh.dm
    c0, c1 = dm.getHeightStratum(0)
    v0, v1 = dm.getDepthStratum(0)
    X = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, mesh.cdim)
    out = []
    for c in range(c0, c1):
        verts = [int(q) - v0 for q in dm.getTransitiveClosure(c)[0] if v0 <= q < v1]
        a, b, d = X[verts]
        n = np.cross(b - a, d - a)
        out.append(n / np.linalg.norm(n))
    return np.array(out)


@pytest.mark.tier_a
def test_the_facet_normal_follows_a_deform_through_the_solve():
    """After a deform, the solve refills the facet normal its projector reads
    (it was filled once and REMAP'd: 1.2e-2 off). The reference is each
    triangle's own vertex cross product."""
    sphere, sheet = _sphere_sheet(lambda s: 1)
    field = sphere._sheet_facet_normal
    before = np.array(np.asarray(field.array)[:, 0, :], copy=True)
    X = np.array(sphere.X.coords, copy=True)
    X[:, 0] *= 1.2                                             # an ellipsoid: the normals turn
    sphere.deform(X)
    try:
        sheet.solve(zero_init_guess=True)
    except Exception:
        # Sanctioned: the sheet's tangency still uses the sphere's analytic
        # normal (#879, left open), so the solve on an ellipsoid need not
        # converge; what is asserted is the refill it performs first.
        pass
    now = np.asarray(field.array)[:, 0, :]
    reference = _triangle_normals(sphere)
    assert np.abs(np.abs((now * reference).sum(axis=1)) - 1).max() < 1e-12
    assert np.abs(now - before).max() > 1e-3                   # it did move


@pytest.mark.tier_b      # minutes: a nonlinear / viscoelastic sphere solve
def test_the_history_projection_is_in_the_transcript_and_refuses_an_adjoint():
    uw.reset_default_model()
    model = uw.get_default_model()
    sphere = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.3, degree=1, qdegree=5)
    x, y, z = sphere.X
    sheet = uw.systems.ThinSheet(sphere)
    sheet.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        sheet.Unknowns, order=1, integrator="etd", objective_rate="jaumann")
    sheet.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    sheet.constitutive_model.Parameters.shear_modulus = 1
    sheet.tolerance = 1.0e-8
    sheet.basal_drag = 1
    sheet.bodyforce = list(sympy.Matrix([[2 * y * z, -2 * x * z, 0]]))
    model.clear_transcript()
    model.record_every = 1
    with model.step(0.1):
        sheet.solve(timestep=0.1, zero_init_guess=False)
    events = list(model.transcript)[-1].events
    kinds = [e["kind"] for e in events]
    assert "history_projection" in kinds, kinds
    assert kinds.index("history_projection") > kinds.index("history_shift")
    projection = events[kinds.index("history_projection")]
    assert projection["adjoint"]["supported"] is False
