"""The full-disc annulus generators mark the centre point when asked.

``centre=True`` with ``radiusInner=0`` puts the centre vertex in a ``Centre``
label, so a Dirichlet condition on ``"Centre"`` pins the field at r = 0.
AnnulusWithSpokes accepted ``centre`` and ignored it: the gmsh model had no
physical group for the centre point, and a condition on Centre was refused
(the example sweep for #944, W1 2026-11-21). QuarterAnnulus
marks the centre always.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]

GENERATORS = {
    "Annulus": lambda: uw.meshing.Annulus(
        radiusOuter=1.0, radiusInner=0.0, cellSize=0.2, centre=True),
    "AnnulusWithSpokes": lambda: uw.meshing.AnnulusWithSpokes(
        radiusOuter=1.0, radiusInner=0.0, centre=True),
    "AnnulusInternalBoundary": lambda: uw.meshing.AnnulusInternalBoundary(
        radiusOuter=1.5, radiusInternal=1.0, radiusInner=0.0, cellSize=0.2, centre=True),
    # marks the centre always (QuarterAnnulus does not use ``centre``)
    "QuarterAnnulus": lambda: uw.meshing.QuarterAnnulus(
        radiusOuter=1.0, radiusInner=0.0, cellSize=0.2),
}


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_centre_label_holds_the_one_centre_vertex(name):
    mesh = GENERATORS[name]()
    dm = mesh.dm
    assert dm.hasLabel("Centre")
    points = dm.getLabel("Centre").getStratumIS(mesh.boundaries.Centre.value)
    points = [] if points is None else points.getIndices()
    v0, v1 = dm.getDepthStratum(0)
    vertices = [p for p in points if v0 <= p < v1]
    assert len(vertices) == 1
    xy = mesh.X.coords[vertices[0] - v0]
    assert np.allclose(xy, 0.0, atol=1.0e-12)


def test_a_dirichlet_condition_on_the_centre_pins_the_field_there():
    mesh = GENERATORS["AnnulusWithSpokes"]()
    u = uw.discretisation.MeshVariable("u_centre", mesh, 1, degree=1)
    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1
    poisson.f = 0.0
    poisson.add_dirichlet_bc((1.0,), "Upper")
    poisson.add_dirichlet_bc((-1.0,), "Centre")
    poisson.solve()
    at_centre = uw.function.evaluate(u.sym[0], np.array([[0.0, 0.0]]))
    assert np.allclose(at_centre, -1.0, atol=1.0e-10)
