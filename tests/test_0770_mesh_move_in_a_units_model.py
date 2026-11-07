"""A mesh move works the same in a model with reference quantities.

mesh.X.coords is in physical units there and the DM holds non-dimensional coordinates.
The remesh transaction read the first and wrote it to the second, so mesh.deform left the
mesh scaled by the length scale, and it remapped a field at physical points as if they
were model points and wrote the dimensional answer into the raw values. See #979.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]

LENGTH_SCALE = 1.0e6    # metres, the reference length below


@pytest.fixture
def units_model():
    uw.reset_default_model()
    model = uw.get_default_model()
    model.set_reference_quantities(
        length=uw.quantity(1000, "km"), temperature_diff=uw.quantity(1000, "K"),
        viscosity=uw.quantity(1e21, "Pa*s"), diffusivity=uw.quantity(1e-6, "m**2/s"))
    yield model
    uw.reset_default_model()


def _dm_coords(mesh):
    return np.array(mesh.dm.getCoordinatesLocal().array).reshape(-1, mesh.cdim)


@pytest.mark.parametrize("form", ["metres", "kilometres", "model"])
def test_deform_keeps_the_mesh_in_its_frame(units_model, form):
    """Coordinates with units are converted, in whatever length unit; a plain array is
    model coordinates, as evaluate reads one."""
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 8, qdegree=3)
    start = _dm_coords(mesh)
    assert np.linalg.norm(start, axis=1).max() == pytest.approx(1.0)

    radius = np.linalg.norm(start, axis=1)
    inside = (radius > 0.5 + 1.0e-9) & (radius < 1.0 - 1.0e-9)
    scale = np.where(inside, 1.02, 1.0)[:, None]
    if form == "model":
        target = start * scale
    else:
        target = mesh.X.coords * scale
        if form == "kilometres":
            target = target.to("km")
    assert mesh.deform(target) is True

    expected = start.copy()
    expected[inside] *= 1.02
    assert np.abs(_dm_coords(mesh) - expected).max() < 1.0e-9
    assert np.abs(np.array(mesh.X.coords) - expected * LENGTH_SCALE).max() < 1.0e-3


def test_deform_remaps_a_field_with_units(units_model):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 8, qdegree=3)
    T = uw.discretisation.MeshVariable("T0770", mesh, 1, degree=1, units="K")
    start = _dm_coords(mesh)
    T.data[:, 0] = 0.3 + 0.2 * start[:, 0]      # raw values, linear in position

    radius = np.linalg.norm(start, axis=1)
    mesh.deform(mesh.X.coords * np.where((radius > 0.5 + 1.0e-9) & (radius < 1.0 - 1.0e-9),
                                         1.02, 1.0)[:, None])

    moved = _dm_coords(mesh)
    interior = np.linalg.norm(moved, axis=1) < 0.99
    raw = np.array(T.data).reshape(-1)
    assert np.abs(raw - (0.3 + 0.2 * moved[:, 0]))[interior].max() < 1.0e-9


def test_the_mover_moves_nodes_and_remaps_in_a_units_model(units_model):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 8, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    T = uw.discretisation.MeshVariable("T0770m", mesh, 1, degree=1, units="K")
    start = _dm_coords(mesh)
    T.data[:, 0] = 0.3 + 0.2 * start[:, 0]
    density = 1 + 24 * sympy.exp(-(((r - 0.9) / 0.06) ** 2))
    values = np.asarray(uw.function.evaluate(density, start))
    assert values.max() > 10.0      # the metric has a feature: a flat one moves nothing
    uw.meshing.node_redistribution(mesh, density, slip_surfaces=True, skip_threshold=None,
                                   method_kwargs=dict(accel="none", n_outer=20))
    moved = _dm_coords(mesh)
    assert np.abs(moved - start).max() > 1.0e-3
    assert np.linalg.norm(moved, axis=1).max() == pytest.approx(1.0)
    raw = np.array(T.data).reshape(-1)
    interior = np.linalg.norm(moved, axis=1) < 0.99
    assert np.abs(raw - (0.3 + 0.2 * moved[:, 0]))[interior].max() < 1.0e-9


def test_ephemeral_coords_restores_the_mesh_in_its_frame(units_model):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 8, qdegree=3)
    start = _dm_coords(mesh)
    with mesh.ephemeral_coords():
        mesh._deform_mesh(start * 1.01)
    assert np.abs(_dm_coords(mesh) - start).max() < 1.0e-12


def test_a_submesh_follows_its_parent_in_its_frame(units_model):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 8, qdegree=3)
    surface = mesh.extract_surface("Upper")
    radius = np.linalg.norm(_dm_coords(mesh), axis=1)
    mesh.deform(mesh.X.coords * np.where(radius > 1.0 - 1.0e-9, 1.05, 1.0)[:, None])
    on_surface = np.linalg.norm(_dm_coords(surface), axis=1)
    assert np.abs(on_surface - 1.05).max() < 1.0e-9
