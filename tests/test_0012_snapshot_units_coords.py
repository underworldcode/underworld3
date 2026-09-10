"""A snapshot must not rescale the mesh.

``mesh.X.coords`` is the UNIT-AWARE view: with a model that declares a length
scale it returns metres, while the DM coordinate vector the restore path writes
back into holds model units. Capturing one and restoring the other multiplies
the mesh by the length scale — silently, because every array keeps its shape
and every field is restored correctly. Only the geometry is wrong, so what
fails afterwards is every integral, every evaluate, and every subsequent solve.

``model.rewind()`` goes straight through this path, which is how it was found.
"""

import pytest

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

import numpy as np


LENGTH_SCALE_M = 500e3


def _model_with_units():
    import underworld3 as uw

    uw.reset_default_model()
    model = uw.get_default_model()
    model.set_reference_quantities(
        domain_depth=uw.quantity(500, "km"),
        material_viscosity=uw.quantity(1e21, "Pa*s"),
        lithostatic_pressure=uw.quantity(3300 * 9.81 * 500e3, "Pa"),
    )
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8.0
    )
    return uw, model, mesh


def test_restore_leaves_the_mesh_at_its_own_size():
    """Round-tripping a snapshot must not scale the coordinates."""
    uw, model, mesh = _model_with_units()

    dimensional_before = np.asarray(mesh.X.coords).copy()
    raw_before = np.asarray(mesh._coords).copy()
    assert dimensional_before.max() == pytest.approx(LENGTH_SCALE_M, rel=1e-6), (
        "the fixture is not exercising the unit-aware view"
    )
    assert raw_before.max() == pytest.approx(1.0, rel=1e-6)

    snap = model.save_state()
    model.load_state(snap)

    assert np.asarray(mesh._coords).max() == pytest.approx(1.0, rel=1e-12), (
        "restore rescaled the mesh: the captured coordinates were dimensional "
        "but were written back as model units"
    )
    assert np.allclose(np.asarray(mesh.X.coords), dimensional_before, rtol=0, atol=0)


def test_repeated_restores_do_not_drift():
    """The scaling error compounds, so check more than one round trip."""
    uw, model, mesh = _model_with_units()
    raw_before = np.asarray(mesh._coords).copy()

    for _ in range(3):
        snap = model.save_state()
        model.load_state(snap)

    assert np.allclose(np.asarray(mesh._coords), raw_before, rtol=0, atol=0)


def test_evaluate_still_works_after_a_restore():
    """The symptom, not the mechanism: a rescaled mesh puts every sample point
    outside the domain, and evaluate quietly returns the value at one corner."""
    uw, model, mesh = _model_with_units()
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    x, y = mesh.X
    T.array[:, 0, 0] = np.asarray(T.coords)[:, 1] / LENGTH_SCALE_M

    sample = np.column_stack([np.full(9, 0.5), np.linspace(0.05, 0.95, 9)])
    before = np.asarray(uw.function.evaluate(T.sym[0], sample)).ravel()
    assert np.ptp(before) > 0.5, "the fixture should vary across the sample line"

    model.load_state(model.save_state())

    after = np.asarray(uw.function.evaluate(T.sym[0], sample)).ravel()
    assert np.allclose(after, before, rtol=1e-10, atol=1e-12), (
        "evaluate disagrees with itself across a snapshot round trip"
    )


def test_rewind_reaches_the_state_the_step_started_from():
    """The path this was found on: record a step, rewind, and check the mesh."""
    uw, model, mesh = _model_with_units()
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    T.array[:, 0, 0] = 1.0

    model.tracker.time = 0.0
    model.tracker.step = 0
    model.record_every = 1

    raw_before = np.asarray(mesh._coords).copy()
    with model.step(0.5):
        T.array[:, 0, 0] = 2.0

    model.rewind()

    assert np.allclose(np.asarray(mesh._coords), raw_before, rtol=0, atol=0)
    assert np.allclose(np.asarray(T.array)[:, 0, 0], 1.0)
    assert model.tracker.time == 0.0
