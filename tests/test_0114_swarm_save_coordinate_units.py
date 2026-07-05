"""
Regression test for Swarm.save() coordinate-system consistency (SWARM-19 / BF-17).

``Swarm.save()`` has two IO branches: a parallel-HDF5 path and a sequential
fallback (``force_sequential=True`` or h5py built without MPI). The parallel
branch saved ``_particle_coordinates.data`` (model units) while the sequential
branch saved the deprecated ``self.points`` property, which multiplies by the
model length scale when coordinate scaling is active — so the two branches
produced checkpoints that differed by the length scale, and the sequential
files could not round-trip through ``read_timestep`` (which re-inserts raw
coordinates as model units).

Both branches must write MODEL-UNIT coordinates (the convention the parallel
branch and ``read_timestep`` already used).
"""

import os

import h5py
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.fixture
def scaled_model():
    """A default model with a length scale, so coordinate scaling is active."""
    uw.reset_default_model()
    model = uw.get_default_model()
    model.set_reference_quantities(
        domain_depth=uw.quantity(1000, "km"),
        plate_velocity=uw.quantity(5, "cm/year"),
    )
    yield model
    uw.reset_default_model()


def _saved_coordinates(filename):
    with h5py.File(filename, "r") as h5f:
        return h5f["coordinates"][:]


def test_save_writes_model_units_in_both_io_branches(scaled_model, tmp_path):
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    swarm = uw.swarm.Swarm(mesh)
    swarm.populate(fill_param=1)

    # Coordinate scaling must actually be active for this test to bite:
    # with a 1000 km depth reference the length scale is 1e6 (metres).
    assert mesh.CoordinateSystem._scaled
    assert mesh.CoordinateSystem._length_scale != 1.0

    # np.asarray: drop the NDArray_With_Callback wrapper for plain numpy ops
    model_coords = np.asarray(swarm._particle_coordinates.data).copy()

    f_seq = str(tmp_path / "swarm_seq.h5")
    swarm.save(f_seq, force_sequential=True)
    seq_coords = _saved_coordinates(f_seq)
    np.testing.assert_allclose(
        np.sort(seq_coords, axis=0),
        np.sort(model_coords, axis=0),
        rtol=1e-12,
        err_msg="sequential save() branch must write model-unit coordinates "
        "(SWARM-19: it used to write physically-scaled self.points)",
    )

    if h5py.h5.get_config().mpi:
        f_par = str(tmp_path / "swarm_par.h5")
        swarm.save(f_par)
        par_coords = _saved_coordinates(f_par)
        np.testing.assert_allclose(
            np.sort(par_coords, axis=0),
            np.sort(seq_coords, axis=0),
            rtol=1e-12,
            err_msg="parallel and sequential save() branches must write the "
            "same coordinate system (SWARM-19)",
        )


def test_save_read_roundtrip_with_scaling(scaled_model, tmp_path):
    """A sequential checkpoint must round-trip through read_timestep."""
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    swarm = uw.swarm.Swarm(mesh)
    swarm.populate(fill_param=1)
    original = np.sort(np.asarray(swarm._particle_coordinates.data), axis=0)

    # save() then read back through the read_timestep naming convention
    base = str(tmp_path / "chk")
    filename = base + ".swarm.00000.h5"
    swarm.save(filename, force_sequential=True)

    swarm2 = uw.swarm.Swarm(mesh)
    swarm2.dm.finalizeFieldRegister()
    swarm2.read_timestep(os.path.basename(base), "swarm", 0, outputPath=str(tmp_path))

    restored = np.sort(np.asarray(swarm2._particle_coordinates.data), axis=0)
    assert restored.shape == original.shape, (
        "read_timestep dropped particles: coordinates were saved in the wrong "
        "unit system (physically scaled, outside the model-unit mesh domain)"
    )
    np.testing.assert_allclose(restored, original, rtol=1e-12)
