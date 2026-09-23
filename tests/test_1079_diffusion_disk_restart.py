"""Run parent in serial; UW_DIFFUSION_TEST_RANKS selects fresh worker ranks."""

import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest
import sympy
import underworld3 as uw
from parallel.serial_reference import _MPI_ENV_PREFIXES

pytestmark = [pytest.mark.tier_b]


class SymbolicDiskContinuationMismatch(AssertionError):
    """Known disk limitation; setup and exact-restore failures are not xfailed."""


def _checkpoint_fields(wrapper):
    """Read the writer's flushed PETSc vectors and layouts, not sidecar logs."""
    fields = {}
    with h5py.File(wrapper) as snapshot:
        bulk = wrapper.parent / snapshot["meshes"].attrs["bulk_dir"]
        for mesh_name, mesh in snapshot["meshes"].items():
            for name, variable in mesh["variables"].items():
                with h5py.File(bulk / variable.attrs["external_file"]) as data:
                    arrays = {}
                    data.visititems(lambda key, obj: arrays.update({key: obj[()]})
                                    if isinstance(obj, h5py.Dataset) else None)
                assert arrays, f"Missing checkpoint datasets for {name}"
                assert f"uw_checkpoint/{name}" in arrays
                fields[mesh_name, name] = arrays
    assert fields
    return fields


def _assert_saved_metadata_equal(reference, restored):
    with h5py.File(reference) as expected, h5py.File(restored) as actual:
        assert set(expected["python_state"]) == set(actual["python_state"])
        for name, group in expected["python_state"].items():
            other = actual["python_state"][name]
            assert set(group.attrs) == set(other.attrs)
            for key in group.attrs:
                np.testing.assert_array_equal(other.attrs[key], group.attrs[key])
        symbolic = [group for name, group in expected["python_state"].items()
                    if name.startswith("Symbolic_")]
        assert len(symbolic) == 1
        assert "psi_star__skipped" in symbolic[0].attrs
        assert "MutableDenseMatrix" in symbolic[0].attrs["psi_star__skipped"]


@pytest.mark.parametrize("value", [object(), [sympy.Matrix([1])]])
@pytest.mark.level_1
def test_skipped_snapshot_state_warns(tmp_path, value):
    from underworld3.checkpoint.disk_snapshot import _serialise_field

    with h5py.File(tmp_path / "state.h5", "w") as f:
        group = f.create_group("state")
        with pytest.warns(RuntimeWarning, match="Snapshot skipped state field /state/history"):
            _serialise_field(group, "history", value)
        assert "history__skipped" in group.attrs


@pytest.mark.level_2
@pytest.mark.xfail(
    strict=True, raises=SymbolicDiskContinuationMismatch,
    reason="PR #708: disk snapshots skip live symbolic flux history",
)
def test_symbolic_diffusion_fresh_process_disk_restart(tmp_path):
    if uw.mpi.size != 1:
        pytest.skip("Run parent in serial; UW_DIFFUSION_TEST_RANKS selects worker ranks.")
    ranks = int(os.environ.get("UW_DIFFUSION_TEST_RANKS", "1"))
    root = Path(__file__).resolve().parents[1]
    worker = root / "tests/parallel/ptest_1079_diffusion_disk_restart.py"
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(_MPI_ENV_PREFIXES)}
    launcher = [] if ranks == 1 else [str(Path(sys.executable).with_name("mpirun")), "-np", str(ranks)]
    for phase in ("write", "resume"):
        command = [sys.executable, str(root / "scripts/mpi_supervisor.py"),
                   "--silence", "45", "--hard-cap", "90", "--",
                   *launcher, sys.executable, "-m", "mpi4py", str(worker), "-uw_phase", phase]
        with (tmp_path / f"{phase}.log").open("w") as log:
            result = subprocess.run(command, cwd=tmp_path, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=100)
        assert result.returncode == 0, (tmp_path / f"{phase}.log").read_text()
    assert "Snapshot skipped state field" in (tmp_path / "write.log").read_text()
    _assert_saved_metadata_equal(tmp_path / "restart.h5", tmp_path / "restored.h5")
    expected = _checkpoint_fields(tmp_path / "restart.h5")
    actual = _checkpoint_fields(tmp_path / "restored.h5")
    assert set(actual) == set(expected)
    for field in expected:
        assert set(actual[field]) == set(expected[field])
        for dataset in expected[field]:
            np.testing.assert_array_equal(actual[field][dataset], expected[field][dataset])

    maximum = 0.0
    for step in range(2):
        expected = _checkpoint_fields(tmp_path / f"write_{step}.h5")
        actual = _checkpoint_fields(tmp_path / f"resume_{step}.h5")
        assert set(actual) == set(expected)
        for field in expected:
            assert set(actual[field]) == set(expected[field])
            for dataset, values in expected[field].items():
                if "/vecs/" in dataset or dataset.startswith("uw_checkpoint/"):
                    assert actual[field][dataset].shape == values.shape
                    assert np.all(np.isfinite(actual[field][dataset]))
                    maximum = max(maximum, float(np.max(np.abs(actual[field][dataset] - values))))
                else:
                    np.testing.assert_array_equal(actual[field][dataset], values)
    # Exact checkpoint restoration and layout checks precede the known
    # numerical limitation. Missing state or corrupt layouts fail normally.
    print(f"DIFFUSION_DISK_REPLAY ranks={ranks} max_error={maximum:.12g}")
    if maximum > 1e-11:
        raise SymbolicDiskContinuationMismatch(f"Disk replay max difference {maximum:.12g}")
