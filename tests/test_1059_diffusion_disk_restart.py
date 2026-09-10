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
    worker = root / "tests/parallel/ptest_1058_diffusion_disk_restart.py"
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(_MPI_ENV_PREFIXES)}
    launcher = [] if ranks == 1 else [str(Path(sys.executable).with_name("mpirun")), "-np", str(ranks)]
    for phase in ("write", "resume"):
        command = [sys.executable, str(root / "scripts/mpi_supervisor.py"),
                   "--silence", "45", "--hard-cap", "90", "--",
                   *launcher, sys.executable, "-m", "mpi4py", str(worker), phase]
        with (tmp_path / f"{phase}.log").open("w") as log:
            result = subprocess.run(command, cwd=tmp_path, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=100)
        assert result.returncode == 0, (tmp_path / f"{phase}.log").read_text()
    assert "Snapshot skipped state field" in (tmp_path / "write.log").read_text()
    maximum = 0.0
    for rank in range(ranks):
        with np.load(tmp_path / f"write_{rank}.npz") as expected, np.load(
                tmp_path / f"resume_{rank}.npz") as actual:
            assert set(actual) == set(expected)
            for name in expected:
                if not name.startswith("T_"):
                    np.testing.assert_array_equal(actual[name], expected[name])
            for name in ("T_0", "T_1"):
                maximum = max(maximum, float(np.max(np.abs(actual[name] - expected[name]))))
    # All exact field/metadata restoration checks on every rank run before
    # recognizing this known numerical limitation. Other failures stay failures.
    print(f"DIFFUSION_DISK_REPLAY ranks={ranks} max_error={maximum:.12g}")
    if maximum > 1e-11:
        raise SymbolicDiskContinuationMismatch(f"Disk replay max difference {maximum:.12g}")
