"""Fresh-process transport snapshots: pc2, CN and BDF2 on tiny tetrahedra.

Run this parent pytest in serial. UW_SUPG_TEST_RANKS=8 requests eight-rank
workers; the default uses singleton workers. Every phase starts a fresh
interpreter. No forked in-memory snapshot can satisfy the restore check.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest

import underworld3 as uw
from parallel.serial_reference import _MPI_ENV_PREFIXES

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


@pytest.mark.parametrize("method", ["pc2", "cn", "bdf2"])
def test_fresh_process_transport_restart(method, tmp_path):
    if uw.mpi.size != 1:
        pytest.skip("Run the parent in serial; UW_SUPG_TEST_RANKS selects worker ranks.")
    ranks = int(os.environ.get("UW_SUPG_TEST_RANKS", "1"))
    root = Path(__file__).resolve().parents[1]
    worker = root / "tests/parallel/ptest_1119_supg_restart.py"
    supervisor = root / "scripts/mpi_supervisor.py"
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(_MPI_ENV_PREFIXES)}
    launcher = []
    if ranks > 1:
        executable = Path(sys.executable).with_name("mpirun")
        if not executable.is_file():
            executable = shutil.which("mpirun")
        assert executable, "Activate the matching MPI environment before this test."
        launcher = [str(executable), "-np", str(ranks)]
    if ranks > 1 and not supervisor.is_file():
        pytest.skip("MPI restart supervision requires development commit #678.")
    for phase in ("full", "write", "resume"):
        worker_command = [*launcher, sys.executable, "-m", "mpi4py", str(worker),
                          "-uw_method", method, "-uw_phase", phase]
        command = ([sys.executable, str(supervisor), "--silence", "45",
                    "--hard-cap", "60", "--", *worker_command]
                   if supervisor.is_file() else worker_command)
        with (tmp_path / f"{phase}.log").open("w") as log:
            status = subprocess.run(command, cwd=tmp_path, env=env,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    timeout=65).returncode
        assert status == 0, (tmp_path / f"{phase}.log").read_text(errors="replace")
    maxima = {"field": 0.0, "estimate": 0.0}
    for rank in range(ranks):
        with h5py.File(tmp_path / f"full_rank{rank}.h5") as full, h5py.File(
                tmp_path / f"resume_rank{rank}.h5") as resumed:
            assert set(full) == set(resumed)
            for name in full:
                expected, actual = full[name][()], resumed[name][()]
                if name == "estimate_dt" or name == "solver_last_change_rate":
                    if isinstance(expected, bytes):
                        assert actual == expected
                        continue
                    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10, err_msg=name)
                    maxima["estimate"] = max(maxima["estimate"], float(np.max(np.abs(actual-expected))))
                elif name in ("coords", "step", "time") or name.startswith(("solver_", "history_")):
                    np.testing.assert_array_equal(actual, expected, err_msg=name)
                else:
                    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12, err_msg=name)
                    maxima["field"] = max(maxima["field"], float(np.max(np.abs(actual-expected))))
    print(f"SUPG_FRESH_RESTART method={method} ranks={ranks} "
          f"max_field_error={maxima['field']:.12g} max_estimator_error={maxima['estimate']:.12g}", flush=True)
