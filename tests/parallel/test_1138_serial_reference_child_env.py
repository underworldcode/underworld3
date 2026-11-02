"""The serial reference child keeps the user's MPI component selections (#922).

``serial_reference`` runs a test module's np=1 diagnostic in a spawned Python, and
scrubs the launcher's variables from the child's environment so that the child
runs as a singleton instead of trying to join the parent's job. The scrub dropped
every ``OMPI_*`` variable, which also dropped the user's own Open MPI component
selections. On kaiju, jobs set ``OMPI_MCA_io=^ompio`` to keep HDF5 off OMPIO, and
under load every HDF5 error 76 was in a spawned reference run that never saw it.

The child now keeps ``OMPI_MCA_io``, ``OMPI_MCA_mtl``, ``OMPI_MCA_btl`` and
``OMPI_MCA_pml``, and still drops the job-identity variables. The first test
checks the scrub; the second spawns a child through the same launch path and asks
it what it received.

This file carries no ``mpi(min_size=2)`` mark: under mpirun the launcher's own
variables are set too, which is the case the scrub exists for.
"""
import json

import pytest

from serial_reference import _run_child

pytestmark = [pytest.mark.level_1]

USER_SELECTIONS = {"OMPI_MCA_io": "^ompio", "OMPI_MCA_mtl": "^ofi",
                   "OMPI_MCA_btl": "self,vader", "OMPI_MCA_pml": "ob1"}
JOB_IDENTITY = {"OMPI_COMM_WORLD_RANK": "0", "OMPI_MCA_ess_base_jobid": "123",
                "OMPI_MCA_orte_hnp_uri": "x", "PMIX_RANK": "0", "PMI_RANK": "0",
                "SLURM_JOB_ID": "9", "PRTE_MCA_x": "1", "OPAL_PREFIX": "/o",
                "HYDRA_CONTROL_FD": "3", "I_MPI_PIN": "1", "MPI_LOCALRANKID": "0"}


@pytest.mark.tier_b
def test_the_child_env_keeps_selections_and_drops_job_identity():
    from serial_reference import _child_env
    env = _child_env({**USER_SELECTIONS, **JOB_IDENTITY, "PATH": "/bin", "HOME": "/h"})
    for name, value in USER_SELECTIONS.items():
        assert env.get(name) == value, f"{name} was not passed to the child"
    leaked = sorted(set(JOB_IDENTITY) & set(env))
    assert not leaked, f"launcher variables passed to the child: {leaked}"
    assert env["PATH"] == "/bin" and env["HOME"] == "/h"


CHILD = '''
import json, os, sys
print("SERIALREF " + json.dumps({name: os.environ.get(name) for name in sys.argv[1].split(",")}))
'''


@pytest.mark.tier_b
def test_a_spawned_child_receives_the_parents_io_selection(tmp_path, monkeypatch):
    for name, value in {**USER_SELECTIONS, "PMIX_RANK": "0"}.items():
        monkeypatch.setenv(name, value)
    child = tmp_path / "child.py"
    child.write_text(CHILD)
    names = ",".join([*USER_SELECTIONS, "PMIX_RANK"])
    payload = _run_child(str(child), names, timeout=60)
    assert isinstance(payload, dict), payload
    for name, value in USER_SELECTIONS.items():
        assert payload[name] == value, f"the child saw {name}={payload[name]!r}"
    assert payload["PMIX_RANK"] is None, json.dumps(payload)
