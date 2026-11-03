"""The serial reference child keeps the user's MPI component selections (#922).

``serial_reference`` runs a test module's np=1 diagnostic in a spawned Python, and
scrubs the launcher's variables from the child's environment so that the child
runs as a singleton instead of trying to join the parent's job. The scrub dropped
every ``OMPI_*`` variable, which also dropped the user's own Open MPI component
selections. On kaiju, jobs set ``OMPI_MCA_io=^ompio`` to keep HDF5 off OMPIO, and
under load every HDF5 error 76 was in a spawned reference run that never saw it.

The child now keeps the user's selections and parameters for io (and its
sub-frameworks), mtl, btl and pml, and still drops the launcher's variables. A
selection is kept with its parameters: ``pml=cm`` without its mtl's provider
settings, or ``pml=ucx`` without its UCX settings, can make the child's MPI_Init
abort where the parent's succeeded. The first test checks the scrub. The second
spawns a child through the same launch path; the child initialises MPI, reports
its world size and every launcher-prefixed variable it received.

This file carries no ``mpi(min_size=2)`` mark: under mpirun the launcher's own
variables are set too, which is the case the scrub exists for.
"""
import pytest

from serial_reference import _MPI_ENV_PREFIXES, _kept_mpi_setting, _run_child

pytestmark = [pytest.mark.level_1]

USER_SELECTIONS = {"OMPI_MCA_io": "^ompio", "OMPI_MCA_mtl": "^ofi",
                   "OMPI_MCA_mtl_ofi_provider_include": "tcp",
                   "OMPI_MCA_btl": "self,vader", "OMPI_MCA_pml": "ob1"}
LAUNCHER_VARIABLES = {"OMPI_COMM_WORLD_RANK": "0", "OMPI_MCA_ess_base_jobid": "123",
                      "OMPI_MCA_orte_hnp_uri": "x", "PMIX_RANK": "0", "PMI_RANK": "0",
                      "SLURM_JOB_ID": "9", "HYDRA_CONTROL_FD": "3",
                      "MPI_LOCALRANKID": "0"}


@pytest.mark.tier_b
def test_the_child_env_keeps_selections_and_drops_job_identity():
    from serial_reference import _child_env
    env = _child_env({**USER_SELECTIONS, **LAUNCHER_VARIABLES, "PATH": "/bin", "HOME": "/h"})
    for name, value in USER_SELECTIONS.items():
        assert env.get(name) == value, f"{name} was not passed to the child"
    leaked = sorted(set(LAUNCHER_VARIABLES) & set(env))
    assert not leaked, f"launcher variables passed to the child: {leaked}"
    assert env["PATH"] == "/bin" and env["HOME"] == "/h"


CHILD = '''
import json, os, sys
from mpi4py import MPI
prefixes = tuple(sys.argv[1].split(","))
print("SERIALREF " + json.dumps({
    "size": MPI.COMM_WORLD.Get_size(),
    "mpi_env": {k: v for k, v in os.environ.items() if k.startswith(prefixes)}}))
'''


@pytest.mark.tier_b
def test_a_spawned_child_receives_the_parents_io_selection(tmp_path, monkeypatch):
    for name, value in {**USER_SELECTIONS, "PMIX_RANK": "0"}.items():
        monkeypatch.setenv(name, value)
    child = tmp_path / "child.py"
    child.write_text(CHILD)
    payload = _run_child(str(child), ",".join(_MPI_ENV_PREFIXES), timeout=120)
    assert isinstance(payload, dict), payload
    assert payload["size"] == 1, f"the child is not a singleton: {payload}"
    received = payload["mpi_env"]
    for name, value in USER_SELECTIONS.items():
        assert received.get(name) == value, f"the child saw {name}={received.get(name)!r}"
    # under mpirun the parent carries the launcher's own variables too: none of
    # them may reach the child, only the kept framework settings
    unexpected = sorted(k for k in received if not _kept_mpi_setting(k))
    assert not unexpected, f"launcher variables reached the child: {unexpected}"
