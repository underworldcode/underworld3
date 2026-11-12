"""Prolongation from a submesh reaches every parent DOF it covers (#1087).

``Mesh.prolongate`` maps parent DOFs to submesh DOFs by coordinate, rank by
rank. A parent DOF on the region's edge can be owned by a rank whose cells
around it all lie outside the region; that rank has no submesh DOF there.
Each rank used to write its matched rows and sync, which keeps the owner's
value, so those DOFs kept their old value: on ``AnnulusInternalBoundary``
with ``extract_region("Inner")`` at np 3, two DOFs on r = 1 came back as 0
(error 1.0). The partitions CI runs (np 2, np 4) happen to miss it, so this
test launches np 3 and np 5 itself, in subprocesses through the MPI
supervisor (this file is also the worker: ``python -m mpi4py <this file>
<out.json>``).

A P2 vector field ``(r, -r)`` on the submesh is prolongated into a zeroed
parent field, in ``replace`` mode and then in ``add`` mode onto a field of
ones. Every parent DOF in the region must take the submesh value (plus one
in ``add`` mode), and every DOF outside it must keep its old value.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def _prolongate(out):
    import underworld3 as uw
    from mpi4py import MPI

    full = uw.meshing.AnnulusInternalBoundary(radiusOuter=1.5, radiusInternal=1.0,
                                              radiusInner=0.5, cellSize=1.0 / 8.0)
    rock = full.extract_region("Inner")
    v_full = uw.discretisation.MeshVariable("Vf", full, full.dim, degree=2)
    v_rock = uw.discretisation.MeshVariable("Vr", rock, rock.dim, degree=2)
    r_rock = np.hypot(*np.asarray(v_rock.coords).T)
    v_rock.data[:, 0] = r_rock
    v_rock.data[:, 1] = -r_rock

    r = np.hypot(*np.asarray(v_full.coords).T)
    inside = r < 1.0 + 1.0e-6
    errors = {}
    for mode, start in (("replace", 0.0), ("add", 1.0)):
        v_full.data[:] = start
        rock.prolongate(v_rock, v_full, mode=mode)
        data = np.asarray(v_full.data)
        expect = np.where(inside[:, None], start + np.column_stack([r, -r]), start)
        err = float(np.abs(data - expect).max()) if len(data) else 0.0
        errors[mode] = uw.mpi.comm.allreduce(err, op=MPI.MAX)
    if uw.mpi.rank == 0:
        Path(out).write_text(json.dumps(errors))


pytestmark = [pytest.mark.level_2, pytest.mark.timeout(600)]


@pytest.mark.tier_b
@pytest.mark.skipif(not Path(sys.executable).with_name("mpirun").exists(),
                    reason="needs mpirun beside the Python interpreter")
@pytest.mark.parametrize("ranks", [3, 5])
def test_prolongate_reaches_dofs_owned_outside_the_region(ranks, tmp_path):
    import underworld3 as uw

    if uw.mpi.size != 1:
        pytest.skip("Run in serial; it launches its own parallel run.")
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "errors.json"
    command = [sys.executable, str(root / "scripts/mpi_supervisor.py"), "--silence", "120", "--",
               str(Path(sys.executable).with_name("mpirun")), "-np", str(ranks),
               sys.executable, "-m", "mpi4py", str(Path(__file__).resolve()), str(out)]
    log = tmp_path / "run.log"
    with log.open("w") as fh:
        result = subprocess.run(command, cwd=tmp_path, stdout=fh, stderr=subprocess.STDOUT,
                                timeout=300)
    assert result.returncode == 0, log.read_text()[-4000:]
    errors = json.loads(out.read_text())
    assert errors["replace"] < 1e-12, errors
    assert errors["add"] < 1e-12, errors


if __name__ == "__main__":
    _prolongate(sys.argv[1])
