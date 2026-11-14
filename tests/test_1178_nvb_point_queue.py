"""The native refine keeps its work list in a queue of its own (#1140).

``nvb_transform.c`` used PETSc's ``DMPlexPointQueue``. In PETSc 3.25.3 that queue
returns values that were never enqueued, and loses entries, once an enqueue grows
it while its contents wrap its buffer. A spherical shell adapt whose closure passes
1024 edges read an edge number of -1230142544 from it and crashed, differently from
run to run (a segmentation fault, PETSc error 73, PETSc error 60).

Two contracts:
- the queue returns every value once, in the order it went in, while it is made to
  move and grow with entries outstanding;
- that shell adapt completes. It runs in a child process, because the failure it
  guards against is a crash of the interpreter.
"""
import subprocess
import sys

import pytest


_SHELL_ADAPT = r"""
import numpy as np, underworld3 as uw
def metric(c):
    d = np.abs(np.linalg.norm(np.asarray(c), axis=1) - 0.75)
    return 1.0 / np.minimum(np.sqrt(0.05**2 + (2.0 * d) ** 2), 0.3) ** 2
base = uw.meshing.SphericalShell(radiusOuter=1.0, radiusInner=0.5, cellSize=0.5, refinement=1, qdegree=2)
child = base.adapt(metric, max_levels={levels}, mg_coarsening_ratio=2.0)
count = lambda mesh: mesh.dm.getDepthStratum(0)[1] - mesh.dm.getDepthStratum(0)[0]
print("ADAPTED", count(base), count(child))
"""


@pytest.mark.level_1
@pytest.mark.tier_b
@pytest.mark.parametrize("capacity, burst, rounds", [(4, 7, 300), (1, 1, 64), (1024, 1500, 8)])
def test_the_point_queue_returns_every_value_once_and_in_order(capacity, burst, rounds):
    from underworld3.utilities import _nvb_transform

    returned, mismatches = _nvb_transform._point_queue_self_test(capacity, burst, rounds)
    assert returned == burst * rounds
    assert mismatches == 0


@pytest.mark.level_2
@pytest.mark.tier_b
@pytest.mark.parametrize("levels", [2, 3])
def test_a_spherical_shell_adapt_whose_closure_outgrows_the_queue_completes(levels):
    run = subprocess.run([sys.executable, "-c", _SHELL_ADAPT.format(levels=levels)],
                         capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stderr[-800:]
    marker = [line.split() for line in run.stdout.splitlines() if line.startswith("ADAPTED")]
    assert len(marker) == 1
    base_vertices, child_vertices = int(marker[0][1]), int(marker[0][2])
    assert child_vertices > base_vertices
