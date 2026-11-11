"""The MMPDE mover runs when some ranks hold no cells (#1085).

Every step of ``_mmpde_mover`` after it builds its cell list is collective.
It used to return ``[]`` on a rank whose cell list came back ``None`` while
the other ranks went on into the reductions, so a partition with an empty
rank aborted the job ("Message truncated"). ``UnstructuredSimplexBox`` with
``cellSize=1`` has four cells, and at np 4 the partition is ``[2, 0, 2, 0]``.

Checked here:

* with some ranks empty, the mover completes on every rank, every rank
  returns the same energy history, the energy does not increase, and no
  cell is inverted;
* on a quadrilateral mesh, where the mover does not apply, every rank
  returns ``[]`` together.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw
from mpi4py import MPI
from underworld3.meshing.smoothing import _mmpde_mover

pytestmark = [pytest.mark.level_2, pytest.mark.mpi(min_size=2), pytest.mark.timeout(120)]

BOUNDARIES = ("Top", "Bottom", "Left", "Right")


def _metric(mesh):
    x, y = mesh.X
    return 1.0 + 4.0 * sympy.exp(-((y - 0.5) / 0.2) ** 2)


@pytest.mark.tier_b
def test_mover_completes_with_empty_ranks():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=1.0)
    cs, ce = mesh.dm.getHeightStratum(0)
    counts = uw.mpi.comm.allgather(ce - cs)
    if min(counts) > 0:
        pytest.skip(f"no rank is empty at np {uw.mpi.size} (cells per rank {counts})")

    energies = _mmpde_mover(mesh, _metric(mesh), BOUNDARIES, False, n_outer=2,
                            metric_eval="sympy")

    every = uw.mpi.comm.allgather([float(e) for e in energies])
    assert all(e == every[0] for e in every), every
    assert len(energies) > 0
    assert np.all(np.diff(np.asarray(energies, dtype=float)) <= 1e-12 * abs(float(energies[0])))

    # No inverted cell, on any rank: the signed areas share one sign.
    coords = np.asarray(mesh.X.coords)
    areas = []
    pStart = mesh.dm.getDepthStratum(0)[0]
    for c in range(cs, ce):
        closure = mesh.dm.getTransitiveClosure(c)[0]
        v = [p - pStart for p in closure if pStart <= p < mesh.dm.getDepthStratum(0)[1]]
        a, b, d = coords[v]
        areas.append((b[0] - a[0]) * (d[1] - a[1]) - (d[0] - a[0]) * (b[1] - a[1]))
    areas = np.asarray(areas)
    positive = uw.mpi.comm.allreduce(int((areas > 0).sum()), op=MPI.SUM)
    negative = uw.mpi.comm.allreduce(int((areas < 0).sum()), op=MPI.SUM)
    assert positive == 0 or negative == 0
    assert positive + negative == sum(counts)


@pytest.mark.tier_b
def test_mover_returns_together_on_quads():
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4), minCoords=(0.0, 0.0),
                                        maxCoords=(1.0, 1.0))
    energies = _mmpde_mover(mesh, _metric(mesh), BOUNDARIES, False, n_outer=2,
                            metric_eval="sympy")
    assert uw.mpi.comm.allgather(list(energies)) == [[]] * uw.mpi.size
