"""MPI tests for ``smooth_surface_field`` on a codim-1 surface submesh.

The graph low-pass is built on the parallel vertex-vertex adjacency
(``A.mult``) + owned-vertex map + ``globalToLocal``, so it must be
bit-identical serial vs parallel and preserve the constant mode exactly on
any rank count. Also exercises ``extract_surface`` in parallel.
"""

import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw
from underworld3 import mpi

pytestmark = [
    pytest.mark.level_2,
    pytest.mark.tier_b,
    pytest.mark.mpi(min_size=2),
    pytest.mark.timeout(180),
]


def _gather_unique(coords, vals):
    """Gather (coord-keyed) values to rank 0, de-duplicating shared/ghost
    vertices by rounded coordinate. Returns (keys_sorted, vals) on rank 0,
    else (None, None)."""
    loc = np.column_stack(
        [np.round(coords[:, 0], 9), np.round(coords[:, 1], 9), vals])
    allrows = mpi.comm.gather(loc, root=0)
    if mpi.rank != 0:
        return None, None
    d = {}
    for x, y, v in np.vstack(allrows):
        d[(x, y)] = v
    keys = sorted(d.keys())
    return keys, np.array([d[k] for k in keys])


def test_surface_extract_and_smooth_parallel():
    ann = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.06)
    surf = ann.extract_surface("Upper")
    assert surf.dm.getDimension() == 1
    assert surf.parent is ann

    h = uw.discretisation.MeshVariable("h_par", surf, 1, degree=1)
    c = np.asarray(h.coords)
    th = np.arctan2(c[:, 1], c[:, 0])
    h.data[:, 0] = np.cos(2 * th) + 0.3 * np.cos(20 * th)

    uw.meshing.smooth_surface_field(h, n_iters=40, alpha=0.6, taubin=True)

    keys, vals = _gather_unique(c, h.data[:, 0])
    if mpi.rank == 0:
        thg = np.arctan2(
            np.array([k[1] for k in keys]), np.array([k[0] for k in keys]))
        # low (signal) mode preserved, high (sawtooth) mode attenuated
        low = (vals * np.cos(2 * thg)).mean() * 2
        high = (vals * np.cos(20 * thg)).mean() * 2
        assert low > 0.9, f"low mode lost in parallel: {low:.3f}"
        assert abs(high) < 0.2, f"high mode not attenuated in parallel: {high:.3f}"


def test_constant_preserved_parallel():
    shell = uw.meshing.SphericalShell(
        radiusOuter=1.0, radiusInner=0.5, cellSize=0.2)
    surf = shell.extract_surface("Upper")           # 2-manifold in parallel
    h = uw.discretisation.MeshVariable("hc_par", surf, 1, degree=1)
    h.data[:, 0] = 0.37
    uw.meshing.smooth_surface_field(h, n_iters=30, alpha=0.6, taubin=True)
    # constant mode is invariant on every rank (owned + ghost)
    local_max_dev = float(np.abs(h.data[:, 0] - 0.37).max()) if h.data.size else 0.0
    global_max_dev = mpi.comm.allreduce(local_max_dev, op=MPI.MAX)
    assert global_max_dev < 1.0e-12
