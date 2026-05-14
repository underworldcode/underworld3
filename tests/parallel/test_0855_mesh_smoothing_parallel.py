"""Parallel regression tests for ``smooth_mesh_interior``.

The smoother runs a local scipy Mat-Vec per sweep followed by a
PETSc halo exchange (``coordDM.localToGlobal`` INSERT, then
``globalToLocal``). These tests verify the parallel-safety
properties:

  * Boundary vertices remain bit-identical on every rank
  * After ``n_iters`` sweeps, every rank's ghost-vertex copies
    agree exactly with the owner's value (halo exchange is doing
    its job)
  * Per-sweep interior displacement decreases monotonically using
    a global reduction (matches the serial guarantee)

Run with:
    mpirun -n 2 python -m pytest --with-mpi \\
      tests/parallel/test_0855_mesh_smoothing_parallel.py
    mpirun -n 4 python -m pytest --with-mpi \\
      tests/parallel/test_0855_mesh_smoothing_parallel.py
"""

import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw
from underworld3.meshing import smooth_mesh_interior


pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.timeout(120)]


def _box_mesh(resolution: int = 12):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=1.0 / resolution,
    )


def _boundary_vertex_mask(mesh):
    dm = mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)
    n_verts = pEnd - pStart
    skip = {"All_Boundaries", "Null_Boundary"}
    mask = np.zeros(n_verts, dtype=bool)
    for member in mesh.boundaries:
        name = getattr(member, "name", None)
        if not name or name in skip:
            continue
        label = dm.getLabel(name)
        if label is None:
            continue
        vIS = label.getValueIS()
        if vIS is None:
            continue
        for val in vIS.getIndices():
            iset = label.getStratumIS(int(val))
            if iset is None:
                continue
            for idx in iset.getIndices():
                if pStart <= idx < pEnd:
                    mask[idx - pStart] = True
    return mask


@pytest.mark.mpi(min_size=2)
def test_parallel_boundary_pinned():
    """Boundary vertices stay bit-identical on every rank."""
    mesh = _box_mesh(resolution=12)
    is_bnd = _boundary_vertex_mask(mesh)
    before = np.asarray(mesh.X.coords).copy()
    smooth_mesh_interior(mesh, n_iters=5, alpha=0.5)
    after = np.asarray(mesh.X.coords)
    if int(is_bnd.sum()) > 0:
        assert np.allclose(
            before[is_bnd], after[is_bnd], rtol=0, atol=0), (
            f"Rank {uw.mpi.rank}: boundary vertices moved.")


@pytest.mark.mpi(min_size=2)
def test_parallel_ghosts_agree_with_owners():
    """After smoothing, every ghost vertex's local coord must
    match the owner's. Verifies the halo exchange did its job."""
    mesh = _box_mesh(resolution=12)
    rng = np.random.default_rng(7 + uw.mpi.rank)
    coords = np.asarray(mesh.X.coords).copy()
    is_bnd = _boundary_vertex_mask(mesh)
    if (~is_bnd).any():
        coords[~is_bnd] += 0.01 * rng.standard_normal(
            (int((~is_bnd).sum()), coords.shape[1]))
        mesh._deform_mesh(coords)
    smooth_mesh_interior(mesh, n_iters=5, alpha=0.5)
    # After the call, the DM's coordinate vector has been updated.
    # We verify ghost-owner agreement by doing a fresh halo exchange:
    # a properly-consistent state is invariant under another
    # globalToLocal — i.e. ghost values are unchanged after refresh.
    dm = mesh.dm
    coord_dm = dm.getCoordinateDM()
    local_vec = dm.getCoordinatesLocal()
    global_vec = dm.getCoordinates()
    before = np.asarray(local_vec.array).copy()
    coord_dm.globalToLocal(global_vec, local_vec)
    after = np.asarray(local_vec.array)
    assert np.allclose(before, after, rtol=0, atol=1.0e-15), (
        f"Rank {uw.mpi.rank}: ghost-owner disagreement of "
        f"max |Δ| = {np.abs(before - after).max():.3e} after "
        f"smoothing — halo exchange did not propagate correctly")


@pytest.mark.mpi(min_size=2)
def test_parallel_sweep_displacement_decreases():
    """Global per-sweep displacement decreases, same as the
    serial guarantee."""
    mesh = _box_mesh(resolution=12)
    rng = np.random.default_rng(13 + uw.mpi.rank)
    coords = np.asarray(mesh.X.coords).copy()
    is_bnd = _boundary_vertex_mask(mesh)
    if (~is_bnd).any():
        coords[~is_bnd] += 0.02 * rng.standard_normal(
            (int((~is_bnd).sum()), coords.shape[1]))
        mesh._deform_mesh(coords)
    comm = MPI.COMM_WORLD
    prev = np.asarray(mesh.X.coords).copy()
    disps = []
    for _ in range(4):
        smooth_mesh_interior(mesh, n_iters=1, alpha=0.5)
        now = np.asarray(mesh.X.coords)
        is_bnd_now = _boundary_vertex_mask(mesh)
        is_int = ~is_bnd_now
        local_sq = (
            float(np.linalg.norm((now - prev)[is_int]) ** 2)
            if is_int.any() else 0.0)
        global_disp = comm.allreduce(local_sq) ** 0.5
        disps.append(global_disp)
        prev = now.copy()
    for k in range(1, len(disps)):
        assert disps[k] <= disps[k - 1] * 1.001, (
            f"Sweep {k+1} global displacement {disps[k]:.3e} > "
            f"{disps[k-1]:.3e} (sweep {k}); series: {disps}")
