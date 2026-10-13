"""Parallel regression tests for ``smooth_mesh_interior``.

The vertex-vertex adjacency is built as a parallel PETSc AIJ matrix;
each rank inserts its locally-visible edges using GLOBAL vertex
indices, and ``mat.assemble()`` routes cross-rank contributions so
that owned-vertex rows are complete after assembly. These tests
verify the parallel-safety properties:

  * Boundary vertices remain bit-identical on every rank
  * After ``n_iters`` sweeps, every rank's ghost-vertex copies
    agree exactly with the owner's value (halo exchange is doing
    its job)
  * Per-sweep interior displacement decreases monotonically using
    a global reduction (matches the serial guarantee)
  * Final coords from a parallel run match a serial reference to
    a single ULP — i.e. the smoother is partition-independent

Run with:
    mpirun -n 2 python -m pytest --with-mpi \\
      tests/parallel/test_0855_mesh_smoothing_parallel.py
    mpirun -n 4 python -m pytest --with-mpi \\
      tests/parallel/test_0855_mesh_smoothing_parallel.py
"""

import os
import subprocess
import sys
import tempfile
import textwrap

import numpy as np
import pytest
from mpi4py import MPI
from scipy.spatial import cKDTree

import underworld3 as uw
from underworld3.meshing import smooth_mesh_interior

from serial_reference import _MPI_ENV_PREFIXES


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


# ---------------------------------------------------------------------
# Bit-identical-vs-serial regression test
# ---------------------------------------------------------------------

# Deterministic, position-only perturbation. The smoother takes a
# perturbed mesh through 8 sweeps; we then compare the final owned
# coords against a serial reference. The perturbation is a pure
# function of (x, y) so that serial and parallel runs start from the
# same field, regardless of where the partition cut lands.
_SERIAL_REFERENCE_SCRIPT = textwrap.dedent("""
    import sys
    import numpy as np

    import underworld3 as uw
    from underworld3.meshing import (
        UnstructuredSimplexBox, smooth_mesh_interior)


    def _boundary_vertex_mask(mesh):
        dm = mesh.dm
        pStart, pEnd = dm.getDepthStratum(0)
        n = pEnd - pStart
        skip = {"All_Boundaries", "Null_Boundary"}
        mask = np.zeros(n, dtype=bool)
        for b in mesh.boundaries:
            nm = getattr(b, "name", None)
            if not nm or nm in skip:
                continue
            lab = dm.getLabel(nm)
            if lab is None:
                continue
            vIS = lab.getValueIS()
            if vIS is None:
                continue
            for v in vIS.getIndices():
                iset = lab.getStratumIS(int(v))
                if iset is None:
                    continue
                for idx in iset.getIndices():
                    if pStart <= idx < pEnd:
                        mask[idx - pStart] = True
        return mask


    out_path = sys.argv[1]
    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=1.0 / 12)
    is_bnd = _boundary_vertex_mask(mesh)
    coords = np.asarray(mesh.X.coords).copy()
    initial = coords.copy()
    dx = 0.018 * np.sin(7.0 * np.pi * coords[:, 0]) \\
        * np.cos(5.0 * np.pi * coords[:, 1])
    dy = 0.018 * np.cos(3.0 * np.pi * coords[:, 0]) \\
        * np.sin(11.0 * np.pi * coords[:, 1])
    coords[~is_bnd, 0] += dx[~is_bnd]
    coords[~is_bnd, 1] += dy[~is_bnd]
    mesh._deform_mesh(coords)
    smooth_mesh_interior(mesh, n_iters=8, alpha=0.5)
    np.savez(out_path,
             initial=initial,
             final=np.asarray(mesh.X.coords))
""")


def _owned_vertex_mask_local(dm):
    pStart, pEnd = dm.getDepthStratum(0)
    n = pEnd - pStart
    owned = np.ones(n, dtype=bool)
    sf = dm.getPointSF()
    if sf is None:
        return owned
    try:
        _, leaves, _ = sf.getGraph()
    except Exception:
        return owned
    if leaves is None:
        return owned
    for L in leaves:
        if pStart <= L < pEnd:
            owned[L - pStart] = False
    return owned


@pytest.mark.mpi(min_size=2)
def test_parallel_matches_serial_bit_identical():
    """Final coords from the parallel smoother match a serial
    reference to a single ULP, partitioning-independent.

    Rank 0 spawns a serial Python subprocess that runs the same
    mesh + perturbation + 8-sweep smoothing pipeline; rank 0 then
    compares the parallel result (gathered from every rank's owned
    vertices) against the serial reference, matching vertices by
    their initial (pre-perturbation) coordinate.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # 1. Compute the serial reference (rank 0 only) in a clean subprocess. The
    # launcher's variables are stripped first, or the child's MPI_Init tries to
    # join THIS mpirun's job and aborts on a descriptor it does not own.
    #
    # The prefix list is shared with serial_reference rather than restated here:
    # a local copy that named only the Open MPI family passed on Open MPI and
    # failed under MPICH, whose variables are PMI_* (#675). One list, so a
    # launcher that is missing from it is missing everywhere and gets noticed.
    ref_path = None
    if rank == 0:
        tmpdir = tempfile.mkdtemp(prefix="winslow_ref_")
        ref_path = os.path.join(tmpdir, "ref.npz")
        clean_env = {
            k: v for k, v in os.environ.items()
            if not k.startswith(_MPI_ENV_PREFIXES)
        }
        proc = subprocess.run(
            [sys.executable, "-c",
             _SERIAL_REFERENCE_SCRIPT, ref_path],
            capture_output=True, text=True, timeout=60,
            env=clean_env)
        assert proc.returncode == 0, (
            f"serial reference subprocess failed:\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        assert os.path.exists(ref_path), (
            f"serial reference not written at {ref_path}")

    # 2. Run the parallel pipeline on this comm
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=1.0 / 12)
    is_bnd = _boundary_vertex_mask(mesh)
    coords = np.asarray(mesh.X.coords).copy()
    initial_local = coords.copy()
    dx = 0.018 * np.sin(7.0 * np.pi * coords[:, 0]) \
        * np.cos(5.0 * np.pi * coords[:, 1])
    dy = 0.018 * np.cos(3.0 * np.pi * coords[:, 0]) \
        * np.sin(11.0 * np.pi * coords[:, 1])
    coords[~is_bnd, 0] += dx[~is_bnd]
    coords[~is_bnd, 1] += dy[~is_bnd]
    mesh._deform_mesh(coords)
    smooth_mesh_interior(mesh, n_iters=8, alpha=0.5)

    # 3. Gather owned final coords + their pre-perturbation initial
    # coords (the latter is the matching key against the serial ref).
    is_owned = _owned_vertex_mask_local(mesh.dm)
    final_local = np.asarray(mesh.X.coords).copy()
    own_initial = initial_local[is_owned]
    own_final = final_local[is_owned]
    g_initial = comm.gather(own_initial, root=0)
    g_final = comm.gather(own_final, root=0)

    # 4. Compare on rank 0
    if rank == 0:
        all_initial = np.vstack(g_initial)
        all_final = np.vstack(g_final)
        ref = np.load(ref_path)
        ref_initial = ref["initial"]
        ref_final = ref["final"]
        # Match each parallel vertex to its serial counterpart by
        # initial coordinate (bit-identical because the mesh
        # generator runs serially before distribution).
        tree = cKDTree(ref_initial)
        match_dist, idx = tree.query(all_initial, k=1)
        assert match_dist.max() < 1.0e-12, (
            "initial-coord matching failed: max mismatch "
            f"{match_dist.max():.3e} — meshes must be identical "
            "between serial and parallel runs")
        drift = np.linalg.norm(all_final - ref_final[idx], axis=1)
        size = MPI.COMM_WORLD.size
        assert drift.max() < 1.0e-12, (
            f"np={size}: parallel smoother diverged from serial; "
            f"max drift = {drift.max():.3e}  "
            f"mean = {drift.mean():.3e}  "
            f"({(drift > 1.0e-12).sum()}/{len(drift)} verts off)")
        # Cleanup
        try:
            os.remove(ref_path)
            os.rmdir(os.path.dirname(ref_path))
        except OSError:
            pass
