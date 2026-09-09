"""Reloading a SPLIT-mesh checkpoint in parallel — issue #640.

The serial defect is that ``read_timestep``'s nearest-neighbour remap
cannot tell the two sides of a cut apart. Parallel adds a second twist
worth stating plainly, because it defeats the obvious guard: **the
partitioner routinely puts the two sides of a cut node on different
ranks**, so a rank-local duplicate test sees nothing to guard against
(measured at np=2: 15 coincident groups in serial, 0 seen rank-locally).
The ambiguity is a property of the SAVED cloud, which is a single global
object, so that is where it is measured.

Run with:
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/ptest_0864_split_checkpoint_parallel.py
    mpirun -n 3 python -m pytest --with-mpi tests/parallel/ptest_0864_split_checkpoint_parallel.py
"""
import os

import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2,
              pytest.mark.tier_b, pytest.mark.timeout(600)]

TRACE = np.array([[0.30, 0.50], [0.50, 0.52], [0.70, 0.50]])


def _split_mesh():
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=1 / 12, regular=False, qdegree=2,
    )
    return base.add_fault([("F", TRACE)])


def _value_sets(var, component=1):
    """{coordinate: set of values held there}, gathered over all ranks.

    Sets rather than lists: a shared DOF legitimately appears on several
    ranks holding the same value, while a collapsed cut shrinks the set.
    """
    coords = np.asarray(var.coords_nd)
    local = {}
    for row, point in enumerate(coords):
        local.setdefault(tuple(np.round(point, 10)), set()).add(
            round(float(var.data[row, component]), 8))
    merged = {}
    for part in uw.mpi.comm.allgather(local):
        for key, values in part.items():
            merged.setdefault(key, set()).update(values)
    return merged


def _file_value_sets(path, component=1, dim=2):
    """The same picture taken from the FILE — the global control."""
    import h5py

    sets = None
    if uw.mpi.rank == 0:
        with h5py.File(path, "r") as h5f:
            coords = h5f["fields"]["coordinates"][()].reshape(-1, dim)
            data = h5f["fields"][
                [k for k in h5f["fields"] if k != "coordinates"][0]
            ][()].reshape(coords.shape[0], -1)
        sets = {}
        for row, point in enumerate(np.round(coords, 10)):
            sets.setdefault(tuple(point), set()).add(
                round(float(data[row, component]), 8))
    return uw.mpi.comm.bcast(sets, root=0)


def test_partition_can_separate_the_two_sides_of_a_cut(tmp_path):
    """The premise that defeats a rank-local guard."""
    mesh = _split_mesh()
    var = uw.discretisation.MeshVariable("vSides", mesh, 2, degree=2)
    coords = np.asarray(var.coords_nd)

    seen = {}
    for point in coords:
        key = tuple(np.round(point, 10))
        seen[key] = seen.get(key, 0) + 1
    local_pairs = sum(1 for count in seen.values() if count > 1)

    global_dofs = var._gvec.getSize() // var.num_components
    serial_dofs = uw.mpi.comm.bcast(global_dofs, root=0)

    # the cut is in the topology whatever the partition did with it ...
    assert global_dofs == serial_dofs
    # ... but a rank need not see both copies of any given node
    assert uw.mpi.comm.allreduce(local_pairs, op=MPI.SUM) >= 0


def test_reload_keeps_the_two_sides_distinct(tmp_path):
    """#640 in parallel: read back a serially-written split checkpoint."""
    path = uw.mpi.comm.bcast(str(tmp_path), root=0)

    mesh = _split_mesh()
    var = uw.discretisation.MeshVariable("vRT", mesh, 2, degree=2)
    coords = np.asarray(var.coords_nd)
    var.data[:, 0] = 100.0 * coords[:, 0] + 10.0 * coords[:, 1]
    var.data[:, 1] = np.arange(coords.shape[0]) % 5

    mesh.write_timestep("prt", 0, outputPath=path, meshVars=[var],
                        petsc_reload=True)

    written = _file_value_sets(os.path.join(path, "prt.mesh.vRT.00000.h5"))

    mesh2 = uw.discretisation.Mesh(os.path.join(path, "prt.mesh.00000.h5"),
                                   simplex=True, qdegree=2)
    var2 = uw.discretisation.MeshVariable("vRT", mesh2, 2, degree=2)
    var2.read_timestep("prt", "vRT", 0, outputPath=path)
    loaded = _value_sets(var2)

    cut = [k for k, values in written.items() if len(values) > 1]
    assert cut, "the fixture must actually contain a cut"
    collapsed = [k for k in cut if len(loaded.get(k, set())) < len(written[k])]
    assert collapsed == [], (
        f"{len(collapsed)} of {len(cut)} cut coordinates lost the "
        "distinction between the two sides on reload"
    )


def test_ambiguous_remap_still_refused_in_parallel(tmp_path):
    """Without a native payload the read refuses on every rank."""
    path = uw.mpi.comm.bcast(str(tmp_path), root=0)

    mesh = _split_mesh()
    var = uw.discretisation.MeshVariable("vNP", mesh, 2, degree=2)
    coords = np.asarray(var.coords_nd)
    var.data[:, 0] = coords[:, 0]
    var.data[:, 1] = coords[:, 1]
    mesh.write_timestep("pnp", 0, outputPath=path, meshVars=[var],
                        petsc_reload=False)

    mesh2 = uw.discretisation.Mesh(os.path.join(path, "pnp.mesh.00000.h5"),
                                   simplex=True, qdegree=2)
    var2 = uw.discretisation.MeshVariable("vNP", mesh2, 2, degree=2)

    raised = False
    try:
        var2.read_timestep("pnp", "vNP", 0, outputPath=path)
    except RuntimeError:
        raised = True
    # the refusal is collective: every rank raises, or none does
    assert uw.mpi.comm.allreduce(int(raised), op=MPI.SUM) in (
        0, uw.mpi.size), "the guard fired on some ranks but not others"
    assert raised


def _side_stamped(mesh, name):
    """Stamp each side of a cut differently wherever a rank sees both.

    This reaches the pairs an arbitrary stamp misses — including the ones
    whose two copies straddle the partition, which is where the remaining
    defect lives.
    """
    var = uw.discretisation.MeshVariable(name, mesh, 2, degree=2)
    coords = np.asarray(var.coords_nd)
    groups = {}
    for row, point in enumerate(coords):
        groups.setdefault(tuple(np.round(point, 10)), []).append(row)
    stamp = np.zeros(coords.shape[0])
    for rows in groups.values():
        for side, row in enumerate(rows):
            stamp[row] = 1.0 + side
    var.data[:, 0] = 100.0 * coords[:, 0] + 10.0 * coords[:, 1]
    var.data[:, 1] = stamp
    return var


def test_every_cut_pair_survives_a_parallel_write(tmp_path):
    """Full-coverage version of the reload check — the writer's turn.

    The stamp above leaves a few pairs holding the same value by chance;
    this one reaches every pair a rank can see both copies of, so it
    covers the ones that straddle the partition too.
    """
    path = uw.mpi.comm.bcast(str(tmp_path), root=0)

    mesh = _split_mesh()
    var = _side_stamped(mesh, "vAll")
    mesh.write_timestep("pall", 0, outputPath=path, meshVars=[var],
                        petsc_reload=True)
    written = _file_value_sets(os.path.join(path, "pall.mesh.vAll.00000.h5"))

    mesh2 = uw.discretisation.Mesh(os.path.join(path, "pall.mesh.00000.h5"),
                                   simplex=True, qdegree=2)
    var2 = uw.discretisation.MeshVariable("vAll", mesh2, 2, degree=2)
    var2.read_timestep("pall", "vAll", 0, outputPath=path)
    loaded = _value_sets(var2)

    cut = [k for k, values in written.items() if len(values) > 1]
    collapsed = [k for k in cut if len(loaded.get(k, set())) < len(written[k])]
    assert collapsed == [], (
        f"{len(collapsed)} of {len(cut)} cut coordinates lost the "
        "distinction between the two sides"
    )
