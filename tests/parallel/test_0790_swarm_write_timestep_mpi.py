"""MPI regression test for swarm.write_timestep deadlock (issue #151).

The bug: ``Swarm.save()`` and ``SwarmVariable.save()`` parallel paths called
``h5f.create_dataset(name, data=local_data)`` collectively, but each rank
passed its own local-sized array. In parallel HDF5 every rank must specify
the *same* dataset shape on a collective ``create_dataset``; passing
different shapes leaves HDF5's internal metadata inconsistent so the
collective close never synchronises, producing a silent hang.

The fix uses ``allgather`` of per-rank sizes, creates the dataset at the
global shape, then has each rank write its slice.

This test runs bknight1's reproducer from #151 and fails fast via the
pytest timeout rather than blocking indefinitely.
"""

import os
import pytest

import underworld3 as uw
from petsc4py import PETSc


pytestmark = [
    pytest.mark.level_2,
    pytest.mark.mpi(min_size=2),
    pytest.mark.timeout(60),
]


@pytest.mark.mpi(min_size=2)
def test_swarm_write_timestep_parallel(tmp_path_factory):
    """swarm.write_timestep() must complete under MPI without deadlock.

    Pre-fix at np>=2 with unequal per-rank particle counts (which is the
    common case), the parallel HDF5 collective close hung indefinitely.
    """
    if uw.mpi.rank == 0:
        out_dir = tmp_path_factory.mktemp("swarm_io_151")
    else:
        out_dir = None
    out_dir = uw.mpi.comm.bcast(out_dir, root=0)
    out_dir = str(out_dir)

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.2,
        regular=False,
        qdegree=3,
    )

    swarm = uw.swarm.Swarm(mesh=mesh)
    material = swarm.add_variable(name="material", size=1, dtype=PETSc.IntType)
    swarm.populate(fill_param=4)

    # Sanity: this is the failing case — local sizes should NOT all be equal.
    sizes = uw.mpi.comm.allgather(swarm.local_size)
    assert sum(sizes) > 0, "swarm.populate produced zero global particles"

    # Pre-fix: this hangs forever. The pytest timeout (60s) catches that
    # rather than letting the run block indefinitely.
    swarm.write_timestep(
        filename="swarm",
        swarmname="swarm",
        index=0,
        outputPath=out_dir,
        swarmVars=[material],
    )

    # Verify file content
    expected_h5 = os.path.join(out_dir, "swarm.swarm.00000.h5")
    expected_var = os.path.join(out_dir, "swarm.swarm.material.00000.h5")
    expected_xdmf = os.path.join(out_dir, "swarm.swarm.00000.xdmf")
    if uw.mpi.rank == 0:
        assert os.path.exists(expected_h5), expected_h5
        assert os.path.exists(expected_var), expected_var
        assert os.path.exists(expected_xdmf), expected_xdmf

        import h5py
        with h5py.File(expected_h5, "r") as f:
            n_global = f["coordinates"].shape[0]
        assert n_global == sum(sizes), (
            f"saved coords shape {n_global} != sum of local sizes {sum(sizes)}"
        )
