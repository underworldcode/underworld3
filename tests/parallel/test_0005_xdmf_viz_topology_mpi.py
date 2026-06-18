"""MPI regression tests for PETSc-native XDMF visualization topology."""

import os
import shutil
import tempfile

import h5py
import pytest

import underworld3 as uw


pytestmark = [
    pytest.mark.level_1,
    pytest.mark.tier_a,
    pytest.mark.mpi(min_size=2),
    pytest.mark.timeout(60),
]


def _shared_tmpdir(prefix):
    if uw.mpi.rank == 0:
        tmpdir = tempfile.mkdtemp(prefix=prefix)
    else:
        tmpdir = None
    return uw.mpi.comm.bcast(tmpdir, root=0)


def _cleanup_shared_tmpdir(tmpdir):
    uw.mpi.comm.barrier()
    if uw.mpi.rank == 0:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _check_mesh_output(out_dir, name):
    mesh_h5 = os.path.join(out_dir, f"{name}.mesh.00000.h5")
    xdmf_file = os.path.join(out_dir, f"{name}.mesh.00000.xdmf")

    if uw.mpi.rank == 0:
        assert os.path.exists(mesh_h5), mesh_h5
        assert os.path.exists(xdmf_file), xdmf_file

        with h5py.File(mesh_h5, "r") as h5f:
            assert "labels" in h5f
            assert "topology/cells" in h5f
            assert "topology/cones" in h5f
            assert "geometry/vertices" in h5f
            assert "viz/topology/cells" in h5f

            cells = h5f["viz/topology/cells"]
            vertices = h5f["geometry/vertices"]
            assert len(cells.shape) == 2
            assert cells.shape[1] > 1
            assert cells[:].min() >= 0
            assert cells[:].max() < vertices.shape[0]

        with open(xdmf_file, "r") as f:
            xdmf_text = f.read()
        assert "&MeshData;:/viz/topology/cells" in xdmf_text
        assert "&MeshData;:/geometry/vertices" in xdmf_text
        assert "&MeshData;:/topology/cells" not in xdmf_text


@pytest.mark.mpi(min_size=2)
def test_xdmf_viz_topology_parallel_2d_and_3d():
    """PETSc writes valid viz topology for timestep output under MPI."""

    out_dir = _shared_tmpdir("uw3_xdmf_viz_topology_mpi_")
    try:
        mesh_2d = uw.meshing.StructuredQuadBox(elementRes=(3, 3))
        mesh_2d.write_timestep("test_topo_2d", index=0, outputPath=out_dir)
        uw.mpi.comm.barrier()
        _check_mesh_output(out_dir, "test_topo_2d")

        mesh_3d = uw.meshing.StructuredQuadBox(elementRes=(2, 2, 2))
        mesh_3d.write_timestep("test_topo_3d", index=0, outputPath=out_dir)
        uw.mpi.comm.barrier()
        _check_mesh_output(out_dir, "test_topo_3d")
    finally:
        _cleanup_shared_tmpdir(out_dir)
