"""A PETSc reload read gives the field that was written, or refuses (UNFILED 48).

``read_checkpoint`` maps the file's points through the mesh's topology SF,
which maps from the file the mesh was loaded from. That holds when the mesh was
rebuilt from the checkpoint. The mesh that wrote the file was loaded from the
gmsh cache, whose points are numbered differently from np 2 on, and a read into
a second variable on it gave the wrong field, silently (max deviation 1.6 at the
nodes for a field of size about 2). ``load_from_h5_plex_vector`` had the mirror
fault: it loads the vector as stored, right on the writing mesh, wrong on a
rebuilt one.

Now the file's saved point order is compared with the order of the mesh's
source file. The writing mesh, in the layout it wrote with, reads the vector
back directly; a rebuilt mesh reads through the point map; any other mesh is
refused, on every rank. ``load_from_h5_plex_vector`` goes through
``read_checkpoint`` unless this mesh wrote the file in this layout.
"""
import os
import tempfile

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b, pytest.mark.timeout(300)]


def _box():
    return uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=1 / 12, qdegree=3)


def _field(c):
    return np.sin(3 * c[:, 0]) * np.cos(2 * c[:, 1]) + c[:, 0] if len(c) else np.empty(0)


def _max_deviation(var):
    c = np.asarray(var.coords)
    local = float(np.abs(np.asarray(var.array[:, 0, 0]) - _field(c)).max()) if len(c) else 0.0
    return uw.mpi.comm.allreduce(local, op=uw.MPI.MAX)


@pytest.fixture
def written():
    mesh = _box()
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    T.array[:, 0, 0] = _field(np.asarray(T.coords))
    out = uw.mpi.comm.bcast(tempfile.mkdtemp() if uw.mpi.rank == 0 else None)
    mesh.write_timestep("ck", index=0, outputPath=out, meshVars=[T], petsc_reload=True)
    return mesh, os.path.join(out, "ck.mesh.T.00000.h5"), os.path.join(out, "ck.mesh.00000.h5")


def test_read_checkpoint_on_the_writing_mesh(written):
    mesh, var_file, _ = written
    R = uw.discretisation.MeshVariable("R_writer", mesh, 1, degree=2)
    R.read_checkpoint(var_file, data_name="T")
    assert _max_deviation(R) < 1e-12


def test_read_checkpoint_on_the_rebuilt_mesh(written):
    _, var_file, mesh_file = written
    rebuilt = uw.discretisation.Mesh(mesh_file, qdegree=3)
    R = uw.discretisation.MeshVariable("R_rebuilt", rebuilt, 1, degree=2)
    R.read_checkpoint(var_file, data_name="T")
    assert _max_deviation(R) < 1e-12


def test_read_checkpoint_on_another_mesh_is_refused_in_parallel(written):
    _, var_file, _ = written
    other = _box()
    R = uw.discretisation.MeshVariable("R_other", other, 1, degree=2)
    if uw.mpi.size == 1:
        R.read_checkpoint(var_file, data_name="T")      # one numbering serially
        assert _max_deviation(R) < 1e-12
        return
    message = None
    try:
        R.read_checkpoint(var_file, data_name="T")
    except RuntimeError as err:
        message = str(err)
    messages = uw.mpi.comm.allgather(message)
    assert messages[0] is not None and "point map does not apply" in messages[0], messages
    assert len(set(messages)) == 1


@pytest.mark.parametrize("on", ["writing", "rebuilt"])
def test_load_from_h5_plex_vector(written, on):
    mesh, var_file, mesh_file = written
    target = mesh if on == "writing" else uw.discretisation.Mesh(mesh_file, qdegree=3)
    R = uw.discretisation.MeshVariable(f"L_{on}", target, 1, degree=2)
    R.load_from_h5_plex_vector(var_file, data_name="T")
    assert _max_deviation(R) < 1e-12


# UNFILED 49: a file written with create_xdmf=False holds no fields/ group.
# read_timestep's rank 0 raised KeyError there while the other ranks waited in
# a bcast. It now reads the file's PETSc payload through read_checkpoint.

@pytest.mark.parametrize("into", ["T", "U"])
def test_read_timestep_of_a_payload_only_file(into):
    mesh = _box()
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    T.array[:, 0, 0] = _field(np.asarray(T.coords))
    out = uw.mpi.comm.bcast(tempfile.mkdtemp() if uw.mpi.rank == 0 else None)
    mesh.write_timestep("ck", index=0, outputPath=out, meshVars=[T],
                        create_xdmf=False, petsc_reload=True)
    R = T if into == "T" else uw.discretisation.MeshVariable(into, mesh, 1, degree=2)
    R.read_timestep("ck", "T", 0, outputPath=out)
    assert _max_deviation(R) < 1e-12
