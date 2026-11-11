"""A refined mesh reloads correctly on the rank count that wrote it (#690).

``Mesh(file)`` rebuilds a refined mesh's multigrid hierarchy by refining the
coarse sidecar, then matches every point of the rebuilt fine mesh to its point
in the file. Until #690 was fixed the match was by array index, and a mesh
written on two or more ranks reloaded with cells joined to the wrong vertices.

Checked here, in parallel:

* a box at refinement 2, moved by nearly a fine cell before writing, reloads
  with its saved geometry, its hierarchy and its field (by section), and a
  Poisson solve with geometric FMG converges on it;
* a file written before the fix (simulated by deleting the reference
  coordinates the fix stores) of an unmoved mesh still restores the hierarchy,
  matched by its saved coordinates;
* such a file of a moved mesh, which cannot be matched, reloads without the
  hierarchy, with a warning, and is still correct.

The rank counts the bug depends on (written on two, read on one; written on
one, read on two) are in ``tests/test_0690_refined_checkpoint_cross_np.py``.
"""

import os
import shutil
import tempfile
import warnings

import numpy as np
import pytest
import underworld3 as uw
from mpi4py import MPI

pytestmark = [
    pytest.mark.level_2,
    pytest.mark.mpi(min_size=2),
    pytest.mark.timeout(300),
]


@pytest.fixture
def shared_dir():
    d = tempfile.mkdtemp(prefix="uw3_0690_") if uw.mpi.rank == 0 else None
    d = uw.mpi.comm.bcast(d, root=0)
    yield d
    uw.mpi.comm.barrier()
    if uw.mpi.rank == 0:
        shutil.rmtree(d, ignore_errors=True)


def _write(out, name, moved):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25, refinement=2)
    if moved:
        # Finest spacing 0.0625; the largest move is 0.06.
        x = np.asarray(mesh.X.coords).copy()
        x[:, 0] += 0.06 * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
        mesh.deform(x)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    T.data[:, 0] = np.asarray(T.coords)[:, 0]
    area = uw.maths.Integral(mesh, 1.0).evaluate()
    mesh.write_checkpoint(name, outputPath=out, meshVars=[T])
    return area


def _drop_reference_coordinates(meshfile):
    """Make ``meshfile`` look like one written before #690 was fixed."""
    if uw.mpi.rank == 0:
        import h5py

        with h5py.File(meshfile, "a") as fh:
            del fh["topologies/uw_mesh/dms/uw_reference_coordinates"]
    uw.mpi.comm.barrier()


def _reload(out, name):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mesh = uw.discretisation.Mesh(os.path.join(out, f"{name}.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    T.read_checkpoint(os.path.join(out, f"{name}.T.00000.h5"), data_name="T")
    d = np.asarray(T.data)[:, 0] - np.asarray(T.coords)[:, 0]
    err = uw.mpi.comm.allreduce(float(np.abs(d).max()) if d.size else 0.0, op=MPI.MAX)
    warned = uw.mpi.comm.allreduce(
        any("without its multigrid hierarchy" in str(w.message) for w in caught), op=MPI.LOR)
    return mesh, uw.maths.Integral(mesh, 1.0).evaluate(), err, warned


@pytest.mark.tier_b
def test_moved_refined_mesh_reloads_with_hierarchy(shared_dir):
    area = _write(shared_dir, "moved", moved=True)
    mesh, got, err, warned = _reload(shared_dir, "moved")
    assert got == pytest.approx(area, rel=1e-12)
    assert err < 1e-12
    assert len(mesh.dm_hierarchy) == 3 and not warned

    poisson = uw.systems.Poisson(mesh)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.f = 1.0
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc(0.0, "Top")
    poisson.petsc_options["pc_type"] = "mg"
    poisson.petsc_options["pc_mg_type"] = "full"
    poisson.petsc_options["pc_mg_levels"] = 3
    poisson.solve()
    assert poisson.snes.getConvergedReason() > 0


@pytest.mark.tier_b
def test_old_file_of_unmoved_mesh_restores_hierarchy(shared_dir):
    area = _write(shared_dir, "old", moved=False)
    _drop_reference_coordinates(os.path.join(shared_dir, "old.mesh.00000.h5"))
    mesh, got, err, warned = _reload(shared_dir, "old")
    assert got == pytest.approx(area, rel=1e-12)
    assert err < 1e-12
    assert len(mesh.dm_hierarchy) == 3 and not warned


@pytest.mark.tier_b
def test_old_file_of_moved_mesh_reloads_flat(shared_dir):
    area = _write(shared_dir, "old_moved", moved=True)
    _drop_reference_coordinates(os.path.join(shared_dir, "old_moved.mesh.00000.h5"))
    mesh, got, err, warned = _reload(shared_dir, "old_moved")
    assert got == pytest.approx(area, rel=1e-12)
    assert err < 1e-12
    assert len(mesh.dm_hierarchy) == 1 and warned
