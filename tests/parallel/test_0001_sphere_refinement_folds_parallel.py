"""A refinement callback cannot invert a cell, at any rank count (#946).

The refinement hierarchy puts back where refinement placed it a vertex whose move by the
callback turned a cell inside out. The ranks that see the inverted cell choose the
vertex; every other rank that holds a copy of it must put its copy back too, or the
copies disagree. Every assert follows the collectives it needs, so a failure fails on every rank instead
of leaving the others waiting. Contracts, on every rank:

- a shell whose inner wall is left creased (gmsh's triangulation, not flipped to the
  convex hull) refines with no inverted cell, and every copy of every vertex matches its
  owner's; it is skipped only on a host whose gmsh builds that shell with no pocket
  cell (no warning and nothing inverted);
- a box whose callback pushes the new top-wall vertex at (0.25, 1) below the row at
  y = 0.75 keeps that vertex where refinement placed it on every rank that holds it,
  and keeps the callback's harmless move of the vertex at (0.75, 1);
- the shell as built, with convex-hull walls, refines with no inverted cell and no
  warning.

On development eddab707 (no guard) at np 2 on Linux (hyperion, 2026-11-06), the folding
callback leaves 2 inverted cells and the shell 5, with no warning, and those tests fail.

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_0001_sphere_refinement_folds_parallel.py
"""
import os
import warnings

import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw
from underworld3.meshing import _sphere_surface
from underworld3.meshing._mesh_files import mesh_file_dir
from underworld3.utilities._plex_topology import closure_points

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(600)]


def _inverted_cells(dm):
    """Cells, owned or overlap, whose orientation is opposite to the rest, over all ranks."""
    vs, _ = dm.getDepthStratum(0)
    cs, ce = dm.getHeightStratum(0)
    cdim = dm.getCoordinateDim()
    x = dm.getCoordinatesLocal().array.reshape(-1, cdim)
    cells = closure_points(dm, np.arange(cs, ce), 0)
    local = 0
    if cells.shape[0] and cells.shape[1] == cdim + 1:    # a rank may hold no cells
        cells = cells - vs
        sign = np.sign(np.linalg.det(x[cells[:, 1:]] - x[cells[:, :1]]))
        local = int(np.count_nonzero(sign != np.sign(np.median(sign))))
    return uw.mpi.comm.allreduce(local)


def _copies_disagree(dm):
    """Vertex copies whose coordinates differ from their owner's, over all ranks."""
    local = dm.getCoordinatesLocal()
    owners = local.duplicate()
    dm.getCoordinateDM().globalToLocal(dm.getCoordinates(), owners)
    return uw.mpi.comm.allreduce(int(np.count_nonzero(owners.array != local.array)))


def _warned(record):
    """Whether any rank caught the hierarchy's warning (it is issued on rank 0)."""
    caught = any("refinement callback" in str(w.message) for w in record)
    return bool(uw.mpi.comm.allreduce(int(caught), op=MPI.MAX))


def test_a_creased_shell_refines_untangled_on_every_rank(monkeypatch):
    monkeypatch.setattr(_sphere_surface, "_flip_creased_sphere_edges", lambda *args, **kwargs: None)
    directory = mesh_file_dir()
    os.makedirs(directory, exist_ok=True)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        mesh = uw.meshing.SphericalShell(
            radiusOuter=1.0, radiusInner=0.5, cellSize=0.5, refinement=1, qdegree=2,
            filename=f"{directory}/uw_test_0001_creased_shell.msh")
    warned = _warned(record)
    inverted, disagree = _inverted_cells(mesh.dm), _copies_disagree(mesh.dm)
    # no warning and nothing inverted: this host's gmsh built no pocket, so there was
    # nothing to hold back; inverted cells without a warning mean the guard is missing
    if not warned and inverted == 0:
        pytest.skip("this host's gmsh built a creased shell with no pocket cell: nothing to hold back")
    assert inverted == 0 and disagree == 0, (warned, inverted, disagree)


def test_a_folding_callback_is_held_back_on_every_rank():
    def push_down(dm):
        c = dm.getCoordinatesLocal()
        x = c.array.reshape(-1, 2)
        x[np.isclose(x[:, 0], 0.25) & np.isclose(x[:, 1], 1.0), 1] = 0.55
        x[np.isclose(x[:, 0], 0.75) & np.isclose(x[:, 1], 1.0), 1] = 0.99
        c.array[...] = x.ravel()
        dm.setCoordinatesLocal(c)

    base = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.5, regular=True)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        mesh = uw.discretisation.Mesh(base.dm.clone(), simplex=True, boundaries=base.boundaries,
                                      refinement=1, refinement_callback=push_down)
    warned = _warned(record)
    inverted, disagree = _inverted_cells(mesh.dm), _copies_disagree(mesh.dm)
    x = mesh.dm.getCoordinatesLocal().array.reshape(-1, 2)
    pushed = uw.mpi.comm.allreduce(int(np.any(np.isclose(x[:, 1], 0.55))))
    held = uw.mpi.comm.allreduce(int(np.any(np.isclose(x[:, 0], 0.25) & np.isclose(x[:, 1], 1.0))))
    moved = uw.mpi.comm.allreduce(int(np.any(np.isclose(x[:, 0], 0.75) & np.isclose(x[:, 1], 0.99))))
    assert warned and inverted == 0 and disagree == 0, (warned, inverted, disagree)
    assert pushed == 0 and held >= 1 and moved >= 1, (pushed, held, moved)


def test_the_shell_refines_untangled_without_a_warning():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        mesh = uw.meshing.SphericalShell(radiusOuter=1.0, radiusInner=0.5, cellSize=0.5,
                                         refinement=1, qdegree=2)
    warned = _warned(record)
    inverted, disagree = _inverted_cells(mesh.dm), _copies_disagree(mesh.dm)
    assert not warned and inverted == 0 and disagree == 0, (warned, inverted, disagree)
