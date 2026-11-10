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
  warning;
- a callback that moves vertices on one rank only leaves every copy of every vertex
  with the moved position, on every level: copies of a shared vertex differ after a
  callback that finds its vertices through the labels a rank holds, and the hierarchy
  reconciles them before it checks a cell;
- a box whose callback kicks about a fifth of its interior vertices far enough to turn
  a cell over refines to the serial mesh, vertex for vertex. A kicked vertex on a
  partition interface can invert cells on one side only; the vertex is held on every
  rank that holds it, so the mesh does not depend on the partition. Copies agreeing
  is not enough to show this: when only one side held the vertex, the copies still
  agreed at np 8, but 31 vertices were held instead of 24 (hyperion, 2026-11-06).

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

from serial_reference import emit, serial_reference

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


def _copies_disagree(mesh):
    """Vertex copies whose coordinates differ from their owner's, on every level of the
    hierarchy and on the mesh's own DM, over all ranks."""
    total = 0
    for dm in list(getattr(mesh, "dm_hierarchy", None) or []) + [mesh.dm]:
        local = dm.getCoordinatesLocal()
        owners = local.duplicate()
        dm.getCoordinateDM().globalToLocal(dm.getCoordinates(), owners)
        total += int(np.count_nonzero(owners.array != local.array))
        owners.destroy()
    return uw.mpi.comm.allreduce(total)


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
    inverted, disagree = _inverted_cells(mesh.dm), _copies_disagree(mesh)
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
    inverted, disagree = _inverted_cells(mesh.dm), _copies_disagree(mesh)
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
    inverted, disagree = _inverted_cells(mesh.dm), _copies_disagree(mesh)
    assert not warned and inverted == 0 and disagree == 0, (warned, inverted, disagree)


def test_copies_of_a_vertex_take_the_move_when_one_rank_makes_it():
    """Rank 0 alone nudges every vertex it holds, as a callback that reads only the
    labels a rank holds can do for a shared wall vertex. The copies on the other side
    of the partition must take the move."""
    def nudge_on_rank_0(dm):
        if uw.mpi.rank != 0:
            return
        c = dm.getCoordinatesLocal()
        x = c.array.reshape(-1, 2)
        x[:, 0] += 1.0e-3 * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
        c.array[...] = x.ravel()
        dm.setCoordinatesLocal(c)

    base = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25, regular=True)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        mesh = uw.discretisation.Mesh(base.dm.clone(), simplex=True, boundaries=base.boundaries,
                                      refinement=1, refinement_callback=nudge_on_rank_0)
    warned = _warned(record)
    inverted, disagree = _inverted_cells(mesh.dm), _copies_disagree(mesh)
    assert not warned and inverted == 0 and disagree == 0, (warned, inverted, disagree)


_H = 0.025     # the refined box's spacing


def _kick(dm):
    """Kick about a fifth of the interior vertices 1.3 h, in a direction that varies with
    position: each kick turns a cell inside out. A function of position only, so every
    rank and the serial run kick the same vertices the same way."""
    c = dm.getCoordinatesLocal()
    x = c.array.reshape(-1, 2)
    inside = ((x > 1.0e-9) & (x < 1.0 - 1.0e-9)).all(axis=1)
    s = np.sin(53.3 * x[:, 0] + 17.9 * x[:, 1]) * np.sin(29.1 * x[:, 0] - 61.7 * x[:, 1] + 0.5)
    phi = 97.3 * x[:, 0] + 13.1 * x[:, 1]
    kicked = inside & (s > 0.5)
    x[kicked] += 1.3 * _H * np.column_stack([np.cos(phi), np.sin(phi)])[kicked]
    c.array[...] = x.ravel()
    dm.setCoordinatesLocal(c)


def _kicked_box():
    base = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=2 * _H, regular=True)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        mesh = uw.discretisation.Mesh(base.dm.clone(), simplex=True, boundaries=base.boundaries,
                                      refinement=1, refinement_callback=_kick)
    return mesh, record


def _owned_vertices(mesh):
    """Every vertex once, at its owner's coordinates, sorted; on rank 0, None elsewhere."""
    owned = np.asarray(mesh.dm.getCoordinates().array).reshape(-1, mesh.cdim)
    gathered = uw.mpi.comm.gather(owned, root=0)
    if uw.mpi.rank != 0:
        return None
    x = np.vstack(gathered)
    return x[np.lexsort(np.round(x, 9).T[::-1])]


def test_a_held_vertex_is_held_on_every_rank_that_holds_it():
    mesh, record = _kicked_box()
    warned = _warned(record)
    inverted, disagree = _inverted_cells(mesh.dm), _copies_disagree(mesh)
    reference = serial_reference(__file__, "kicked_box")
    x = _owned_vertices(mesh)
    differ = None
    if uw.mpi.rank == 0:
        expected = np.asarray(reference["values"]).reshape(-1, 2)
        differ = (int(np.count_nonzero(np.abs(x - expected).max(axis=1) > 1.0e-12))
                  if x.shape == expected.shape else f"{len(x)} vertices, serial {len(expected)}")
    differ = uw.mpi.comm.bcast(differ, root=0)
    assert warned and inverted == 0 and disagree == 0 and differ == 0, (
        warned, inverted, disagree, differ)


if __name__ == "__main__":
    # The single-rank child of test_a_held_vertex_is_held_on_every_rank_that_holds_it
    # (see serial_reference): `python <thisfile> kicked_box`.
    import sys
    assert sys.argv[1:] == ["kicked_box"], sys.argv
    _mesh, _record = _kicked_box()
    _x = _owned_vertices(_mesh)
    emit(_x.ravel(), [len(_x), 0])
