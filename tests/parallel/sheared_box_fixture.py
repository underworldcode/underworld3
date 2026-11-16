"""#1123's sheared 3-D box, built the same on every platform and never tangled.

The box is a structured grid of n^3 cubes, each split into six tetrahedra along
the axis-ordered paths from one corner to the opposite one, so the mesh does not
depend on gmsh. (gmsh's box differed between builds: 1094 cells on macOS against
1086 on Linux, and with the old wave amplitude 0.03 one macOS cell was turned
inside out, so the locator tests had no right answer there.)

The map is #1123's linear shear A (det 0.196, smallest singular value 0.175) plus
sine terms of amplitude SHEAR_WAVE. The sine terms' gradient is a scaled
permutation matrix of norm 2 pi SHEAR_WAVE = 0.157 at 0.025, below 0.175, so the
map is invertible; `sheared_box` asserts, collectively, that no cell turned over.
"""
import itertools

import numpy as np
from petsc4py import PETSc

import underworld3 as uw

SHEAR = np.array([[1.0, 1.2, 0.3], [0.1, 0.7, 0.5], [0.0, 0.2, 0.5]])
SHEAR_WAVE = 0.025


def kuhn_box(n):
    """A DM of the unit cube in n^3 cubes, six tetrahedra each, on rank 0 (the
    Mesh distributes it). Collective."""
    coords = np.zeros((0, 3))
    cells = np.zeros((0, 4), dtype=PETSc.IntType)
    if PETSc.COMM_WORLD.getRank() == 0:
        def node(i, j, k):
            return (i * (n + 1) + j) * (n + 1) + k
        coords = np.array([[i / n, j / n, k / n]
                           for i in range(n + 1) for j in range(n + 1) for k in range(n + 1)])
        tets = []
        for i, j, k in itertools.product(range(n), repeat=3):
            for order in itertools.permutations(range(3)):
                corner = [i, j, k]
                path = [node(*corner)]
                for axis in order:
                    corner[axis] += 1
                    path.append(node(*corner))
                tets.append(path)
        cells = np.asarray(tets, dtype=PETSc.IntType)
    dm = PETSc.DMPlex().createFromCellList(3, cells, coords, interpolate=True, comm=PETSc.COMM_WORLD)
    dm.createLabel("All_Boundaries")
    dm.markBoundaryFaces("All_Boundaries", 1001)
    return dm


def cell_orientations(mesh):
    """The sign of each local cell's volume."""
    from underworld3.utilities._plex_topology import closure_points
    cs, ce = mesh.dm.getHeightStratum(0)
    if ce == cs:
        return np.zeros(0)
    Xl = np.asarray(mesh._coords).reshape(-1, mesh.cdim)
    V = Xl[mesh._coord_rows_for_points(
        mesh.dm, closure_points(mesh.dm, np.arange(cs, ce), 0).ravel())].reshape(ce - cs, -1, mesh.cdim)
    return np.sign(np.linalg.det(V[:, 1:, :] - V[:, :1, :]))


def sheared_box(n=6, qdegree=3):
    """The sheared box; fails loudly if any cell turned over. Collective."""
    from mpi4py import MPI

    mesh = uw.discretisation.Mesh(kuhn_box(n), degree=1, qdegree=qdegree)
    before = cell_orientations(mesh)
    X = np.asarray(mesh.X.coords).copy()
    Y = X @ SHEAR.T
    Y[:, 0] += SHEAR_WAVE * np.sin(2 * np.pi * X[:, 1])
    Y[:, 1] += SHEAR_WAVE * np.sin(2 * np.pi * X[:, 2])
    Y[:, 2] += SHEAR_WAVE * np.sin(2 * np.pi * X[:, 0])
    mesh.deform(Y)
    flipped = uw.mpi.comm.allreduce(int((cell_orientations(mesh) != before).sum()), op=MPI.SUM)
    assert flipped == 0, f"the sheared fixture turned {flipped} cells over"
    return mesh
