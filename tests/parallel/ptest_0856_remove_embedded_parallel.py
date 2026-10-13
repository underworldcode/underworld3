"""Parallel removal (:func:`place_surface.remove_embedded`) — the lifecycle
on a distributed mesh, with no redistribution ever.

The removal gathers the object's star exactly as the placements do, carves
and refills on one rank, and every rank rebuilds collectively — the rest of
the mesh never moves. The object's geometry is allgathered from its labels
before marking, so removal works whatever the current distribution (the
object may have been scattered by a checkpoint reload).

Deterministic assertions only: labels globally empty, volume conserved (the
routine's own collective gate — asserting the info proves the gates ran and
agreed), the in-call validity battery (check_faces at every rank count), and
refusals collective. Cell counts after a refill are NOT pinned: the fill is
gmsh's, and its output depends on shell node ordering, which the gather
sets. Run:

    mpirun -np 2 python -m pytest tests/parallel/ptest_0856_remove_embedded_parallel.py --with-mpi
"""
import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw
from underworld3.utilities.place_surface import (place_thin_volume,
                                                 remove_embedded)

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2,
              pytest.mark.tier_b, pytest.mark.timeout(600)]

PATCH = np.array([[0.3, 0.3, 0.5], [0.7, 0.3, 0.5],
                  [0.7, 0.7, 0.5], [0.3, 0.7, 0.5]])


def test_a_distributed_zone_is_removed_cleanly():
    comm = uw.mpi.comm
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.11, regular=False, qdegree=2)
    zoned, _ = place_thin_volume(mesh.dm, [PATCH], width=0.045,
                                 label="Zone", label_value=5)
    cleared, info = remove_embedded(zoned, "Zone", label_value=5)

    assert info["n_removed_cells"] > 0 and info["n_filled_cells"] > 0
    for name in ("Zone", "Zone_skin"):
        left = (cleared.getLabel(name).getStratumSize(5)
                if cleared.hasLabel(name) else 0)
        assert int(comm.allreduce(left, op=MPI.SUM)) == 0

    gathered = comm.allgather(info)
    assert all(g == gathered[0] for g in gathered)


def test_removal_refusals_are_collective():
    comm = uw.mpi.comm
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.2, regular=False, qdegree=2)
    message = None
    try:
        remove_embedded(mesh.dm, "Ghost", label_value=3)
    except (RuntimeError, ValueError) as exc:
        message = str(exc)
    messages = comm.allgather(message)
    assert all(m is not None for m in messages), (
        f"some rank did NOT raise: {[m is None for m in messages]}")
    assert len(set(messages)) == 1, "ranks raised different errors"


def _fe_quadratic(dm, base, bounds, tag):
    import sympy
    dim = dm.getDimension()
    mesh = uw.discretisation.Mesh(
        dm.clone(), simplex=True, qdegree=3, boundaries=bounds,
        coordinate_system_type=base.CoordinateSystem.coordinate_type)
    xs = mesh.X
    exact = sum(c**2 for c in xs)
    t = uw.discretisation.MeshVariable(f"T_lc_{tag}", mesh, 1, degree=2)
    poisson = uw.systems.Poisson(mesh, u_Field=t)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = -2.0 * dim
    walls = (("Bottom", "Top", "Left", "Right") if dim == 2 else
             ("Bottom", "Top", "Left", "Right", "Front", "Back"))
    for wall in walls:
        poisson.add_dirichlet_bc(sympy.Matrix([exact]), wall)
    poisson.tolerance = 1e-11
    poisson.solve()
    X = np.asarray(t.coords)
    err = np.abs(np.asarray(t.data[:, 0]) - (X**2).sum(axis=1))
    return uw.mpi.comm.allreduce(float(err.max()) if len(err) else 0.0,
                                 op=MPI.MAX)


def test_the_distributed_lifecycle_composes_in_3d():
    """Add a zone to the RUNNING distributed mesh, solve, remove, solve —
    no redistribution anywhere; FE exactness is the whole-chain oracle."""
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.11, regular=False, qdegree=2)
    bounds = base._boundaries_with("Zone")
    zoned, _ = place_thin_volume(base.dm, [PATCH], width=0.045,
                                 label="Zone",
                                 label_value=bounds["Zone"].value)
    assert _fe_quadratic(zoned, base, bounds, "z3") < 1e-8
    cleared, _ = remove_embedded(zoned, "Zone",
                                 label_value=bounds["Zone"].value)
    assert _fe_quadratic(cleared, base, bounds, "c3") < 1e-8


def test_the_distributed_lifecycle_composes_in_2d():
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.05,
        regular=False, qdegree=2)
    bounds = base._boundaries_with("Zone")
    l1 = np.array([[0.3, 0.35], [0.7, 0.65]])
    l2 = np.array([[0.3, 0.65], [0.7, 0.35]])
    zoned, _ = place_thin_volume(base.dm, [l1, l2], width=0.02,
                                 label="Zone",
                                 label_value=bounds["Zone"].value)
    assert _fe_quadratic(zoned, base, bounds, "z2") < 1e-8
    cleared, info = remove_embedded(zoned, "Zone",
                                    label_value=bounds["Zone"].value)
    assert info["n_removed_cells"] > 0
    assert _fe_quadratic(cleared, base, bounds, "c2") < 1e-8
