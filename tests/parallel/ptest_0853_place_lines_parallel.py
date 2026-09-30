"""Parallel 2-D placement (:func:`place_surface.place_along_lines`).

Gather-first, one rebuild for every rank count — the 3-D mechanism one
dimension down, serving the lifecycle ruling: surfaces are added to the
RUNNING distributed mesh, never via redistribution. Scope: INTERIOR
surfaces (a wall-touching end needs the serial end-settling machinery and
must refuse collectively — asserted here, since a rank-local raise is a
hang at np>=3).

The correctness oracle is FE exactness: P2 reproduces x^2 + y^2 through the
placed surface to solver precision only if the rebuilt chart, its seam
cones and its star-forest are all right — the probe class every
topological gate is blind to (issue #520's lesson). Run:

    mpirun -np 2 python -m pytest tests/parallel/ptest_0853_place_lines_parallel.py --with-mpi
"""
import numpy as np
import pytest
import sympy
from mpi4py import MPI

import underworld3 as uw
from underworld3.utilities.place_surface import place_along_lines

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2,
              pytest.mark.tier_b, pytest.mark.timeout(600)]


def test_an_interior_surface_places_and_solves_exactly():
    comm = uw.mpi.comm
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.05,
        regular=False, qdegree=2)
    bounds = base._boundaries_with("Fault")
    line = np.array([[0.25, 0.4], [0.75, 0.6]])
    new, info = place_along_lines(base.dm, [line], label="Fault",
                                  label_value=bounds["Fault"].value)

    assert info["n_surface_facets"] == (info["n_placed"]
                                        + info["n_on_surface"] - 1)
    gathered = comm.allgather(info)
    assert all(g == gathered[0] for g in gathered)

    mesh = uw.discretisation.Mesh(
        new, simplex=True, qdegree=3, boundaries=bounds,
        coordinate_system_type=base.CoordinateSystem.coordinate_type)
    x, y = mesh.X
    exact = x**2 + y**2
    t = uw.discretisation.MeshVariable("T_p2d", mesh, 1, degree=2)
    poisson = uw.systems.Poisson(mesh, u_Field=t)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = -4.0
    for wall in ("Bottom", "Top", "Left", "Right"):
        poisson.add_dirichlet_bc(sympy.Matrix([exact]), wall)
    poisson.tolerance = 1e-11
    poisson.solve()
    X = np.asarray(t.coords)
    err = np.abs(np.asarray(t.data[:, 0]) - (X[:, 0]**2 + X[:, 1]**2))
    worst = comm.allreduce(float(err.max()) if len(err) else 0.0, op=MPI.MAX)
    assert worst < 1e-8, f"wrong operator on the placed mesh: {worst:.3e}"


def test_a_wall_touching_surface_refuses_collectively():
    comm = uw.mpi.comm
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1,
        regular=False, qdegree=2)
    crossing = np.array([[-0.1, 0.45], [1.1, 0.55]])
    message = None
    try:
        place_along_lines(base.dm, [crossing], label="Bad")
    except (NotImplementedError, RuntimeError, ValueError) as exc:
        message = str(exc)
    messages = comm.allgather(message)
    assert all(m is not None for m in messages), (
        f"some rank did NOT raise: {[m is None for m in messages]}")
    assert len(set(messages)) == 1, "ranks raised different errors"
    assert "serial" in messages[0] or "INTERIOR" in messages[0].upper()
