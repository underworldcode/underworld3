"""A material index read at the integration points, or as a fraction per cell.

``IndexSwarmVariable(..., proxy_location=...)`` chooses where the level sets
live. At the integration points each one is exactly 0 or 1 (the material of
the nearest particle), which is what keeps a layered viscosity exact; per
cell they are a polynomial fraction, clamped and renormalised so the masks
still sum to one.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _layered(tag, location, cell_size=0.1, fill=3, degree=1, h=0.5):
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=cell_size, qdegree=2, regular=True)
    swarm = uw.swarm.Swarm(mesh)
    mat = uw.swarm.IndexSwarmVariable(tag, swarm, indices=2, proxy_degree=degree,
                                      proxy_location=location)
    swarm.populate(fill_param=fill)
    X = np.asarray(swarm._particle_coordinates.data)
    with uw.synchronised_array_update():
        mat.data[:, 0] = (X[:, 1] > h).astype(int)
    return mesh, swarm, mat


@pytest.mark.parametrize("location", ["nodes", "integration_points", "cells"])
def test_masks_are_bounded_and_sum_to_one(location):
    mesh, swarm, mat = _layered(f"P{location[:4]}", location)
    for var in mat._meshLevelSetVars:
        d = np.asarray(var.data[:, 0])
        assert d.min() >= -1e-12 and d.max() <= 1 + 1e-12, (d.min(), d.max())
    total = sum(np.asarray(v.data[:, 0]) for v in mat._meshLevelSetVars)
    assert np.allclose(total, 1.0, atol=1e-12)
    # and through the weak form
    assert abs(uw.maths.Integral(mesh, mat.sym[0] + mat.sym[1]).evaluate() - 1.0) < 1e-10


def test_integration_point_masks_are_exactly_zero_or_one():
    """Each integration point takes the material of its nearest particle: a
    label, not an average. That is what a nodal level set cannot do."""
    mesh, swarm, mat = _layered("Q", "integration_points")
    for var in mat._meshLevelSetVars:
        d = np.asarray(var.data[:, 0])
        assert set(np.unique(d).tolist()) <= {0.0, 1.0}, np.unique(d)
    assert mat._meshLevelSetVars[0].is_integration_point
    # the nodal proxy, by contrast, averages across the interface
    _, _, nodal = _layered("N", "nodes")
    d = np.asarray(nodal._meshLevelSetVars[0].data[:, 0])
    assert ((d > 1e-6) & (d < 1 - 1e-6)).any()


def test_layered_couette_is_exact_at_the_integration_points():
    """Interface on mesh edges: the exact velocity is in the P2 space, so the
    only error is the material representation."""
    eta_top, h = 1.0e3, 0.5
    results = {}
    for location in ("nodes", "integration_points"):
        mesh, swarm, mat = _layered(f"C{location[:4]}", location, h=h)
        v = uw.discretisation.MeshVariable(f"v{location[:4]}", mesh, mesh.dim, degree=2)
        p = uw.discretisation.MeshVariable(f"p{location[:4]}", mesh, 1, degree=1)
        stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
        stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
        stokes.constitutive_model.Parameters.shear_viscosity_0 = mat.createMask([1.0, eta_top])
        stokes.add_dirichlet_bc((1.0, 0.0), "Top")
        stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
        stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
        stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
        stokes.tolerance = 1e-8
        stokes.solve()
        A = 1.0 / (h + (1.0 - h) / eta_top)
        Xv = np.asarray(v.coords)
        exact = np.where(Xv[:, 1] < h, A * Xv[:, 1], A * h + A / eta_top * (Xv[:, 1] - h))
        results[location] = np.abs(np.asarray(v.data[:, 0]) - exact).max()
    assert results["integration_points"] < 1e-5, results
    assert results["nodes"] > 1e-2, results            # the control


def test_repopulation_keeps_a_material_index_a_label():
    """A new particle takes its nearest neighbour's index whole: averaging two
    labels does not give a label."""
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.15, qdegree=2)
    swarm = uw.swarm.Swarm(mesh)
    mat = uw.swarm.IndexSwarmVariable("R", swarm, indices=3, proxy_location="integration_points")
    scalar = uw.swarm.SwarmVariable("Rs", swarm, 1)
    swarm.populate(fill_param=3)
    X = np.asarray(swarm._particle_coordinates.data)
    with uw.synchronised_array_update():
        mat.data[:, 0] = np.digitize(X[:, 1], [0.33, 0.66])
        scalar.data[:, 0] = X[:, 0]
    for i in np.sort(np.nonzero(X[:, 0] < 0.4)[0])[::-1]:
        swarm.dm.removePointAtIndex(int(i))
    swarm._invalidate_canonical_data()
    added, _ = swarm.repopulate(order=1)
    from mpi4py import MPI
    assert uw.mpi.comm.allreduce(added, op=MPI.SUM) > 0     # added is rank-local
    Xn = np.asarray(swarm._particle_coordinates.data)
    idx = np.asarray(mat.data[:, 0])
    assert set(np.unique(idx).tolist()) <= {0, 1, 2}, np.unique(idx)
    # right material for all but the particles nearest an interface
    if idx.shape[0] > 0:
        assert np.mean(idx == np.digitize(Xn[:, 1], [0.33, 0.66])) > 0.9
    # a float field is still reconstructed, not copied from one neighbour
    assert np.abs(np.asarray(scalar.data[:, 0]) - Xn[:, 0]).max() < 1e-10


def test_population_control_keeps_an_extending_box_sampled():
    """Pure shear with outflow sides and inflow top and bottom: without
    population control the inflow cells empty."""
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(-1.0, -0.5), maxCoords=(1.0, 0.5),
                                             cellSize=0.15, qdegree=2)
    mesh.return_coords_to_bounds = None
    x, y = mesh.X
    V = sympy.Matrix([[x, -y]])
    c0, c1 = mesh.dm.getHeightStratum(0)

    def empty_cells(swarm):
        P = np.asarray(swarm._particle_coordinates.data)
        cells = np.asarray(mesh._robust_owning_cells(P))
        return int((np.bincount(cells[cells >= 0], minlength=c1 - c0) == 0).sum())

    counts = {}
    for control in (False, True):
        swarm = uw.swarm.Swarm(mesh)
        swarm.populate(fill_param=3)
        if control:
            swarm.population_control = dict(min_per_cell=6)
        for _ in range(10):
            swarm.advection(V, 0.1, order=2)
        counts[control] = empty_cells(swarm)
    from mpi4py import MPI
    assert uw.mpi.comm.allreduce(counts[True], op=MPI.SUM) == 0, counts
    assert uw.mpi.comm.allreduce(counts[False], op=MPI.SUM) > 0, counts
