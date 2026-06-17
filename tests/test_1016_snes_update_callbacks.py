"""SNES per-iteration update callbacks (PETSc SNESSetUpdate).

Covers:
  - the callback actually fires during the nonlinear solve;
  - the pressure-gauge helper drives the surface-mean pressure to zero on an
    enclosed (pressure-null-space) problem.

The Helmholtz / shear-band smoother use case exercises the same machinery but
needs the boundary-FEM-correct field scatter (non-zero Dirichlet velocity DOFs);
that is tracked separately and not asserted here.
"""
import numpy as np
import pytest
import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _lid_driven(cellSize=0.1):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=cellSize, qdegree=3)
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.add_dirichlet_bc((1.0, 0.0), "Top")
    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
    stokes.add_dirichlet_bc((0.0, 0.0), "Left")
    stokes.add_dirichlet_bc((0.0, 0.0), "Right")
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-8
    return mesh, v, p, stokes


def test_update_callback_fires():
    mesh, v, p, stokes = _lid_driven()
    calls = []
    stokes.add_update_callback(lambda solver, iteration: calls.append(iteration))
    stokes.solve()
    assert len(calls) > 0, "SNES update callback was never called"


def test_pressure_gauge_zero_mean_on_boundary():
    mesh, v, p, stokes = _lid_driven()
    stokes.set_pressure_gauge("Top")
    stokes.solve()

    area = uw.maths.BdIntegral(mesh, 1.0, "Top").evaluate()
    mean_top = uw.maths.BdIntegral(mesh, p.sym[0, 0], "Top").evaluate() / area
    assert np.isclose(mean_top, 0.0, atol=1.0e-8), f"top-mean pressure = {mean_top}"


def test_no_callback_path_unchanged():
    # With no callbacks registered the solve must succeed exactly as before
    # (the feature is a no-op).
    mesh, v, p, stokes = _lid_driven()
    stokes.solve()
    assert stokes.snes.getConvergedReason() > 0
