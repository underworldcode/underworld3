"""SNES per-iteration update callbacks (PETSc SNESSetUpdate).

Covers:
  - the callback actually fires during the nonlinear solve;
  - the pressure-gauge helper drives the surface-mean pressure to zero on an
    enclosed (pressure-null-space) problem;
  - the mid-solve field scatter is correct on driven (non-zero Dirichlet)
    boundaries, so a callback reading the velocity on a lid sees the imposed
    value rather than a stale one (boundary-FEM-correct scatter);
  - the same machinery works generically on single-field scalar and vector
    solvers (not just Stokes);
  - an auxiliary Helmholtz/Projection smoother fired each iteration is
    self-consistent with the converged velocity (the shear-band use case).
"""
import numpy as np
import sympy
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


def test_callback_sees_driven_boundary_velocity():
    # Core guard for the boundary-FEM-correct scatter: a callback reading the
    # velocity on the driven (lid) boundary must see the imposed value (1, 0),
    # not the stale/zero value left by a plain global->local scatter (which does
    # not carry non-zero Dirichlet DOFs). Read via uw.function.evaluate, the
    # same path an auxiliary smoother uses.
    mesh, v, p, stokes = _lid_driven()

    # interior Top-boundary sample points (avoid the two-BC corner nodes)
    xs = np.linspace(0.2, 0.8, 5)
    pts = np.column_stack([xs, np.full_like(xs, 1.0)])

    seen = {}

    def probe(solver, iteration):
        seen["vx"] = np.asarray(uw.function.evaluate(v.sym[0], pts)).ravel()
        seen["vy"] = np.asarray(uw.function.evaluate(v.sym[1], pts)).ravel()

    stokes.add_update_callback(probe)
    stokes.solve()

    assert "vx" in seen, "callback never fired"
    # Driven boundary value must reach the callback (was ~0 before the fix).
    assert np.allclose(seen["vx"], 1.0, atol=1.0e-8), f"vx on Top = {seen['vx']}"
    assert np.allclose(seen["vy"], 0.0, atol=1.0e-8), f"vy on Top = {seen['vy']}"


def test_scalar_solver_callback_sees_driven_boundary():
    # The callback machinery is generic across solvers. On a single-field scalar
    # (Poisson) solver a callback must (a) fire without error and (b) read the
    # field on a non-zero Dirichlet boundary as the imposed value, via the
    # boundary-FEM-correct scatter. Before generalisation a callback on a scalar
    # solver raised AttributeError ('no attribute fields').
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2, qdegree=3)
    phi = uw.discretisation.MeshVariable("phi", mesh, 1, degree=2)
    poisson = uw.systems.Poisson(mesh, u_Field=phi)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 0.0
    poisson.add_dirichlet_bc(3.0, "Top")       # non-zero (driven) value
    poisson.add_dirichlet_bc(0.0, "Bottom")

    xs = np.linspace(0.2, 0.8, 4)
    top = np.column_stack([xs, np.full_like(xs, 1.0)])
    seen = {}

    def cb(solver, iteration):
        seen["top"] = np.asarray(uw.function.evaluate(phi.sym[0], top)).ravel()

    poisson.add_update_callback(cb)
    poisson.solve()

    assert "top" in seen, "callback never fired on scalar solver"
    assert np.allclose(seen["top"], 3.0, atol=1.0e-8), f"phi on Top = {seen['top']}"


def test_final_iterate_dispatch_on_scalar_solver():
    # The callbacks are documented as being applied once to the final converged
    # iterate (SNESSetUpdate only fires at iteration START). That final dispatch
    # is centralised in _snes_solve_with_retries, so non-Stokes solvers get it
    # too -- signalled by an iteration == -1 call after the solve completes.
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, qdegree=3)
    phi = uw.discretisation.MeshVariable("phif", mesh, 1, degree=2)
    poisson = uw.systems.Poisson(mesh, u_Field=phi)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 1.0
    for bc in ("Bottom", "Top", "Left", "Right"):
        poisson.add_dirichlet_bc(0.0, bc)

    iters = []
    poisson.add_update_callback(lambda solver, iteration: iters.append(iteration))
    poisson.solve()

    assert -1 in iters, f"final-iterate dispatch did not fire on scalar solver: {iters}"


def test_vector_solver_callback_sees_driven_boundary():
    # Vector-solver analogue: a callback on a Vector_Projection reads both
    # components of the field on a non-zero Dirichlet boundary as imposed.
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2, qdegree=3)
    U = uw.discretisation.MeshVariable("Uv", mesh, mesh.dim, degree=2)
    vp = uw.systems.Vector_Projection(mesh, U)
    x, y = mesh.X
    vp.uw_function = sympy.Matrix([[0.5 * x, 0.5 * y]])
    vp.add_dirichlet_bc((2.0, -1.0), "Top")

    xs = np.linspace(0.2, 0.8, 4)
    top = np.column_stack([xs, np.full_like(xs, 1.0)])
    seen = {}

    def cb(solver, iteration):
        seen["ux"] = np.asarray(uw.function.evaluate(U.sym[0], top)).ravel()
        seen["uy"] = np.asarray(uw.function.evaluate(U.sym[1], top)).ravel()

    vp.add_update_callback(cb)
    vp.solve()

    assert "ux" in seen, "callback never fired on vector solver"
    assert np.allclose(seen["ux"], 2.0, atol=1.0e-8), f"Ux on Top = {seen['ux']}"
    assert np.allclose(seen["uy"], -1.0, atol=1.0e-8), f"Uy on Top = {seen['uy']}"


def test_helmholtz_smoother_self_consistent():
    # Shear-band use case: viscosity depends on a projected (smoothed) strain-rate
    # field ebar, refreshed by a Projection smoother fired every SNES iteration.
    # At convergence ebar must equal the re-projection of the converged velocity.
    # This exercises the boundary-correct scatter in the final post-solve dispatch:
    # before the fix the smoother saw a stale lid velocity and ebar disagreed with
    # a standalone re-projection by ~35%.
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1, qdegree=3)
    v = uw.discretisation.MeshVariable("Uh", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("Ph", mesh, 1, degree=1)
    ebar = uw.discretisation.MeshVariable("ebar", mesh, 1, degree=1)
    ebar.data[:] = 1.0

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    # viscosity depends on the SMOOTHED strain-rate field (nonlocal coupling)
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0 + 2.0 / (1.0 + ebar.sym[0, 0])

    edot = stokes.strainrate
    e_local = sympy.sqrt((edot * edot).trace() / 2 + 1.0e-6)
    proj = uw.systems.Projection(mesh, ebar)
    proj.uw_function = e_local
    proj.smoothing_length = 0.1

    stokes.add_update_callback(lambda solver, iteration: proj.solve())

    stokes.add_dirichlet_bc((1.0, 0.0), "Top")
    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
    stokes.add_dirichlet_bc((0.0, None), "Left")
    stokes.add_dirichlet_bc((0.0, None), "Right")
    stokes.tolerance = 1.0e-7
    stokes.solve()

    assert stokes.snes.getConvergedReason() > 0, "coupled solve did not converge"

    ebar_before = np.array(ebar.data[:, 0])
    proj.solve()
    ebar_after = np.array(ebar.data[:, 0])
    rel = np.linalg.norm(ebar_after - ebar_before) / (np.linalg.norm(ebar_after) + 1.0e-30)
    assert rel < 1.0e-4, f"ebar not self-consistent at convergence: rel = {rel}"
