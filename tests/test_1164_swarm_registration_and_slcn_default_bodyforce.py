"""Four small defects from the probe pass (development 01dec8d2).

* #1034: ``NavierStokesSLCN`` with no ``bodyforce`` set raised
  ``ShapeError: (2, 1) + (1, 2)`` at its first solve: the default body force was
  a column, while the setter and the time term are rows.
* #866: a proxied swarm variable's REINIT callback was set on the user-facing
  wrapper, but the remesh pass reads the base variables in ``mesh.vars``, so the
  proxy was never marked stale.
* #867: ``swarm.mesh = new`` unregistered the swarm from the model's primary
  mesh, not from the mesh it was leaving, so the old mesh kept synchronising it.
* #868: the rebuild of a mesh's DM for a new variable now empties petsc4py's
  deferred-destroy list. Serially that list is not used; this checks that the
  cleanup destroys nothing still in use (the parallel count is in
  tests/parallel/test_1164_variable_rebuild_frees_petsc_garbage_parallel.py).
"""
import pytest
import underworld3 as uw

pytestmark = [pytest.mark.level_1]


@pytest.mark.tier_b
def test_navier_stokes_slcn_solves_with_its_default_body_force():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.25)
    v = uw.discretisation.MeshVariable("U1164", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P1164", mesh, 1, degree=1)
    ns = uw.systems.NavierStokesSLCN(mesh, velocityField=v, pressureField=p, rho=1.0)
    ns.constitutive_model = uw.constitutive_models.ViscousFlowModel
    ns.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    ns.add_dirichlet_bc((0.0, 0.0), "Bottom")
    ns.add_dirichlet_bc((1.0, 0.0), "Top")
    ns.solve(timestep=0.01)
    assert ns.snes.getConvergedReason() > 0
    assert ns.bodyforce.shape == (1, mesh.dim)


@pytest.mark.tier_b
def test_the_remesh_pass_finds_a_proxied_swarm_variables_callback():
    from underworld3.discretisation.remesh import _mark_reinit_stale
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.25)
    swarm = uw.swarm.Swarm(mesh)
    s = uw.swarm.SwarmVariable("s1164", swarm, 1, proxy_degree=1)
    swarm.populate(fill_param=2)
    s._proxy_stale = False
    entries = [v for k, v in mesh.vars.items() if "s1164" in k]
    assert entries, "the proxy is not registered on the mesh"
    for entry in entries:                 # what remesh iterates over
        _mark_reinit_stale(entry)
    assert s._proxy_stale is True


@pytest.mark.tier_b
def test_a_swarm_moved_to_another_mesh_leaves_the_mesh_it_was_on():
    primary = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.5)
    old = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.4)
    new = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.3)
    swarm = uw.swarm.Swarm(old)
    swarm.populate(fill_param=1)

    def on(mesh):
        return any(x is swarm for x in getattr(mesh, "_registered_swarms", []))

    assert on(old)
    swarm.mesh = new
    assert not on(old)
    assert on(new)


@pytest.mark.tier_b
def test_twelve_variables_on_one_mesh_keep_their_values_and_a_solve_runs():
    import numpy as np
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 8)
    held = []
    for k in range(12):
        held.append(uw.discretisation.MeshVariable(f"K1164_{k}", mesh, 1, degree=2))
        held[-1].array[...] = float(k)
    for k, var in enumerate(held):
        assert np.abs(np.asarray(var.array) - k).max() == 0.0
    T = held[-1]
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 0.0
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc(1.0, "Top")
    poisson.tolerance = 1e-10
    poisson.solve(zero_init_guess=True)
    assert poisson.snes.getConvergedReason() > 0
    X = np.asarray(T.coords)
    assert np.abs(np.asarray(T.array)[:, 0, 0] - X[:, 1]).max() < 1e-8     # T = y, exact in P2
