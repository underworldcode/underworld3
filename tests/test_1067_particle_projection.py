"""The global L2 projection of scattered points onto continuous P1
(ParticleL2Projector), and the forward integration-point history that uses it in
place of its per-cell fit (reconstruction='global')."""
import numpy as np
import pytest
import sympy
import underworld3 as uw
from underworld3.utilities.particle_projection import ParticleL2Projector

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def test_a_linear_field_sampled_anywhere_in_the_cells_is_projected_exactly():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1, qdegree=3)
    pj = ParticleL2Projector(mesh)
    pj._build()
    ncell = pj._rows.shape[0]
    rng = np.random.default_rng(7 + uw.mpi.rank)
    lam = rng.dirichlet(np.ones(3), size=(ncell, 5))
    X = np.einsum("cpi,cid->cpd", lam, pj._Xv).reshape(-1, 2)
    cell = np.repeat(np.arange(ncell), 5)
    w = np.repeat(pj.cell_measure / 5, 5)

    def f(P):
        return np.c_[1.0 + 2.0 * P[:, 0] - 3.0 * P[:, 1], 0.5 * P[:, 0]]

    u = pj.project(X, f(X), w, cell, old=np.zeros((pj.n_local_rows, 2)), eps=1.0e-12)
    assert np.abs(u - f(np.asarray(pj._var.coords))).max() < 1.0e-9


def test_a_node_no_point_reached_keeps_its_previous_value():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, qdegree=3)
    pj = ParticleL2Projector(mesh)
    pj._build()
    old = np.full((pj.n_local_rows, 1), 7.0)
    u = pj.project(np.zeros((0, 2)), np.zeros((0, 1)), np.zeros(0), np.zeros(0, dtype=int), old=old, eps=1.0e-8)
    assert np.abs(u - 7.0).max() < 1.0e-9


@pytest.mark.parametrize("reconstruction", ["cell", "global"])
def test_both_reconstructions_carry_the_maxwell_shear_stress(reconstruction):
    # eta = G = dt = 1, wall speed 0.5: tau_xy = 1 - 2^-n after n steps
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(-1.0, -0.5), maxCoords=(1.0, 0.5), cellSize=0.125, qdegree=3)
    v = uw.discretisation.MeshVariable(f"U_pp_{reconstruction}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P_pp_{reconstruction}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.stress_transport = "forward_integration_points"
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=1, integrator="bdf")
    cm = stokes.constitutive_model
    cm.Parameters.shear_viscosity_0 = 1.0
    cm.Parameters.shear_modulus = 1.0
    cm.Parameters.dt_elastic = 1.0
    stokes.add_dirichlet_bc((0.5, 0.0), "Top")
    stokes.add_dirichlet_bc((-0.5, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    stokes.tolerance = 1.0e-8
    stokes.DFDt.reconstruction = reconstruction
    for n in range(1, 5):
        stokes.solve(timestep=1.0, zero_init_guess=False)
        xy = np.asarray(stokes.DFDt.psi_star[0].data)[:, 2]
        assert np.abs(xy - (1.0 - 0.5 ** n)).max() < 1.0e-6


@pytest.mark.parametrize("reconstruction", ["cell", "global"])
def test_both_reconstructions_hold_a_linear_stress_at_the_stores_own_dofs(reconstruction):
    """A stress varying linearly in space, read at the launch points and
    reconstructed, is reproduced exactly at the store's dofs, which sit INSIDE
    each cell (a uniform field cannot tell a dof at a vertex from one inside the
    cell; writing vertex values into interior dofs steepens every cell)."""
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2, qdegree=3)
    v = uw.discretisation.MeshVariable(f"U_pp_lin_{reconstruction}", mesh, 2, degree=2)
    history = uw.systems.ddt.ForwardIntegrationPointsSemiLagrangian(
        mesh, sympy.Matrix.zeros(2, 2), v.sym, uw.VarType.SYM_TENSOR, reconstruction=reconstruction)

    def stress(P):
        return np.c_[1.0 + P[:, 0], 2.0 - 3.0 * P[:, 1], 0.5 * P[:, 0] + 0.25 * P[:, 1]]

    history._fit_arrivals(history._launch, stress(history._launch), cell=history._launch_cell)
    dofs = np.asarray(history.psi_star[0].coords_nd)
    # (the global projection's weak pull towards the previous, zero, field is 1e-8 of the mass)
    assert np.abs(np.asarray(history.psi_star[0].data) - stress(dofs)).max() < 1.0e-6


def test_the_global_reconstruction_refuses_the_per_cell_limiter():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, qdegree=3)
    v = uw.discretisation.MeshVariable("U_pp_lim", mesh, 2, degree=2)
    history = uw.systems.ddt.ForwardIntegrationPointsSemiLagrangian(
        mesh, sympy.Matrix.zeros(2, 2), v.sym, uw.VarType.SYM_TENSOR, fit_limiter=True)
    with pytest.raises(ValueError, match="per-cell fit"):
        history.reconstruction = "global"
