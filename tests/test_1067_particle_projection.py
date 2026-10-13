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


@pytest.mark.parametrize("transport", ["forward_integration_points", "forward_nodes"])
@pytest.mark.parametrize("reconstruction", ["cell", "global"])
def test_both_reconstructions_carry_the_maxwell_shear_stress(transport, reconstruction):
    # eta = G = dt = 1, wall speed 0.5: tau_xy = 1 - 2^-n after n steps
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(-1.0, -0.5), maxCoords=(1.0, 0.5), cellSize=0.125, qdegree=3)
    v = uw.discretisation.MeshVariable(f"U_pp_{transport[8]}{reconstruction}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P_pp_{transport[8]}{reconstruction}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.stress_transport = transport
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


def test_the_forward_nodes_global_projection_reproduces_a_linear_field_from_its_launch_set():
    """The forward-nodes history launched from the nodes and the interior lattice,
    carried by a zero velocity (the arrivals are the launch points), projected
    globally: a linear field comes back exactly at the nodes."""
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2, qdegree=3)
    x, y = mesh.X
    T = uw.discretisation.MeshVariable("T_fwn_lin", mesh, 1, degree=1)
    T.data[:, 0] = 1.0 + 2.0 * np.asarray(T.coords)[:, 0] - 3.0 * np.asarray(T.coords)[:, 1]
    history = uw.systems.ddt.ForwardNodesSemiLagrangian(mesh, T.sym, sympy.Matrix([[0.0, 0.0]]),
                                                        reconstruction="global")
    history.update_pre_solve(0.1)
    expected = 1.0 + 2.0 * np.asarray(T.coords)[:, 0] - 3.0 * np.asarray(T.coords)[:, 1]
    assert np.abs(np.asarray(history.psi_star[0].data)[:, 0] - expected).max() < 1.0e-6
    assert history._fit_overshoot < 1.0e-6


def test_a_quadratic_field_sampled_anywhere_in_the_cells_is_projected_exactly_at_degree_two():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1, qdegree=3)
    pj = ParticleL2Projector(mesh, degree=2)
    pj._build()
    ncell = pj._rows.shape[0]
    rng = np.random.default_rng(11 + uw.mpi.rank)
    lam = rng.dirichlet(np.ones(3), size=(ncell, 9))
    X = np.einsum("cpi,cid->cpd", lam, pj._Xv).reshape(-1, 2)
    cell = np.repeat(np.arange(ncell), 9)
    w = np.repeat(pj.cell_measure / 9, 9)

    def f(P):
        return np.c_[1.0 + 2.0 * P[:, 0] - 3.0 * P[:, 1] + P[:, 0] ** 2 - 0.5 * P[:, 0] * P[:, 1],
                     P[:, 1] ** 2 - P[:, 0]]

    u = pj.project(X, f(X), w, cell, old=np.zeros((pj.n_local_rows, 2)), eps=1.0e-12)
    assert np.abs(u - f(np.asarray(pj._var.coords))).max() < 1.0e-8
    # the element mass matrix integrates to the cell measure
    assert abs(pj._Me.sum() - pj.cell_measure.sum()) < 1.0e-12


def test_the_forward_nodes_global_projection_reproduces_a_quadratic_velocity_from_its_launch_set():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2, qdegree=3)
    v = uw.discretisation.MeshVariable("U_fwn_q2", mesh, 2, degree=2)
    P = np.asarray(v.coords)
    v.data[:, 0] = 1.0 + P[:, 0] ** 2 - 0.5 * P[:, 0] * P[:, 1]
    v.data[:, 1] = P[:, 1] ** 2 - P[:, 0]
    history = uw.systems.ddt.ForwardNodesSemiLagrangian(mesh, v.sym, sympy.Matrix([[0.0, 0.0]]), uw.VarType.VECTOR,
                                                        degree=2, reconstruction="global")
    history.update_pre_solve(0.1)
    assert np.abs(np.asarray(history.psi_star[0].data) - np.asarray(v.data)).max() < 1.0e-6
    assert history._fit_overshoot < 1.0e-6


def test_the_bubble_penalty_acts_on_the_quadratic_content_only():
    """The penalty leaves a linear field exact at any beta. A quadratic field fully
    sampled in every cell loses its curvature progressively: the least-squares fit
    trades a cell's bubble against its neighbours' dofs, so the data hold a bubble
    with an effective weight of only about 0.04 in units of the bubble's own mass,
    and beta 0.01 removes about a fifth of a resolved quadratic, 0.1 about three
    quarters (measured on this mesh: 4.5e-4 and 1.6e-3 of the 2.2e-3 that
    removing it all costs)."""
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1, qdegree=3)
    pj = ParticleL2Projector(mesh, degree=2)
    pj._build()
    ncell = pj._rows.shape[0]
    rng = np.random.default_rng(5 + uw.mpi.rank)
    lam = rng.dirichlet(np.ones(3), size=(ncell, 9))
    X = np.einsum("cpi,cid->cpd", lam, pj._Xv).reshape(-1, 2)
    cell = np.repeat(np.arange(ncell), 9)
    w = np.repeat(pj.cell_measure / 9, 9)
    P = np.asarray(pj._var.coords)
    linear, quadratic = lambda Q: 1.0 + 2.0 * Q[:, 0] - 3.0 * Q[:, 1], lambda Q: Q[:, 0] ** 2 - 0.5 * Q[:, 0] * Q[:, 1]
    errs = {}
    for beta in (0.0, 0.01, 0.1, 1000.0):
        pj.bubble_penalty = beta
        u = pj.project(X, np.c_[linear(X), quadratic(X)], w, cell, old=np.zeros((pj.n_local_rows, 2)), eps=1.0e-12)
        assert np.abs(u[:, 0] - linear(P)).max() < 1.0e-8, beta
        errs[beta] = np.abs(u[:, 1] - quadratic(P)).max()
    assert errs[0.0] < 1.0e-8
    assert 0.15 * errs[1000.0] < errs[0.01] < 0.30 * errs[1000.0]
    assert 0.60 * errs[1000.0] < errs[0.1] < 0.85 * errs[1000.0]
    assert 1.5e-3 < errs[1000.0] < 3.0e-3            # the quadratic content of x^2 on h ~ 0.1 cells


@pytest.mark.skipif(uw.mpi.size > 1, reason="selects rows by position; serial only")
def test_the_deficit_fill_holds_an_emptied_cell_to_the_previous_field():
    """Half the cells receive no points at all: their rows take the previous
    field at full weight (not a 1e-8 pull), while the sampled half is fitted."""
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1, qdegree=3)
    pj = ParticleL2Projector(mesh, degree=1)
    pj._build()
    ncell = pj._rows.shape[0]
    rng = np.random.default_rng(3)
    lam = rng.dirichlet(np.ones(3), size=(ncell, 5))
    X = np.einsum("cpi,cid->cpd", lam, pj._Xv).reshape(-1, 2)
    cell = np.repeat(np.arange(ncell), 5)
    w = np.repeat(pj.cell_measure / 5, 5)
    left = X[:, 0] < 0.5                                   # points only in the left half
    old = np.full((pj.n_local_rows, 1), 3.0)
    u = pj.project(X[left], np.full((left.sum(), 1), 1.0), w[left], cell[left], old=old, eps=1.0e-8)
    P = np.asarray(pj._var.coords)
    # the consistent mass couples neighbours, so the step from the data (1) to the
    # previous field (3) is spread over about two cells and decays geometrically
    # beyond: 0.48 one cell in, 1.6e-2 at x > 0.75, 1.2e-3 at x > 0.85
    assert np.abs(u[P[:, 0] > 0.9] - 3.0).max() < 2.0e-3     # unreached: the previous field
    assert np.abs(u[P[:, 0] < 0.1] - 1.0).max() < 5.0e-3     # sampled: the data (the same tail, 2.9e-3 four cells in)
    # with no points at all the previous field comes back exactly
    u0 = pj.project(np.zeros((0, 2)), np.zeros((0, 1)), np.zeros(0), np.zeros(0, dtype=int), old=old, eps=1.0e-8)
    assert np.abs(u0 - 3.0).max() < 1.0e-9


def test_an_arrival_on_a_shared_face_is_counted_once():
    """A point on a face shared by two cells is offered to both; the global
    projection must weigh it once, so the multiplicity it comes back with is 2
    and the weight split is w/2 per row."""
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, qdegree=3)
    T = uw.discretisation.MeshVariable("T_face", mesh, 1, degree=1)
    history = uw.systems.ddt.ForwardNodesSemiLagrangian(mesh, T.sym, sympy.Matrix([[0.0, 0.0]]),
                                                        reconstruction="global")
    pj = history._global_projector
    pj._build()
    # the midpoint of an interior edge: the edge of cell 0 whose two vertices
    # are both inside the box
    Xv = pj._Xv[0]
    inner = [k for k in range(3) if all(1e-9 < c < 1 - 1e-9 for c in Xv[k])]
    a, b = (inner + [k for k in range(3) if k not in inner])[:2]
    X = 0.5 * (Xv[a] + Xv[b])[None, :]
    arrivals, vw, cells, mult = history._arrivals_by_cell(X, np.array([[7.0, 3.0]]))
    n_cells_sharing = len(np.unique(cells))
    assert n_cells_sharing >= 1 and len(cells) == n_cells_sharing
    assert np.all(mult == n_cells_sharing)
    assert abs((vw[:, 1] / mult).sum() - 3.0) < 1e-12     # the weight column, split once over its takers


def test_the_forward_nodes_launch_weights_sum_to_the_domain():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, qdegree=3)
    T = uw.discretisation.MeshVariable("T_w", mesh, 1, degree=1)
    history = uw.systems.ddt.ForwardNodesSemiLagrangian(mesh, T.sym, sympy.Matrix([[0.0, 0.0]]),
                                                        reconstruction="global")
    w = history._weights_of_launch()
    node_w, lattice_w = history._launch_weights
    assert abs(node_w.sum() + lattice_w.sum() - history._global_projector.cell_measure.sum()) < 1e-12
    assert w.shape[0] == int(history._owned.sum()) + history._interior.shape[0]


def test_the_global_projection_refuses_a_cubic_store():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, qdegree=3)
    with pytest.raises(NotImplementedError, match="degree 3"):
        ParticleL2Projector(mesh, degree=3)


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
