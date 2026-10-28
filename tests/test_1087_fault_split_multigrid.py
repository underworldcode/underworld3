"""Split-level multigrid for split-node faults (``utilities.fault_mg``).

``mesh.add_fault`` on an add-on mesh carries the fault through every add-on
level of the inherited multigrid tail and leaves the static background
untouched. The coarse spaces then contain the discontinuity, and the velocity
solve stays mesh-independent as the fault is refined — measured on this box
(2026-10-13): 6, 5 and 4 velocity iterations at k = 2, 3, 4 add-on levels
against 13, 18 and 23 with the welded tail, the peak slip identical to six
figures. Design note: ``docs/developer/design/FAULT_MULTIGRID_HIERARCHIES.md``.
"""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

TIP_A, TIP_B = np.array([0.30, 0.45]), np.array([0.70, 0.55])
H_BASE = 0.05


@pytest.fixture(scope="module")
def base():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1, refinement=1,
        qdegree=2)


def _adapted(base, k, centre=(0.5, 0.5)):
    """The add-on mesh of ``k`` levels toward the fault, and the fault trace."""
    half = (TIP_B - TIP_A) / 2
    a, b = np.asarray(centre) - half, np.asarray(centre) + half
    pts = np.column_stack([np.linspace(a[0], b[0], 9), np.linspace(a[1], b[1], 9),
                           np.zeros(9)])
    guide = uw.meshing.Surface(f"guide_{k}_{centre[0]:.2f}", base, pts)
    guide.discretize()
    child = base.adapt(guide.refinement_metric_function(
        h_near=H_BASE / 2**k, h_far=H_BASE, width=0.08, profile="linear"),
        max_levels=k)
    return child, np.vstack([a, b])


def _solve(mesh, law=0):
    x, y = mesh.X
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
    stokes.tolerance = 1.0e-6
    stokes.petsc_use_pressure_nullspace = True
    for side in ("Top", "Bottom", "Left", "Right"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0), side)
    stokes.add_fault_bc(law, "Flt")
    stokes.solve()
    info = stokes._rotated_freeslip_info
    from underworld3.utilities.fault_contact import fault_slip
    _s, V, _leak = fault_slip(stokes, "Flt", info)
    return info, float(np.abs(V).max())


def test_the_add_on_levels_are_split_and_the_background_is_not(base):
    from underworld3.utilities.fault_mg import is_split_level

    child, trace = _adapted(base, 2)
    mesh = child.add_fault([("Flt", trace)])
    tail = mesh._custom_mg_coarse_meshes
    background = base._coarse_level_meshes()
    assert len(tail) == len(background) + 1
    # the static background levels are the base's own wraps, untouched
    for level, wrap in zip(tail, background):
        assert level is wrap and not is_split_level(level)
    # every add-on level and the finest carry the fault's pairing
    assert all(is_split_level(level) for level in tail[len(background):])
    assert is_split_level(mesh)
    # the welded tail is still available
    welded = child.add_fault([("Flt", trace)], split_levels=False)
    assert not any(is_split_level(level) for level in welded._custom_mg_coarse_meshes)


def test_iterations_fall_with_refinement_and_the_slip_is_unchanged(base):
    its, slips = {}, {}
    for k in (2, 3):
        child, trace = _adapted(base, k)
        info, slip = _solve(child.add_fault([("Flt", trace)]))
        assert info["velocity_pc"] == "custom-FMG" and info["converged"], info
        its[k] = info["vel_its_last"][-1]
        slips[k] = slip
    # measured 6 and 5; the welded tail takes 13 and 18 and grows with k
    assert its[2] <= 8 and its[3] <= 8, its
    assert its[3] <= its[2] + 1, its
    child, trace = _adapted(base, 3)
    info_w, slip_w = _solve(child.add_fault([("Flt", trace)], split_levels=False))
    assert info_w["vel_its_last"][-1] >= its[3] + 6, (info_w["vel_its_last"], its)
    assert abs(slip_w - slips[3]) < 1.0e-4 * slip_w, (slip_w, slips[3])


def test_a_viscous_fault_solves_in_as_few_iterations(base):
    child, trace = _adapted(base, 3)
    info, slip = _solve(child.add_fault([("Flt", trace)]), law=1.0)
    assert info["converged"] and info["vel_its_last"][-1] <= 8, info["vel_its_last"]
    assert slip > 0.1


def test_a_moving_fault_re_splits_the_levels_at_every_position(base):
    for cx in (0.35, 0.5, 0.65):
        child, trace = _adapted(base, 2, centre=(cx, 0.5))
        mesh = child.add_fault([("Flt", trace)])
        info, slip = _solve(mesh)
        assert info["converged"] and info["vel_its_last"][-1] <= 8, (cx, info["vel_its_last"])
        assert slip > 0.1, (cx, slip)


def test_the_cell_located_transfer_keeps_each_side_on_its_own_side(base):
    """A coarse field equal to +1 on one side of the fault and -1 on the other
    prolongs to the same values on the fine level, at the slit nodes included: a
    node on the slit interpolates from its own side only. The side of a node is
    that of a cell listing it — a slit node is listed only by its own side's
    cells — so no pairing orientation is assumed. Nodes within two coarse cells of
    a tip are left out (the tip vertex is shared by both sides; measured, the
    disagreement reaches 0.033 from a tip on this hierarchy). And a quadratic
    field is reproduced exactly between the split levels."""
    from underworld3.utilities.fault_mg import cell_located_prolongation

    child, trace = _adapted(base, 2)
    mesh = child.add_fault([("Flt", trace)])
    coarse = mesh._custom_mg_coarse_meshes[-1]
    assert getattr(coarse, "_fault_point_pairs", None)
    degree, continuous = 2, True
    P = cell_located_prolongation(coarse, mesh, degree, continuous)
    Xc = np.asarray(coarse._get_coords_for_basis(degree, continuous))
    Xf = np.asarray(mesh._get_coords_for_basis(degree, continuous))
    assert P.shape == (Xf.shape[0], Xc.shape[0])
    # partition of unity, bounded weights (the clamp)
    assert np.abs(np.asarray(P.sum(axis=1)).ravel() - 1.0).max() < 1.0e-10
    assert np.abs(P.data).max() <= 1.0 + 1.0e-10

    a, b = trace
    n = np.array([-(b - a)[1], (b - a)[0]])
    n /= np.linalg.norm(n)

    def side_by_cell(m, X):
        cn = np.asarray(m._cell_node_indices(degree, continuous))
        cen = X[cn].mean(axis=1)                      # P2 node centroid = cell centroid
        side = np.zeros(X.shape[0])
        for k in range(cn.shape[0]):
            side[cn[k]] = np.sign((cen[k] - a) @ n)
        return side

    fc = side_by_cell(coarse, Xc)
    ff = P @ fc
    expected = side_by_cell(mesh, Xf)
    away_from_tips = np.minimum(np.linalg.norm(Xf - a, axis=1),
                                np.linalg.norm(Xf - b, axis=1)) > 0.05
    on_slit = np.abs((Xf - a) @ n) < 1.0e-10
    assert (on_slit & away_from_tips).sum() > 20
    assert np.abs(ff - expected)[away_from_tips].max() < 1.0e-10

    # the levels are nested away from the trace (one adapt call's bisections),
    # so a quadratic is reproduced exactly there; within a coarse cell or two of
    # the trace each cut pulls vertices onto it, the levels are not nested, and
    # the clamp bounds the weights at the cost of exactness (measured: 0.018 at
    # worst, 3e-4 at 0.03 from the trace, 3e-14 beyond 0.04)
    q = lambda X: (1.0 + 2.0 * X[:, 0] - X[:, 1] + 3.0 * X[:, 0]**2 - X[:, 0] * X[:, 1]
                   + 0.5 * X[:, 1]**2)
    err = np.abs(P @ q(Xc) - q(Xf))
    off_trace = np.abs((Xf - a) @ n) > 0.05
    assert err[off_trace].max() < 1.0e-9
    assert err.max() < 0.05
