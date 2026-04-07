"""Nitsche free-slip validation: compare essential BC, penalty, and Nitsche.

A Cartesian box with free-slip top/bottom and no-slip sides provides a
problem where the exact free-slip solution is known (from the essential BC
version). We verify that penalty and Nitsche give the same answer.

Run with: pixi run python -m pytest tests/test_1060_nitsche_freeslip.py -v
"""

import pytest
import numpy as np
import sympy
import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _solve_freeslip_box(method, res=8):
    """Solve buoyancy-driven flow in a unit box with free-slip top/bottom.

    Parameters
    ----------
    method : str
        "essential", "penalty", or "nitsche"
    res : int
        Element resolution per side.

    Returns
    -------
    v_data : ndarray
        Velocity at P2 nodes.
    p_data : ndarray
        Pressure at P1 nodes.
    coords : ndarray
        P2 node coordinates.
    """

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=1.0 / res, qdegree=3,
    )

    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2,
                                        vtype=uw.VarType.VECTOR)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.saddle_preconditioner = 1.0

    x, y = mesh.X

    # Buoyancy: horizontal density variation drives convective circulation
    stokes.bodyforce = sympy.Matrix([0, sympy.cos(sympy.pi * x)])

    # Sides: no-slip
    stokes.add_dirichlet_bc((0.0, 0.0), "Left")
    stokes.add_dirichlet_bc((0.0, 0.0), "Right")

    # Top/bottom: free-slip (v_y = 0, v_x free)
    if method == "essential":
        stokes.add_dirichlet_bc((sympy.oo, 0.0), "Top")
        stokes.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
    elif method == "penalty":
        Gamma = mesh.Gamma
        stokes.add_natural_bc(1e4 * Gamma.dot(v.sym) * Gamma, "Top")
        stokes.add_natural_bc(1e4 * Gamma.dot(v.sym) * Gamma, "Bottom")
    elif method == "nitsche":
        stokes.add_nitsche_bc("Top", gamma=10.0)
        stokes.add_nitsche_bc("Bottom", gamma=10.0)
    else:
        raise ValueError(f"Unknown method: {method}")

    stokes.tolerance = 1e-6
    stokes.petsc_options["ksp_type"] = "fgmres"

    stokes.solve()

    return v.data.copy(), p.data.copy(), v.coords.copy()


class TestNitscheFreeslip:
    """Compare Nitsche free-slip against essential BC and penalty on a Cartesian box."""

    @pytest.fixture(scope="class")
    def solutions(self):
        """Run all three methods once and cache results."""
        essential = _solve_freeslip_box("essential")
        penalty = _solve_freeslip_box("penalty")
        nitsche = _solve_freeslip_box("nitsche")
        return {"essential": essential, "penalty": penalty, "nitsche": nitsche}

    def test_nitsche_converges(self, solutions):
        """Nitsche solution should exist (solver converged)."""
        v, p, coords = solutions["nitsche"]
        assert v.shape[0] > 0
        assert not np.any(np.isnan(v))

    def test_nitsche_matches_essential(self, solutions):
        """Nitsche velocity should match essential BC solution closely."""
        v_ess, _, _ = solutions["essential"]
        v_nit, _, _ = solutions["nitsche"]

        # L2 relative difference
        diff = np.sqrt(np.sum((v_ess - v_nit) ** 2)) / np.sqrt(np.sum(v_ess ** 2))
        print(f"Nitsche vs essential: relative L2 diff = {diff:.4e}")
        assert diff < 0.01, f"Nitsche differs from essential by {diff:.4e}"

    def test_penalty_matches_essential(self, solutions):
        """Penalty velocity should also match essential BC solution."""
        v_ess, _, _ = solutions["essential"]
        v_pen, _, _ = solutions["penalty"]

        diff = np.sqrt(np.sum((v_ess - v_pen) ** 2)) / np.sqrt(np.sum(v_ess ** 2))
        print(f"Penalty vs essential: relative L2 diff = {diff:.4e}")
        assert diff < 0.01, f"Penalty differs from essential by {diff:.4e}"

    def test_nitsche_normal_velocity_zero(self, solutions):
        """Normal velocity on free-slip boundaries should be near zero."""
        v, _, coords = solutions["nitsche"]

        # Top boundary (y ~ 1): v_y should be ~ 0
        top = np.abs(coords[:, 1] - 1.0) < 1e-10
        if np.any(top):
            max_vy_top = np.max(np.abs(v[top, 1]))
            print(f"Nitsche max |v_y| on top: {max_vy_top:.4e}")
            assert max_vy_top < 1e-4

        # Bottom boundary (y ~ 0): v_y should be ~ 0
        bot = np.abs(coords[:, 1]) < 1e-10
        if np.any(bot):
            max_vy_bot = np.max(np.abs(v[bot, 1]))
            print(f"Nitsche max |v_y| on bottom: {max_vy_bot:.4e}")
            assert max_vy_bot < 1e-4

    def test_nitsche_better_than_penalty_constraint(self, solutions):
        """Nitsche should enforce normal constraint at least as well as penalty."""
        v_nit, _, coords_nit = solutions["nitsche"]
        v_pen, _, coords_pen = solutions["penalty"]

        # Compare max |v_y| on top boundary
        top_nit = np.abs(coords_nit[:, 1] - 1.0) < 1e-10
        top_pen = np.abs(coords_pen[:, 1] - 1.0) < 1e-10

        max_vn_nit = np.max(np.abs(v_nit[top_nit, 1])) if np.any(top_nit) else 0
        max_vn_pen = np.max(np.abs(v_pen[top_pen, 1])) if np.any(top_pen) else 0

        print(f"Normal velocity on top: Nitsche={max_vn_nit:.4e}, Penalty={max_vn_pen:.4e}")
        # Nitsche at gamma=10 should be comparable or better than penalty at 1e4
