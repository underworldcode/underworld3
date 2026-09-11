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
        stokes.add_nitsche_bc(0.0, "Top", gamma=10.0)
        stokes.add_nitsche_bc(0.0, "Bottom", gamma=10.0)
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

    def test_nitsche_constrains_the_wall_normal_velocity(self, solutions):
        """The Nitsche term is wired up and acting on the wall-normal component.

        This is a WIRING check, not an accuracy claim. Nitsche is a weak
        constraint: it does not drive v.n to machine precision and is not
        supposed to (see docs/developer/subsystems/rotated-freeslip.md, which is
        why rotated strong free-slip exists). How small the residual leak is
        depends on gamma, the mesh, the forcing and the viscosity, so any tight
        number here would be a property of this fixture rather than of the code.

        The bound is therefore deliberately loose — it fails if the constraint
        is absent or has no effect, and passes for any configuration in which it
        is doing its job. Do NOT tighten it to track a measured value: the
        previous version of this test asserted an ABSOLUTE 1e-4, which scaled
        with the buoyancy forcing rather than with anything about Nitsche, and
        sat a fraction of a percent from failing for unrelated reasons.
        """
        v, _, coords = solutions["nitsche"]
        scale = np.max(np.abs(v))

        for name, on_wall in (("top", np.abs(coords[:, 1] - 1.0) < 1e-10),
                              ("bottom", np.abs(coords[:, 1]) < 1e-10)):
            if not np.any(on_wall):
                continue
            leak = np.max(np.abs(v[on_wall, 1])) / scale
            print(f"Nitsche relative |v_y| on {name}: {leak:.3e}")
            assert leak < 0.1, (
                f"wall-normal velocity on {name} is {leak:.3e} of the velocity "
                "scale — the free-slip constraint is not being applied"
            )

    @pytest.mark.tier_c
    def test_constraint_strength_ordering_characterisation(self, solutions):
        """Characterisation: strong is exact, both weak paths leak; tier C.

        This test CAN fail because the code got better, and that is why it is
        tier C: a failure here demands an explanation, not a revert. It is not
        a contract, and nothing should be reverted to make it pass.

        It exists because the ordering is what the free-slip rulings rest on —
        an essential BC (and a rotated strong free-slip) holds v.n to machine
        precision, while Nitsche and penalty are weak constraints that leave a
        finite leak. If a weak path starts coming out exact, or the strong path
        stops being exact, the documented reasoning in
        docs/developer/subsystems/rotated-freeslip.md needs revisiting and
        somebody should say why.

        The numbers are a characterisation of THIS fixture, measured
        2026-09-12 at res=8: essential 0.0, penalty 1.5e-3, nitsche 5.7e-3
        relative to the velocity scale. They are deliberately not a ranking —
        which of the two weak methods leaks less is problem-dependent, moves
        with gamma and the penalty coefficient, and is not asserted here.
        """
        leaks = {}
        for method in ("essential", "penalty", "nitsche"):
            v, _, coords = solutions[method]
            on_top = np.abs(coords[:, 1] - 1.0) < 1e-10
            if not np.any(on_top):
                pytest.skip("no nodes on the top wall in this partition")
            leaks[method] = np.max(np.abs(v[on_top, 1])) / np.max(np.abs(v))

        print("Relative |v_y| on top: "
              + ", ".join(f"{k}={v:.3e}" for k, v in leaks.items()))

        assert leaks["essential"] < 1.0e-12, (
            f"an essential BC is expected to hold v.n to machine precision; got "
            f"{leaks['essential']:.3e}. This is the contract half of this test."
        )
        for method in ("penalty", "nitsche"):
            assert leaks[method] > 1.0e-5, (
                f"{method} leaked only {leaks[method]:.3e} — a weak constraint "
                "reaching machine precision is GOOD NEWS and a change in "
                "behaviour. Explain it and re-characterise; do not revert to "
                "make this pass."
            )
