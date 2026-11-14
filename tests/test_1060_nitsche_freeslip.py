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


def _build_freeslip_box(method, res=8):
    """The solver for buoyancy-driven flow in a unit box with free-slip top/bottom.

    Parameters
    ----------
    method : str
        "essential", "penalty", or "nitsche"
    res : int
        Element resolution per side.

    Returns
    -------
    stokes, v, p
        The solver, not yet solved, and its velocity and pressure variables.
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
        stokes.add_nitsche_bc(0.0, "Top")
        stokes.add_nitsche_bc(0.0, "Bottom")
    else:
        raise ValueError(f"Unknown method: {method}")

    stokes.tolerance = 1e-6
    stokes.petsc_options["ksp_type"] = "fgmres"

    return stokes, v, p


def _solve_freeslip_box(method, res=8, reasons=None):
    """Solve the box of :func:`_build_freeslip_box`.

    Parameters
    ----------
    reasons : dict, optional
        If given, the SNES converged reason is stored in it under ``method``.

    Returns
    -------
    v_data : ndarray
        Velocity at P2 nodes.
    p_data : ndarray
        Pressure at P1 nodes.
    coords : ndarray
        P2 node coordinates.
    """
    stokes, v, p = _build_freeslip_box(method, res)
    stokes.solve()
    if reasons is not None:
        reasons[method] = stokes.snes.getConvergedReason()

    return v.data.copy(), p.data.copy(), v.coords.copy()


def _relative_l2_difference(reference, other):
    """The L2 difference of two nodal fields over every rank, relative to the first.

    Both sums are taken over every rank's nodes, so each rank asserts the same
    number. A node held by two ranks is counted by both.
    """
    from mpi4py import MPI

    difference = uw.mpi.comm.allreduce(float(np.sum((reference - other) ** 2)), op=MPI.SUM)
    size = uw.mpi.comm.allreduce(float(np.sum(reference ** 2)), op=MPI.SUM)
    return np.sqrt(difference / size)


def _velocity_and_pressure_unknowns(stokes):
    """The index sets of the velocity and the pressure unknowns of a solved solver."""
    _, index_sets, _ = stokes.dm.createFieldDecomposition()
    return index_sets[0], index_sets[1]


@pytest.fixture(scope="module")
def solved_nitsche_box():
    stokes, _, _ = _build_freeslip_box("nitsche")
    stokes.solve()
    return stokes


@pytest.mark.tier_b
def test_the_continuity_row_carries_the_nitsche_term(solved_nitsche_box):
    """The pressure-velocity block is minus the transpose of the velocity-pressure block.

    The Nitsche condition puts the pressure into the momentum rows on the wall, and
    the matching term belongs in the continuity rows. It is integrated only if the
    wall is registered on the pressure field as well as the velocity field: PETSc
    visits a natural boundary for the field it is registered on and no other. With
    the term missing the two blocks differ by 0.28 on this fixture (#1072).
    """
    stokes = solved_nitsche_box
    jacobian = stokes.snes.getJacobian()[0]
    velocity, pressure = _velocity_and_pressure_unknowns(stokes)

    pressure_velocity = jacobian.createSubMatrix(pressure, velocity)
    velocity_pressure = jacobian.createSubMatrix(velocity, pressure)
    scale = velocity_pressure.norm()
    # in place, on the copy createSubMatrix made
    velocity_pressure.transpose()
    pressure_velocity.axpy(1.0, velocity_pressure)
    mismatch = pressure_velocity.norm()

    assert scale > 0.5
    assert mismatch < 1.0e-12 * scale, (
        f"the pressure-velocity block differs from minus the transpose of the "
        f"velocity-pressure block by {mismatch:.3e} (block norm {scale:.3e})"
    )


@pytest.mark.tier_b
def test_the_continuity_residual_sums_to_zero_in_a_closed_box(solved_nitsche_box):
    """No velocity field has a net continuity residual when no wall lets flow through.

    Summed over the pressure unknowns, the continuity residual is the integral of
    the divergence less the wall-normal velocity the Nitsche walls measure, which is
    the flux through the other walls: zero here, whatever the velocity. That is what
    makes the constant pressure a null vector of the transposed system as well, and
    so the system solvable. Without the Nitsche term in the continuity rows the sum
    is the flux through Top and Bottom (#1072).
    """
    from petsc4py import PETSc

    stokes = solved_nitsche_box
    _, pressure = _velocity_and_pressure_unknowns(stokes)

    state = stokes.snes.getJacobian()[0].createVecRight()
    state.setRandom(PETSc.Random().create(comm=state.comm))
    residual = state.duplicate()
    stokes.snes.computeFunction(state, residual)

    continuity = residual.getSubVector(pressure)
    net, size = continuity.sum(), continuity.norm(PETSc.NormType.NORM_1)
    residual.restoreSubVector(pressure, continuity)

    assert size > 1.0e-3
    assert abs(net) < 1.0e-12 * size, (
        f"the continuity residual sums to {net:.3e} (sum of magnitudes {size:.3e})"
    )


@pytest.mark.tier_b
def test_the_boundary_residual_diagnostic_returns_the_continuity_row_term():
    """``compute_boundary_residual_fields`` reaches the entry on the pressure field.

    For the velocity (0, x(1-x)), which the quadratic elements hold exactly and
    which is zero on the no-slip sides, the continuity-row term of a free-slip
    wall sums to minus the flow out through it: -1/6 on Top and +1/6 on Bottom.
    The sum is over every rank: a rank that holds no facet of the wall returns
    zeros, and takes part in the call and in the reduction.
    """
    from mpi4py import MPI

    stokes, v, _ = _build_freeslip_box("nitsche")
    x = v.coords[:, 0]
    v.array[:, 0, 0] = 0.0
    v.array[:, 0, 1] = x * (1.0 - x)

    for wall, flow_out in (("Top", 1.0 / 6.0), ("Bottom", -1.0 / 6.0)):
        rows = stokes.compute_boundary_residual_fields(wall, residual_field_id=1)
        momentum = uw.mpi.comm.allreduce(
            float(np.abs(rows["velocity"]).max(initial=0.0)), op=MPI.MAX)
        continuity = uw.mpi.comm.allreduce(float(rows["pressure"].sum()), op=MPI.SUM)
        assert momentum == 0.0
        assert abs(continuity + flow_out) < 1.0e-12, (
            f"{wall}: the continuity-row term sums to {continuity:.15e}"
        )


class TestNitscheFreeslip:
    """Compare Nitsche free-slip against essential BC and penalty on a Cartesian box."""

    @pytest.fixture(scope="class")
    def solutions(self):
        """Run all three methods once and cache results."""
        reasons = {}
        essential = _solve_freeslip_box("essential", reasons=reasons)
        penalty = _solve_freeslip_box("penalty", reasons=reasons)
        nitsche = _solve_freeslip_box("nitsche", reasons=reasons)
        return {"essential": essential, "penalty": penalty, "nitsche": nitsche,
                "reasons": reasons}

    def test_the_reference_solves_converge(self, solutions):
        """The essential and penalty solutions the comparisons rest on converged."""
        reasons = solutions["reasons"]
        assert reasons["essential"] > 0, reasons
        assert reasons["penalty"] > 0, reasons

    def test_nitsche_converges(self, solutions):
        """The Nitsche solve reaches its tolerance."""
        assert solutions["reasons"]["nitsche"] > 0, solutions["reasons"]

    def test_nitsche_solution_is_finite(self, solutions):
        """The Nitsche solve leaves a finite velocity field."""
        v, p, coords = solutions["nitsche"]
        assert v.shape[0] > 0
        assert not np.any(np.isnan(v))

    def test_nitsche_matches_essential(self, solutions):
        """The Nitsche velocity is within 1.5e-3 of the essential one at this resolution.

        Measured 1.17e-3 at the default penalty (gamma 12.5). Without the
        Nitsche term in the continuity rows it is 3.36e-3 (#1072), and at
        gamma 10 it is 1.98e-3, so the bound separates all three.
        """
        v_ess, _, _ = solutions["essential"]
        v_nit, _, _ = solutions["nitsche"]

        diff = _relative_l2_difference(v_ess, v_nit)
        print(f"Nitsche vs essential: relative L2 diff = {diff:.4e}")
        assert diff < 1.5e-3, f"Nitsche differs from essential by {diff:.4e}"

    def test_penalty_matches_essential(self, solutions):
        """Penalty velocity should also match essential BC solution."""
        v_ess, _, _ = solutions["essential"]
        v_pen, _, _ = solutions["penalty"]

        diff = _relative_l2_difference(v_ess, v_pen)
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

    @pytest.mark.tier_b
    def test_essential_bc_holds_the_wall_normal_velocity_exactly(self, solutions):
        """Contract: an essential BC holds v.n to machine precision.

        Exactness is a hard baseline — a strong Dirichlet constraint eliminates
        the degree of freedom, so the wall-normal velocity is zero to round-off
        and any departure names a specific defect in how the BC is applied.

        This is the half of the free-slip reasoning that can be asserted. The
        other half — that the WEAK paths leave a finite leak — used to be
        asserted alongside it as `leaks[method] > 1e-5`, i.e. that Nitsche and
        penalty must STAY inaccurate. That assertion would fail if either method
        improved, and a failure could not say which of the fixture, the gamma,
        the penalty coefficient or the method had moved. The measured leaks are
        printed instead: read them, do not gate on them.
        """
        from mpi4py import MPI

        leaks = {}
        for method in ("essential", "penalty", "nitsche"):
            v, _, coords = solutions[method]
            on_top = np.abs(coords[:, 1] - 1.0) < 1e-10
            # over every rank's nodes, so each rank asserts the same numbers (a rank
            # holding no top-wall node used to skip on its own)
            n_top = uw.mpi.comm.allreduce(int(np.count_nonzero(on_top)), op=MPI.SUM)
            assert n_top > 0, "the mesh has no node on the top wall"
            top = uw.mpi.comm.allreduce(
                float(np.max(np.abs(v[on_top, 1]))) if np.any(on_top) else 0.0, op=MPI.MAX)
            scale = uw.mpi.comm.allreduce(
                float(np.max(np.abs(v))) if v.size else 0.0, op=MPI.MAX)
            leaks[method] = top / scale

        print("Relative |v_y| on top: "
              + ", ".join(f"{k}={v:.3e}" for k, v in leaks.items()))

        assert leaks["essential"] < 1.0e-12, (
            f"an essential BC must hold v.n to machine precision; got "
            f"{leaks['essential']:.3e}"
        )
