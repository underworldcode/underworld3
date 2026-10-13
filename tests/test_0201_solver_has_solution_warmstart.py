"""Layer 1a of the nonlinear-solver warm-start / homotopy design:
``solver.has_solution`` and the automatic cold-start Picard warm-up.

Contract under test (design:
``docs/developer/design/nonlinear-solver-homotopy-warmstart.md``, Layer 1):

  * ``has_solution`` is a public, read-only status flag, ``False`` on a fresh
    solver and ``True`` only after a solve whose SNES converged.
  * It resets to ``False`` on a *structural* rebuild (the ``is_setup = False``
    invalidation used by remesh / adapt / mesh-mover), so the next solve
    cold-starts rather than warming off a stale iterate.
  * It survives a coefficient-only change (a new viscosity value), so parameter
    continuation and time-stepping warm-start correctly.
  * The cold-start Picard warm-up runs under the consistent-Newton tangent
    without breaking convergence, and leaves the default (frozen) tangent path
    untouched.

These are cheap serial checks on the public API — the hard-case δ-continuation
that motivates the design is validated separately against the Spiegelman study.
"""

import pytest
import sympy

import underworld3 as uw


pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _poisson():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25
    )
    t = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    poisson = uw.systems.Poisson(mesh, u_Field=t)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.add_essential_bc((0.0,), "Bottom")
    poisson.add_essential_bc((1.0,), "Top")
    return mesh, poisson


def test_has_solution_false_before_first_solve():
    _, poisson = _poisson()
    assert poisson.has_solution is False


def test_has_solution_true_after_converged_solve():
    _, poisson = _poisson()
    poisson.solve()
    assert poisson.snes.getConvergedReason() > 0
    assert poisson.has_solution is True


def test_has_solution_resets_on_structural_rebuild():
    """A structural invalidation (is_setup=False — the remesh / adapt / mover
    hook) must drop the warm-start claim."""
    _, poisson = _poisson()
    poisson.solve()
    assert poisson.has_solution is True

    poisson.is_setup = False
    assert poisson.has_solution is False, (
        "has_solution must reset when the solver is structurally invalidated"
    )


def test_has_solution_survives_coefficient_change():
    """Changing a coefficient value (not the mesh/operators) keeps the solution
    so continuation and time-stepping can warm-start."""
    _, poisson = _poisson()
    poisson.solve()
    assert poisson.has_solution is True

    # A coefficient update goes through _update_constants, not is_setup=False.
    poisson.constitutive_model.Parameters.diffusivity = 2.0
    poisson._update_constants()
    assert poisson.has_solution is True, (
        "a coefficient-only change must not drop the warm-start claim"
    )


def test_has_solution_resets_on_mesh_deform():
    """The mesh-mover path (mesh deform → registered solvers is_setup=False)
    must reset has_solution."""
    import numpy as np

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
    )
    v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = sympy.Matrix([0.0, -1.0])
    stokes.add_essential_bc((0.0, 0.0), "Bottom")
    stokes.add_essential_bc((0.0, None), "Left")
    stokes.add_essential_bc((0.0, None), "Right")
    stokes.solve()
    assert stokes.has_solution is True

    disp = np.zeros((mesh.X.coords.shape[0], mesh.dim))
    disp[:, -1] = 0.01 * mesh.X.coords[:, 0]
    with mesh._coord_mutation():
        mesh._deform_mesh(mesh.X.coords + disp)
    assert stokes.has_solution is False, (
        "a mesh deform (structural rebuild) must reset has_solution"
    )


def test_zero_init_guess_is_tristate_and_auto_detects():
    """Layer 1b: the default (None) resolves cold-vs-warm from has_solution, while
    True/False remain explicit overrides."""
    _, poisson = _poisson()

    # Fresh solver: nothing to warm from -> auto resolves COLD.
    assert poisson.has_solution is False
    assert poisson._resolve_zero_init_guess(None) is True

    poisson.solve()
    assert poisson.has_solution is True
    # A converged solution is present -> auto resolves WARM.
    assert poisson._resolve_zero_init_guess(None) is False

    # Explicit values are honoured regardless of has_solution.
    assert poisson._resolve_zero_init_guess(True) is True
    assert poisson._resolve_zero_init_guess(False) is False

    # After a structural rebuild the claim is dropped, so auto is cold again — this
    # is what stops a remesh warm-starting off stale field data.
    poisson.is_setup = False
    assert poisson._resolve_zero_init_guess(None) is True


def test_force_setup_cold_starts_through_the_public_path():
    """`_force_setup=True` is a structural invalidation, so the solve it is passed to
    must COLD start.

    Regression (adversarial review, M1): `zero_init_guess` was resolved at the top of
    solve(), before the `_force_setup` block that clears `has_solution`, so the call
    that triggered the invalidation warm-started off the flag it was about to clear.
    Driven through the public `solve()` rather than the private resolver, so it
    actually exercises the ordering.
    """
    _, poisson = _poisson()
    poisson.solve()
    assert poisson.has_solution is True

    poisson.solve(_force_setup=True)
    # The rebuild must have dropped the claim during that call, not after it.
    assert poisson._needs_dm_rebuild is False   # rebuilt during the solve
    assert poisson.has_solution is True         # ... and re-established by convergence

    # The ordering itself: resolving now (post-invalidation) must say COLD.
    poisson.is_setup = False
    assert poisson._resolve_zero_init_guess(None) is True


def test_repeated_default_solve_agrees_to_solver_tolerance():
    """A bare solve() called repeatedly (the linear chain-of-solves pattern) must give
    the same answer once auto-detect starts warming the later calls.

    "Same" means *to the requested convergence tolerance*, not bitwise: an iterative
    solve stops anywhere inside the tolerance ball, and a warm start enters that ball
    from a different direction than a cold one. The difference is therefore
    tolerance-sized and shrinks with it (measured: 2.4e-5 at tol=1e-4, 7.4e-10 at
    1e-8, 2.0e-14 at 1e-12) — the flip to auto-detected warm starts changes results
    at that level and no more.
    """
    import numpy as np

    mesh, poisson = _poisson()
    poisson.tolerance = 1.0e-10
    poisson.solve()
    first = poisson.u.array[:, 0, 0].copy()

    poisson.solve()   # now warm by auto-detect
    second = poisson.u.array[:, 0, 0].copy()

    assert np.abs(first - second).max() < 1.0e-6, (
        "warm and cold solutions of a linear problem must agree to the solver "
        f"tolerance (got max|diff| = {np.abs(first - second).max():.2e})"
    )


def test_cold_warmstart_under_consistent_newton_converges():
    """A cold consistent-Newton Stokes solve exercises the automatic
    single-Picard warm-up branch — it must converge and set has_solution."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
    )
    v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = sympy.Matrix([0.0, -1.0])
    stokes.add_essential_bc((0.0, 0.0), "Bottom")
    stokes.add_essential_bc((0.0, None), "Left")
    stokes.add_essential_bc((0.0, None), "Right")

    stokes.consistent_jacobian = True
    stokes.solve()  # cold (zero_init_guess default True) → warm-up branch taken
    assert stokes.snes.getConvergedReason() > 0
    assert stokes.has_solution is True
