"""
Regression test for ``uw.function.evaluate`` on tensor-valued expressions
containing derivatives.

Background
----------
``_project_to_work_variable`` in ``_function.pyx`` is reached from
``evaluate_nd`` whenever the input expression contains derivatives that
can't be evaluated directly at points (e.g. ``grad(u)``, ``strain_tensor``,
viscous flux). For non-scalar expressions it allocates a multi-component
work mesh variable and projects each component into it.

Until this PR the matrix branch was dead code with a ``TODO(BUG)`` marker
— it called ``MeshVariable(..., num_components=rows*cols)`` (flat int)
which can't infer ``vtype``, raising ``ValueError`` on every invocation.
That bubbled up to ``DDt.update_pre_solve``'s outer ``except Exception``,
forcing a wasteful try-fail-fallback on every NavierStokes step.

The fix uses ``SNES_MultiComponent_Projection`` against a flat
``(1, Nc) MATRIX`` work var, then fans the flat result into a tensor-shaped
work var so the caller's ``work_var.sym`` keeps the original shape.

See: issue #180, PR comment by @ss2098 pinpointing the SLCN projection
path.
"""
import numpy as np
import pytest
import sympy
import underworld3 as uw


@pytest.mark.level_1
@pytest.mark.tier_a
def test_evaluate_strain_tensor_returns_correct_shape():
    """``evaluate(strain_tensor, pts)`` returns shape (n_pts, dim, dim)."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2
    )
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)

    # Pure rotation: strain tensor is identically zero (antisymmetric gradient).
    v.array[:, 0, 0] = v.coords[:, 1]
    v.array[:, 0, 1] = -v.coords[:, 0]

    strain = mesh.vector.strain_tensor(v.sym)
    assert strain.shape == (2, 2)

    pts = np.array([[0.5, 0.5], [0.3, 0.7]])
    result = np.asarray(uw.function.evaluate(strain, pts))

    assert result.shape == (2, 2, 2), f"expected (n_pts, 2, 2), got {result.shape}"
    # Strain rate of pure rotation is zero up to projection noise.
    assert np.allclose(result, 0.0, atol=1e-6)


@pytest.mark.level_1
@pytest.mark.tier_a
def test_evaluate_shear_field_strain_nonzero():
    """A simple shear field produces a known non-zero off-diagonal strain."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2
    )
    v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)

    # Simple shear u = (y, 0) — strain has off-diagonal entries = 0.5.
    v.array[:, 0, 0] = v.coords[:, 1]
    v.array[:, 0, 1] = 0.0

    strain = mesh.vector.strain_tensor(v.sym)
    result = np.asarray(uw.function.evaluate(strain, np.array([[0.5, 0.5]])))
    # Off-diagonal entries should be ~0.5 (projection has some smoothing).
    assert abs(result[0, 0, 1] - 0.5) < 0.05
    assert abs(result[0, 1, 0] - 0.5) < 0.05


@pytest.mark.level_2
@pytest.mark.tier_a
def test_navier_stokes_solve_does_not_trigger_ddt_fallback():
    """A NavierStokes solve completes without entering the DDt projection fallback.

    The DDt fallback in ``update_pre_solve`` exists for expressions where
    ``uw.function.evaluate`` raises. Once tensor evaluate works, the
    fallback should not be needed for the NS viscous flux.
    """
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 8, regular=False
    )

    v = uw.discretisation.MeshVariable("u", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1, continuous=True)

    ns = uw.systems.NavierStokes(
        mesh, velocityField=v, pressureField=p, rho=1.0, order=2
    )
    ns.constitutive_model = uw.constitutive_models.ViscousFlowModel
    ns.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    ns.bodyforce = sympy.Matrix([0, 0]).T
    ns.add_dirichlet_bc((1.0, 0.0), "Top")
    ns.add_dirichlet_bc((0.0, 0.0), "Bottom")
    ns.add_dirichlet_bc((0.0, 0.0), "Left")
    ns.add_dirichlet_bc((0.0, 0.0), "Right")

    # Pre-fix: this raised ShapeError("(1, 3) + (2, 2)") because
    # _project_to_work_variable failed (ValueError) and the DDt fallback
    # used the wrong source shape. Post-fix: evaluate succeeds, fallback
    # isn't reached, solve completes.
    ns.solve(timestep=0.01, zero_init_guess=True)
    ns.solve(timestep=0.01, zero_init_guess=False)
