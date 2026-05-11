"""
Regression test for the NavierStokesSLCN DFDt projection source shape.

When ``SNES_MultiComponent_Projection`` was wired into the ``SemiLagrangian``
DDt path, the ``psi_fn`` setter and ``_setup_projections`` were updated to
flatten the source tensor to a ``(1, Nc)`` row matrix via
``_build_projection_source``. The fallback path inside ``update_pre_solve``
(taken when ``uw.function.evaluate`` raises on expressions containing
derivatives — which is the NavierStokes viscous flux every step) missed
the migration and assigned ``self.psi_fn`` directly, producing:

    sympy.matrices.exceptions.ShapeError:
        Matrix size mismatch: (1, 3) + (2, 2).

See: https://github.com/underworldcode/underworld3/issues/180

The test does one ``solve(timestep=dt)`` of NavierStokesSLCN on a tiny mesh.
Before the fix this raises ``ShapeError`` from the DDt fallback. After the
fix it converges normally.
"""
import pytest
import sympy
import underworld3 as uw


@pytest.mark.level_2
@pytest.mark.tier_a
def test_navier_stokes_slcn_solve_does_not_raise_shape_error():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=1 / 8,
        regular=False,
    )

    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True)

    ns = uw.systems.NavierStokes(
        mesh,
        velocityField=v,
        pressureField=p,
        rho=1.0,
        order=2,
    )
    ns.constitutive_model = uw.constitutive_models.ViscousFlowModel
    ns.constitutive_model.Parameters.shear_viscosity_0 = 1.0

    ns.bodyforce = sympy.Matrix([0, 0]).T
    ns.add_dirichlet_bc((1.0, 0.0), "Top")
    ns.add_dirichlet_bc((0.0, 0.0), "Bottom")
    ns.add_dirichlet_bc((0.0, 0.0), "Left")
    ns.add_dirichlet_bc((0.0, 0.0), "Right")

    # Take a single step. Pre-fix this raised ShapeError("(1, 3) + (2, 2)")
    # from the DDt projection fallback. Post-fix it just converges.
    ns.solve(timestep=0.01, zero_init_guess=True)
