# %%
import underworld3 as uw
import numpy as np
import pytest

# Physics solver tests - full solver execution
pytestmark = pytest.mark.level_3
from sympy import Piecewise
import sympy


# ### Set up variables of the model
res = 25

# ### Set up the mesh
minX, maxX = -1.0, 0.0
minY, maxY = -1.0, 0.0

# +
### Quads
meshStructuredQuadBox = uw.meshing.StructuredQuadBox(
    elementRes=(int(res), int(res)),
    minCoords=(minX, minY),
    maxCoords=(maxX, maxY),
    qdegree=2,
)

### Tris
meshSimplex_box_irregular = uw.meshing.UnstructuredSimplexBox(
    minCoords=(minX, minY),
    maxCoords=(maxX, maxY),
    cellSize=1 / res,
    qdegree=2,
    regular=False,
)

meshSimplex_box_regular = uw.meshing.UnstructuredSimplexBox(
    minCoords=(minX, minY),
    maxCoords=(maxX, maxY),
    cellSize=1 / res,
    qdegree=2,
    regular=True,
)


# %%
@pytest.mark.parametrize(
    "mesh",
    [
        meshStructuredQuadBox,
        meshSimplex_box_irregular,
        meshSimplex_box_regular,
    ],
)
def test_Darcy_boxmesh_G_and_noG(mesh):
    """Test Darcy flow with layered permeability, with and without gravity."""
    # Reset the mesh if it still has things lying around from earlier tests
    # mesh.dm.clearDS()
    # mesh.dm.clearFields()
    # mesh.nuke_coords_and_rebuild()
    # mesh.dm.createDS()

    p_soln = uw.discretisation.MeshVariable("P", mesh, 1, degree=2)
    v_soln = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=1)

    # x and y coordinates
    x = mesh.N.x
    y = mesh.N.y

    # #### Set up the Darcy solver
    darcy = uw.systems.SteadyStateDarcy(mesh, p_soln, v_soln)
    darcy.petsc_options.delValue("ksp_monitor")
    darcy.petsc_options["snes_rtol"] = 1.0e-6  # Needs to be smaller than the contrast in properties
    darcy.constitutive_model = uw.constitutive_models.DarcyFlowModel

    # #### Set up the hydraulic conductivity layout
    ### Groundwater pressure boundary condition on the bottom wall

    max_pressure = 0.5

    # +
    # set up two materials

    interfaceY = -0.26

    k1 = 1.0
    k2 = 1.0e-4

    #### The piecewise version
    kFunc = Piecewise((k1, y >= interfaceY), (k2, y < interfaceY), (1.0, True))

    darcy.constitutive_model.Parameters.permeability = kFunc
    darcy.constitutive_model.Parameters.s = sympy.Matrix(
        [0, 0]
    ).T  # Row vector to match grad_u shape
    darcy.f = 0.0

    # set up boundary conditions
    darcy.add_dirichlet_bc([0.0], "Top")
    darcy.add_dirichlet_bc([-1.0 * minY * max_pressure], "Bottom")

    # Zero pressure gradient at sides / base (implied bc)

    darcy._v_projector.petsc_options["snes_rtol"] = 1.0e-6
    darcy._v_projector.smoothing = 1.0e-6
    # darcy._v_projector.add_dirichlet_bc(0.0, "Left", [0])
    # darcy._v_projector.add_dirichlet_bc(0.0, "Right", [0])

    # Solve darcy
    darcy.solve(verbose=True)

    # set up interpolation coordinates
    ycoords = np.linspace(minY + 0.001 * (maxY - minY), maxY - 0.001 * (maxY - minY), 100)
    xcoords = np.full_like(ycoords, -0.5)
    xy_coords = np.column_stack([xcoords, ycoords])

    pressure_interp = uw.function.evaluate(p_soln.sym[0], xy_coords).squeeze()

    # #### Get analytical solution
    La = -1.0 * interfaceY
    Lb = 1.0 + interfaceY
    dP = max_pressure

    S = 0
    Pa = (dP / Lb - S + k1 / k2 * S) / (1.0 / Lb + k1 / k2 / La)
    pressure_analytic_noG = np.piecewise(
        ycoords,
        [ycoords >= -La, ycoords < -La],
        [
            lambda ycoords: -Pa * ycoords / La,
            lambda ycoords: Pa + (dP - Pa) * (-ycoords - La) / Lb,
        ],
    )

    print(pressure_interp)
    print(pressure_analytic_noG)

    # ### Compare analytical and numerical solution
    assert np.allclose(pressure_analytic_noG, pressure_interp, atol=3e-2)

    print("=" * 30)

    ## Suggest we re-solve right here for version with G to avoid all the re-definitions

    S = 1
    Pa = (dP / Lb - S + k1 / k2 * S) / (1.0 / Lb + k1 / k2 / La)
    pressure_analytic = np.piecewise(
        ycoords,
        [ycoords >= -La, ycoords < -La],
        [
            lambda ycoords: -Pa * ycoords / La,
            lambda ycoords: Pa + (dP - Pa) * (-ycoords - La) / Lb,
        ],
    )

    darcy.constitutive_model.Parameters.s = sympy.Matrix(
        [0, -1]
    ).T  # Row vector to match grad_u shape
    darcy.solve()
    darcy.view()

    print(darcy.F1.sym)

    pressure_interp = uw.function.evaluate(p_soln.sym[0], xy_coords).squeeze()

    # ### Compare analytical and numerical solution
    try:
        assert np.allclose(pressure_analytic, pressure_interp, atol=0.1)
        print(pressure_interp)
        print(pressure_analytic)
    finally:
        # Plot diagnostics regardless of test pass/fail
        # This allows visualization when running as notebook or for debugging
        if uw.is_notebook:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # Plot 1: Pressure field (with gravity - current state)
            ax1 = axes[0, 0]
            coords = p_soln.coords
            pressure_vals = p_soln.array[...].flatten()
            scatter1 = ax1.scatter(
                coords[:, 0], coords[:, 1], c=pressure_vals, s=15, cmap="viridis", alpha=0.8
            )
            ax1.axhline(
                y=interfaceY,
                color="r",
                linestyle="--",
                linewidth=2,
                label=f"Interface (y={interfaceY})",
            )
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            ax1.set_title("Pressure Field (with gravity)")
            ax1.set_aspect("equal")
            ax1.legend()
            plt.colorbar(scatter1, ax=ax1, label="Pressure")

            # Plot 2: Velocity field
            ax2 = axes[0, 1]
            vel_coords = v_soln.coords
            vel_vals = v_soln.array
            ax2.quiver(
                vel_coords[::3, 0],
                vel_coords[::3, 1],
                vel_vals[::3, 0, 0],
                vel_vals[::3, 0, 1],
                alpha=0.6,
                scale=5,
                width=0.003,
            )
            ax2.axhline(y=interfaceY, color="r", linestyle="--", linewidth=2)
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")
            ax2.set_title(f"Velocity Field (k1={k1}, k2={k2})")
            ax2.set_aspect("equal")

            # Plot 3: Pressure profile comparison (no gravity)
            ax3 = axes[1, 0]
            ax3.plot(ycoords, pressure_analytic_noG, "r-", linewidth=2, label="Analytical (no G)")

            # Re-solve without gravity for profile comparison
            darcy.constitutive_model.Parameters.s = sympy.Matrix(
                [0, 0]
            ).T  # Row vector to match grad_u shape
            darcy.solve(verbose=False)
            pressure_interp_noG = uw.function.evaluate(p_soln.sym[0], xy_coords).squeeze()
            ax3.plot(
                ycoords,
                pressure_interp_noG,
                "bo",
                markersize=4,
                alpha=0.6,
                label="Numerical (no G)",
            )

            ax3.axvline(x=interfaceY, color="gray", linestyle="--", alpha=0.5)
            ax3.set_xlabel("y coordinate")
            ax3.set_ylabel("Pressure")
            ax3.set_title("Vertical Pressure Profile (no gravity)")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Plot 4: Pressure profile comparison (with gravity)
            ax4 = axes[1, 1]
            ax4.plot(ycoords, pressure_analytic, "r-", linewidth=2, label="Analytical (with G)")
            ax4.plot(
                ycoords, pressure_interp, "go", markersize=4, alpha=0.6, label="Numerical (with G)"
            )
            ax4.axvline(x=interfaceY, color="gray", linestyle="--", alpha=0.5)
            ax4.set_xlabel("y coordinate")
            ax4.set_ylabel("Pressure")
            ax4.set_title("Vertical Pressure Profile (with gravity)")
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()


#

# %%
if uw.is_notebook:
    test_Darcy_boxmesh_G_and_noG(meshSimplex_box_regular)

# %%
