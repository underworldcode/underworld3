import pytest
import underworld3 as uw


annulus = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.1, qdegree=2)


spherical_shell = uw.meshing.SphericalShell(
    radiusOuter=1.0, radiusInner=0.5, cellSize=0.5, qdegree=2
)

# cubed_sphere = uw.meshing.CubedSphere(
#     radiusOuter=1.0, radiusInner=0.5, numElements=3, qdegree=2
# )

# Maybe lower and upper would work better for the names of the box mesh boundaries too.


# %%
@pytest.mark.parametrize("mesh", [annulus, spherical_shell])
def test_poisson_sphere(mesh):
    """Test Poisson equation on spherical/annular meshes with Dirichlet BCs."""

    u = uw.discretisation.MeshVariable(r"mathbf{u}", mesh, 1, vtype=uw.VarType.SCALAR, degree=2)

    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1

    poisson.f = 0.0
    poisson.add_dirichlet_bc(1.0, "Lower")
    poisson.add_dirichlet_bc(0.0, "Upper")
    poisson.solve()

    assert poisson.snes.getConvergedReason() > 0

    # %%
    if uw.is_notebook:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot solution field
        coords = u.coords
        values = u.data.flatten()

        if mesh.dim == 2:
            # For 2D annulus, plot in Cartesian coordinates
            scatter1 = ax1.scatter(
                coords[:, 0], coords[:, 1], c=values, s=20, cmap="coolwarm", alpha=0.8
            )
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            ax1.set_title(f"Poisson Solution (T: inner=1, outer=0)")
            ax1.set_aspect("equal")
            plt.colorbar(scatter1, ax=ax1, label="T")

            # Plot radial profile
            r = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
            ax2.scatter(r, values, s=10, alpha=0.5, label="Numerical")

            # Analytical solution for radial diffusion: T = (log(r/r_o))/(log(r_i/r_o))
            r_sorted = np.linspace(mesh.radiusInner, mesh.radiusOuter, 100)
            T_analytical = np.log(r_sorted / mesh.radiusOuter) / np.log(
                mesh.radiusInner / mesh.radiusOuter
            )
            ax2.plot(r_sorted, T_analytical, "r-", linewidth=2, label="Analytical")

            ax2.set_xlabel("Radius")
            ax2.set_ylabel("T")
            ax2.set_title("Radial Profile")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            # For 3D spherical shell, show one slice
            ax1.text(
                0.5,
                0.5,
                f"3D Spherical Shell\nConverged: {poisson.snes.getConvergedReason() > 0}\nNodes: {len(values)}",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )
            ax1.axis("off")

            r = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2 + coords[:, 2] ** 2)
            ax2.scatter(r, values, s=10, alpha=0.5, label="Numerical")

            # Analytical solution
            r_sorted = np.linspace(mesh.radiusInner, mesh.radiusOuter, 100)
            T_analytical = (1.0 / r_sorted - 1.0 / mesh.radiusOuter) / (
                1.0 / mesh.radiusInner - 1.0 / mesh.radiusOuter
            )
            ax2.plot(r_sorted, T_analytical, "r-", linewidth=2, label="Analytical")

            ax2.set_xlabel("Radius")
            ax2.set_ylabel("T")
            ax2.set_title("Radial Profile (3D)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    del poisson
    del mesh
