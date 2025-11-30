import pytest

# Physics solver tests - full solver execution
pytestmark = pytest.mark.level_3
import sympy
import underworld3 as uw
import numpy as np


# %%
def test_adv_diff_annulus():
    """Test advection-diffusion on annulus with rigid body rotation."""
    mesh = uw.meshing.Annulus(
        radiusOuter=1.0, radiusInner=0.5, cellSize=0.2, refinement=1, qdegree=3
    )

    v_soln = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    t_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)
    t_0 = uw.discretisation.MeshVariable("T0", mesh, 1, degree=3, varsymbol=r"T_{0}")

    # Create adv_diff object

    # Set some things
    k = 0.01
    h = 0.1
    t_i = 2.0
    t_o = 1.0
    r_i = 0.5
    r_o = 1.0
    delta_t = 0.05  ## 1/20 rotation in one step

    adv_diff = uw.systems.AdvDiffusion(
        mesh,
        u_Field=t_soln,
        V_fn=v_soln,
    )

    adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
    adv_diff.constitutive_model.Parameters.diffusivity = k

    radius_fn = sympy.sqrt(mesh.rvec.dot(mesh.rvec))  # normalise by outer radius if not 1.0
    unit_rvec = mesh.rvec / (1.0e-10 + radius_fn)

    # Some useful coordinate stuff

    x, y = mesh.X
    r, th = mesh.CoordinateSystem.xR

    # Rigid body rotation v_theta = constant, v_r = 0.0

    theta_dot = 2.0 * np.pi  # i.e one revolution in time 1.0
    v_x = -r * theta_dot * sympy.sin(th)
    v_y = r * theta_dot * sympy.cos(th)

    with uw.synchronised_array_update():
        v_soln.array[:, 0, 0] = uw.function.evaluate(v_x, v_soln.coords).squeeze()
        v_soln.array[:, 0, 1] = uw.function.evaluate(v_y, v_soln.coords).squeeze()

    abs_r = sympy.sqrt(mesh.rvec.dot(mesh.rvec))

    init_t = sympy.exp(-30.0 * (mesh.N.x**2 + (mesh.N.y - 0.75) ** 2))

    adv_diff.add_dirichlet_bc(0.0, "Lower")
    adv_diff.add_dirichlet_bc(0.0, "Upper")

    with uw.synchronised_array_update():
        t_0.array[...] = uw.function.evaluate(init_t, t_0.coords).reshape(t_0.array.shape)
        t_soln.array[...] = t_0.array[...]

    t_soln.array[...] = uw.function.evaluate(init_t, t_0.coords).reshape(t_soln.array.shape)

    scalar_projection_solver = uw.systems.solvers.SNES_Projection(mesh, t_0)
    scalar_projection_solver.uw_function = t_soln.sym[0]
    scalar_projection_solver.bcs = adv_diff.bcs
    scalar_projection_solver.solve()

    scalar_projection_solver.uw_function = t_soln.sym[0]
    scalar_projection_solver.bcs = adv_diff.bcs
    scalar_projection_solver.solve()

    adv_diff.solve(timestep=delta_t)

    # %%
    if uw.is_notebook:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Initial temperature
        ax1 = axes[0]
        coords_t0 = t_0.coords
        t0_vals = t_0.data.flatten()
        scatter1 = ax1.scatter(
            coords_t0[:, 0], coords_t0[:, 1], c=t0_vals, s=20, cmap="hot", alpha=0.8, vmin=0
        )
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_title(f"Initial Temperature (t=0)")
        ax1.set_aspect("equal")
        plt.colorbar(scatter1, ax=ax1, label="T")

        # Plot 2: Final temperature after advection-diffusion
        ax2 = axes[1]
        coords_t = t_soln.coords
        t_vals = t_soln.data.flatten()
        scatter2 = ax2.scatter(
            coords_t[:, 0], coords_t[:, 1], c=t_vals, s=20, cmap="hot", alpha=0.8, vmin=0
        )

        # Add velocity field
        vel_coords = v_soln.coords
        vel_data = v_soln.data
        skip = max(1, len(vel_coords) // 30)
        vel_mag = np.sqrt(vel_data[:, 0] ** 2 + vel_data[:, 1] ** 2)
        scale_factor = vel_mag.max() * 10 if vel_mag.max() > 0 else 1
        ax2.quiver(
            vel_coords[::skip, 0],
            vel_coords[::skip, 1],
            vel_data[::skip, 0],
            vel_data[::skip, 1],
            alpha=0.4,
            scale=scale_factor,
            color="blue",
        )

        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_title(f"After Rotation (Δt={delta_t:.3f}, κ={k})")
        ax2.set_aspect("equal")
        plt.colorbar(scatter2, ax=ax2, label="T")

        # Plot 3: Radial profiles
        ax3 = axes[2]

        # Sample along a radial line at different angles
        r_sample = np.linspace(r_i, r_o, 50)

        for angle_deg in [0, 45, 90]:
            angle = np.radians(angle_deg)
            x_sample = r_sample * np.cos(angle)
            y_sample = r_sample * np.sin(angle)
            sample_coords = np.column_stack([x_sample, y_sample])

            T_sample_init = uw.function.evaluate(init_t, sample_coords).flatten()
            T_sample_final = uw.function.evaluate(t_soln.sym[0], sample_coords).flatten()

            ax3.plot(r_sample, T_sample_init, "--", alpha=0.5, label=f"Initial θ={angle_deg}°")
            ax3.plot(r_sample, T_sample_final, "-", label=f"Final θ={angle_deg}°")

        ax3.set_xlabel("Radius")
        ax3.set_ylabel("Temperature")
        ax3.set_title("Radial Temperature Profiles")
        ax3.legend(fontsize=8, ncol=2)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
