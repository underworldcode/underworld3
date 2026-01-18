# %%
import pytest

# Physics solver tests - full solver execution
pytestmark = pytest.mark.level_3
import sympy
import underworld3 as uw

r_o = 1.0
r_i = 0.6
res = 0.33

annulus = uw.meshing.Annulus(
    radiusOuter=r_o,
    radiusInner=r_i,
    cellSize=0.1,
    qdegree=2,
)

spherical_shell = uw.meshing.SphericalShell(
    radiusOuter=r_o,
    radiusInner=r_i,
    cellSize=0.4,
    qdegree=2,
)

cubed_sphere = uw.meshing.CubedSphere(
    radiusOuter=r_o,
    radiusInner=r_i,
    numElements=3,
    qdegree=2,
    refinement=0,
)


# %%


# %%
@pytest.mark.parametrize("mesh", [annulus, cubed_sphere, spherical_shell])
def test_stokes_sphere(mesh):
    """Test Stokes flow on spherical/annular meshes with localized buoyancy forcing."""
    if mesh.dim == 2:
        x, y = mesh.X
        z = 0
    else:
        x, y, z = mesh.X

    ra = mesh.CoordinateSystem.R[0]

    u = uw.discretisation.MeshVariable(
        "u",
        mesh,
        mesh.dim,
        vtype=uw.VarType.VECTOR,
        degree=2,
        varsymbol=r"\mathbf{u}",
    )
    p = uw.discretisation.MeshVariable(
        "p",
        mesh,
        1,
        vtype=uw.VarType.SCALAR,
        degree=1,
        continuous=True,
        varsymbol=r"\mathbf{p}",
    )

    # Create a density structure / buoyancy force

    radius_fn = sympy.sqrt(mesh.rvec.dot(mesh.rvec))  # normalise by outer radius if not 1.0
    unit_rvec = mesh.X / (radius_fn)

    ## Buoyancy (T) field

    t_forcing_fn = 1.0 * (
        sympy.exp(-10.0 * (x**2 + (y - 0.8) ** 2 + z**2))
        + sympy.exp(-10.0 * ((x - 0.8) ** 2 + y**2 + z**2))
        + sympy.exp(-10.0 * (x**2 + y**2 + (z - 0.8) ** 2))
    )

    stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p, verbose=False)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1

    stokes.tolerance = 1.0e-2
    stokes.petsc_options["ksp_monitor"] = None

    stokes.petsc_options["snes_type"] = "newtonls"
    stokes.petsc_options["ksp_type"] = "fgmres"

    # stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
    stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "multiplicative")
    stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "v")

    stokes.petsc_options[f"fieldsplit_velocity_ksp_type"] = "fcg"
    stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_type"] = "fgmres"
    stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_max_it"] = 5
    stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

    buoyancy_force = 1.0e6 * t_forcing_fn

    # Free slip condition by penalizing radial velocity at the surface (non-linear term)
    # free_slip_penalty_upper = u.sym.dot(unit_rvec) * unit_rvec * surface_fn
    # free_slip_penalty_lower = u.sym.dot(unit_rvec) * unit_rvec * base_fn

    stokes.bodyforce = unit_rvec * buoyancy_force

    Gamma = mesh.Gamma

    stokes.add_natural_bc(10000 * Gamma.dot(u) * Gamma, "Upper")
    stokes.add_natural_bc(10000 * Gamma.dot(u) * Gamma, "Lower")

    stokes.solve()

    if uw.is_notebook:
        return stokes

    assert stokes.snes.getConvergedReason() > 0


# %%
# The following is to help work out the nature of any regression - run in a notebook to see what is happening

# %%
if uw.is_notebook:
    import matplotlib.pyplot as plt
    import numpy as np

    mesh = annulus
    solver = test_stokes_sphere(mesh)

    t_forcing_fn = solver.F0.sym.dot(solver.F0.sym)

    if mesh.dim == 2:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Buoyancy forcing
        ax1 = axes[0]
        coords = solver.u.coords
        forcing_vals = uw.function.evaluate(t_forcing_fn, coords).flatten()
        scatter1 = ax1.scatter(
            coords[:, 0], coords[:, 1], c=forcing_vals, s=20, cmap="hot", alpha=0.8
        )
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_title("Buoyancy Forcing")
        ax1.set_aspect("equal")
        plt.colorbar(scatter1, ax=ax1, label="T")

        # Plot 2: Velocity magnitude
        vel_data = solver.u.array[:, 0, :]
        vel_mag = np.sqrt(vel_data[:, 0] ** 2 + vel_data[:, 1] ** 2)
        scatter2 = ax1 = axes[1]
        scatter2 = ax1.scatter(
            coords[:, 0], coords[:, 1], c=vel_mag, s=20, cmap="plasma", alpha=0.8
        )
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_title("Velocity Magnitude")
        ax1.set_aspect("equal")
        plt.colorbar(scatter2, ax=ax1, label="|v|")

        # Plot 3: Streamlines
        ax3 = axes[2]
        skip = 3
        ax3.quiver(
            coords[::skip, 0],
            coords[::skip, 1],
            vel_data[::skip, 0],
            vel_data[::skip, 1],
            alpha=0.7,
            scale=vel_mag.max() * 15,
        )

        p_coords = solver.p.coords
        p_vals = solver.p.array[:, 0, 0]
        scatter3 = ax3.scatter(
            p_coords[:, 0], p_coords[:, 1], c=p_vals, s=15, cmap="RdBu_r", alpha=0.3
        )
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_title("Flow Field + Pressure")
        ax3.set_aspect("equal")
        plt.colorbar(scatter3, ax=ax3, label="Pressure")

    else:
        # For 3D spherical shell, show radial slices
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        coords = solver.u.coords
        vel_data = solver.u.array[:, 0, :]

        # XY plane (z ~ 0)
        z_slice = np.abs(coords[:, 2]) < 0.15
        coords_slice = coords[z_slice]
        vel_slice = vel_data[z_slice]
        vel_mag_slice = np.sqrt(vel_slice[:, 0] ** 2 + vel_slice[:, 1] ** 2 + vel_slice[:, 2] ** 2)

        # Forcing
        forcing_vals = uw.function.evaluate(t_forcing_fn, coords_slice).flatten()
        ax1 = axes[0]
        scatter1 = ax1.scatter(
            coords_slice[:, 0],
            coords_slice[:, 1],
            c=forcing_vals,
            s=25,
            cmap="hot",
            alpha=0.8,
        )
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_title("3D Buoyancy (z≈0 slice)")
        ax1.set_aspect("equal")
        plt.colorbar(scatter1, ax=ax1, label="T")

        # Velocity
        ax2 = axes[1]
        scatter2 = ax2.scatter(
            coords_slice[:, 0],
            coords_slice[:, 1],
            c=vel_mag_slice,
            s=25,
            cmap="plasma",
            alpha=0.8,
        )
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_title("3D Velocity (z≈0 slice)")
        ax2.set_aspect("equal")
        plt.colorbar(scatter2, ax=ax2, label="|v|")

        # Pressure
        p_coords = p.coords
        p_vals = p.data.flatten()
        p_slice = np.abs(p_coords[:, 2]) < 0.15

        ax3 = axes[2]
        scatter3 = ax3.scatter(
            p_coords[p_slice, 0],
            p_coords[p_slice, 1],
            c=p_vals[p_slice],
            s=25,
            cmap="RdBu_r",
            alpha=0.8,
        )
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_title("3D Pressure (z≈0 slice)")
        ax3.set_aspect("equal")
        plt.colorbar(scatter3, ax=ax3, label="Pressure")

    plt.tight_layout()
    plt.show()
