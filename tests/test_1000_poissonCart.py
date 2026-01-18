# %% [markdown]
# # Poisson Solver Unit Tests
#
# This file can run both as a pytest test suite and as a Jupyter notebook.
# - **pytest**: Runs tests without plots
# - **Jupyter**: Shows visualizations to verify solver behavior

# %%
import numpy as np
import pytest

# Physics solver tests - full solver execution
pytestmark = pytest.mark.level_3
import sympy
import underworld3 as uw

import os

os.environ["SYMPY_USE_CACHE"] = "no"


@pytest.fixture(autouse=True)
def reset_model_state():
    """Reset model state before each test to prevent pollution from other tests.

    Tests in test_0850_units_propagation set reference quantities which can affect
    solver behavior and cause units enforcement errors.
    """
    uw.reset_default_model()
    uw.use_strict_units(False)
    uw.use_nondimensional_scaling(False)
    yield
    uw.reset_default_model()
    uw.use_strict_units(False)
    uw.use_nondimensional_scaling(False)


# Try to import plotting libraries - only show plots if available and in Jupyter
try:
    import matplotlib.pyplot as plt

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


# Helper function for conditional plotting
def show_plot_if_jupyter(fig_func, title=""):
    """Show plot only if running in Jupyter with plotting available."""
    if uw.is_notebook and PLOTTING_AVAILABLE:
        print(f"\n=== {title} ===")
        try:
            fig_func()
            plt.show()
        except Exception as e:
            print(f"⚠️  Plotting failed: {e}")
            print("Continuing without visualization...")


# Helper function for robust 2D plotting
def plot_2d_solution(mesh, u, ax, title="2D Solution"):
    """Robust 2D solution plotting using meshVariable.coords for proper coordinate matching."""
    coords = u.coords
    values = u.data.flatten()

    im = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=15, cmap="viridis", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax)

    ax.text(
        0.02,
        0.98,
        f"{len(coords)} DOF points",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        verticalalignment="top",
        fontsize=8,
    )

    return im


# %% [markdown]
# ## Test Mesh Definitions
# Define various mesh types for comprehensive testing

# %%
structured_quad_box = uw.meshing.StructuredQuadBox(elementRes=(5,) * 2)

unstructured_quad_box_irregular = uw.meshing.UnstructuredSimplexBox(cellSize=0.1, regular=False)
unstructured_quad_box_regular = uw.meshing.UnstructuredSimplexBox(cellSize=0.1, regular=True)
unstructured_quad_box_regular_3D = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0), cellSize=0.1, regular=True
)


# %% [markdown]
# ## Basic Poisson Test on Different Mesh Types
# Test that the Poisson solver converges on various mesh geometries


# %%
@pytest.mark.parametrize(
    "mesh",
    [
        structured_quad_box,
        unstructured_quad_box_irregular,
        unstructured_quad_box_regular,
        unstructured_quad_box_regular_3D,
    ],
)
def test_poisson_boxmesh(mesh):
    u = uw.discretisation.MeshVariable(r"mathbf{u}", mesh, 1, vtype=uw.VarType.SCALAR, degree=2)

    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1
    poisson.f = 0.0
    poisson.add_dirichlet_bc(1.0, "Bottom")
    poisson.add_dirichlet_bc(0.0, "Top")
    poisson.solve()

    assert poisson.snes.getConvergedReason() > 0

    del poisson
    del mesh


# %% [markdown]
# ## Linear Profile Test
# Test Poisson equation ∇²u = 0 with BCs u(y=0)=1, u(y=1)=0
# Expected analytical solution: u(y) = 1 - y


# %%
def test_poisson_linear_profile():
    """Test Poisson solver with no source term against 1D linear analytical solution."""
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.2)
    x, y = mesh.X

    # Poisson setup: ∇²u = 0
    u = uw.discretisation.MeshVariable("u", mesh, 1, vtype=uw.VarType.SCALAR, degree=2)
    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1
    poisson.f = 0.0

    # BCs: u(y=0) = 1, u(y=1) = 0 gives linear profile u(y) = 1 - y
    poisson.add_dirichlet_bc(1.0, "Bottom")
    poisson.add_dirichlet_bc(0.0, "Top")
    poisson.solve()

    assert poisson.snes.getConvergedReason() > 0

    # Sample vertical profile through center
    sample_y = np.linspace(0, 1, 20)
    sample_x = np.zeros_like(sample_y) + 0.5
    sample_points = np.column_stack([sample_x, sample_y])

    # Compare with analytical solution: u(y) = 1 - y
    u_numerical = uw.function.evaluate(u.sym[0], sample_points, rbf=False).squeeze()
    u_analytical = 1 - sample_y

    error = np.sqrt(np.mean((u_numerical - u_analytical) ** 2))

    # Visualization for debugging
    def plot_linear_comparison():
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

        # Plot 1: Full 2D solution
        plot_2d_solution(mesh, u, ax1, "2D Numerical Solution")

        # Plot 2: 1D profile comparison
        ax2.plot(sample_y, u_numerical, "bo-", label="Numerical", markersize=4)
        ax2.plot(sample_y, u_analytical, "r-", label="Analytical u=1-y", linewidth=2)
        ax2.set_xlabel("y")
        ax2.set_ylabel("u")
        ax2.set_title(f"Profile Comparison (x=0.5)\nL2 Error: {error:.3e}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Error distribution
        error_field = u_numerical - u_analytical
        ax3.plot(sample_y, error_field, "go-", markersize=4)
        ax3.set_xlabel("y")
        ax3.set_ylabel("Error")
        ax3.set_title("Numerical - Analytical")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

    show_plot_if_jupyter(plot_linear_comparison, "Linear Profile Test Results")

    assert error < 1e-3, f"Linear profile error {error:.3e} exceeds tolerance"

    del poisson
    del mesh


# %% [markdown]
# ## Constant Source Test
# Test Poisson equation ∇²u = -2 with BCs u(y=0)=0, u(y=1)=0
# Expected analytical solution: u(y) = y(1-y)


# %%
def test_poisson_constant_source():
    """Test Poisson solver with constant source term against 1D quadratic analytical solution."""
    mesh = uw.meshing.StructuredQuadBox(elementRes=(10, 10))
    x, y = mesh.X

    # Poisson setup: ∇²u = -2
    u = uw.discretisation.MeshVariable("u", mesh, 1, vtype=uw.VarType.SCALAR, degree=2)
    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1
    poisson.f = 2.0  # Source term (note sign convention: -∇²u = f)

    # BCs: u(y=0) = 0, u(y=1) = 0 gives parabolic profile u(y) = y(1-y)
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc(0.0, "Top")
    poisson.solve()

    assert poisson.snes.getConvergedReason() > 0

    # Sample vertical profile through center
    sample_y = np.linspace(0, 1, 20)
    sample_x = np.zeros_like(sample_y) + 0.5
    sample_points = np.column_stack([sample_x, sample_y])

    # Analytical solution: u(y) = y(1-y) when ∇²u = -2
    u_numerical = uw.function.evaluate(u.sym[0], sample_points, rbf=False).squeeze()
    u_analytical = sample_y * (1 - sample_y)

    error = np.sqrt(np.mean((u_numerical - u_analytical) ** 2))

    # Visualization
    def plot_quadratic_comparison():
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

        # Plot 1: Full 2D solution
        plot_2d_solution(mesh, u, ax1, "2D Numerical Solution\n(Constant Source f=2)")

        # Plot 2: 1D profile comparison
        ax2.plot(sample_y, u_numerical, "bo-", label="Numerical", markersize=4)
        ax2.plot(sample_y, u_analytical, "r-", label="Analytical u=y(1-y)", linewidth=2)
        ax2.set_xlabel("y")
        ax2.set_ylabel("u")
        ax2.set_title(f"Profile Comparison (x=0.5)\nL2 Error: {error:.3e}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Error distribution
        error_field = u_numerical - u_analytical
        ax3.plot(sample_y, error_field, "go-", markersize=4)
        ax3.set_xlabel("y")
        ax3.set_ylabel("Error")
        ax3.set_title("Numerical - Analytical")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

    show_plot_if_jupyter(plot_quadratic_comparison, "Constant Source Test Results")

    assert error < 1e-3, f"Quadratic source error {error:.3e} exceeds tolerance"

    del poisson
    del mesh


# %% [markdown]
# ## Sinusoidal Source Test
# Test Poisson equation ∇²u = -π²sin(πy) with BCs u(y=0)=0, u(y=1)=0
# Expected analytical solution: u(y) = sin(πy)


# %%
def test_poisson_sinusoidal_source():
    """Test Poisson solver with sinusoidal source term - critical for time-dependent validation."""
    mesh = uw.meshing.StructuredQuadBox(elementRes=(12, 12))
    x, y = mesh.X

    # Poisson setup: ∇²u = -π²sin(πy)
    u = uw.discretisation.MeshVariable("u", mesh, 1, vtype=uw.VarType.SCALAR, degree=2)
    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1

    # Symbolic source term: f = π²sin(πy) so that ∇²u = -π²sin(πy) has solution u = sin(πy)
    poisson.f = sympy.pi**2 * sympy.sin(sympy.pi * y)

    # BCs: u(y=0) = 0, u(y=1) = 0 (consistent with sin(πy))
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc(0.0, "Top")
    poisson.solve()

    assert poisson.snes.getConvergedReason() > 0

    # Sample and compare with analytical solution
    sample_y = np.linspace(0, 1, 25)
    sample_x = np.zeros_like(sample_y) + 0.5
    sample_points = np.column_stack([sample_x, sample_y])

    # Analytical solution: u(y) = sin(πy)
    u_numerical = uw.function.evaluate(u.sym[0], sample_points, rbf=False).squeeze()
    u_analytical = np.sin(np.pi * sample_y)

    error = np.sqrt(np.mean((u_numerical - u_analytical) ** 2))

    # Visualization
    def plot_sinusoidal_comparison():
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

        # Plot 1: Full 2D solution
        plot_2d_solution(mesh, u, ax1, "2D Numerical Solution\n(Sinusoidal Source f=π²sin(πy))")

        # Plot 2: 1D profile comparison
        ax2.plot(sample_y, u_numerical, "bo-", label="Numerical", markersize=4)
        ax2.plot(sample_y, u_analytical, "r-", label="Analytical u=sin(πy)", linewidth=2)
        ax2.set_xlabel("y")
        ax2.set_ylabel("u")
        ax2.set_title(f"Profile Comparison (x=0.5)\nL2 Error: {error:.3e}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Error distribution
        error_field = u_numerical - u_analytical
        ax3.plot(sample_y, error_field, "go-", markersize=4)
        ax3.set_xlabel("y")
        ax3.set_ylabel("Error")
        ax3.set_title("Numerical - Analytical")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

    show_plot_if_jupyter(plot_sinusoidal_comparison, "Sinusoidal Source Test Results")

    assert error < 2e-3, f"Sinusoidal source error {error:.3e} exceeds tolerance"

    del poisson
    del mesh


# %% [markdown]
# ## Run All Tests for Jupyter
# When running in Jupyter, execute all tests to see visualizations

# %%
if uw.is_notebook:
    print("Running Poisson tests with visualizations...")

    try:
        test_poisson_linear_profile()
        print("✅ Linear profile test completed")
    except Exception as e:
        print(f"❌ Linear profile test failed: {e}")

    print("\n" + "=" * 60)

    try:
        test_poisson_constant_source()
        print("✅ Constant source test completed")
    except Exception as e:
        print(f"❌ Constant source test failed: {e}")

    print("\n" + "=" * 60)

    try:
        test_poisson_sinusoidal_source()
        print("✅ Sinusoidal source test completed")
    except Exception as e:
        print(f"❌ Sinusoidal source test failed: {e}")

    print("\n" + "=" * 60)
    print("All tests completed. Check plots above for results.")

# %%
