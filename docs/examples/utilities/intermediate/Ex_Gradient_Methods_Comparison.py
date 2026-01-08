# %% [markdown]
# # Gradient Computation Methods: Projection vs Clement Evaluation
#
# This notebook compares two approaches for computing gradients of mesh variables:
#
# 1. **L2 Projection**: Solve a mass matrix system for optimal L2 gradient recovery
#    - Accuracy: O(h²) for smooth solutions
#    - Cost: Requires solving a linear system
#
# 2. **Clement Interpolation**: Average cell-wise gradients at vertices
#    - Accuracy: O(h) - first order convergence
#    - Cost: No linear solve required
#
# We test with two functions to demonstrate convergence behaviour:
# - **Smooth**: $f(x,y) = \sin(\pi x)\sin(\pi y)$ - infinitely differentiable
# - **Less smooth**: Cone function with gradient discontinuity at the apex

# %%

# %%
import numpy as np
import underworld3 as uw
import sympy
import matplotlib.pyplot as plt

# %% [markdown]
# ## Test Functions
#
# ### Smooth function
# $$f_1(x,y) = \sin(\pi x) \sin(\pi y)$$
#
# ### Steep gradient function (tanh layer)
# $$f_2(x,y) = \tanh(k(x - 0.5))$$
#
# where $k$ controls the steepness. This represents a smooth but rapidly-varying
# transition layer - common in boundary layers, phase fields, and reaction fronts.
# The gradient $\partial f/\partial x = k \, \text{sech}^2(k(x-0.5))$ is large near $x=0.5$.

# %%
# Smooth function and its derivatives
def f_smooth(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def dfdx_smooth(x, y):
    return np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)

def dfdy_smooth(x, y):
    return np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)

# Tanh layer function (steep but smooth gradient)
k_steepness = 20  # Controls layer width (~1/k)

def f_tanh(x, y):
    return np.tanh(k_steepness * (x - 0.5))

def dfdx_tanh(x, y):
    return k_steepness / np.cosh(k_steepness * (x - 0.5))**2

def dfdy_tanh(x, y):
    return np.zeros_like(x)

# %% [markdown]
# ## Convergence Study Setup

# %%
# Cell sizes to test (controls mesh resolution) - 5 levels for better rate estimation
cell_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625]

# Sample points for evaluation
n_sample = 200
np.random.seed(42)
sample_coords = np.random.uniform(0.1, 0.9, size=(n_sample, 2))
print(f"Using {n_sample} sample points")

# %% [markdown]
# ## Run Convergence Study

# %%
def run_convergence_study(f_func, dfdx_func, dfdy_func, name):
    """Run convergence study for a given test function."""

    # Exact gradients at sample points
    exact_dfdx = dfdx_func(sample_coords[:, 0], sample_coords[:, 1])
    exact_dfdy = dfdy_func(sample_coords[:, 0], sample_coords[:, 1])

    # Storage for errors
    errors_proj_x = []
    errors_proj_y = []
    errors_clement_x = []
    errors_clement_y = []
    h_values = []

    print(f"\n{'='*60}")
    print(f"Test function: {name}")
    print(f"{'='*60}")

    for cell_size in cell_sizes:
        print(f"\n--- Cell size: {cell_size} ---")

        # Create simplex mesh
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0),
            maxCoords=(1.0, 1.0),
            cellSize=cell_size,
            regular=True,
        )

        x, y = mesh.X
        h_values.append(cell_size)

        # Create mesh variable and set to test function
        f_var = uw.discretisation.MeshVariable("f", mesh, num_components=1, degree=1)
        f_var.array[:, 0, 0] = f_func(f_var.coords[:, 0], f_var.coords[:, 1])

        # ----- L2 Projection -----
        grad_proj_x = uw.discretisation.MeshVariable("dfdx", mesh, num_components=1, degree=1)
        grad_proj_y = uw.discretisation.MeshVariable("dfdy", mesh, num_components=1, degree=1)

        projector_x = uw.systems.Projection(mesh, grad_proj_x)
        projector_x.uw_function = f_var.sym[0, 0].diff(x)
        projector_x.smoothing = 0.0
        projector_x.solve()

        projector_y = uw.systems.Projection(mesh, grad_proj_y)
        projector_y.uw_function = f_var.sym[0, 0].diff(y)
        projector_y.smoothing = 0.0
        projector_y.solve()

        proj_dfdx = uw.function.evaluate(grad_proj_x.sym, sample_coords).flatten()
        proj_dfdy = uw.function.evaluate(grad_proj_y.sym, sample_coords).flatten()

        error_proj_x = np.sqrt(np.mean((proj_dfdx - exact_dfdx)**2))
        error_proj_y = np.sqrt(np.mean((proj_dfdy - exact_dfdy)**2))
        errors_proj_x.append(error_proj_x)
        errors_proj_y.append(error_proj_y)

        # ----- Clement Evaluation -----
        clement_dfdx = uw.function.evaluate(f_var.sym.diff(x), sample_coords).flatten()
        clement_dfdy = uw.function.evaluate(f_var.sym.diff(y), sample_coords).flatten()

        error_clement_x = np.sqrt(np.mean((clement_dfdx - exact_dfdx)**2))
        error_clement_y = np.sqrt(np.mean((clement_dfdy - exact_dfdy)**2))
        errors_clement_x.append(error_clement_x)
        errors_clement_y.append(error_clement_y)

        print(f"  Projection:  err_x={error_proj_x:.2e}, err_y={error_proj_y:.2e}")
        print(f"  Clement:     err_x={error_clement_x:.2e}, err_y={error_clement_y:.2e}")

        del mesh, f_var, grad_proj_x, grad_proj_y, projector_x, projector_y

    return {
        'h': np.array(h_values),
        'proj_x': np.array(errors_proj_x),
        'proj_y': np.array(errors_proj_y),
        'clement_x': np.array(errors_clement_x),
        'clement_y': np.array(errors_clement_y),
    }

# %%
# Run studies for both functions
results_smooth = run_convergence_study(f_smooth, dfdx_smooth, dfdy_smooth, "Smooth: sin(πx)sin(πy)")
results_tanh = run_convergence_study(f_tanh, dfdx_tanh, dfdy_tanh, f"Steep layer: tanh({k_steepness}(x-0.5))")

# %% [markdown]
# ## Compute Convergence Rates

# %%
def compute_rate(h, errors):
    """Compute convergence rate from log-log slope."""
    log_h = np.log(h)
    log_e = np.log(errors)
    return np.polyfit(log_h, log_e, 1)[0]

print("\n" + "="*70)
print("CONVERGENCE RATES")
print("="*70)

for name, results in [("Smooth function", results_smooth), ("Tanh layer", results_tanh)]:
    rate_proj = (compute_rate(results['h'], results['proj_x']) +
                 compute_rate(results['h'], results['proj_y'])) / 2
    rate_clement = (compute_rate(results['h'], results['clement_x']) +
                    compute_rate(results['h'], results['clement_y'])) / 2

    print(f"\n{name}:")
    print(f"  L2 Projection:     O(h^{rate_proj:.2f})")
    print(f"  Clement Evaluation: O(h^{rate_clement:.2f})")

# %% [markdown]
# ## Visualise Convergence

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (name, results, title) in zip(axes, [
    ("Smooth", results_smooth, r"Smooth: $\sin(\pi x)\sin(\pi y)$"),
    ("Tanh", results_tanh, rf"Steep layer: $\tanh({k_steepness}(x-0.5))$"),
]):
    h = results['h']

    # Average x and y errors for cleaner plot
    err_proj = (results['proj_x'] + results['proj_y']) / 2
    err_clement = (results['clement_x'] + results['clement_y']) / 2

    rate_proj = compute_rate(h, err_proj)
    rate_clement = compute_rate(h, err_clement)

    ax.loglog(h, err_proj, 'o-', label=f'L2 Projection (O(h^{rate_proj:.2f}))',
              markersize=10, linewidth=2, color='C0')
    ax.loglog(h, err_clement, 's-', label=f'Clement (O(h^{rate_clement:.2f}))',
              markersize=10, linewidth=2, color='C1')

    # Reference lines
    h_ref = np.array([h.max(), h.min()])
    scale_h1 = err_clement[-1] / h[-1]
    scale_h2 = err_proj[-1] / h[-1]**2
    ax.loglog(h_ref, scale_h1 * h_ref, 'k--', alpha=0.4, linewidth=1.5, label='O(h) reference')
    ax.loglog(h_ref, scale_h2 * h_ref**2, 'k:', alpha=0.4, linewidth=1.5, label='O(h²) reference')

    ax.set_xlabel('Mesh size h', fontsize=12)
    ax.set_ylabel('L2 Error (averaged over components)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gradient_convergence.png", dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Summary Tables

# %%
print("\n" + "="*70)
print("SMOOTH FUNCTION: sin(πx)sin(πy)")
print("="*70)
print(f"\n{'Cell Size':<12} {'Projection':<14} {'Clement':<14} {'Ratio':<8}")
print("-"*50)
for i, h in enumerate(results_smooth['h']):
    err_proj = (results_smooth['proj_x'][i] + results_smooth['proj_y'][i]) / 2
    err_clem = (results_smooth['clement_x'][i] + results_smooth['clement_y'][i]) / 2
    print(f"{h:<12.5f} {err_proj:<14.2e} {err_clem:<14.2e} {err_clem/err_proj:<8.1f}")

print("\n" + "="*70)
print(f"STEEP LAYER FUNCTION: tanh({k_steepness}(x-0.5))")
print("="*70)
print(f"\n{'Cell Size':<12} {'Projection':<14} {'Clement':<14} {'Ratio':<8}")
print("-"*50)
for i, h in enumerate(results_tanh['h']):
    err_proj = (results_tanh['proj_x'][i] + results_tanh['proj_y'][i]) / 2
    err_clem = (results_tanh['clement_x'][i] + results_tanh['clement_y'][i]) / 2
    print(f"{h:<12.5f} {err_proj:<14.2e} {err_clem:<14.2e} {err_clem/err_proj:<8.1f}")

# %% [markdown]
# ## Conclusions
#
# | Function | Method | Theoretical | Observed |
# |----------|--------|-------------|----------|
# | Smooth (sin) | L2 Projection | O(h²) | ~O(h²) |
# | Smooth (sin) | Clement | O(h) | ~O(h²)* |
# | Steep layer (tanh) | L2 Projection | O(h²) | ? |
# | Steep layer (tanh) | Clement | O(h) | ? |
#
# **Key observations:**
#
# 1. **Smooth functions**: Both methods achieve approximately O(h²) convergence.
#    The Clement method "superconverges" for smooth solutions on regular meshes.
#
# 2. **Steep gradient regions**: The tanh function has a transition layer of width
#    ~1/k where the gradient varies rapidly. When the mesh cannot resolve this layer
#    (h > 1/k), both methods struggle and show degraded convergence. As the mesh
#    resolves the layer (h << 1/k), asymptotic rates are recovered.
#
# 3. **Error magnitude**: L2 projection consistently produces smaller errors,
#    justifying the additional solve cost when accuracy is critical.
#
# ### When to use each method:
#
# **Clement Evaluation** (`uw.function.evaluate(T.diff(x), coords)`):
# - Quick gradient estimates during iteration
# - Error indicators for adaptive mesh refinement
# - Visualisation and debugging
# - Situations where solve overhead matters more than accuracy
#
# **L2 Projection** (`uw.systems.Projection`):
# - High-accuracy requirements in constitutive relations
# - Post-processing for publication-quality results
# - When gradient accuracy directly affects solution quality

# %% [markdown]
# ## Appendix: API Usage
#
# ### Direct gradient evaluation (Clement)
# ```python
# x, y = mesh.X
#
# # Simple derivative
# dTdx = uw.function.evaluate(T.sym.diff(x), coords)
#
# # Combined expressions
# result = uw.function.evaluate(T.sym * T.sym.diff(x) + A.sym.diff(y), coords)
# ```
#
# ### L2 Projection
# ```python
# grad_var = uw.discretisation.MeshVariable("grad", mesh, num_components=1, degree=1)
# projector = uw.systems.Projection(mesh, grad_var)
# projector.uw_function = T.sym[0,0].diff(x)
# projector.smoothing = 0.0
# projector.solve()
#
# result = uw.function.evaluate(grad_var.sym, coords)
# ```

# %%
from mpi4py import MPI
MPI.Finalize()

# %%
