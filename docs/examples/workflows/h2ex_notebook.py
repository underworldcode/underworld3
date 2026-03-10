# %% [markdown]
r"""
# H2Ex Fault-Controlled Flow — Product-Aware Workflow

**PHYSICS:** crustal fluid flow through fault zones

**DIFFICULTY:** advanced

**RUNTIME:** ~10 minutes (with adaptation); seconds (isoviscous, no adaptation)

## Description

Demonstrates the **product-aware workflow pattern** where expensive serial
steps (mesh creation, fault loading, adaptation) produce named products
that are saved to disk and reloaded for parameter studies.

Two usage patterns:

- **Pattern A (full serial):** Build everything from scratch, save products.
- **Pattern B (product reload):** Load pre-built mesh and faults, vary
  rheology or boundary conditions without repeating the expensive steps.

## Key Concepts

- `WorkflowProducts` for make-like persistence
- `produces`/`requires` metadata on workflow steps for DAG visualisation
- Geographic mesh with fault-adaptive refinement
- Anisotropic rheology via fault influence fields
"""

# %%
#| echo: false
import nest_asyncio
nest_asyncio.apply()

# %%
import os
import underworld3 as uw
from underworld3.workflows import WorkflowProducts

import h2ex_config as h2ex

# %% [markdown]
"""
## 1. Configuration

All tunable parameters for the H2Ex workflow.
"""

# %%
config = h2ex.H2ExConfig(
    adapt=False,              # Set True for mesh adaptation (requires MMG)
    rheology="isoviscous",    # "isoviscous" for fast smoke test
    num_elements=(8, 8, 4),   # Coarse for demonstration
)

config.view()

# %% [markdown]
"""
### Inspect workflow DAG

`view()` shows all steps with their product dependencies.
"""

# %%
h2ex.view()

# %% [markdown]
"""
### YAML round-trip
"""

# %%
os.makedirs(config.output_dir, exist_ok=True)
config.save_yaml(f"{config.output_dir}/params.yaml")

config2 = h2ex.H2ExConfig.from_yaml(f"{config.output_dir}/params.yaml")
assert config2.rheology == config.rheology
uw.pprint("YAML round-trip OK")

# %% [markdown]
"""
## 2. Products Manager

`WorkflowProducts` handles save/load dispatch and tracks a manifest.
"""

# %%
products = WorkflowProducts(config)

# %% [markdown]
"""
## 3. Pattern A — Full Serial Build

Build everything from scratch.  After the expensive steps (mesh, faults,
adaptation), save products so they can be reloaded later.
"""

# %%
model = config.setup_model()

# %% [markdown]
"""
### Create mesh
"""

# %%
mesh = h2ex.create_mesh(config)
uw.pprint(f"Mesh: {mesh.elements_count} elements")

# %% [markdown]
"""
### Load faults and build surfaces

This step requires fault trace data.  If the data file doesn't exist,
an empty SurfaceCollection is returned and the workflow proceeds
without fault-controlled features.
"""

# %%
surfaces = h2ex.load_and_build_faults(mesh, config)
uw.pprint(f"Fault surfaces: {len(surfaces.surfaces)}")

# %% [markdown]
"""
### Adapt mesh (optional)

When `config.adapt=True` and faults are loaded, the mesh is refined
near fault surfaces using a distance-based metric.
"""

# %%
mesh = h2ex.adapt_mesh(mesh, surfaces, config)

# %% [markdown]
"""
### Save products

These are the expensive-to-compute objects.  Save them so Pattern B
can skip straight to solver setup.
"""

# %%
products.save("mesh", mesh)
if len(surfaces.surfaces) > 0:
    products.save("fault_surfaces", surfaces)

# Check what we saved
products.list()

# %% [markdown]
"""
## 4. Solver Setup
"""

# %%
stokes, v, p = h2ex.create_stokes(mesh, config)
fields = h2ex.setup_rheology(stokes, surfaces, mesh, config)
h2ex.set_boundary_conditions(stokes, mesh, config)

# %% [markdown]
"""
## 5. Solve
"""

# %%
stokes.solve(zero_init_guess=True)
uw.pprint(f"Solved.  v_max = {v.max()}")

# %% [markdown]
"""
## 6. Product Status

Cross-reference the workflow's product metadata against what's on disk.
"""

# %%
products.status(h2ex)

# %% [markdown]
"""
## 7. Pattern B — Reload Products for Parameter Study

In a separate session (or a second run), load pre-built products and
vary only the downstream parameters.
"""

# %%
# Uncomment to demonstrate product reload:
#
# products = WorkflowProducts(config)
# mesh = products.load("mesh")
# surfaces = products.load("fault_surfaces", mesh=mesh)
#
# # Change rheology without rebuilding the mesh
# config.rheology = "anisotropic"
# config.eta_1_ratio = 0.01  # More extreme contrast
#
# stokes, v, p = h2ex.create_stokes(mesh, config)
# fields = h2ex.setup_rheology(stokes, surfaces, mesh, config)
# h2ex.set_boundary_conditions(stokes, mesh, config)
# stokes.solve(zero_init_guess=True)

# %% [markdown]
"""
## 8. Visualization (optional)
"""

# %%
uw.pause("Waiting before visualisation")

# %%
h2ex.plot_results(mesh, stokes, surfaces=surfaces, config=config)

# %% [markdown]
"""
## Summary

This notebook demonstrates the product-aware workflow pattern:

1. **H2ExConfig** — validated, serializable parameters for fault-controlled flow
2. **WorkflowProducts** — make-like persistence (save, load, exists, status)
3. **DAG metadata** — `produces`/`requires` on each step, visible in `view()`
4. **Two patterns** — full serial build vs product-reload for parameter studies

The key insight: expensive setup (mesh, faults, adaptation) runs once,
cheap-to-vary steps (rheology, BCs, solve) run many times.
"""
