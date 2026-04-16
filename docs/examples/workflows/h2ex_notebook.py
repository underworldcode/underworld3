# %% [markdown]
r"""
# H2Ex Fault-Controlled Flow — Product-Aware Workflow

**PHYSICS:** crustal fluid flow through fault zones → surface accumulation

**DIFFICULTY:** advanced

**RUNTIME:** ~10 minutes (with adaptation); seconds (isoviscous, no adaptation)

## Description

Demonstrates the **product-aware workflow pattern** for hydrogen exploration.
The full pipeline:

1. Build geographic mesh and load fault surfaces
2. Adapt mesh near faults
3. Solve stress with TransverseIsotropic rheology → strain rate
4. Solve reference stress (no faults) → background strain rate
5. Compute permeability from strain-rate delta
6. Solve Darcy flow with computed permeability
7. Advect tracers → surface accumulation heat map

Two usage patterns:

- **Pattern A (full serial):** Build everything from scratch, save products.
- **Pattern B (product reload):** Load pre-built mesh and faults, vary
  stress azimuth or fault weakness without repeating the expensive steps.

## Key Concepts

- `WorkflowProducts` for make-like persistence
- `produces`/`requires` metadata on workflow steps for DAG visualisation
- Geographic mesh with fault-adaptive refinement
- Anisotropic rheology via TransverseIsotropicFlowModel
- Piecewise permeability from strain-rate anomaly
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
    adapt=False,                # Set True for mesh adaptation (requires MMG)
    num_elements=(8, 8, 4),     # Coarse for demonstration
    stress_azimuth_deg=0.0,     # E-W stress orientation
    fault_weakness=1e-5,        # Strong fault anisotropy
    max_advection_steps=100,    # Reduced for smoke test
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
assert config2.stress_azimuth_deg == config.stress_azimuth_deg
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

# %% [markdown]
"""
### Load faults and build surfaces

This step requires fault trace data.  If the data file doesn't exist,
an empty SurfaceCollection is returned and the workflow proceeds
without fault-controlled features.
"""

# %%
surfaces = h2ex.load_and_build_faults(mesh, config)

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
### Save mesh products

These are the expensive-to-compute objects.  Save them so Pattern B
can skip straight to the stress calculation.
"""

# %%
products.save("mesh", mesh)
if len(surfaces.surfaces) > 0:
    products.save("fault_surfaces", surfaces)

products.list()

# %% [markdown]
"""
## 4. Stress Calculations

Solve the Stokes equation twice:
- Once with fault weakness → strain rate on faults
- Once without faults → reference strain rate
"""

# %%
stokes, strain_rate = h2ex.solve_stress(mesh, surfaces, config)

# %%
strain_rate_ref = h2ex.solve_reference_stress(mesh, config)

# %% [markdown]
"""
## 5. Permeability

Compute permeability from the strain-rate delta (fault - reference).
Positive delta → enhanced permeability.
"""

# %%
permeability = h2ex.compute_permeability(mesh, strain_rate, strain_rate_ref, config)

# %% [markdown]
"""
## 6. Darcy Flow

Solve steady-state Darcy flow with the computed permeability.
Source at depth drives fluid upward through fault conduits.
"""

# %%
darcy, v_darcy, p_darcy = h2ex.solve_darcy(mesh, permeability, config)

# %% [markdown]
"""
## 7. Tracer Advection

Seed tracers throughout the domain and advect them in the
negative Darcy velocity.  Stopped at the surface, they form
an accumulation heat map.
"""

# %%
accumulation = h2ex.advect_tracers(mesh, v_darcy, config)

# %% [markdown]
"""
## 8. Product Status

Cross-reference the workflow's product metadata against what's on disk.
"""

# %%
products.status(h2ex)

# %% [markdown]
"""
## 9. Pattern B — Reload Products for Parameter Study

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
# # Change stress orientation without rebuilding the mesh
# config.stress_azimuth_deg = 45.0
#
# stokes, strain_rate = h2ex.solve_stress(mesh, surfaces, config)
# strain_rate_ref = h2ex.solve_reference_stress(mesh, config)
# permeability = h2ex.compute_permeability(mesh, strain_rate, strain_rate_ref, config)
# darcy, v_darcy, p_darcy = h2ex.solve_darcy(mesh, permeability, config)
# accumulation = h2ex.advect_tracers(mesh, v_darcy, config)

# %% [markdown]
"""
## 10. Visualization (optional)
"""

# %%
uw.pause("Waiting before visualisation")

# %%
h2ex.plot_results(mesh, v_darcy, surfaces=surfaces, config=config)

# %% [markdown]
"""
## Summary

This notebook demonstrates the full H2Ex pipeline:

1. **Mesh preparation** — geographic mesh, fault loading, adaptation
2. **Stress calculation** — TransverseIsotropic with oriented BCs
3. **Permeability** — derived from strain-rate anomaly
4. **Darcy flow** — driven by source at depth
5. **Accumulation** — tracers advected to surface, density heat map

The product-aware pattern (save/load/status) enables:

- **Parameter studies**: vary `stress_azimuth_deg` or `fault_weakness`
  without rebuilding the mesh
- **Reproducibility**: YAML config + products directory captures the full state
- **DAG visualization**: `view()` shows step dependencies
"""
