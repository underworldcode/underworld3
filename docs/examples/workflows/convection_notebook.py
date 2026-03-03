# %% [markdown]
r"""
# Thermal Convection — Workflow Pattern Example

**PHYSICS:** convection

**DIFFICULTY:** intermediate

**RUNTIME:** ~2 minutes (50 steps at 1/16 resolution)

## Description

Rayleigh–Bénard convection with constant viscosity, demonstrating the
**workflow package pattern** from `docs/developer/guides/workflow-packages.md`.

All parameters and setup functions live in `convection_config.py`,
imported as `convection`.  Every call reads as `convection.create_mesh(...)`,
making it clear where each helper comes from.  When building a standalone
package the config module becomes the pip-installable library and the
notebook goes in `notebooks/`.

## Key Concepts

- `WorkflowConfig` subclass for validated, serializable parameters
- Helper functions that return standard UW3 objects
- YAML round-trip for parameter management
- Coupled Stokes + advection–diffusion time-stepping

## Physical Setup

| Parameter | Value |
|-----------|-------|
| Domain | `aspect_ratio` × 1 (non-dimensional) |
| Ra | $10^6$ |
| Viscosity | 1 (constant) |
| BCs (velocity) | Free-slip all walls |
| BCs (temperature) | T = 1 bottom, T = 0 top |
| Initial T | Linear + sinusoidal perturbation |
"""

# %%
#| echo: false
# Required to fix pyvista (visualisation) crashes
# in interactive notebooks (including on binder)

import nest_asyncio
nest_asyncio.apply()

# %%
import os
import convection_config as convection

# %% [markdown]
"""
## 1. Configuration

All tunable parameters in one validated object.  Override defaults by keyword
or load from a YAML file.
"""

# %%
config = convection.ConvectionConfig(
    rayleigh=1e6,
    cellsize=1.0 / 16,
    n_steps=50,
)

config.view()

# %% [markdown]
"""
### YAML round-trip

Save parameters for reproducibility:
"""

# %%
os.makedirs(config.output_dir, exist_ok=True)
config.save_yaml(f"{config.output_dir}/params.yaml")

# Reload — identical config
config2 = convection.ConvectionConfig.from_yaml(f"{config.output_dir}/params.yaml")
assert config2.rayleigh == config.rayleigh
print("YAML round-trip OK")

# %% [markdown]
"""
## 2. Model Setup

`setup_model()` creates a `uw.Model` and registers any reference
quantities.  For non-dimensional convection the defaults are fine.
"""

# %%
model = config.setup_model()

# %% [markdown]
"""
## 3. Mesh
"""

# %%
mesh = convection.create_mesh(config)
mesh.view()

# %% [markdown]
"""
## 4. Solvers

Each helper returns standard UW3 objects — no wrappers.
"""

# %%
stokes, v, p = convection.create_stokes(mesh, config)
adv_diff, T = convection.create_advdiff(mesh, v, config)
convection.set_buoyancy(stokes, T, config)

# %%
convection.view()

# %% [markdown]
"""
## 5. Initial Conditions
"""

# %%
convection.set_initial_temperature(T, mesh, config)

# %% [markdown]
"""
## 6. Initial Solve
"""

# %%
import underworld3 as uw

stokes.solve(zero_init_guess=True)
adv_diff.solve(timestep=config.dt_factor * stokes.estimate_dt(), zero_init_guess=True)

uw.pprint(f"Ra = {config.rayleigh:.0e}, initial dt = {stokes.estimate_dt():.2e}")

# %% [markdown]
"""
## 7. Time Evolution
"""

# %%
for step in range(config.n_steps):
    stokes.solve(zero_init_guess=False)
    delta_t = config.dt_factor * stokes.estimate_dt()
    adv_diff.solve(timestep=delta_t, zero_init_guess=False)

    if step % config.save_every == 0 or step == config.n_steps - 1:
        uw.pprint(f"Step {step:4d}, dt = {delta_t:.2e}, T range = {T.stats()}\n ")

# %% [markdown]
"""
## 8. Visualization (optional)
"""

# %%
uw.pause("Waiting before visualisation")

# %%
convection.plot_temperature(mesh, T, v=v, config=config)

# %% [markdown]
"""
## Summary

This notebook demonstrates the workflow package pattern with the classic
Rayleigh–Bénard convection problem:

1. **Config** — `ConvectionConfig(WorkflowConfig)` validates all parameters
2. **Helpers** — `convection.create_mesh`, `convection.create_stokes`, etc. return UW3 objects
3. **Notebook** — clean script that imports the workflow module, configures, solves, visualizes
4. **YAML** — `save_yaml` / `from_yaml` for reproducible parameter management

To build your own workflow package, copy this structure into a separate
repository.  See `docs/developer/guides/workflow-packages.md` for the full guide.
"""
