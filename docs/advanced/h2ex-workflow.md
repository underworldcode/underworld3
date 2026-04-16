# H2Ex: Fault-Controlled Hydrogen Exploration Workflow

This workflow models crustal fluid flow through fault zones to predict where
hydrogen-bearing fluids might reach the surface. It was developed for the
Eyre Peninsula (South Australia) but applies to any region with mapped fault data.

## Scientific Background

Natural hydrogen seeps occur where deep fluids migrate upward through
crustal fracture networks. The key physical insight is that faults act as
**anisotropic conduits**: they are weak in the fault-parallel direction,
concentrating strain and creating pathways for fluid flow.

The workflow computes permeability from mechanical stress, then solves for
fluid flow through that permeability field. This links **structural geology**
(fault geometry and orientation) to **hydrogeology** (fluid migration paths).

## Pipeline Overview

The workflow has six stages, each implemented as a `@workflow_step` function
in the `h2ex_config` module:

```
mesh → faults → adapt → stress → permeability → Darcy flow
                                                     ↓
                                              tracer advection
                                                     ↓
                                            surface accumulation
```

| Step | Function | What it does |
|------|----------|--------------|
| 1 | `create_mesh` | Geographic mesh over lon/lat/depth region |
| 2 | `load_and_build_faults` | Load fault traces, extrude to 3D surfaces |
| 3 | `adapt_mesh` | Refine mesh near fault surfaces |
| 4 | `solve_stress` | Stokes with anisotropic rheology on faults |
| 5 | `solve_reference_stress` | Stokes without faults (background) |
| 6 | `compute_permeability` | Strain-rate anomaly → permeability field |
| 7 | `solve_darcy` | Steady-state Darcy flow with computed $\kappa$ |
| 8 | `advect_tracers` | Particle tracking → surface density map |

## How Permeability Is Computed

The permeability field is derived from the **strain-rate anomaly** —
the difference between strain rate with faults and without:

$$\Delta\dot\varepsilon = \dot\varepsilon_\text{faults} - \dot\varepsilon_\text{reference}$$

Where faults concentrate strain ($\Delta\dot\varepsilon > 0$), permeability is
enhanced. The mapping uses configurable thresholds on a log$_{10}$ scale:

| Strain-rate delta | Permeability | Interpretation |
|-------------------|-------------|----------------|
| $\Delta < \delta_\text{low}$ | $10^{-2}$ | Strain shadow (reduced flow) |
| $\delta_\text{low} \le \Delta < 0$ | $10^{-1}$ | Mildly reduced |
| $0 \le \Delta \le \delta_\text{high}$ | $10^{0}$ | Background |
| $\delta_\text{high} < \Delta \le \delta_\text{very high}$ | $10^{1}$ | Enhanced (fault zone) |
| $\Delta > \delta_\text{very high}$ | $10^{2}$ | Highly enhanced (fault core) |

This is a simplified model. In practice, permeability depends on many factors
(mineralogy, confining pressure, fluid chemistry). The threshold-based approach
captures the first-order effect: faults that slip more carry more fluid.

## Stress Calculation

The stress field comes from solving the Stokes equation with:

- **Anisotropic rheology**: `TransverseIsotropicFlowModel` with fault normals
  as directors. Viscosity is reduced in the fault-parallel direction by a
  factor `fault_weakness` (typically $10^{-5}$).
- **Oriented boundary conditions**: velocity BCs on vertical faces at a
  configurable azimuth angle, representing the regional stress field.
- **Free surface**: stress-free top boundary (open to flow).
- **Free-slip base**: no vertical velocity at the bottom.

The reference solve uses `ViscousFlowModel` (isotropic, uniform viscosity)
with identical boundary conditions. The difference in strain rate isolates
the fault contribution.

## Configuration

All parameters are in the `H2ExConfig` class (Pydantic-validated):

```python
import h2ex_config as h2ex

config = h2ex.H2ExConfig(
    # Domain
    lon_range=(135.5, 137.5),
    lat_range=(-34.5, -33.0),
    depth_range_km=(0.0, 50.0),
    num_elements=(16, 16, 8),

    # Faults
    fault_data_path="Structures/faults_as_swarm_points_xyz.npz",
    trace_resolution_km=3.0,

    # Mesh adaptation
    adapt=True,
    h_near_km=2.0,      # Fine elements near faults
    h_far_km=20.0,       # Coarse elements far from faults
    transition_km=10.0,  # Transition width

    # Stress
    stress_azimuth_deg=30.0,   # Regional stress orientation
    fault_weakness=1e-5,        # Fault-zone viscosity ratio

    # Permeability thresholds
    delta_low=-2.0,
    delta_high=5.0,
    delta_very_high=10.0,

    # Darcy flow
    source_depth_km=30.0,
    source_magnitude=10.0,
)

config.view()   # Display all parameters as a table
config.save_yaml("params.yaml")  # Save for reproducibility
```

## Running the Workflow

### Full serial pipeline

```python
from underworld3.workflows import WorkflowProducts

model = config.setup_model()
products = WorkflowProducts(config)

# Expensive steps — run once, save products
mesh = h2ex.create_mesh(config)
surfaces = h2ex.load_and_build_faults(mesh, config)
mesh = h2ex.adapt_mesh(mesh, surfaces, config)
products.save("mesh", mesh)
products.save("fault_surfaces", surfaces)

# Stress and permeability
stokes, strain_rate = h2ex.solve_stress(mesh, surfaces, config)
strain_rate_ref = h2ex.solve_reference_stress(mesh, config)
permeability = h2ex.compute_permeability(
    mesh, strain_rate, strain_rate_ref, config
)

# Darcy flow
darcy, v_darcy, p_darcy = h2ex.solve_darcy(mesh, permeability, config)

# Tracer accumulation (optional)
accumulation = h2ex.advect_tracers(mesh, v_darcy, config)
```

### Parameter study (reload products)

```python
# Later session: skip mesh building
products = WorkflowProducts(config)
mesh = products.load("mesh")
surfaces = products.load("fault_surfaces", mesh=mesh)

# Vary stress orientation
config.stress_azimuth_deg = 45.0

stokes, strain_rate = h2ex.solve_stress(mesh, surfaces, config)
strain_rate_ref = h2ex.solve_reference_stress(mesh, config)
permeability = h2ex.compute_permeability(
    mesh, strain_rate, strain_rate_ref, config
)
darcy, v_darcy, p_darcy = h2ex.solve_darcy(mesh, permeability, config)
```

## Fault Data Format

Fault traces are stored as a NumPy `.npz` file with one array per fault:

```python
import numpy as np

# Each fault is an (N, 2) array of (longitude, latitude) points
np.savez("faults.npz",
    fault_A=np.array([[135.8, -34.0], [136.0, -33.8], [136.2, -33.6]]),
    fault_B=np.array([[136.5, -34.2], [136.7, -34.0], [136.9, -33.8]]),
)
```

The workflow extrudes each trace to a 3D surface spanning the configured
depth range, interpolates points to uniform spacing (`trace_resolution_km`),
and computes surface normals for the anisotropic rheology.

## Inspecting the Workflow

```python
# Show all steps with their dependencies (produces/requires)
h2ex.view()

# Check which products exist on disk
products.status(h2ex)
```

## Requirements

- **Underworld3** with the `workflows` module
- **pyvista** (optional, for fault triangulation and visualization)
- **MMG** (optional, for mesh adaptation — available in `amr-dev` environment)
- Fault trace data in `.npz` format

## See Also

- {doc}`../developer/guides/workflow-packages` — How to build workflow packages
- {doc}`porous-flow` — Darcy and Richards solver documentation
- {doc}`mesh-adaptation` — Mesh adaptation guide
