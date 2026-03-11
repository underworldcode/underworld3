# Building Workflow Packages on Underworld3

Complex simulations — hydrogen exploration, groundwater modelling, mantle
convection — share a large amount of structural boilerplate (mesh setup,
reference quantities, solver configuration) on top of a smaller body of
domain-specific code.  Underworld3 provides **infrastructure and patterns**
so that domain-specific workflows can be built and pip-installed as
independent packages.

## What goes where

| Content | Location | Maintained by |
|---------|----------|---------------|
| `WorkflowConfig` base class | `underworld3.workflows` | UW3 team |
| Common utilities | `underworld3.workflows` | UW3 team |
| Pattern docs (this guide) | `docs/developer/guides/` | UW3 team |
| Example workflow (convection) | `docs/examples/workflows/` | UW3 team (demo) |
| Real workflow packages | Separate repos | Domain communities |

## Quick start

```python
from underworld3.workflows import WorkflowConfig
from pydantic import Field

class HydrogenConfig(WorkflowConfig):
    """Parameters for hydrogen exploration simulations."""

    region: str = Field(default="eyre_peninsula")
    depth_km: float = Field(default=50.0, gt=0)
    fault_influence_km: float = Field(default=5.0, gt=0)

    # Reference quantities for non-dimensionalisation
    ref_length: str = "50 km"
    ref_viscosity: str = "1e21 Pa*s"
```

`WorkflowConfig` gives you:

- **Pydantic validation** — type checks, bounds, IDE autocompletion.
- **YAML serialisation** — `config.save_yaml("params.yaml")` / `Config.from_yaml(...)`.
- **Model integration** — `config.setup_model()` creates a `uw.Model` with reference
  quantities already set.
- **Extra fields** — `extra="allow"` means users can add ad-hoc parameters without
  subclassing.

## Design principles

### 1. Helpers return standard UW3 objects

```python
# GOOD — returns a mesh, caller decides what to do with it
def create_mesh(config: HydrogenConfig):
    mesh = uw.meshing.UnstructuredSimplexBox(...)
    return mesh

# BAD — hides UW3 objects behind a wrapper
class HydrogenModel:
    def __init__(self, config):
        self._mesh = uw.meshing.UnstructuredSimplexBox(...)
```

Helpers are **convenience functions**, not an abstraction layer.  Users who
need finer control can call the UW3 API directly.

### 2. No automatic execution

Setup functions configure objects; the user calls `solve()`:

```python
mesh = create_mesh(config)
stokes = setup_stokes(mesh, config)

# User controls the solve
stokes.solve()
```

This keeps the notebook readable and debuggable.

### 3. Dependency checking at import time

If your workflow needs optional packages (e.g. `geopandas`, `stripy`),
check them early:

```python
from underworld3.workflows import check_dependencies

check_dependencies({
    "geopandas": "pip install geopandas",
    "stripy": "pip install stripy",
})
```

This gives users a clear error message with install instructions instead of
a confusing `ImportError` deep in the solve loop.

### 4. Progressive disclosure

| User level | Experience |
|------------|------------|
| Student | `config = FaultFlowConfig(); mesh = create_mesh(config)` |
| Researcher | Reads `create_mesh` source, tweaks parameters |
| Expert | Uses raw `uw.meshing` API with their own mesh generation |

The workflow helpers are the **on-ramp**, not the only way.

## Package structure

A workflow package lives in its own repo and is pip-installable:

```
uw3-hydrogen/
    pyproject.toml
    src/uw3_hydrogen/
        __init__.py
        config.py           # HydrogenConfig(WorkflowConfig)
        helpers.py           # create_mesh, setup_stokes, load_faults, ...
        data/                # Bundled reference data (optional)
    notebooks/
        tutorial.ipynb
```

### `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68", "setuptools-scm"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "uw3-hydrogen"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "underworld3",
]

[project.optional-dependencies]
geo = ["geopandas", "stripy"]
```

Install: `pip install uw3-hydrogen` or `pixi add --pypi uw3-hydrogen`.

### `config.py`

```python
from underworld3.workflows import WorkflowConfig
from pydantic import Field

class HydrogenConfig(WorkflowConfig):
    region: str = "eyre_peninsula"
    depth_km: float = Field(default=50.0, gt=0)
    fault_influence_km: float = Field(default=5.0, gt=0)
    cellsize_km: float = Field(default=2.5, gt=0)

    ref_length: str = "50 km"
    ref_viscosity: str = "1e21 Pa*s"
```

### `helpers.py`

```python
import underworld3 as uw
from .config import HydrogenConfig

def create_mesh(config: HydrogenConfig):
    """Build a simplex mesh for the study region."""
    depth = uw.quantity(config.depth_km, "km")
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, -float(depth.to("m").magnitude)),
        maxCoords=(200e3, 0.0),
        cellSize=config.cellsize_km * 1e3,
    )
    return mesh

def setup_stokes(mesh, config: HydrogenConfig):
    """Configure a Stokes solver with isotropic viscosity."""
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    return stokes
```

### User notebook

Import the workflow as a module so every call is self-documenting:

```python
import underworld3 as uw
import uw3_hydrogen as hydrogen

config = hydrogen.HydrogenConfig(region="eyre_peninsula", depth_km=50)
model = config.setup_model()
mesh = hydrogen.create_mesh(config)
stokes = hydrogen.setup_stokes(mesh, config)
stokes.solve()
```

## `WorkflowConfig` API reference

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `workflow_name` | `str` | `""` | Short identifier |
| `description` | `str` | `""` | Human-readable description |
| `output_dir` | `str` | `"output"` | Output directory |
| `ref_length` | `str` or `None` | `None` | e.g. `"1000 km"` |
| `ref_viscosity` | `str` or `None` | `None` | e.g. `"1e21 Pa*s"` |
| `ref_diffusivity` | `str` or `None` | `None` | e.g. `"1e-6 m**2/s"` |
| `ref_temperature` | `str` or `None` | `None` | e.g. `"1500 kelvin"` |
| `ref_density` | `str` or `None` | `None` | e.g. `"3300 kg/m**3"` |
| `ref_velocity` | `str` or `None` | `None` | e.g. `"5 cm/year"` |

### Methods

- **`setup_model(name=None)`** — Create/reset a `uw.Model`, register reference
  quantities, store config in metadata.  Returns the model.
- **`save_yaml(path)`** — Serialise to YAML.
- **`from_yaml(path)`** (classmethod) — Deserialise from YAML.  Works with
  subclasses: `HydrogenConfig.from_yaml("params.yaml")` returns a
  `HydrogenConfig`.

## Utility functions

```python
from underworld3.workflows import check_dependencies, parse_quantity
```

- **`check_dependencies(packages)`** — Verify optional imports are available.
  `packages` maps import names to install instructions.
- **`parse_quantity(s)`** — Parse `"1000 km"` into `uw.quantity(1000, "km")`.

## Products and persistence

Real workflows have **expensive serial steps** (mesh adaptation, fault
processing) and **cheap-to-vary parallel steps** (Stokes solve with different
rheology).  The product system lets you save the output of expensive steps
and reload them for parameter studies — like `make` rules with targets,
prerequisites, and recipes.

### `WorkflowProducts` — named product manager

```python
from underworld3.workflows import WorkflowProducts

products = WorkflowProducts(config)       # uses config.output_dir/products
products = WorkflowProducts(products_dir="my_products")  # explicit path

# Save products after expensive steps
products.save("adapted_mesh", mesh)
products.save("fault_surfaces", surface_collection)

# Check what's available
products.exists("adapted_mesh")   # True
products.list()                   # HTML table in Jupyter, plain text otherwise

# Load products in a later session
mesh = products.load("adapted_mesh")
surfaces = products.load("fault_surfaces", mesh=mesh)

# Cross-reference with workflow DAG
products.status(h2ex_module)      # shows built vs missing products

# Cleanup
products.remove("adapted_mesh")
products.clear()                  # remove all
```

Type dispatch is automatic:

| Object Type | Save Format | Notes |
|-------------|-------------|-------|
| Mesh | HDF5 via `write_timestep` | Portable, ParaView-readable |
| MeshVariable | HDF5 via `write_timestep` | Vertex values + XDMF; kd-tree reload |
| Surface | VTK (`.save()`) | |
| SurfaceCollection | Directory of VTK files | |
| ndarray | `.npz` | |
| Other | YAML (if serializable) | Fallback |

MeshVariables are loaded via `read_timestep`, which uses kd-tree interpolation.
This means you can save on one mesh and reload onto a different one (e.g. after
re-adaptation):

```python
# Save a variable
products.save("temperature", T)

# Load onto same or different mesh — pass the target variable
products.load("temperature", mesh_variable=T_new)
```

A YAML manifest (`products/manifest.yaml`) tracks product names, types,
file paths, and timestamps.

### Step metadata — `produces` and `requires`

Workflow steps can declare what products they create and depend on:

```python
@workflow_step(
    description="Adapt mesh near fault surfaces",
    produces=["adapted_mesh"],
    requires=["mesh", "fault_surfaces"],
)
def adapt_mesh(mesh, surfaces, config):
    ...
```

This metadata is used by:
- **`view(module)`** — shows a DAG table with Produces / Requires columns
- **`products.status(module)`** — cross-references declared products with
  what exists on disk

There is no automatic DAG traversal — the user orchestrates in the notebook.
The infrastructure provides save/load/exists and DAG visualisation.

### Two notebook patterns

**Pattern A — Full serial build** (development / first run):

```python
mesh = h2ex.create_mesh(config)
surfaces = h2ex.load_and_build_faults(mesh, config)
mesh = h2ex.adapt_mesh(mesh, surfaces, config)
products.save("adapted_mesh", mesh)

stokes, strain_rate = h2ex.solve_stress(mesh, surfaces, config)
strain_rate_ref = h2ex.solve_reference_stress(mesh, config)
permeability = h2ex.compute_permeability(mesh, strain_rate, strain_rate_ref, config)
darcy, v_darcy, p_darcy = h2ex.solve_darcy(mesh, permeability, config)
accumulation = h2ex.advect_tracers(mesh, v_darcy, config)
```

**Pattern B — Product reload** (parameter studies):

```python
mesh = products.load("adapted_mesh")
surfaces = products.load("fault_surfaces", mesh=mesh)

# Vary stress orientation without rebuilding the mesh
config.stress_azimuth_deg = 45.0

stokes, strain_rate = h2ex.solve_stress(mesh, surfaces, config)
strain_rate_ref = h2ex.solve_reference_stress(mesh, config)
permeability = h2ex.compute_permeability(mesh, strain_rate, strain_rate_ref, config)
darcy, v_darcy, p_darcy = h2ex.solve_darcy(mesh, permeability, config)
```

The expensive steps (mesh, faults, adaptation) run once; the downstream
steps (stress, permeability, Darcy flow) run many times with different
stress orientations or fault weakness values.

## In-repo examples

See `docs/examples/workflows/` for complete working examples:

- `convection_config.py` — `ConvectionConfig(WorkflowConfig)` with helpers
- `convection_notebook.py` — Clean notebook using the config + helpers
- `h2ex_config.py` — `H2ExConfig(WorkflowConfig)` with product-aware steps
- `h2ex_notebook.py` — Product-aware notebook with save/load patterns

The convection example demonstrates the basic pattern; the H2Ex example
demonstrates the product-aware extension for multi-stage workflows.
