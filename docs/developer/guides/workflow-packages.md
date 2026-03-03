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

## In-repo example

See `docs/examples/workflows/` for a complete working example:

- `convection_config.py` — `ConvectionConfig(WorkflowConfig)` with helpers
- `convection_notebook.py` — Clean notebook using the config + helpers

This serves as the template to copy when starting a new workflow package.
