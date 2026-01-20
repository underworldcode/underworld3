---
title: "Script Parameters"
description: "Configurable parameters for notebooks and command-line scripts"
authors:
- name: Underworld Team
---

## Overview

`uw.Params` provides a clean way to define configurable parameters that can be:

1. **Set in notebooks** - Just assign new values
2. **Overridden from command line** - Via PETSc options (`-uw_param_name value`)

This makes scripts portable between interactive development and HPC batch execution.

## Basic Usage

```python
import underworld3 as uw

# Define parameters with defaults
params = uw.Params(
    uw_resolution = 0.05,    # Cell size for mesh
    uw_diffusivity = 1.0,    # Material property
    uw_max_steps = 100,      # Integer parameter
    uw_verbose = True,       # Boolean flag
    uw_solver = "mumps",     # String option
)

# Use in your model
mesh = uw.meshing.Box(cellSize=params.uw_resolution)
```

### Naming Convention

Use the `uw_` prefix for parameter names. This:

- Avoids collisions with PETSc solver options
- Makes it clear these are underworld example parameters
- The CLI flag matches the Python name exactly

### Command Line Override

```bash
# Override resolution and solver
python script.py -uw_resolution 0.025 -uw_solver superlu_dist

# Works with mpirun
mpirun -np 4 python script.py -uw_resolution 0.01
```

### Notebook Override

```python
# Just assign a new value
params.uw_resolution = 0.025
```

## Unit-Aware Parameters

For physical quantities with units, use the `uw.Param()` wrapper:

```python
params = uw.Params(
    # Physical quantities with units
    uw_cell_size = uw.Param(0.5, units="km", description="Mesh cell size"),
    uw_viscosity = uw.Param(1e21, units="Pa*s"),
    uw_latitude = uw.Param(45.0, units="degree"),

    # Dimensionless ratios
    uw_rayleigh = uw.Param(1e5, type=uw.ParamType.RATIO,
                           description="Rayleigh number"),

    # Plain types still work
    uw_elements = 32,
)
```

### CLI with Units

Units **must** be provided on the command line for unit-aware parameters:

```bash
# These work
python script.py -uw_cell_size "500 m"
python script.py -uw_cell_size 0.5km
python script.py -uw_viscosity "1e22 Pa*s"
python script.py -uw_latitude "0.785 radian"  # Converts to degrees

# These fail with helpful errors
python script.py -uw_cell_size 500        # ERROR: Units required
python script.py -uw_cell_size "500 s"    # ERROR: Dimension mismatch
```

### Accessing Values

Unit-aware parameters return `UWQuantity` objects that integrate with the scaling system:

```python
cell_size = params.uw_cell_size      # UWQuantity(0.5, "km")
cell_size_m = cell_size.to("meter")  # Convert to meters

# For mesh creation (handles unit conversion automatically)
mesh = uw.meshing.Box(cellSize=params.uw_cell_size)
```

## Parameter Options

The `uw.Param()` wrapper supports:

| Option | Description |
|--------|-------------|
| `value` | Default value (required) |
| `units` | Unit string, e.g., `"km"`, `"Pa*s"`, `"degree"` |
| `type` | Explicit type: `uw.ParamType.QUANTITY`, `RATIO`, `INTEGER`, etc. |
| `bounds` | Tuple `(min, max)` for validation |
| `description` | Help text shown in `cli_help()` |

### Bounds Validation

```python
params = uw.Params(
    uw_cell_size = uw.Param(0.5, units="km",
                            bounds=(0.01, 100),
                            description="Must be 0.01-100 km"),
)

# This would raise ValueError
# python script.py -uw_cell_size "0.001 km"  # Below minimum
```

## Inspecting Parameters

### Display in Notebooks

`Params` has rich display with source tracking:

```python
params  # Shows table with values, units, types, and sources
```

| Parameter | Value | Units | Type | Source |
|-----------|-------|-------|------|--------|
| `uw_cell_size` | `0.5 kilometer` | km | quantity | default |
| `uw_viscosity` | `1e+21 pascal * second` | Pa*s | quantity | **CLI** |

### CLI Help

```python
print(params.cli_help())
```

```
Command-line options (PETSc format):

  -uw_cell_size <quantity>   Units: km
                             Must be 0.01-100 km
                             (default: 0.5 km)

  -uw_viscosity <quantity>   Units: Pa*s
                             (default: 1e+21 Pa*s)

  -uw_rayleigh <ratio>       Dimensionless ratio
                             Rayleigh number
                             (default: 100000.0)

Example:
  python script.py -uw_cell_size 0.5km -uw_viscosity 1e+21Pa*s
```

## Complete Example

```python
import underworld3 as uw

# Define all configurable parameters at the top
params = uw.Params(
    # Mesh parameters
    uw_cell_size = uw.Param(50.0, units="km",
                            bounds=(10, 200),
                            description="Target cell size"),
    uw_depth = uw.Param(660.0, units="km",
                        description="Model depth"),

    # Physical properties
    uw_viscosity = uw.Param(1e21, units="Pa*s"),
    uw_density_diff = uw.Param(50.0, units="kg/m^3"),

    # Solver settings
    uw_max_iterations = 50,
    uw_tolerance = 1e-6,
)

# Show help (useful at script start)
if uw.mpi.rank == 0:
    print(params.cli_help())

# Build model using parameters
mesh = uw.meshing.Box(
    minCoords=(0, 0),
    maxCoords=(uw.quantity(2000, "km"), params.uw_depth),
    cellSize=params.uw_cell_size,
)

# ... rest of model setup
```

Run with:

```bash
# Default parameters
python convection.py

# Override for higher resolution
python convection.py -uw_cell_size 25km -uw_depth 1000km

# HPC run with many overrides
mpirun -np 256 python convection.py \
    -uw_cell_size 10km \
    -uw_viscosity "5e20 Pa*s" \
    -uw_max_iterations 100
```

## Angle Units

Angles work naturally - you can define in degrees and provide radians (or vice versa):

```python
params = uw.Params(
    uw_latitude = uw.Param(45.0, units="degree"),
)

# CLI accepts any angle unit
# python script.py -uw_latitude "0.785 radian"
# python script.py -uw_latitude "45 deg"

# Convert as needed
lat_rad = params.uw_latitude.to("radian")
```
