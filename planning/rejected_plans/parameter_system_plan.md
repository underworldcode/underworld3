# Underworld3 Parameter Management System Plan

## Overview

This document outlines a plan for an enhanced parameter management system that integrates Underworld's existing `UWexpression` system with PETSc's options database to provide:

- Default parameter values with validation
- Command-line parameter overrides via PETSc options
- Claude-friendly parameter discovery and documentation
- Auto-generated parameter tables with symbols and descriptions

## Current State Analysis

Underworld3 currently uses:
- Global `PETSc.Options()` instances created locally in functions
- A prefixed `PETSc.Options("uw_")` in `_petsc_tools.py`
- Manual command-line parsing in `parse_cmd_line_options()`
- `UWexpression` system for connecting symbols, values, and descriptions
- No centralized parameter management with defaults

## Architecture Design

### Core Components

#### 1. ParameterExpression Class (extends UWexpression)

```python
class ParameterExpression(UWexpression):
    """
    UWexpression that synchronizes with PETSc options database.
    Combines symbol, value, description with PETSc command-line integration.
    """
    
    def __init__(self, name, default_value, description="No description", 
                 units=None, valid_range=None, petsc_option=None):
        # Initialize UWexpression with symbol and default
        super().__init__(name, sym=default_value, description=description)
        
        # Additional parameter metadata
        self.units = units
        self.valid_range = valid_range
        self.petsc_option = petsc_option or name.replace(" ", "_").lower()
        
        # Register with PETSc options database
        self._sync_with_petsc()
```

#### 2. Parameter Registry System

- **Storage**: Use PETSc options database as the single source of truth
- **Command Line**: Standard PETSc `-option_name value` syntax
- **Defaults**: Programmatic way to set defaults before PETSc reads command line
- **Documentation**: Rich metadata attached to each parameter via UWexpression

#### 3. Auto-Generated Documentation

```python
def generate_parameter_table(parameters):
    """Generate markdown table of all parameters"""
    table = "| Symbol | Parameter | Default | Description | Units | Command Line |\n"
    table += "|--------|-----------|---------|-------------|-------|-------------|\n"
    
    for param in parameters:
        table += f"| ${param.name}$ | {param.petsc_option} | {param.sym} | "
        table += f"{param.description} | {param.units or '-'} | `-{param.petsc_option}` |\n"
    
    return table
```

## Usage Patterns

### 1. Natural Parameter Declaration

Parameters reference constants defined earlier in the script, making notebooks and jupytext files easy to modify:

```python
import underworld3 as uw

# Define parameter constants at the top of the script/notebook
# These are easily visible and modifiable by users
MESH_RESOLUTION = 32
REFERENCE_VISCOSITY = 1e21
RAYLEIGH_NUMBER = 1e6
PRANDTL_NUMBER = 1.0

# Declare parameters near their usage, referencing the constants
mesh_resolution = ParameterExpression(
    name=r"N_{mesh}",                    # LaTeX symbol for docs
    default_value=MESH_RESOLUTION,       # References constant defined above
    description="Mesh resolution: number of elements per axis",
    units="elements",
    valid_range=(4, 512),
    petsc_option="mesh_resolution"       # Command line: -mesh_resolution 64
)

viscosity = ParameterExpression(
    name=r"\eta_0", 
    default_value=REFERENCE_VISCOSITY,   # References constant defined above
    description="Reference viscosity", 
    units="Pa⋅s",
    petsc_option="viscosity"
)

Ra = ParameterExpression(
    name=r"Ra", 
    default_value=RAYLEIGH_NUMBER,       # References constant defined above
    description="Rayleigh number", 
    petsc_option="rayleigh"
)

Pr = ParameterExpression(
    name=r"Pr", 
    default_value=PRANDTL_NUMBER,        # References constant defined above
    description="Prandtl number", 
    petsc_option="prandtl"
)
```

### 2. Mathematical Expression Integration

Parameters work naturally in sympy expressions:

```python
# Use in mathematical expressions naturally
buoyancy_force = density * gravity * mesh_resolution**2  # Sympy expression

# Parameter appears in mathematical formulation
temperature_equation = Ra * Pr * temperature_field + viscosity * velocity_gradient
```

### 3. Mesh and System Integration

```python
# Create mesh using parameter
mesh = uw.meshing.BoxMesh(
    elementRes=(mesh_resolution.sym, mesh_resolution.sym),  # Uses current value
    minCoords=(0, 0), 
    maxCoords=(1, 1)
)

# Access current value (from PETSc options database)
print(f"Using mesh resolution: {mesh_resolution.sym}")  # Default or command-line value
```

### 4. Command Line Usage

```bash
# Users can override defaults via standard PETSc options
python my_simulation.py -mesh_resolution 64 -viscosity 1e22 -rayleigh 1e7
```

## Claude-Friendly Features

### 1. Pattern Recognition for Easy Parsing
The constant-reference pattern makes it very easy for Claude to:
- **Identify parameter sections**: Constants are grouped at the top with clear naming
- **Understand parameter hierarchy**: Constants → ParameterExpression → Usage
- **Generate variations**: Can easily modify constants to create different scenarios
- **Trace parameter flow**: Follow from constant definition to expression to usage

```python
# Claude can easily parse this pattern:
# 1. Find constants section (ALL_CAPS variables)
# 2. Find ParameterExpression definitions referencing constants  
# 3. Find usage locations in mesh/solver setup
# 4. Generate modified versions by changing constants

# Example: Claude generating a high-resolution variant
HIGH_RES_MESH = 128        # Modified from original MESH_RESOLUTION = 32
FINE_VISCOSITY = 5e20     # Modified from REFERENCE_VISCOSITY = 1e21
```

### 2. Context Awareness
- Parameters declared near usage provide clear location hints
- Claude can understand where parameters affect the simulation

### 3. Mathematical Understanding
- LaTeX symbols (η₀, Ra, Pr) help Claude understand physical meaning
- Rich descriptions explain parameter significance

### 4. Command-Line Mapping
- Clear mapping between mathematical symbols and CLI options
- Claude can help users with command-line parameter setting

### 5. Documentation Integration
- Auto-generated parameter tables for documentation
- Type safety and validation help Claude provide better guidance

### 6. Parameter Discovery
```python
# List all parameters with their current values
def show_all_parameters():
    for param in ParameterExpression._registry:
        print(f"{param.name}: {param.sym} ({param.description})")
        print(f"  Command line: -{param.petsc_option}")
        print(f"  Units: {param.units}")
        print(f"  Valid range: {param.valid_range}")
```

### 7. Easy Example Generation for Users

Claude can easily generate parameter variations by modifying the constants section:

```python
# Original constants section
MESH_RESOLUTION = 32
REFERENCE_VISCOSITY = 1e21
RAYLEIGH_NUMBER = 1e6

# Claude can generate variations like:
# High-resolution study:
MESH_RESOLUTION = 128
REFERENCE_VISCOSITY = 1e21
RAYLEIGH_NUMBER = 1e6

# Low-viscosity study:  
MESH_RESOLUTION = 32
REFERENCE_VISCOSITY = 1e19
RAYLEIGH_NUMBER = 1e6

# Parameter sweep study:
MESH_RESOLUTIONS = [32, 64, 128]
for res in MESH_RESOLUTIONS:
    MESH_RESOLUTION = res
    # ... run simulation
```

## Integration Points

### 1. PETSc Options Database
- All parameters stored in PETSc options for consistent access
- Automatic synchronization between expressions and options
- Command-line parsing handled by PETSc

### 2. Existing UWexpression System
- Extends familiar UWexpression interface
- Maintains compatibility with existing mathematical expressions
- Preserves sympy integration

### 3. Documentation Generation
- Automatic parameter table generation for user manuals
- LaTeX symbol rendering in documentation
- Parameter metadata for API docs

## Example Documentation Output

| Symbol | Parameter | Default | Description | Units | Command Line |
|--------|-----------|---------|-------------|-------|-------------|
| $N_{mesh}$ | mesh_resolution | 32 | Mesh resolution: number of elements per axis | elements | `-mesh_resolution` |
| $\eta_0$ | viscosity | 1e21 | Reference viscosity | Pa⋅s | `-viscosity` |
| $Ra$ | rayleigh | 1e6 | Rayleigh number | - | `-rayleigh` |
| $Pr$ | prandtl | 1.0 | Prandtl number | - | `-prandtl` |

## Benefits

### For Users
- **Familiar Interface**: Builds on existing UWexpression system
- **Clear Documentation**: Rich parameter descriptions with units and symbols
- **Command-Line Control**: Easy parameter override via PETSc options
- **Validation**: Built-in parameter validation and type checking

### For Claude
- **Context Understanding**: Parameters declared where they're used
- **Mathematical Meaning**: LaTeX symbols provide physical context
- **Parameter Discovery**: Easy to find and understand all parameters
- **Documentation Integration**: Auto-generated tables for reference

### For Developers
- **Centralized Management**: Single system for all parameter handling
- **PETSc Integration**: Leverages existing PETSc infrastructure
- **Backward Compatibility**: Works with existing code patterns
- **Extensible**: Easy to add new parameters and metadata

## Implementation Steps

1. **Create ParameterExpression class** extending UWexpression
2. **Add PETSc synchronization** for command-line integration
3. **Implement validation** and type checking
4. **Create documentation generators** for parameter tables
5. **Add parameter discovery** and introspection methods
6. **Write comprehensive tests** for the parameter system
7. **Create usage examples** for common Underworld scenarios

## Future Enhancements

- **Config file support**: YAML/JSON parameter file loading
- **Parameter groups**: Organize related parameters (solver, physical, etc.)
- **Dynamic validation**: Runtime parameter constraint checking
- **Parameter history**: Track parameter changes during simulation
- **GUI integration**: Web-based parameter configuration interface

This system will provide a robust, user-friendly, and Claude-compatible parameter management solution that enhances the Underworld3 user experience while maintaining integration with PETSc's powerful options system.