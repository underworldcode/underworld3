# Underworld3 Notebook Style Guide

## Overall Approach

Notebooks should be **simple, informal, and educational** - not overly congratulatory or verbose.

## Key Principles

1. **Jupyter-Native Display**
   - Use implicit display (last line of cell) to show results
   - Prefer `object` over `print(object)` when showing a single result
   - Use markdown cells for explanations and context

2. **Minimal Print Statements**
   - Avoid excessive `uw.pprint()` - notebooks are single-process
   - Use `print()` sparingly, mainly for showing multiple related values
   - Let Jupyter's display system do the work

3. **Tone**
   - Simple and direct
   - Informal but professional
   - Not overly congratulatory (avoid "Amazing!", "Fantastic!", etc.)
   - Focus on teaching, not cheerleading

4. **Code Style**
   - Show the "uw3 way" of doing things
   - Use implicit display for single results
   - Group related output with formatted strings when needed

## Examples

### Good Style

```python
# Create a quantity
temperature = 1500 * uw.units.K

# Display it (implicit)
temperature
```

```python
# Show multiple related values
print(f"Temperature: {temperature}")
print(f"Pressure: {pressure}")
print(f"Density: {density}")
```

### Avoid

```python
# Too verbose
print("Creating temperature...")
temperature = 1500 * uw.units.K
print(f"Temperature created: {temperature}")
print("Success!")  # Unnecessary
```

## Notebook Structure

1. **Title and Introduction** (markdown)
   - What the notebook teaches
   - Brief list of topics

2. **Import Cell**
   ```python
   import nest_asyncio
   nest_asyncio.apply()

   import underworld3 as uw
   import numpy as np
   ```

3. **Parameters Cell** (see [Parameters and Configuration](#parameters-and-configuration) below)
   - Named constants for defaults, then `uw.Params` block
   - Markdown cell above explaining CLI override syntax

4. **Concept Sections**
   - Markdown header (##) for each major concept
   - Brief explanation in markdown
   - Code cells demonstrating the concept
   - Minimal output cells (let Jupyter display)

5. **Summary** (markdown)
   - Key takeaways in bullet points
   - When to use what

6. **Try It Yourself** (markdown)
   - Optional exercises in code fences
   - Encourage exploration

## Display Patterns

### Showing a Single Value
```python
# Implicit display
temperature.units
```

### Showing Multiple Values
```python
print(f"Temperature: {T.min():.1f} to {T.max():.1f}")
print(f"Pressure: {P.min():.1f} to {P.max():.1f}")
```

### Inspecting Objects
```python
# Let Jupyter display the repr
mesh.units
```

```python
# Or use .view() for detailed info
mesh.view()
```

## Parameters and Configuration

Every notebook or example script that accepts tuneable settings should use
`uw.Params`.  The standard pattern has two parts:

1. **Named constants** — plain Python variables holding the default values.
   These are the first thing a notebook user sees and edits.
2. **`uw.Params` block** — wraps the constants with units, bounds,
   descriptions, and CLI override support.

### Standard Pattern

A markdown cell introduces the parameters and shows CLI usage:

~~~markdown
### Configurable parameters

Default values are defined as named constants below.  From the command
line, override them with PETSc-style flags:

```bash
python script.py -uw_viscosity "5e20 Pa*s" -uw_cell_size 25km
```
~~~

Followed by the code cell:

```python
# --- Default values (edit these in a notebook) ---
VISCOSITY  = 1e21   # Pa·s – reference viscosity
CELL_SIZE  = 50.0   # km – target cell size
DEPTH      = 660.0  # km – model depth
MAX_STEPS  = 100    # solver iterations

params = uw.Params(
    uw_viscosity = uw.Param(VISCOSITY, units="Pa*s", description="reference viscosity"),
    uw_cell_size = uw.Param(CELL_SIZE, units="km",   description="target cell size"),
    uw_depth     = uw.Param(DEPTH,     units="km",   description="model depth"),
    uw_max_steps = MAX_STEPS,
)
```

### Why Named Constants

- **Visibility**: The reader sees the default values at a glance without
  having to parse the `uw.Param(...)` wrapper.
- **Editability**: In a notebook, changing a default is a single number
  edit at the top of the cell — no need to find it inside a function call.
- **Separation of concerns**: The constants say *what* the defaults are;
  the `uw.Params` block says *how* they are validated and overridden.

### Naming Conventions

- Named constants: `UPPER_CASE` with a brief inline comment showing
  units and purpose.
- Parameter names: `uw_` prefix to avoid PETSc option collisions.
- Add a `description=` string for any parameter that will appear in
  `params.cli_help()`.

### What Not To Do

```python
# Avoid: inline literals with no named constant
params = uw.Params(
    uw_viscosity = uw.Param(1e21, units="Pa*s"),  # hard to scan
)

# Avoid: parameters scattered through the notebook
viscosity = 1e21        # defined in cell 3
# ... 20 cells later ...
params.uw_viscosity = viscosity   # reader has lost context
```

## The workflow pattern (provisional)

> **Status: provisional.** The pattern below is the target for example
> notebooks. It depends on `uw.workflows`, which lives on
> `feature/fault-system-workflow` and is not in a release, and on
> [#760](https://github.com/underworldcode/underworld3/issues/760), which has
> to be settled before `produces`/`requires` mean what this section says they
> mean. Until both land, examples are plain scripts and notebooks following
> the rest of this guide. Written down now so the design is not lost.

An example is **two files**, not one:

- `<topic>_config.py` — the library half. In a standalone package this is what
  gets pip-installed.
- `<topic>_notebook.py` — the user half, in jupytext percent format, which
  opens as a notebook and stores as reviewable text.

Calls in the notebook read `convection.create_mesh(config)`, so where every
helper comes from is visible on the page.

### One validated config object

Every tunable is a field on a `WorkflowConfig` subclass, carrying its own
bounds and description:

```python
class ConvectionConfig(WorkflowConfig):
    """Parameters for constant-viscosity Rayleigh–Bénard convection."""

    rayleigh: float = Field(default=1e6, gt=0, description="Rayleigh number")
    cellsize: float = Field(default=1.0 / 16, gt=0, description="Mesh cell size")
    n_steps: int = Field(default=50, ge=0, description="Number of time steps")
```

`config.save_yaml(...)` and `Config.from_yaml(...)` make the parameters of a
run a file rather than a memory of which flags were typed. (This overlaps with
[Parameters and Configuration](#parameters-and-configuration) above, which
specifies `uw.Params`; which applies where is part of #760.)

### Steps declare what they make and what they need

```python
@workflow_step(
    description="Solve to steady state and checkpoint",
    produces=["T_checkpoint"],
    requires=["mesh"],
)
def solve(mesh, config): ...
```

`uw.workflows.view(module)` renders those as a table, so the dependency chart
is generated from the steps and nobody maintains a diagram separately.
`step.view()` shows one step's source.

**Name products, not objects.** A product is a file that outlives the process
— a checkpoint, a figure, a dump — not a live Python object like `mesh` or
`stokes`. The dependency that matters runs over files:

```
figure  ←  checkpoint  ←  solve  ←  mesh
```

That is the chain that lets a reader open the notebook, load the checkpoint
and redraw a figure in seconds instead of re-solving. An example whose figures
can only be had by running the whole simulation is not launchable in one click,
whatever its decorators say. `WorkflowProducts` is the layer for this;
#760 covers the file product type and the staleness check it still needs.

### Helpers return standard UW3 objects

No wrapper types. A helper returns a mesh, a solver, a variable — the same
things the rest of Underworld3 takes — so a reader can step outside the
pattern at any point without unpicking it.

### The notebook opens with its own description

```
# Thermal Convection — Workflow Pattern Example

**PHYSICS:** convection
**DIFFICULTY:** intermediate
**RUNTIME:** ~2 minutes (50 steps at 1/16 resolution)
```

then a short Description, Key Concepts, and a Physical Setup table. Someone
arriving from a one-click launch reads that before anything runs, and it is
what tells them whether this is the example they wanted.

Reference implementation: `docs/examples/workflows/convection_config.py` and
`convection_notebook.py` (on `feature/fault-system-workflow`). They show the
semantics; neither yet shows products as files, which is #760.

## What to Avoid

- ❌ Excessive congratulation ("Great job!", "Excellent!")
- ❌ Over-explanation of obvious things
- ❌ Too many print statements for simple output
- ❌ Long chains of verification cells
- ❌ Debug/exploration cells left in production notebooks

## Examples from Notebook 12 and 13

These notebooks demonstrate the preferred style:
- Simple introduction
- Concept-focused sections
- Implicit display where appropriate
- Minimal but useful output
- Practical exercises at the end
