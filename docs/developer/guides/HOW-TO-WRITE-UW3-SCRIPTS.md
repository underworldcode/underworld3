# How to Write Underworld3 Scripts

**Last Updated**: 2025-11-15
**Status**: Living Document - Updated as new patterns emerge

This guide captures critical lessons learned from writing and debugging Underworld3 tests and scripts. It serves as a reference for common patterns, pitfalls, and best practices.

---

## Table of Contents

1. [Critical Ordering Rules](#critical-ordering-rules)
2. [Swarm Usage Patterns](#swarm-usage-patterns)
3. [Units System Integration](#units-system-integration)
4. [Mesh and Variable Creation](#mesh-and-variable-creation)
5. [Solver Setup and Execution](#solver-setup-and-execution)
6. [The Timestepping Pattern](#the-timestepping-pattern)
7. [Common Pitfalls and Anti-Patterns](#common-pitfalls-and-anti-patterns)
8. [Testing Best Practices](#testing-best-practices)
9. [Debugging Techniques](#debugging-techniques)

---

## Critical Ordering Rules

### ⚠️ RULE #1: Swarm Variables Before Population

**THE MOST COMMON ERROR**: Creating swarm variables AFTER populating the swarm.

```python
# ❌ WRONG - Will cause PETSc DM structure errors
swarm = uw.swarm.Swarm(mesh)
swarm.populate(fill_param=3)  # ERROR: No variables defined yet!
s_var = uw.swarm.SwarmVariable("scalar", swarm, 1)  # Too late!

# ✅ CORRECT - Variables must exist before population
swarm = uw.swarm.Swarm(mesh)
s_var = uw.swarm.SwarmVariable("scalar", swarm, 1)  # Create first
v_var = uw.swarm.SwarmVariable("vector", swarm, 2)  # All variables
swarm.populate(fill_param=3)  # Now populate
```

**Why this matters:**
- `populate()` and `add_particles_with_coordinates()` set up the PETSc DM structure
- The DM must know about all fields (variables) before particle allocation
- Creating variables after population causes DM inconsistencies and cryptic PETSc errors

**Error symptoms:**
- `petsc4py.PETSc.Error: error code 63`
- `PETSC ERROR: Argument out of range`
- Long comma-separated coordinate strings in error messages

**Applies to:**
- `swarm.populate()`
- `swarm.add_particles_with_coordinates()`
- Any operation that adds particles to the swarm

### RULE #2: Reference Quantities Before Mesh Creation

When using units, set reference quantities BEFORE creating meshes:

```python
# ✅ CORRECT - Reference quantities first
uw.reset_default_model()
model = uw.get_default_model()
model.set_reference_quantities(
    domain_depth=uw.quantity(500, "m"),
    material_density=uw.quantity(3300, "kg/m**3"),
)
mesh = uw.meshing.StructuredQuadBox(...)  # Mesh inherits reference quantities

# ❌ WRONG - Setting after mesh creation won't apply to existing mesh
mesh = uw.meshing.StructuredQuadBox(...)
model.set_reference_quantities(...)  # Too late for this mesh!
```

**Why this matters:**
- Reference quantities are immutable after mesh creation
- Coordinate units are established during mesh initialization
- Late setting won't retroactively apply to existing objects

### RULE #3: Model Reset Between Tests

```python
# ✅ CORRECT - Reset at start of each test
def test_something_with_units():
    uw.reset_default_model()  # Clean slate
    model = uw.get_default_model()
    model.set_reference_quantities(...)
    # ... rest of test

# ❌ WRONG - Reusing model from previous test
def test_something_with_units():
    model = uw.get_default_model()  # May have stale state!
    # ... rest of test
```

**Why this matters:**
- Tests can pollute each other's model state
- Strict units mode and reference quantities persist across tests
- Prevents mysterious failures from test ordering dependencies

---

## Swarm Usage Patterns

### Basic Swarm Creation

```python
# Complete swarm setup pattern
mesh = uw.meshing.StructuredQuadBox(elementRes=(8, 8))
swarm = uw.swarm.Swarm(mesh)

# Create ALL variables BEFORE populating
scalar_var = uw.swarm.SwarmVariable("material", swarm, 1)
vector_var = uw.swarm.SwarmVariable("velocity", swarm, 2)

# Create proxy variables if needed for integration/derivatives
proxy_var = uw.swarm.SwarmVariable("proxy", swarm, 1, proxy_degree=2)

# NOW populate the swarm
swarm.populate(fill_param=3)  # Uniform layout-based population

# OR use specific coordinates
coords = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
swarm.add_particles_with_coordinates(coords)
```

### Proxy Variables for Integration

When you need to integrate swarm data or compute derivatives:

```python
# Create proxy variable during initialization
s_var = uw.swarm.SwarmVariable("scalar", swarm, 1, proxy_degree=2)

# Populate swarm
swarm.populate(fill_param=3)

# Set swarm data
x_coords = swarm._particle_coordinates.data[:, 0]
s_var.data[:, 0] = 2.0 + x_coords

# Use symbolic representation for integration
I_f = uw.maths.Integral(mesh, fn=s_var.sym[0])  # Integrates proxy
result = I_f.evaluate()
```

**Key points:**
- Proxy variables create mesh-based RBF interpolations of swarm data
- Required for integrals (can't integrate point data directly)
- Required for derivatives (need continuous field representation)
- `proxy_degree` controls RBF interpolation quality

### Swarm Data Access Patterns

```python
# ✅ CORRECT - Direct array access
scalar_var.data[:, 0] = values  # Single variable, automatic sync

# ✅ CORRECT - Batch updates for multiple variables
with uw.synchronised_array_update():
    scalar_var.data[:, 0] = values1
    vector_var.data[:, 0] = values2
    vector_var.data[:, 1] = values3

# ❌ AVOID - Old access context pattern (legacy)
with swarm.access(scalar_var):
    scalar_var.data[:, 0] = values  # Unnecessary, use direct access
```

---

## Units System Integration

### Units Everywhere or Nowhere Principle

**Core Principle**: Either ALL quantities have units (when reference quantities set) OR all are plain numbers (when not set). No mixing.

```python
# Mode 1: Units Everywhere
uw.reset_default_model()
model = uw.get_default_model()
model.set_reference_quantities(
    domain_depth=uw.quantity(1000, "km"),
    plate_velocity=uw.quantity(5, "cm/year"),
    mantle_viscosity=uw.quantity(1e21, "Pa*s"),  # Provides [M] dimension
)

mesh = uw.meshing.StructuredQuadBox(...)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, units="kelvin")

poisson = uw.systems.Poisson(mesh, u_Field=T)
poisson.f = uw.quantity(2.0, "K")  # ✓ Units required
poisson.f = 2.0  # ✗ ERROR - plain number not allowed

# Mode 2: Plain Numbers Everywhere
uw.reset_default_model()
uw.use_nondimensional_scaling(False)

mesh = uw.meshing.StructuredQuadBox(...)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)  # No units

poisson = uw.systems.Poisson(mesh, u_Field=T)
poisson.f = 2.0  # ✓ Plain number allowed
```

### Understanding [M] Dimension Requirements

If you use quantities with mass dimension (pressure, stress, viscosity), you MUST provide [M]:

```python
# ✅ CORRECT - Complete dimensional specification
model.set_reference_quantities(
    domain_depth=uw.quantity(500, "m"),         # [L]
    material_density=uw.quantity(3300, "kg/m**3"),  # [M]
)

# Create pressure variable (has [M L⁻¹ T⁻²] dimensions)
p = uw.discretisation.MeshVariable("p", mesh, 1, units="pascal")  # Works!

# ❌ WRONG - Missing [M] dimension
model.set_reference_quantities(
    domain_depth=uw.quantity(500, "m"),  # Only [L], no [M]
)
p = uw.discretisation.MeshVariable("p", mesh, 1, units="pascal")  # ERROR at use-time!
```

**How to provide [M]:**
1. Explicit: `material_density=uw.quantity(3300, "kg/m**3")`
2. Via proxy: `mantle_viscosity=uw.quantity(1e21, "Pa*s")` (Pa contains kg)
3. Direct scale: `pressure_scale=uw.quantity(1e7, "Pa")` (no dimensional analysis)

### Checking Units Mode

```python
model = uw.get_default_model()

if model.has_units():
    # Units mode - all quantities need units
    value = uw.quantity(100, "m")
else:
    # Plain numbers mode
    value = 100
```

### Unit-Aware Variable Creation

```python
# With reference quantities set:
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, units="kelvin")
v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2, units="m/s")

# Coordinates automatically get units from reference quantities
x, y = mesh.X  # These have units from domain_depth
u = uw.get_units(x)  # Returns 'kilometer' if domain_depth in km

# Symbolic operations preserve units
grad_T = T.sym.diff(y)  # Has units: kelvin / kilometer
flux = grad_T * v.sym[1]  # Units: kelvin * m / (s * km)
```

---

## Mesh and Variable Creation

### Basic Mesh Creation

```python
# 2D structured mesh
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(16, 16),
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0),
)

# With units (requires reference quantities set first!)
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(16, 16),
    minCoords=(0.0, 0.0),
    maxCoords=(uw.quantity(1000, "km"), uw.quantity(500, "km")),
    units="kilometer",  # Explicit coordinate units
)

# 3D structured mesh
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(8, 8, 8),
    minCoords=(0.0, 0.0, 0.0),
    maxCoords=(1.0, 1.0, 1.0),
)
```

### Variable Creation Best Practices

```python
# Scalar variable
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

# Vector variable (2D)
v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)

# Vector variable (3D)
v = uw.discretisation.MeshVariable("v", mesh, 3, degree=2)

# Pressure (typically degree=1 for Stokes)
p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)

# With units
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, units="kelvin")
v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2, units="m/s")

# Private variables (not saved/loaded with model)
temp_var = uw.discretisation.MeshVariable("_temp", mesh, 1, _register=False)
```

### Variable Naming Convention

```python
# ✅ GOOD - Clear, descriptive names
temperature = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
velocity = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
pressure = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)

# ⚠️ AVOID - Ambiguous 'model' variable name
# 'model' is ambiguous: uw.Model vs constitutive models
# Use specific names:
constitutive_model = stokes.constitutive_model  # ✓
viscous_model = poisson.constitutive_model  # ✓
model = stokes.constitutive_model  # ✗ Ambiguous
```

### Direct Data Access (Modern Pattern)

```python
# ✅ MODERN - Direct array access
var.array[..., 0] = values  # Single variable

# ✅ MODERN - Batch updates
with uw.synchronised_array_update():
    var1.array[..., 0] = values1
    var2.array[..., 0] = values2

# ❌ LEGACY - Old access context (still works but not needed)
with mesh.access(var):
    var.data[...] = values
```

---

## Solver Setup and Execution

### Poisson Solver

```python
# Basic setup
mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 16))
u = uw.discretisation.MeshVariable("u", mesh, 1, degree=2)

poisson = uw.systems.Poisson(mesh, u_Field=u)
poisson.constitutive_model = uw.constitutive_models.DiffusionModel
poisson.constitutive_model.Parameters.diffusivity = 1.0

# Source term (respects units mode!)
if model.has_units():
    poisson.f = uw.quantity(1.0, "appropriate_units")
else:
    poisson.f = 1.0

# Boundary conditions
poisson.add_dirichlet_bc(T_bottom, "Bottom")
poisson.add_dirichlet_bc(T_top, "Top")

# Solve
poisson.solve()
```

### Stokes Solver

```python
# Create variables BEFORE solver
mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 16))
v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)

# Setup solver
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0

# Boundary conditions
stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
stokes.add_dirichlet_bc((0.0, 0.0), "Top")

# Solve
stokes.solve()

# Access results
velocity_data = v.array
pressure_data = p.array
```

### Constitutive Model Access

```python
# ✅ CORRECT - Use 'constitutive_model' variable name
constitutive_model = stokes.constitutive_model
constitutive_model.Parameters.viscosity = 1e21

# ✅ CORRECT - Inline access is fine too
stokes.constitutive_model.Parameters.viscosity = 1e21

# ⚠️ AVOID - Ambiguous 'model' variable name
model = stokes.constitutive_model  # Confusing with uw.Model
```

---

## The Timestepping Pattern

### ⚠️ RULE: the clock lives on `model.tracker`, never in a loose variable

Every time-dependent script needs a clock. Declare the model and its reference
quantities first (see [RULE #2](#rule-2-reference-quantities-before-mesh-creation)
— they must precede mesh creation), then keep the clock on the tracker:

```python
uw.reset_default_model()
model = uw.get_default_model()
model.set_reference_quantities(
    domain_depth=uw.quantity(500, "km"),
    material_density=uw.quantity(3300, "kg/m**3"),
    material_viscosity=uw.quantity(1e21, "Pa*s"),
)

mesh = uw.meshing.UnstructuredSimplexBox(...)      # inherits the reference quantities
# ... variables, solvers ...

model.tracker.time = uw.quantity(0.0, "Myr")
model.tracker.step = 0
model.tracker.dt = None

while model.tracker.time < end_time:
    dt = adv_diff.estimate_dt()        # a dimensional quantity when units are active

    adv_diff.solve(timestep=dt)
    stokes.solve(zero_init_guess=False)

    model.tracker.time = model.tracker.time + dt
    model.tracker.step += 1
    model.tracker.dt = dt
```

**Start from the model, not from the mesh.** A default model is created for you
if you never ask for one, so it is easy to write a whole script without noticing
it exists — and then reference quantities, which must be set before the mesh, are
already too late. Declaring the model on the first line makes the units decision
explicit at the only point where it can still be made. `estimate_dt()` then
returns a dimensional quantity and the clock carries units with no extra work.

Neither the model nor the units are enforced today; both arrived after much of
the surrounding code. Treat this ordering as the pattern regardless, because
retrofitting units to a script written without them means rebuilding the mesh.

**Why this and not `t = 0.0; t += dt`.** The tracker is captured by
`model.save_state()` and reverted by `model.load_state()`. A loose Python
variable is not — the tracker's own docstring says so:

> Everything on the tracker is captured by `snapshot` and reverted by
> `restore`; loose Python variables are not.

So a script that keeps its clock in a local and then backsteps gets its
**fields** restored and its **time** left in the future. Nothing raises; the
run simply carries a clock that disagrees with the state it is describing, and
every output written from that point is mislabelled. Measured on a real
five-step run: after restoring a snapshot taken at t = 1140, a loose `t` still
read 2280 while the fields were correctly back at 1140.

This holds for both snapshot flavours — the in-memory token and the on-disk
snapshot used for restart.

`time`, `step` and `dt` are pre-seeded on the tracker as a convention. Anything
else you assign to it (`model.tracker.rms_velocity = ...`) is captured and
restored the same way, so a diagnostic you want to survive a backstep belongs
there too.

### Wrap the step

`model.step(dt)` makes the loop a transaction:

```python
while model.tracker.time < end_time:
    dt = adv_diff.estimate_dt()

    with model.step(dt):
        adv_diff.solve(timestep=dt)
        stokes.solve(zero_init_guess=False)
```

Three things follow, and none of them requires anything else in the script to
change. The clock reads the END of the interval for the whole block, which is
where an implicit scheme centres its residual, so a time-dependent coefficient
is evaluated at the right time. The advance commits only on clean exit, so a
step that raises — or one abandoned because the Courant number came out too
large — leaves the clock exactly as it was. And everything the block did is
recorded:

```python
>>> for entry in model.journal[-3:]:
...     print(entry)
<step 0 'convect' dt=0.01 solve:SNES_AdvectionDiffusion(T) -> solve:SNES_Stokes(V)>
<step 1 'convect' dt=0.01 solve:SNES_AdvectionDiffusion(T) -> solve:SNES_Stokes(V)>
<step 2 'convect' dt=0.01 solve:SNES_AdvectionDiffusion(T) -> solve:SNES_Stokes(V)>
```

That record is worth having on its own. It answers what a run actually did,
in order, without the script being instrumented for it — which is the question
you want to ask of someone else's model, or your own six months later.

Opening a step is optional. A script that never does behaves exactly as before.

### Recording a run

Ask the step to keep the state it started from and the journal becomes a
restorable record:

```python
model.record_every = 1      # keep every step; None (default) keeps none
model.record_limit = 8      # how many snapshots to retain

while model.tracker.time < end_time:
    with model.step(dt):
        adv_diff.solve(timestep=dt)
        stokes.solve(zero_init_guess=False)

model.rewind()              # undo the last step: fields, history and clock
```

The snapshot is taken before the operators run, which is the only correct
point — a `DDt` shifts its history in its post-solve hook, so a snapshot taken
afterwards holds the shifted history rather than the step's input.

Two things this buys beyond backstepping. Replaying a step from its own
snapshot reproduces it exactly, where re-running the script does not, so a step
that misbehaved can be looked at twice. And an adjoint needs precisely this: the
state at each step and the order the operators were applied in.

Snapshots cost roughly 13 bytes per primary degree of freedom per step. Older
steps lose their snapshot and keep their journal record, so the account of what
happened outlives the state it happened to.

A driver that runs the same model more than once — an inversion, a parameter
sweep, a restart — should start each run with a clean account:

```python
model.clear_journal()
model.tracker.time = uw.quantity(0.0, "Myr")
model.tracker.step = 0
```

Without it the journal is the concatenation of every run the process has done,
and `rewind()` will walk back into the previous one.

On a mesh that deforms or adapts the snapshot cannot be taken yet; the run
warns once, keeps journalling, and `rewind()` will not reach those steps.

### Writing the record down

`model.journal` is what the run can still undo. It lives in memory, it is
bounded, and it dies with the process. `model.journal_file` is what the run
*did*:

```python
model.journal_file = "output/run.log"
```

One aligned line per step, appended and flushed as it closes, so `tail -f`
follows a running job:

```
# underworld3 step log · model 'default' · started 2026-09-10T21:22:40+00:00
# scales: length 2.2e+06 m | time 4.84e+18 s | mass 1.065e+47 kg | temperature 2500 K
# step           t/Myr          dt/Myr    wall/s  outcome    operators, in order
      0        0.175907        0.175907      0.44  ok         [convect] solve:SNES_AdvectionDiffusion_Composed(T) > history_shift:EulerianSUPG(T) > solve:SNES_Stokes(v)
      1        0.501546        0.325639      0.09  ok         [convect] solve:SNES_AdvectionDiffusion_Composed(T) > history_shift:EulerianSUPG(T) > solve:SNES_Stokes(v)
      2        0.990939        0.489393      0.09  ok         [convect] solve:SNES_AdvectionDiffusion_Composed(T) > history_shift:EulerianSUPG(T) > solve:SNES_Stokes(v)
      3         1.51459        0.523655      0.09  ok         [convect] solve:SNES_AdvectionDiffusion_Composed(T) > history_shift:EulerianSUPG(T) > solve:SNES_Stokes(v)
      4         30.5663         29.0517      0.09  ABANDONED  [too big] solve:SNES_AdvectionDiffusion_Composed(T) > history_shift:EulerianSUPG(T) > solve:SNES_Stokes(v)
  -- restore from a snapshot; the clock now reads 1.51459 Myr
  -- rewind to the start of step 3 (t = 0.990939 Myr); 1 step(s) undone
      3         1.51459        0.523655      0.42  ok         [replay] solve:SNES_AdvectionDiffusion_Composed(T) > history_shift:EulerianSUPG(T) > solve:SNES_Stokes(v)
```

`wall/s` is how long the block took. It is not physics, but it is the number
you want when watching: a step that suddenly takes ten times as long is the
first sign of a solver in trouble.

**A true log records the backtracks.** `rewind()` and a bare `load_state()`
each write their own line, because a log that shows step 3, then step 3 again
with nothing in between, is not a log of what happened. `rewind` writes the
more specific note and suppresses the generic one.

Four other things go in the file that are not in `model.journal`, all
deliberate:

- **An abandoned step.** A rejected step is the part of a run's history that is
  otherwise invisible, and it is usually what you want when asking why a run
  went the way it did.
- **A step aged out by `journal_limit`.** The account of what happened outlives
  both the state and the bounded in-memory list.
- **An invariant complaint**, as an `invariant` event on the step, so it
  survives the terminal the run happened to have.
- **Everything up to a kill.** The file is flushed per step.

### For parsing: JSON lines

A path ending `.jsonl`, `.ndjson` or `.json` — or `model.journal_format =
"jsonl"` — writes the same record as one JSON object per line:

```json
{"kind": "run",  "model": "default", "started": "2026-09-10T21:22:40+00:00", "scales": {"length": {"magnitude": 2200000.0, "units": "meter"}, ...}}
{"kind": "step", "index": 0, "label": "convect",
 "t0": {"magnitude": 0.0, "units": "megayear"},
 "t1": {"magnitude": 0.1759, "units": "megayear"},
 "dt": {"magnitude": 5551210433127.7, "units": "second"},
 "completed": true, "restorable": true, "wall": 0.44,
 "events": [{"kind": "solve", "name": "SNES_AdvectionDiffusion_Composed(T)"},
            {"kind": "history_shift", "name": "EulerianSUPG(T)", "dt": 1.1469e-06},
            {"kind": "solve", "name": "SNES_Stokes(v)"}]}
{"kind": "rewind", "message": "...", "to_step": 3, "steps_undone": 1, "t": {"magnitude": 0.9909, "units": "megayear"}}
```

Read it back with `uw.read_journal(path)`, which returns one entry per run — an
inversion driver that ran the forward model thirteen times leaves thirteen runs
in one file, delimited by the header `clear_journal()` writes.

**Why JSON lines and not YAML.** One self-contained record per line is the
whole point. A killed run leaves a truncated final line that *fails* to parse,
so `read_journal` drops it and keeps everything before; a half-written YAML
mapping frequently still parses, as a real record with its last key missing.
Line-oriented also means `grep`, `wc -l` and `jq -c` work without a parser, and
`json` is stdlib with predictable float round-tripping. YAML is the right
format for a whole document written once and edited by hand — which is what
`Model.to_yaml` uses it for — but a log is a stream.

The two formats differ in one more way. The text log is a **report**: the time
column is converted into one unit, named in the header. The JSON log is a
**record**: every value keeps the units the run actually held it in, which is
why `t0` may read in Myr beside a `dt` in seconds — the clock came from the
tracker and the interval from `estimate_dt()`.

Rank 0 writes; the other ranks record in memory as usual.

### The same account, as a figure

A terminal is not where a run belongs in a paper.

```python
uw.journal_diagram(model, out="figures/run.pdf")     # or a .jsonl log
uw.journal_diagram(model, out="figures/run.svg")     # same figure, SVG
uw.journal_flowchart(model)                          # Mermaid, for docs
```

`journal_diagram` puts **time down the page**: one row per step, A4 portrait,
paginated, so it drops into a document column and opens anywhere. Each row
carries the step index, the clock, `dt` as a number and as a bar, a wall-clock
tick, and one letter. Backtracks are drawn in the left gutter as an arrow from
the step that ended back up to the step it returned to.

The PDF and the SVG are both written directly — no plotting library, no
rasterisation, nothing fetched at render time, and a print-safe palette that
separates in greyscale. The `dt` axis goes logarithmic when the range exceeds
20x and says so: a rejected step is often tens of times the accepted ones,
which is *why* it was rejected, and on a linear axis it flattens everything
else to nothing.

**The letter is the layout.** Each distinct operator sequence gets one, defined
once at the foot of the figure:

```
step   t/Myr   dt/Myr   seq   dt
   9   6.472    1.176    A    ▇▇▇▇▇▇▇▇▇▇▇▇
  10   7.936    1.464    A    ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
  11   9.856    1.920    A    ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
  12     145    135.2    A    ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄  abandoned
  11   9.856    1.920    A    ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
  12   12.56    2.704    B !  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇

A   AdvectionDiffusion(T)  >  shift EulerianSUPG(T)  >  Stokes(v)        14 steps
B   AdvectionDiffusion(T)  >  shift EulerianSUPG(T)  >  Stokes(v)  >
    AdvectionDiffusion(T)  >  shift EulerianSUPG(T)  >  Stokes(v)         1 step
```

A column of `A` with a single `B` in it says at a glance that one step did
something different. A hundred spelled-out sequences say nothing and hide the
one that matters.

`journal_flowchart` renders one step's operator flow as Mermaid. When a run has
more than one distinct sequence, each becomes its own subgraph labelled with
the steps that took it, so an anomalous step is visible rather than averaged
away.

Both accept a live model, a `.jsonl` log, or the list `read_journal` returns.
Not a text log: that one is a report, and reading it back is refused with the
one line that fixes it.

### What the record checks

A step also checks that it can be what it claims to be. One invariant so far:
a history manager must advance exactly once per step.

```
<step 0 'convect' dt=0.01 solve:SNES_AdvectionDiffusion(T) -> history_shift:EulerianSUPG(T) -> solve:SNES_Stokes(V)>
```

Call a solver twice inside one step — a corrector, a Picard iteration on a
coupled system, a retry — and its history advances twice, so the physical step
is taken twice. The solve counter and the timestep history look identical to a
single step, so nothing else in the library can see it. The step warns.

If a solver genuinely is called more than once within a step, only the last
call should carry the timestep.

### Backstepping

The pattern above is what makes speculative stepping safe:

```python
snap = model.save_state()          # before the step, not after

try:
    with model.step(big_dt):
        adv_diff.solve(timestep=big_dt)
        stokes.solve(zero_init_guess=False)
        if courant_number() > courant_limit:
            raise StepRejected            # abandons the step; the clock stays put
except StepRejected:
    model.load_state(snap)                # fields go back; the clock never moved
    for sub_dt in substeps(big_dt):
        with model.step(sub_dt):
            ...
```

Take the snapshot **before** the operator, not after. A `DDt` history plugin
shifts its history in its post-solve hook, so a snapshot taken after a solve
holds the shifted history, which is not the state that step ran from.

Restoring a snapshot is bit-exact and repeatable. Re-running the same script is
not: warm starts and preconditioner reuse are solver history that is not part
of model state, so two independent runs of the same problem on the same solver
objects diverge at the 1e-13 level from the first step. If you need to look at a
step twice, restore it rather than re-run it.

### Two worked cases

**`docs/examples/convection/intermediate/Ex_Convection_Annulus_Recorded.py`** —
Boussinesq convection in an annulus. Four reference quantities, a body force
written as a force (Ra falls out of the nondimensionalisation rather than being
typed in), rotated free-slip on the curved boundaries, and a varying
`estimate_dt()`. It then demonstrates the four things the record buys, in
order: the journal, a rejected step, a bit-exact replay, and the invariant
catching a step that was taken twice. Compare
`../advanced/Ex_Convection_Cylinder.py`, which solves the same physics with a
bare `for step in range(n)` loop and no clock at all.

**An adjoint driven from the journal.** The backward pass of a discrete adjoint
needs exactly what the record holds: the state at each step and the order the
operators were applied in. Walking `model.journal` backwards —
`load_state(entry.snapshot)`, replay, transpose-solve — replaces the
hand-written checkpoint dictionary that an adjoint normally carries, and
removes its dependence on knowing in advance which arrays the backward pass
will want.

### Time-dependent expressions

`mesh.t` is the model clock as a symbol. It is repacked from
`model.tracker.time` before every solve, so a time-dependent source or
boundary condition follows the loop above with no recompilation per step:

```python
omega = 2 * sympy.pi / period
stokes.add_dirichlet_bc((V0 * sympy.sin(omega * mesh.t), 0.0), "Top")
```

Two things to know. A script that never advances `model.tracker.time` leaves
`mesh.t` at zero, so the clock and the pattern above are the same subject. And
`mesh.t` should appear inside an expression rather than be handed bare to a
scalar setter — `poisson.f = mesh.t` stores a value, `poisson.f = 1.0 * mesh.t`
keeps the symbol.

---

## Common Pitfalls and Anti-Patterns

### ❌ Rebinding the name instead of setting `.sym`

To change the value of an expression, set `.sym`. It is the only settable
property — `.value` and `.data` are derived, read-only views.

```python
# ✅ CORRECT - a value change; the container keeps its identity
viscosity.sym = sympy.Integer(0)
solver._update_constants()      # only if you are not about to solve

# ❌ WRONG - rebinds a Python name and changes nothing
viscosity = 0
```

The second line leaves every expression that already references the atom
pointing at the old object with its old value, and nothing complains. The
identity is the point: because the container is unchanged, a ramped value
reaches every residual that mentions it with no rebuild.

`expr.copy(other)` does the same job from another expression, and assigning to
a constitutive parameter slot (`Parameters.diffusivity = 0.0`) is also a value
change rather than a replacement.

### ❌ Swarm Variable Creation After Population

```python
# ❌ WRONG - Most common error!
swarm = uw.swarm.Swarm(mesh)
swarm.populate(fill_param=3)
var = uw.swarm.SwarmVariable("s", swarm, 1)  # ERROR!

# ✅ CORRECT
swarm = uw.swarm.Swarm(mesh)
var = uw.swarm.SwarmVariable("s", swarm, 1)
swarm.populate(fill_param=3)
```

### ❌ Mixing Units and Plain Numbers

```python
# ❌ WRONG - Inconsistent unit usage
model.set_reference_quantities(domain_depth=uw.quantity(1000, "km"))
poisson.f = 2.0  # ERROR: Units required when reference quantities set

# ✅ CORRECT - Consistent approach
model.set_reference_quantities(domain_depth=uw.quantity(1000, "km"))
poisson.f = uw.quantity(2.0, "kelvin")
```

### ❌ Missing Reference Quantities for [M] Dimension

```python
# ❌ WRONG - Incomplete dimensional specification
model.set_reference_quantities(domain_depth=uw.quantity(500, "m"))
p = uw.discretisation.MeshVariable("p", mesh, 1, units="pascal")  # Needs [M]!

# ✅ CORRECT - Complete specification
model.set_reference_quantities(
    domain_depth=uw.quantity(500, "m"),
    material_density=uw.quantity(3300, "kg/m**3"),
)
p = uw.discretisation.MeshVariable("p", mesh, 1, units="pascal")
```

### ❌ Batman Pattern (Anti-Pattern - DO NOT USE)

**The Batman Pattern**: Declaring all variables upfront before any computational work, "just in case."

```python
# ❌ BATMAN PATTERN - DO NOT DO THIS
mesh = uw.meshing.StructuredQuadBox(...)
T = uw.discretisation.MeshVariable('T', mesh, 1, degree=2)
gradT = uw.discretisation.MeshVariable('gradT', mesh, 1, degree=1)  # Don't need yet!
flux = uw.discretisation.MeshVariable('flux', mesh, mesh.dim, degree=1)  # Don't need yet!
# ... 20 more variables you MIGHT need later

# Solve the actual problem (far from declarations)
poisson = uw.systems.Poisson(mesh, u_Field=T)
poisson.solve()

# Use pre-declared variables
proj = uw.systems.Projection(mesh, gradT, ...)
proj.solve()
```

**Why this is wrong:**
- Unnatural workflow - requires predicting all future needs
- Poor software design - violates principle of locality
- Breaks exploratory analysis - can't create variables on demand
- **WAS** required due to old DM state corruption bug
- **NOW FIXED** (2025-10-14) - variables can be created anytime

```python
# ✅ CORRECT - Natural workflow
mesh = uw.meshing.StructuredQuadBox(...)
T = uw.discretisation.MeshVariable('T', mesh, 1, degree=2)

# Solve
poisson = uw.systems.Poisson(mesh, u_Field=T)
poisson.solve()

# Create gradient variable WHEN YOU NEED IT
gradT = uw.discretisation.MeshVariable('gradT', mesh, 1, degree=1)
proj = uw.systems.Projection(mesh, gradT, ...)
proj.solve()
```

**See**: `CLAUDE.md` "NO BATMAN" section for full history

### ❌ Not Resetting Model Between Tests

```python
# ❌ WRONG - Tests can interfere
def test_with_units():
    model = uw.get_default_model()  # May have state from previous test!
    # ...

# ✅ CORRECT - Clean slate each test
def test_with_units():
    uw.reset_default_model()
    model = uw.get_default_model()
    # ...
```

---

## Testing Best Practices

### Test Structure Template

```python
import pytest
import underworld3 as uw
import numpy as np

@pytest.mark.level_2  # Complexity: 1=quick, 2=intermediate, 3=physics
@pytest.mark.tier_a   # Reliability: a=production, b=validated, c=experimental
def test_descriptive_name():
    """
    Clear docstring explaining what is being tested and why.

    Include any important context about expected behavior.
    """
    # ALWAYS reset model at start of test
    uw.reset_default_model()

    # Setup with units if needed
    model = uw.get_default_model()
    model.set_reference_quantities(
        domain_depth=uw.quantity(500, "m"),
        material_density=uw.quantity(3300, "kg/m**3"),
    )

    # Create mesh
    mesh = uw.meshing.StructuredQuadBox(elementRes=(8, 8))

    # Create variables
    var = uw.discretisation.MeshVariable("var", mesh, 1, degree=2, units="kelvin")

    # Test operations
    # ...

    # Assertions with clear failure messages
    assert result == expected, f"Expected {expected}, got {result}"
```

### Test Markers Usage

```python
# Level markers (complexity/runtime)
@pytest.mark.level_1  # Quick - imports, setup, no solving (~seconds)
@pytest.mark.level_2  # Intermediate - integration, simple solves (~minutes)
@pytest.mark.level_3  # Physics - benchmarks, complex solves (~minutes to hours)

# Tier markers (reliability/trust)
@pytest.mark.tier_a   # Production-ready - trusted for TDD, CI
@pytest.mark.tier_b   # Validated - use with caution, needs more testing
@pytest.mark.tier_c   # Experimental - development only, not for automation

# Expected failures
@pytest.mark.xfail(reason="Clear explanation of why this fails")

# Multiple markers
@pytest.mark.level_2
@pytest.mark.tier_b
@pytest.mark.xfail(reason="Unit-aware derivative bug: UnitAwareDerivativeMatrix * NegativeOne")
def test_something():
    pass
```

### Testing Swarms (Critical!)

```python
def test_swarm_functionality():
    uw.reset_default_model()

    mesh = uw.meshing.StructuredQuadBox(elementRes=(8, 8))
    swarm = uw.swarm.Swarm(mesh)

    # ⚠️ CRITICAL: Create ALL variables BEFORE populating!
    scalar_var = uw.swarm.SwarmVariable("scalar", swarm, 1)
    vector_var = uw.swarm.SwarmVariable("vector", swarm, 2)

    # NOW populate
    swarm.populate(fill_param=3)

    # Set data and test
    scalar_var.data[:, 0] = 1.0
    assert scalar_var.array.mean() == pytest.approx(1.0)
```

---

## Debugging Techniques

### JIT Compilation Issues

**Slow time-stepping loops**: If each `solver.solve()` call takes 10+ seconds
in a time-stepping loop, the solver is probably recompiling the JIT extension
every step. Use `UWexpression` objects for any parameter that changes between
steps:

```python
# FAST — expression parameter, no recompilation on change
dt_e = uw.expression("dt_e", 0.01)
model.Parameters.dt_elastic = dt_e

for step in range(100):
    dt_e.sym = compute_timestep()  # Updates constants[], ~0ms
    solver.solve()                  # No JIT rebuild
```

**Symbolic names in generated C code**: If you see LaTeX-like names in the
generated C code (e.g., `\eta` instead of a number or `constants[i]`):

```text
// ERROR symptom — expression not unwrapped:
out[0] = 1.0/{ \eta \hspace{ 0.0006pt } };
```

**Cause**: A `UWexpression` was not properly detected as constant or unwrapped.

**Solution**: Check that the expression resolves to a pure number when fully
unwrapped. Composite expressions containing mesh variables or coordinates
cannot be routed through `constants[]`.

### PETSc DM Errors with Swarms

Error pattern:
```
[0]PETSC ERROR: Argument out of range
[0]PETSC ERROR: Input string 0.5488135039273248,0.7151893663724195,...
```

**Cause**: Swarm variables created AFTER `populate()` or `add_particles_with_coordinates()`.

**Solution**: Create ALL swarm variables BEFORE adding particles.

### Units Dimension Errors

Error pattern:
```
ValueError: Cannot create variable with units when model has no reference quantities
```

**Cause**: Missing reference quantities, or trying to use [M] dimension without providing [M].

**Solution**:
1. Set reference quantities before mesh creation
2. Ensure [M] dimension available if using pressure/stress/viscosity units

### Test State Pollution

Symptom: Test passes in isolation but fails in suite, or vice versa.

**Cause**: Model state persisting between tests.

**Solution**: ALWAYS call `uw.reset_default_model()` at start of each test.

### Derivative Units Bugs

Error pattern:
```
TypeError: unsupported operand type(s) for *: 'UnitAwareDerivativeMatrix' and 'NegativeOne'
```

**Cause**: Unit-aware derivative arithmetic not fully implemented.

**Status**: Known bug as of 2025-11-15, mark tests with xfail.

---

## Quick Reference Checklist

### Starting a New Script

- [ ] Import underworld3: `import underworld3 as uw`
- [ ] Decide units mode: With reference quantities or plain numbers?
- [ ] If using units: Set reference quantities BEFORE mesh creation
- [ ] If using units with [M]: Provide material_density or equivalent

### Writing a Timestepping Loop

- [ ] Declare the model and its reference quantities BEFORE creating the mesh
- [ ] Keep `time`, `step` and `dt` on `model.tracker`, not in local variables
- [ ] Wrap each step in `with model.step(dt):`
- [ ] To change an expression's value set `.sym`, never rebind the name
- [ ] Take snapshots BEFORE the operator you might want to undo
- [ ] Use `mesh.t` inside an expression for time dependence, never bare

### Creating a Swarm

- [ ] Create mesh first
- [ ] Create swarm: `swarm = uw.swarm.Swarm(mesh)`
- [ ] Create ALL swarm variables (regular + proxy)
- [ ] THEN populate: `swarm.populate()` or `add_particles_with_coordinates()`
- [ ] Set swarm data
- [ ] Use `.sym` for symbolic operations (integration, derivatives)

### Writing a Test

- [ ] Add `@pytest.mark.level_N` marker (1, 2, or 3)
- [ ] Add `@pytest.mark.tier_X` marker (a, b, or c)
- [ ] Start with `uw.reset_default_model()`
- [ ] Set reference quantities if using units
- [ ] Follow correct ordering (mesh → variables → populate for swarms)
- [ ] Clear docstring explaining what's being tested
- [ ] Meaningful assertion messages

### Setting Up Units

- [ ] Call `uw.reset_default_model()` for clean state
- [ ] Get model: `model = uw.get_default_model()`
- [ ] Set reference quantities: `model.set_reference_quantities(...)`
- [ ] Include [M] dimension if needed (density, viscosity, or pressure_scale)
- [ ] Create mesh (coordinates inherit units)
- [ ] Create variables with `units=` parameter
- [ ] Use `uw.quantity()` for all numerical values in solver parameters

---

## Version History

- **2026-09-09**: The timestepping pattern
  - Start from the model and its reference quantities, not from the mesh
  - Clock on `model.tracker`, not loose variables (snapshot consistency)
  - Disk snapshots now carry dimensional values (magnitude + units)
  - `mesh.t` now resolves to the model clock (#410)
  - `model.step(dt)` — the step as a transaction, and the step journal
  - `model.record_every` / `model.rewind()` — the journal as a restorable record
  - A step warns when a history advances more than once
  - Set `.sym` to change a value; rebinding the name changes nothing
  - Backstepping recipe; snapshot before the operator
  - `mesh.t` is not the model clock and is silently zero in a solve
- **2025-11-15**: Initial version
  - Swarm ordering rules from test_0850/0851 debugging
  - Units everywhere-or-nowhere principle
  - [M] dimension requirements
  - Batman pattern documentation
  - Test classification system integration
  - JIT unwrapping debugging patterns

---

## See Also

- `CLAUDE.md`: Project status, coding conventions, "NO BATMAN" section
- `docs/developer/TESTING-RELIABILITY-SYSTEM.md`: Test tier classification
- `docs/developer/COORDINATE-UNITS-TECHNICAL-NOTE.md`: Coordinate units implementation
- `docs/beginner/tutorials/12-Units_System.ipynb`: Units system tutorial
- `docs/beginner/tutorials/13-Non_Dimensional_Scaling.ipynb`: Dimensional analysis
- `docs/advanced/snapshot-restore.md`: Snapshot and restore semantics
- `tests/test_0009_model_tracker.py`: The pattern, enforced
