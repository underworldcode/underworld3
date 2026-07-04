---
title: "What is a uw.workflow?"
---

# What is a `uw.workflow`?

A **workflow** in Underworld3 is a structured way to organise a
simulation — and everything that surrounds it — as a graph of
named, cached computations.  This page explains the mental model.
The [API reference](../../api/workflows.md) documents the symbols;
the [convection example user guide](../../examples/workflows/convection-user.md)
walks through one workflow end to end.

## The problem a workflow solves

Geodynamic simulations rarely live in isolation.  A typical project
ends up with:

- A **simulation** that runs for hours or days and produces large
  HDF5 checkpoint chains.
- **Diagnostics** computed during the run (Nu, Vrms, viscosity profiles, ...).
- **Aggregations** across multiple runs (Nu vs Ra plots, parameter sweeps).
- **Visualisations** rendered after-the-fact (frame stacks, movies).
- **Restarts**, **warm-starts**, **parameter ramps** that re-use prior work.

Without structure, all of this becomes ad-hoc: scripts that orchestrate
each other, files in semi-conventional locations, no record of what
config produced what output.  Re-running the analysis often means
re-running the simulation because the script doesn't know what's
already been done.

A workflow encodes the structure once.  Then "give me the
Nu-vs-Ra plot" is a single function call: the framework figures out
which simulations are still needed, runs only those, and assembles
the plot.

## The mental model: a DAG of products

A workflow is a directed acyclic graph (DAG) where:

- **Nodes** are *products* — named, persistable outputs.  Examples:
  `mesh`, `run_directory`, `nu_vs_ra_csv`, `temperature_movie`.
- **Edges** encode dependencies.  An edge from `mesh` to
  `run_directory` says "to build the run directory, you first need
  the mesh".

Each product is built by exactly one **step** — a Python function
decorated with `@workflow_step`, declaring what it `produces` and
what it `requires`.  Together the steps form the DAG.

```python
from underworld3.workflows import workflow_step

@workflow_step(produces=["mesh"])
def create_mesh(config):
    return uw.meshing.UnstructuredSimplexBox(...)

@workflow_step(produces=["run_directory"], requires=["mesh"])
def evolve(mesh, config):
    ...
    return run

@workflow_step(produces=["run_summary"], requires=["run_directory"])
def summarise_run(run_directory, config):
    ...
    return summary
```

When you ask the runner to `build("run_summary")`, it walks the
DAG, runs whatever steps are needed (and only those), and caches
the results.  Re-asking for the same product on the next session
hits the cache.

## The vocabulary

A small set of words you'll see across every workflow.  The same
glossary appears at the bottom of each per-workflow user guide so
non-technical readers don't need to come here first.

### Workflow

A Python module of `@workflow_step`-decorated functions plus a
Pydantic config class.  Every workflow exports the same shape, so
once you've learned the mechanics of one (see the convection
example), you can read any other.

### Step

A single function decorated with `@workflow_step`, declaring what
it `produces` (one or more named outputs) and `requires` (named
outputs of upstream steps).

### Product

A named, cached output of a step.  The runner persists products
through `WorkflowProducts` to a `<output_dir>/products/` directory
with type-aware serialisation:

| Object type | On-disk form |
|-------------|--------------|
| `Mesh` | HDF5 + XDMF |
| `MeshVariable` | HDF5 + XDMF |
| `Surface` / `SurfaceCollection` | VTK |
| `ndarray` | NPZ |
| `Run` (run-directory) | recorded path; the directory itself is the artefact |
| `Path` (file artefact) | recorded path; the producer wrote the file |
| anything else | YAML fallback |

### Cache key

A short hex digest derived from a product's inputs (the relevant
config fields plus upstream products' cache keys).  Two products
with the same cache key are equivalent under deterministic
producers; mismatched cache keys mean "the inputs changed, the
cached version is stale, rebuild".  Stored next to each product in
the manifest.

### Identity fields

The subset of a config whose change should *invalidate* cached
products.  Mesh and physics fields are identity; step caps and
tolerances aren't.  Declared on each `WorkflowConfig` subclass:

```python
class MyConfig(WorkflowConfig):
    _identity_fields = ("mesh_resolution", "rayleigh", "viscosity")
    ...
```

If `_identity_fields` is `None` (the default), all products fall
back to existence-based caching — the legacy pre-cache_key
behaviour, preserved for backward compatibility.

### Freshness

The property of a cached product whose cache key still matches
what the current config expects.  When the runner is asked for a
product, it walks the DAG; products that are fresh hit the cache,
products that are stale rebuild.

### Recipe

An *example script* that composes the workflow's primitives —
e.g. `warm_start`, `ramp_rayleigh`.  Recipes live alongside the
workflow code; they are not part of the public API.  The
discipline ([Run/Manifest/etc.](../../api/workflows.md) is API,
recipes are example code) keeps the public surface small while
letting per-workflow patterns live where they belong.

### Run directory

The on-disk folder for one model run: manifest, h5 chain,
timeseries.csv, summary.  Wrapped by the `Run` class.  For
time-loop workflows, the run directory is itself a product (a
`run_directory` typed entry in the products manifest).

## How the pieces fit together

```
┌──────────────────────┐
│ WorkflowConfig       │  Parameters with types, bounds, identity
│ (Pydantic)           │  hashable via .cache_key()
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Workflow module      │  @workflow_step functions declaring
│ (your code)          │  produces/requires
└──────────┬───────────┘
           │ inspected by
           ▼
┌──────────────────────┐    ┌──────────────────────┐
│ WorkflowRunner       │◄──►│ WorkflowProducts     │
│ runner.build(...)    │    │ <output_dir>/products│
│ runner.dag()         │    │ /manifest.yaml       │
└──────────────────────┘    └──────────────────────┘
```

- **`WorkflowConfig`** is your parameter object.  Subclass it,
  declare `_identity_fields` (which fields invalidate caches),
  populate with Pydantic `Field`s.
- **Your workflow module** has `@workflow_step`-decorated functions
  with `produces=` and `requires=` declared.
- **`WorkflowRunner(module, config, products=...)`** ties them
  together.  `runner.build("name")` resolves the DAG, runs whatever
  is stale, caches results in memory and on disk.
- **`WorkflowProducts`** is the on-disk persistence layer.  Type
  registry handles save/load; cache keys handle freshness.

## Two flavours of workflow

Workflows come in two structural shapes — same machinery, different
patterns.

### Time-loop workflow

A simulation that integrates forward in time.  The dominant product
is a **run directory** that grows incrementally (h5 chain plus
timeseries.csv).  Cached at two granularities:

- **Inner**: the run directory's own manifest tracks "is this run
  steady?" — short-circuits when re-invoked on the same directory
  with matching identity hash.
- **Outer**: the runner records the run directory as a product
  with a cache_key derived from inputs.  Config changes propagate
  freshness to downstream products (summary, tables, plots, movies).

The convection example is a time-loop workflow.

### Product-graph workflow

A pipeline of one-shot computations: mesh → adapted mesh → stress →
permeability → ...  No time loop; each step's output is built
entirely or not at all.  All caching is at the product granularity.

The H2Ex example is a product-graph workflow.

## Recipes vs API

Some operations naturally compose the workflow's primitives:

- `warm_start(source_dir, target_dir, **overrides)` — start a new
  run from an existing converged run, optionally with different
  settings.
- `ramp_rayleigh(values, base_dir=...)` — sequential warm-starts
  through a sequence of Ra values.

These are **recipes** — example scripts that compose
`Run.open` / `Run.load_field` / `Run.create` / `Run.append_step` /
`WorkflowRunner.build`.  They live next to the workflow's code, not
in the `underworld3.workflows` package.

The discipline: round-trip primitives (the methods and classes that
consumed-by every workflow) are API.  Composition patterns
(`warm_start`, `ramp`, `branch_run`, `ensemble_from_ic`) are recipes
until 3+ workflows write nearly-identical versions.  Then they're
worth promoting to API.

This keeps the public surface small and the recipes flexible.
You're encouraged to copy a recipe into your workflow's repository
and adapt it; that's the point.

## Versioning

The workflow runtime is at `underworld3.workflows.__api_version__ =
"0.2"` — pre-1.0.  The shape of `Run`, `Manifest`, manifest
schemas, and the `cache_key` computation may shift in 0.x.  When a
second consumer of the time-loop primitives (a fault-mechanics
workflow, a subduction workflow) lands and exercises the API
without major changes, we bump to 1.0 and commit to backward
compatibility.

Each manifest carries a `workflow_api` field stamping the version
that wrote it.  Reading a pre-stamped manifest is non-fatal —
`Manifest.workflow_api` returns `None`, and freshness checks fall
back to existence-based.

## What's currently in the package

Public API surface (see [API reference](../../api/workflows.md) for
detail):

- **Configuration** — `WorkflowConfig` (Pydantic base),
  `config_cache_key`, `config_snapshot`.
- **Runtime** — `@workflow_step`, `WorkflowRunner`,
  `WorkflowProducts`.
- **Run-directory primitives** — `Run`, `Manifest`, `RUN_NAME`.
- **CLI helper** — `cli_from_config`, `config_from_args`.
- **Diagram** — `diagram(module)`, `render(module, output_path)`.
- **Discovery** — `view`, `list_workflows`, `init_workflow`.

Out of scope on purpose:

- Solver wrappers — workflows use UW3 solvers directly.
- A "workflow GUI" — the package gives observers + DAG metadata so
  a UI layer can be built on top, but doesn't ship one.
- Generic time-loop scaffold (`run_loop`) — stays in the example
  until a second time-loop consumer makes the right callback shape
  obvious.

## Where to go next

- The [convection user guide](../../examples/workflows/convection-user.md)
  walks through a complete time-loop workflow from a runner's
  perspective.
- The [convection developer guide](../../examples/workflows/convection-developer.md)
  walks through the same workflow from a builder's perspective.
- The [API reference](../../api/workflows.md) lists every public
  symbol and its docstring.
- [Building workflow packages](workflow-packages.md) covers the
  practical mechanics of pip-installable workflow repositories.
