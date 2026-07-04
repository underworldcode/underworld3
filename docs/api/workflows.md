# Workflows

The `underworld3.workflows` package provides infrastructure for
organising simulations as DAGs of cached, idempotent computations.
For the conceptual overview see the
[workflow concepts guide](../developer/guides/workflow-concepts.md);
the [convection example](../examples/workflows/convection-user.md)
walks through one workflow end to end.

```{eval-rst}
.. automodule:: underworld3.workflows
   :no-members:
```

The package's public API version is exposed as
`underworld3.workflows.__api_version__`.  Every manifest written
by `Run.write_manifest` or `WorkflowProducts.save` carries this
stamp under the key `workflow_api` so older artefacts can be
identified.

## Configuration

### WorkflowConfig

The Pydantic base class every workflow's config inherits from.
Subclasses declare `_identity_fields` to mark which fields
invalidate cached products on change.

```{eval-rst}
.. autoclass:: underworld3.workflows.WorkflowConfig
   :members:
   :show-inheritance:
```

### config_cache_key / config_snapshot

Helpers used by `WorkflowConfig.cache_key()` and the runner.
Most code shouldn't call these directly, but they're exported for
custom workflows that need fine-grained control.

```{eval-rst}
.. autofunction:: underworld3.workflows.config_cache_key
```

```{eval-rst}
.. autofunction:: underworld3.workflows.config_snapshot
```

## Step decorator

### workflow_step

Mark a function as a workflow step with declared `produces` /
`requires` lists.  The runner walks these to resolve the DAG.

```{eval-rst}
.. autofunction:: underworld3.workflows.workflow_step
```

## Run-directory primitives

### Run

Thin wrapper around an on-disk run directory: manifest, h5 chain,
timeseries CSV, summary.  Used by time-loop workflows where the
run directory is itself a workflow product.

```{eval-rst}
.. autoclass:: underworld3.workflows.Run
   :members:
   :special-members: __init__
```

### Manifest

Read-only view onto a run directory's `manifest.yaml` with
convenience properties for `workflow`, `config_hash`,
`config_snapshot`, `started_at`, `workflow_api`, `cache_key`,
and `inputs`.

```{eval-rst}
.. autoclass:: underworld3.workflows.Manifest
   :members:
```

### RUN_NAME

The filename stem used by `Run` for its h5/xdmf chain
(`<RUN_NAME>.mesh.NNNNN.h5`).  Defaults to `"run"`.

```{eval-rst}
.. autodata:: underworld3.workflows.RUN_NAME
```

## Runner

### WorkflowRunner

Resolves the DAG of `@workflow_step`-decorated functions in a
workflow module, building products on demand and caching them in
memory and on disk via `WorkflowProducts`.

Key methods:

- **`build(name)` / `get(name)`** — return product *name*, building
  if needed.  Synonym pair.
- **`build_all()`** — build every leaf product.
- **`rebuild(name)`** — invalidate and rebuild.
- **`invalidate(name)`** — drop a product from cache + disk.
- **`status(name)`** — `"cached"`, `"on_disk"`, `"missing"`.
- **`dag()`** — display steps with status (HTML in Jupyter, plain
  text in a terminal).
- **`diagram()`** — Graphviz DOT source for the runner's DAG with
  per-product status colours.
- **`observe(callback)`** — register a hook fired on
  cache/load/build events.
- **`what_invalidates(name)`** — set of products that would
  rebuild if *name* changed.

```{eval-rst}
.. autoclass:: underworld3.workflows.WorkflowRunner
   :members:
   :special-members: __init__
```

## Products

### WorkflowProducts

The on-disk persistence layer.  Type-aware save/load (`Mesh` →
HDF5, `Run` → directory pointer, `Path` → file pointer, ndarray →
NPZ, ...).  Maintains a YAML manifest with per-product cache keys
and input audit.

```{eval-rst}
.. autoclass:: underworld3.workflows.WorkflowProducts
   :members:
   :special-members: __init__
```

## CLI helper

### cli_from_config

Auto-derive an argparse parser from a `WorkflowConfig` subclass.
Every Pydantic field becomes a CLI flag, with type-aware mapping:

| Pydantic type | CLI form |
|---------------|----------|
| `bool` | `--flag` / `--no-flag` (BooleanOptionalAction) |
| `int`, `float`, `str` | typed value |
| `Literal[...]` | `choices=` constraint |
| anything else | silently skipped |

Most workflows use this to build their CLI driver in five lines:

```python
from underworld3.workflows import cli_from_config, config_from_args
import my_workflow_config as mw

parser = cli_from_config(mw.MyConfig)
parser.add_argument("--no-evolve", action="store_true")
args = parser.parse_args()
config = config_from_args(mw.MyConfig, args)
```

```{eval-rst}
.. autofunction:: underworld3.workflows.cli_from_config
```

```{eval-rst}
.. autofunction:: underworld3.workflows.config_from_args
```

## DAG diagrams

### diagram

Generate a Graphviz DOT source string for a workflow module's
produces/requires graph.  Pass to `dot -Tpng` (or `-Tsvg` /
`-Tpdf`) to render.

```{eval-rst}
.. autofunction:: underworld3.workflows.diagram
```

### render

One-call wrapper around `diagram` plus the `dot` binary —
generates a rendered PNG/SVG/PDF directly.  Requires Graphviz
on `PATH`.

```{eval-rst}
.. autofunction:: underworld3.workflows.render
```

## Discovery

### view

Display the workflow steps and config classes in a workflow
module.  Convenient inside a Jupyter notebook (renders an HTML
table); falls back to plain text in a terminal.

```{eval-rst}
.. autofunction:: underworld3.workflows.view
```

### list_workflows / init_workflow

Discover available workflows on the system; scaffold a new
workflow into a target directory.  See
`underworld3.workflows.scaffold` for details.

```{eval-rst}
.. autofunction:: underworld3.workflows.list_workflows
```

```{eval-rst}
.. autofunction:: underworld3.workflows.init_workflow
```

## Utilities

### check_dependencies

Check that optional packages a workflow needs are installed; emit
a clear error with install instructions if not.  Use at workflow
module top so users don't see a confusing `ImportError` deep in a
solve.

```{eval-rst}
.. autofunction:: underworld3.workflows.check_dependencies
```

### parse_quantity

Parse a quantity string (`"50 km"`, `"1e21 Pa*s"`) into a
`uw.quantity`.  Used internally by `WorkflowConfig.setup_model`.

```{eval-rst}
.. autofunction:: underworld3.workflows.parse_quantity
```

### show_source

Display the source of a workflow function.  Useful for
notebook-side introspection without leaving the cell.

```{eval-rst}
.. autofunction:: underworld3.workflows.show_source
```
