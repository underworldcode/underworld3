# Descriptions and views

Every core object says what it is once, as data, and one renderer shows it.
`describe()` returns a plain tree; `view()` renders that tree for wherever
you are; `uw.render(...)` gives the same tree as a string in a named form.
The run transcript serialises the same tree when a solver acts, so a
notebook, a note, a run record and a query cannot disagree about what an
object is.

## The record

```python
d = stokes.describe()       # a dict
d["kind"], d["name"], d["summary"]
```

Every description carries `kind`, `name` and `summary`. The rest depends
on what the object holds:

| key | holds |
|---|---|
| `facts` | scalar facts, `{label: value}`, in a stable order |
| `terms` | the named quantities the object was given: `name`, `symbol`, `latex`, `text`, `units`, `description`, and `where` |
| `forms` | its equations, `{name: {symbol, latex, text, description, where}}` |
| `conditions` | its boundary conditions (`boundary_conditions` on a solver, the transcript's name for the same list) |
| `children` | the objects it contains, each a description of its own |

`where` is the list of named expressions inside a value, each carrying the
same keys as a term and its own `where`, followed to the `depth` asked
for. Only strings, numbers, lists and dicts appear in the tree, so it can
be written as YAML or JSON without further work. The live SymPy objects
stay on the object.

The kinds and what they contain:

| kind | children | facts worth knowing |
|---|---|---|
| `solver` | its constitutive model and its histories | `unknown`, `dim`; `forms` are `F0`, `F1`, `PF0` as implemented |
| `constitutive_model` | none | parameters as terms; the flux as a form |
| `history` | none | `scheme`, `order`, `theta`: the time integrator a part used |
| `mesh` | its variables | dimension, cells, coordinate system, units, boundaries, cell quality |
| `variable`, `swarm_variable` | none | symbol, shape, degree, continuity, type, units, proxy |
| `swarm` | its variables | particle count |
| `model` | meshes, swarms, solvers | scales as declared, clock, counts |

## Rendering

```python
stokes.view()                      # Markdown with mathematics in a notebook, text in a terminal
stokes.view(format="latex")        # a fragment for a note
stokes.view(format="yaml")         # the record, for a query or a tool
uw.render(stokes.describe(), "markdown", depth=1)
```

The formats are `markdown`, `text`, `latex`, `yaml` and `json`. `depth`
limits how many levels of children are shown; `describe(depth=...)` limits
how far named expressions are followed into each other. A solver with a
constitutive model written in terms of further named quantities renders
them as a nested "where" list under each form.

`view()` on a class, or `view(class_documentation=True)` on an instance,
shows the class documentation as well.

## Adding a description to a class

Override `describe(self, depth=4)` and return a record built with
`underworld3.utilities.describe.record` and `term`:

```python
from underworld3.utilities.describe import record, term

def describe(self, depth=4):
    facts = {"order": self.order}
    terms = [term("psi", self.psi_fn, description="the quantity tracked")]
    return record("history", type(self).__name__, "a time history", facts=facts, terms=terms)
```

`term()` takes a SymPy expression, a named Underworld expression (whose
symbol, units and description it reads), a quantity or a number. Nothing
else is needed: `view()` finds the description and renders it, and the
contract test (`test_0017`) checks that every kind carries the shared keys
and renders in every format.

A class without `describe()` falls back to its `_object_viewer()`, the
notebook-only display from before this layer existed. New classes should
not add one.

## Where the same tree goes

- The transcript writes a solver's description as its `part` record when
  the solver first acts and again if its forms change; `uw.transcript_key`
  renders those records. A part record is a `kind: part` and carries the
  solver's `forms`, `boundary_conditions` and `terms`; the solver's own
  `kind` and `children` are left out, since the children record themselves
  when they act.
- A query interface or an MCP tool returns `describe()` as YAML or JSON
  with nothing added; `depth=0` is the summary, deeper is the detail.
