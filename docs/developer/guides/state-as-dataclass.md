# State-as-dataclass — the snapshot contract for solver-internal helpers

When adding a new solver-internal helper class (a time-derivative
manager, a controller, a convergence tracker, anything carrying
mutable evolution state in Python attributes), you should declare its
mutable state as a dataclass exposed via a `.state` attribute. The
unitary snapshot / restore toolkit
([design note](../design/in_memory_checkpoint_design.md),
implementation in `src/underworld3/checkpoint/`) discovers state
through this attribute automatically — no per-class registration of
which attributes are "state" required.

## Why a contract

Without a uniform contract, every helper invents its own way of
persisting mutable state — or doesn't, and silently loses information
across snapshot / restore. The audit that motivated this work found
several classes (DDt, parameter mutation history, swarm population
counter, ...) each with bespoke state-tracking and no path to
snapshot. The contract makes the obligation explicit: if your class
has mutable state, declare it as a dataclass, end of story.

Side benefits of the contract beyond snapshot:
- `obj_a.state == obj_b.state` becomes a meaningful comparison — useful
  for regression-testing solver-internal behaviour.
- `repr(obj.state)` is automatic and useful for debugging.
- Schema versioning is tractable: a `_schema_version` field on each
  State dataclass plus a migration registry (v1.1) handles
  cross-version compatibility for on-disk snapshots.

## The interface

```python
from underworld3.checkpoint import SnapshottableState

# Inherit from SnapshottableState — gives you the _schema_version field.
@dataclass
class MyHelperState(SnapshottableState):
    # _schema_version: int = 1  # inherited; override on schema change
    counter: int = 0
    history: list[float] = field(default_factory=list)
    config_name: str = ""
```

The host class exposes the dataclass via a `.state` attribute
(property is fine; stored attribute is fine).

```python
class MyHelper(uw_object):  # uw_object gives you instance_number
    def __init__(self, ...):
        super().__init__()
        # ... set up your state ...

        # Self-register so Model.snapshot() discovers you.
        try:
            import underworld3 as _uw
            _uw.get_default_model()._register_state_bearer(self)
        except Exception:
            pass

    @property
    def state(self) -> MyHelperState:
        return MyHelperState(
            counter=self._counter,
            history=list(self._history),
            config_name=self._config_name,
        )

    @state.setter
    def state(self, s: MyHelperState) -> None:
        if s._schema_version != MyHelperState._schema_version:
            raise ValueError("schema mismatch")
        self._counter = s.counter
        self._history = list(s.history)
        self._config_name = s.config_name
        # If your class has *derived* state (caches, downstream
        # coefficients, ...) recompute it here so post-restore reads
        # are consistent without waiting for the next solve.
```

The protocol check is structural: any object exposing a `.state`
attribute that is a `SnapshottableState` instance is `Snapshottable`.

## Option (B) vs (C): stored vs derived dataclass

The design note discusses two implementation styles. Both satisfy the
protocol; pick based on whether the class is new or being retrofitted.

### Option (C): state is the authoritative store. *Prefer for new code.*

```python
class MyNewHelper:
    def __init__(self, ...):
        self.state = MyHelperState(counter=0, history=[])

    def step(self, dx):
        self.state.counter += 1
        self.state.history.append(dx)
```

Every mutation site reads/writes `self.state.<field>` directly. The
dataclass *is* the storage. Self-documenting; mistakes (forgetting to
add a new field to the state object) are obvious because adding state
involves adding a field to the dataclass.

### Option (B): state is a derived view over private attrs. *Use for retrofits.*

```python
class MyExistingHelper:
    def __init__(self, ...):
        self._counter = 0
        self._history = []
        # ... legacy private attrs throughout the class ...

    @property
    def state(self) -> MyHelperState:
        return MyHelperState(counter=self._counter, history=list(self._history))

    @state.setter
    def state(self, s):
        self._counter = s.counter
        self._history = list(s.history)
```

The existing private attrs stay as-is; `.state` is a façade. Less
invasive than option (C) because the existing call sites don't
change — only the new accessor is added.

The five `DDt` flavors in `src/underworld3/systems/ddt.py` (`Symbolic`,
`Eulerian`, `SemiLagrangian`, `Lagrangian`, `Lagrangian_Swarm`) use
option (B) for this reason: they predate the contract, and rewriting
every mutation site would be churn without behavior change.

## What goes in the State dataclass

**Yes:**
- Mutable evolution-tracking state — anything that changes after
  construction and affects subsequent behaviour. Counters, history
  buffers, current-step values, mutation logs.
- Names / stable IDs of bound objects (mesh-variable names, swarm
  names) — restore uses these to verify the binding still holds.
- Configuration flags whose state is meaningful at snapshot time
  (`with_forcing_history`, `recycle_rate`, ...) — they help restore
  detect a mid-run reconfiguration that would invalidate the snapshot.

**No:**
- Live PETSc Vec, DM, or solver handles. Tokens must stay plain
  Python / numpy so they survive DM lifecycle changes (and so v1.1's
  on-disk serialisation works).
- Constructor-time arguments that never change (mesh reference,
  variable type, polynomial degree). Re-derive on restore from the
  surviving wrapper.
- Bulk numerical data like mesh-variable DOFs or swarm-variable
  arrays. Those travel via the dedicated mesh-var / swarm-var
  snapshot paths.
- Caches, derived coefficients, anything you can recompute from
  primary state in the `.state` setter.

## Schema versioning

`_schema_version` exists for a v1.1 / v1.2 feature: on-disk snapshots
that survive across UW3 versions. When you change the shape of a
State dataclass (rename a field, add a required field, change a
type), bump the version and add a migration entry. Within a single
process (v1 in-memory only), the version is checked for strict
equality — any mismatch is a programming error.

The migration registry itself is v1.1 work (item 6 of the design
note). For now: define `_schema_version: int = 1` on every State
dataclass and don't worry about migrations until on-disk lands.

## Where to put your State dataclass

Next to the host class. `DDtSymbolicState` lives next to
`Symbolic` in `src/underworld3/systems/ddt.py`. This keeps the
dataclass and the class it describes co-located; changes to one show
up in the diff against the other.

## Testing

A typical snapshot test for a state-bearing class:

```python
def test_my_helper_roundtrip():
    uw.reset_default_model()
    model = uw.get_default_model()
    h = MyHelper(...)

    # Advance state.
    h.step(0.1)
    h.step(0.2)
    state_pre = h.state

    snap = model.save_state()

    # Mutate.
    h.step(0.5)

    model.load_state(snap)

    # Verify primary state recovered.
    assert h.state == state_pre
```

The dataclass `__eq__` makes the final assertion a one-liner.

## Related

- [Design note](../design/in_memory_checkpoint_design.md) — full design rationale, scope, and roadmap.
- `src/underworld3/checkpoint/state.py` — protocol definitions.
- `src/underworld3/checkpoint/snapshot.py` — capture / restore orchestration.
- `src/underworld3/systems/ddt.py` — five working examples (option (B) retrofits).
