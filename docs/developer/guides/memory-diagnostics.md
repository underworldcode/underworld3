# Memory diagnostics

Long parallel runs occasionally OOM on HPC even when each step looks small.
The `uw.utilities.memprobe` module gives you a way to "light up" memory
tracking on demand, sample at regular intervals, and pin which subsystem is
growing.

## Quick start

```bash
UW_MEMPROBE=1 mpirun -n 16 python my_long_run.py
```

With this flag set, the `Stokes.solve()` and `NavierStokes.solve()` paths
emit a one-line growth report each time they're called:

```
[memprobe] Stokes.solve:
  RSS +0.42 MiB
  kdtree: live +1, total_constructed +1
```

A clean run shows mostly zero deltas. A leak shows the same component
growing on every step.

## What's tracked

| Signal | Source | Cost |
|---|---|---|
| Process RSS (MiB) | `resource.getrusage` | free |
| KDTree live count | `uw.kdtree.live_count()` | free |
| KDTree total constructed | `uw.kdtree.total_constructed()` | free |
| Per-class Python instance counts | `gc.get_objects()` walk | slow — gated behind `full=True` |

`KDTree` instances are tracked via Cython class counters in `__cinit__` and
`__dealloc__`. Cython's deterministic destruction makes the live count
accurate without weak-references.

PETSc-side object and allocation tracking is **not** parsed from Python —
PETSc's own `-log_view` and `-malloc_dump` runtime flags give the same
information more reliably. To enable them from Python:

```python
from underworld3.utilities import memprobe
memprobe.dump_petsc_leaks_at_finalize()  # equivalent to -malloc_dump -objects_dump
```

## API

### Snapshots and diffs

```python
from underworld3.utilities import memprobe

before = memprobe.snapshot()
do_work()
after = memprobe.snapshot()
print(memprobe.format_diff("after-work", memprobe.diff(before, after)))
```

Add `full=True` to also walk Python-class counts:

```python
snap = memprobe.snapshot(full=True)
# snap["py_classes"] = {"underworld3.swarm.Swarm": 3, ...}
```

### Probe context manager

```python
with memprobe.probe("step 42"):
    advance_one_step()
# On exit, the diff is emitted via `print` (configurable).
```

The `emit` keyword takes any callable: `with probe(..., emit=logger.info):`
to route into the logging system, or a rank-aware writer for parallel runs:

```python
import underworld3 as uw
emit = lambda s: uw.pprint(0, s)  # rank-0 only
with memprobe.probe("step 42", emit=emit):
    ...
```

### Decorator

```python
@memprobe.instrument("my-hot-loop")
def step():
    ...
```

When `memprobe.ENABLED` is `False` (the default) the decorator's wrapper is
a single attribute lookup + branch — sub-microsecond — so it's safe to
leave on hot paths permanently.

`Stokes.solve()` and `NavierStokes.solve()` are pre-decorated. Add more if
you want them.

### Runtime toggles

```python
memprobe.enable()
# ...probed region...
memprobe.disable()
```

`UW_MEMPROBE=1` flips it on at import time.

## Debugging recipes

### "RSS grows X MiB per step — which component is it?"

1. Set `UW_MEMPROBE=1` and run for ~20 steps to confirm the per-solve
   growth pattern.
2. Add `with memprobe.probe("step N", full=True):` around your step loop.
   The `full=True` walks `gc.get_objects()` and lists Python class growth
   sorted by absolute change — usually the dominant suspect is on top.
3. If RSS grows but no Python class does, the leak is in PETSc memory or
   C extensions. Re-run with `-log_view -malloc_dump` and inspect the PETSc
   reports written at finalize.

### "Are kd-trees being released properly?"

Check `uw.kdtree.live_count()` directly, or look for the `kdtree: live +N`
line in the diff. KDTrees should drop to zero when their owning object
(typically a `Mesh` or `Swarm`) is destroyed.

### Parallel runs

`memprobe` runs on each rank independently. For meaningful aggregate
output, pipe `emit` through a rank filter:

```python
import underworld3 as uw
def root_only(s):
    if uw.mpi.rank == 0:
        print(s)

with memprobe.probe("step", emit=root_only):
    ...
```

Or compare per-rank snapshots manually for a load-imbalance view.
