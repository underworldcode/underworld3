---
title: "State Snapshots & Restore"
---

# State Snapshots & Restore

## Overview

`Model.save_state()` and `Model.load_state()` are a *stash for timesteps* —
a quick "hold that thought, I might need to come back" mechanism for
time-stepping code. Take a snapshot, try a step, and if you don't like
the result, restore and try again. The system is put back exactly as it
was, as if the discarded step never happened.

Typical uses:

- **Backtrack past an instability** — a step blows up; restore and
  continue with a smaller Δt or a different scheme.
- **Adaptive Δt with an error / CFL check** — take a step, measure it,
  restore and retry if it violated your criterion.
- **Predictor–corrector probing** — try a predictor, inspect the
  corrector, fall back if it isn't converging.
- **Multi-stage time integration** (RK-style) — restore to the start
  of a step between stages.

The same entry points serve two storage modes — in-memory for fast
intra-run stash, and on-disk for persistent restart / postprocessing.
You pick by giving (or not giving) a ``file=``. The existing
``mesh.write_timestep()`` / ``read_timestep()`` and
``mesh.write_checkpoint()`` / ``MeshVariable.read_checkpoint()``
paths are unchanged and continue to serve their existing roles (see
"Choosing between paths" at the bottom).

## The API

```python
import underworld3 as uw

model = uw.get_default_model()

# ... set up mesh, variables, swarm, solvers, step a few times ...

# In-memory: the "stash for timesteps" use case.
token = model.save_state()       # capture everything, return a token

# ... take a speculative step you might regret ...

model.load_state(token)          # put everything back exactly
```

To persist on disk instead, pass a path. The same call captures the
same state — only the storage layer differs:

```python
# On-disk: persistent snapshot for restart, postprocessing,
# bisection studies, or transferring a run to another machine.
model.save_state(file="step42.snap.h5")
# ... later, or in a fresh process with the same model set up ...
model.load_state("step42.snap.h5")
```

``save_state()`` returns a :class:`Snapshot` token when called
without ``file``; you can hold several at once and restore any of
them. With ``file=...`` it writes a self-contained on-disk snapshot
and returns the path.

``load_state()`` takes either a token or a path string — it figures
out which from the argument type, so the same call works for both
storage modes.

## What is captured

You do not enumerate anything — `save_state()` captures the full state
of the model automatically:

- mesh coordinates,
- all mesh-variable values,
- all swarm particle positions and swarm-variable values,
- solver-internal time-integration history (the `DDt` operators that
  drive `AdvDiffusion`, viscoelastic stress history, etc.),
- everything on the model tracker (see below).

Restore rebuilds swarm populations from the snapshot, so it is correct
even if particles migrated, were added, or were lost between snapshot
and restore — that is exactly the situation restore exists for.

## The model tracker: time, step, and your own quantities

A subtle trap in time-stepping scripts: your loop counter and
simulation time usually live in plain Python variables, and
`load_state()` has no way to know about them.

```python
model_time = 0.0
token = model.save_state()
model_time = 5.0          # advance
model.load_state(token)
# model_time is still 5.0 — restore cannot reach a local variable
```

`Model.tracker` solves this. It is a model-dwelling record of where the
run is — and anything you put on it is automatically captured and
restored.

```python
model.tracker.time = 0.0
model.tracker.step = 0

token = model.save_state()

model.tracker.time = 5.0
model.tracker.step = 100

model.load_state(token)

model.tracker.time   # 0.0  — reverted automatically
model.tracker.step   # 0    — reverted automatically
```

`time`, `step` and `dt` come pre-seeded as conventions, but they have
no special status. Any attribute you assign becomes managed state:

```python
model.tracker.peak_velocity = 0.0
model.tracker.energy_history = np.zeros(3)
```

These now travel with every snapshot and revert on every restore — no
extra code, no special handling in your solvers. Using the tracker is
optional; solvers do not depend on it. It is simply the place to keep
the things you want `load_state()` to manage.

```{note} Reserved name
`state` is reserved on the tracker (it is the snapshot mechanism's own
hook). Do not use `model.tracker.state` for your own quantity.
```

```{note} git-stash semantics
Restore returns to *exactly* the captured point. A quantity you add to
the tracker *after* taking a snapshot is removed by a restore of that
snapshot — the same way `git stash pop` does not keep work you started
afterwards.
```

## Worked example: adaptive-Δt backtracking

A canonical CFL-controlled stepping loop. The speculative step is
taken, checked, and either kept or discarded:

```python
import numpy as np
import underworld3 as uw

model = uw.get_default_model()
# ... mesh, swarm, velocity field V_fn, solvers set up ...

cfl_limit = mesh.get_min_radius()
dt = 0.5

while model.tracker.time < t_end:
    token = model.save_state()
    coords_before = swarm._particle_coordinates.data.copy()

    # Speculative step at the current Δt.
    swarm.advection(V_fn, delta_t=dt)
    # ... your solves for this step ...

    # CFL check.
    moved = np.linalg.norm(
        swarm._particle_coordinates.data - coords_before, axis=1
    ).max()

    if moved > cfl_limit:
        # Too big — discard and retry with a smaller Δt.
        model.load_state(token)
        dt *= 0.5
        continue

    # Good step — commit.
    model.tracker.time += dt
    model.tracker.step += 1
    dt = min(dt * 1.1, dt_max)   # let Δt grow again
```

Because the swarm, fields, solver history *and* the tracker's `time` /
`step` are all captured, the `continue` path leaves no trace: the next
attempt starts from precisely where the failed one began.

## Guarantees and scope

```{note} What is guaranteed
- **Discarding a step leaves no trace.** A snapshot → speculative
  step → restore → continue reproduces a run that never took the
  speculative step *bit-for-bit*, including across MPI ranks and
  through real PETSc solves.
- **Parallel-correct.** Works under MPI at any (fixed) rank count.
  Restore recovers the exact global state even if the discarded step
  migrated or lost particles across ranks.
```

```{warning} Limitations
- **In-memory tokens are intra-run only.** Tokens returned by
  ``save_state()`` (no ``file``) live in process memory and do not
  survive the process exiting. They are also a full copy of model
  state — holding many large tokens at once costs memory. To persist
  across runs, use ``save_state(file=…)`` instead.
- **Same rank count.** A snapshot written on *N* MPI ranks must be
  read on *N* ranks. Cross-rank-count restart is not supported by
  this mechanism (use the ``mesh.write_timestep`` path for that).
- **No mesh adaptation across a snapshot.** If the mesh is adapted
  between save and load, ``load_state`` refuses with a clear error
  rather than corrupting state.
- **Recovery vs. a never-snapshotted run** is bit-exact for the
  *discarded-step* guarantee above. Continuing after a load_state
  that ran a real solver may differ from a run that never
  snapshotted by a small amount within solver tolerance — load_state
  resyncs solver fields rather than reproducing their exact internal
  buffers. This does not affect the correctness of backtracking.
```

## On-disk file layout (when ``save_state(file=…)`` is used)

A disk snapshot is **two artifacts**: a small wrapper HDF5 file plus
a sibling ``.bulk/`` directory holding the bulk data. They are a
unit — move them together. The wrapper is rich in metadata and
inspectable with standard h5 tools without UW3 in the loop:

```text
my_run.snap.h5         (~tens of KB; metadata, group structure)
my_run.snap.bulk/      (per-mesh + per-swarm sidecars)
    {mesh}.mesh.00000.h5
    {mesh}.{var}.00000.h5     (one per mesh-variable)
    {swarm}.swarm.h5          (one per swarm)
```

A quick ``h5ls -v my_run.snap.h5/metadata`` shows you the run name,
schema version, simulation time, step, dimensions, MPI rank count at
write, and inventories of meshes / swarms / variables / state-bearer
classes. For a Python-side summary use
``uw.checkpoint.inspect_snapshot(path)``.

## Choosing between paths

| Need | Use |
|---|---|
| Backtrack a few steps inside a running script (RK staging, adaptive Δt, predictor–corrector probes) | ``save_state()`` → token |
| Persist whole-model state across runs (crash recovery, bisection studies, full restart) | ``save_state(file=…)`` / ``load_state(file=…)`` |
| Restart from a previous run on a *different* rank count or remap onto a *different* resolution | ``mesh.write_timestep()`` / ``MeshVariable.read_timestep()`` (KDTree remap) |
| Efficient same-rank restart writing only specific variables for postprocessing | ``mesh.write_checkpoint()`` / ``MeshVariable.read_checkpoint()`` (PETSc DMPlex per-variable) |
| Visualisation for ParaView (XDMF + per-step HDF5) | ``mesh.write_timestep(create_xdmf=True)`` |

## Related

- [Parallel-Safe Scripting](parallel-computing.md) — MPI patterns;
  snapshot/restore is parallel-correct at fixed rank count.
- Developer reference: the state-as-dataclass contract for adding new
  snapshot-managed solver helpers lives in the developer guide.
