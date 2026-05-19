---
title: "State Snapshots & Restore"
---

# State Snapshots & Restore

## Overview

`Model.snapshot()` and `Model.restore()` are a *stash for timesteps* —
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

This is intentionally *not* archival checkpointing. It is fast,
in-memory, and meant to be used freely within a run. For long-term,
on-disk restart files, use the existing `mesh.write_timestep()` /
`read_timestep()` path, which is unchanged and serves a different
purpose.

## The API

```python
import underworld3 as uw

model = uw.get_default_model()

# ... set up mesh, variables, swarm, solvers, step a few times ...

token = model.snapshot()      # capture everything, return a token

# ... take a speculative step you might regret ...

model.restore(token)          # put everything back exactly
```

`snapshot()` returns a plain in-memory token. You can hold several at
once and restore any of them. `restore()` returns the model to the
exact state at the moment that token was captured.

## What is captured

You do not enumerate anything — `snapshot()` captures the full state
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
`restore()` has no way to know about them.

```python
model_time = 0.0
token = model.snapshot()
model_time = 5.0          # advance
model.restore(token)
# model_time is still 5.0 — restore cannot reach a local variable
```

`Model.tracker` solves this. It is a model-dwelling record of where the
run is — and anything you put on it is automatically captured and
restored.

```python
model.tracker.time = 0.0
model.tracker.step = 0

token = model.snapshot()

model.tracker.time = 5.0
model.tracker.step = 100

model.restore(token)

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
the things you want `restore()` to manage.

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
    token = model.snapshot()
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
        model.restore(token)
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
- **In-memory only.** Snapshots live in process memory and are not
  written to disk; they do not survive the process exiting. They are
  also a full copy of model state — holding many large snapshots at
  once costs memory.
- **Same rank count.** A snapshot taken on *N* MPI ranks is restored
  on *N* ranks. Changing the rank count is not supported by this
  mechanism (use the `write_timestep` restart path for that).
- **No mesh adaptation across a snapshot.** If the mesh is adapted
  between snapshot and restore, restore refuses with a clear error
  rather than corrupting state.
- **Recovery vs. a never-snapshotted run** is bit-exact for the
  *discarded-step* guarantee above. Continuing after a restore that
  ran a real solver may differ from a run that never snapshotted by a
  small amount within solver tolerance — restore resyncs solver
  fields rather than reproducing their exact internal buffers. This
  does not affect the correctness of backtracking.
```

## Related

- [Parallel-Safe Scripting](parallel-computing.md) — MPI patterns;
  snapshot/restore is parallel-correct at fixed rank count.
- Developer reference: the state-as-dataclass contract for adding new
  snapshot-managed solver helpers lives in the developer guide.
