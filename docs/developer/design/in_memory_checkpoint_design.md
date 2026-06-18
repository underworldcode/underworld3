# In-memory checkpoint as a general UW3 capability

Design note. Spun off from the deformable-surface architectural
discussion (2026-05-11) as a self-contained capability that is bounded,
useful in its own right, and not specifically tied to free-surface code.
The companion note that motivated this is
`docs/developer/design/deformable_surface_metronome_design_note.md` (in
the `feature/exp-integrator-freesurface` worktree).

## Motivation

**Primary use case: backtrack past unstable timestepping.** A timestepper
hits an instability or a sudden regime change; the cleanest recovery is
to restore the last known-good state and continue with smaller Δt
(or a different scheme). Today this is done ad-hoc by users where it
happens at all.

**Secondary uses, all sharing the same primitive:**

- Multi-stage time integration (RK4 between stages — restore to
  start-of-step, deform to next stage configuration, sample rate). The
  free-surface integrator-zoo work is the immediate case; the same
  primitive applies to any multi-stage scheme.
- Adaptive Δt with error estimate (take a step, estimate per-step
  error, restore + retry if too large).
- Predictor-corrector probing (try a predictor, check the corrector
  residual, fall back if not converging — relevant to the VEP work).
- Regime-change feeling-out (e.g., elastic predictor → check yield
  surface → restore + plastic split if violated).

All five want one thing: at one moment, capture *enough* state that the
system can be put back at any point afterwards as if nothing had
happened.

## Reframe: same operation as on-disk checkpoint, different backend

Rather than build a parallel snapshot mechanism, treat in-memory
snapshot as a *backend variant* of the existing checkpoint code. Two
benefits:

1. **One source of truth for "what is system state."** Whatever set of
   (mesh coords, MV DOFs, swarm positions, swarm-var values, algorithm
   history, ...) the checkpoint serialiser captures becomes the
   contract for in-memory snapshots too. Adding new state-bearers means
   adding them once; both backends pick them up.
2. **The existing checkpoint code gets exercised harder and improved
   as a side effect.** In-memory roundtrip is a cheap unit test —
   every snapshot/restore cycle exercises the serialise/deserialise
   paths, which surfaces gaps and bugs that disk-only checkpointing
   exposes only at quarterly-test cadence.

## Two mesh-variable output paths

This design note predates the unified timestep writer. The current public API
keeps the same two reload semantics, but exposes new output through
`Mesh.write_timestep(...)`:

**Coordinate-remap path.**
- Write with `Mesh.write_timestep(...)`.
- Read selected variables with `MeshVariable.read_timestep(...)`.
- Uses `/fields` coordinate/value datasets and coordinate/KDTree remapping.
- Can load data onto a different mesh or MPI decomposition.

**PETSc-native reload path.**
- Write with `Mesh.write_timestep(..., petsc_reload=True)`.
- Read selected variables with `MeshVariable.read_checkpoint(...)`.
- Uses PETSc DMPlex topology, section, vector, and `PetscSF` metadata.
- Intended for exact same-mesh finite-element vector reload.

`Mesh.write_checkpoint(...)` is retained as a compatibility wrapper for older
scripts. New code should use `Mesh.write_timestep(..., petsc_reload=True)` for
PETSc-native reload output.

**For in-memory snapshot, the PETSc-native path is the conceptual model.**
Restore goes back to the same DM, so the resolution/decomposition flexibility
of coordinate remap is unneeded. But the in-memory backend does not use the
HDF5 Viewer — it copies section structure and vector data directly into numpy
arrays. Same conceptual capture, different mechanism.

## What state must be captured

Audit-informed inventory:

**Captured today (in Path A):**
- DM topology and section
- Deformed mesh coordinates
- Mesh-variable global vectors (DOF values) — **including `DDt.psi_star`,
  which is itself a mesh variable; its DOF data goes through the
  section path automatically**
- Swarm-variable values (via mesh-DM proxy fields)

**NOT captured today, required for full-state in-memory restore:**

| state | location | priority |
|---|---|---|
| `DDt._dt_history` (variable-Δt BDF history; list of floats) | `systems/ddt.py:386` | **high** |
| `DDt._history_initialised` (bool) | `systems/ddt.py:383` | **high** |
| `DDt._n_solves_completed` (int) | `systems/ddt.py:384` | **high** |
| Binding `DDt instance ↔ psi_star MV(s)` | implicit in `__init__` | high |
| Simulation time, step counter | not in any object today | high |
| Parameter mutation history | `parameters.py:145` (`_history`) | medium |
| `Swarm._mesh_version` | `swarm.py:2421` | medium |
| Solver iteration counts / convergence history | scattered | low |

**The DDt example is representative, not exceptional.** Algorithm-internal
scalar/list state on Python objects is the architecturally significant
gap — it lives in solver-adjacent Python objects, not in any PETScSection.
The in-memory snapshot must compose PETScSection state + Python-side
mutable state into a single token. The next section addresses how this
composition should be designed in general — not as a per-class bolt-on,
but as a contract that new algorithm-helper classes follow from day one.

## General serialisation contract for solver-internal state

The opportunity. Bolting per-class snapshot hooks onto each algorithm
helper as it appears would produce a half-baked system that silently
misses state whenever a new helper is added without a corresponding hook.
DDt today, the next adaptive-Δt controller tomorrow, the gamma estimator
the day after — the pattern recurs.

This is the right moment to design a general serialisation contract for
solver-internal state, before it leaks into the user API. New
algorithm-helper classes should declare their state slots from day one;
existing classes (DDt, parameter mutation history, solver convergence
bookkeeping) get retrofitted as they're touched.

**Three options for the contract.**

(A) **Declarative slots.** Class declares a class-level list of
attribute names that constitute state.
```python
class DDt:
    _state_attrs = ('_dt_history', '_history_initialised',
                    '_n_solves_completed')
```
Snapshot copies named attrs; restore writes them back. Stringly-typed;
silent breakage when attribute names drift. Least invasive — works as a
mixin on any existing class.

(B) **Explicit save/restore methods.** Class implements paired methods.
```python
class DDt:
    def _save_state(self) -> dict: ...
    def _restore_state(self, d: dict) -> None: ...
```
Most flexible (transform on save, validate on restore). Most boilerplate;
easy to forget to update when adding a new state attribute. Standard
Python pickling pattern (`__getstate__` / `__setstate__`) is essentially
this option.

(C) **State as a first-class object.** Computation and state are
separated; the State object is a dataclass / pydantic model that
trivially serialises.
```python
@dataclass
class DDtState:
    dt_history: list[float]
    history_initialised: bool
    n_solves_completed: int
    psi_star_var_names: list[str]   # binding to MVs lives here too

class DDt:
    def __init__(self, ...):
        self.state = DDtState(...)
    # All mutations go via self.state.<attr>
```
Most self-documenting; enables side benefits beyond serialisation
(deep-copy, equality testing, repr, schema-versioned migrations). Most
invasive — changes how new solver-internal classes are written.

**Recommendation.** (C) for new code, (B)-style adapters for the small
number of existing classes that need retrofitting (primarily DDt and
parameter-mutation-history). (A) is rejected because the silent-drift
failure mode is exactly the kind of half-baked outcome we're trying to
avoid by designing this now.

The decision matters because it sets the pattern for every
algorithm-internal class added over the next few years. The cost of
choosing wrong is that snapshot tokens silently miss state for any
class added without proper instrumentation; the cost of choosing right
is that new solver-internal classes get serialisation, deep-copy, and
equality testing automatically.

**Side benefits of (C) beyond checkpoint.**
- `solver_a.state == solver_b.state` becomes a meaningful comparison —
  useful for regression testing of solver-internal behaviour
- `repr(obj.state)` is automatic and useful for debugging
- snapshot tokens compose trivially: `{id(obj): obj.state.copy()}`
- schema versioning is tractable: a `_schema_version` field on each
  State object plus a migration registry handles cross-version
  compatibility (relevant for on-disk checkpoints that survive UW3
  upgrades)
- the `Snapshottable` interface becomes a one-liner: "has a `.state`
  attribute that is a dataclass."

**Bindings to PETSc-owned objects.** State objects must NOT hold direct
PETSc Vec / DM / MV handles — only stable identifiers (variable names,
mesh names) that can resolve back to the live object on restore. This
keeps tokens plain-Python and avoids the DM-lifecycle hazards
identified in earlier work.

## A third backend: on-disk full-state snapshot

The in-memory backend is one storage option. The same capture/restore
machinery supports a slower-but-persistent **on-disk full-state**
backend with no architectural changes — only the serialisation layer
differs.

This is **distinct from the existing `write_timestep` on-disk path**,
which is selective (per-variable), designed for visualisation, emits
XDMF for ParaView, and does not restore solver-internal state. Three
backends serving three different needs:

| dimension | existing `write_timestep` | new on-disk full-state | new in-memory |
|---|---|---|---|
| storage | HDF5 + XDMF, per-variable | HDF5, monolithic | dict of numpy arrays |
| selectivity | user picks variables | always full state | always full state |
| Python-side state | not captured | captured | captured |
| restorability | partial (re-init solvers, lose history) | bit-equivalent | bit-equivalent |
| persistence | survives process exit | survives process exit | intra-run only |
| speed | medium | slow (HDF5 I/O) | fast (RAM copy) |
| typical use | visualisation, restart-with-changes | crash recovery, bisection, debugging | RK staging, backtrack, adaptive Δt |

The existing `write_timestep` continues to serve its role unchanged.
The new mechanism (in-memory + on-disk-full-state) addresses a
different need: faithful state restore for algorithmic uses where
"approximate restart" would silently corrupt the run.

**Use cases the on-disk full-state backend opens up:**

- **Crash recovery for long simulations.** Periodic snapshots written
  to disk; on restart, restore from the most recent. No lost work
  beyond the snapshot interval.
- **Cross-run resumption.** A simulation that ran to $t = T_1$ can be
  picked up at $t = T_1$ in a later session, bit-equivalent — including
  DDt history and any other algorithm-internal state.
- **Bisection / branching exploration.** Snapshot at a decision point;
  try one parameter path; if unsatisfactory, restore and try another.
  Powerful for sensitivity studies on long runs.
- **Debugging captures.** Snapshot at a problem point; examine offline;
  iterate without re-running the costly setup.

**What this adds to the design.** Three implications, all bounded:

1. **Backend abstraction must serialise to bytes from day one.** The
   in-memory backend stores numpy arrays directly; the on-disk backend
   serialises them to HDF5. Both go through the same `save_*` / `load_*`
   protocol on the backend interface — the abstraction is shaped by
   needing both.
2. **Schema versioning becomes critical.** In-memory tokens are
   short-lived; on-disk tokens may be loaded by a future UW3 version.
   Each `State` dataclass carries a `_schema_version`; a migration
   registry handles cross-version restore. (The on-disk
   `write_timestep` path has no equivalent need because it doesn't
   restore solver state.)
3. **Generation-counter semantics need a cross-process branch.**
   Within-process: counter check ensures you don't restore an
   invalidated snapshot. Across-process (restoring from disk on a
   fresh run): the model is being initialised *from* the snapshot,
   not restored *to* a previous state — counters are set to the
   snapshot's values, no invalidation check needed. Restore code
   distinguishes the two paths.

## API shape

Full-state always; backend chosen at save time by passing (or
omitting) a ``file=``. (Final names landed after a phase-5 rename
pass that replaced the draft ``snapshot``/``restore`` verbs — see
the user guide at ``docs/advanced/snapshot-restore.md``.)

```python
# Same call, different storage — dispatch on whether a file is given.
token = model.save_state()                       # in-memory (default)
model.save_state(file='step42.snap.h5')          # on-disk full-state

model.load_state(token)                          # in-memory restore
model.load_state('step42.snap.h5')               # on-disk restore

# Existing per-variable selective on-disk path is unchanged:
mesh.write_timestep('step42.h5', ...)            # visualisation; not full-state
```

Backends share a single `Snapshot` structure — only the serialisation
layer differs:

```python
class Snapshot:
    section_state: dict[DM, SectionAndVecData]   # PETSc state
    python_state: dict[ObjectId, ObjectState]    # State dataclasses
    generations: dict[ObjectId, int]             # within-process invalidation
    metadata: dict                                # sim time, step #, schema version
```

The in-memory backend stores `section_state` arrays as numpy buffers
directly; the on-disk backend serialises them to HDF5 datasets. The
Python-side `State` dataclasses serialise to HDF5 attributes (small
scalars / short lists) and groups (arrays).

Tokens are plain Python / numpy — never PETSc Vec or DM handles.
On-disk "tokens" are paths to single HDF5 files. Either way, the
DM-lifecycle hazards identified in earlier work do not apply.

## Restore semantics for swarms — rebuild, do not refuse

**Correction (2026-05-11, post-review).** An earlier draft of this
section proposed using a per-swarm `_population_generation` counter as
an *invalidation gate*: if the counter at restore differs from the
counter at capture, raise rather than restore. That design is wrong.
The whole point of the toolkit is to undo state changes — including
particle motion / migration / repopulation between capture and
restore. Refusing on counter mismatch breaks the central use cases (RK
staging, backtrack-on-instability, adaptive Δt retry — all of which
*will* migrate particles between capture and restore).

**The correct semantics: restore rebuilds the swarm's particle
population from the snapshot.** Specifically:

1. Clear the swarm's current local particles.
2. Re-add the captured per-rank coordinates via
   `add_particles_with_coordinates(saved_local_coords, migrate=False)`.
   The mesh partition is deterministic and unchanged within v1 scope
   (mesh-version check still applies), so particles that were local at
   capture are local at restore — no migration step needed.
3. Write the captured per-particle variable data back into the
   newly-added particles, in their captured order.

The `Swarm._population_generation` counter is still useful as
*informational metadata* — it can flag in logs / metadata what
happened between capture and restore, it can feed cache invalidation
in other consumers, it can power future optimisations (e.g., a fast
in-place restore when the counter happens to match). But it is **not**
a restore gate.

Mutation sites where the counter is incremented (current line numbers
re-derived per audit; the design's correctness does not depend on the
exact set as long as we over-bump rather than under-bump):

| file | site | call |
|---|---|---|
| `swarm.py` | end of `populate()` | covers internal `dm.addNPoints()` calls |
| `swarm.py` | `Swarm.migrate()` after `migration_disabled` early-exit | bumps unconditionally; conservative no-op safe |
| `swarm.py` | after `dm.migrate()` in `add_particles_with_coordinates()` | direct PETSc DM call, not via `Swarm.migrate` |
| `swarm.py` | after `addNPoints` in `add_particles_with_global_coordinates()` | catches `migrate=False` callers |
| `swarm.py` | `advection()` remesh path after re-injection `addNPoints` | recycle-mode reinitialisation |

`Mesh._mesh_version` is a separate counter on the mesh side. In v1
the mesh-version mismatch *does* refuse restore — see the next
section.

## Architectural work required

In rough dependency order:

1. **Backend abstraction layer.** Extract the "capture" logic from the
   output-storage logic used for PETSc-native reload. Introduce a
   `CheckpointBackend` protocol (e.g., `save_section`, `save_vector`,
   `save_metadata`, `load_section`, `load_vector`, `load_metadata`).
   Shaped from day one to support both backends — the in-memory case
   is the cheapest test of the abstraction's correctness; the on-disk
   case is what locks in the byte-level serialisation contract. The
   existing HDF5 path is refactored to fit the protocol; the wedge
   is clean because section/vector ops are PETSc abstractions and the
   HDF5 coupling lives in the Viewer construction.
2. **Backend implementations.** Two concrete backends:
   - `InMemoryBackend` — dict of numpy arrays. Trivial once the
     abstraction exists.
   - `OnDiskFullStateBackend` — single monolithic HDF5 file. Shares
   PETSc-state serialisation with the existing PETSc-native reload path
   (already HDF5); adds Python-state serialisation as HDF5 attributes/groups.
3. **Adopt the serialisation contract for new solver-internal code.**
   Decision: option (C) — state as first-class dataclass (see "General
   serialisation contract" section). Any new algorithm-helper class
   added from this point on declares its state as a separate
   dataclass; the checkpoint mechanism reads `.state` automatically.
   Document the pattern in the developer guide; add a check to CI that
   new classes in `systems/` and `solvers/` declare `.state`.
4. **Retrofit existing solver-internal classes** to the contract.
   In priority order: `DDt` (`systems/ddt.py`), parameter mutation
   history (`parameters.py:145`), any solver convergence-tracking
   state (audit pending). Each retrofit is small; total bounded by
   the number of classes (probably under ten).
5. **Swarm rebuild on restore + informational
   `_population_generation` counter.** Snapshot captures per-rank
   particle coordinates and per-variable arrays. Restore clears the
   current local population and re-adds the captured particles at
   their captured coords (see "Restore semantics for swarms" section
   above for the corrected design). The counter is bumped at every
   identified mutation site for informational use; it is **not** an
   invalidation gate.
6. **Schema versioning + migration registry.** Each `State` dataclass
   carries a `_schema_version` integer. A central registry maps
   `(class, version)` to migration functions that lift older State
   data to the current schema. In-memory restore checks version
   equality (any mismatch is a programming error since both sides are
   the same process); on-disk restore consults the migration registry
   and applies migrations in sequence. Without this infrastructure,
   on-disk snapshots break the moment any retrofitted class evolves.
7. **`Model.snapshot()` / `Model.restore()` orchestration.** Walks
   registered meshes, swarms, MVs, and any object exposing a `.state`
   attribute; composes the token; routes to backend (in-memory or
   on-disk) based on whether `path=` is given; restores in safe order;
   distinguishes within-process restore (counter validation enforced)
   from cross-process restore (counters initialised from snapshot).
8. **Tests.** In-memory roundtrip on standard setups (Stokes benchmark,
   swarm-bearing setup, DDt-using setup). On-disk roundtrip on the
   same setups. Cross-process restore: write to disk in process A,
   load in fresh process B, verify the restored model continues
   bit-equivalently from the snapshot point. Verify generation
   invalidation raises cleanly; verify `.state == .state` after
   roundtrip for retrofitted classes.

## Scope boundaries (NOT in v1)

- **Mesh adaptation roundtrip — scheduled for v1.2, not a permanent
  limitation.** A snapshot taken before a `mesh.adapt()` event in v1
  refuses restore via the `_mesh_version` check, because the captured
  DOF arrays are sized for the pre-adapt section and writing them
  in-place into the post-adapt DM would corrupt the run. **v1.2 will
  replace the refusal with a mesh-rebuild path**: capture enough
  topology / section info to destroy the post-adapt DM and rebuild
  the pre-adapt one, then write DOFs into the rebuilt DM. The
  principle is the same as the swarm rebuild (capture-the-state,
  rebuild-on-restore); the implementation is more invasive because
  every MeshVariable / Swarm / solver holds references into the DM
  and those wrappers need to re-bind. v1's snapshot **captures the
  topology / section info even though v1 restore ignores it**, so the
  payload is forward-compatible with v1.2 without a schema bump.
- **Replacing the existing `write_timestep` path.** The selective
  per-variable on-disk path continues unchanged. The new full-state
  on-disk backend is additive and serves a different need (faithful
  restore vs visualisation/restart-with-changes).
- **Cross-rank-count restore.** On-disk full-state snapshots written
  on N ranks restore on N ranks. Restoring on a different rank count
  requires the existing `write_timestep` path's interpolation
  machinery and is out of scope for the full-state backend.
- **Lazy / copy-on-write in-memory tokens.** Always-eager copy for
  v1. The use cases that drove this (RK staging, single-step
  backtrack) all happen on a per-step cadence where eager copy is
  affordable. Lazy/COW is an optimisation for later if someone proves
  it matters.

## Open implementation questions

1. **Does PETSc expose a clean way to copy section + vector state
   directly into numpy without going through a Viewer?** The PETSc
   `Vec.getArray()` / `Section.getDof()` low-level APIs should
   suffice, but it needs a quick prototype to confirm. Alternative:
   `PETSc.Viewer.Type.MEMORY` may exist but support for DM operations
   is uncertain.
2. **How does `Model.snapshot()` discover state-bearing objects?**
   With the contract decision made (option C — `.state` dataclass),
   discovery options narrow to: (a) walk Model's registered solvers
   and recurse into anything with a `.state` attr; (b) explicit
   registration at construction time. (a) is more ergonomic but
   relies on solvers maintaining proper registration with the Model.
   (b) is more explicit but requires every state-bearing object to
   know about Model. (a) is probably right; verify the existing
   solver-Model relationship can support recursive `.state` discovery.
3. **What does `Model.restore(token)` order look like?** Mesh
   coords first (since MV DOFs are tied to mesh layout), then MV
   DOFs, then swarm positions + migrate, then swarm-var values, then
   DDt and Python-side state, then generation counters validated last.
   Confirm this order is correct; particularly whether MV restoration
   needs the mesh to be at restored coords first.
4. **Memory cost on realistic setups.** For a typical coupled-physics
   run (mesh + swarm + several MVs + DDt history), what's the
   per-snapshot byte cost? Drives the answer to "can users keep N
   snapshots in memory without thinking about it." Probably bounded
   by one-Stokes-solve worth of memory, but worth measuring early.

## Status

Audit complete. Design pending review. Implementation has not started.
The work is bounded (no open-ended research items in any of the seven
architectural-work items above) and decomposes into commits that can
land sequentially: backend abstraction → InMemoryBackend → Python-side
registration → swarm generation counter → Model orchestration → tests.

Expected size: around four weeks of careful work, dominated by:
- the backend-abstraction refactor (item 1), which touches existing
  checkpoint code other things depend on, and must be shaped to
  support both backends from day one;
- the contract retrofit (item 4), mechanically small per class but
  needing careful audit so no state is silently missed;
- the on-disk backend (item 2b) and schema versioning (item 6),
  together adding around a week beyond the in-memory-only scope but
  unlocking the cross-run / crash-recovery / bisection use cases.

The contract decision (item 3) and the schema-versioning decision
(item 6) are the highest-leverage pieces — they set the pattern for
every algorithm-internal class added over the next few years and the
durability guarantee for every on-disk full-state snapshot. Worth
getting right even if they slow the immediate work, because the
alternative is years of half-baked snapshots that silently miss state
or break across UW3 versions.

## Cross-references

- `docs/developer/design/deformable_surface_metronome_design_note.md`
  (in `feature/exp-integrator-freesurface` worktree) — the parent
  design discussion that motivated spinning this off as a separate
  capability.
- `publications/free-surface-paper/integrator_zoo_supplementary.md` —
  the empirical work that exposed the need for snapshot/restore as
  part of multi-stage time integration.
