# Memory retained per faulted-mesh rebuild (hyperion, 2026-10-26)

Kaiju production (feature/box-faults-warm 99259d35, np 16, the ZG95 annulus
slab driver): a constant ~7 GB per node (0.44 GB per rank) per faulted-mesh
rebuild, one full mesh footprint, until an out-of-memory kill after 24
rebuilds. Branch `bugfix/faulted-rebuild-leak` from 99259d35.

## Fixture

`probe_rebuild_memory.py`: the moving-fault box of fault-split-multigrid.md,
k = 3, 20 rebuilds with three solves each; every reference dropped and the
garbage collected after each rebuild, as the driver does. Resident set and
live Mesh / MeshVariable / PETSc objects after every solve (`mem_*.jsonl`).

| | resident growth per rebuild | live after 20 rebuilds (Mesh / DM / Mat / KSP) |
|---|---|---|
| serial, before | 53.7 MB | 183 / 427 / 240 / 20 |
| serial, after | 0.1 MB | 12 / 28 / 12 / 1 |
| np 2, before | 56.0 MB | 189 / 439 / 240 / 20 |
| np 2, Python owners fixed only | 52.7 MB | 12 / 28 / 12 / 1 |
| np 2, after | 0.1 MB | 12 / 28 / 12 / 1 |

Ordinary steps add nothing measurable (about 6 MB on a mesh's first solve,
flat after): the growth is per rebuild, the whole previous mesh retained.

## What held each dropped mesh

`probe_who_holds.py` walks the referrers of a dropped mesh to a root.

1. The default model's registries (`underworld3.model`): `_meshes` and
   `_solvers` were strong dicts. Now `WeakValueDictionary`, as `_swarms`
   already was. `_variables` stays strong (a mesh holds its variables weakly,
   so the registry is their owner) and a finalizer on each mesh drops its
   variables when it is freed.
2. sympy's function caches holding the coordinate symbols, which held their
   mesh: `UWCoordinate`, `CoordinateSystem` and its geographic and spherical
   accessors now hold the mesh weakly.
3. `MeshVariable._init_params`, the constructor arguments kept for life with a
   strong `mesh`: dropped once `__init__` has read them.
4. Parallel only: petsc4py does not destroy a collected object's PETSc object
   in parallel (the ranks collect in different orders and destroy is
   collective); it defers it to a list that only `PETSc.garbage_cleanup`
   empties, and nothing called it. PETSc's object table after five rebuilds:
   2217 live DMs at np 2 against 98 serial. `Mesh.__init__`, collective over
   its DM's communicator, now collects and cleans up on that communicator
   (`underworld3.mpi._free_petsc_garbage`).

## Tests

- `tests/test_0077_model_registry_does_not_pin.py`: a dropped mesh with a
  variable and a solver, and a dropped faulted rebuild with a solve, are freed.
  Failed before on every object; passes after.
- `tests/parallel/test_0078_rebuild_memory_parallel.py`: at np 2, resident
  growth per rebuild under 10 MB on every rank. Before: 39.6 MB; passes after.

Tier A level 1 (serial): 1 failed, 1495 passed; the failure
(`test_0864_split_checkpoint_roundtrip`) fails identically on 99259d35. At
np 2, custom-MG, rotated free-slip, fault-split multigrid and the new test:
19 passed, 4 skipped. A first version that made `_variables` weak broke 24
tier-A tests (a variable the caller did not keep vanished); the strong
registry with the mesh finalizer replaced it.

## Not covered

The driver's own references. If it keeps the previous child (for example in
its warm-start transfer of V onto the new mesh) past the rebuild, that child
stays alive by design; the library no longer adds an owner of its own.

## Adversarial review (2026-10-27)

Two independent read-only passes on 99259d35..547c2e54 (the collective
cleanup in mesh construction; object lifetimes after the registry change),
then fixes in 56df136b and ff782c4d.

Found and fixed:
- `Model._part_objects` (the step transcript's part -> object map) was a strong
  dict keyed per instance: every solver that solved inside `model.step` was
  kept with its mesh. test_0077 ran with transcripts off, so it could not see
  it. Now weak; test_0077 gains a solve-in-a-step test with transcripts on.
- My fix had made a variable stop keeping its mesh (a helper returning only a
  variable lost the mesh). Ownership is now: a variable holds its mesh, a mesh
  owns its variables (`mesh._vars` a plain dict; the cycle is collected with
  the mesh), the model indexes meshes, variables and solvers weakly. Variable
  symbols hold their variable weakly and coordinates their mesh weakly, so
  sympy's caches pin nothing. This also removes the stale `stats()`
  temporaries the registry purge missed (a RuntimeError after a rebuild).
- `_primary_mesh_id` outlived its mesh and could be taken by a reused address,
  differently per rank. The primary mesh is now a weak reference, a freed one
  is replaced by the next mesh, and `set_primary_mesh` always sets it (it did
  not for an unregistered mesh while a live primary existed).
- A `.msh` read with `comm=` would clean up on COMM_WORLD; the cleanup now uses
  the reader's communicator. Latent (no caller passes `comm=`).
- test_1058 made its orphaned field by resetting the model, which no longer
  releases a variable; it now orphans one the way the statistics helpers do
  (taken off `mesh.vars`), with its assertions unchanged.

New tests in test_0077 (a variable keeps its mesh; a solver that solved in a
step is freed; the primary mesh is released and replaced) fail on 547c2e54
and pass on ff782c4d.

Attacks that failed: no `Mesh(` is built by a subset of ranks or on a
sub-communicator in src (COMM_SELF readers return raw DMs, never a Mesh);
`PETSc.garbage_cleanup` touches only the given communicator's garbage and
destroys only what every rank queued; no CoordinateSystem is pickled or
copied; nothing reads `_init_params` after `__init__`; snapshot capture and
restore work off live meshes; no internal solver is reachable only through the
model's `_solvers`.

Not changed: the `gc.collect()` in every `Mesh(...)` costs time in deep
hierarchy builds (kaiju: +0.3 s, 3%, per faulted build); `extract_meshes`
reports a freed mesh's coordinates as a foreign mesh "None" (a wrong message,
not a wrong result).

Validation on ff782c4d:
- memory probe, 15 rebuilds: 0.1 MB per rebuild serial and np 2 (meshes, DMs
  and variables constant);
- tier A level 1 serial, on 56df136b: 1495 passed, 4 failed (test_0864,
  failing identically on 99259d35, and the three test_1058 premise failures
  fixed in ff782c4d; test_1058 passes there). Rerun on ff782c4d: 1498
  passed, 1 failed (test_0864 only);
- tests/parallel/test_*.py: 163 passed, 16 skipped at np 2; 171 passed, 8
  skipped at np 4; no failures, no hangs.

## Review 3: the development-based branch (2026-10-27, after shutdown; NOT yet acted on)

Independent read-only review of bugfix/faulted-rebuild-leak-dev (35244335)
against development 7222caf3. Must be addressed before the PR.

- should-fix, parallel hang: a dropped mesh with variables is in a reference
  cycle (mesh._vars <-> var._mesh_owner), so it stays in the weak registries
  until a gc pass, whose timing differs per rank. write_snapshot
  (checkpoint/disk_snapshot.py:317) loops over model._meshes calling the
  collective mesh.write_checkpoint: rebuild, drop, model.save_state at np 2
  can mismatch the collective writes. Serial: snapshots may include the
  dropped mesh (read_snapshot :507 "mesh not registered"; snapshot.py:206
  "duplicate mesh name"). Fix: gc.collect() on every rank (or
  _free_petsc_garbage) at the top of write_snapshot / snapshot() /
  read_snapshot.
- should-fix: Swarm.mesh (swarm.py:3782) resolves its mesh by id through the
  weak registry, holding no reference: a swarm on a mesh nobody else holds
  (no proxied variable yet) loses it at the next Mesh(...) (gc in __init__),
  or silently gets an unrelated mesh at a reused address. Fix: the swarm holds
  its mesh (strongly, or weakref with a clear error), not by id.
- should-fix: "a mesh owns its variables" holds for the base variable only:
  EnhancedMeshVariable (the user-facing MeshVariable) registers the wrapper in
  model._variables while mesh._vars holds _base_var (enhanced_variables.py:
  135-143). A dropped wrapper vanishes from the registry: get_variable returns
  None, units extraction (units.py:375) and nondimensional scaling
  (nondimensional.py:34, 467) skip it (silently unscaled), adjoint tokens
  (adjoint.py:626-636) lose it. Related: the bare-name key points at the most
  recent same-named variable only (model.py:658-695).
- should-fix (latent): MeshVariable.__del__ destroys _gvec explicitly
  (discretisation_mesh_variables.py:2452); it now runs from cyclic gc at
  per-rank times, bypassing petsc4py's deferred list. Drop the explicit
  destroys and let petsc4py defer them.
- nit: checkpoint/snapshot.py:398-399 builds dicts from model._meshes.values()
  without list(...).

Failed attacks: no partial-rank or sub-communicator Mesh build in src
(Mesh.__init__ already had a COMM_WORLD barrier and a dm.comm allgather, so
the cleanup adds no new collectivity); _init_params read only in __init__; no
Mesh / CoordinateSystem / UWCoordinate pickled or deep-copied; model.mesh
deterministic (collection precedes registration); strong mesh._vars only stops
variables vanishing.
