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
