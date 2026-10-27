# Kaiju's two per-rebuild costs (hyperion, 2026-10-27)

Branch `bugfix/rebuild-efficiency`, from the leak fix (`bugfix/faulted-rebuild-leak`
fb9b1ba1), the production overlay for faulted runs.

## The solver's functions recompiled on every rebuild (kaiju: 1.4 s)

Not reproduced on the library fixture. `probe_jit_per_rebuild.py`: the
moving-fault box (viscous Stokes, frictionless fault, k = 3), six rebuilds.
One JIT module after the first rebuild and no new one after; `getext` (code
generation plus the cache lookup) takes 0.02 s per rebuild; the solver's
`_build` 0.09 s. The JIT key is a hash of the generated C source with the
module names normalised out, so an unchanged solver hits the cache.

So kaiju's 1.4 s comes from something in its model whose generated source
changes between rebuilds, or from a compile path this fixture does not take
(a viscoelastic history, the swarm, the fault law's terms, a projection).
Needed from kaiju: which solver's functions recompile (a `JIT compiled module`
log line per rebuild with the hash, or `UW_JIT_VERBOSE`), and whether the
source differs (two consecutive rebuilds' generated C).

## adopt_hierarchy aligning the base levels for nothing (kaiju: 0.8-2 s)

`adopt_hierarchy` (called by `FaultNetwork.build` and `place_fault_ribbon_2d`)
co-locates every level of the base's tail with the placed mesh
(`_colocate_level`, a collective redistribution) unless the base was
partitioned for the add-on. A driver that then calls `set_custom_fmg` gives
the solver its own levels, which `build_transfers` takes first; the
co-located tail is never read.

A lazy tail is not a safe fix: `_custom_mg_coarse_meshes` is read in a dozen
places (the add_fault, adapt and extrude children inherit it, fault_split
copies it, visualisation reads it, possibly on one rank), and the colocation
is collective, so it would have to be resolved at a point every rank reaches,
against a finest level that an inheriting child changes. Options:
1. an explicit switch on the two placement routes to skip adopting the tail
   when the caller supplies its own levels (simple, explicit);
2. drop the driver's `set_custom_fmg` and use the mesh-owned tail, if its
   levels are what the driver wants anyway.
Needed from kaiju: why the driver calls `set_custom_fmg` on a mesh that owns a
tail, and with which levels.
