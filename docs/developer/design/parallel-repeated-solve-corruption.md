# Parallel repeated-FE-solve heap corruption (np ≥ 5)

**Status:** ROOT CAUSE FOUND (2026-05-27) — it is the **`_use_direct_solver`
(lagged MUMPS LU)** path the movers wire in, *not* the Poisson solve, the DM, or
singularity. The UW3 **default GMRES+GAMG solver is clean at np=5** (10/10, even
for the singular `constant_nullspace` case). Fix is mover-local + low-risk.
**Worktree:** `bugfix/parallel-singular-corruption` (off `origin/development`).

> ## ROOT CAUSE (supersedes the #96-class framing below)
> Fixed mesh, np=5, 15 looped solves — measured:
> ```
> nullspace (singular), DEFAULT GMRES+GAMG :  clean 10/10
> dirichlet (non-sing), DEFAULT GMRES+GAMG :  clean 10/10
> nullspace (singular), _use_direct_solver :  crash  5/6   (lagged MUMPS LU)
> ```
> Every repro and every mover wired the Poisson through
> `sm._use_direct_solver` (`smoothing.py:859`): `snes_type=ksponly`,
> **`snes_lag_jacobian=-2` + `snes_lag_preconditioner=-2`** (factor once, reuse
> forever), `pc_type=lu`, `pc_factor_mat_solver_type=mumps`. The lagged-MUMPS
> reuse corrupts the heap over repeated solves at np≥5. The Poisson/DM/nullspace
> are all fine — the **default solver is the proof**. This is why normal UW3
> parallel runs are healthy. The earlier "general / Dirichlet 25%" rates were all
> measured *with* `_use_direct_solver`. The #96 framing below (DMClone cache) was
> disproved (independent DM still crashes) and is kept only for the record.
>
> **Narrowed (np=5, nullspace):** it is **MUMPS itself, not the lagging** —
> non-lagged MUMPS (`snes_lag_*=1`, refactor every solve) still crashes 7/8, and
> un-lagging only the PC crashes 8/8. So `pc_type=lu` + `pc_factor_mat_solver_type=mumps`
> repeated at np≥5 corrupts; the factorisation-reuse is incidental.
>
> **Fix direction:** the movers must **not use MUMPS in parallel**. Use the
> (clean) iterative GAMG path when `uw.mpi.size > 1`, keep MUMPS (lagged, fast)
> for serial. Concretely: gate in `_use_direct_solver` (fall back to
> `_use_iterative_solver` under MPI) or in the movers' `_wire` (choose by
> `uw.mpi.size`). The singular `constant_nullspace` φ-Poisson is clean AND
> convergent with GAMG (default test 10/10 + all iterations OK), so the docstring's
> "GAMG fragile" note is about convergence quality, not crashes. Verify the full
> movers at np=5 with the oracle (in progress). Whether this is a MUMPS bug in
> this build or a UW3 MUMPS misconfig is left open — irrelevant to the fix.

---

## Symptom

Underworld3 solvers that **repeatedly `.solve()` in a loop at np ≥ 5** suffer a
**probabilistic heap corruption** — crash (SIGSEGV / SIGBUS / SIGABRT, signal
varies = heap corruption) **or MPI deadlock (hang)**. This blocks the adaptive
mesh-motion movers (`_winslow_equidistribute` / OT and `_winslow_elliptic` / MA
in `meshing/smoothing.py`), which drive Picard loops of cached solvers.

It is **not** mover-specific. Measured crash/hang rates at **np = 5** (release
arch `petsc-325-uw-openmpi`, MUMPS, 15 looped solves per run):

| solver | operator | rate |
|--------|----------|------|
| pure-Neumann Poisson (`constant_nullspace=True`) | singular | **100%** (8/8) |
| Poisson with a Dirichlet BC | non-singular | **~25%** (2/8) — *also crashes* |
| `SNES_MultiComponent` Hessian recovery (Nc=4, flux from aux φ) | — | ~50–75% |
| `MultiComponent_Projection` (Nc=4, F1=0 mass-only) | — | **0%** clean |

**np = 3 mostly escapes; np ≤ 2 escapes entirely** (why this was missed — almost
all prior parallel UW3 work ran at np ≤ 2). Singularity *amplifies* the rate to
100% but is **not** the root cause: a non-singular Dirichlet solve still corrupts
~25%. So this is a **general** repeated-parallel-FE-solve corruption.

## What it is NOT (ruled out)

`constant_nullspace` object · MUMPS `ICNTL(24)` · the linear solver (MUMPS **and**
GAMG both) · per-solve nullspace re-attach · the NullSpace comm · FE order (P1/P2)
· `n_components > dim` (disproved — see below) · the IS/SF/DM per-solve leak
(identical in the clean Dirichlet path) · DM field sizes · **bare PETSc** (pure
AIJ/DMDA + KSP/SNES + NullSpace is clean at np=5 — needs UW3's DMPlex+FE C
callbacks).

### "Nc > dim" was a red herring
Earlier hypothesis (and the prior handoff) blamed `SNES_MultiComponent` with
`n_components > mesh.dim`. **Disproved this session:** `MultiComponent_Projection`
at **Nc=4 > dim=2 is clean** (0% — it has no flux, F1=0), while the Hessian-row
recovery at **Nc=dim=2 still crashes** (it has a flux from an auxiliary field).
So Nc>dim is neither necessary nor the trigger; the discriminator is the
**assembly path** (flux / no-essential-BC), and ultimately the **#96 shared-state
mechanism** below.

## Root-cause class — NOT the #96 DMClone-shared-cache (disproved 2026-05-27)

Initial hypothesis (and the prior handoff) was the Issue-#96 mechanism: `DMClone`
shares `mesh.dm`'s `DM_Plex` mutable caches by refcount, so a repeated parallel
assembly corrupts the shared state. **Disproved this session** — see "Isolation
attempts". A fully **independent** solver DM (fresh `DM_Plex`, no `DMClone`,
reloaded from the mesh `.h5` + re-distributed, verified layout-identical to
`mesh.dm`) **still crashes 7/8 at np=5**, same as baseline. So the corruption is
**not** the DMClone-shared `DM_Plex` cache, and the #96 "new DM, no clones"
isolation does **not** transfer to this bug.

The corruption is **intrinsic to a single UW3 solver doing repeated parallel FE
solves at np ≥ 5** — the OT mover crashes **5/5 alone** (analytic metric, no other
solver instantiated), so it is not about coexisting solvers either. What remains
coupled to the mesh in a single solve (the new suspects):

1. **The auxiliary vector.** Every `SNES_*.solve()` calls `mesh.update_lvec()`
   then `DMSetAuxiliaryVec(solver.dm, mesh.lvec)` — sharing `mesh.lvec` and, on
   first build, **mutating `mesh.dm`** (`clearDS` + `createDS` +
   `createFieldDecomposition`; `discretisation_mesh.py:1975` `update_lvec`).
2. **The per-solve IS/SF/DM leak.** `-log_view` showed ~92 Index Set, ~74 Star
   Forest, ~46 DM objects leaked per 15 solves. The prior handoff dismissed this
   as benign "because the clean Dirichlet path leaks identically" — but **Dirichlet
   is NOT clean (it crashes ~25%)**, so the accumulating leak is back as a prime
   suspect (heap fragmentation/corruption from accumulating un-destroyed PETSc
   objects at np ≥ 5).

A full isolation per the #96 recipe (independent DM **and** independent
fields/aux-data copied via numpy, never touching `mesh.lvec`/`update_lvec`) was
**not** completed — it requires bypassing the aux-vec sourcing inside the compiled
`solve()` path. That, plus the leak, is the next investigation front.

## The proven fix recipe (from the #96 campaign)

> Completely separate the solver out: **create a NEW dm (NOT a clone)**, add the
> fields the DM needs, and **COPY the values in from `mesh.dm` via numpy** (not
> shared PETSc objects). Establish that the fully-isolated solver is clean, then
> **back off** (re-share progressively) **until it breaks** — that locates the
> minimal sufficient isolation, which is what ships.

### Implementation path (designed; not yet built)

1. **Independent DM, no clone.** The mesh persists its gmsh topology as
   `mesh.name + ".h5"`. Reload it per-solver via `_from_plexh5(...)` → a fresh
   `DM_Plex` (no refcount sharing with `mesh.dm`). **Must then distribute it**:
   the raw reload lands undistributed (all cells on rank 0 — verified by
   `probe_reload_layout.py`), so its partition will **not** match `mesh.dm`.
2. **numpy data bridge.** Because the independent DM's parallel layout differs
   from `mesh.dm`, map all data (coordinates, aux/coefficient fields, the solution
   back) **by coordinate matching via numpy** — not by sharing/copying PETSc Vecs.
   This is the substantial, careful, parallel part.
3. **Replicate the DM setup** the assembly needs on the independent DM:
   `createCoordinateSpace(degree, …)` + `dm_force_coordinate_field`, boundary
   labels (`labelsLoad` + named-boundary patching), FE field + `createDS`.
4. **Gate narrow-first** (a solver/mesh flag): apply only to the mover solvers
   first; generalise to `clone_dm_hierarchy` for all solvers only after
   tier-A/B benchmarking (Solver Stability is Paramount).
5. **Back off to minimal** using the oracle below: once full isolation is clean,
   re-introduce sharing incrementally to find the cheapest sufficient isolation.

### Isolation attempts already tried (this session) — both FAILED
- **Rebuild each clone's coordinate space in `clone_dm_hierarchy`**
  (`createCoordinateSpace` + `dm_force_coordinate_field` on the clone): no rate
  reduction. Reverted.
- **Fully independent solver DM (no `DMClone`)**: `clone_dm_hierarchy` returns a
  fresh DM reloaded from `mesh.name + ".h5"` + `distribute()` (verified
  layout-identical to `mesh.dm` — `probe_reload_distribute.py` shows
  `same_order=True` all ranks, so the aux-vec/section coupling is compatible with
  no remapping). **Still 7/8 crash at np=5** (baseline 100%). Reverted. ⇒ the
  DMClone-shared `DM_Plex` is NOT the culprit; isolating only the DM is
  insufficient. (Repro: `repro_ns_isolated.py` with `mesh._isolate_solver_dm`.)

### Band-aids (insufficient — for the record)
A single-DOF pin or a **near-zero mass term** (`-∇·κ∇φ + εφ = f`, εφ in the **F0
operator**, never in `ps.f` — that breaks the Jacobian) makes the φ-Poisson
non-singular → drops it from 100% to the ~25% **general** baseline. Still not
production-usable, because the general 25% remains. The mass term is the right
*formulation* for a mesh-motion potential, but it does not fix the corruption.

## Reproduction & tooling (all under `~/+Simulations/StagnantLid/`)

- **Repros** (`parallel_corruption_repros/`): `test_ns_loop.py`
  (MODE=nullspace|dirichlet — primary), `repro_hessian_loop.py`
  (MODE=full|row — secondary), `repro_mc_projection.py` /
  `repro_flux_chars.py` (the clean Projection controls),
  `repro_screened.py` (mass-term band-aid), `probe_reload_layout.py`.
- **Oracle:** `rate_test.sh` — runs each config N times at np=5 with a per-run
  **timeout** (the corruption can deadlock, so a plain loop hangs forever) and
  classifies CLEAN / CRASH / HANG. This is the scoring harness for isolation
  candidates. `TIMEOUT=100 NP=5 N1=.. N2=.. bash rate_test.sh`.
- **Run env var:** `UW_NO_USAGE_METRICS=1` (silences the telemetry thread).

### Debug PETSc (built this session)
A minimal `--with-debugging=1 -O0` arch **`petsc-325-uw-openmpi-debug`** is built
(coexists with the release arch; release untouched). Use env **`amr-debug`**
(activation sets the `-debug` arch). **It does NOT reliably reproduce the crash** —
the debug allocator's padding absorbs the overrun (the process completes or dies
silently; `-malloc_debug` guard bytes did not trip). This matches the #96
experience: the debug build "exits before the SEGV". So **the release arch + the
timeout oracle is the working reproduction**, not the debug build. (The 3.24
`petsc-4-uw-openmpi-debug` arch is **ABI-incompatible** with the now-3.25.0 shared
source — do not use it. To rebuild a 3.25 debug arch: a minimal manual
`./configure --with-petsc-arch=petsc-325-uw-openmpi-debug --with-debugging=1
--with-mpi-dir=$CONDA_PREFIX --with-hdf5-dir=$CONDA_PREFIX --download-*=0
--with-petsc4py=0 --with-pragmatic=0 --with-slepc=0 --with-x=0 COPTFLAGS="-g -O0"`
under `pixi run -e amr-debug`, then `make … all`, then build petsc4py + `pip
install .` under `amr-debug`.)

## THE FIX (implemented in `meshing/smoothing.py`, 2026-05-27)

Avoid MUMPS in parallel; keep it for the validated serial speedup. Three changes,
all mover-local:
1. **`_use_direct_solver`** — under `uw.mpi.size > 1`, fall back to
   `_use_iterative_solver` (MUMPS-free GMRES+GAMG / CG+Jacobi). Serial keeps the
   lagged MUMPS LU (the 10× Picard efficiency lever). `elliptic` is now a param,
   forwarded to the fallback.
2. **`_use_iterative_solver`** — the GAMG **coarse** solver was `lu`+`mumps`;
   under MPI it now uses `redundant`+`svd` (verified clean + convergent on the
   singular pure-Neumann coarse). Serial keeps the MUMPS coarse.
3. **`_wire`** (all three movers) — forwards `elliptic` to `_use_direct_solver`
   so the parallel fallback picks GAMG (φ-Poisson) vs CG+Jacobi (mass) correctly.

### Verification (np=5, timeout oracle; baselines in parentheses)
- OT mover, default `linear_solver="direct"`: **clean 6/6** (was 5/5 crash).
- MA mover (`_winslow_elliptic`, original Nc=4 Hessian recovery): **clean 5/5** (was 5/5).
- Public `mesh.OT_adapt(field)` (full metric-density + mover): **clean 5/5** (was crash).
- φ-Poisson `constant_nullspace`, `_use_direct_solver`: **clean 6/6** (was 5/6).
- Nc=4 Hessian recovery loop (`_use_direct_solver`): **clean 6/6** (was 4/4 crash).
- Serial (np=1, MUMPS): unchanged, DONE.
- **Regression:** tier-A (`level_1 and tier_a`) **177 passed, 6 skipped, 0 failed**
  (serial path is bit-identical — the fix only changes the parallel branch).

### Convergence + correctness (not just crash-free)
All three mover solver types **converge** in parallel (positive KSP reason) and
**match the serial MUMPS answer** to ~1e-10 relative (within `ksp_rtol=1e-7`),
via partition-invariant `uw.maths.Integral` diagnostics:
- φ-Poisson (GAMG, singular `constant_nullspace`): converged 29 its, resnorm
  1.1e-9; ∫|∇φ|² serial 7.354489536e-2 vs parallel 7.354489548e-2 (Δ 1.6e-9).
- ∇φ Vector_Projection (CG+Jacobi): converged 8 its; ∫ serial 3.536143815311 vs
  parallel 3.536143815630 (Δ 9e-11).
- Nc=4 Hessian recovery (CG+Jacobi): converged 12 its; ∫ serial 1268.973923874
  vs parallel 1268.973923661 (Δ 1.7e-10).
The full OT mover yields a valid non-tangled adapted mesh in parallel.

### Second change: parallel-correct `_patch_volumes` (the equidistribution source)
`_patch_volumes` (the per-vertex dual area driving the equidistribution metric)
is exactly the **lumped P1 mass diagonal** `M_ii = ∫N_i dV`. The hand-rolled
local `np.add.at` sum **under-counts shared vertices on rank-partition
boundaries** (it never sums the neighbouring rank's incident triangles), so the
parallel grading was systematically too weak (q_min 0.82→0.150 vs serial
0.82→0.108 — under-refined). Fixed by computing it through the **FE mass matrix**
in parallel (`_lumped_vertex_volumes`): `M·1` row-sums = the lumped diagonal, with
PETSc's `localToGlobal(ADD)` doing the cross-rank reduction. Verified:
serial result **bit-identical** to the numpy version (maxdiff 2.6e-18, ordering
preserved), parallel **conserves total area** (∑lumped = mesh area) and the
parallel grading now tracks serial (q_min→0.093, aspect→24.8 ≈ serial 21.3,
vs the under-refined 0.150/15.3 before). Serial keeps the fast numpy path.
Uses petsc4py's bound `DM.createMassMatrix` + `M·1`; annotated `TODO(petsc4py)`
to switch to the purpose-built `DMCreateMassMatrixLumped` (which returns the
lumped diagonal directly with the cross-rank ADD built in) once petsc4py
binds it (it exists in the 3.25 C API but is not exposed).

### Fix 2 (Nc>dim guard + Hessian row-restructure) — REVERTED
Premised on the disproved "Nc>dim" theory. The Hessian recovery crashed only
because it too used MUMPS (`_use_direct_solver`); `MultiComponent_Projection` at
Nc=4 with the default (non-MUMPS) solver was always clean. The MUMPS fix makes
the **original** single Nc=4 recovery parallel-safe (CG+Jacobi), so the
row-restructure is unnecessary and the guard would false-positive. Reverted.

### Open / broader caveat
**MUMPS-in-parallel-over-repeated-solves is unsafe in this build generally** — any
UW3 code setting `pc_factor_mat_solver_type=mumps` (or `pc_type=lu` in parallel)
and re-solving at np≥3 risks the same corruption. This fix covers the movers;
a broader guard/warning (or pinning whether it is a MUMPS bug vs a UW3/PETSc-MUMPS
interface issue — possibly the per-solve IS/SF/DM leak being MUMPS factorisation
objects) is worth a follow-up. The `petsc-custom/build-petsc.sh` worktree-local
`UW_PETSC_DEBUG=1` patch must be reverted before any PR (the debug arch is already
built).

## Next steps (DM isolation is ruled out — pursue the aux-vec / leak fronts)

1. **Per-solve leak.** Re-run `-log_view` at np=2 on `test_ns_loop` (both modes)
   and locate the un-destroyed IS/SF/DM allocations per solve (in the `solve()`
   path and/or `update_lvec`'s `createFieldDecomposition`). Plug them and re-score
   with `rate_test.sh`. Test whether the crash rate scales with solve count /
   `NREP` (accumulation signature).
2. **Aux-vec / `update_lvec`.** Try a complete isolation that also avoids
   `mesh.lvec`: build an independent aux vec with numpy-copied data and bypass
   `update_lvec`'s mutation of `mesh.dm` (`clearDS`/`createDS`/`createFieldDecomposition`
   every first-build). This needs a hook in the compiled `solve()` aux-vec
   sourcing.
3. **ASan.** Because the debug build absorbs the overrun, a definitive pinpoint
   likely needs an AddressSanitizer PETSc build (`-fsanitize=address`, fiddly via
   Python on macOS — needs the ASan runtime preloaded) run at np ≥ 5.
4. Score every candidate with `rate_test.sh` (CLEAN/CRASH/HANG); benchmark
   tier-A/B before any change to the shared solve path (Solver Stability is
   Paramount).
