# Restore exactness and the step ratio: adversarial review record

Branch `bugfix/snapshot-restore-exact` on `development` (`45fc6120`). One round of three
independent reviewers (restore semantics; the step ratio; tests, style and contracts),
run against the installed build of `e30f5777` and `737b4e15`, with a build of the base
commit for old-vs-new comparisons. Fixes in `25129756`.

## What the branch does

| Change | Why |
|---|---|
| A restore moves the mesh only when its coordinates changed | the move bumped the geometry version and every solver rebuilt; 1.5e-5 in velocity on a viscoelastic channel |
| Solver step state (`SolverStepState`: warm-start flag; on the Eulerian NS and advection-diffusion solvers the last step length, the rate `estimate_dt` reads, the extrapolation's previous step) captured and restored, in memory and on disk | a regretted step set how the next one ran |
| The characteristic trace's step state (valid velocity levels, step lengths) travels with the DDt state and the swarm payload | the mid-time velocity of the next trace-back depends on it |
| A restore replaces each GAMG preconditioner and drops the rotated free-slip workspace | GAMG keeps its hierarchy and eigenvalue estimates through `PCReset`; a regretted solve left ~2e-8 behind |
| NS advecting velocity `u^n + r(u^n - u^{n-1})`, trace-back mid-time velocity `v^n + (r/2)(v^n - v^{n-1})`, `r = dt_n/dt_{n-1}` | both were written for equal steps and dropped to first order on every step change |

A regretted step leaves no trace bit for bit (AdvDiffusion SLCN, viscous and
viscoelastic NS, custom-FMG Stokes, rotated free-slip Stokes; serial and 2 ranks).
Against a never-restored run the difference is the warm multigrid: 2.5e-8 relative on
the AdvDiffusion test (8.4e-7 before), 5.5e-7 in velocity on the viscoelastic channel.
Constant-step runs are bit-identical to `development`.

## Findings and their state

- **Blocking — a restore crashed every solver on code-set multigrid transfers**
  (`set_custom_fmg`, adapt-on-top): `PCReset` on a PCMG drops the transfers and the next
  setup calls `DMCoarsen` ("Grid adaptor mmg not registered"). Fixed: only GAMG is
  replaced. Test `test_a_restore_keeps_multigrid_transfers_set_in_code` (fails on `737b4e15`).
- **Swarm-owned trace not captured**: a regretted swarm step left its step length
  (5.6e-5 in position). Fixed; `test_a_restore_brings_back_the_step_length_of_a_swarm_trace`.
- **Rotated free-slip workspace survived a restore** (~1e-15; its fast-path hint can steer
  a nonlinear run). Fixed; `test_a_restore_discards_the_rotated_free_slip_workspace`.
- **Disk restore ignored solver state** (in-process `load_state(path)` kept the regretted
  step's `last_timestep`, `previous_timestep`). Fixed: written and read with the snapshot;
  `test_a_disk_snapshot_carries_the_solver_step_state`.
- **A float32 / sympy timestep made the trace record unreadable from disk** (`KeyError`).
  Fixed: non-dimensional floats, tolerant read.
- **PETSc error block printed on every restore** of a never-set-up field split. Fixed.
- **Tests**: the end-to-end viscoelastic test guards only the solver step state and took
  213 s (now level 3, two steps; the rest moved to a viscous channel, seconds each);
  `test_0008`'s 1e-5 floor passed the old 8.4e-7 (now 1e-7 relative); level-1 NS solve
  (now level 2); the step ratio was untested through any caller (now through
  `swarm.advection`).
- **Docs and dead code**: five statements of the equal-step formula; the dead
  `SemiLagrangian._midtime_velocity_expr`; one comment and two guide passages. Fixed.
- **Not changed, with reason**: the trace record nested in the DDt state stays a plain
  dict (the disk snapshot writes state field by field and cannot write a nested
  dataclass; commented); `_DDtCoreState` keeps schema 1 (the new field has a default and
  old files load — checked both ways); a Schur inner KSP is not reached by the
  preconditioner discard (UW never configures one); a capture taken before the first
  solve differs at round-off (1e-15) because objects built on first use cannot be un-built.

## Attacks that failed

- PC discard on standard paths: GAMG Stokes 40/8 iterations, refined native-mg Stokes 3
  iterations, Poisson GAMG 7 and FMG 3 — same counts as never-restored; refined mg Stokes
  bit-identical.
- Mesh deformed on one rank only between save and load, np=2 and 3: every rank moves, no
  hang, coordinates exact.
- Old disk snapshots read by the new code and new ones by the old code.
- Step ratio through every caller (nodal, integration-point and forward SL histories,
  AdvDiffusion SLCN with the shared trace, swarm-owned trace): mid-time displacement error
  1e-15 after the first step; dimensional timesteps give the right ratio.
- Second order with alternating steps h, h/2: swarm rates 1.98–1.99 (development
  1.05–1.17).
- Constant steps: AdvDiffusion SLCN order 2, NS orders 1 and 2 (with Picard), VE Stokes
  order 2 bit-identical to development; `advection="implicit"` unaffected.
- Territory suites (42 files, `-n 8`): 412 passed on the branch, 406 on development, the
  difference being the new tests. Level 1: 2045 passed; level 2: 1122 passed.

## Issues adjudicated

- #773 (snapshots did not restore parameters) — fixed on development before this branch;
  flagged on the issue with the probe.
- #296 (a deform invalidated a snapshot) — fixed on development; flagged with the probe.
- #791, #817 (Picard warm-up) — still live; this branch only restores the flag that
  decides whether it runs.
- #742 (`psi_star[0]` semantics), #690 and #809 (refined-mesh reload) — still live, not
  touched.
