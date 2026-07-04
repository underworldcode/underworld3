# Free-surface convection — session findings (2026-05-13)

Diagnostics + remediation experiments for SLCN+CN ringing in
`docs/developer/design/_phase_i_fs_convection_zoo.py`. Runs labelled
v4-v19; output snapshots in `output/convection_zoo_snapshots_*` and
work logs in `/tmp/conv_*.csv`.

## What works (known-good baseline)

- bdf2_sl + Anderson m=5 + ALE correction at Ra=10⁵, ρg=10⁵, no Nitsche, P3 T,
  uniform mesh res=20: clean dynamics through ~step 27, then ringing at step 28-30,
  catastrophic blowup at step 33-35.
- Same params with **end-result T clip** (`--clip-t`, no other changes): runs
  bounded for 35 steps with recurring small ringing events every ~2 steps,
  but never catastrophic.
- Same params with **launch-point clip** (env var `UW_LAUNCH_CLIP=1`,
  clips `psi_star[0]` before adv_diff.solve): **dramatically better** than
  end-clip — post-solve overshoots drop from O(2) to O(0.05),
  bad-DOF count from O(100) to O(7). Fewer mass-conservation artefacts.
- Refined-BL mesh alone (`--outer-refine-factor` 1.3 or 2.0) is **WORSE**:
  triggers ringing earlier (step 20 vs 30) and snowballs faster, because
  fine cells get crushed by accumulated surface deformation.
- Higher polynomial degree T (`--t-degree 5`) is much WORSE: catastrophic
  at step 28 with |ΔT|=35.

## Diagnosis (definitive)

The contamination originates **inside `adv_diff.solve()`** as Crank-Nicolson +
BDF2 on P3 Lagrange basis ringing on the sharp thermal boundary layer in
deformed elements (Runge phenomenon at the SL trace-back interpolation,
amplified by CN+BDF2's non-monotone implicit step). Confirmed by per-step
T extrema instrumentation: T transitions from strict [0,1] before
adv_diff.solve to [-0.18, +2.30] after, with NO restore-call activity
(the kdtree restore is innocent). The bug is in the discretisation, not
in any boundary handling.

Reproducibility: from-scratch is bit-deterministic (v3 ≡ v4 to all digits)
but restart-from-checkpoint at the SAME state does NOT reproduce —
confirming non-checkpointed internal state (Anderson buffers, KSP
preconditioner caches) carries part of the bifurcation path. This is
expected for a marginal numerical instability with sudden onset.

## What doesn't work

- **Mesh adapt** via `mesh.adapt(uniform_metric)`: silent C-level crash in
  PETSc/MMG in this environment. Worked around with kdtree-NN
  shape-rollback as a coarse stand-in (only delays ringing by 1 step
  because element distortion is not the carrier — it's the discretisation).
- **Boundary-layer refinement without smoothing**: the graded BL cells
  get crushed by the radial-velocity diffuser within ~15 steps because
  the diffuser smooths radial displacement but does nothing tangential.
- **Winslow smoother inside `deform_by_inc`** (per RK-stage / Picard-iter):
  Lagrangian DOFs ride the smoothing → SLCN trace-back samples T at moved
  positions → spurious mode-5 amplification appears in step 1.
  Currently reverted; see "User recommendations" below.

## User recommendations (carry forward into next session)

> "Go back to what works, and then introduce the mesh smoothing AFTER all
> the other mesh processing is done. This can be an infrequent step (or a
> snapshot / backtrack when we get that working)."

Implemented as: `--winslow-every-n-steps N` + `--winslow-iters K`. When both
are non-zero, Winslow runs once every N completed timesteps, K Jacobi
sweeps per call, AFTER all of the step's Stokes/Picard/SLCN/adv_diff
work is done. Operates on interior nodes only (boundary pinned via DMPlex
`Upper`/`Lower` labels). Implementation uses DMPlex edge iteration for
adjacency + scipy.sparse Mat-Vec for the Jacobi sweep — **parallel-safe via
DMPlex local point chart**, but the scipy sparse Mat-Vec needs to become
PETSc Mat-Vec for full parallel scaling.

Other carry-forward suggestions from the user:
- **Use `read_timestep` for plot scripts**: don't rebuild meshes from gmsh
  in plot scripts. Pattern is `mesh = uw.discretisation.Mesh(filename=mesh_h5)`
  + `var.read_timestep(...)`. RBF interpolation handles any DOF-ordering
  mismatch automatically.
- **Spectral filtering is not parallel-safe**: replace the Fourier
  `disp_internal → inc_fn` step in `deform_by_inc` with a per-vertex
  representation eventually. Tried it via MeshVariable BC at the diffuser;
  this broke step-1 dynamics for unclear reasons (T evolved to a
  fully-developed mode-5 pattern at step 1). Reverted to Fourier for now;
  needs deeper investigation before next attempt.
- **Weighted Winslow for adaptive density** (no node addition): same Jacobi
  sparse matrix, but row-weights non-uniform (smaller weight to neighbours
  far from a feature → vertices drift toward that feature). Simpler than
  full h-refinement, immediately implementable on top of current code.

## Files changed (worktree state at end of session)

- `docs/developer/design/_phase_i_fs_convection_zoo.py`:
  - Added `winslow_smooth_interior(state, n_iters, alpha)` and
    `_winslow_build_adjacency(mesh)` utilities (DMPlex edge iteration
    + scipy CSR Jacobi)
  - Added per-step T-extrema instrumentation (T_DIAG lines)
  - Added per-call kdtree-restore instrumentation (RESTORE lines)
  - Added clip path: `--clip-t` (end-result clip) and env-var
    `UW_LAUNCH_CLIP=1` (launch-point clip on `psi_star[0]`)
  - Added `--t-degree N` (default 3)
  - Added `--outer-refine-factor F` (default 1.0)
  - Added `--remesh-at-step N` (kdtree-NN shape-rollback proof of concept;
    not a true remesh — `mesh.adapt` failed in environment)
  - Added `--winslow-iters K` + `--winslow-every-n-steps N`
  - `deform_by_inc` is at the **original Fourier+diffuser baseline**
    (Winslow removed from inside)
- `MEMORY.md`: added `feedback_env_flag_hacks_need_todo.md` —
  every `os.environ.get("UW_*", ...)` toggle gets a paired
  `TODO(env-flag-hack)` comment for findability.

## Worth running first thing next session

A clean baseline matching v5 (uniform mesh, no Winslow, no clip) to
confirm no regressions from all the editing. Then v5 with launch-point
clip enabled + an end-of-step Winslow every 5-10 steps. Compare to
v5 alone to see whether the periodic Winslow helps without breaking
anything.

```bash
# Sanity baseline
pixi run -e amr-dev python -u docs/developer/design/_phase_i_fs_convection_zoo.py \
  --schemes bdf2_sl --n-steps 35 --Ra 1e5 --rho-g 1e5 \
  --nitsche-penalty 0 --stokes-tol 1e-7 --ale-correction --anderson-m 5 \
  --capture-every 1 --snap-suffix "_sanity_baseline" \
  --work-log /tmp/conv_sanity_baseline.csv

# With launch-clip + periodic Winslow
UW_LAUNCH_CLIP=1 pixi run -e amr-dev python -u docs/developer/design/_phase_i_fs_convection_zoo.py \
  --schemes bdf2_sl --n-steps 35 --Ra 1e5 --rho-g 1e5 \
  --nitsche-penalty 0 --stokes-tol 1e-7 --ale-correction --anderson-m 5 \
  --winslow-iters 5 --winslow-every-n-steps 5 --winslow-alpha 0.5 \
  --capture-every 1 --snap-suffix "_launchclip_winslow_n5"
```
