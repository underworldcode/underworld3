# Adversarial review: `feature/ve-stretch-split` (stretch_in_momentum, preconditioner_flux)

**Date**: 2026-10-18. **Branch**: `feature/ve-stretch-split`, base `05d84a02`.
**Reviewers**: four, in parallel, each against a build of the base — correctness
and parallel; numerics and timing; tests, contracts, documentation and issues;
interactions with open PRs and with `development` since the branch point.
**Record of the work**: wiki page *Viscoelastic stretch at large steps*.

## Findings and what was done

| # | finding | severity | done |
|---|---|---|---|
| 1 | `preconditioner_flux="model"` did nothing with rotated free-slip (that path preconditions with Qᵀ J Q) and on `NavierStokesSLCN`/`Swarm` (flux not built through `stress_deviator`), while the solve report recorded `"model"` | should-fix (#477 class) | refused with `NotImplementedError` on both paths; the report records `"model"` only when the block was built (`_preconditioner_flux_applied`) |
| 2 | `stretch_in_momentum` other than `"implicit"` did nothing under BDF, outside the deformation step, and on `NavierStokesSLCN`/`Swarm`, while the report recorded the form | should-fix | BDF now honours the form; a non-default form needs `convected_step="deformation"` (constructor and setter refuse otherwise); `SNES_NavierStokes` refuses it |
| 3 | default ETD + deformation path not bit-identical to the base: a factored `L Lᵀ` changed the generated C (round-off, 1e-14 in velocity) | should-fix | base operand order restored for `"full"`; 16 configurations now bitwise identical in velocity, pressure, stored stress and Newton counts |
| 4 | merging `development` (#744, discrete adjoint): the Pmat alias was taken before the adjoint transpose, a `KeyError` for any non-symmetric tangent (3 tests in `test_0019`) | blocking on merge | development merged; the alias is retaken after the transpose; the adjoint preconditions with the transposed Jacobian; `test_0019` passes |
| 5 | `preconditioner_flux` setter forced a full DM teardown and a cold start | should-fix | rewires the kernels only; the test asserts the solution survives |
| 6 | the preconditioner block ignored the model's smooth surrogate (`flux_jacobian`, `set_jacobian_F1_source`) | should-fix | refused when a surrogate is active (no preconditioner counterpart exists) |
| 7 | duplicated G2/G3 loops (Charter §5) | should-fix | `_flux_velocity_kernels` |
| 8 | tests that could not fail: no test caught `"linear"` == `"implicit"`; the switch test switched before the first solve; `"tangent"` untested | should-fix | exactness tests: the flux is quadratic in L under `"implicit"` and not under `"linear"`/`"tangent"`; `"tangent"` equals `"implicit"` at L = Lⁿ; the compiled residual is quadratic before a mid-run switch and not after |
| 9 | docstrings: "symmetric positive-definite whatever the stress" false with yield and on Navier-Stokes | should-fix | qualified |
| 10 | `.data` write in a new test (Charter §7); non-raw docstrings with `:math:` | nit | `.array`; raw docstrings (the constructor docstring's pre-existing escape warning goes too) |
| 11 | documentation numbers: Pakdel–McKinley M reported as 3.6 (the value nearest the wall; inside the layer it is 24–30); backward Euler's cost 9 % (it is 14.5 %); `"tangent"` "stalls where implicit does" (at Δt 1/120 it went unstable earlier); the conformation check does not cover the Eulerian history; the yielding argument has no run behind it | should-fix | corrected in `stress-transport.md` and on the wiki |
| 12 | `plasticity-solvers.md` names the VEP family and had no row for the stall | should-fix | failure-mode row added |

## Attacks that failed

- Timing: concurrent identically configured Waters–King runs, base vs branch, per-step 9.0–9.3 s vs 9.0–9.1 s, JIT within noise, results bitwise identical. The slow tier-C tests (test_1060/1061, 53 and 68 min) cost the same on the base: the durations written in those files are stale for this machine, not a regression.
- The new forms against an independent calculation: implicit − linear equals the dropped O(Δt² L²) terms to 1.5e-7, implicit − tangent the same in (L − Lⁿ) to 3.3e-8; `history_flux` identical for every form.
- `"model"` leaves Newton unchanged: identical iteration counts, residuals to six figures, velocity to 1e-15 (Stokes and Eulerian NS); the Stokes preconditioner block is symmetric to 5e-17.
- Parallel: no collectives in the new code; both test files pass on 2 ranks.
- Named-expression restore after the preconditioner build, exception safety of the swap flag, determinism, CI reach (`tests/test_107*`), the capability-guide and solver-description contracts.

## Interactions

- **#800** (forward-from-nodes, FENE-P) edits the same constitutive code: five hunks conflict. It lands first; this branch then takes four hand resolutions (the strain-form objective term and spring factor in the ETD and BDF branches) and routes #800's `_theta_rule_flux` through `stress_deviator`, or `"model"` becomes a no-op on `uw.systems.NavierStokes`.
- **#744** (on development): handled above. **#846**, **#677**, **#848**, **#850** inherit only that conflict. No other open PR or merged commit duplicates or contradicts this work.

## Issues adjudicated

- **#768** (wall-strain-rate timestep, conformation health check): still live in part — the check refuses the Eulerian SUPG history.
- **#755** (Schur scaling omits solvent and DEVSS viscosities): still live; this branch touches only the velocity block of the Pmat.
- **#806** (solver introspection of the configuration): already fixed on development (the solve report carries snes, ksp, line search and tangent); this branch extends it.
- **#827** (PETSc banner from the velocity preconditioner check): still live here; PR #850.
- **#823** (unwrap cost): probably fixed (memoised unwrap); needs its timing probe before closing. This branch adds one unwrap per setup when `"model"` is set.
