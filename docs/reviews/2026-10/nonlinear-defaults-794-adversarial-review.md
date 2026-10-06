# Adversarial review — nonlinear solver defaults and the declared plastic regularisation (PR #794)

Branch `bugfix/picard-warmup-791` vs `development`.

Method: two rounds.
- Round 1 (at a8aa68c0): three independent reviewers, one each on the solver control flow, the constitutive models and η_reg, and the solve record, docs and tests.
- Round 2: one reviewer on the fix commit 8010d335.

Every reviewer was told to break the change and to verify each finding by probe. Probe scripts are in `~/+Simulations/review_794/`. All findings below are fixed on the branch (8010d335, 6535935b) unless listed as open.

## Round 1 — blockers

- **Cold-start verdict was per rank** (`_solution_is_trivially_zero` read `u.vec.norm()`, a LOCAL vector).
  - At np=2, with the initial guess non-zero in one corner, rank 0 took the warm-up and rank 1 did not. Rank 1 failed with `MPI_ERR_BUFFER`; rank 0 spun until killed.
  - The verdict is now a reduced any-nonzero test. `tests/parallel/test_1081` checks it; the old verdict disagrees at np=2 and np=3.
- **Krylov restart 100 never took effect for Stokes.** It was written inside the `strategy` setter, which construction never runs; the live KSP and the report both showed 30. It is now set once at construction as a managed option.
- **`picard` was inserted second** in the Scalar/Vector/MultiComponent `solve()`. Five subclasses pass `_force_setup` positionally, so `_force_setup=True` became `picard=1`: a warm-up and a blended compile on a linear AdvDiffusion solve. `picard` is now the last argument.
- **`solve(picard=N)` raised on rotated free-slip** under the new default. It now stages N Picard iterations, then Newton (power law: 17 its, error 1.2e-8 against the essential-BC answer).
- **The default yield anchor "yield"**, sqrt family, δ 0.1, had two defects:
  - every unyielded cell was 1/(1 − δ/2) stiffer: η/η_ve = 1.0499 at f → 0, and a box that never yields moved 4.75%;
  - the yielded tangent was negative: τ/τ_y peaks at 1.033 at f = 1.2, and dτ/dε̇ = −0.029 at f = 1.5.

  The maintainer chose sqrt / "onset" / δ 0.1: η(0) = 1, τ(f=1) = 0.955 τ_y, tangent ≥ −1e-4.
- **A YAML colon in the guide front matter** dropped `nonlinear-solver` from `uw.capabilities()`; `test_0030` failed 2 tests.
- **An uncommented `except: pass`** in `yield_continuation.py` failed the Charter S4 style gate.

## Round 1 — should-fix

- **Linearity test on every warm solve.** The symbolic test ran before the cheap cold-start test: 3.7 s per warm step on a 16×8 VEP box, and 5.8 against 2.2 s/step overall. The cold test now runs first, and the facts are cached.
- **The warm-up ran outside the solve's contracts**: before the update hooks and the wall-clock guard, outside the `estimate_difficulty` cap (cap 1 ran 2 iterations), and missing from the report (`fnorm0` was 4.04e-2 against a cold F of 0.120). It now runs inside `_snes_solve_with_retries`.
- **Newton after the warm-up measured rtol from the post-warm-up residual**, and stagnated on SNORM on near-linear boxes (2 its without the warm-up). It is now anchored at `tolerance`·‖F(u_cold)‖.
- **Changing the tangent after a solve did not recompile**, so the report described a tangent that was not running (Poisson False → True: 9 its, against 3 for a fresh solver). α was one process-wide container shared by every solver. Both setters now mark a rewire, and α is unique per solver.
- **Ladder defects:**
  - a retry restarted at the largest converged m (1e6), not the last good rung;
  - a zero or Piecewise η_reg gave zoo or stale zeros, and a ladder that did nothing reported `converged=True`;
  - the entry multiplier was absolute;
  - each rung recompiled (5.2 s against 0.23 s);
  - VEP failed with an AttributeError instead of a clean refusal;
  - the probe variable was named from `id(solver)`, which differs between ranks.
- **Zero test spelled three ways.** `Float(0.0) == 0` is False in SymPy 1.14, so `plastic_rate_strengthening = 0.0` compiled a zero term and suppressed the warning.
- **Warning hygiene.** The gauge warning fired twice per solve from an internal frame. The Drucker-Prager warning fired under the frozen tangent and for a p-dependent η0 with a constant yield stress.
- **The record had gaps.** Rotated reports carried no config. The transcript dropped the regularisation. Guides and docstrings still described the old defaults, and the new controls were undocumented.
- **Tests that could not fail.** test_1069 asserted only that a "restart" key existed (it was 30). test_1071's string checks could not tell 0.05 from 0, and it never checked that the ladder ran. test_1018's rotated frozen-tangent coverage had silently become Newton.
- **Duplicated code.** The warm-up policy was pasted 4 times and had drifted: `"continuation"` ignored `picard=N` in three classes.

## Round 2 (on 8010d335)

- **blocker**: the form-fact cache went stale across a rebuild. Viscous → Drucker-Prager, solved warm, recorded `pressure_in_rheology=False` with no warning. A flux made nonlinear skipped the cold step. The cache is now dropped wherever a rewire or a setup completes.
- **blocker**: test_1018 still expected the removed `NotImplementedError`. It now asserts the staging.
- **should-fix**: the rotated path discarded the warning notes.
- **should-fix**: `KSPSetType` on the single-field FMG route reset the live restart to 30 while the report said 100. The restart is re-applied, and test_1069 now reads it from the live KSP.
- **should-fix**: the ladder's early exits skipped its warning. A failed first rung marked non-solution fields as converged.
- **nit**: the duplicated homotopy check, the warm-up warning frame, the δ-march anchor docstring.

## Attacks that failed

- **Linearity guard:** correct on 19 cases (Poisson k=1/k(T)/k(u), Stokes η=1/η(T)/η(|v|), VE, VP, DP, VEP, power law, units-aware viscous and VP, projections, AdvDiffusion, Navier-Stokes).
- **Warm loop:** no warm-up on steps 2+. Nonlinear Poisson matches Picard-throughout to 7.7e-11 (3 vs 9 its).
- **Eisenstat-Walker:** the warm-up iterate is identical with EW on and off.
- **`newton_pressure_coupling=False` with a p-independent rheology:** identical iterations, dv = 0, ‖J_T − J_F‖ = 0.
- **Units:** `uw.quantity(1e20, "Pa*s")` with η_ref 1e21 gives deep-yield η = 0.1005. The default `0 [Pa s]` omits the term.
- **Ladder:** ends at exactly 1.0 on a clean run, after a mid-ladder failure, and after a failure at m = 1. It refuses NaN through a separate flag reduction, and is bounded by `max_steps`.
- **Names:** two VP models get distinct `m_reg` atoms. `describe()` names η_reg and m_reg.
- **Record:** JSON-safe. A transcript round-trips through `uw.Transcript`. The snapshot creates no NPC or line search and prints nothing.
- **Probe cap:** `picard` 2/5/3 with cap 3 gives 3 iterations. Tolerances are restored.
- **Anchor:** never loosens a solve (boundary-driven DP, VP, n = 0.5 and 3).
- **Rank uniformity:** the ladder runs at np=3. Every new collective is reached by every rank.
- **Restart:** reaches the live KSP for default, fast, robust, gamg and refinement=2 FMG Stokes, and for plain, gamg and fmg Poisson. A user value is kept. The composed solver keeps its 200.

## Open (not fixed on this branch)

- **#817 is misdiagnosed.** `Stokes_Constrained` fails on its second Newton iteration whatever the tangent:
  - with η(1 + 1e-6 ε̇_II): automatic warm-up DIVERGED_DTOL; `picard=-1` DIVERGED_MAX_IT oscillating; `consistent_jacobian=False` identical;
  - the linearity guard only hides this for linear fluxes.
- **VEP under the Newton default.** The guide recommends the frozen tangent for VEP, but `test_1052`'s loading through yield converges under the new default (nl = 1 per step).
- **Retries in the record:** with divergence retries, the report keeps the warm-up plus the last attempt (pre-existing behaviour for retries).
- **mpi4py reductions:** `_solution_is_trivially_zero` and `_masked_max_ratio` reduce through `uw.mpi.comm` (mpi4py), which the library already does elsewhere. A PETSc global-Vec norm would serve.
