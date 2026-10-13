# Exponential Integrator for VE / VEP Constitutive Updates — Implementation Plan

**Status**: **ETD-1 ships as the recommended default** (2026-04-29, 27 commits). ETD-1 reproduces BDF-1 essentially exactly on the deep-yield TI killer test (σ_∥ peak 1.04·τ_y, |u_y| peak 0.0320, SNES 1.8 mean iters — all identical to BDF-1) AND inherits ETD's analytical exponential factor for the linear-relaxation part. Phase B (ETD-2, single α/φ), Phase D (per-component split), and Phase E (hybrid BDF/ETD) remain on the branch as instructive failures — they don't ship.

**TL;DR**:
- **The lesson**: the drift/blow-up on VEP+yield is order-driven, not algorithm-driven. **First-order methods (BDF-1, ETD-1) are L-stable and damp the high-frequency modes that plastic yield transitions excite**; higher-order methods (BDF-2, ETD-2 lumped/split/hybrid) preserve those modes and let them grow. Recognising this collapses the whole "ETD doesn't work for fault mechanics" narrative — it's *higher-order* ETD that doesn't work, same as higher-order BDF.
- **Production recommendation**: `integrator='etd', order=1` for everything. Single-step like BDF-1, no forcing-history mesh variable, fully L-stable, with the analytical exp factor for the linear part. Killer-test trajectory **byte-identical to BDF-1** in σ_∥ and |u_y|; ~5% slower wall-clock.
- **Higher-order ETD on smooth VE** (no yield): ETD-2 still beats BDF-2 by 4.3× on `bench_ve_harmonic`. Available as `integrator='etd', order=2` for users who know their problem is fully VE.
- **Higher-order anything on tight-yield TI**: don't use. BDF-2, ETD-2 lumped, ETD-2 split + lag + cap, ETD-2 hybrid — all show drift or blow-up of various flavours.

**Branch**: `feature/exp-integrator-investigation`

**Provenance artifacts**: the experiment scripts (`_exp_integrator_*.py`,
`_phase_*.py`, `_plot_phase_*.py`, `_exp_jury_rig_minimal.py`) and result
images (`exp_integrator_*.png`) cited throughout this document live in
`experiments/exp-integrator/` (moved from this directory 2026-07, DOC-06).
The raw `.trace.txt` solver logs were deleted — they are reproducible from
the scripts and preserved in git history.

**API (production)**:
- `ViscoElasticPlasticFlowModel(unknowns, integrator='etd', order=1)` — ETD-1 (first-order). **Default-recommended for new code** — BDF-1 stability + analytical exp factor for the linear-relaxation part.
- `TransverseIsotropicVEPFlowModel(unknowns, integrator='etd', order=1)` — same, TI variant.
- `integrator='bdf'` on the same classes (with `order=1` or `2`) — production default, unchanged behaviour. Same accuracy class as ETD-1 but with rational rather than analytical relaxation factor.
- `integrator='etd', order=2` (Phase B ETD-2) — second-order, accurate on smooth VE (4.3× better than BDF-2 on `bench_ve_harmonic`); **avoid in VEP+yield regime** (catastrophic σ/u runaway documented in lessons #7, #9).
- Sibling `MaxwellExponentialFlowModel` / `TransverseIsotropicMaxwellExponentialFlowModel` survive as thin aliases for backwards compat.

**API (experimental — investigative, not for production)**:
- `TransverseIsotropicVEPSplitFlowModel` (Phase D): per-component split with τ-cap. σ enforcement OK, `|u_y|` ratchets.
- `TransverseIsotropicVEPFlowModel(integrator='hybrid', fault_weight=...)` (Phase E): spatial blend. σ enforcement OK, `|u_y|` drifts.
- Both retained on the branch for reference; docstrings marked EXPERIMENTAL.

---

## TL;DR

For Maxwell-type viscoelasticity $\dot\sigma + \sigma/\tau = \mu\dot\gamma$, integrate the relaxation operator analytically and approximate only the forcing:

$$\sigma^{n+1} = \alpha\,\sigma^n + \mu(A\,\dot\gamma^{n+1} + B\,\dot\gamma^n)$$

with $\alpha = e^{-\Delta t/\tau}$, $\varphi = (1-\alpha)\tau/\Delta t$, $A = \tau(1-\varphi)$, $B = \tau(\varphi-\alpha)$.

Numerically validated: **5–12× more accurate than BDF-2** at small Δt, **decisively better at Δt ≈ τ** (where BDF-1/2 over-damp to near-zero output), **structurally avoids the BDF-2 multistep instability** seen in TI-VEP + spatial yield_stress (no second history term to amplify through the autodiff Jacobian).

The integrator stores one slot of σ-history *and* one slot of γ̇-history. Yield handling via standard return-mapping. The DDt class hierarchy already supports multiple parallel integrator coefficient sets (`_bdf_coeffs`, `_am_coeffs`); adding `_exp_coeffs` and an optional forcing-history storage stream is a peer extension, ~200 lines.

---

## Implementation phasing

### Phase B — UW3 prototype (next session, est. 3–5 days)

**Goal**: Match BDF-2's `bench_ve_harmonic` accuracy (1.34e-3) with single-step exponential, in a clean implementation.

**Tasks** (in order):

1. **Resolve UWexpression-to-JIT propagation** (~half day)
   - The Phase B jury-rig (`_exp_integrator_uw3_jury_rig.py`) hit a JIT propagation snag: setting `cm._exp_alpha.sym = X` per step doesn't reach the JIT-compiled flux. The BDF path's `_bdf_c0..c3` *do* propagate via `_update_constants()` — replicate that mechanism for `_exp_alpha`, `_exp_phi`.
   - Likely fix: subclass-level `_update_constants` or piggyback on the existing constants-manifest registration in `SolverBaseClass`.

2. **Extend `SemiLagrangian` DDt with exponential integrator** (~1 day, ~200 lines)
   - Add `_exp_coeffs = _create_exp_coefficients(...)` parallel to existing `_bdf_coeffs`/`_am_coeffs`
   - Add `with_forcing_history=False` constructor parameter; when True, allocate `forcing_star` MeshVariable and wire projection-snapshot machinery (mirror what's done for `psi_star`)
   - Add `update_post_solve` branch that calls `_update_exp_values(dt, tau_eff)` and projects the current strain rate into `forcing_star[0]` (use `SNES_MultiComponent_Projection` — already used for VE-Stokes' tau projection)
   - Add `exp_history_term()` peer method to `bdf()` and `adams_moulton_flux()`

3. **Add `MaxwellExponentialFlowModel`** (~half day, ~150 lines)
   - Sibling of `ViscoElasticPlasticFlowModel`. `requires_stress_history = True`, but the auto-DDt creation path uses `with_forcing_history=True` instead of `order=k`
   - Stress: `σ = 2η(1-φ)·ε̇ + DFDt.exp_history_term()`
   - Yield handling: the `viscosity` property wraps with softmin/min as today, replacing η(1-φ) where it appears
   - Lagged-τ: each `_update_constants()` call pulls τ_eff from the most recent post-solve projected stress and uses it for next step's α, φ, A, B

4. **Validate on existing benchmarks** (~half day)
   - `bench_ve_harmonic` — must match BDF-2's max\|err\| = 1.34e-3 at peak-start IC, or be stricter
   - `bench_ve_square_vardt` — must match BDF-2's accuracy under variable Δt
   - `bench_vep_square` (Min mode) — peak \|σ\| within 1% of τ_y, matching the snapshot-fix BDF-2 baseline
   - All 20 existing VE/VEP regression tests still pass

5. **The killer test** (~half day)
   - `bench_ti_vep_harmonic` at θ ∈ {0°, ±15°}, τ_y ∈ {0.15, 0.30}, with the spatial yield_stress field
   - **Decision gate**: peak \|σ_xy\| must stay bounded (≲ 1.1·τ_y in fault zone, ≲ A_∞ in bulk) for all 6 (θ, τ_y) combinations. BDF-2 currently produces 10⁸ blow-up here; exp should run cleanly. This is the empirical proof of the structural argument.

### Phase C — Particle / Lagrangian extension (later session)

`Lagrangian_DDt` and `Lagrangian_Swarm_DDt` are siblings of `SemiLagrangian`; they already share the BDF/AM coefficient API. Mirror the Phase B changes:
- Add `_exp_coeffs` and `exp_history_term()`
- Add forcing-history slot (a swarm variable in the `Lagrangian_Swarm` case)
- The integrator-method API is storage-agnostic; nothing the constitutive model calls needs to change

### Phase D — Per-component (α_⊥, φ_⊥)/(α_∥, φ_∥) for TI VEP — DONE (2026-04-29)

The rank-4 TI modulus splits cleanly into two orthogonal projectors:
$$\mathbf{C} = 2\eta_0 \, \mathbf{P}_\perp + 2\eta_\parallel^\text{eff} \, \mathbf{P}_\parallel$$

with `P_∥` the director-aligned projector (the `K` kernel of the original `_build_c_tensor`) and `P_⊥ = I_4 - P_∥`. Each projector has its own Maxwell relaxation time during yielding (τ_⊥ = η_0/μ stays at the matrix value while τ_∥ = η_∥_eff/μ collapses). Phase B's single lumped (α, φ) cannot represent both timescales; per-component decomposition can.

**Validated in 1D cleanroom first** (`_exp_integrator_phase_d_split.py`): two parallel Maxwell branches with disparate τ, sinusoidal forcing, closed-form analytical reference. Per-component matches analytical to discretisation order (slope-2 in Δt, max\|err\|/A_∞ ≈ 5e-6 at Δt=0.005); every lumped variant carries Δt-independent error 7%-142% — the splitting is structurally required when τ_⊥ ≠ τ_∥, not a Δt issue.

**UW3 implementation** as `TransverseIsotropicVEPSplitFlowModel` (`src/underworld3/constitutive_models.py`):

1. **Sub-moduli**: `_build_split_c_tensors(η_⊥, η_∥)` returns `C_⊥ = 2η_⊥·(I-K)` and `C_∥ = 2η_∥·K` by zeroing one viscosity in the existing `_build_c_tensor` loop. Sum recovers original C.

2. **Lagged η_∥ via `forcing_star`**: `_eta_par_eff_lagged()` reuses the parent's softmin envelope but reads the rate from `forcing_star.sym` (projected previous-step ε̇) instead of `self.E_eff.sym` (current Newton iterate). Breaks the per-quad-split's 1-iter trivial-Newton failure mode (where α_∥ depends on η_∥ depends on E_eff depends on Newton's u, collapsing to fixed point).

3. **Explicit-parallel plasticity**: both `α_∥, φ_∥` AND the C_∥ multiplier use the lagged η — fully Picard for the parallel branch. ETD's E_eff has weak σ-history coupling (`α/(2η_1) ≈ 0.5` vs BDF's `1/(2μΔt) ≈ 10`) so the parent's _eta_par_eff would not see the yielded state on the current iterate; using forcing_star sees it because |γ̇*| is large there. BDF-1 effectively does the same Picard treatment via its E_eff magnification.

4. **Soft cap on x_par** (recommendation #4): `x_eff = (1 - exp(-c·x_natural))/c` keeps `α_∥ ≥ exp(-1/c)`, equivalent to `τ_∥ ≥ c·Δt`. User-tunable via `cm.tau_par_cap_factor` (default c=1.0). This shape pre-evaluates to a finite scalar at codegen-time defaults (dt=∞, μ=∞, Pint(1, "Pa·s") for η) where additive forms hit `oo+Pint` dimensional clashes.

5. **σ_∥ probe added** to all three killer-test runners: resolved fault-shear `|σ_∥| = √(|σ·n|² - (n·σ·n)²)` measured at fault centre per step. The previously-used `|σ_xy|` global-frame probe overshoots the yield surface in BDF too (2.15·τ_y) — `|σ_∥|` is the right comparator and shows BDF sits at 1.04·τ_y (essentially exact).

**Killer-test outcome** (θ=+15°, τ_y=0.05, RES=32, 1.5T):

| metric | BDF-1 | ETD lumped | split (Newton-impl, c=0) | split + cap (c=1.0) |
| --- | --- | --- | --- | --- |
| centre `\|σ_∥\|` peak | **1.04·τ_y** | 2.06·τ_y | 4.15·τ_y | **1.21·τ_y** |
| centre `\|σ_xy\|` peak | 2.15·τ_y | 29.10·τ_y | 4.92·τ_y | 2.47·τ_y |
| global max `\|σ\|_II` | 1.05 | 17.82 | 0.41 | 1.32 |
| global max `\|u_y\|` | **0.032** | 18.49 | 0.070 | 0.681 |
| SNES iters mean / max | 1.8 / 4 | 8.1 / 22 | 1.0 / 1 | 1.0 / 1 |
| wall / step | 1.7 s | 5.6 s | 4.1 s | 1.9 s |

(τ_y=0.15 sanity check: split + cap gives σ_∥ = 1.03·τ_y, |u_y| = 0.012, SNES 1 iter mean — Phase B regime preserved.)

**What works**: σ_∥ enforcement to within 21% of τ_y (vs BDF's 4%); no global runaway; physically correct fault-mechanics structure (PyVista plots `output/exp_integrator_phase_d_pyvista_split_*.png` show strain rate localised on fault, σ saturated at yield surface, bipolar u_y indicating along-fault slip). 1-iter Newton (linear in parallel branch) makes per-step cost competitive with BDF.

**Open**: `|u_y|` is 16-21× BDF-1's. The yield surface is correctly enforced; the difference is in how much slip accumulates per yield cycle. Mechanism (lesson #9 below): BDF's E_eff = ε̇ + σ*/(2μΔt) has built-in elastic damping that absorbs boundary motion into elastic accumulation rather than slip. ETD's E_eff with α_∥ → 0 at yield wipes elastic memory each step; even with the soft cap, the flux structure keeps slip accumulating at near-boundary rate. Not a yield-criterion failure — both integrators sit on the yield surface — but a difference in how the constitutive law is integrated through the yielded regime.

### Phase E — Hybrid BDF/ETD with spatial fault weight — DONE (2026-04-29)

User-suggested structural insight: in the TI fault model the user already supplies the fault geometry through `yield_stress(x)`; we know a priori where yielding *can* happen. So let each integrator handle its sweet spot:

- Inside the fault zone (where `τ_y(x)` is reachable): **BDF-1** — its `σ*/(2μΔt)` magnification provides the elastic damping that the cyclic-yield regime needs.
- Outside the fault (where `τ_y(x) → τ_y_bulk` ≫ A_∞ and yielding is structurally unreachable): **ETD-2** — strictly more accurate VE; its lack of plastic damping doesn't matter because plasticity isn't activated.

**Math**: `σ(x) = w(x)·σ_BDF + (1-w(x))·σ_ETD` with `w(x) = (1/τ_y(x) - 1/τ_y_bulk) / (1/τ_y_fault - 1/τ_y_bulk) ∈ [0, 1]`.

**Implementation** (in `TransverseIsotropicVEPFlowModel`):
- New `integrator='hybrid'` option; constructor takes `fault_weight` (sympy expression).
- `_eta_for_tensor(integrator_mode, apply_yield)` extracts (η_0, η_1_eff) per integrator/yield combination.
- `_assemble_c_tensor(η_0, η_1_eff)` builds the rank-4 tensor from given values.
- `_build_c_tensor` for `'hybrid'` builds both `_c_bdf` (yield-clipped) and `_c_etd` (raw).
- `_e_eff_for(integrator_mode)` returns the right E_eff form.
- `stress()` for `'hybrid'` blends `w·(C_BDF:E_eff_BDF) + (1-w)·(C_ETD:E_eff_ETD)`.
- Both BDF and ETD coefficients update each step. Single shared psi_star + forcing_star.

**Killer-test outcome** (θ=+15°, τ_y=0.05, RES=32, 1.5T):

| metric | BDF-1 | ETD lumped | split + cap | **hybrid** |
| --- | --- | --- | --- | --- |
| centre `\|σ_∥\|` peak | **1.04·τ_y** | 2.06·τ_y | 1.21·τ_y | 1.12·τ_y |
| centre `\|σ_xy\|` peak | 2.15·τ_y | 29·τ_y | 2.47·τ_y | 2.35·τ_y |
| global max `\|σ\|_II` | 1.05 | 17.82 | 1.32 | **0.95** |
| global max `\|u_y\|` | **0.032** | 18.49 | 0.681 | 0.109 |
| SNES iters mean / max | 1.8 / 4 | 8.1 / 22 | 1.0 / 1 | 2.1 / 4 |
| wall / step | 1.7 s | 5.6 s | 1.9 s | 2.3 s |

(τ_y=0.15 sanity: σ_∥=1.05·τ_y, |u_y|=0.014, SNES 1.5 mean — matches BDF.)

**What works**: σ_∥ peak 1.12·τ_y (closest to BDF's 1.04 of any ETD variant), |σ|_II peak 0.95 (actually slightly tighter than BDF's 1.05 at this snapshot), Newton iterates normally (2.1 vs split's degenerate 1.0). PyVista field plots show physically clean structure: u_y range ±0.017 at chosen step (no boundary overshoot), strain-rate localised on the fault band, no fault-tip stress concentrations.

**Why we still don't ship it**: the trajectory plot reveals `|u_y|` ramps monotonically from ~1e-5 to 0.109 over 1.5 periods — slow accumulation, not bounded oscillation like BDF-1 (which oscillates around 0.01-0.03 returning to baseline between yield events). At any single snapshot the field looks BDF-class; over cycles, drift accumulates.

**Likely cause**: shared σ* history. Both BDF and ETD branches read from the same `psi_star`, but `psi_star` is updated to the *blended* σ each step. Inside the fault, the BDF branch's σ* is "previous step's blended σ" — not "previous step's BDF-pure σ". Bulk's ETD-stored history leaks into the fault's BDF computation, slowly amplifying fault slip over cycles. Fixing this would need two independent history fields with parallel updates — a significant refactor.

**Decision**: Phase E as committed is the cleanest hybrid we tried, but doesn't deliver BDF-class temporal behaviour and fundamentally can't without the independent-history rework. Keep on branch as documented investigation; not advertised in user-facing API.

### Phase F — Generic `TimeIntegrator` refactor (deferred — only if needed)

If we end up with five-plus integrator methods on the DDt class and want to add another (e.g., Crank-Nicolson or higher-order ETD), refactor to separate `HistoryStorage` from a `TimeIntegrator` strategy object. Not needed for current scope.

---

## Open architectural questions to resolve during Phase B

1. **Lagged-τ vs SNES sub-iteration for VEP**

   For yield-active VEP, $\tau_{\text{eff}} = \eta_{\text{eff}}/\mu$ depends on σ (nonlinear). Two strategies:
   - *Lagged-τ (Picard)*: Compute α, φ, A, B from previous step's η_eff. First-order in the nonlinear coupling, trivial to implement. **Phase B starts with this.**
   - *Self-consistent τ via SNES*: Include τ in the iterate so the inner Newton converges τ↔σ together. More accurate but couples the time-integration to the SNES tolerance. Add only if lagged-τ shows insufficient accuracy.

2. **Per-quad α, φ when τ is spatial**

   When η_eff is a spatial field (yield zone, weakness map), α = exp(-Δt/τ) becomes a spatial expression. Sympy handles `exp(spatial_expr)` symbolically, but JIT codegen has to evaluate `exp` per quadrature point per residual eval — potentially expensive.

   Mitigation: project (α, φ) onto a scalar mesh variable at the start of each step. They're constant within a step. The JIT then sees a scalar-field reference, not an `exp` to evaluate. ~one extra projection per step.

3. **Forcing-history projection cost**

   ε̇* needs to be projected into `forcing_star[0]` after each solve. UW3's `SNES_MultiComponent_Projection` (committed in 2026-04 for VE-Stokes' tau projection, see `docs/developer/CHANGELOG.md`) makes this cheap and direct. Memory cost: one extra `SYM_TENSOR` MeshVariable per VE/VEP solver.

4. **TI-VEP per-component decomposition**

   The TI rank-4 tensor has separate timescales: $\tau_0 = \eta_0/\mu$ for bulk, $\tau_{1,\text{eff}} = \eta_{1,\text{eff}}/\mu$ for fault-tangent. The clean approach is to construct the rank-4 stress tensor with separate $(\alpha_0, \varphi_0)$ for the isotropic part and $(\alpha_1, \varphi_1)$ for the director-aligned correction, matching how `_build_c_tensor` already does separate viscosities. **Validate at Phase B step 5; bug-fix in this session if needed.**

5. **Asymmetric fine-Δt windows around BC flips**

   Phase B 1D evaluation showed that centred fine windows around BC discontinuities waste their pre-flip half (σ is near peak and barely changes) while the post-flip half does the real work. Production benchmarks should use asymmetric windows (small pre, larger post). Affects the `bench_*_vardt` schedule, not the integrator itself.

---

## Validation gates (achieved — Phase B closed)

| Test | Baseline | ETD-2 result | Status |
|---|---|---|---|
| `bench_ve_harmonic` | BDF-2 max\|err\| = 1.34e-3 | **3.14e-4** | ✅ **4.3× more accurate** than BDF-2 |
| `bench_ve_square` (const-Δt) | BDF-1 2.83e-2, BDF-2 8.07e-2 | 8.72e-2 | ✅ matches BDF-2 within 10% |
| `bench_vep_square` (Min) | peak\|σ\| = 0.5000 | peak\|σ\| = 0.4899, **0/160 violations** | ✅ saturated under τ_y |
| **`bench_ti_vep_harmonic` order=2 (killer test)** | **BDF-2: 10⁵-10⁹ blow-up on every yield-active combo** | **6/6 PASS, σ ≲ 1.12·τ_y at fault centre** | ✅ **decision gate met** |
| 20 existing VE/VEP regression tests | pass | pass | ✅ no BDF regression |

### Killer-test detail (`bench_ti_vep_harmonic`)

Centre-probe metrics (apples-to-apples with BDF-1 production), ETD-2 vs BDF-1:

| θ | τ_y | ETD-2 \|τ_resolved\| | BDF-1 \|τ_resolved\| | ETD-2 \|σ_xy\| | BDF-1 \|σ_xy\| |
|---|---|---|---|---|---|
| 0°  | 0.15 | **1.103·τ_y** | 1.122·τ_y | 1.103·τ_y | 1.122·τ_y |
| +15°| 0.15 | **1.118·τ_y** | 1.143·τ_y | 1.410·τ_y | 1.447·τ_y |
| -15°| 0.15 | **1.120·τ_y** | 1.127·τ_y | 1.408·τ_y | 1.440·τ_y |
| 0°  | 0.30 | **0.922·τ_y** | 1.150·τ_y | 0.922·τ_y | 1.150·τ_y |
| +15°| 0.30 | **0.804·τ_y** | 1.139·τ_y | 0.929·τ_y | 1.049·τ_y |
| -15°| 0.30 | **0.803·τ_y** | 1.138·τ_y | 0.929·τ_y | 1.047·τ_y |

ETD-2 **is tighter than BDF-1 production on every probe**. BDF-2 (the higher-order method ETD-2 replaces) blows up to 10⁵-10⁹ on every yield-active combo (τ_y=0.15) — confirming the structural argument empirically.

Runner: `docs/developer/design/experiments/exp-integrator/_exp_integrator_phase_b_killer.py`.

---

## Future work (out of scope for Phase B but relevant)

- **Backtracking timestepping**: when a step contains an event that the integrator can't capture in one piece (e.g., a steep change in γ̇ or yield-onset), back up and retry with smaller Δt. Logically separate from the integrator choice; both BDF and exp would benefit. Useful for adaptive timestep strategies that don't know flip times a priori.

- **Higher-order ETDs**: ETD-3, ETD-4 would store 2 or 3 forcing-history slots and use cubic/quartic interpolation in the integral. Not needed unless second-order forcing accuracy proves insufficient (unlikely for typical mantle/lithosphere problems).

- **Higher-order yield treatment**: the lagged-τ approach is first-order in the nonlinear coupling. For sharp yield onset under variable Δt, a self-consistent τ via SNES sub-iteration may be needed. Bridge from Phase B if observed.

- **Symbolic τ_eff in non-Maxwell rheologies** (Burgers, Maxwell-Voigt, etc.): the exponential framework generalises to any linear relaxation operator. Each relaxation timescale gets its own (α, φ); the rank-4 contraction picks them up via a matrix exponential of the relaxation tensor. Out of scope, but the architecture leaves the door open.

---

## What we learned — deviations from the original plan

These notes capture decisions taken during Phase B that diverge from or refine the pre-Phase-B plan above. They should inform Phase C and Phase D scope.

### 1. The "JIT propagation" task was a red herring

The plan's Task 1 anticipated a UWexpression-to-JIT propagation issue based on the jury-rig's failure. The actual cause was simpler: the jury-rig subclassed `ViscousFlowModel` which has `requires_stress_history = False`, so the Stokes solver took the viscous branch where `cm.flux` is **never compiled** — it builds the flux from `cm.viscosity` instead. The custom flux containing `_exp_alpha`, `_exp_phi`, etc. was effectively dead code. Once the model declares `requires_stress_history = True` (as the new sibling did), the existing constants-manifest infrastructure handles α, φ propagation correctly with no new plumbing.

### 2. Predictor-corrector return mapping (the 1D Phase B's yield approach) is wrong for 2D Stokes

The 1D Phase B evaluator used predictor-corrector return mapping: solve pure VE, then clip σ to satisfy yield. In 2D Stokes that breaks momentum balance — the SNES finds u that satisfies `∇·σ_VE = body force` (no yield), then we clip σ but leave u unchanged, so the velocity field corresponds to the unclipped stress. **In 2D Stokes-VEP, yield must live inside the SNES residual** via the standard viscosity-wrapping pattern (`viscosity = softmin(η, η_pl)`), the same as the production BDF VEP path. Refactored mid-Phase-B (commit `aba93c2`).

### 3. Lagged-τ aggregation experiments did not tighten yield-surface saturation

Multiple lagged-τ approaches were tried (scalar `min η_eff` over yield-active nodes; scalar `median η_eff`; per-node spatial α via projected scalar mesh variables) — all gave **worse** σ overshoot than the raw τ_VE baseline. Analysis showed the ETD-2 history term `2η_raw·(φ-α)·ε̇*` uses raw η (not yield-clipped) — a Picard-style approximation — and produces a non-zero floor on σ under harmonic forcing that is insensitive to τ_eff except via the α·σ* scaling. The effect is geometric, not a τ-choice issue. Reverted to raw τ_VE = η/μ (commit `584dea8`).

The ETD-2 result at parity-with-BDF-1-production — 1.10-1.14·τ_y at the fault centre — reflects the same kind of overshoot BDF-1 itself shows. Tightening past that is a Phase D concern requiring per-component (α₀, φ₀) for the rank-4 TI tensor, **not** a fix on the lagged-τ aggregation.

### 4. Probe-metric mismatch caused a false alarm

`max σ_II/τ_y_local` over a fault-zone mask reads larger than `σ_xy at fault centre / τ_y_at_fault` because the Gaussian-weakened τ_y(x) varies sharply across the mask: shoulder nodes have τ_y_local much larger than the centerline value, and σ_II saturates accordingly at those local τ_y, which inflates the ratio when the centerline τ_y_at_fault is used as the denominator. **Use the per-node ratio `max σ_II(x)/τ_y(x)`**, or stick to the centre-probe metric (the one BDF-1 production reports). The killer-test runner now reports both.

### 5. Architectural collapse landed at the parameter level

The plan envisaged Phase B with sibling classes (`MaxwellExponentialFlowModel`, TI variant) and Phase D moving integrator state onto the DDt with a strategy parameter. The collapse landed at the **constructor parameter** level instead: `ViscoElasticPlasticFlowModel(unknowns, integrator='etd')` (and the same on TI-VEP). Coefficients still live where the existing infrastructure naturally wants them (`_bdf_c0..c3` on the model, `_exp_coeffs` on the DDt) — the dispatch in `E_eff`, `viscosity`, `_build_c_tensor`, and the uniform `_update_history_*` hooks all branch on `self._integrator`. Sibling classes survive as ~10-line aliases for backwards compatibility (commit `ae79664`).

### 6. Unit-handling in any new array touchpoints needs explicit care

A predictor-corrector clip of `psi_star[0].array` via raw numpy initially looked correct in the non-units case (production benches don't use units) but stripped UnitAwareArray wrappers silently. The user flagged this as accumulating tech debt. The audit fix landed in commit `aba93c2`: `forcing_star` allocated `units=None` (ε̇ has different physical dimensions from σ), `update_forcing_history` non-dimensionalises eval results before storing. Future Phase D work touching `.array` should follow the same pattern (see `update_pre_solve` for the canonical example).

### 7. Empirical range of validity: τ_y / A_∞ ≥ ~0.5; below that, Phase B ETD-2 is strictly worse than BDF-1 production

A direct test by tightening τ_y from 0.15 to **0.05** on `bench_ti_vep_harmonic` (so τ_y / A_∞ = 0.05/0.27 ≈ 0.19) at RES=32 over 1.5 periods produced a **catastrophic step-by-step runaway** for Phase B ETD-2 even though Newton converged on every step. Apples-to-apples comparison with BDF-1 production on the *same* setup (saved at `output/phase_b_th{0,15}_ty0p05.*` and `output/phase_b_bdf_th+15_ty0p05.npz`):

| metric (θ=+15°, τ_y=0.05) | **ETD-2** (Phase B) | **BDF-1** (production) |
|---|---|---|
| max σ_II in domain | **17.8** | **1.05** |
| u_y range | ±18 | ±0.032 |
| SNES iter mean / max | 8 / 22 | 1.8 / 4 |
| Wall / step | 5.9 s | 1.7 s |
| Centre \|σ_xy\| peak | 17.8 (356·τ_y) | 0.108 (2.15·τ_y) |
| Diverged SNES steps | 0/120 | 0/120 |

The catastrophe **is specific to the ETD-2 implementation**, not a problem-class issue: BDF-1 production handles τ_y=0.05 cleanly with bounded σ, faster Newton (mean 1.8 iters), and 3.5× faster wall time per step.

Time-series comparison, both integrators run on the same RES=32 mesh with matching driver and step size: ``output/exp_integrator_phase_b_bdf_vs_etd.png`` (generated by `_plot_phase_b_bdf_vs_etd.py` from `output/phase_b_{bdf,etd}_th+15_ty0p05.npz`). The ETD-2 trace tracks BDF-1 inside the ±τ_y band for the first half-cycle, then breaks loose at the second yield event and runs away through the second period — peak centre |σ_xy| reaches 1.46 (29·τ_y) and global max |u_y| reaches ~18, while BDF-1 stays at 0.11 and 0.03 respectively. The divergence point is the first deep yield, not a steady accumulation.

The mechanism is the one items 3 and 5 in this list already identified: the ETD-2 history term ``α·σ* + 2η·(φ-α)·ε̇*`` uses raw η (Picard approximation when yield is active). The analytical-floor σ-magnitude under harmonic forcing is ~A_∞, independent of τ_y. When A_∞ > τ_y, σ* feeds back through α·σ* on each step and grows without bound; the leading viscous term is yield-clipped but has small (1-φ) coefficient at typical Δt, so it can't dominate the runaway history.

**Newton's "convergence" reports in this regime are physically meaningless** — Newton finds the residual minimum each step, but the time-integration loop diverges. SNES iteration counts are *not* an early-warning signal (they actually *drop* from typical levels because the residual structure becomes degenerate); the warning is in σ_II / u_y magnitudes themselves.

Practical implications for Phase B as committed:

* ``integrator='etd'`` with the raw τ_VE = η/μ in `α, φ` works for **τ_y / A_∞ ≥ ~0.5** — at parity with BDF-1 production for accuracy at ratio 0.55, beats BDF-2 by 4.3× at no-yield ``bench_ve_harmonic``.
* Below that ratio (the **typical fault-mechanics regime**): solution diverges silently (no SNES error). Phase B ETD-2 is **strictly worse than BDF-1** — slower, less accurate, unstable.
* Phase B as currently committed should be treated as a structural-argument demo, not a drop-in replacement for the BDF integrators. Production users should keep ``integrator='bdf'`` (the default) until Phase D lands.
* **Phase D (per-component (α₀, φ₀)/(α₁, φ₁) for TI) is blocking, not "future work"**, for any production use of ETD-2 on tight-yield problems.

The Phase B design-doc note that "lagged-τ doesn't help" applies in this regime too — the failure is structural to the Picard approximation, not a τ-choice issue.

### 8. The diagnostic that mattered: |σ_∥| (resolved fault shear), not |σ_xy|

Throughout Phase B and into the early Phase D iterations, the killer-test trajectories used `|σ_xy|` at fault centre as the yield-surface diagnostic. That was wrong. The yield criterion `|σ_∥| ≤ τ_y` lives in the fault frame; `|σ_xy|` is global-frame and includes contributions that the limiter doesn't constrain (off-fault stress, geometric tilts).

Adding the resolved fault-plane shear `|σ_∥| = √(|σ·n|² - (n·σ·n)²)` as a per-step probe (commits 59ab769 onwards) revealed that:

* BDF-1 sits **right on** the yield surface (peak `|σ_∥|` = 1.04·τ_y, essentially exact) despite `|σ_xy|` peaking at 2.15·τ_y.
* Lumped Phase B ETD-2 stays at 2.06·τ_y in `|σ_∥|` even though `|σ_xy|` runs away to 29·τ_y — the catastrophe is off-fault, the *fault* is doing fine.
* Phase D's first split implementations (per-quad and Newton-implicit lag) overshot to 4·τ_y *on the fault plane*; this needed fixing.

Without `|σ_∥|`, Phase D would have been judged on `|σ_xy|` alone — and the cure (explicit-parallel + cap) would have looked like it just lowered the global-frame number without engaging with the actual yield-criterion physics.

### 9. The structural BDF-vs-ETD slip-rate difference — physics, not numerics

After Phase D's σ_∥ enforcement reached BDF parity (1.21 vs 1.04·τ_y), `|u_y|` remained 16-21× BDF-1's at τ_y=0.05. The mechanism is a structural difference in how each integrator handles the yielded regime, not a numerical defect:

* **BDF**: E_eff = ε̇ + σ*/(2μΔt). At Δt=0.05, μ=1, the σ-history prefactor is **10**. When σ_∥ saturates near τ_y, this term *dominates* E_eff_∥ — boundary motion is preferentially absorbed into elastic accumulation rather than slip. The integrator has built-in elastic damping during yield.
* **ETD (Phase D)**: E_eff_∥ = (1-φ_∥)·ε̇ + α_∥/(2η_∥)·σ* + (φ_∥-α_∥)·ε̇*. At yield with α_∥, φ_∥ → 0 (or even with the soft cap clamping them at 0.37, 0.63), the σ-history coefficient is at most O(1). Boundary motion goes into γ̇_∥ at the imposed BC rate — the fault slips freely.

Both integrators correctly enforce `|σ_∥| ≤ τ_y` (the limiter works). They just distribute the boundary motion differently between elastic and plastic strain. BDF's behaviour is closer to a typical seismic-cycle picture (elastic energy stores and releases episodically); ETD's is closer to steady-flow plasticity (boundary motion drives free slip at yield). Neither is "wrong"; they're modelling different limits of the same constitutive law.

Implication: when comparing integrators on a tight-yield problem, σ-amplitude is a poor metric (both at τ_y); the meaningful difference is in time-integrated slip per cycle, which depends on the elastic-damping strength and is integrator-specific.

### 10. Phase D recommendations checklist — what worked, what didn't

The chatGPT advisor's stabilisation strategy was on the money for the issues we hit:

| Recommendation | Phase D status |
| --- | --- |
| 1. Lag τ in the exponential — use τⁿ, never τⁿ⁺¹ | **Implemented.** `_eta_par_eff_lagged()` reads forcing_star (previous-step ε̇). Cured the per-quad split's 1-iter trivial-Newton failure mode. |
| 2. Plastic correction *after* VE update (predictor-corrector) | **Rejected.** Tried earlier in Phase B; broke 2D Stokes momentum balance. Yield-in-residual via softmin is the working pattern. |
| 3. Under-relax stress update (ω ~ 0.5) | **Not implemented.** Open follow-up. Would smooth Newton's hop and might tame the slip ratchet (lesson #9) without affecting the yield surface. |
| 4. Cap τ_eff ≥ c·Δt to avoid α_∥ → 0 | **Implemented** as a soft x_par cap `(1-exp(-c·x))/c`. Tunable via `cm.tau_par_cap_factor` (default 1.0). Modestly improves σ_∥ enforcement (1.31 → 1.21·τ_y at τ_y=0.05) but slightly worsens the slip ratchet (0.525 → 0.681) — the inconsistent capping (η_C natural, η_α capped) shrinks the (1-φ_∥)·E term in proportion to the σ*-contribution, so σ_∥ stays controlled but flux balance is more sensitive between yield events. |
| 5. Consistent viscosity in Stokes + constitutive | **Implemented.** Both C_∥ and (α_∥, φ_∥) use the lagged forcing_star-based η. Earlier inconsistency (C_∥ on current η, α_∥ on lagged η) had Newton converge in 1 iter and σ_∥ drift to 4·τ_y. |

Also-tried, rejected:
* **Raw E (current strain rate) as yield-criterion rate input**: adds explicit u-dependence into η_∥_eff, which propagates into C_∥ and produces a singular GAMG operator at u=0 (start-up zero state). Smooth-floor regularisation didn't fix the SNES 0-iter divergence. Reverted — the parent's E_eff-based criterion is the right shape, just needs the right rate input (forcing_star, lesson above).
* **`σ*/(2ε̇*)` back-derivation for lagged η**: appears intuitive (it's the *effective* viscosity from histories alone) but breaks elastic regime (where σ ≈ μ·γ·dt, not η·ε̇). Produced startup spikes. Replaced by the parent-softmin-on-forcing_star pattern.
* **`sympy.Min` cap on η**: catastrophic 29/120 SNES diverged; non-smooth derivative breaks Newton. Replaced by smooth `(1-exp(-c·x))/c`.

### 11. Phase E (hybrid) drifts because of shared σ* history — and that's structural

The Phase E hybrid (`σ = w·σ_BDF + (1-w)·σ_ETD`) was conceptually the cleanest fix for the BDF-vs-ETD mismatch — let each integrator handle its sweet spot. Snapshot-by-snapshot the field structure is BDF-class (no boundary overshoot, fault-band strain-rate localisation, σ_∥ within 8% of τ_y). But the time-trajectory shows `|u_y|` ramping monotonically over cycles, ending at ~3× BDF.

Mechanism: both branches share a single ``psi_star`` history slot, which is updated to the *blended* σ each step. So inside the fault, the BDF branch's σ* is "previous step's blended σ" — contaminated with ETD's looser-history contribution from the bulk. Over many cycles, that contamination amplifies fault slip slightly each pass.

The fix would require two independent history fields with parallel updates (BDF history fed only by BDF flux, ETD history fed only by ETD flux, plus the spatial blend at flux time). That's a real DDt refactor, not a one-line fix, and even then there's no guarantee the slow drift fully closes — the underlying physics-mismatch (lesson #9) lives in how each integrator handles the yielded regime, not just in the history bookkeeping.

The investigation-level lesson: **patches that share history between BDF and ETD branches will leak the missing damping into temporal drift.** Whether it's a per-quad split (Phase D, Newton-implicit), explicit-parallel split (Phase D with cap), or spatial blend (Phase E), the slow drift keeps reappearing in different magnitudes. ETD-as-designed is a beautiful integrator for VE; trying to retrofit it onto deep-yield VEP without rebuilding from the ground up consistently leaves residual non-physical behaviour.

### 12. (superseded by #13)

The conclusion in earlier drafts of this section ("for deep-yield fault mechanics BDF-1 is the right integrator; don't retrofit ETD") was correct *for higher-order ETD*. Lesson #13 below shows it's wrong for ETD generally — first-order ETD works fine.

### 13. The drift was order-driven, not algorithm-driven — ETD-1 ships

User's structural insight: "all the integrators have this growing instability except the first order one." ETD-1 (first-order ETD with `φ = α`) confirms it empirically — it reproduces BDF-1 essentially exactly on the killer test:

| metric (θ=+15°, τ_y=0.05) | BDF-1 | ETD-1 |
| --- | --- | --- |
| centre `\|σ_∥\|` peak | 1.04·τ_y | 1.04·τ_y |
| global max `\|u_y\|` | 0.0320 | 0.0320 |
| SNES iters mean | 1.8 | 1.8 |
| diverged | 0/120 | 0/120 |

Mechanism: BDF-1 and ETD-1 are both **L-stable** (`|R(z)| ≤ 1` on the entire negative-real-part half-plane → every mode is damped). BDF-2 is only A-stable; ETD-2 is exact for the linear ODE so has *zero* numerical dissipation. The plastic yield transitions create effective high-frequency modes (residual structure flips discontinuously when σ crosses τ_y); first-order methods damp them with the same numerical viscosity they apply to everything else, while higher-order methods preserve them and let them grow.

Same general principle as Crank-Nicolson failing on stiff problems while implicit Euler doesn't.

This collapses the "ETD doesn't work for fault mechanics" narrative the earlier lessons #7, #9, #11, #12 were converging toward. The actual statement is "*higher-order* ETD doesn't work for fault mechanics," same as higher-order BDF. ETD-1 (single-step, no forcing-history slot, analytical exp factor) is the right shape: BDF-1 stability + ETD's exact treatment of the linear-relaxation part.

**Production recommendation**: `integrator='etd', order=1` as the default for VEP and TI-VEP. Wall-clock cost ~5% over BDF-1 (one extra `exp` per coefficient update); accuracy is per-iteration the same as BDF-1 (both first-order) but the analytical factor handles the linear-relaxation limit cleanly without the rational-approximation error at large `Δt/τ`. Phase B's ETD-2 (`integrator='etd', order=2`) remains available for users with smooth VE problems who can certify yield is never active — it beats BDF-2 by 4× there.

The Phase D and Phase E artefacts stay on the branch as instructive failures of the higher-order-ETD idea — useful documentation of what doesn't work and why, but not part of the production API.

---

## Appendix A — Numerical evidence

### Phase A (1D linear Maxwell, sinusoidal forcing) — DONE

`_exp_integrator_phase_a.py` solves $\dot\sigma + \sigma/\tau = \mu\dot\gamma$ with $\dot\gamma = \dot\gamma_0 \cos(\omega t)$, $\omega = \pi/2$, $\eta = \mu = \tau = 1$.

| Δt/τ | Exp max\|err\| | BDF-1 | BDF-2 |
|---|---|---|---|
| 0.01 | 1.1e-5 | 4.5e-3 | 7.4e-5 |
| 0.05 | 2.9e-4 | 2.2e-2 | 1.8e-3 |
| 0.10 | 1.1e-3 | 4.4e-2 | 6.8e-3 |
| 0.50 | 2.8e-2 | 1.9e-1 | 1.1e-1 |
| **1.00** | **1.0e-1** | 3.5e-1 | 3.5e-1 |
| **2.00** | **5.7e-2** | 3.4e-1 | 3.4e-1 |

Exp shows clean second-order slope at small Δt and stays accurate at Δt ≥ τ where both BDFs collapse to near-zero output. Figure: `experiments/exp-integrator/exp_integrator_phase_a.png`.

### Phase B (VEP, large Δt, square wave, variable-Δt) — DONE

`_exp_integrator_phase_b_eval.py` extends to:

- **VEP harmonic** ($\omega = \pi/4$, return-mapping yield): both Exp and BDF-1 clip correctly at τ_y; agreement to ~1% at small Δt because yield mechanism dominates over time integrator.

- **Pure VE at large Δt/τ**: at Δt/τ ≤ 1, Exp 5–12× more accurate; at Δt/τ ≥ 2, both struggle but Exp degrades more gracefully (gives bounded under-shoot vs BDF's wrong-shape output).

- **Square wave VE/VEP**: exp consistently ~2× more accurate than BDF-1 for VE; the yield+BC-discontinuity error dominates both for VEP at small Δt.

- **Variable-Δt around BC flips** (correctly schedule, with fine-zone clamp): improvement of 11–19% in max error for both VE and VEP, both Exp and BDF-1. The exp's plateau-period exactness shows clearly as the per-step error drops to near machine precision once the BC discontinuity is well-resolved.

Figures (in `experiments/exp-integrator/`): `exp_integrator_phase_b_yield.png`, `exp_integrator_phase_b_largedt.png`, `exp_integrator_phase_b_square.png`, `exp_integrator_phase_b_vardt.png`.

### Phase B UW3 jury-rig — partial (propagation snag identified)

`_exp_integrator_uw3_jury_rig.py` attempted to wire ETD-2 into UW3 via a custom `MaxwellExpFlowModel(ViscousFlowModel)` subclass. Hit a JIT propagation issue: `cm._exp_alpha.sym = X` per-step updates don't reach the JIT-compiled flux. Minimal incremental test (`_exp_jury_rig_minimal.py`) confirmed the constitutive-model class plumbing works in isolation; the issue is specific to per-step updates of UWexpression coefficients. **First task of Phase B is resolving this**, by replicating the BDF coefficient propagation pattern.

---

## Appendix B — Architecture details

### What the exponential integrator stores

| Integrator | psi_star slots | forcing_star slots | Coefficients |
|---|---|---|---|
| BDF-1 | 1 | 0 | c_0, c_1 |
| BDF-2 | 2 | 0 | c_0, c_1, c_2 |
| AM-2 | 1 | 0 | a_0, a_1, a_2 |
| ETD-1 (Lawson) | 1 | 0 | α |
| **ETD-2 (this proposal)** | **1** | **1** | **α, A, B** |
| ETD-3 | 1 | 2 | α, A, B, C |

The `SemiLagrangian` already maintains parallel `_bdf_coeffs` and `_am_coeffs`. Adding `_exp_coeffs` is the same kind of peer extension.

### What stays the same vs BDF in the constitutive model

The factorisation σ = η_eff·γ̇ + (history) is preserved. Yield-mode logic (softmin/min/harmonic) wraps η_eff identically to today. The Stokes weak-form structure is unchanged. What changes:
- Different formula for η_eff_VE: η(1-φ) replaces η Δt/(τ+Δt)
- Different history term: α·σⁿ + 2η(φ-α)·ε̇ⁿ replaces the BDF Σ c_i ψ*_i sum
- New ε̇* storage slot

### Why this avoids the BDF-2 instability (TI-VEP + spatial yield)

The instability we documented arises from the c_2·ψ*_{n-1} term in BDF-2's history sum getting autodiff'd into the Jacobian, where it picks up the spatial gradient of η_1_eff (via $\partial\eta_{1,\text{eff}}/\partial\nabla u$), and then gets *directionally amplified* by the rank-4 tensor's $\hat n\otimes\hat n\otimes\hat n\otimes\hat n$ coupling. The amplification compounds across history, exploding |σ| over ~10 t_r.

Exponential has **no second history term**. There's no c_2·ψ*_1 to amplify. The α·σⁿ contribution is autodiff-trivial (σⁿ is a known mesh variable, treated as constant w.r.t. ∇u). The ε̇ⁿ contribution likewise. The Jacobian's only ∇u-dependent term is the leading 2η_eff(1-φ)·ε̇, which is well-behaved.

This is why the structural argument carries to TI-VEP via the per-component decomposition (Phase B step 5): each component of the rank-4 tensor gets its own (α, φ, A, B), each with single-history-slot relaxation, no cross-component amplification.
