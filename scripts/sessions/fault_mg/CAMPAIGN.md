# ZG95 fault convection campaign: hand-over (2026-10-15)

Driver `zg95_annulus.py` (annulus, Rayleigh-scaled yielding lid, SUPG transport,
rotated free-slip, split-node fault on the add-on hierarchy), renderer
`zg95_stream.py` (surface velocity over streamlines; red fault trace; vertical
exaggeration 2 of the unrolled section). Outputs live on hyperion under
`~/+Simulations/zg95/<run>/` (checkpoints every 25 steps: `zg95.mesh.*.h5`,
`state_NNNN.json`, `records.json`, `surface_NNNN.npz` with the trench trace,
`stream_NNNN.png`, `step_NNNN.png`). Frames are on the wiki page `Fault-multigrid`.

## Physics (Moresi & Solomatov 1998 reference case)
Ra0 = 1e7 on the basal viscosity (`--eta_ref 1`), contrast 3e4, yield 1e5 + 1e7 z
(tau1/Ra0 = 1, mobile per their Fig 6), internal heating 7, interior from 0.8,
base H 0.16 with 2 halvings (finest 0.04), P2 velocity.

    --H 0.16 --ref 2 --ra 1e7 --eta_ref 1.0 --advect supg --heating 7 --deta 3e4
    --yield0 1e5 --yield_depth 1e7 --T_int 0.8 --dt_mult 3 --dt_max 1e-3
    --picard 2 --yield_soft 0.2 --fig_every 50 --ckpt_every 50

## Runs
- `conv_ms98` (control, 1500 spin-up steps, DONE): steady six-cell mobile lid from
  step ~100: mobility 0.5-0.7, lid 0.105, a third of the lid yielding, T_mid 0.70,
  plateau surface speeds ~80, 27-35 s/step at 8 ranks (tol 1e-3, Newton 6-8).
- `conv_ms98_step` (restart from ckpt 100 with `--deta 1e5 --yield0 2e5`, DONE):
  sluggish distributed-deformation regime, lid 0.15, mobility 0.2-0.4, no overturn.
- `fault_ms98_split` (fault phase from ckpt 150, 300 steps, trench velocity sampled
  1.7 deg into the hanging wall -> trench effectively frozen, DONE): localised slip
  (85-120), stationary sliver, one-sided dipping slab (t 0.12-0.165), slab break-off
  (0.177), overturn (0.183, overriding lid at 250-300), ridge jump to the trench
  (0.19+). Lid yielding mean 0.40 vs control 0.33. Montage: `sequence.png`.
- `fault_ms98_move` (same, trench follows the overriding plate: median surface
  velocity over 0.1-0.3 rad beyond the fault, `--trench_off/--trench_span`;
  RUNNING on hyperion, 8 ranks, ~2 min/step): the slab follows the advancing
  fault at ~12 deg lag, still symmetric at t 0.093.

Fault phase launch (from a control checkpoint N copied into the run directory):

    cp conv_ms98/*000NN* conv_ms98/zg95.mesh.00000.h5 conv_ms98/state_0NN.json conv_ms98/records.json <out>/
    mpirun -np 8 python zg95_annulus.py --out <out> --restart N --spinup 0 --steps 300 --arm split \
        <physics flags> --tol 1e-2 --newton_switch 1e-2 --fig_every 10 --ckpt_every 25 --verbose

## Solver state (the item to fix before production)
On the faulted add-on the pressure (Schur) solve hits `fieldsplit_pres_ksp_max_it`
= 200 once the nonlinear residual falls below ~1e-2 of its start, under the Newton
tangent AND under Picard (42 Picard iterations at tol 1e-3, cap from iteration 14).
The base mesh never shows it. Each pressure iteration nests a full velocity solve,
so a capped solve costs ~3400 MG cycles (~10 min). Hypotheses: the nested Schur
stalls on the inner-solve noise floor once the outer residual is small; or the
yielded viscosity on the faulted add-on breaks the 1/eta Schur preconditioner.
Probe: one step at np 2 from ckpt 150 with the pressure KSP monitor and converged
reason per outer iteration; compare base vs add-on; check the inner velocity rtol
against the pressure rtol (0.033 x tol). The fault runs above use tol 1e-2,
Picard-only (`--newton_switch 1e-2`) to stay below the cap. Warm start across
add-on rebuilds (`AddOn.warm_start`) halves the step cost.

## Other open items
- Dynamic topography sign: peaks over the downwellings, lows over upwellings in the
  step figures; check the reaction sign convention before quantitative use.
- Parallel-safety lessons applied in the driver (keep them): mesh-variable writes
  are collective, decide on every rank; cross-mesh evaluation of an expression with
  velocity gradients fails, evaluate on the owning mesh into a P1 variable and
  transfer that; build-line counts are global reductions.
- Wiki page `Fault-multigrid` carries the frames; keep it current.
