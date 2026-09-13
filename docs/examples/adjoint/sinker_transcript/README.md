# The sinking-blob adjoint, written in the timestepping pattern

Same problem and same mathematics as `../supg/`. What changes is the
scaffolding — and the point of the exercise is that the scaffolding is now
library machinery rather than something this project invented for itself.

Run it:

```bash
python generate_target_transcript.py     # twin experiment: v(T) at the true centre
python forward_sinker_transcript.py      # one forward run, printing its transcript
python taylor_test_transcript.py         # the gate
```

---

## What the forward model looks like now

```python
uw.reset_default_model()
uwmodel = uw.get_default_model()
uwmodel.set_reference_quantities(
    domain_depth=uw.quantity(500, "km"),
    material_viscosity=uw.quantity(1e21, "Pa*s"),
    lithostatic_pressure=REF_DENSITY * GRAVITY * DOMAIN_DEPTH,
)

# ... mesh, variables, solvers ...

uwmodel.clear_transcript()
uwmodel.tracker.time = uw.quantity(0.0, "Myr")
uwmodel.tracker.step = 0
uwmodel.record_every = 1

for _ in range(nsteps):
    with uwmodel.step(dt, label="sink"):
        adv.solve(timestep=dt, zero_init_guess=False)
        stokes.solve(zero_init_guess=False)
```

and it prints its own account of itself:

```
<step 0 'sink' dt=1.1158811388500671 [megayear] solve:SNES_AdvectionDiffusion_Composed(beta) -> history_shift:EulerianSUPG(beta) -> solve:SNES_Stokes(v)>
<step 1 'sink' dt=1.1158811388500671 [megayear] ... >
...
restorable: 5 of 5
```

### Three scales, not ten constants

`../supg/forward_sinker_supg.py` opens with

```python
REF_LENGTH = 500e3
REF_DENSITY = 3300.0
REF_GRAVITY = 9.81
REF_VISCOSITY = 1e21
REF_PRESSURE = REF_DENSITY * REF_GRAVITY * REF_LENGTH
REF_TIME = REF_VISCOSITY / REF_PRESSURE
REF_VELOCITY = REF_LENGTH / REF_TIME
DENSITY_BACKGROUND = 3200.0 / REF_DENSITY
...
DT_FIXED = 570.0
```

and every number after that is a ratio you have to keep in your head. Here the
three scales that actually fix this problem are declared once — a length, a
viscosity, and the lithostatic stress `rho g L` — and everything downstream is
written in the units it is quoted in: `50 km`, `3300 kg/m^3`, `1.1159 Myr`.

Density is deliberately **not** one of the reference quantities. The Stokes
sinker uses it only as a ratio, which is exactly why `REF_DENSITY` cancelled
out of every nondimensional number the old script produced.

The nondimensional problem that reaches the solver is bit-for-bit the one the
old script assembled by hand — the interface at `t = T` sits at 271.2 km and
371.2 km on the centreline either way, against 0.5425 and 0.7425 of the box
before. The difference is that the scaling is now stated and checked rather
than spread across a module header.

### There is no checkpoint dictionary

The old forward model carried its own recording:

```python
ck = {"B": [b0.copy()], "V": [...], "P": [...], "dt": []}
for k in range(nsteps):
    adv.solve(timestep=dt, zero_init_guess=False)
    stokes.solve(zero_init_guess=False)
    ck["B"].append(np.asarray(beta.array)[:, 0, 0].copy())
    ck["V"].append(np.asarray(v.array)[:, 0, :].copy())
    ck["P"].append(np.asarray(p.array)[:, 0, 0].copy())
    ck["dt"].append(dt)
```

Four lists holding precisely the arrays the adjoint turned out to want. That is
a recording you can only write once you already know the adjoint — which is the
wrong way round, and is a large part of why an adjoint is normally a rewrite of
the forward model rather than an addition to it.

It is also silently incomplete. It does not hold the transport history; it
works only because backward Euler happens to make `psi_star` recoverable from
`B[k]`. Change `theta`, add a second history level, put the level set on a
swarm, and the dictionary is quietly wrong in a way nothing detects.

`model.record_every = 1` replaces all of it with a request. The snapshot each
step keeps is the *whole* model state — every mesh variable, every swarm, every
registered state-bearer including the DDt history, and the clock — captured
before the step's operators ran, which is the only correct point.

---

## What the adjoint looks like now

The three blocks and the backward recursion are untouched. What changed is
where the states come from.

| | `../supg/` | here |
|---|---|---|
| the record | `ck` dict built by the forward loop | `model.transcript`, built by the library |
| a state | `ck["B"][k]`, `ck["V"][k]`, `ck["P"][k]` | `model.load_state(transcript[k].snapshot)` |
| the history | reconstructed from `ck["B"][k]` | restored with everything else |
| the clock | not recorded | restored with everything else |
| what ran | assumed | `transcript[k].events`, in order |

```python
def linearisation_state(self, k):
    """Restore the point step k's transport residual was linearised at."""
    entry = self.transcript[k]
    self.state_at(k)                                  # beta_k, V_k, history
    beta_in = np.asarray(beta.array)[:, 0, 0].copy()
    adv.solve(timestep=entry.dt, zero_init_guess=False)   # -> beta_{k+1}
    self.psi_star.array[:, 0, 0] = beta_in            # where the residual reads it
```

Two things are worth stating plainly about that replay.

**It is exact.** Restoring a snapshot and re-solving reproduces the step to the
last bit (measured: `maxdiff 0.00e+00` on every step). Re-*running* the script
does not — warm starts and preconditioner reuse are solver history, not model
state, so two independent runs of the same problem diverge at the 1e-13 level
from the first step. If you need to look at a step twice, restore it.

**It costs one extra transport solve per step**, and that is what buys the
recording being generic: the forward run does not have to know an adjoint is
coming. Trading a solve for not having to write a bespoke tape is the right
trade at this size; on a long run you would raise `record_every` and recompute
between restore points, which is the standard checkpointing schedule and is
what `record_every` / `record_limit` are for.

**One line is still scheme-specific.** Putting `beta_in` back into `psi_star`
is a statement about backward Euler, not about the record: the residual of step
`k` reads the step's input from the history slot, and the solve's post-hook has
already shifted it forward. A `solver.adjoint(...)` method would own that line;
today the driver does.

### The one thing the transcript cannot hold

An N-step run has N+1 time levels, and the transcript records *steps*. So
`solve_forward` returns `(transcript, final_state)`, and `state_at(N)` reads the
final state rather than a transcript entry. That asymmetry is real, not an
oversight — the last level is the run's output, not the input to anything.

---

## Two library changes this exercise produced

**`model.clear_transcript()` (new).** A driver that runs the same model many times
— an inversion, a parameter sweep — needs each run to have its own account.
Without it the transcript is the concatenation of every run the process has done,
and `rewind()` walks back into the previous one. The Taylor test runs the
forward model thirteen times; it found this immediately.

**A snapshot no longer rescales the mesh (fixed).** `mesh.X.coords` is the
unit-aware view and returns metres once a model declares a length scale; the
DM coordinate vector that restore writes back into holds model units. Capture
took the first and restore wrote the second, so **every restore multiplied the
mesh by the length scale** — 500 km became 250,000,000 km. Nothing raised:
shapes matched, fields came back correctly, only the geometry was wrong. The
symptom is that `uw.function.evaluate` starts returning the value at one corner
for every sample point, because every sample point is now outside the domain.

`model.rewind()` goes straight through that path, which is how it surfaced.
Covered now by `tests/test_0012_snapshot_units_coords.py`.

---

## The gate

`taylor_test_transcript.py`, control at (300 km, 350 km), true centre
(250 km, 375 km), five steps, contrast 1000:

```
J = 5.473279e-10   adjoint dJ/dcx0 = 2.022362e-14 /m   dJ/dcy0 = -9.883878e-16 /m
h=5.0  km:  FD dJ/dcx0  2.021281e-14  ratio 0.99947  |  FD dJ/dcy0 -9.842465e-16  ratio 0.99581
h=0.5  km:  FD dJ/dcx0  2.022355e-14  ratio 1.00000  |  FD dJ/dcy0 -9.883048e-16  ratio 0.99992
h=0.05 km:  FD dJ/dcx0  2.022356e-14  ratio 1.00000  |  FD dJ/dcy0 -9.884208e-16  ratio 1.00003
```

Same quality as `../supg/` (1.00000 / 0.99993). The gradients are small
because the control is now a length in metres rather than a fraction of the
box; multiply by 5e5 to compare with the old numbers.

---

## What is unchanged, and still load-bearing

The three physics choices from `../supg/` that make this adjoint exact:

- **Eulerian SUPG transport**, so one timestep is a residual and every
  sensitivity is a SymPy derivative of it.
- **Backward Euler (`theta = 1`)**, so `beta_old` appears in the residual
  undifferentiated and the history coupling is a pointwise expression rather
  than a weak form.
- **A fixed timestep**, so the objective does not depend on the control through
  the schedule. This was the dominant defect in the original adjoint.

See `../supg/README.md` for the derivation and the ablation that established
each of them.
