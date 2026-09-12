# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Convection in an Annulus — a recorded run

**PHYSICS:** convection
**DIFFICULTY:** intermediate

## Description

Boussinesq thermal convection in a 2D annulus, written in the timestepping
pattern: the model and its reference quantities come first, the clock lives on
`model.tracker`, and each timestep is a `model.step(dt)` block.

Compare `../advanced/Ex_Convection_Cylinder.py`, which solves the same physics
with a bare `for step in range(n)` loop and no clock at all. The physics here is
unchanged. What the pattern adds is that the run keeps an account of itself, and
that account is worth four things this script demonstrates in turn:

1. **what ran** — an ordered transcript, named by what each solver solves
2. **a rejected step** — the clock does not move when a step is abandoned
3. **playback** — a recorded step replays bit-for-bit, where a re-run does not
4. **recording without judging** — a step taken twice is visible in the
   transcript, and the transcript makes no claim about whether that is wrong
5. **a log on disk** — the same account in aligned columns, flushed as each
   step closes, so a run that dies keeps its history and a run in progress can
   be watched with `tail -f`
6. **a figure** — the same account as a portrait PDF (or SVG), and the step's
   operator flow as Mermaid

## Key concepts

- **Units first.** Four reference quantities fix the scaling, and the body
  force is then written as the physics — `-rho0 alpha T g rhat` — rather than
  as a Rayleigh number. Ra falls out of the nondimensionalisation; the script
  prints it so you can check.
- **Rotated free-slip on curved boundaries.** `add_rotated_freeslip_bc`
  enforces `v.n = 0` to machine precision on a circle, where a penalty or
  Nitsche condition leaks at ~1e-3.
- **A varying timestep.** `estimate_dt()` returns a dimensional quantity that
  goes straight into `model.step(dt)` and `adv.solve(timestep=dt)`. With an
  implicit SUPG transport this is an accuracy choice, not a stability limit.

## Parameters

Override from the command line, e.g. `-uw_n_steps 20 -uw_cell_size 0.075`.
"""

# %%
import os

import numpy as np
import sympy

import underworld3 as uw


def say(*args):
    """Rank-safe print that keeps its own formatting.

    `uw.pprint`'s default `clean_display=True` rewrites the string it is given
    — it strips braces and collapses runs of whitespace — so an aligned table
    printed through it loses its columns. Pass `clean_display=False` whenever
    the layout is yours rather than SymPy's.
    """
    uw.pprint(*args, clean_display=False)


params = uw.Params(
    uw_cell_size=0.1,          # mesh resolution, as a fraction of the outer radius
    uw_n_steps=8,              # timesteps in the recorded run
    uw_dt_fraction=0.5,        # accuracy factor on estimate_dt()
    uw_demos=1,                # run the transcript demonstrations after the loop
)

# %% [markdown]
"""
## The model comes first

Reference quantities must be set BEFORE the mesh is created, so the model is
the first thing the script declares rather than something the mesh conjures for
you. Four quantities fix all four dimensions this problem uses:

| quantity | fixes |
|---|---|
| `shell_thickness` | length |
| `thermal_diffusivity` | time, as `d^2 / kappa` |
| `mantle_viscosity` | mass |
| `temperature_contrast` | temperature |
"""

# %%
uw.reset_default_model()
model = uw.get_default_model()

SHELL_THICKNESS = uw.quantity(2200, "km")
KAPPA = uw.quantity(1e-6, "m**2/s")
ETA = uw.quantity(1e22, "Pa*s")
DELTA_T = uw.quantity(2500, "K")

RHO0 = uw.quantity(3300, "kg/m**3")
ALPHA = uw.quantity(3e-5, "1/K")
GRAVITY = uw.quantity(9.81, "m/s**2")

model.set_reference_quantities(
    shell_thickness=SHELL_THICKNESS,
    thermal_diffusivity=KAPPA,
    mantle_viscosity=ETA,
    temperature_contrast=DELTA_T,
)

RAYLEIGH = (RHO0 * ALPHA * DELTA_T * GRAVITY * SHELL_THICKNESS**3 / (KAPPA * ETA))
say(f"Ra = {float(RAYLEIGH.to('dimensionless').magnitude):.3e}")
say(f"diffusion time d^2/kappa = "
          f"{model.get_fundamental_scales()['time'].to('Gyr')}")

# %% [markdown]
"""
## Mesh and variables

An annulus with `radiusInner / radiusOuter = 0.55`, roughly Earth's
core-mantle ratio. The mesh is built in model units; `mesh.X.coords` reads back
in metres because the model declares a length scale.
"""

# %%
R_OUTER = 1.0
R_INNER = 0.55

mesh = uw.meshing.Annulus(
    radiusInner=R_INNER,
    radiusOuter=R_OUTER,
    cellSize=params.uw_cell_size,
    degree=1,
    qdegree=3,
)

v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)

# %% [markdown]
"""
## Stokes: rotated free-slip, and the body force as physics

`add_rotated_freeslip_bc(0.0, boundary)` rotates each boundary node into its
own normal / tangent frame and constrains the normal component strongly. On a
circle that is exact to machine precision; a penalty or Nitsche condition
leaks at around 1e-3, which on a convection run shows up as spurious radial
flow at the boundary.

The buoyancy is written as the force it is, in the units it has. The
nondimensionalisation turns it into `Ra T rhat` — that is where the Rayleigh
number printed above comes from, and writing it this way means the script
never has to be told what Ra is.
"""

# %%
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = ETA
stokes.tolerance = 1.0e-8
stokes.petsc_options.delValue("ksp_monitor")

stokes.add_rotated_freeslip_bc(0.0, "Upper")
stokes.add_rotated_freeslip_bc(0.0, "Lower")

radius = sympy.sqrt(mesh.X.dot(mesh.X))
rhat = mesh.X / radius
# Name the coefficient rather than letting the product collapse into an
# anonymous number. Python multiplies the three quantities at assignment, so
# without this the run records a bare -0.97119 kg/(K m^2 s^2) and nothing
# saying where it came from.
BUOYANCY = uw.expression(
    r"\rho_0 \alpha g",
    RHO0 * ALPHA * GRAVITY,
    "buoyancy coefficient: reference density x thermal expansivity x gravity",
)
stokes.bodyforce = -BUOYANCY * T.sym[0] * rhat

# %% [markdown]
"""
## Transport

`uw.systems.AdvDiffusion` composes an implicit Eulerian SUPG transport step
(Crank-Nicolson by default). Hot inner boundary, cold outer.
"""

# %%
adv = uw.systems.AdvDiffusion(mesh, u_Field=T, V_fn=v.sym)
adv.constitutive_model = uw.constitutive_models.DiffusionModel
adv.constitutive_model.Parameters.diffusivity = KAPPA
adv.add_dirichlet_bc(1.0, "Lower")
adv.add_dirichlet_bc(0.0, "Upper")
adv.tolerance = 1.0e-8
adv.petsc_options.delValue("ksp_monitor")

# %% [markdown]
"""
## Initial condition

A conductive profile with a mode-5 perturbation. Built from the nodal
coordinates in model units, which is what `beta`-style level sets and initial
conditions generally want: `T.coords` reads in metres, so divide by the length
scale once and work in the box's own units.
"""

# %%
LENGTH_SCALE = float(model.get_fundamental_scales()["length"].to("m").magnitude)


def myr(q):
    """A time quantity as a Myr string. `UWQuantity.__format__` delegates to
    the bare float, so pint's `~` format specs do not apply to it."""
    return f"{float(q.to('Myr').magnitude):.4f} Myr"

Xn = np.asarray(T.coords)[:, :2] / LENGTH_SCALE
rn = np.sqrt((Xn**2).sum(axis=1))
thn = np.arctan2(Xn[:, 1], Xn[:, 0])
shell = (rn - R_INNER) / (R_OUTER - R_INNER)

T.array[:, 0, 0] = (1.0 - shell) + 0.1 * np.sin(5.0 * thn) * np.sin(np.pi * shell)

adv.Unknowns.DuDt.initialise_history()
stokes.solve(zero_init_guess=True)

# %% [markdown]
"""
## Diagnostics on the tracker

`model.tracker.time`, `.step` and `.dt` are pre-seeded by convention. Anything
else you assign to it is captured by a snapshot and restored by a rewind, in
the same breath as the fields — which is exactly what a diagnostic wants, and
what a loose Python variable cannot give you.
"""

# %%
v_rms_fn = sympy.sqrt(v.sym.dot(v.sym))
area = float(uw.maths.Integral(mesh, sympy.sympify(1.0)).evaluate())


def v_rms():
    return float(uw.maths.Integral(mesh, v_rms_fn).evaluate()) / area


# %% [markdown]
"""
## The loop

Everything above is ordinary. This is the pattern:

```python
while ...:
    dt = adv.estimate_dt()
    with model.step(dt, label="convect"):
        adv.solve(timestep=dt, zero_init_guess=False)
        stokes.solve(zero_init_guess=False)
```

`record_every = 1` asks each step to keep the state it started from — every
field, the transport history, the clock and the tracker diagnostics together —
captured before the operators run, which is the only correct point.
"""

# %%
model.tracker.time = uw.quantity(0.0, "Myr")
model.tracker.step = 0
model.tracker.dt = None
model.tracker.v_rms = v_rms()

model.record_every = 1
model.record_limit = params.uw_n_steps

# The on-disk transcript needs no setting up: it is on by default, and this run
# will leave one under `transcripts/` beside a copy of this script. Section 5
# shows what landed. To send it elsewhere, or to turn it off:
#
#     model.transcript_file = "output/annulus.log"     # somewhere else
#     model.transcript_file = "output/annulus.jsonl"   # JSON lines, for parsing
#     model.transcript_file = None                     # off
say(f"transcript: {model.transcript_file or '(off)'}")

for _ in range(int(params.uw_n_steps)):
    dt = params.uw_dt_fraction * adv.estimate_dt()

    with model.step(dt, label="convect"):
        adv.solve(timestep=dt, zero_init_guess=False)
        stokes.solve(zero_init_guess=False)
        model.tracker.v_rms = v_rms()

    say(f"step {model.tracker.step:>3d}  "
              f"t = {myr(model.tracker.time)}  "
              f"dt = {myr(dt)}  "
              f"v_rms = {model.tracker.v_rms:.4e}")

# %% [markdown]
"""
## 1. What ran

The transcript is an ordered account of each step: the interval it covered and
the operators it applied, named by what they solve. It answers "is this model
doing the thing the write-up says it does" without the script being
instrumented for it — which is the question you want to ask of someone else's
model, or of your own six months later.

Note the `history_shift` between the two solves. That is the transport history
advancing — the thing that makes a step taken twice visible at all.
"""

# %%
if params.uw_demos:
    say("")
    say("--- 1. the transcript " + "-" * 55)
    for entry in model.transcript:
        say(f"  step {entry.index:>2d}  dt = {myr(entry.dt):>12s}   "
            + " -> ".join(f"{e['kind']}:{e['name']}" for e in entry.events))
    say(f"  {len(model.restore_points)} of {len(model.transcript)} steps "
        f"are restorable")

# %% [markdown]
"""
## 2. A rejected step

A `model.step` block is a transaction. If it does not exit cleanly — an
exception, or a step abandoned because a diagnostic came out wrong — the clock
and the step counter are left exactly as they were, and nothing is added to the
transcript. Backstepping no longer has to remember to unwind a counter.

The fields are yours to restore: take a snapshot before the block, and load it
in the handler. The clock never moved, so the two stay consistent.
"""


# %%
class StepRejected(Exception):
    """Raised inside a step block to abandon it."""


if params.uw_demos:
    say("")
    say("--- 2. a rejected step " + "-" * 51)

    before = (myr(model.tracker.time), model.tracker.step, len(model.transcript))
    snap = model.save_state()          # BEFORE the step, not after

    reckless_dt = 50.0 * params.uw_dt_fraction * adv.estimate_dt()
    try:
        with model.step(reckless_dt, label="too big"):
            adv.solve(timestep=reckless_dt, zero_init_guess=False)
            stokes.solve(zero_init_guess=False)
            if v_rms() > 4.0 * model.tracker.v_rms:
                raise StepRejected("v_rms jumped; the step is not resolved")
    except StepRejected as why:
        model.load_state(snap)
        say(f"  rejected: {why}")

    after = (myr(model.tracker.time), model.tracker.step, len(model.transcript))
    say(f"  clock/step/transcript before : {before}")
    say(f"  clock/step/transcript after  : {after}")
    say(f"  unchanged: {before == after}")

# %% [markdown]
"""
## 3. Playback

`model.rewind()` puts the run back to the start of a completed step — fields,
transport history, clock and tracker diagnostics together — and truncates the
transcript to match, so it continues to describe the run that actually happened.

Replaying the step from there reproduces it exactly. Re-*running* the script
does not: warm starts and preconditioner reuse are solver history rather than
model state, so two independent runs of the same problem diverge at the 1e-13
level from the first step. If you need to look at a step twice, restore it
rather than re-run it.
"""

# %%
if params.uw_demos:
    say("")
    say("--- 3. playback " + "-" * 58)

    T_end = np.asarray(T.array)[:, 0, 0].copy()
    v_rms_end = model.tracker.v_rms

    target = model.rewind()
    say(f"  rewound to the start of step {target.index}: "
              f"t = {myr(model.tracker.time)}, "
              f"v_rms = {model.tracker.v_rms:.4e}  "
              f"(was {v_rms_end:.4e})")

    with model.step(target.dt, label="replay"):
        adv.solve(timestep=target.dt, zero_init_guess=False)
        stokes.solve(zero_init_guess=False)
        model.tracker.v_rms = v_rms()

    T_replay = np.asarray(T.array)[:, 0, 0]
    say(f"  replayed: identical to the original step: "
              f"{np.array_equal(T_replay, T_end)}   "
              f"max |dT| = {np.abs(T_replay - T_end).max():.3e}")

# %% [markdown]
"""
## 4. Recording, not judging

Call a solver twice inside one step — a predictor/corrector, a Picard iteration
on the coupled system, a retry — and its history advances twice, so the
physical step is taken twice. The timestep history and the solve counter look
identical to a single step, so the transcript is the only place it shows.

The step records that and says nothing about it. Whether two shifts in one bar
are a mistake or legitimate sub-cycling is a reading of the transcript, made by
a later pass that can look across bars; inside the loop it would have to be
guessed. See `docs/developer/design/run-score-and-transcript.md`.

What you see below is the bar's operator sequence with everything in it twice —
which is also why the figure gives that bar its own letter.
"""

# %%
if params.uw_demos:
    say("")
    say("--- 4. a step taken twice " + "-" * 47)

    dt = params.uw_dt_fraction * adv.estimate_dt()
    with model.step(dt, label="taken twice"):
        adv.solve(timestep=dt, zero_init_guess=False)     # a "predictor"
        stokes.solve(zero_init_guess=False)
        adv.solve(timestep=dt, zero_init_guess=False)     # and a "corrector"
        stokes.solve(zero_init_guess=False)

    entry = model.transcript[-1]
    shifts = [e for e in entry.events if e["kind"] == "history_shift"]
    say(f"  history shifts in this one step: {len(shifts)}")
    say(f"  the step as recorded: {entry}")

# %% [markdown]
"""
## 5. The log on disk

The transcript lands on disk without being asked, in a stamped directory under
`transcripts/` beside a copy of the script that launched it and a `launch.json`
recording what invoked it. One line per step, flushed as it closes — so
`tail -f` follows a running job, and a run that is killed keeps everything up
to the moment it died. Nothing is created for a script that never takes a step.

Three differences from `model.transcript`, all deliberate. An **abandoned** step
appears in the file and not in memory. A step aged out by `transcript_limit`
leaves memory but stays in the file. And a **backtrack** — `rewind()` or a
bare `load_state()` — writes its own line, because a log that shows step 7 and
then step 7 again, with nothing in between, is not a log of what happened.
"""

# %%
if params.uw_demos:
    say("")
    say("--- 5. the log on disk " + "-" * 51)
    say(f"  {model.transcript_file}")

    run_dir = os.path.dirname(model.transcript_file) if model.transcript_file else ""
    if run_dir:
        say(f"  the run directory holds: "
            f"{', '.join(sorted(os.listdir(run_dir)))}")
    say("")
    with open(model.transcript_file, encoding="utf-8") as handle:
        for line in handle.read().splitlines():
            say("  " + line)

# %% [markdown]
"""
## 6. The same account, as a figure

A terminal is not where a run belongs in a paper. `uw.transcript_diagram` renders
the record with **time running down the page** — one row per step, A4 portrait,
paginated — and writes it as a PDF, which opens anywhere, or an SVG if the
suffix says so. Both are written directly: no plotting library, no
rasterisation, no theme to fight with. `uw.transcript_flowchart` renders one
step's operator flow as Mermaid, for dropping into documentation.

The layout decision worth knowing about: each distinct operator sequence gets a
**letter**, defined once at the foot of the figure. A column of `A` with a
single `B` in it says at a glance that one step did something different, where
a hundred spelled-out sequences say nothing and hide the one that matters. The
doubled step below is found that way rather than by reading.

Both take a live model, which matters here because the text log is a report and
cannot be read back — pass a `.jsonl` log or the model itself.
"""

# %%
if params.uw_demos:
    say("")
    say("--- 6. the figure " + "-" * 56)

    stem = model.transcript_file.rsplit(".", 1)[0]   # beside the transcript
    say(f"  {uw.transcript_diagram(model, out=stem + '.pdf', title='Annulus convection - run log')}")
    say(f"  {uw.transcript_diagram(model, out=stem + '.svg', title='Annulus convection - run log')}")
    say("")
    for line in uw.transcript_flowchart(model).splitlines():
        say("  " + line)

# %%
say("")
say(f"final: t = {myr(model.tracker.time)}, "
          f"{model.tracker.step} steps, "
          f"v_rms = {model.tracker.v_rms:.4e}")
