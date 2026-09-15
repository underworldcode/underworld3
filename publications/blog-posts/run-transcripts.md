---
title: "A Transcript for Every Run"
status: draft
feeds_into: [paper-2, release-post]
target: underworldcode.org (Ghost)
figures: figures/run-transcripts
---

Six months after a result, the questions are always the same: which version of
the code produced it, what the parameters were, and whether the run did what
the caption says it did. Underworld 1 could answer all three from one file.
Underworld3 could answer none of them until recently, and the reason is worth
a paragraph, because it is a consequence of what makes the current code usable
at all.

## Why a Python model has no document to keep

A model in Underworld 1 was an XML document. The launcher read it, StGermain
assembled the components it named, and the run followed. Provenance came for
free: the XML *was* the model, so keeping it kept everything. The cost was that
the XML was a programming language without being one — no expressions worth
the name, no control flow, no debugger — and anything the schema had not
anticipated needed C.

A model in Underworld3 is a Python program. The library takes SymPy
expressions for constitutive behaviour, boundary conditions and sources, it
puts no constraint on what happens between solves, and the timestep loop is
written by the user. That openness is what makes the library drivable by a
colleague who has never used it, or by a language model: the surface is
compositional, so a reader can predict what a call does from what the pieces
mean. It also means no serialisable model document exists, even in principle.
The model is a program, and what it did is a sequence of calls whose arguments
depended on runtime state.

So we keep a record of what the run *did*. The rest of this post is about
what that record looks like, because a record nobody can read is a record
nobody checks.

## What a run leaves

Every run that takes a timestep writes a transcript. There is nothing to
configure, a run that takes no step writes nothing, and a run under `pytest`
writes nothing, so the default costs nothing where it would only be noise.

```
transcripts/2026-09-15T13-02-27-Ex_Convection_Annulus_Recorded/
    Ex_Convection_Annulus_Recorded.py    the script that launched it, verbatim
    launch.json                          argv, interpreter, cwd, version, commit
    transcript.log                       one aligned line per step — for watching
    transcript.jsonl                     one JSON object per step — for reading back
    transcript.svg, transcript.pdf       the run as a figure
transcripts/latest -> 2026-09-15T13-02-27-Ex_Convection_Annulus_Recorded
```

The directory is stamped, so this morning's run is still there this afternoon;
`latest` points at the newest; the path is announced when the run ends. The
script is copied because a run's own launch line is the first thing a reader
wants and the first thing that is gone.

## One line per step, aligned, flushed

The text log is written to be watched. Each step is one line, appended and
flushed as the step closes, so `tail -f` follows a running job and a job that
is killed keeps every step it finished.

```
# underworld3 run transcript · Ex_Convection_Annulus_Recorded · started 2026-09-15T13:02:27+10:00
# scales: length 2.2e+06 m | time 4.84e+18 s | mass 1.065e+47 kg | temperature 2500 K
  -- solves: AdvDiffusion(T)  [F0, F1]  (the form is in the record; uw.transcript_key renders it)
  -- solves: Stokes(v)  [F0, F1, PF0]  (the form is in the record; uw.transcript_key renders it)
# step           t/Myr          dt/Myr    wall/s  outcome    operators, in order
      0        0.175907        0.175907      0.26  ok         [convect] AdvDiffusion(T) > shift EulerianSUPG(T) > Stokes(v)
      1        0.501546        0.325639      0.06  ok         [convect] AdvDiffusion(T) > shift EulerianSUPG(T) > Stokes(v)
      2        0.990939        0.489393      0.06  ok         [convect] AdvDiffusion(T) > shift EulerianSUPG(T) > Stokes(v)
     ...
     11         9.85635         1.92019      0.06  ok         [convect] AdvDiffusion(T) > shift EulerianSUPG(T) > Stokes(v)
     12         145.049         135.192      0.06  ABANDONED  [too big] AdvDiffusion(T) > shift EulerianSUPG(T) > Stokes(v)
  -- restore from a snapshot; the clock now reads 9.85635 Myr
  -- rewind to the start of step 11 (t = 7.93616 Myr); 1 step(s) undone
     11         9.85635         1.92019      0.30  ok         [replay] AdvDiffusion(T) > shift EulerianSUPG(T) > Stokes(v)
     12         12.5602         2.70384      0.12  ok         [taken twice] AdvDiffusion(T) > shift EulerianSUPG(T) > Stokes(v) > AdvDiffusion(T) > shift EulerianSUPG(T) > Stokes(v)
# run ended after 13 recorded step(s) · 2026-09-15T13:02:29+10:00
```

Every column is there because someone watching a run wants it. `t` and `dt`
are in one unit, chosen once from the run's own scales and named in the
header, so a reader is not converting between lines. `wall/s` is the first
number to move when a solver gets into trouble — the
replayed step 11 took 0.30 s against the original's 0.06, because the restore
discarded the warm start. The label in brackets is the name the script gave
the step, so a run that does different things at different times says which.
And the operators are listed in the order they ran, which is the thing a
caption asserts and a reader cannot otherwise check.

The names are the names the user wrote. The record itself holds the class that
implemented each operator — `SNES_AdvectionDiffusion_Composed` — but the log
says `AdvDiffusion(T)`, which is what the script said, and `Stokes(v)`, and
`shift EulerianSUPG(T)` for the transport history that advanced. The same
vocabulary is used by every view of the record, so a name learned in the log
is the name found in the figure.

## How it went, not just that it ran

The `outcome` column reads `ok`, `DIVERGED` or `ABANDONED`. A solve that did
not converge is named under its step, with the reason and the work:

```
      7         2.63841        0.523655      4.81  DIVERGED   [convect] Stokes(v) > ...
  !! Stokes(v): DIVERGED_LINEAR_SOLVE after 6 its (1200 ksp), |F| 3.11e-04
```

Behind the column, each solve's record carries the converged reason, the
nonlinear and Krylov counts, the residual norm, and whether a fieldsplit block
hit its iteration cap. That last one is worth its own state: a Stokes solve
whose velocity block gave up reports converged, and the pressure it returned
was solved on an operator that was still moving. The record says which, and
the figure marks it.

Warnings raised inside a step are recorded with where they came from:

```
  ~~ RuntimeWarning: Stokes: the velocity block fell back to 'gamg' — no mesh hierarchy was available...
```

"The velocity block fell back to gamg" changes what the numbers mean. A record
that kept the residual norms but not that line would be an account of the run
with the explanation removed.

## Notes only when something changes

A log that repeats itself is a log nobody reads. The step lines repeat because
each is a fact; everything else is written once, when it happens. A restore
and a rewind are notes between the steps they sit between, so a log that shows
step 11, then step 11 again, says what happened in between. Identical warnings
within a step are written once with a count. The one-line-per-step promise
holds, and the notes are the places to look.

## The chart: collapsing what repeated, by rule

The log is complete, and that creates the problem a profiler has: most steps
are identical, and the ones that matter are buried. Removing the repetitive
ones by hand would be a judgement about what mattered. The chart removes them
by rule instead — consecutive steps that did exactly the same thing collapse
into one line, which carries the first and last value of anything that changed
across them.

```
 step      t/Myr     dt/Myr │ AdvDiffusion(T) │ EulerianSUPG(T) │    Stokes(v)    │
───────────────────────────────────────────────────────────────────────────────────
    0   0.175907   0.175907 │        1        │        2        │        3        │
  …11    9.85635    1.92019 │        ↓        │        ↓        │        ↓        │   ×11 unchanged, dt 0.1759 → 1.92
   12    145.049    135.192 │        1        │        2        │        3        │   ABANDONED
                            │ restore from a snapshot; the clock now reads 9.85635 Myr
                            │ rewind to the start of step 11 (t = 7.93616 Myr); 1 step(s) undone
   11    9.85635    1.92019 │        1        │        2        │        3        │
   12    12.5602    2.70384 │       1,4       │       2,5       │       3,6       │
```

Each column is a participant, and the digits give the order it ran within the
step. The collapsed line asserts that steps 1 to 11 were *identical* to step 0
in what ran and in what order, so nothing hides behind it — and it still says
that `dt` grew from 0.18 to 1.92 Myr across them, because a timestep that
grew tenfold is a diagnostic. Fifteen records become six rows, and the rows
that remain are the ones a reader would have picked out: the step that tried
135 Myr and was rejected, the step taken again, and the last step that ran
everything twice — `1,4 │ 2,5 │ 3,6` — a predictor-corrector written without
noticing that the transport history advances on every solve, so the
temperature advanced two intervals while the clock advanced one.

Collapsing is optional. A run whose timestep is itself the thing under
examination is easier to read one row at a time.

## The figure

```{figure} figures/run-transcripts/run-chart-key.svg
:label: fig-chart
:alt: The annulus run as a chart. Three columns — AdvDiffusion(T), EulerianSUPG(T), Stokes(v) — and a bar per step in which the marks descend, one line each, joined by a path that steps down and to the right; steps 1 to 10 collapsed into one band; step 12 marked abandoned with a rewind arrow back to step 11 and an arrow down to the replayed step 11; the last step's bar twice as tall, its path making two descents. Below, a legend, and a key listing each part's named quantities with values and units and its boundary conditions.

The same run as a figure. Each step is a bar, and inside it what ran sits one
line below what ran before, in its own column, joined by a path: the shape of
the path is the sequence. A step that runs everything twice is twice as tall,
and shows it. Each mark says how the solve went — converged, converged with a
block at its cap, or diverged. A run of identical steps collapses into a band
carrying the range of anything that changed. The backtrack is drawn where it
happened: back out of the abandoned step, down to the step taken again. The
key beneath is the model, by the quantities in its residuals.
```

The three marks are the three things a solve can do. A page that says every
solve converged is a page that can be skimmed; a page with one amber mark in
three hundred is a page that says where to look. The figure is SVG for the
web and PDF for print, drawn without a plotting dependency, so the record and
its figure need nothing installed to be looked at.

```{figure} figures/run-transcripts/run-transcript.svg
:label: fig-run-transcript
:alt: The same run with time down the page and dt as a horizontal bar per step, growing from 0.18 to 1.92 Myr; the abandoned step drawn as a dashed bar; the rewind and replay arrows in the gutter; each row lettered by its operator sequence, with the sequences defined once at the foot.

Time down the page, `dt` across it. Each distinct operator sequence gets a
letter, defined once at the foot, so the one step that did something different
is the one letter that differs.
```

## The equations, as implemented

The header lines say each solver's form is in the record. It is: each solver
records the residual it assembled — the templates with their own symbols and
docstrings, the named expressions inside them expanded down to the
constitutive model, the boundary conditions, and the terms it was given — once
per run and again if it changes. `uw.transcript_key` renders that as a key, in
Markdown with LaTeX for a note or a notebook, or in plain text for a terminal.
For the Stokes solver in the run above:

$$\mathbf{f}_0(\mathbf{u}) = \left[\begin{matrix}\dfrac{x\,\rho_0 \alpha g\,T}{\sqrt{x^{2} + y^{2}}}\\[6pt] \dfrac{y\,\rho_0 \alpha g\,T}{\sqrt{x^{2} + y^{2}}}\end{matrix}\right]
\qquad
\mathbf{F}_1(\mathbf{u}) = 2\eta\,\dot{\boldsymbol\varepsilon} + \eta\lambda\,(\nabla\cdot\mathbf{v})\,\mathbf{I} - p\,\mathbf{I}$$

where $\rho_0 \alpha g = 0.9712\ \mathrm{kg/K/m^{2}/s^{2}}$ — buoyancy
coefficient: reference density × thermal expansivity × gravity; $\eta =
10^{22}\ \mathrm{Pa\cdot s}$ — shear viscosity; $\lambda = 0$ — numerical
penalty; with rotated free-slip on the upper and lower boundaries,
$\mathbf{u}\cdot\hat{\mathbf{n}} = 0$.

Two things make that more than a pretty-print. The equation is read from the
solver at the moment it solved, so it is the equation that ran, not the one
the script intended. And the names are the user's: the buoyancy coefficient
appears under the name it was given, with its description, rather than as the
number it collapsed to. A coefficient written as an anonymous float would have
appeared as a float. The key is where the discipline of naming things pays
off, and it pays off in the record rather than in the source.

## Reproducing a result from the record

Restoring is exact where re-running is not. Two independent runs of the same
script on the same machine diverge at the 1e-13 level from the first step,
because warm starts and preconditioner reuse are solver history that sits
outside model state. A step restored from the snapshot it began with and taken
again reproduces its temperature field bit for bit — `max |ΔT| = 0` across
every step of the run above.

That asymmetry is why a run can keep its own restore points.
`model.record_every = 1` keeps the state each step started from — fields,
transport history and clock together — and `model.rewind()` returns to it;
`read_transcript()` gives the steps back in order with the interval each
covered and the operators it applied, and any recorded step can be re-entered
with `load_state(entry.snapshot)`. A step that misbehaved can be looked at
twice, which re-running the script cannot give you. Snapshots cost about 13
bytes per primary degree of freedom per step and are off unless asked for;
the transcript itself is bytes.

## Where this lands against FAIR

*Findable* is the stamped directory, the `latest` pointer and the announced
path. *Accessible* is the two formats: the log for `tail -f`, the JSON Lines
for `jq` or `read_transcript`, neither needing Underworld3 to open.
*Interoperable* is how dimensional values are stored — each with its own
magnitude and unit string, `{"magnitude": 4.31, "units": "megayear"}`, rather
than a convention a reader has to be told. *Reusable* is the launch record and
the key together: what ran it, and what it solved.

## What is not in it yet

A part that does nothing in a step is marked as having done nothing only in
the chart, from the roster of parts that played; the log does not mention it.
Mesh deformation and adaptation are not recorded as events, so a history
stored before the mesh moved and read after it has nothing in the record to
flag it. And what a step is *supposed* to contain is inferred from what
repeated rather than declared by the script, so the record can say that one
step differs from its neighbours and cannot yet say that a run disagrees with
its own description. That last one is where the Underworld 1 comparison ends
up: a declared plan would be the XML's descendant, a statement of what a step
should contain, checked against the transcript rather than executed from it.
