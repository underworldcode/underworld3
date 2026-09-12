---
title: "A Transcript for Every Run"
status: draft
feeds_into: [paper-2, release-post]
target: underworldcode.org (Ghost)
figures: figures/run-transcripts
---

Six months after a result, the questions are always the same: which version of
the code produced it, what the parameters were, and whether the run did what
the caption says it did. Underworld 1 could answer all three from a single
file. Underworld3 could answer none of them, until recently, and the reason is
worth setting out, because it is a consequence of what makes the current code
usable at all.

## The Underworld 1 answer

A model in Underworld 1 was an XML document. The launcher read it, StGermain
assembled the components it named, and the run followed. Provenance came for
free: the XML *was* the model, so keeping it kept everything. Two people with
the same XML and the same binary were running the same experiment, and could
say so without qualification.

The cost was that the XML was a programming language without being one. It had
no expressions worth the name, no control flow a reader could follow, and no
debugger. Anything the schema had not anticipated could not be said, and a
model that needed something new needed C. A great deal of scientific intent
ended up encoded as combinations of components that happened to compose, which
is a poor medium for explaining what a model does.

## Why that answer is not available now

A model in Underworld3 is a Python program. The library takes SymPy expressions
for constitutive behaviour, boundary conditions and source terms; it puts no
constraint on what happens between solves; and the timestep loop is written by
the user. That openness is deliberate, and it is the same property that makes
the library drivable by a colleague who has never used it, or by a language
model: the surface is compositional, so a reader can predict what a call will
do from what the pieces mean.

It also means no serialisable model document exists, even in principle. The
model is an arbitrary program, and what it does is a sequence of calls —
including calls whose arguments depend on runtime state. Nothing can be
recovered from the objects afterwards, because the objects do not record the
order in which anyone touched them.

That is a real trade. Underworld 1 had a complete statement of intent and very
little expressive power. Underworld3 has the expressive power and, until
recently, kept no statement of anything at all.

## What a run leaves now

Every Underworld3 run that takes a timestep writes a **transcript**: a record
of what it did, appended and flushed as each step closes. No configuration is
involved. A run that takes no step writes nothing, and a run under `pytest`
writes nothing, so the default costs nothing where it would only be noise.

```
transcripts/2026-09-12T03-31-34-make_figures/
    make_figures.py      the script that launched it, verbatim
    launch.json          argv, interpreter, cwd, versions, commit
    transcript.log       one aligned line per step, flushed
    transcript.jsonl     the same record, machine-readable
```

The directory is stamped with the time the run started, so a second run does
not overwrite the first. The path is printed when the run begins and again
when it ends, and a `latest` symlink points at the most recent one.

`launch.json` is the replacement for the XML, and it is a weaker thing
honestly labelled. It cannot state what the model *is*; it states what was
*run*:

```json
{
  "started": "2026-09-12T03:31:34+00:00",
  "argv": ["make_figures.py"],
  "python": "3.12.12",
  "underworld3": "0.0.0",
  "mpi_size": 1,
  "git_commit": "638a6c3c5a3859fc40e2a87f4f918ba9f3aebcb1",
  "git_dirty": true,
  "script": "make_figures.py"
}
```

The entry script is copied beside it. Modules it imports are not, which is
what the commit id is there to cover, and the file says so rather than leaving
a reader to find out. `git_dirty` is the field that earns its place: a commit
id with uncommitted changes on top of it identifies nothing, and recording the
flag is the difference between provenance and the appearance of it.

## Where this lands against FAIR

**Findable** is the stamped directory, the `latest` pointer and the announced
path. A run that produced a figure can be located without anyone having
remembered to write down where it went.

**Accessible** is the two file formats. The text transcript is read with
`tail -f` while a job is running; the JSON Lines record is read with `jq`, or
with `uw.read_transcript`. Neither needs Underworld3 installed to open.

**Interoperable** is the way dimensional values are stored. Each carries its
own magnitude and unit string — `{"magnitude": 4.31, "units": "megayear"}` —
rather than a bespoke convention that a reader has to be told about.

**Reusable** is the launch record together with the transcript. The two
together let someone decide what to change, which is what reuse requires.

## Provenance is not reproducibility

The transcript identifies a run. It does not promise that running the script
again produces the same numbers, and on this code it will not. Warm starts and
preconditioner reuse are solver history that sits outside model state, so two
independent runs of the same script on the same machine diverge at the 1e-13
level from the first step.

Restoring is exact where re-running is not. A step restored from the snapshot
it began with and taken again reproduces its temperature field bit for bit —
`max |ΔT| = 0` across every step of the annulus run below. That asymmetry is
the practical reason a run keeps its own restore points: a step that
misbehaved can be looked at twice, which re-running cannot give you.

## Reading a run

The transcript is complete, which creates the problem a profiler has: every
step is in it, and most steps are identical. Removing the repetitive ones by
hand would be a judgement about what mattered. A **chart** removes them by
rule instead — it groups consecutive steps that did exactly the same thing,
and carries the first and last value of anything that changed across the
group, so a timestep that grew by a factor of eight survives the grouping.

```
transcript · model 'default'
started 2026-09-12T15:22:22+00:00
no terminator: this run is still going, or it was interrupted. What follows is the transcript of a prefix.

 step      t/Myr     dt/Myr │ AdvectionDiffus │    Stokes(v)    │ EulerianSUPG(T) │
───────────────────────────────────────────────────────────────────────────────────
    0   0.175907   0.175907 │        1        │        3        │        2        │
  …13    16.8708    4.31057 │        ↓        │        ↓        │        ↓        │   ×13 unchanged, dt 0.1759 → 4.311
   14    446.031    429.161 │        1        │        3        │        2        │   ABANDONED
                            │ restore from a snapshot; the clock now reads 16.8708 Myr
                            │ rewind to the start of step 13 (t = 12.5602 Myr); 1 step(s) undone
   13    16.8708    4.31057 │        1        │        3        │        2        │
   14     25.454    8.58321 │       1,4       │       3,6       │       2,5       │

17 step(s), 3 part(s): AdvectionDiffusion(T), Stokes(v), EulerianSUPG(T)
↓  the steps between did exactly this, unchanged
·  this part did nothing in that step
digits are the order the parts ran within the step
```

Each column is a participant: the advection-diffusion solver, the Stokes
solver, and the transport history the first of them holds. The digits give the
order they ran within the step. The grouped line asserts that steps 1 to 13
were *identical* to step 0 in what ran and in what order, so nothing can hide
behind it. Seventeen steps become six rows, and the rows that remain are the
ones a reader would have picked out. Grouping is optional, because a run whose
timestep is itself the thing under examination is easier to read one row at a
time.

Three things are visible in those six rows, and in @fig-chart, that no print
statement was written to report. Step 14 attempted 429 Myr and was rejected on a velocity
diagnostic, so the run went back to step 13 and took it again. The replayed
step 13 took 0.30 s of wall clock against the original's 0.06, because the
restore discarded the warm start. And the last step ran everything twice —
`1,4 │ 3,6 │ 2,5` — which is a predictor-corrector written without noticing
that the transport history advances on every solve, so the temperature
advanced two intervals while the clock advanced one.

```{figure} figures/run-transcripts/run-chart.svg
:label: fig-chart
:alt: A chart of a 17-step annulus convection run. Three columns — AdvectionDiffusion(T), Stokes(v) and EulerianSUPG(T) — each carry a filled mark per step with the order it ran: 1, 3 and 2. Steps 1 to 12 group into one shaded band with a downward arrow in each column, labelled x12, showing t running 0.5015 to 12.56 Myr and dt 0.3256 to 2.704 Myr. Step 14 at 446.031 Myr is drawn with hollow dashed marks and labelled abandoned. A dashed red arrow in the left gutter labelled "rewind 1 (+1)" runs back from it to step 13, and a blue arrow labelled "again" runs down to the replayed step 13. The final step carries two marks in every column, numbered 1 and 4, 3 and 6, 2 and 5.

The same run as a chart. Each column is a participant and each row a step; a
filled mark carries the order that participant ran within the step. Steps 1 to
12 are grouped into one band, which asserts that they did exactly what step 0
did, and reports the range of `t` and `dt` across them. The final step carries
two marks in every column, which is the predictor-corrector advancing the
transport history twice.
```

```{figure} figures/run-transcripts/run-transcript.svg
:label: fig-run-transcript
:alt: A portrait figure of a 17-step annulus convection run. Steps run down the page with dt as a horizontal bar; dt grows from 0.18 to 4.31 Myr over the first fourteen. Step 14 is drawn hatched in red at 429 Myr and marked abandoned. A dashed arrow in the left gutter runs back from it to step 13, and a solid arrow labelled "again" runs down to the replayed step 13. The final step carries the letter B where every other step carries A.

The same run as a figure. Time runs down the page, `dt` across it, and the two
backtracks are drawn in the left gutter: back out of the abandoned step, then
down to the step that was taken again. Each distinct operator sequence gets a
letter, defined once at the foot, so the one step that did something different
is the one letter that differs.
```

The figure and the chart are rendered from the JSON record after the run, or
during it — the chart above is of a run still in progress, which is why it
says so. Both are produced by
[`make_figures.py`](figures/run-transcripts/make_figures.py), which is also
the run they describe.

## What it costs, and what it does not do

A step appends two lines, one to each format, and flushes. The snapshots that
make `rewind` possible cost about 13 bytes per primary degree of freedom per
step and are off unless asked for; the transcript itself is bytes.

Three things are not in it yet. A part that does nothing in a step does not
appear, so silence and absence are indistinguishable. Mesh deformation and
adaptation are not recorded as events, so a history stored before the mesh
moved and read after it has nothing in the record to flag it. And what a step
is supposed to contain is inferred from what repeated rather than declared by
the script, so the record can say that one step differs from its neighbours
and cannot yet say that a run disagrees with its own description.

The last of those is where the Underworld 1 comparison ends up. A declared
plan would be the XML's descendant — a statement of what a step is supposed
to contain — checked against the transcript rather than executed from it. The
document would describe the model without having to be the only way to express
it.
