# The run score and the run transcript

*Status: design note. Vocabulary and the model it implies. No implementation.*

## Why this note exists

Underworld3 records what a run did — `model.transcript`, the on-disk log, the
figure. Building those made it clear that the hard part is not the recording
but the **view**: the record has to make a *relation* visible, and until we
knew which relation, each addition was a patch on the last.

The relation is this. A value stored by one action in one cycle is picked up by
a read in the next. Every silent defect found while building the timestepping
machinery was a read taking the wrong write:

- an Eulerian history that initialises only on its first solve, so a solver
  reused for a second run silently reads the **previous run's** store
- the history-before-advection ordering trap
- `old_frame_traceback` on a deforming mesh (#423), where a semi-Lagrangian
  history recorded in the old frame and read after the mesh moved amplifies
  ~10% per cycle, with no error and no symptom other than the growth rate
- reading a solver's `F0`/`F1` templates before the solve configured them

None of those is visible in a single cycle. All of them are structure across
cycles.

## Two documents, not one

**The score** is what the run is *supposed* to do: which parts play, in what
order, with what meter. It is the same for every cycle of a well-behaved run.

**The transcript** is what the run *actually did*, including the things that
are not in the piece — a cycle abandoned, a jump back to an earlier cycle, a
re-take.

Both already exist in the current implementation without those names. The
figure's operator-sequence legend (`A`, `B`, ...) is an **inferred score**; the
rows are the **transcript**. A run in which every bar is `A` played the score.
A `B` is a bar played differently.

That naming makes the original motivating question mechanical: *is the model
doing the scientific task you say it is* becomes **a diff of the transcript
against the score**.

## Vocabulary

| term | meaning here |
|---|---|
| **bar** | one turn of the orchestrating loop. Numbered monotonically, never reused, and the thing you refer to. NOT necessarily a physical time interval — it may be a task. |
| **beat** | a position inside a bar at which alignment is required. **Barriers** sit on beats: a mesh deform, an adapt, a migration, a remesh. |
| **part** | a participant with its own stave. Two kinds: *actors* (solvers, swarm pushes, mesh movers) and *state-holders* (DDt histories, fields, particle coordinates). |
| **note** | what a part did in a bar, carrying its own duration — its `dt`, which need not be the bar's. |
| **rest** | notated absence. Distinguishes *did nothing this bar* from *was not being watched*. |
| **tuplet** | n notes in the space of the bar, bracketed with the ratio. Sub-cycling, notated as ordinary rather than flagged as anomalous. |
| **tie** | a value written in one bar and read in the next, drawn as an arc across the barline. |
| **tempo** | how bar numbers map to real time. Deliberately separate from the meter: `dt` varies, wall clock varies more, and the vertical axis is ordinal. |
| **performance event** | not part of the piece: a bar abandoned, a jump back, a re-take. Transcript only. |

Two conventions borrowed with the vocabulary and worth keeping:

- **Score order.** Parts appear in a stable, conventional order (actors, then
  histories, then swarms, then mesh), not order of first appearance, so a
  reader finds the same part in the same place in every run's transcript.
- **Rests are compulsory.** In a score every part accounts for every beat. A
  part that does nothing is written as a rest, never left blank.

## Why two dimensions

Different parts advance on different `dt`. A swarm may sub-cycle twice for one
Stokes solve; two histories on the same field may be at different orders. Those
cannot be laid on one axis without pretending they share a clock — which is the
failure being looked for.

So: **parts across the page, bars down the page.** Vertical is ordinal, not
time. A part whose accumulated time drifts away from its neighbours' is then a
visible misalignment rather than something you would have to instrument for.

## What the notation makes checkable

Each check is a reading of the notation rather than a separate assertion:

| reading | defect |
|---|---|
| a tie with no note at its head | a read of uninitialised history |
| two notes tied into one read | the physical step taken twice |
| a tuplet whose ratio does not fill its bar | sub-cycling that fails to tile the interval |
| a tie crossing a beat that carries a barrier, with nothing re-expressing it | the `old_frame_traceback` class (#423) |
| transcript ≠ score | the model is not doing what the script says |

**None of these belong in the loop.** An earlier version asserted one of them
there — "a history must advance exactly once per bar" — and it was wrong twice
over: it would fire on legitimate sub-cycling, and it could not see the check
that matters most, since #423's signature is a growth rate across bars. It has
been removed. Derived from the notation the rule is the honest one — **the
notes in a bar tile its interval exactly once** — and sub-cycling satisfies it.

The separation that follows: **execution records, analysis judges.** A step
raises only on structural failures, where the transcript could not be
well-formed — a step opened inside another step, a rewind to a bar that kept no
snapshot. Everything above is a *finding*, produced by a pass over a finished
transcript, which can look across bars, can be re-run on an old transcript when
a new pathology is learned, and never has to decide mid-run whether something
was deliberate.

## Where reads and writes come from

Feasibility, because this is the part that decides whether the model is
buildable:

- **Writes** come from the runtime hooks already in place: the unknown after a
  solve, `psi_star[i]` in `update_post_solve`, particle coordinates after an
  advection.
- **Reads** are derivable *symbolically*. A solve's residual is a SymPy form,
  so the variables it depends on are in `F0` / `F1`'s atoms. This is how the
  discrete adjoint obtained `∂F/∂ψ*` at all. No kernel instrumentation.

Within one solve, reads and writes are not ordered — the kernel reads
`psi_star` throughout the Newton iteration — so the edge is *write → solve*,
not write → instant. The cell is atomic; the meaning is in the edges between
cells. That is exactly where the cross-cycle relation lives, so the limitation
does not bite.

## What is missing today

Everything above except two items is a renaming of something already captured,
which is a reasonable sign the vocabulary fits rather than being imposed.

1. **Barriers are not events.** `_deform_mesh` does not declare itself, so
   there is no beat to draw the rule at — and the check that most wants the
   barrier (#423) has no anchor without it.
2. **Rests are not recorded.** A part that does nothing in a bar simply does
   not appear, so silence and absence are indistinguishable.

A third, smaller: **a bar is not necessarily an interval, and `model.step(dt)`
insists that it is.** The signature requires a `dt`, so there is no container
for "the next task". If the event clock is the general thing, the timestep is
the common case rather than the definition.

## Inferred score, then declared score

The score is **inferred** today — the figure takes the most common bar as the
norm. That costs nothing and can only ever say *this bar differs from its
neighbours*.

A **declared** score — the script stating what a bar is supposed to contain —
turns the diff into *this run disagrees with its own description*, which is the
stronger claim and the one that motivated the work. The path is to work
towards the declarative model from the inferred one rather than to require it
up front; nothing in the vocabulary above depends on which we have.

## A note on names

"Transcript" rather than tape or record: a transcript is what was actually
played, including the false starts and the re-takes, which is precisely the
thing being kept. The word also carries its own contrast with the score, so
the pair names itself.

The API followed: what was `model.journal` is `model.transcript`, and the
renderers follow from the pair — `transcript_diagram` draws the transcript,
`transcript_flowchart` draws the score it implies.

"Record" is kept as a **verb**. A step records what it did; the thing it
produces is the transcript.
