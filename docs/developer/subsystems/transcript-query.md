# Querying a run's transcript

A run writes a transcript: a header with the scales, one line per step with
the operators it applied and how each went, the description of each part
when it first acts, and notes for what was not a step, a rewind above all.
`uw.read_transcript` reads it back as data. `uw.Transcript` puts one query
object over that data, so a notebook, a test, the digest and a tool ask the
same questions of the same interpretation.

```python
t = uw.Transcript("transcripts/latest/transcript.jsonl")   # a path, a live model, or read_transcript's list
t.view()                        # the summary, rendered for the session
t.view(format="yaml")           # the same as data
```

## The questions

| call | answers |
|---|---|
| `t.abandoned()` | steps that did not commit, each with `abandoned_by`: the exception's class and message |
| `t.backtracks()` | rewinds and restores, with where they happened, the step they went back to, and the `reason` and `detail` the caller gave |
| `t.failed()`, `t.capped()` | solves that diverged; solves the SNES called converged while an inner block hit its cap or its deadline |
| `t.solves(part=...)`, `t.events(kind=..., outcome=..., step=...)` | events as `(step_index, event)`, filtered |
| `t.patterns()` | the run collapsed to its distinct step patterns: same operators, outcomes, label and completion, with nothing recorded between |
| `t.step(i)`, `t.sequence(i)` | one step as recorded; its operators in order |
| `t.compare(a, b)` | operators in one step and not the other, outcomes that changed, and the interval, wall time and completion of each |
| `t.part(name, at_step=i)` | what a part was solving at step `i`: the description recorded at or before it |
| `t.changes()` | parts whose form changed during the run, and when |
| `t.between(t0, t1)` | steps starting in an interval of the run's own time |
| `t.adjoint_segments()` | the run partitioned by adjoint support |

Every answer is plain data, the dicts the record holds. The outcome of a
solve is read by the same rule the figure and the table use, so a solve the
digest marks amber is the one `capped()` returns.

## Recording decisions

The transcript cannot infer why a run went back, since the acceptance test
lives in the caller's loop. Say so when rewinding:

```python
if displacement > limit:
    model.rewind(1, reason="free surface displacement over the limit",
                 observed=displacement, threshold=limit, action="halve dt")
    dt = dt / 2
```

The note then carries `reason` and `detail`, and `t.backtracks()` returns
them. A step abandoned by an exception records the exception's class and
message as `abandoned_by` without anything from the caller.

## The same tree as everything else

`t.describe()` is a record in the shape every object uses (see
[Descriptions and views](describe-and-view.md)): facts for the header and
the counts, the step patterns, and the parts as children. `uw.render(...)`
turns it into Markdown, text, LaTeX, YAML or JSON, and the renderers
`uw.transcript_table`, `uw.transcript_figure` and `uw.transcript_key` take
the query object as their source.
