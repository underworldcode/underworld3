r"""A run's transcript, queryable.

The transcript on disk is a record of what a run did: a header with the
scales, one line per step with the operators it applied and how each went,
the description of each part when it first acted, and notes for the things
that were not steps, a rewind above all. :func:`underworld3.read_transcript`
reads it back as data; this module puts one query object over that data so
a notebook, a test, the digest and a tool all ask the same questions of the
same interpretation.

    t = uw.Transcript("transcripts/latest/transcript.jsonl")
    t.view()                       # the summary, in the form the session wants
    t.failed(), t.capped()         # solves that diverged, solves with a block at its cap
    t.abandoned(), t.backtracks()  # rejected steps, and the rewinds with their reasons
    t.patterns()                   # the run collapsed to its distinct step patterns
    t.part("SNES_Stokes#3", at_step=40)   # the equation that was being solved at step 40
    t.compare(12, 13)              # what changed between two steps

Every method returns plain data, the same dicts the record holds, with the
outcome of a solve read by the same rule the figure and the table use.
"""

from .describe import record, view as _view


def _operator_text(event):
    """A short name for what an event did: the solver's own name for a
    solve, ``history shift T`` for a shift, ``advect swarm`` for a push."""
    kind, name = event.get("kind"), event.get("name")
    if kind == "solve":
        return str(name)
    if kind == "adjoint_solve":
        return f"adjoint {name}"
    if kind == "history_shift":
        return f"history shift {name}"
    if kind == "swarm_advect":
        return f"advect {name}"
    return f"{kind} {name}"


class Transcript:
    """One run, from a transcript file, the list :func:`read_transcript`
    returns, or a live model. ``run`` picks a run when the file holds
    several, the last by default."""

    def __init__(self, source, run=-1):
        from .transcript_report import _as_runs
        if isinstance(source, Transcript):
            runs = source.runs
        else:
            runs = _as_runs(source)
        if not runs:
            raise ValueError("this transcript holds no run")
        self.runs = runs
        self.entry = runs[run]
        self.header = self.entry.get("run") or {}
        self.steps = list(self.entry.get("steps") or [])
        self.notes = list(self.entry.get("notes") or [])
        self.parts = list(self.entry.get("parts") or [])
        self.ended = self.entry.get("ended")
        self.live = bool(self.entry.get("live"))

    # --- the shape of the run -------------------------------------------

    def __len__(self):
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    @property
    def name(self):
        from .transcript_report import _run_title
        return _run_title(self.header, fallback="")

    def step(self, index, attempt=-1):
        """The step with this index, as recorded. After a rewind the run
        numbers its steps again from where it went back to, so an index can
        name several attempts: ``attempt`` picks one, the last by default,
        which is the one that stands. :meth:`attempts` lists them all."""
        found = self.attempts(index)
        if not found:
            raise KeyError(f"no step {index} in this run")
        return found[attempt]

    def attempts(self, index):
        """Every recorded step with this index, in the order they happened:
        the rejected ones first, the one that stands last."""
        return [s for s in self.steps if int(s.get("index", -1)) == int(index)]

    def position(self, step):
        """Where a step record sits in the run's sequence."""
        for i, s in enumerate(self.steps):
            if s is step:
                return i
        raise ValueError("this step is not in the run")

    def sequence(self, step):
        """The operators a step applied, in order, as short names."""
        step = self.step(step) if not isinstance(step, dict) else step
        return [_operator_text(e) for e in step.get("events", []) if e.get("kind") != "warning"]

    def outcome(self, event):
        """``"ok"``, ``"capped"``, ``"diverged"`` or ``None`` for an event: the
        rule the figure and the table apply."""
        from .transcript_report import _outcome
        return _outcome(event)

    # --- events ---------------------------------------------------------

    def events(self, kind=None, part=None, outcome=None, step=None):
        """Events across the run, each as ``(step_index, event)``, filtered
        by kind (``"solve"``, ``"adjoint_solve"``, ``"history_shift"``,
        ``"swarm_advect"``, ``"warning"``), by part, by outcome, or to one
        step."""
        out = []
        for s in self.steps:
            if step is not None and int(s.get("index", -1)) != int(step):
                continue
            for e in s.get("events", []):
                if kind is not None and e.get("kind") != kind:
                    continue
                if part is not None and e.get("part") != part and e.get("name") != part:
                    continue
                if outcome is not None and self.outcome(e) != outcome:
                    continue
                out.append((int(s.get("index", -1)), e))
        return out

    def solves(self, outcome=None, part=None):
        return self.events(kind="solve", part=part, outcome=outcome)

    def failed(self):
        """Solves that did not converge."""
        return self.solves(outcome="diverged")

    def capped(self):
        """Solves the SNES called converged while an inner block hit its
        iteration cap or its deadline: the amber mark."""
        return self.solves(outcome="capped")

    def warnings(self):
        return self.events(kind="warning")

    def abandoned(self):
        """Steps that did not commit: an exception inside the block, or a
        step the caller rejected. Each carries ``abandoned_by`` when the
        run recorded what stopped it."""
        return [s for s in self.steps if not s.get("completed")]

    def backtracks(self):
        """The rewinds and restores, each with where in the sequence it
        happened (``after_position``), the step it went back to
        (``to_step``), and the ``reason`` and ``detail`` the caller gave."""
        return [n for n in self.notes if n.get("kind") in ("rewind", "restore")]

    # --- parts -----------------------------------------------------------

    def part_names(self):
        seen = []
        for p in self.parts:
            if p.get("part") not in seen:
                seen.append(p.get("part"))
        return seen

    def part(self, name, at_step=None):
        """The description of a part as recorded: what it solved. With
        ``at_step``, the description in force at that step, which is the
        latest recorded at or before it — a part records itself again when
        its form changes."""
        records = [p for p in self.parts if p.get("part") == name or p.get("label") == name]
        if not records:
            raise KeyError(f"no part {name!r} recorded in this run")
        if at_step is None:
            return records[-1]
        before = [p for p in records if int(p.get("at_step", -1)) <= int(at_step)]
        return before[-1] if before else records[0]

    def changes(self):
        """Parts whose form changed during the run: ``(part, at_step)`` for
        each re-recording after the first."""
        seen, out = {}, []
        for p in self.parts:
            key = p.get("part")
            if key in seen and seen[key] != p.get("fingerprint"):
                out.append((key, p.get("at_step")))
            seen[key] = p.get("fingerprint")
        return out

    # --- structure -------------------------------------------------------

    def patterns(self):
        """The run collapsed to its distinct step patterns: consecutive
        steps that applied the same operators in the same order with the
        same outcomes, the same label and the same completion are one
        pattern. Each is ``{"from", "to", "count", "sequence", "outcomes",
        "label", "completed"}``; a run that never changes has one."""
        out = []
        for position, s in enumerate(self.steps):
            signature = (tuple(self.sequence(s)),
                         tuple(self.outcome(e) for e in s.get("events", []) if e.get("kind") != "warning"),
                         s.get("label"), bool(s.get("completed")))
            index = int(s.get("index", -1))
            if (out and out[-1]["_signature"] == signature
                    and not self._note_between(out[-1]["positions"][1], position)):
                out[-1]["to"] = index
                out[-1]["positions"] = (out[-1]["positions"][0], position)
                out[-1]["count"] += 1
                continue
            out.append({"from": index, "to": index, "positions": (position, position), "count": 1,
                        "sequence": list(signature[0]), "outcomes": list(signature[1]),
                        "label": signature[2], "completed": signature[3], "_signature": signature})
        for p in out:
            p.pop("_signature")
        return out

    def _note_between(self, position_a, position_b):
        """Whether a note (a rewind, a restore) sits between two positions."""
        return any(position_a <= int(n.get("after_position", -1)) < position_b for n in self.notes)

    def compare(self, a, b):
        """What differs between two steps, given by index (the attempt that
        stands) or as records from :meth:`step`: operators in one and not
        the other, outcomes that changed for the same operator, and the
        interval, wall time and completion of each."""
        sa = a if isinstance(a, dict) else self.step(a)
        sb = b if isinstance(b, dict) else self.step(b)
        seq_a, seq_b = self.sequence(sa), self.sequence(sb)
        oa = {self._event_key(e): self.outcome(e) for e in sa.get("events", [])}
        ob = {self._event_key(e): self.outcome(e) for e in sb.get("events", [])}
        return {
            "only_in_a": [x for x in seq_a if x not in seq_b],
            "only_in_b": [x for x in seq_b if x not in seq_a],
            "order_differs": seq_a != seq_b and sorted(seq_a) == sorted(seq_b),
            "outcome_changes": {k: (oa[k], ob[k]) for k in oa if k in ob and oa[k] != ob[k]},
            "dt": (sa.get("dt"), sb.get("dt")),
            "wall": (sa.get("wall"), sb.get("wall")),
            "completed": (bool(sa.get("completed")), bool(sb.get("completed"))),
        }

    @staticmethod
    def _event_key(event):
        return (event.get("kind"), event.get("part") or event.get("name"))

    def between(self, t0, t1):
        """Steps whose interval starts in ``[t0, t1]``, in the run's own
        time unit."""
        from .transcript_report import _magnitude
        return [s for s in self.steps if t0 <= _magnitude(s.get("t0")) <= t1]

    def adjoint_segments(self):
        """The run partitioned by adjoint support: see
        :func:`underworld3.transcript_adjoint_segments`."""
        from .transcript_report import transcript_adjoint_segments
        return transcript_adjoint_segments(self.runs, run=self.runs.index(self.entry))

    # --- description ---------------------------------------------------

    def describe(self, depth=1):
        """The run as data, in the shape every other object uses: facts for
        the header and the counts, and the parts as children."""
        facts = {}
        if self.header.get("started"):
            facts["started"] = self.header["started"]
        if self.ended:
            facts["ended"] = self.ended.get("ended") or self.ended.get("at") or "yes"
        elif self.live:
            facts["state"] = "in progress"
        else:
            facts["state"] = "no terminator: still running, or interrupted"
        scales = self.header.get("reference") or self.header.get("scales") or {}
        if scales:
            facts["scales"] = " | ".join(f"{k} {v['magnitude']:.4g} {v['units']}"
                                         for k, v in scales.items() if isinstance(v, dict))
        facts["steps"] = len(self.steps)
        abandoned = self.abandoned()
        if abandoned:
            facts["abandoned"] = [int(s.get("index", -1)) for s in abandoned]
        backtracks = self.backtracks()
        if backtracks:
            facts["backtracks"] = len(backtracks)
        failed, capped = self.failed(), self.capped()
        if failed:
            facts["failed solves"] = [f"{i}: {e.get('name')}" for i, e in failed]
        if capped:
            facts["capped solves"] = len(capped)
        patterns = self.patterns()
        facts["patterns"] = [f"{p['from']}-{p['to']}: " + " > ".join(p["sequence"]) if p["count"] > 1
                             else f"{p['from']}: " + " > ".join(p["sequence"]) for p in patterns[:12]]
        if len(patterns) > 12:
            facts["patterns"].append(f"... {len(patterns) - 12} more")
        children = []
        if depth > 0:
            for p in self.parts:
                child = dict(p)
                child.setdefault("kind", "part")
                child.setdefault("name", p.get("label") or p.get("part"))
                child.setdefault("summary", f"recorded at step {p.get('at_step')}")
                children.append(child)
        summary = f"{len(self.steps)} step(s)"
        if abandoned:
            summary += f", {len(abandoned)} abandoned"
        if backtracks:
            summary += f", {len(backtracks)} backtrack(s)"
        if failed:
            summary += f", {len(failed)} failed solve(s)"
        return record("transcript", self.name, summary, facts=facts, children=children)

    def view(self, format=None, depth=None):
        """Show the run: :meth:`describe` rendered for the session, or in
        the ``format`` named."""
        _view(self, format=format, depth=depth)
