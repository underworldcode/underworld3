# Adversarial review: refute the change before it opens as a PR

**Audience**: anyone (human or AI) preparing an Underworld3 branch for review.

The [code review process](CODE-REVIEW-PROCESS.md) describes how a change is
reviewed once it is a PR. This document is the pass that runs *before* that: an
independent attempt to refute the change, aimed at the failure modes this
codebase actually produces. Four parallel reviewers on three already-pushed
branches found twenty-odd defects in 2026-09, including a SEGV reachable from
the API's own error message and a JIT change that aborted half of all `np>=2`
runs. Every one of them was on work already pushed, which is why the ordering
matters.

## Running it

1. Read `docs/developer/UW3_STYLE_CHARTER.md`. It is two pages and it is
   normative; the style findings that reach a PR are almost always in it.
2. Run the review on the branch diff — an independent pass whose goal is to
   break the change, not to confirm it.
3. Fix what it finds, then open the PR.
4. Adjudicate every open issue in the territory the branch touches: close it
   with evidence, or refuse to close it and say what is still live. See
   [Issues the review closes](#issues-the-review-closes).
5. Post the findings and the adjudications on the PR, including the attacks
   that failed.
6. Run it **again** after any substantial post-review commit. A refactor
   landing after the review is when a second pass is worth most.

The posted review is terse: findings and evidence (numbers, `file:line`, probe
name), one line each; a short list of attacks that failed, with numbers; merge
conditions as bullets. Voice is "we". Cite a rule at a defect
("Charter §4: uncommented swallow"), never to justify the review itself.

## What we attack

**Parallel.** Rank asymmetry, collective ordering, empty-rank edges. A
collective call inside a rank-dependent branch hangs the job rather than
failing it, so it survives every serial test.

**Frames and units.** Does the value cross a boundary in the units the other
side expects? Snapshot capture read `mesh.X.coords` (metres) while restore
wrote model units, and silently rescaled the mesh on every restore.

**Determinism.** Anything whose result can depend on hash seed, dict order, or
partition. A JIT change made the generated C hash-seed dependent and aborted
roughly half of multi-rank runs.

**Tests that cannot fail.** Every regression test must be shown to fail on its
bug. A ramp of 1000 → 1e-6 where `"1.0e-6"` sorts as a prefix of `"1.0"` moves
nothing; an MPI property that is seed-flaky caught its bug zero times in six.
Delete a test that cannot fail. A flaky test is the test's fault until proven
otherwise.

**Diagnostics that do not discriminate.** Before building on a measure, check
it gives a different answer with the feature on and off. Three field-level
measures of population control did not.

**CI reach.** Does the new test file match the CI glob? Eleven test files —
an entire subsystem's suite — never ran because they fell outside the pattern.

## Contracts a change must not quietly leave

These are enforced by tests. The review checks the test is still the one doing
the enforcing, and that no exemption list has grown a new entry without a
reason beside it.

**A solver says what it solves.** Every solver class declares `_solver_terms`,
a tuple of `(attribute, description)` naming the terms it is given, from which
`_declared_terms()` is built. `describe()` then returns the residual templates,
the named expressions inside them (followed recursively into the constitutive
model), the boundary conditions and those terms. `view()` renders that
description and the run transcript serialises it, so a note and a run cannot
quote different equations. A new solver that skips the roster fails
`tests/test_0016_solver_description_contract.py`; adding its name to that
file's exemption list is not the fix.

**Constraints have one enumeration.** `_constraint_mechanisms()` lists every
way a constraint can be on a solver, and both the mixed-mechanism guard
(#464) and `describe()` read it. A mechanism added without registering there
breaks the guard — loudly — rather than vanishing from every description.

**A run records what it did.** `model.step(dt)` is the transaction that carries
the transcript. A new operation that changes model state inside a step — a
solve, a history shift, a mesh deformation — records itself, or the transcript
silently understates the run. See
[HOW-TO-WRITE-UW3-SCRIPTS](HOW-TO-WRITE-UW3-SCRIPTS.md) and
`docs/developer/design/run-plan-and-transcript.md`.

**Named quantities keep their names.** A coefficient written as
`uw.expression(r"\rho_0 \alpha g", ...)` appears in the description under that
name. An anonymous float collapses into the assembled product and the
transcript can only show the number. Examples in `docs/` name their
coefficients.

**A family's guide says what the family can do now.** The curated guides in
`docs/developer/guides/` carry front matter naming the families they apply
to, and `uw.capabilities()` lists them beside each family. A change to a
solver family, a constitutive model, a history scheme or a boundary
mechanism is reviewed against every guide that names it: the guide is
updated in the same change, or the review says why it still holds. The AI
skills in `.claude/skills` are symlinks to those pages;
`tests/test_0030_capability_guides.py` fails on a copy, on a guide without
front matter, and on a family name no class carries.

## Issues the review closes

The review decides issue state, and it is the only pass that is in a position
to. It has the branch diff, it has just read the code the issue is about, and
it is already looking for the ways a fix fails to be a fix.

**Before opening the PR, list the open issues in the territory the branch
touches** — the files in the diff, the subsystem, the symptom. For each one,
the review returns exactly one of three verdicts, and every verdict is posted
on the PR with its evidence:

| verdict | what it means | what it requires |
|---|---|---|
| **Closed by this branch** | the defect is gone because of this change | the PR body carries `Closes #N`, and the branch carries the regression test that would have caught it |
| **Already fixed** | it was fixed earlier and nobody closed it | a probe run against `development`, pasted into the issue as the closing comment, showing the behaviour the issue describes no longer happens |
| **Still live** | the branch passes through it and leaves it | one line saying which part still reproduces, so the next session does not re-derive it |

**Refusing to close is a verdict, not a gap.** A branch that touches the
territory of an issue and does not fix it says so, in a line. The failure this
replaces is silence: the fix lands, the issue reads as familiar to anyone who
knows the PR work, and nothing ever goes back to adjudicate it. Measured over
2026-09: 61 issues filed, 19 closed, and a random sample of five found two
already fixed on `development` and still open.

**Close on a probe, never on a claim.** "PR #NNN fixed this" is not evidence —
a PR that merged in the territory of an issue very often addressed its
neighbour. The probe goes in the issue so any close can be audited and
reopened.

**What becomes an issue in the first place.** A finding the branch fixes is a
commit and a line in the posted review. It is not also an issue. Only a finding
we decide **not** to fix in this branch is filed, and then the PR body says
which issues it is leaving open and why. One defect produced an issue and a PR
comment through 2026-09, the comment was addressed on merge, and the issue
stayed — which is most of the backlog.

## Where the reviews live

`docs/reviews/[YYYY-MM]/`, indexed by `docs/reviews/README.md`, and posted on
the PR itself.
