---
title: Raw notes for "Our AI Development Strategy"
status: notes
feeds_into: [standalone]
target: underworldcode.org (Ghost)
---

# Framework: Our AI Development Strategy

## The Hook

Small team, complex codebase, ambitious scope. Underworld3 has ~50k lines
of Python/Cython wrapping PETSc, SymPy, and a JIT compiler. The core
development team is tiny by any software project standard. AI assistance
isn't a luxury — it's how we stay viable.

## Thesis (Revised)

This is a **co-evolution** story, not an adoption story. AI tools didn't
just accelerate our existing workflow — they couldn't, because the
codebase wasn't ready for them. The breakthrough was recognising that
making the code work better for AI also made it work better for humans.
The code and the process evolved together.

---

## Section Sketch (Revised)

### 1. The Starting Point: It Didn't Work

AI tools (Claude Sonnet) entered the workflow in **August 2025**. The
early commits tell the story of what we tackled first:

**Aug–Sep 2025: Data structures and evaluation rewrites.**
The first AI-assisted work was on the fundamentals — swarm and mesh
variable data structures, the global evaluation routine, kdtree-based
particle migration, NDArray_with_callback. Also: "re-working structure
to be more AI friendly", "First pass at AI training examples and
documentation." The codebase was being reshaped for AI comprehension
from the very first month.

**Oct–Nov 2025: Units system and API consistency.**
Complete data access migration (`mesh.access()` → direct access),
universal units system, coordinate interface cleanup, Parameter
descriptor migration across constitutive models, test migration to
pytest markers. This was the intensive API-consistency phase — making
every module follow the same patterns.

**Dec 2025 – Jan 2026: Infrastructure maturity.**
Pixi build system, PETSc 3.24 compatibility, symbol disambiguation,
documentation reorganisation, CI/CD improvements, Binder integration.
The codebase was becoming stable enough for external users.

**Feb–Mar 2026: Application-driven development.**
DDt hierarchy refactor, PetscDS constants mechanism, boundary integral
work, surface/quantity support, Navier-Stokes improvements. Working
from application needs (benchmarks, tutorials) back to infrastructure
changes. PR-based workflow with proper branching.

AI tools initially struggled to make sense of UW3's design — both the
internal API and the Python user interface. The codebase had:
- Inconsistent patterns across modules
- Implicit conventions that a human could absorb over time but an AI
  would miss or hallucinate alternatives for
- High context requirements: you needed to understand multiple subsystems
  simultaneously to make changes safely

The AI would generate plausible-looking code that was subtly wrong because
it couldn't predict which pattern applied where. This wasn't an AI
problem — it was a code clarity problem that the AI made visible.

### 2. The Breakthrough: Rewrite for AI Readability

The turning point was reframing the goal: **make this code work better
in an AI development environment.** This turned out to mean making it
work better, period.

**Removing API inconsistencies.** Where one module used one pattern and
another used a different pattern for the same operation, the AI would
guess wrong. Fixing this for the AI fixed it for every new contributor.

**Iterative API evolution with AI tools.** We used AI tools as a
litmus test: if the AI consistently misused an interface, the interface
was probably confusing. We evolved the API iteratively, using the AI's
mistakes as signal. Predictability and consistency across modules became
explicit design goals.

**More Pythonic UI.** We looked for more Python-like approaches to the
user interface — standard patterns, familiar idioms — so that both AI
tools and humans had lower context requirements when developing or
writing notebooks. If a Python programmer's first instinct would be
`var.data[...] = values`, then that should work, not
`with mesh.access(var): var.data[...] = values`.

**Developer policies consistent with AI tool use.** CLAUDE.md,
branching conventions, commit attribution, planning-before-execution —
all designed so that an AI session has enough context to be useful and
enough guardrails to be safe.

**Test-driven development.** We moved to a test-driven approach for
issues and new features. This was partly for the usual TDD reasons, but
also because tests are the clearest possible specification for an AI:
"make this pass" is unambiguous in a way that "fix the boundary
conditions" is not.

**Application-driven development.** Recently — leveraging all the above
improvements — we were able to move to an application-driven approach.
Instead of working bottom-up on infrastructure, we specify a target
application (a benchmark, a tutorial, a scientific problem) and let the
AI-assisted workflow identify what needs to change to make it work. This
is only possible because the API is now consistent enough and the test
suite robust enough to catch regressions.

### 3. CLAUDE.md as Institutional Memory

- The idea: a machine-readable project brief that persists across sessions
- What's in it: architecture, conventions, build constraints, design decisions
- How it evolved: started small, grew as we discovered what the AI needed
  to know to be useful
- The meta-problem: the AI that maintains the codebase also needs to
  understand the AI-assistance conventions
- Living document: it changes as the code changes, which is the point
- Memory system: session-persistent memory files capture feedback, user
  preferences, project state across conversations

### 4. What AI Is Good At (For Us)

**Tracing symbolic pipelines** — the early win. The lazy evaluation chain
from user expression → UWexpression → unwrapping → C code generation →
PETSc callback is long and the logic is spread across multiple files.
An AI that can hold the whole chain in context and trace a specific
expression through it catches bugs and identifies simplification
opportunities that are hard to see manually.

**Refactoring with confidence** — changing data access patterns across
50+ files, updating API conventions, renaming consistently. The AI
reads the whole codebase, understands the pattern, and applies it.
A human doing this gets bored and makes mistakes on file 37.

**Test generation and classification** — writing tests for edge cases,
classifying existing tests into reliability tiers, identifying which
tests are actually testing what they claim to test.

**Documentation that stays current** — generating docs from code rather
than writing docs about code. The AI reads the implementation and
produces documentation that matches what the code actually does, not
what it did six months ago.

**Cross-referencing** — "does this change break anything?" requires
reading widely. The AI can check all consumers of a changed interface
in seconds.

### 5. What Did Not Work Well

**Loss of focus without clear targets.** Quite a few times we lost focus
when we did not properly specify targets and did not isolate features
well. The AI is very willing to keep going — it doesn't get tired or
question whether the current direction is right. Without a clear
stopping condition, sessions would drift.

**Large refactors while maintaining functionality.** Redesigning large
pieces of UW3's structure while keeping existing functionality working
was particularly challenging with AI assistance. The AI would make
changes that were locally correct but broke assumptions elsewhere. Clear
tests and constraints were required to resolve this — you need the
test suite to be the AI's conscience.

**Recently: large context helps.** The availability of larger AI context
windows has made major refactors significantly easier to plan and
execute. When the AI can hold the entire affected subsystem in context
simultaneously, it makes fewer "locally correct, globally wrong" mistakes.

**Other ongoing challenges:**

*Numerical algorithm design* — the AI can implement an algorithm you
describe, but it won't invent a better preconditioner or spot that your
time-stepping scheme is only first-order accurate when you think it's
second-order. Domain expertise matters.

*PETSc subtleties* — PETSc's API is vast and the AI's training data
includes many versions. It will confidently suggest API calls that don't
exist in our version, or miss the distinction between a collective and a
local operation.

*Architectural judgement* — the AI will happily refactor code into a
beautiful abstraction that nobody needs. "Should we do this?" is a human
question. The AI is good at "how should we do this?" once the decision
is made.

### 6. The Workflow

**Planning mode as review gate** — for any non-trivial change, we start
in planning mode. The AI proposes a strategy, we review and adjust, then
execution proceeds with minimal interruption.

**Worktrees for isolation** — multiple AI sessions can work on the same
repo without conflicts. Each gets a git worktree with its own branch.

**Attribution** — commits and PRs include "Underworld development team
with AI support from Claude Code". Honest attribution, not co-authorship.

**Test-first specification** — write the test that defines success before
asking the AI to implement the change. The test is the spec.

### 7. The Deeper Question: AI and Scientific Software

- **Verification through visibility.** Scientific software has a
  verification problem: how do you know the code implements the
  mathematics correctly? UW3's symbolic introspection helps — you can
  *see* what was assembled. AI tracing helps — it can follow the chain
  and flag discrepancies. The combination is powerful.

- **Making the code AI-readable made it human-readable.** This is the
  central insight. Every improvement we made for AI comprehension —
  consistent patterns, clear naming, explicit conventions, good tests —
  was also an improvement for human comprehension. The AI was an
  unusually honest user: it couldn't fill in gaps with institutional
  knowledge, so it exposed every place where the code was unclear.

- **Risk: over-reliance.** If the AI writes code that the team doesn't
  fully understand, that's technical debt with extra steps. We try to
  use the AI as a force multiplier for understanding, not a substitute
  for it.

- **Reproducibility.** The AI leaves a trail: commits, PR descriptions,
  CLAUDE.md updates, plan files. The reasoning is captured, not just
  the code.

### 8. Concrete Examples

- **Lazy derivative tracing**: early win — AI could follow the symbolic
  chain across files and identify where derivatives were being dropped
- **Data access migration**: `with mesh.access()` → direct `.data[...]`
  across the entire codebase, with the AI as consistency enforcer
- **Test tier classification**: systematic analysis of test reliability,
  categorising into tiers A/B/C
- **PetscDS constants mechanism**: AI helped design and implement the
  routing of UWexpressions to C-level constants
- **Documentation generation**: subsystem docs, design docs, style guides
  written by AI from code reading, reviewed and refined by humans
- **Bug hunting**: the BDF/AM coefficient freeze bug (NS solver F0/F1
  frozen at BDF1/AM1 despite effective_order ramp) was found through
  AI-assisted code tracing
- **API consistency audit**: AI identified where modules diverged from
  agreed patterns, enabling systematic cleanup

### 9. What We'd Tell Other Small Teams

- **Your code probably isn't AI-ready.** The AI will show you where.
  Treat its confusion as signal, not noise.
- **Start with the project brief.** CLAUDE.md (or equivalent). The
  upfront investment in documenting your architecture for the AI pays
  off immediately — and you'll discover gaps in your own understanding.
- **Tests are specifications.** Move to test-driven development not
  just for quality, but because tests are the clearest instructions
  you can give an AI.
- **Plan before executing.** The AI is very willing to charge ahead.
  Planning mode (review the strategy before starting work) prevents
  drift and wasted effort.
- **Use AI for the work that's important but doesn't get done:** tests,
  docs, consistency checks, pattern migration.
- **Keep humans in the loop** for design decisions and numerical
  correctness.
- **Attribution matters.** Be honest about what the AI contributed.
- **Large context is transformative.** If you're doing complex refactors,
  the ability for the AI to hold the whole affected subsystem in context
  changes what's feasible.

---

## Tone Notes
- Honest, not promotional. We use AI because we have to, not because
  it's trendy.
- **Co-evolution**, not adoption. The code changed to suit the tools,
  and the tools became more useful as the code improved.
- Specific examples, not generalities. Show the actual wins and the
  actual failures.
- Aimed at other scientific software teams, not AI enthusiasts.
- No hype. "Force multiplier for a small team" is the message, not
  "AI writes our code."

## Timeline (from Louis + git log)

**August 2025**: First use of Claude (Sonnet). Started with data
structures, swarm/mesh variable access patterns.

**September 2025**: Early wins — removing `with mesh.access()`,
NDArray_with_callback, kdtree particle migration rewrite, "re-working
structure to be more AI friendly", "First pass at AI training examples."
This is where confidence built: learning to monitor progress, adjust
prompts, see results.

**October–November 2025**: The units/non-dimensionalisation slog.
Pervasive changes, not well enough planned, context overflow multiple
times. "Many moments of frustration. Many." In hindsight: needed
multiple passes with clear boundaries, not one monolithic push. This
was the painful learning about scoping AI-assisted refactors.
Eventually resolved — but expensively.

**December 2025 – January 2026**: Infrastructure maturity. Pixi build
system, PETSc 3.24 compat, docs reorg, CI/CD. More controlled work.

**February–March 2026**: "Massive productivity jumps." Application-driven
development. DDt hierarchy, PetscDS constants, boundary integrals,
NS solver improvements. Working from science targets back to code.
PR-based workflow, proper branching.

## Usage Data (Claude Code report, Feb 2 – Mar 13 2026)

34 sessions, 663 messages, +11,597/-1,055 lines, 179 files, 21 days.
82% goal achievement. 19 parallel-session overlap events (21% of msgs).

**What helped most**: Multi-file changes (12 sessions), good debugging
(8), correct code edits (6).

**Friction**: Wrong initial approach (21 events), buggy code (14),
misunderstood request (9), ignoring conventions (repeated), CI/build
fragility (repeated).

**Louis's reflection on the arc**:
- Phase 1 (data access) = confidence building, learning the workflow
- Phase 2 (units) = overreach, context overflow, frustration — taught
  us about scoping and planning
- Phase 3 (infrastructure) = steady, controlled work
- Phase 4 (applications) = massive productivity, the payoff

**Key insight from Louis**: "If I did the units work now, I would have
multiple passes of changes." The lesson was about decomposition and
planning, not about AI capability. The tools got somewhat better (larger
context), but mostly *we* got better at using them.

## Test Tiers and AI Trust Boundaries

The test classification system is itself a statement about AI trust:

- **Tier A** (human-written): Target tests that define correct behaviour.
  The AI is *allowed to change code to make these pass*. These are the
  ground truth — if Tier A fails, the code is wrong.
- **Tier B** (AI-written, validated): Tests written by Claude, reviewed
  and validated by humans. Trusted for regression, but the AI should
  not change production code solely to satisfy a Tier B test without
  human review.
- **Tier C** (AI prototype): Experimental tests that should not be
  trusted enough to drive code changes. Useful for exploration,
  not for TDD.

This is a practical trust hierarchy: the AI can write tests (B, C) and
the AI can fix code (against A), but the AI cannot write tests that
authorise itself to change code. The human stays in the loop at the
boundary between "what is correct" and "what needs fixing."

## Team Adoption

Gradually bringing in a larger team. Mixed AI tooling:
- Some team members use Claude Code
- One uses Codex
- All pay attention to GitHub Copilot reviews on PRs

**The most useful innovation for team scaling**: switching to the
worktrees + feature branches mode of isolated development. Each
developer (human or AI-assisted) works in their own worktree on their
own branch. No stepping on each other. PRs are the integration point.
This was essential — multiple AI sessions sharing one working directory
was a recipe for overwritten work.

## External Planning System

A separate Claude-based planning system provides high-level oversight
across the project. This is *not* the same Claude that writes code:

- External planning file (`underworld.md` in Box cloud storage)
- Tracks active work, bugs, priorities across the whole project
- Code-writing Claude sessions check in with the planning file at
  conversation start and annotate completed items
- Prevents the tunnel-vision problem: individual sessions optimise
  locally, the planning system maintains the global view

This separation matters: the planning Claude sees across sessions and
across time. The coding Claude sees deeply into one problem. Neither
alone is sufficient.

## Remaining Questions
- Any other specific incidents worth narrating?
- Screenshots or excerpts to include? (e.g., a symbolic trace,
  a before/after of an API pattern)
