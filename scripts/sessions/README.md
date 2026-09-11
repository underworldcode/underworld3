# Session scripts

Drivers, plotting scripts, benchmarks and profilers written during development
sessions. They are not tests and are not run by CI — they lived in `tests/`
until the 2026-09 audit, where 22 of them sat among the real test files and made
the test tree hard to read.

Nothing imports these; they are run directly. Three of them
(`vep_fault_weakening.py`, `vep_strain_weakening.py`, `vep_timedep_yield.py`)
wrote their figures to an absolute path inside a worktree that no longer exists,
so they had been writing nowhere since that worktree was removed; they now write
beside themselves.

A test helper belongs in `tests/` with a leading underscore (`_mg_ladder.py`),
not here.
