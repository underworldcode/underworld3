# Figures for UWTN 2026-022, "Faults on an add-on mesh"

Made on hyperion, 2026-10-25, branch `feature/coarse-solve-nullspace`. Library
source: commit `e135429f` (every later commit on the branch touches
`scripts/sessions/` only): `feature/parallel-addon-distribution` at 8fe9ad76,
the empty-rank fixes (aed17e23, bfae3e65), and the coarse-solve prototype,
which is inactive unless `UW_ROTATED_COARSE` is set (it was not). `dev`
environment, PETSc 3.25.3.

Fixture for all three: the unit box in simple shear (`v = (y - 0.5, 0)` on
every wall), eta = 1, a straight fault; background
`UnstructuredSimplexBox(cellSize=0.1, refinement=1)` (h = 0.1, 0.05, never
cut); add-on part `base.adapt(metric, max_levels=k)` toward the trace
(`h_near = 0.05 / 2**k`, band width 0.08, linear profile); `add_fault` splits
the finest and, by default, every add-on level. Rotated fieldsplit solve,
velocity block custom FMG; tolerance 1e-6.

## stack.png (`fig_stack.py`)

k = 3, fault (0.30, 0.45) to (0.70, 0.55).

| level | role | cells | fault |
|---|---|---|---|
| 0 | background | 242 | welded (trace drawn) |
| 1 | background | 968 | welded (trace drawn) |
| 2 | add-on | 1274 | split |
| 3 | add-on | 1644 | split |
| 4 | finest | 2006 | split |

The two copies are drawn 0.01 apart along the fault normal; the tips are
single vertices.

## iterations.png (`fig_iterations.py`)

Re-measured, serial, fault centred at (0.5, 0.5), half-vector (0.20, 0.05).
A and C through the library (`../library_c.py`, records in
`~/+Simulations/fault_split_mg/lib_*_np1.json`); B2 and D exist only in the
exploration driver (`../stage5_variants.py`, `RESULT` lines).

Velocity iterations per solve, frictionless fault:

| k | near-fault h | A welded | B2 TI band | C split | D split + weak zone |
|---|---|---|---|---|---|
| 2 | 0.0125 | 8 | 14 | 5 | 7 |
| 3 | 0.00625 | 13 | 22 | 5 | 4 |
| 4 | 0.0031 | 18 | 30 | 4 | 4 |

Viscous fault (eta_f = 1) at k = 3: C 5, A 12.

Peak slip, A against C (library): k = 2: 0.1595810283 / 0.1595808977;
k = 3: 0.1589859512 / 0.1589860839; k = 4: 0.1592639163 / 0.1592638795;
viscous k = 3: 0.1365087172 / 0.1365088233. Agreement to six figures at every
k. Cells on the finest: 1470, 2006, 3034 at k = 2, 3, 4.

## moving-fault.png (`fig_moving_fault.py`)

k = 4, the fault (half-vector (0.20, 0.05)) centred at x = 0.3, 0.5, 0.7,
frictionless. Each step rebuilds the add-on part from the one background,
cuts and splits every add-on level, and solves. Wall times per step, seconds
(`timings_np8.json`, `timings_np1.json`):

| x | np | refine | conform + split | solve | velocity its |
|---|---|---|---|---|---|
| 0.3 | 8 | 0.58 | 0.89 | 4.83 | 7 |
| 0.5 | 8 | 0.26 | 0.78 | 4.63 | 7 |
| 0.7 | 8 | 0.29 | 0.75 | 4.54 | 7 |
| 0.3 | 1 | 1.42 | 1.22 | 5.98 | 4 |
| 0.5 | 1 | 0.77 | 1.08 | 6.21 | 4 |
| 0.7 | 1 | 0.83 | 1.11 | 6.48 | 5 |

The np 1 rows are the run that drew the figure (the first serial run gave
1.40 / 1.22 / 6.38, 1.26 / 1.09 / 6.32, 0.86 / 1.11 / 6.80). "solve" includes
building the solver and its transfers. Finest cells (serial): 3102, 3034, 3080;
six levels. Peak slip: 0.153319, 0.159264, 0.153278. The machine was otherwise
idle.

Background cells outside the band: of the cells farther than 0.15 from the
trace, 760 of 840, 747 of 833 and 762 of 832 coincide with cells of the
background's finest level (about 90%); at 0.08 (the band width) 836 of 945.
Refinement grading and conformity reach somewhat past the band, so "the base
cells are identical outside the band" is true of most, not all, of them.

## The draft's three claims, checked at this commit

(a) "Iterations 6/5/4 split vs 13/18/23 welded at k = 2/3/4, slip equal to
six figures." **Stale.** Now 5/5/4 split against 8/13/18 welded; the slip does
agree to six figures. The viscous k = 3 pair is 5 against 12 (the draft's
17 is stale too). The welded tail improved since 2026-10-13; the split tail
still wins at every depth and the gap still grows with k.

(b) "The junction ligament closes at the coarser level and the replay stops
there." **Half right.** `probe_junction_replay.py` (test_0866's kissing Y,
ligament 1.0): the replay does stop at the first add-on level below the
finest and leaves it and everything beneath welded, at one and at two add-on
levels. The reasons it gives are not the ligament closing: at one level,
"the label meets itself at a junction. Junctions need a side assignment this
version does not define"; at two levels, "snapping inverted 1 cell(s);
snap_frac=0.1 is too large for this mesh".

(c) "The 3-D sheet placement on a two-level child is refused by the volume
guard." **Stale.** `../library_c_3d.py C 2` places the sheet at
`clearance=0.9` and solves: 44382 tetrahedra, 7 velocity iterations, peak
slip 0.17967. No intermediate add-on level survives in the tail
(`split_levels` 0, three levels: two background and the finest), so at k = 2
the 3-D hierarchy still has no split coarse level.

## NOTES

- 2026-10-25 (hyperion): figures and numbers above at library `e135429f`; the
  note's tables should take these, and claims (a), (b), (c) need the
  corrections above.
