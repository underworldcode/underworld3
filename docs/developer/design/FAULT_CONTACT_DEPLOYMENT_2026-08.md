# Split-node fault contacts: deployment design

**Status**: agreed direction (maintainer + AI session, 2026-08-04). The
backend primitives are prototyped and validated on `feature/fault-split-node`;
this note records the architecture and the path from prototype to deployed
capability. Companion study with all reference numbers:
`~/+Simulations/fault_split_gate/`.

## What exists and what it measured

A fault as a **zero-thickness contact**: the mesh's nodes are duplicated along
a conforming facet chain (`utilities/fault_split.py`), so continuous FE spaces
jump across the fault and nowhere else; interface conditions act on the
coincident DOF pairs through an orthogonal mean/jump extension of the rotated
strong-BC machinery (`utilities/fault_contact.py`, pair blocks in
`rotated_bc.build_rotation`).

Validated so far, all at the fault-study resolution (13k cells):

| what | result |
|---|---|
| Schur conditioning | 10 iterations vs the thin weak inclusion's 147, resolution-independent |
| no-opening constraint | machine zero (5e-18) in every configuration tested |
| frictionless (stress-driven) slip | elliptical to 1.35 % RMS; peak 92 % of the infinite-medium crack (finite box) |
| viscous interface law τ = η_f·V | welded↔free family follows the crack compliance 1/(1 + 0.91 η_f a/η) |
| welded limit | removes the fault (recovers the uncut continuum) — it penalises the JUMP only; the mean velocity is never constrained, so nothing becomes rigid |

The Coulomb lobe geometry matches the inclusion representation across all
three fault laws tried (prescribed slip, frictionless, viscous), with
amplitudes ordered exactly by their slip.

## The three-layer architecture (settled)

1. **The fault manifold** — persistent, owns the physics identity. A genuine
   lower-dimensional mesh (polyline in 2-D; the placed-surface branch's
   pyvista triangulation in 3-D), **replicated on every rank**. Its NODES are
   the Lagrangian markers of the fault: per-node internal state (rate-state
   θ, slip history) and property fields live here — arc length is NOT a
   material coordinate under fault deformation, nodes are. A network is a
   set of manifold components plus a junction table.
2. **The split bulk mesh** — ephemeral, rebuilt whenever the fault moves
   (the non-cumulative `adapt` pattern: re-cut and re-split from the static
   base). It carries only the **trace mapping**: the Minus→Plus point pairs
   (`mesh._fault_point_pairs`, from the split's clone map — the sides are
   coincident, so no coordinate query can ever recover the pairing), and
   what derives from them (slip rows, normals, s-coordinates for sampling
   against the manifold).
3. **The constraint/constitutive layer** — the pair transform (no opening
   strong; slip row free = frictionless, datum = prescribed jump-only slip,
   linear operator = viscous, nonlinear relation = friction). Interface
   operators assemble as slip-row trace-mass blocks; tips need no interface
   BC because the jump space vanishes where the sides share a point.

## Two entry paths, one backend

- **(a) Submesh-surface path**: a persistent fault object supplies the
  manifold and its data; the bulk mesh conforms to it and splits along it.
  This is the natural interface for models where the fault is a first-class
  design object (earthquake studies, faulted margins).
- **(b) Ephemeral adapt-on-top path**: per step or per re-adaptation, the
  manifold drives the refinement metric, the cut, and the split on a child of
  the static base; fields and manifold state transfer across rebuilds
  (bulk fields via the existing re-adaptation machinery, manifold state by
  identity — it never lived on the bulk mesh).

Both funnel through the same backend contract:

    manifold → conforming labelled chain on the bulk mesh
             → split (pairs recorded)
             → trace mapping (pairs ↔ manifold nodes)
             → interface laws on the pairs

API sketch (interface-first discipline: land the signatures on `development`
as stubs before the feature branch fills them):

```python
fault = uw.meshing.FaultSurface(name, geometry, properties=..., )   # layer 1
child = mesh.add_fault(fault, ...)          # adapt→cut→split, or split-only
stokes.add_fault_bc(law, fault.name)        # "free" | ("viscous", eta_f) | friction later
```

The manifold contract is a PROTOCOL, not one class (placed-surface
session's counter-proposal, 2026-08-05, adopted): a fault manifold is
anything with a ``name``, geometry, signed-distance/director sampling,
and per-node ``point_data``. The concrete types are ``uw.meshing.Surface``
(the 2-D polyline — what ``add_fault`` consumes today) and
``uw.meshing.FaultSurface`` (the 3-D pyvista-triangulated sheet, which
predates both branches and has no 2-D form). Documentation must not name
``FaultSurface`` as "the" manifold object — the landed 2-D capability
would then have none.

## Parallel: the seam-vertex crossing rule (maintainer ruling, 2026-08-04)

Never synchronise interior replicas across ranks. Instead **pin a mesh vertex
exactly at every fault–seam crossing** (`pull_vertex_onto` is already
collective and partition-consistent), and require crossings to be
transversal. Then:

- every fault facet, and every interior replica, is rank-local — the current
  machinery applies unchanged;
- the ONLY shared object is the crossing vertex, which needs one shared
  replica pair: a single new star-forest entry per crossing, associated
  deterministically by the key **(original root point, side)** — both ranks
  compute the side identically because both hold the replicated manifold and
  the chain-direction rule (lexicographically smaller tip first) is global;
- replicas inherit the original's owner, so the pair blocks in Q stay inside
  one rank's diagonal portion and the existing "pair straddles ranks" guard
  remains an invariant, not a limitation.

Still refused, deliberately: a fault running *along* a seam (facets on the
seam), and junctions on a seam (first version). The current blanket seam
refusal stays in place until this lands, and remains the fallback error.

## Representation policy: the fault chooses its implementation per equation

The fault object is the PHYSICAL entity; how an equation system feels it is
a per-solver choice, and different equations on one model may choose
differently — a sharp interface for Stokes, a damage zone for Darcy flow.
The fault object supplies the ingredients; each solver picks its adapter:

| representation | mechanism | status |
|---|---|---|
| **surface** (sharp contact) | split mesh + pair constraints, `add_fault_bc` | built (this branch) |
| **volume** (weak zone / damage) | distance field from the same manifold → viscosity, damage, permeability k(d) | exists (`meshing/faults.py`: `compute_distance_field`, `create_weakness_function`) |
| **TI weak zone** | director transfer from the manifold normals → `TransverseIsotropicFlowModel` | exists (`meshing/faults.py`: `transfer_normals`) |

Composition across equations, two patterns:

1. **Different meshes per continuity class (the default).** Stokes solves on
   the split child; Darcy / thermal solve on the PARENT (continuous) with a
   volume representation from the same manifold's distance field. No new
   machinery — multi-mesh field transfer exists, and each equation gets a
   mesh whose continuity matches its physics.
2. **One shared split mesh (tight coupling only).** A solver that must NOT
   see the slit needs a continuity (weld) constraint: jump = 0 on its field
   across every pair. For Stokes this is the measured welded limit; for a
   scalar system it is a small new piece (the pair transform on a scalar
   field). Build it when a tightly coupled problem actually demands one
   mesh; until then pattern 1 is simpler and exact.

## Parallel robustness (measured, np = 3/4/5)

`~/+Simulations/fault_split_gate/crossing_sweep.py`: nine fault geometries
per rank count. 17/27 split cleanly (every success: zero star-forest drift,
global Euler 0, conformity; one frictionless solve per np with machine-zero
leak); every refusal was COLLECTIVE and categorised — three-rank corners on
the line, and faults running along a seam. No hangs, no invalid meshes, no
unclassified failures. The sweep also established the kink rule: EVERY
polyline control point needs a vertex pulled onto it, interior kinks
included — a kink is the same problem as a tip.

## Crossing and branching faults: the three-level strategy

Faults branch even where they cannot cross, and crossings evolve (an
inactive crossing deforms into two abutments and reactivates). Three levels,
each usable on its own; the approximate ones are legitimate models, not
placeholders.

- **J0 — offset junctions (available now).** A branch or crossing is a set
  of DISJOINT segments separated by a ligament of one or two local h. Each
  splits independently under the existing refusals; the intact ligament
  transmits stress between them. Junction process zones are physical, the
  error is ligament-scale and shrinks with the near-fault refinement.
  Validation: the King two-fault interaction pattern. A crossing is two
  offset abutments.
- **J1 — true abutment, tip-on-fault (a refusal refinement).** Split the
  through-going fault FIRST; the abutter's tip vertex then belongs to one
  side's copy (the sector it approaches from), and the abutter splits as an
  ordinary fault whose UNSPLIT tip sits on the master's slit. Tips are never
  fan-walked, so the only change is the boundary-touching refusal learning
  to distinguish, for tips only, a domain boundary from a prior fault's slit
  (the labels tell them apart). Exact T kinematics: the master slips through
  freely; branch slip tapers to zero over its last facet. This is the
  locked-branch case — sufficient wherever the branch does not actively
  partition slip.
- **J2 — the exact d-sector split (scheduled).** A degree-d junction vertex
  receives d sector copies (the fan walk opened at ALL incident fault
  edges); branch compatibility — trunk slip partitioning onto splays — is a
  TELESCOPING IDENTITY of sector differences around the vertex, so no cycle
  constraint and no multiplier exists; per-branch no-opening plus per-branch
  interface laws close the system, and a crossing is simply d = 4. New
  pieces, all contained: the generalised fan walk, d−1 replicas per junction
  vertex, per-branch pairing in the trace mapping, and a d-sector BC block
  at junction nodes in place of the pair block. Acceptance test: the vector
  sum of branch slips at the junction closes to machine precision — an
  identity of the representation, not a physics tolerance. The ordinary
  chain vertex is d = 2 and the tip is d = 1 under the same rule.

Junctions stay OFF partition seams at every level, consistent with the
seam-vertex crossing rule.

## Staging

1. **API layer + common backend, serial** — `FaultSurface`, `add_fault`,
   `add_fault_bc`; docs; promote the module functions behind the methods.
   Includes J0 (offset junctions) as a documented pattern with a two-fault
   interaction test, and J1 (tip-on-fault abutment) as the refusal
   refinement.
2. **Parallel crossing rule** — the one-entry-per-crossing SF extension,
   np=2–4 tests including a fault deliberately spanning two ranks.
3. **Nonlinear interface laws** — regularised Coulomb with σ_n from the
   normal-row reaction (Picard-lagged), tangent 2·(dτ/dV)·M rebuilt per
   Newton iteration in the existing hook; oracle: at fixed τ the solution
   must land on the measured compliance curve at the secant η_f = τ/V.
   Then rate-state (arcsinh form; θ ODE on manifold nodes between steps).
4. **Manifold state plumbing** — trace↔manifold sampling, the gather/reduce
   collective for V and σ_n feeding the θ ODE, checkpointing via the
   manifold object.

Geometric multigrid is a separate track (split only the finest 1–2 levels;
side-aware transfer at the single split/unsplit seam) and is not a
prerequisite for any of the above — GAMG holds comfortably at current scale.

## Traps already paid for (do not rediscover)

- petsc4py `getLabel` on a missing name and `getStratumIS` on an empty
  stratum return NULL-wrapping objects that SEGFAULT on first use — guard
  with `hasLabel` / `getStratumSize`.
- The rotated path's field copy-back dropped inhomogeneous essential values
  (#497, fixed): essential DOFs are absent from the global vector. The tell
  is the divergence theorem failing on a field diagnostic.
- Never inject interface entries into the `ptap`-refreshed rotated operator —
  add them to a copy; re-attach the null space on each rebuilt copy.
- The pairing comes from the clone map only; coordinate matching is
  impossible-by-construction at zero distance. Protect the property that the
  two trace spaces are identical with exactly paired DOFs — losing it means
  genuine mortar machinery.
