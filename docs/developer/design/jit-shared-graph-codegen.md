# Generating JIT kernels from the shared expression graph

**Status**: Proposed, 2026-10-07. Staging steps 2 and 3 implemented behind the private
switch `UW_JIT_GRAPH=1` on `feature/jit-graph-codegen`, 2026-10-08, and measured against
tier 1 end to end ({ref}`jit-graph-in-the-library`).
Tier 2 of [#823](https://github.com/underworldcode/underworld3/issues/823).
Tier 1 ([#830](https://github.com/underworldcode/underworld3/pull/830), merged
2026-10-07) is the fix for today: the memoised
unwrap, realness for field values, coordinates and constant slots, derivatives with
respect to fields through `uw.function.diff_wrt_field`, and the power-mean sharpness as
an atom of its own. Tier 2 is the design we would have chosen from the start, and tier
1 is the competitor it is measured against: no slower, significantly more robust and
easier to maintain.

## Summary

A constitutive law in Underworld3 is a graph: named sub-expressions (`UWexpression`
atoms) that refer to one another, with constants and mesh-variable symbols at the
leaves. The JIT expands that graph into a tree before it differentiates and prints it.
On the Spiegelman notch the viscosity is a graph of 137 distinct nodes and a tree of
12,191, and every stage after the expansion — the derivative, the constant scans, the C
printer, the C compiler and the kernel at run time — works on the tree.

We propose that the JIT keeps the graph. Each named sub-expression is lowered once, to
a C temporary, and the Jacobian is formed by the chain rule through the named
sub-expressions instead of by differentiating the expanded tree. A prototype builds the
Newton `uu_G3` block of the notch in half a second against 9 s for the library's route
on tier 1, emits 6.6 KB of C against 3.2 MB, and evaluates it in an eighth of the time.
The compiled kernels — residual, Picard and Newton — agree with the library's to
round-off, and the generated source depends neither on the hash seed nor on what the
script created before the law.

Tier 1 has removed most of the setup time the tree used to cost: on the notch the
library's full Newton setup went from 181 s to 16 s. What tier 1 cannot remove is
proportional to the tree — the size of the generated C, the compiler's time and memory,
and the kernel's run time at every quadrature point of every assembly — and the tree
grows with every named layer a law gains.

## Where we want to be

The JIT should compile the model the user wrote, not a copy of it with every name
erased. Five properties follow:

1. **Named quantities are the unit of compilation.** Each is lowered once, evaluated
   once per quadrature point, and differentiated once per argument; the tangent is
   assembled from those derivatives by the chain rule.
2. **SymPy's algebra acts locally.** Simplification, assumption queries and
   differentiation run on one named quantity's body at a time, never on the whole
   expanded model, so their cost and their surprises stay the size of a body.
3. **One walk produces the C and the constants manifest**, so the two cannot disagree
   and no guard has to check that they do.
4. **The generated C is a pure function of the mathematics and the data layout.** No
   class is mutated to make it print, and no name, counter, hash seed or rank enters it.
5. **The generated C is readable.** One line per named quantity, so a kernel can be
   checked by eye and a wrong one can be found.

Today's JIT has none of these. Tier 1 makes it fast enough and correct on the cases we
have found, with patches placed where each failure surfaced.

## The current pipeline expands every named sub-expression

A solver hands the JIT a `JITCallbackSet` of residual and Jacobian expressions.
`getext()` turns them into one C function per callback. For a Newton tangent:

1. The solver collects its residual sources `F0` and `F1`, which still contain the
   named sub-expressions as symbols.
2. `_jacobian_source` passes each source through `_jacobian_unwrap`, which expands
   every non-constant atom recursively (`unwrap_expression(mode="symbolic_keep_constants")`)
   and then adds $10^{-36}$ to the base of every half-integer power whose base has free
   symbols, constant atoms included (the sqrt guard, so that the tangent is finite at a
   state of rest).
3. The solver differentiates the expanded sources into the `G0`–`G3` blocks with
   explicit loops of `sympy.diff`.
4. `getext()` builds the constants manifest by testing every atom with
   `_is_truly_constant`, which itself unwraps the atom completely. For each callback it
   then reveals nested constants (a second complete unwrap), replaces the manifested
   constants by `_JITConstant` placeholders, unwraps what remains, scans the result
   for unconvertible symbols, and prints each output entry with SymPy's C99 printer.
5. The C source is hashed; the hash is the cache key. On a miss, rank 0 compiles the
   module and the other ranks load it.

The Picard tangent skips step 2. The atoms stay symbols when the solver differentiates,
so the derivative treats the effective viscosity as a constant, and `getext()` expands
them only when it prints. The residual is never guarded. A constitutive model's own
`flux_jacobian`, when it supplies one, also skips step 2 and is differentiated as given.

## The expanded tree is the cost, and it grows with every named layer

The notch viscosity (ViscoPlastic, Drucker–Prager yield with residual strength,
viscosity floor, the campaign law in `notch_model.py`) on tier 1 (`eb167214`; #830 generates byte-identical C), counted by
`scripts/sessions/jit_graph/graph_size_probe.py` without expanding anything:

| | graph | expanded tree |
|---|---:|---:|
| viscosity | 137 distinct nodes; 6 named non-constant atoms, 12 constants | 12,191 nodes |
| `F1`, all four entries | the same 137 nodes | 73,548 nodes |

The tree is large because each named atom is copied into every place that refers to
it, and the copies nest: the yield stress appears inside the plastic viscosity, which
appears inside the effective viscosity, which appears in every entry of `F1` and, after
differentiation, in every entry of `G3` — $d^4$ entries, 16 in two dimensions and 81 in
three. Each layer of named sub-expression a law gains multiplies the tree.

Issue #823 profiled the notch setup on the soft-min law of PR #794. Before tier 1, the
keep-constants unwrap of the viscosity alone took 563 s, because each fixed-point pass
re-traversed the growing tree and re-tested every atom for constancy, and a declared
`plastic_rate_strengthening` took the setup past 40 minutes. Tier 1 makes the unwrap a
single memoised pass, removes the repeated scans, keeps fields real through
differentiation, and holds the power-mean sharpness as an atom of its own; the notch's
full Newton setup falls from 181 s to 16 s. What remains is the work proportional to the
tree — differentiating it, printing it, compiling it and running it. #547 recorded a
1.45 MB header and 1.3 GB of `gcc -O3` memory on a 16-phase collision model. At run time
the expanded kernel evaluates the viscosity's tree once for each output entry that
contains it, unless the compiler recognises the repetition, and it runs at every
quadrature point of every residual and Jacobian assembly.

## The proposal: lower each named sub-expression once

### A node is an applied function of the leaves its value depends on

During lowering, each non-constant atom becomes a **node**: an applied undefined
function whose arguments are the leaves its value depends on, and whose body is its
`.sym` with each child atom replaced by the child's node. The leaves are the symbols the
kernel reads:

| leaf | C |
|---|---|
| field component and gradient (unknowns) | `petsc_u[i]`, `petsc_u_x[i]` |
| field component and gradient (auxiliary) | `petsc_a[i]`, `petsc_a_x[i]` |
| coordinate, boundary normal (base scalars) | `petsc_x[i]`, `petsc_n[i]` |
| constant atom | `constants[i]` |

A node is identified by its body: two atoms whose lowered bodies are equal share one
node. SymPy's equality sees the difference between two constants that share a display
name — the two $\eta$ of a two-material model — so bodies that use them are different
bodies and different nodes. The node's class name is a hash of its body written with
display identities only (a field's function name, a coordinate's system and index, a
constant's name), and its arguments are sorted by those identities, ties broken by
relative creation order as the manifest breaks them. Neither contains a creation
counter. The class also carries the serial number of the compile that made it in its
SymPy identity (`UndefinedFunction(..., _ctx=serial)`), so classes from two compiles
never compare equal and SymPy's cache cannot return a node whose body belongs to an
earlier version of the law.

Inside a body, a sub-expression that repeats without a name is split into an anonymous
node by common sub-expression elimination on that body alone, which is cheap because the
body is small.

### Derivatives follow from SymPy's own chain rule

A node's `fdiff(i)` returns its partial derivative with respect to argument slot $i$,
and SymPy's `Function._eval_derivative` assembles

$$
\frac{\partial n(a_1,\dots,a_k)}{\partial v} \;=\; \sum_{i=1}^{k} n_{,i}(a_1,\dots,a_k)\,\frac{\partial a_i}{\partial v}.
$$

So `sympy.diff`, `derive_by_array`, `Matrix.diff`, second derivatives and derivatives
with respect to a constant (a sensitivity) work through nodes with no change at the call
site. Tier 1 routes every Jacobian derivative in the solvers through
`uw.function.diff_wrt_field` / `uw.function.derive_by_array_wrt_field`, which differentiate with respect to a
field through a stand-in symbol that keeps the field's realness: the field is
`xreplace`d by the stand-in, the expression is differentiated, and the field is
substituted back. Nodes pass through it: the stand-in reaches their arguments, the
derivative follows their `fdiff`, and the substitution rebuilds them. On both prototype
fixtures, every derivative of the lowered `F1` with respect to the velocity gradient and
the pressure is identical through the stand-in and through `sympy.diff`. Tier 2 changes
none of those call sites.

The partial derivative is a node too, built on demand. Five details carry its
correctness:

- **It is keyed by argument slot, never by the leaf object.** Before calling `fdiff`,
  SymPy replaces the differentiation variable by a `Dummy`, so the arguments `fdiff`
  sees are not the leaves the body was written in. A prototype that looked the leaf up
  by object returned zero.
- **It is a partial derivative.** The body is differentiated with every leaf replaced
  by an independent dummy symbol. Differentiated directly, a field that is itself a
  function of the coordinates would be differentiated a second time through a
  coordinate argument.
- **Its arguments are the leaves its own body uses.** A constant that differentiates
  away does not remain an argument, so it does not reach the manifest.
- **A derivative that is zero, a number or a single leaf is returned as itself**, not
  as a node. Cancellation between levels of the graph is then seen when the lower level
  is a number ($A = x - y$, $B = A + y$ gives $\partial B/\partial y = -1 + 1 = 0$).
  Deeper symbolic cancellation is not seen: such an entry stays an expression that
  evaluates to zero. No solver tests a Jacobian block for zero, so the cost is the
  assembly of that entry, not a wrong answer.
- **A node cannot be a `Symbol` subclass.** `Derivative` returns zero before it calls
  `_eval_derivative` when the variable is an applied function that does not occur
  visibly in the expression — and the strain-rate symbols $L_{ij}$ are applied
  functions. Measured: a `Symbol` subclass whose `_eval_derivative` returns the right
  answer is differentiated to zero by `sympy.diff`. An applied function shows its
  dependencies as arguments, so the variable is visible.

Body partials are taken with respect to real dummy symbols, the same remedy tier 1's
stand-in applies at the call sites, for the same reason: with a real field $P$,
`sqrt((C + μP − τ)**2)` becomes `Abs(C + μP − τ)`, and SymPy 1.14 differentiating that with
respect to $P(x, y)$ through its own assumption-free dummy leaves `Derivative(P, P)`
unevaluated, which the C printer refuses.

### Picard, Newton and continuation keep their meaning

The tangent mode is decided, as now, by whether an atom is visible to the derivative.
Each atom has two node variants, and they are different nodes:

- **plain**: the body as written. The residual and the Picard tangent use it; the
  residual is never guarded.
- **guarded**: the body with the sqrt guard applied. `_jacobian_unwrap` replaces each
  non-constant atom of a Newton source by its guarded node instead of expanding it, so
  the derivative passes through it.

In the Picard tangent the atom stays a `UWexpression` symbol while the solver
differentiates, so the coefficient is frozen, and the JIT lowers it to its plain node
when it emits the kernel. The continuation blend contains both, and each lowers to its
own temporaries.

The guard moves from the expanded tree to each node body, under the same rule. The two
placements could differ only where SymPy merges powers across a sub-expression
boundary: a bare `sqrt(g)**(-2/3)` becomes `g**(-1/3)`, which is no longer a
half-integer power. A real law raises a quotient — $(\dot\varepsilon_{II}/\dot\varepsilon_0)^{1/n-1}$ —
which SymPy does not merge, and on a power law over a named invariant, with $n$ as a
constant atom or as the number 3, both routes give a finite Newton tangent at a state of
rest, equal entry for entry.

### Each lowering is read back by `getext()` from its nodes

Each call of `_jacobian_unwrap` lowers its source in a context of its own, and
`getext()` lowers the remaining atoms — those of the residual and the Picard blocks — in
another. Each node class holds its body and its context, so `getext()` reads every body
from the nodes it is handed and needs no new argument, and a caller that is not a solver
(`Integral`, `CellWiseIntegral`, `BdIntegral`) needs no change. An atom lowered in two
contexts is two nodes in Python and one temporary in C, because emission merges
temporaries by the C they compute. The solvers are therefore unchanged apart from
`_jacobian_unwrap`; one context per setup would only save the constancy test of an atom
that both lowerings meet.

`getext()` no longer runs `unwrap_expression` over a whole kernel. Its present phases —
reveal the constants, substitute them, unwrap the rest — become: lower atoms to nodes,
collect the manifest from the leaves, emit.

### A node expands to its body on request

Code outside the JIT that evaluates a Jacobian block must still see a plain expression:
`test_1066` builds its finite-difference oracle by passing `_uu_G3` through
`unwrap_expression(mode="nondimensional")` and `lambdify`. The unwrappers therefore
treat a node application as one more atom, whose one-level expansion is its body with
its arguments substituted. There are two: `unwrap_expression`, whose walkers
(`_unwrap_expression_once` and tier 1's complete expansion) look at `free_symbols`, which
never contains an application, and `unwrap_for_evaluate`, which `uw.function.evaluate`
uses. Both must visit node applications. `pure_sympy_evaluator` classifies any function
whose `__module__` is `None` as mesh-variable data; a node's class has `__module__`
`None`, so evaluation unwraps first. `getext()` must have stopped calling
`unwrap_expression` on whole kernels before the unwrappers learn to expand nodes, or a
kernel expands silently back into the tree.

A block unwrapped completely is the same function as today's tree. It is not always the
same expression: where SymPy merged a power across a sub-expression boundary, the tree
escaped the guard and the expanded nodes keep it.

### Emission writes one temporary per distinct computation

For each callback the JIT gives every node application reachable from its outputs a
**canonical key**: a hash of its body with each leaf written as the C the kernel reads
(`petsc_u[3]`, `petsc_x[0]`, `constants[2]`) and each child replaced by the child's key.
Applications with equal keys compute the same C and share one temporary. The
temporaries are written so that each follows the ones it uses, ties broken by key:

```c
const double t0 = ...;     /* one line per distinct computation, evaluated once */
const double t1 = ...;
out[0] = ...;              /* outputs in terms of t0, t1, ... and the leaves */
```

The generated source is therefore a function of the mathematics and of the kernel's data
layout alone. Python class names, creation counters, object identities and set
iteration order never reach it. Each body passes through the existing validation for
unconvertible symbols and integration-point derivatives.

### The constants manifest is built from the leaves

The manifest is the set of constant atoms among the leaves of the emitted kernels,
tested with the same predicate (`_is_truly_constant`, memoised once per setup) and
ordered by the same key (name, then creation order). It contains every slot today's
manifest contains. It can contain more: where the tree cancels a constant across a name
boundary ($A = c\,x$, then $A/c$), the graph keeps it, and the constant keeps its slot.
An extra slot is harmless — it is packed and read — but each one must be traced to such
a cancellation. The prototype finds none on its two fixtures, and the benchmark adds a
fixture whose Newton source differs from its residual (`set_jacobian_F1_source`).

### The cache key and the parallel agreement keep their mechanism

The key remains the hash of the generated source. Every cached module recompiles once
after the change, because the source text changes; the ABI salt already includes the
Underworld version.

The prototype's generated C is byte-identical under `PYTHONHASHSEED` 0, 1 and 2, after
a preamble that creates extra objects before the law (shifting every creation counter),
and when the law is declared twice in one process, as a re-run notebook cell does.
Whether canonical emission also removes the cross-rank disagreement of #752 is a
hypothesis we test (np ≥ 3, counting how often `_agree_source_across_ranks` repairs); the
repair stays.

## The prototype agrees with the library to round-off

`scripts/sessions/jit_graph/kernel_graph.py` is the lowering;
`scripts/sessions/jit_graph/graph_vs_library.py` builds three kernels of a fixture twice
and compiles each into a C function of the same leaves:

- the residual flux `F1`, lowered with plain nodes; the library route unwraps it as
  `getext()` does (keep-constants);
- the Picard `uu_G3`: `F1` differentiated with the atoms frozen, then lowered with plain
  nodes or unwrapped;
- the Newton `uu_G3`: the library's own `_jacobian_source`, or guarded nodes.

Derivatives in both routes go through tier 1's `diff_wrt_field` in the solver's explicit
loops. Two fixtures:

- **box**: `ViscoPlasticFlowModel` with a Drucker–Prager yield stress $C + \mu p$, a
  temperature-dependent viscosity $\eta_0 e^{-\theta T}$, a yield-stress floor and a
  viscosity floor, on a unit square. Built in the script.
- **notch**: the campaign law (`~/+Simulations/spiegelman_hardcase/drivers/notch_model.py`),
  with the power-mean soft minimum.

Measured on tier 1 at `eb167214` (#830 generates byte-identical C) with Apple clang `-O3`, while another session's test
suite held six of the machine's eight cores; the run times are indicative. One kernel
call is timed at one state, with the cost of an empty kernel's `ctypes` call (145–165 ns)
subtracted. Library time first, graph time second:

| | box | notch |
|---|---|---|
| residual `F1`: unwrap or lower | 0.08 s / 0.04 s | 0.07 s / 0.04 s |
| Newton source or lowering | 0.06 s / 0.03 s | 0.05 s / 0.04 s |
| Newton differentiate | 0.36 s / 0.15 s | 3.5 s / 0.35 s |
| Newton print or emit | 0.36 s / 0.04 s | 5.0 s / 0.08 s |
| C source: residual | 5.3 KB / 0.8 KB | 75 KB / 1.4 KB |
| C source: Picard `G3` | 8.5 KB / 0.8 KB | 125 KB / 1.4 KB |
| C source: Newton `G3` | 91 KB / 3.3 KB (20 temporaries) | 3.2 MB / 6.6 KB (65 temporaries) |
| `cc -O3`, Newton `G3` | 0.41 s / 0.58 s | 0.80 s / 0.40 s |
| one call: residual | 31 ns / 22 ns | 195 ns / 83 ns |
| one call: Picard `G3` | 51 ns / 32 ns | 355 ns / 91 ns |
| one call: Newton `G3` | 196 ns / 23 ns | 984 ns / 125 ns |

| agreement | box | notch |
|---|---|---|
| states | 2,000 | 3,000 |
| largest difference, relative to the block's largest entry: residual, Picard, Newton | $6.2\times10^{-16}$, $6.0\times10^{-15}$, $6.1\times10^{-15}$ | $9.2\times10^{-16}$, $4.4\times10^{-16}$, $4.4\times10^{-16}$ |
| bit-identical entries: residual, Picard, Newton | 93%, 88%, 65% | 79%, 64%, 34% |
| NaN in one route, or in both | none | none |
| entries the library computes as exactly 0 and the graph as round-off | 31 of 17,409 (Newton) | none |
| constants manifest | identical, 7 slots | identical, 12 slots |
| structural zeros | identical | identical |
| state of rest, all three kernels | finite, elementwise equal | finite, elementwise equal |
| generated C, all three kernels, both routes | byte-identical under `PYTHONHASHSEED` 0, 1, 2, after a preamble of extra objects, and with the law declared twice | — |

The states draw the velocity gradient over eight decades and every other field from a
normal distribution, except fields that are bounded by construction: the box's
temperature and the notch's material fraction lie in $[0, 1]$, and the notch's rate cap
is non-negative. Entries that differ by more than $10^{-12}$ of themselves are round-off
zeros, at most $10^{-16}$ of the block's largest entry, which cancel in both routes.

Before its last two commits, tier 1's route spent 43 s on the notch's Newton source and
72–131 s differentiating it. The source's time went into four `im(base)` constructions
inside `Pow.__new__` on whole expanded bases: the notch fixture uses the power-mean soft
minimum (opt-in; the default is the square-root form), whose exponent
$s = 1/(\delta + 0.001)$ has a sum in its denominator, and SymPy then evaluated the sign
of the base's imaginary part on every rebuild of the tree. Holding $s$ as a constant
atom of its own removed it. The graph never met that cost, because the bases of its
powers are small. What tier 1 cannot reduce is proportional to the tree: megabytes of C
and a kernel several times slower.

Outside those bounds the routes can disagree. With a material fraction above one, the
notch law's linear blend of yield stresses is negative, the soft maximum with a zero
floor is then exactly zero, and the law divides by it. The graph computes the yield
stress once, the soft maximum cancels exactly, and the tangent is NaN; the tree has
distributed constants differently in its two copies of the yield stress, so the
cancellation leaves rounding noise and the tangent is finite but meaningless. The law is
singular there. The notch cannot reach it — its material fraction is a P0 field holding
exactly 0 or 1 — but a model with a projected or higher-degree material field could.

(jit-graph-in-the-library)=
## In the library, against tier 1

Steps 2 and 3 are implemented on `feature/jit-graph-codegen`
(`src/underworld3/utilities/_jit_graph.py`, with branches in `getext()`,
`generate_c_source()`, `_jacobian_unwrap` and the two unwrappers), selected by
`UW_JIT_GRAPH=1`. Unset, every path is tier 1's. Both routes therefore run on one build,
and each fixture is run once per route in a fresh process with the JIT cache off
(`scripts/sessions/jit_graph/route_ab.py`): the solver's whole pointwise setup, a Newton
solve, then the residual and the Jacobian assembled repeatedly at the solution. The
fixtures are those of the prototype, with the box ones solved as a lid-driven box
started from simple shear; the notch is solved from rest to a relative tolerance of
$10^{-6}$. Apple clang `-O3`, macOS, single runs on a machine shared with another
session's seven solver processes, so the times are indicative. Tree first, graph second:

| | notch | VEP | box | TI VEP | power law | linear |
|---|---|---|---|---|---|---|
| pointwise setup, s | 15.9 / 2.5 | 3.4 / 2.3 | 2.5 / 1.8 | 1.5 / 1.5 | 1.5 / 1.3 | 1.2 / 1.0 |
| of which C generation, s | 8.6 / 0.23 | 0.61 / 0.21 | 0.33 / 0.10 | 0.17 / 0.14 | 0.05 / 0.04 | 0.02 / 0.01 |
| generated C, all modules of the solve | 4.0 MB / 22 KB | 380 KB / 40 KB | 104 KB / 16 KB | 42 KB / 31 KB | 28 KB / 14 KB | 10.7 KB, byte-identical |
| Jacobian assembly, ms | 180 / 132 | 2.40 / 2.19 | 1.58 / 1.45 | 2.23 / 2.22 | 1.48 / 1.40 | 1.39 / 1.44 |
| residual assembly, ms | 21.4 / 18.7 | 0.62 / 0.59 | 0.35 / 0.33 | 0.61 / 0.60 | 0.32 / 0.32 | 0.33 / 0.33 |
| Newton iterations, nonlinear / linear | 75 / 496 and 57 / 386 (see below) | 6 / 6, both | 30 / 30, both (limit) | 2 / 2, both | 16 / 16, both | 1 / 1, both |

The compile time is not shown separately: it counts every module the solve builds (the
VEP's history projections among them), and on these fixtures it is 1–4 s on either
route, never slower on the graph.

**The callbacks run about ten times faster; the assembly is mostly not callbacks.**
`assembly_split.py` times the notch's assembly at a solved state (minimum of 30), then
swaps the law for a constant viscosity on the same mesh, fields and boundary
conditions and times it again; the difference is the cost of the law's callbacks. Per
quadrature point and per Jacobian assembly, which builds the Jacobian and its
preconditioner as two matrices, the callbacks cost 3.1 µs on the tree and 0.29 µs on the
graph, 30% and 4% of the assembly; per residual, 300 ns and 63 ns. The rest of the
assembly, about 150–165 ms here, is the finite-element machinery, the same on both
routes, so on the notch the assembly can gain at most a third. A law with more named
layers, or a three-dimensional `G3` with 81 entries, gives the callbacks a larger share.

**On Linux with gcc the verdict holds, and gcc's `-fmath-errno` explains part of the
tree's cost on some laws.** On a Linux server (2× Xeon Gold 6240R, the environment's
gcc 14.3 at the default `-O3 -g0`, so with gcc's `-fmath-errno`; the machine fully
loaded by other jobs, a noise floor of about ±10%), the five self-contained fixtures take
identical solver paths on both routes. Setup falls from 10.3 to 8.8 s (box) and from
12.0 to 4.8 s (VEP); the Jacobian assembly ratios, graph to tree, are 0.87 (box), 0.94
(VEP, power law), 1.00 (TI) and 1.04 (linear, byte-identical C, so noise). Per
quadrature point, the box's Jacobian callbacks cost 8.2 µs on the tree and about 1 µs
on the graph, the noise floor of the subtraction. With `-fno-math-errno`, which lets gcc
treat `sqrt`, `pow` and `exp` as pure and merge repeated calls, the box's tree callbacks
fall to 2.8 µs (reproduced in a reverse-order repeat) and the graph's do not move. The
VEP's do not move on either route (tree 2.7–3.1 µs, graph 0.7–1.0 µs). So on gcc, part
of the tree's cost is repeated math calls that the flag alone would recover for tier 1;
the rest, and all of it in the VEP, is repeated arithmetic that only computing each
named quantity once removes. Apple clang does not set `errno` by default, which is why
the Mac does not show the effect. On the same server the #752 fixture agreed across
ranks in all 10 runs at np = 3 and at np = 4 on both routes, and
`tests/parallel/ptest_jit_cache.py` passed at np = 4 on both.

**The operators agree to round-off.** `route_assemble.py` assembles the residual and the
Jacobian on each route at the same state — rest (zero velocity, boundary values
imposed), the tree's final state and the graph's final state — and compares them entry
by entry. On every fixture, at every state, and for the Newton, Picard and continuation
tangents, no Jacobian entry differs from the other route's by more than $5.1\times10^{-16}$
of the largest entry in its row, and the sparsity patterns are identical. Residuals
agree to $10^{-14}$ of their largest entry, except at converged states, where the
residual is itself round-off ($10^{-13}$). The uncapped power law is singular at rest:
both routes give a non-finite residual there, in the same 126 entries.

**The solves agree, and the notch's iteration count is not a property of the route.** On
every fixture but the notch, both routes take the same path: equal nonlinear and linear
iteration counts, residual histories equal to $10^{-5}$ and solutions to $10^{-11}$ or
better. On the notch both converge, the tree in 75 Newton iterations and the graph in
57, to solutions that agree to $8\times10^{-7}$ of the largest velocity and
$2\times10^{-4}$ of the largest pressure, consistent with the tolerance. The two
residual histories agree to $10^{-9}$ for four iterations and part from the fifth: the
notch's Newton iteration amplifies differences at the level of round-off, and the
operators that drive it agree at that level at three states. Changing the body force by
$\pm1$ to $\pm7\times10^{-12}$ of itself shows how far: over 17 such solves per route,
the tree converged in 60 to 111 iterations (median 75) and once failed to converge in
300, and the graph converged every time, in 43 to 86 (median 69). The two distributions
are the same within the sample; the notch's iteration count is a property of the
problem, not of the route.

**The test suite passes on the graph route.** Every serial batch of `scripts/test.sh`
(tier A and B, levels 1 to 3) run with `UW_JIT_GRAPH=1`: 3,023 passed, none failed. The
first run found two defects, both in the fault-network laws and both fixed with a test:
a repeated Piecewise condition shared as a node (a value, which Piecewise refuses as a
condition), and a coordinate leaf without the C name the mesh sets.

**The source is canonical.** On the notch, box and VEP fixtures the graph route's C is
byte-identical under `PYTHONHASHSEED` 0, 1 and 2; `test_0024` checks that a law
declared after a preamble of unrelated objects and declared twice emits the same
header.

**The stand-in derivative is no longer load-bearing.** `plain_diff_probe.py` replaces
`diff_wrt_field` and `derive_by_array_wrt_field` in the solvers by plain `sympy.diff`
and `sympy.derive_by_array` and runs the Newton solve that motivated them (the
Drucker–Prager yield with a yield-stress floor at softness 0). The tree route fails in
the C printer, on the `Derivative` SymPy leaves beside the `Abs`; the graph route
converges, because every `Abs` sits in a node body, whose partials are taken against
real dummies.

## Against tier 1: no slower, more robust, less code

Tier 1 is the competitor. Each criterion is measured against it, not against the code
before it.

**No slower.** In the library, end to end, every stage on every fixture is as fast or
faster: the notch's pointwise setup takes 2.5 s against 15.9 s, its C is 22 KB against
4.0 MB, and its Jacobian assembles in 132 ms against 180 ms. A constant-viscosity Stokes
emits byte-identical C, because a constant law has no node to lower. The assembly gain
is smaller than the prototype's kernel timings suggested (an eighth of the time per
call) because the pointwise kernel is one part of the assembly, beside quadrature and
the element loop. On Linux with gcc the ratios are the same (below). Still to measure:
an idle machine with repeated runs, a three-dimensional fixture, and the small kernels
of `Integral` and `BdIntegral`.

**More robust.** Each failure class below needed a patch in tier 1, placed where it
surfaced; in the graph it cannot arise, or arises only in one body:

| failure class | tier 1 | graph | shown |
|---|---|---|---|
| SymPy's automatic algebra on an expanded base is slow (`im()` inside `Pow.__new__`, 40 s on the notch) | a constant atom added to the power-mean law; realness for unit-carrying parameters | bases are bodies, not trees | graph lowering took 0.04 s on every tier 1 commit, including those where the library took 40 s |
| realness turns `sqrt(x**2)` into `Abs`, whose derivative with respect to a field SymPy leaves unevaluated | a stand-in derivative at 49 call sites | body partials are taken against real dummies | with plain `sympy.diff` at every solver site, the Drucker–Prager floor law fails to compile on the tree route and solves on the graph route |
| manifest and C built by two walks (#302) | two consistency guards and `_reveal_constants` | one walk | by construction; manifests identical on all fixtures |
| generated C that differs between ranks (#752, open) | rank 0's source adopted | canonical emission | byte-identical under hash seeds 0–2 (notch, box, VEP), after a preamble and when re-declared (`test_0024`). Across ranks untested: the #752 fixture (`rank_agreement_752.py`, np = 2) disagreed in 0 of 10 runs on either route, against about 2 in 10 before tier 1, so it no longer reproduces the defect |
| field symbols given their C names by mutating their classes, in an order that matters | `ccode_patch_fns`, the coordinate recovery block | an explicit map from leaf to C | not yet: steps 2–3 still spell leaves through the patched printer; the map belongs to step 4 |
| generated C too large to read or to compile (#547) | opt-in CSE, lower optimisation flags | one line per named quantity | 22 KB against 4.0 MB for the notch's whole solve |

The graph brings one failure class tier 1 does not have: it cannot cancel a quantity
against its own reciprocal across a name.

**Less global state, not less code.** The earlier estimate here (530 lines replaced by
360) does not survive the implementation. The lowering module is 442 lines, about a
third of them docstrings, and the branches it needs elsewhere about 60. Step 4 deletes
about 390 lines of the tree route: `_reveal_constants`, the scanning half of
`_extract_constants`, `_collect_constant_atoms`, `_xreplace_shared`, `_unique_symbols`,
the tree lowering and its two consistency guards in `generate_c_source`, the coordinate
recovery, the opt-in CSE path, the tree guard in `_jacobian_unwrap`, and the unused
`prepare_for_cache_key` and `_createext`. Replacing the class patching of
`ccode_patch_fns` (148 lines with its comments) by a map from leaf to C would remove
about 100 more. The line count comes out about even. What changes is where the
correctness lives: in one module whose every risk has a test that fails when the
mechanism is broken (`test_0024`, each test checked against a mutation of the lowering),
instead of in patches at the places each failure surfaced. The stand-in derivative at
49 call sites becomes belt-and-braces, as the probe above shows; so would realness for
unit-carrying parameters and the patch to SymPy's private `BaseScalar._prop_handler`
table, which we would want to remove — neither is yet shown.

## Results agree to round-off, not bit for bit

The graph changes the order of evaluation. A temporary is rounded once, where the tree
may have folded constants across a sub-expression boundary: $2\,(\tau/(2\dot\varepsilon))$
is the tree's $\tau/\dot\varepsilon$, but in the graph the inner quotient is a temporary
and the factor of two stays. Kernel outputs therefore differ in the last few bits, and
converged solutions differ at the solver tolerance's round-off.

### The graph does not cancel across a name

SymPy cancels on construction: $x \cdot x^{-1}$ is $1$ as soon as both factors meet in
one product. In the tree they meet wherever the law multiplies a named quantity by its
own reciprocal through another name — $A = 1/x$, then $B = A\,x\,T$ is the tree's $T$.
In the graph $A$ is a temporary, $B$ is $t_A\,x\,T$, and at $x = 0$ that is
$\infty \cdot 0 = \text{NaN}$ where the tree gave the limit. The law is not singular
there; the cancellation was simply lost. In a Newton source the sqrt guard keeps the
usual reciprocal, $1/\dot\varepsilon_{II}$, finite; the residual and the Picard tangent
are not guarded. The acceptance criteria therefore include every residual and Picard
kernel at a state of rest, and the constitutive models are read for a quantity
multiplied by a name that holds its reciprocal.

### Acceptance

- every tier A and tier B test passes with its current tolerance — none is loosened;
- on every fixture, the assembled residual and Jacobian agree with the expanded route
  to $10^{-13}$ relative at a cold state, a converged state and a perturbed state, and
  every residual, Picard and Newton kernel is finite and equal to the tree's at a state
  of rest;
- a Newton solve of the notch converges under both routes with the same SNES reasons,
  and nonlinear and linear iteration counts on the fixtures are unchanged, or each
  change is explained;
- the constants manifest contains today's, with every extra slot traced to a
  cancellation, and every Jacobian entry that is zero today is zero or evaluates to zero.

## What does not change

- The solver code: the explicit Jacobian loops, the PETSc block layout
  (`petsc-jacobian-layout.md`), `JITCallbackSet` and the `getext()` signature.
- The `constants[]` contract: a manifested constant ramps without recompiling.
- The cache protocol: memory, disk, rank-0 compile behind a collective decision.
- The residual is never guarded, and the Picard tangent is frozen exactly as now.
- The slot dictionaries `getext()` returns are keyed by the callback objects the solver
  passed in, so the solvers' lookups (`ext_dict.jac[self._uu_G3]`) are unaffected.
- `describe()`, the transcript and the model fingerprints record the named, unexpanded
  forms (`str(template.sym)`) and the packed constants; none reads the generated C.
- A model's `flux_jacobian` is still differentiated as given.

## Risks, and the test that closes each

| risk | how it would fail | closed by |
|---|---|---|
| a dependency missing from a node's arguments | its derivative is silently zero: a Picard-like tangent | arguments from a complete leaf walk, base scalars included; assembled Jacobian against the expanded route on every fixture; a finite-difference oracle with an $h$ sweep (`test_1066`) |
| a slot derivative taken as a total derivative | a field differentiated twice through a coordinate | body partials against independent dummies; a unit test with a field and a coordinate in one node |
| derivative keyed by leaf instead of slot | zero derivative | slot keying; a unit test through `sympy.diff`, where the `Dummy` substitution happens |
| two constants with one display name merged | one material gets the other's coefficient | nodes identified by body equality, which distinguishes them; `test_0103`'s two-material case |
| a node class from an earlier compile reused | a changed `.sym` is ignored | the compile serial in each class's identity; a test that changes a body and rebuilds in one process without clearing SymPy's cache |
| source that depends on the hash seed, a creation counter or a re-declaration | every parallel run repairs; a new process or a re-run notebook cell misses the cache | canonical emission; `test_0105` extended with a Newton fixture, a preamble of extra objects and a re-declared law |
| a cancellation across a name lost | NaN at a state where the tree gave a finite limit, in an unguarded residual or Picard kernel | every residual and Picard kernel compared with the tree at a state of rest on every fixture |
| `test_0022` (tier 1) asserts the expanded, guarded tree of `_jacobian_unwrap`, and checks a block for `Derivative` with `has()`, which does not see node bodies | one test fails by construction, the other cannot fail | both rewritten against expanded nodes in the change that alters `_jacobian_unwrap` |
| an entry that is zero today becomes a non-zero expression | assembly of an entry that evaluates to zero | numbers and leaves inlined; zero patterns compared on every fixture |
| guard placement changes cold-start behaviour | NaN at a state of rest | `test_1067`, extended to a power-law viscosity (the prototype finds both routes finite and equal) |
| `getext()` or another walker expands nodes back into the tree | the cost returns, silently | `getext()` lowers atoms itself; a size check on the emitted source of the notch fixture |
| an atom whose `.sym` is a matrix | it cannot be a scalar temporary | such atoms are expanded in place, as now |
| verbose-output assertions in `test_0004` | a test fails on wording, not on a defect | rewrite those assertions against the kernel contract |
| overhead on small kernels | constant-viscosity problems get slower to set up | measured on the small fixtures; budget: no slower than today |
| a law singular at a state a model reaches (a zero denominator) | NaN where the tree gave rounding noise | the Newton solve of the acceptance criteria; such a law is the model's to fix |

## Benchmark plan

During development both routes run in one process on identical inputs, selected by a
private switch, so every comparison is A against B on one build. The switch and the
expanded route are removed before the change merges. The fixtures are built in
`scripts/sessions/jit_graph/` so that the measurements can be repeated from the
repository; the notch is the one exception, and its driver is named.

**Fixtures**

1. The notch, Newton: default soft-min, power-mean soft-min, and a declared
   `plastic_rate_strengthening`; refinement 1 and 3.
2. Poisson, linear and with $k(u)$; Stokes, viscous and power-law; viscoplastic hard-min
   under Picard and Newton, with the sqrt and power-mean smoothers and the continuation
   blend — the nine cases tier 1 checked for identical source.
3. A Newton source that differs from the residual (`set_jacobian_F1_source`).
4. SolCx with Nitsche free-slip, whose boundary terms carry the full stress.
5. Transversely isotropic power-law Stokes with rotated free-slip (the #752 fixture).
6. Visco-elasto-plastic Stokes with a stress history (`scripts/sessions/profile_jit_phases.py`).
7. A three-dimensional viscoplastic Stokes, where `G3` has 81 entries.
8. Small kernels: constant-viscosity Stokes, Poisson, `Integral` and `BdIntegral`.

**Measurements**

- setup time by phase (`uw.timing`): lowering, differentiation, constants, emission,
  compile — with `UW_NO_USAGE_METRICS=1`, since the import-time usage report makes an
  HTTP request on a thread that runs during setup;
- source size, `.so` size, and peak compiler memory;
- kernel run time: residual and Jacobian assembly (`SNESFunctionEval`,
  `SNESJacobianEval`) repeated on a fixed state, on macOS and on Linux, where gcc's
  default `-fmath-errno` keeps it from merging repeated `pow`, `exp` and `log` calls;
- agreement: compiled kernel outputs at random states (fraction bit-identical, largest
  difference relative to the block); assembled residual and Jacobian at the three
  states; converged solutions; SNES reasons and iteration counts; manifest and zero
  patterns;
- determinism: generated source under `PYTHONHASHSEED` 0, 1 and 2 on every fixture;
- parallel: source-hash agreement in ten runs each at np = 3 and np = 4 (within the
  eight-core budget), counting repairs, and `tests/parallel/ptest_jit_cache.py` run by
  hand at np = 4 — it matches no CI glob, so CI has never run it;
- cache: ramping a constant leaves the key unchanged, a second process hits the disk
  cache, and so does a second process whose script creates other objects first.

## Staging

1. Tier 1 merges.
2. `getext()` emits every callback from the graph and stops unwrapping whole kernels.
   Residual and Picard kernels gain the temporaries; Newton sources still arrive
   expanded and print as before. Nodes exist only inside `getext()`.
3. `_jacobian_unwrap` builds nodes instead of expanding, so nodes reach the solver's
   blocks; in the same change the two unwrappers learn to expand them and `test_0022`'s
   tree-shaped guard test is pinned to the tree route.

   Steps 2 and 3 are implemented together behind `UW_JIT_GRAPH=1` (2026-10-08).
4. The expanded route and the switch are removed, together with the two functions that
   mirror it and have no callers (`prepare_for_cache_key`, `_createext`), and
   `jit-cache.md`, `expressions-functions.md` and `jacobian-consistent-tangent.md` are
   updated. `jit-cache.md` already describes the cross-rank check as an abort and every
   rank as compiling; both stopped being true before this change.

Each step is benchmarked against the one before it and reviewed adversarially before
the next begins.

## CSE on the tree, call-site chain rules and Symbol nodes were rejected

- **Common sub-expression elimination on the expanded tree** (`UW_JIT_CSE=1`, opt-in
  today). It shrinks the C, but only after the tree has been built, differentiated and
  scanned, and SymPy's `cse` on a tree of $10^5$ nodes is itself slow.
- **A chain-rule function called at each Jacobian site.** Tier 1 has since put one
  function at every site, for realness, and a chain rule could live there. It would
  still leave the derivative's correctness to the call: any derivative taken another
  way (a test's oracle, an adjoint, a user's `sympy.diff`) would see frozen nodes. With
  `fdiff` on the node, every route gives the same answer.
- **A `Symbol` subclass with its own `_eval_derivative`.** SymPy returns zero before
  calling it.

## Decisions

- **2026-10-07: tier 2 may change the generated C.** Tier 1's speed work keeps the
  generated C byte-identical, because bit-identical kernels are what the JIT cache rests
  on. Tier 2 redesigns the JIT, and that constraint does not bind it: every kernel
  containing a named non-constant quantity is evaluated in a new order and agrees with
  tier 1 to round-off, not bit for bit. A law with no such quantity emits byte-identical
  C. After the change the C is canonical, bit-identical across hash seeds, preambles,
  re-declarations and processes by construction.

## Questions for the maintainer

1. One PR for steps 2–4, or one PR per step.
2. Whether the generated C should carry each temporary's display name as a comment.
   It makes kernels readable; it also puts display names into the cache key, so a
   `rename()` recompiles.
3. Whether the adjoint's own unwrap (`_peel_except`, on the adjoint branches) adopts the
   nodes in this change or after it.
4. Whether `UW_JIT_CSE` is retired once this lands.
