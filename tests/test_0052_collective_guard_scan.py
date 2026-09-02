"""Static guard: an MPI collective must be reached on every rank.

The recurring parallel defect in this library is not a missing collective, it
is a collective that *some* ranks reach. The starved ranks run on; the rest sit
in the reduction until the job is killed. It survives review because the guard
looks like ordinary control flow --- ``if module is None:``, ``if
undecided.size:``, ``if snapped_any.any():`` --- and it survives testing
because at two ranks a mismatched pair often still meets. Every instance found
so far passed at ``np=2`` and hung at ``np=4``.

``uw.mpi.selective_ranks()`` already catches the *deliberate* form, where the
author has said which ranks execute. It cannot catch this one: a data
predicate never enters that context, because the author does not think of it
as rank selection.

So this scans the source instead, and asks one question of every ``comm.<op>``
call: is it reached unconditionally? The rule the library already follows,
stated in ``line_cut.py`` and ``reconnect.py``, is **reduce first, then
branch** ---

    # COLLECTIVE, and reached on every rank: one with nothing to flip still
    # has to vote or its peers block waiting for it.
    n = uw.mpi.comm.allreduce(len(new_edges), op=MPI.SUM)
    if n == 0:
        break

Guards that every rank evaluates identically are fine and are classified as
such: the communicator geometry (``uw.mpi.size > 1``), the mesh's own
rank-invariant description, a function parameter, or a value that has itself
been reduced.

This is a name-level scan, not a proof. It cannot see through a helper, and a
predicate it calls rank-local may in fact be uniform for reasons it cannot
read. That is why :data:`ACCEPTED` exists --- but an entry there is a claim
that *this* predicate is rank-uniform, and it needs the reason written next to
it.
"""

import ast
import pathlib
import textwrap

import pytest

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "underworld3"

# mpi4py collective entry points. Capitalised is the buffer form, lower case
# the pickling form; both block until every rank in the communicator arrives.
MPI_COLLECTIVES = frozenset({
    "Barrier", "barrier",
    "Allreduce", "allreduce", "Reduce", "reduce",
    "Bcast", "bcast",
    "Gather", "gather", "Gatherv",
    "Allgather", "allgather", "Allgatherv",
    "Scatter", "scatter", "Scatterv",
    "Alltoall", "alltoall", "Alltoallv",
    "Scan", "scan", "Exscan", "exscan",
})

# Receivers that denote a communicator. Matching on the attribute name alone
# would catch `h5file.write` and `sympy_expr.gather`, so the call must be made
# ON something called `comm` (`uw.mpi.comm.allreduce`, `self._comm.bcast`).
COMM_NAMES = frozenset({"comm", "COMM_WORLD", "_comm", "mpi_comm"})

# Quantities every rank agrees on by construction.
UNIFORM_TOKENS = (
    "mpi.size", "comm.size", "mpi.rank", "comm.rank",
    ".dim", ".cdim", "dimension", ".degree", ".simplex", ".order",
    "isinstance", "hasattr", "__debug__",
)
# A value that has been through a collective is uniform thereafter.
REDUCED_TOKENS = ("allreduce", "bcast", "allgather", "gather", "Allreduce")
# Shapes and emptiness of arrays this rank computed: the defect shape.
LOCAL_TOKENS = (
    ".size", "len(", ".any()", ".all()", ".shape", ".sum()", ".empty",
    "isNone", "isnotNone", "==None", "==0", "!=0", ">0", "in",
)

#: Sites where the guard is rank-uniform for a reason the scan cannot read.
#: Keyed by ``(file, function)``; the value is why it is safe. Adding an entry
#: is asserting that every rank takes the same branch --- not that the failure
#: has not been seen yet.
ACCEPTED = {
    ("utilities/place_surface.py", "_place_thin_volume_2d"):
        "`outcropping` is a pure function of `asm_pts`, which arrives by "
        "`comm.bcast`, and of the domain complex from "
        "`_domain_boundary_facets`, which is allgathered and deduplicated by "
        "exact coordinate identity. Both are the same bytes everywhere, so "
        "every rank computes the same flag.",
    ("utilities/rotated_bc.py", "solve_rotated_freeslip"):
        "the cache is created and destroyed on all ranks together, so "
        "`cache is not None` is uniform; the allgather inside exists "
        "precisely to keep the verdict collective once it is.",
}

#: Real defects of this shape whose fix is already in an open pull request.
#: Distinct from :data:`ACCEPTED`: nothing here is safe, it is merely already
#: written down. An entry must name the issue, and
#: :func:`test_accepted_list_has_no_stale_entries` deletes it on merge.
IN_FLIGHT = {
    ("utilities/_jitextension.py", "getext"):
        "#612 -- the JIT compile gate reaches `comm.Barrier` only on ranks "
        "that missed the in-memory cache. Fixed on bugfix/jit-collective-"
        "gate by reducing `module is None` before the gate.",
}


def reduced_names(node):
    """Names assigned the result of a collective, anywhere under *node*.

    ``n = comm.allreduce(len(victims), op=MPI.SUM)`` makes ``n`` the same on
    every rank, so ``if n == 0: break`` is uniform even though the predicate
    reads like a local count. Without this the house idiom --- reduce first,
    then branch --- is exactly what the scan complains about.
    """
    assigned = set()
    for statement in ast.walk(node):
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        value = statement.value
        if value is None:
            continue
        if not any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr in MPI_COLLECTIVES
            for call in ast.walk(value)
        ):
            continue
        targets = (statement.targets if isinstance(statement, ast.Assign)
                   else [statement.target])
        for target in targets:
            assigned |= {n.id for n in ast.walk(target)
                         if isinstance(n, ast.Name)}
    return assigned


def classify(test, params, reduced=frozenset()):
    """Classify a guard predicate as uniform across ranks, or rank-local."""
    source = ast.unparse(test)
    squashed = source.replace(" ", "")

    if any(token in squashed for token in REDUCED_TOKENS):
        return "reduced"

    names = {n.id for n in ast.walk(test) if isinstance(n, ast.Name)}
    if names and names <= set(reduced):
        return "reduced"

    # Strip the uniform quantities before looking for local ones, so that
    # `uw.mpi.size > 1` is not read as a size test on somebody's array.
    residue = squashed
    for token in UNIFORM_TOKENS:
        residue = residue.replace(token, "")
    if any(t in squashed for t in UNIFORM_TOKENS) and not any(
        t in residue for t in LOCAL_TOKENS
    ):
        return "uniform"

    if any(token in squashed for token in LOCAL_TOKENS):
        # A guard on nothing but this function's own parameters is uniform
        # whenever callers pass the same value everywhere, which is the
        # ordinary contract for a keyword such as `axis=` or `percentile=`.
        if names and names <= params:
            return "argument"
        return "local-data"

    return "other"


class Scan(ast.NodeVisitor):
    """Collect collective calls that are not reached unconditionally."""

    def __init__(self):
        self.guards = []
        self.params = set()
        self.reduced = set()
        self.function = None
        self.loop = None
        self.findings = []

    def visit_FunctionDef(self, node):
        arguments = node.args
        saved = (self.function, self.guards, self.params, self.loop,
                 self.reduced)
        self.function = node.name
        self.guards = []
        self.loop = None
        # A nested def closes over the enclosing parameters, so they stay in
        # scope: `_reduce_dt` branching on its parent's `percentile=` keyword
        # is as uniform as if it took the keyword itself.
        self.params = self.params | {
            a.arg for a in [*arguments.posonlyargs, *arguments.args,
                            *arguments.kwonlyargs]
        }
        self.reduced = self.reduced | reduced_names(node)
        self.generic_visit(node)
        (self.function, self.guards, self.params, self.loop,
         self.reduced) = saved

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_If(self, node):
        kind = classify(node.test, self.params, self.reduced)
        source = ast.unparse(node.test)
        for branch in (node.body, node.orelse):
            self.guards.append((kind, source))
            for statement in branch:
                self.visit(statement)
            self.guards.pop()

    def _visit_loop(self, node):
        # A `break` under a rank-local guard skips the remaining iterations —
        # and with them any collective they would have reached. Recorded per
        # loop so the two halves can be matched up after the walk.
        saved, self.loop = self.loop, {"exits": [], "collectives": []}
        self.generic_visit(node)
        finished, self.loop = self.loop, saved
        if finished["exits"] and finished["collectives"]:
            lineno, guard = finished["exits"][0]
            call = finished["collectives"][0][1]
            self.findings.append((lineno, self.function, call, guard))
        # A collective in an inner loop is inside the outer one as well, so
        # carry it up; otherwise a `break` in the outer loop looks harmless.
        if self.loop is not None:
            self.loop["collectives"].extend(finished["collectives"])

    visit_For = _visit_loop
    visit_While = _visit_loop
    visit_AsyncFor = _visit_loop

    def _visit_exit(self, node):
        divergent = [g for g in self.guards if g[0] == "local-data"]
        if self.loop is not None and divergent:
            self.loop["exits"].append((node.lineno, divergent[0][1]))
        self.generic_visit(node)

    visit_Break = _visit_exit
    visit_Return = _visit_exit

    def visit_Call(self, node):
        function = node.func
        if isinstance(function, ast.Attribute):
            base = function.value
            receiver = base.attr if isinstance(base, ast.Attribute) else (
                base.id if isinstance(base, ast.Name) else None)
            if function.attr in MPI_COLLECTIVES and receiver in COMM_NAMES:
                call = f"{receiver}.{function.attr}"
                if self.loop is not None:
                    self.loop["collectives"].append((node.lineno, call))
                divergent = [g for g in self.guards if g[0] == "local-data"]
                if divergent:
                    self.findings.append(
                        (node.lineno, self.function, call, divergent[0][1])
                    )
        self.generic_visit(node)


def scan_source():
    """Every collective under a rank-local guard, as (file, line, ...)."""
    findings = []
    for path in sorted(SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        scan = Scan()
        scan.visit(tree)
        relative = path.relative_to(SRC).as_posix()
        for lineno, function, call, guard in scan.findings:
            findings.append((relative, lineno, function, call, guard))
    return findings


def test_source_tree_is_scannable():
    """The scan must actually see the source, or it passes by finding nothing.

    A guard that cannot fail is not a guard: if the path were wrong, or the
    matching silently stopped working, every assertion below would hold.
    """
    assert SRC.is_dir(), f"source tree not found at {SRC}"

    collectives = 0
    for path in SRC.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if not isinstance(f, ast.Attribute):
                continue
            base = f.value
            receiver = base.attr if isinstance(base, ast.Attribute) else (
                base.id if isinstance(base, ast.Name) else None)
            if f.attr in MPI_COLLECTIVES and receiver in COMM_NAMES:
                collectives += 1

    assert collectives > 100, (
        f"only {collectives} collective calls found in {SRC}; the scan has "
        "stopped matching and every other assertion here is vacuous"
    )


def test_guard_classification_fires_on_the_known_shape():
    """Negative control: the classifier must call the real defect rank-local.

    The three predicates below are the ones that actually deadlocked --- the
    JIT compile gate (#612), the swarm migration working set, and the adapt
    boundary snap. The uniform cases beneath them must NOT be flagged, or the
    scan drowns its own signal.
    """
    def guard_kind(source, params=frozenset()):
        return classify(ast.parse(source, mode="eval").body, set(params))

    assert guard_kind("module is None") == "local-data"
    assert guard_kind("undecided.size") == "local-data"
    assert guard_kind("snapped_any.any()") == "local-data"
    assert guard_kind("len(refine) > 0") == "local-data"

    assert guard_kind("uw.mpi.size > 1") == "uniform"
    assert guard_kind("self.mesh.dim == 2") == "uniform"
    assert guard_kind("axis is None", params={"axis"}) == "argument"
    assert guard_kind("uw.mpi.comm.allreduce(int(n)) == 0") == "reduced"


def test_scan_finds_an_injected_defect_and_spares_the_fix():
    """End-to-end control: the walker must flag the bug and clear the fix.

    Classifying the predicate correctly is not enough --- the finding only
    reaches the report if the visitor also tracks which guards enclose the
    call. Both shapes below are taken from real fixes: the swarm working set
    and the adapt boundary snap.
    """
    def findings(source):
        scan = Scan()
        scan.visit(ast.parse(textwrap.dedent(source)))
        return scan.findings

    defective = findings("""
        def migrate(self):
            undecided = numpy.where(~already)[0]
            if undecided.size:
                total = uw.mpi.comm.allreduce(undecided.size, op=MPI.SUM)
    """)
    assert len(defective) == 1, f"the guarded collective was missed: {defective}"
    assert defective[0][2] == "comm.allreduce"
    assert "undecided.size" in defective[0][3]

    # The same code with the reduction hoisted out is what the fix looks like.
    assert not findings("""
        def migrate(self):
            undecided = numpy.where(~already)[0]
            total = uw.mpi.comm.allreduce(undecided.size, op=MPI.SUM)
            if total:
                pass
    """)

    # A `break` on rank-local state skips the remaining iterations, and any
    # collective they would have reached.
    early_exit = findings("""
        def sweep(self):
            for step in range(10):
                if not victims.size:
                    break
                n = uw.mpi.comm.allreduce(len(victims), op=MPI.SUM)
    """)
    assert len(early_exit) == 1, f"the local `break` was missed: {early_exit}"

    # Exiting on the reduced value instead is the house idiom, and is clean.
    assert not findings("""
        def sweep(self):
            for step in range(10):
                n = uw.mpi.comm.allreduce(len(victims), op=MPI.SUM)
                if n == 0:
                    break
    """)


def test_no_collective_under_a_rank_local_guard():
    """Every ``comm.<op>`` is reached on every rank, or is listed in ACCEPTED.

    A failure here is a latent ``np>1`` hang, not a style point.
    """
    known = set(ACCEPTED) | set(IN_FLIGHT)
    unexpected = [f for f in scan_source() if (f[0], f[2]) not in known]

    if unexpected:
        report = "\n".join(
            f"  {path}:{line} in {function}()\n"
            f"      {call} is skipped when: not ({guard})"
            for path, line, function, call, guard in unexpected
        )
        pytest.fail(
            f"{len(unexpected)} collective call(s) reached only on some "
            f"ranks:\n{report}\n\n"
            "A rank that takes the other branch never arrives, and its peers "
            "wait in the reduction until the job is killed. Reduce first, "
            "then branch:\n"
            "    n = uw.mpi.comm.allreduce(int(local.size), op=MPI.SUM)\n"
            "    if n == 0:\n"
            "        return\n"
            "If the predicate really is the same on every rank, add it to "
            "ACCEPTED in this file with the reason."
        )


def test_accepted_list_has_no_stale_entries():
    """An ACCEPTED entry whose site is gone must be removed, not left to rot.

    Otherwise the list silently grows into a blanket exemption and a genuine
    new defect at the same coordinates is waved through.
    """
    live = {(path, function) for path, _line, function, _call, _guard
            in scan_source()}
    stale = sorted((set(ACCEPTED) | set(IN_FLIGHT)) - live)
    assert not stale, (
        "ACCEPTED / IN_FLIGHT list sites that no longer trip the scan; "
        "delete them:\n"
        + "\n".join(f"  {path}::{function}" for path, function in stale)
    )
