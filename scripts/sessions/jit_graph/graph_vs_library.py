"""Kernels built two ways, compiled, and compared (tier 2 of #823).

  library: today's route on tier 1 -- the Newton source from the solver's own
           ``_jacobian_source``, residual and Picard kernels unwrapped as ``getext``
           unwraps them (keep-constants), derivatives through ``diff_wrt_field`` with the
           solver's explicit uu_G3 loops, each output printed whole;
  graph:   ``kernel_graph`` nodes, the same loops, one temporary per node.

Three kernels per fixture: the residual flux F1 (plain nodes), the Picard uu_G3 (atoms
frozen while differentiating, plain nodes after) and the Newton uu_G3 (guarded nodes).
Each pair is compiled into C functions of the same leaves and evaluated at random
states and at a state of rest.

Determinism checks: run under several PYTHONHASHSEED values, with ``-uw_preamble N``
(N throwaway objects created first, shifting every creation counter) and with
``-uw_redeclare 1`` (the law built twice in one process, as a re-run notebook cell
does); the printed md5s must not change.

Fixtures are in fixtures.py: box, powerlaw, linear, vep and ti are built there and
reproducible from the repository; notch is the Spiegelman campaign law
(~/+Simulations/spiegelman_hardcase/drivers/notch_model.py).
"""
import ctypes
import hashlib
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import sympy
from sympy.core.function import AppliedUndef
from sympy.printing.c import c_code_printers
from sympy.vector.scalar import BaseScalar

import underworld3 as uw
from underworld3.function.expressions import UWexpression, unwrap_expression
from underworld3.function import diff_wrt_field
from underworld3.utilities._jitextension import _extract_constants, _pack_constants

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel_graph import KernelGraph, _KernelNode, emission_order, guard   # noqa: E402
import fixtures   # noqa: E402

params = uw.Params(
    uw_fixture=uw.Param("box", description="box | powerlaw | linear | vep | ti | notch"),
    uw_states=uw.Param(2000, description="random states for the value comparison"),
    uw_calls=uw.Param(200000, description="kernel calls for the run-time measurement"),
    uw_preamble=uw.Param(0, description="throwaway expressions and one mesh created before the law"),
    uw_redeclare=uw.Param(0, description="1 = build the law twice and use the second"),
    uw_numeric_n=uw.Param(0, description="powerlaw: 1 = the stress exponent as a plain number, not a constant atom"),
)
fixture = str(params.uw_fixture)


for i in range(int(params.uw_preamble)):
    uw.expression(f"preamble_{i}", float(i))
if int(params.uw_preamble):
    uw.meshing.UnstructuredSimplexBox(cellSize=0.5)
kwargs = {"numeric_n": bool(int(params.uw_numeric_n))} if fixture == "powerlaw" else {}
stokes, admissible = fixtures.build(fixture, **kwargs)
if int(params.uw_redeclare):
    stokes, admissible = fixtures.build(fixture, **kwargs)
mesh = stokes.mesh
dim = mesh.dim
L = stokes.Unknowns.L
F1 = sympy.Array(stokes.F1.sym).reshape(dim, dim)


def g3(F):
    """The solver's explicit uu_G3 loops, differentiating through diff_wrt_field."""
    F = sympy.Array(F).reshape(dim, dim)
    G = sympy.zeros(dim * dim, dim * dim)
    for gc in range(dim):
        for dg in range(dim):
            dF = diff_wrt_field(F, L[gc, dg])
            for fc in range(dim):
                for df in range(dim):
                    G[fc * dim + gc, df * dim + dg] = dF[fc, df]
    return list(G)


def keep_constants(e):
    return unwrap_expression(e, mode="symbolic_keep_constants")


timing = {}
def timed(label, fn):
    t = time.perf_counter()
    out = fn()
    timing[label] = time.perf_counter() - t
    return out


F1_flat = [F1[i, j] for i in range(dim) for j in range(dim)]
graph = KernelGraph()
kernels = {}
kernels["residual F1"] = (
    timed("library residual: unwrap", lambda: [keep_constants(e) for e in F1_flat]),
    timed("graph residual: lower", lambda: [graph.lower(e, guarded=False) for e in F1_flat]))
picard_raw = timed("both Picard: differentiate (atoms frozen)", lambda: g3(F1))
kernels["Picard G3"] = (
    timed("library Picard: unwrap", lambda: [keep_constants(e) for e in picard_raw]),
    timed("graph Picard: lower", lambda: [graph.lower(e, guarded=False) for e in picard_raw]))
F1_lib = timed("library Newton: source (unwrap + guard)",
               lambda: stokes._jacobian_source(F1, stokes._newton_flux(F1)))
G3_lib = timed("library Newton: differentiate", lambda: g3(F1_lib))
F1_graph = timed("graph Newton: lower",
                 lambda: sympy.Array([guard(graph.lower(e, guarded=True)) for e in F1_flat],
                                     (dim, dim)))
G3_graph = timed("graph Newton: differentiate", lambda: g3(F1_graph))
kernels["Newton G3"] = (G3_lib, G3_graph)

# ------------------------------------------------------------------ spelling of leaves
def leaves_of(exprs):
    out = set()
    for x in exprs:
        x = sympy.sympify(x)
        out |= {a for a in x.atoms(AppliedUndef) if not isinstance(a, _KernelNode)}
        out |= {s for s in x.free_symbols if isinstance(s, (UWexpression, BaseScalar))}
    return out

graph_bodies = [b for _, gr in kernels.values() for _, b in emission_order(gr, str)[0]]
graph_leaves = leaves_of([x for _, gr in kernels.values() for x in gr] + graph_bodies)
library_leaves = leaves_of([x for lib, _ in kernels.values() for x in lib])
leaves = graph_leaves | library_leaves
fields = sorted((s for s in leaves if isinstance(s, AppliedUndef)), key=KernelGraph.display_key)
slot = {f: k for k, f in enumerate(fields)}
library_consts = {e for _, e in _extract_constants(
    tuple(sympy.ImmutableMatrix([list(lib)]) for lib, _ in kernels.values()), mesh)[0]}
graph_consts = {s for s in graph_leaves if isinstance(s, UWexpression)}
consts = sorted(library_consts | graph_consts, key=lambda e: (e.name, e.instance_number))
cindex = {c: i for i, c in enumerate(consts)}
cvals = np.asarray(_pack_constants(list(enumerate(consts))), dtype=float)


def spell(s):
    if s in slot:
        return f"petsc_u[{slot[s]}]"
    if s in cindex:
        return f"constants[{cindex[s]}]"
    if isinstance(s, BaseScalar):
        return f"petsc_x[{s._id[0]}]"
    return str(s)


spelled = {s: sympy.Symbol(spell(s)) for s in leaves}
printer = c_code_printers["c99"]({})
def c_of(e, temps=None):
    e = sympy.sympify(e)
    if temps:
        e = e.xreplace(temps)
    return printer.doprint(e.xreplace(spelled))

# ------------------------------------------------------------------ compile and compare
SIG = "void k(const double *petsc_u, const double *petsc_x, const double *constants, double *out)"
HDR = ("#include <math.h>\n"
       "static inline double Heaviside_1(double x){return x<0?0:x>0?1:0.5;}\n")
work = tempfile.mkdtemp(dir=os.path.expanduser("~/+Simulations"), prefix=f"jit_graph_{fixture}_")
P = ctypes.POINTER(ctypes.c_double)
md5 = lambda s: hashlib.md5(s.encode()).hexdigest()[:10]


def compile_c(tag, body):
    cfile = os.path.join(work, f"{tag}.c")
    open(cfile, "w").write(HDR + SIG + " {\n" + body.replace("Heaviside(", "Heaviside_1(") + "\n}\n")
    so = os.path.join(work, f"{tag}.so")
    t = time.perf_counter()
    subprocess.run(["cc", "-std=c99", "-O3", "-g0", "-shared", "-fPIC", cfile, "-o", so], check=True)
    lib = ctypes.CDLL(so)
    lib.k.restype = None
    return lib, time.perf_counter() - t, os.path.getsize(so)


empty, _, _ = compile_c("empty", "")
print(f"[{fixture}] seed {os.environ.get('PYTHONHASHSEED', 'random')}  preamble {params.uw_preamble}  "
      f"redeclare {params.uw_redeclare}")
for k, v in timing.items():
    print(f"  {k:44s} {v:9.2f} s")
extra = sorted(c.name for c in graph_consts - library_consts)
missing = sorted(c.name for c in library_consts - graph_consts)
print(f"constants: library {len(library_consts)}, graph {len(graph_consts)}; "
      f"graph has the library's: {not missing}; extra in graph: {extra}")

rng = np.random.default_rng(7)
states = []
Lfields = {f for f in fields if f in set(L)}
for _ in range(int(params.uw_states)):
    scale = 10.0 ** rng.uniform(-6, 2)
    u = rng.normal(size=len(fields))
    for f, k in slot.items():
        if f in Lfields:
            u[k] *= scale
        for name, (lo, hi) in admissible.items():
            if f.func.__name__.strip("{}") == name:
                u[k] = rng.uniform(lo, hi)
    states.append((u, rng.uniform(0, 1, size=3)))
rest = (np.zeros(len(fields)), np.full(3, 0.5))

for name, (lib_exprs, graph_exprs) in kernels.items():
    t = time.perf_counter()
    order, key = emission_order(graph_exprs, spell)
    temps_by_key = {k: sympy.Symbol(f"t{i}", real=True) for i, (k, _) in enumerate(order)}
    temps = {app: temps_by_key[k] for app, k in key.items()}
    src_graph = "\n".join([f"const double {temps_by_key[k]} = {c_of(body, temps)};" for k, body in order]
                          + [f"out[{i}] = {c_of(x, temps)};" for i, x in enumerate(graph_exprs)])
    t_emit = time.perf_counter() - t
    t = time.perf_counter()
    src_lib = "\n".join(f"out[{i}] = {c_of(x)};" for i, x in enumerate(lib_exprs))
    t_print = time.perf_counter() - t
    tag = name.replace(" ", "_")
    lib_g, cc_g, so_g = compile_c(f"{tag}_graph", src_graph)
    lib_l, cc_l, so_l = compile_c(f"{tag}_library", src_lib)
    nout = len(graph_exprs)

    def call(lib, u, x):
        out = np.zeros(nout)
        lib.k(u.ctypes.data_as(P), x.ctypes.data_as(P), cvals.ctypes.data_as(P), out.ctypes.data_as(P))
        return out

    nonzero = exact = nan_mismatch = nan_both = lib_zero_graph_not = 0
    worst_block = 0.0
    for u, x in states:
        a, b = call(lib_g, u, x), call(lib_l, u, x)
        nan_mismatch += int(np.any(np.isnan(a) != np.isnan(b)))
        nan_both += int(np.sum(np.isnan(a) & np.isnan(b)))
        lib_zero_graph_not += int(np.sum((a != 0) & (b == 0)))
        m = (b != 0) & np.isfinite(b) & np.isfinite(a)
        nonzero += m.sum(); exact += (a[m] == b[m]).sum()
        if m.any():
            worst_block = max(worst_block, float(np.max(np.abs(a[m] - b[m])) / np.max(np.abs(b[m]))))
    a0, b0 = call(lib_g, *rest), call(lib_l, *rest)
    zl = sum(sympy.sympify(x) == 0 for x in lib_exprs)
    zg = sum(sympy.sympify(x) == 0 for x in graph_exprs)

    per = {}
    u, x = states[0]
    for tg, lib in (("graph", lib_g), ("library", lib_l), ("empty", empty)):
        out = np.zeros(max(nout, 1))
        args = (u.ctypes.data_as(P), x.ctypes.data_as(P), cvals.ctypes.data_as(P), out.ctypes.data_as(P))
        n = int(params.uw_calls)
        t = time.perf_counter()
        for _ in range(n):
            lib.k(*args)
        per[tg] = 1e9 * (time.perf_counter() - t) / n

    print(f"--- {name}: md5 graph {md5(src_graph)}  library {md5(src_lib)}")
    print(f"    emit {t_emit:.2f} s, print {t_print:.2f} s; C graph {len(src_graph):,} B "
          f"({len(order)} temporaries), library {len(src_lib):,} B; cc -O3 {cc_g:.2f} s / {cc_l:.2f} s; "
          f".so {so_g:,} / {so_l:,} B")
    print(f"    {len(states)} states: {nonzero} non-zero entries, bit-identical {exact} "
          f"({100 * exact / max(nonzero, 1):.0f}%), max |diff|/max|block| {worst_block:.1e}, "
          f"NaN in one route {nan_mismatch}, NaN in both {nan_both}, "
          f"graph non-zero where library exactly zero {lib_zero_graph_not}")
    print(f"    structural zeros: library {zl}, graph {zg}; state of rest: finite graph "
          f"{bool(np.isfinite(a0).all())} library {bool(np.isfinite(b0).all())}, "
          f"elementwise equal {bool(np.array_equal(a0, b0))}")
    print(f"    one call: graph {per['graph'] - per['empty']:.1f} ns, library "
          f"{per['library'] - per['empty']:.1f} ns (empty-kernel ctypes call {per['empty']:.1f} ns subtracted)")
print("work dir:", work)
