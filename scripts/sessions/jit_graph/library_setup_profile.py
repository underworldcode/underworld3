"""The library's whole Newton setup on one fixture: wall time by phase, an identity hash
of the generated C, and optionally a cProfile with the hotspots grouped.

The identity hash is the md5 of the generated header with the module name and symbol
prefix canonicalised, as ``getext`` canonicalises them before it hashes; it is
independent of the Underworld version string, so it compares two builds directly.
"""
import cProfile
import hashlib
import io
import os
import pstats
import sys
import time

import underworld3 as uw
import underworld3.utilities._jitextension as jx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixtures   # noqa: E402

params = uw.Params(
    uw_fixture=uw.Param("notch", description="box | powerlaw | linear | vep | ti | notch"),
    uw_profile=uw.Param(0, description="1 = cProfile the setup and print the hotspots"),
    uw_dump=uw.Param("", description="write the canonicalised header to this path"),
)
fixture = str(params.uw_fixture)
stokes, _ = fixtures.build(fixture)

seen = {}
generate, compile_ = jx.generate_c_source, jx.compile_and_load


def timed_generate(*a, **k):
    t = time.perf_counter()
    modname, codeguys, diag = generate(*a, **k)
    seen["generate"] = time.perf_counter() - t
    header = dict(codeguys)["cy_ext.h"]
    header = header.replace(modname, "__MOD__").replace(diag["randstr"], "__RS__")
    seen["header md5"] = hashlib.md5(header.encode()).hexdigest()[:10]
    seen["header bytes"] = len(header)
    if str(params.uw_dump):
        with open(os.path.expanduser(str(params.uw_dump)), "w") as fh:
            fh.write(header)
    return modname, codeguys, diag


def timed_compile(*a, **k):
    t = time.perf_counter()
    out = compile_(*a, **k)
    seen["compile"] = time.perf_counter() - t
    return out


jx.generate_c_source, jx.compile_and_load = timed_generate, timed_compile

profiler = cProfile.Profile() if int(params.uw_profile) else None
t = time.perf_counter()
if profiler:
    profiler.enable()
stokes._setup_pointwise_functions()
if profiler:
    profiler.disable()
wall = time.perf_counter() - t

print(f"[{fixture}] Newton setup {wall:.1f} s{' (under cProfile)' if profiler else ''}; "
      f"C generation {seen.get('generate', float('nan')):.1f} s; compile {seen.get('compile', float('nan')):.1f} s; "
      f"header {seen.get('header bytes', 0):,} B, md5 {seen.get('header md5')}")

if profiler:
    out = os.path.expanduser(f"~/+Simulations/jit_graph/{fixture}_setup.prof")
    profiler.dump_stats(out)
    st = pstats.Stats(profiler)
    groups = {
        "C printing (CodePrinter.doprint)": ("sympy/printing/codeprinter.py", "doprint"),
        "  term ordering (printer._as_ordered_terms)": ("sympy/printing/printer.py", "_as_ordered_terms"),
        "  _handle_UnevaluatedExpr": ("sympy/printing/codeprinter.py", "_handle_UnevaluatedExpr"),
        "differentiation (_dispatch_eval_derivative_n_times)": ("sympy/core/function.py", "_dispatch_eval_derivative_n_times"),
        "replace": ("sympy/core/basic.py", "replace"),
        "xreplace": ("sympy/core/basic.py", "xreplace"),
        "subs": ("sympy/core/basic.py", "subs"),
        "as_real_imag (Pow)": ("sympy/core/power.py", "as_real_imag"),
        "_jacobian_unwrap": ("underworld3/cython/petsc_generic_snes_solvers", "_jacobian_unwrap"),
        "_extract_constants (tier 1 build)": ("underworld3/utilities/_jitextension.py", "_extract_constants"),
        "lower_callbacks (graph build)": ("underworld3/utilities/_jit_graph.py", "lower_callbacks"),
        "emit (graph build)": ("underworld3/utilities/_jit_graph.py", "emit"),
        "compile_and_load": ("underworld3/utilities/_jitextension.py", "compile_and_load"),
    }
    for label, (path, func) in groups.items():
        total = sum(v[3] for k, v in st.stats.items() if path in k[0] and k[2] == func)
        print(f"  {label:52s} {total:7.2f} s")
    print(f"  profile: {out}")
