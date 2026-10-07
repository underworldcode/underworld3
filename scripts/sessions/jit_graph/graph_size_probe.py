"""Size of a kernel's shared graph of named sub-expressions against the tree the
current JIT expands it into (#823). Builds the campaign notch model, does NOT solve,
and never expands: the expanded size is counted by dynamic programming over the graph.
"""
import os, sys, time
import sympy
import underworld3 as uw
from underworld3.function.expressions import UWexpression, _unwrap_atom
from underworld3.utilities._jitextension import _is_truly_constant

params = uw.Params(
    uw_mesh=uw.Param(os.path.expanduser("~/+Simulations/spiegelman_hardcase/meshes/notch_mesh1.msh")),
)
sys.path.insert(0, os.path.expanduser("~/+Simulations/spiegelman_hardcase/drivers"))
import notch_model as nm

S = nm.build(str(params.uw_mesh), 1.0e24, 2.5, 30.0, 1, xi=0.0, seed=False, p_degree=0,
             p_continuous=False, floor_scalar=1.0e-3, unique_params=True)
st, cm = S["stokes"], S["cm"]

constant = {}
def is_const(a):
    if id(a) not in constant:
        constant[id(a)] = (a, _is_truly_constant(a, UWexpression))
    return constant[id(a)][1]

named, consts = {}, {}
def body(a):
    return _unwrap_atom(a, "symbolic")

tree_size, seen_nodes = {}, set()
def size(e):
    """Expanded-tree node count of e, with non-constant UW atoms expanded."""
    k = id(e)
    if k in tree_size:
        return tree_size[k][1]
    seen_nodes.add(k)
    if isinstance(e, UWexpression):
        if is_const(e):
            consts[id(e)] = e
            n = 1
        else:
            named[id(e)] = e
            n = size(body(e))
    elif isinstance(e, (sympy.MatrixBase, sympy.NDimArray)):
        n = sum(size(x) for x in e)
    elif isinstance(e, sympy.Basic) and e.args:
        n = 1 + sum(size(a) for a in e.args)
    else:
        n = 1
    tree_size[k] = (e, n)
    return n

t0 = time.perf_counter()
F1 = st.F1.sym
eta = cm.viscosity
n_eta, n_F1 = size(eta), size(F1)
graph_nodes = len(seen_nodes)
print(f"[probe] {time.perf_counter()-t0:.2f} s")
print(f"named non-constant sub-expressions : {len(named)}")
print(f"constants[] atoms reached          : {len(consts)}")
print(f"distinct sympy nodes in the graph  : {graph_nodes}")
print(f"expanded tree, viscosity           : {n_eta}")
print(f"expanded tree, F1 (all entries)    : {n_F1}")
for e in sorted(named.values(), key=lambda a: -tree_size[id(a)][1])[:12]:
    print(f"   {tree_size[id(e)][1]:>8d}  {e.name}")
