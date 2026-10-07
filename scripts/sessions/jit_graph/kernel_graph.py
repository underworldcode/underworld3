"""Prototype of the shared-graph lowering for JIT kernels (tier 2 of #823).

Not library code: the design note it supports is
docs/developer/design/jit-shared-graph-codegen.md.

A non-constant UWexpression atom becomes a NODE: an applied undefined function of the
leaves its value depends on. SymPy's own chain rule then differentiates through it,
because ``fdiff(i)`` returns the partial derivative with respect to argument slot i.

Two kinds of identity are kept apart:

- In Python, a node is identified by its body. Two atoms with equal bodies share a
  node; bodies that use different constants are different even when the constants
  share a display name, because SymPy equality sees the difference. The class name is
  a hash of the body written with DISPLAY identities only (no creation counters), and
  each class carries its graph's serial in its SymPy identity (``_ctx``), so classes
  from two compiles never compare equal.
- In C, a temporary is identified by a hash of the C it computes: its body with every
  leaf written as the C the kernel reads (``petsc_u[3]``, ``constants[2]``) and every
  child as the child's hash. Temporaries are ordered and merged by that hash, so the
  source is a function of the mathematics and the kernel's data layout, not of Python
  names, creation counters or hash seeds.

Partial derivatives are taken with every leaf replaced by an independent real dummy.
A derivative that is zero, a number or a single leaf is returned as itself.
"""
import hashlib
import itertools

import sympy
from sympy.core.function import AppliedUndef, UndefinedFunction
from sympy.vector.scalar import BaseScalar

from underworld3.function.expressions import UWexpression
from underworld3.function._function import UnderworldAppliedFunction
from underworld3.utilities._jitextension import _is_truly_constant

EPS2 = sympy.Float(1.0e-36)
_graph_serial = itertools.count()


def guard(e):
    """The sqrt guard of ``_jacobian_unwrap``: every half-integer power whose base has
    free symbols (constants included) gets ``+1e-36`` in its base. Memoised on identity
    (as tier 1 does); node applications are not entered."""
    memo = {}

    def g(n):
        hit = memo.get(id(n))
        if hit is not None:
            return hit[1]
        out = n
        args = getattr(n, "args", None)
        if args and not isinstance(n, _KernelNode):
            new_args = tuple(g(a) for a in args)
            if any(a is not b for a, b in zip(args, new_args)) and args != new_args:
                out = n.func(*new_args)
            if (out.is_Pow and out.exp.is_Rational and out.exp.q == 2
                    and out.args[0].free_symbols):
                out = sympy.Pow(out.args[0] + EPS2, out.exp)
        memo[id(n)] = (n, out)
        return out

    return g(e)


class _KernelNode(AppliedUndef):
    """A named sub-expression applied to the leaves its value depends on."""

    def fdiff(self, argindex=1):
        return self._graph.slot_derivative(self, argindex - 1)


def body_of(app):
    """A node application's body, with the application's arguments substituted."""
    cls = app.func
    if app.args == cls._deps:
        return cls._body
    return cls._body.xreplace(dict(zip(cls._deps, app.args)))


class KernelGraph:
    """One lowering context: the nodes of one compile."""

    def __init__(self):
        self.serial = next(_graph_serial)
        self._const = {}     # id(atom) -> (atom, bool)
        self._node = {}      # (id(atom), guarded) -> (atom, application)
        self._by_body = {}   # body -> class
        self._deriv = {}     # (class, slot) -> derivative in terms of the class's deps
        self._busy = set()
        self._splitting = False

    # ------------------------------------------------------------------ leaves
    def is_constant(self, atom):
        hit = self._const.get(id(atom))
        if hit is None:
            hit = self._const[id(atom)] = (atom, _is_truly_constant(atom, UWexpression))
        return hit[1]

    def is_leaf(self, s):
        if isinstance(s, _KernelNode):
            return False
        if isinstance(s, (AppliedUndef, BaseScalar)):
            return True
        if isinstance(s, UWexpression):
            return self.is_constant(s)
        return isinstance(s, sympy.Symbol)

    @staticmethod
    def display_key(s):
        """What a leaf is called, without creation counters: stable across processes,
        ranks, preambles and re-declarations. Not unique (two constants may share a
        display name); used only to name and order, never to identify."""
        if isinstance(s, UnderworldAppliedFunction):
            return f"F|{s.func.__name__}"
        if isinstance(s, BaseScalar):
            return f"X|{s._id[1]}|{s._id[0]}"
        if isinstance(s, UWexpression):
            return f"C|{s.name}"
        if isinstance(s, AppliedUndef):
            return f"A|{s.func.__name__}|{','.join(map(str, s.args))}"
        return f"S|{s.name}"

    @staticmethod
    def order_key(s):
        """Display key, ties broken by RELATIVE creation order (as the constants
        manifest orders same-named constants)."""
        return (KernelGraph.display_key(s), getattr(s, "instance_number", 0))

    def leaves(self, e):
        out, seen, stack = set(), set(), [e]
        while stack:
            a = stack.pop()
            if id(a) in seen:
                continue
            seen.add(id(a))
            if isinstance(a, _KernelNode):
                out.update(a.args)
            elif self.is_leaf(a):
                out.add(a)
            elif isinstance(a, sympy.Basic):
                stack.extend(a.args)
        return out

    # ------------------------------------------------------------------ nodes
    def lower(self, e, guarded):
        """``e`` with each non-constant UW atom replaced by its node."""
        if not isinstance(e, sympy.Basic):
            return e
        atoms = sorted((s for s in e.free_symbols
                        if isinstance(s, UWexpression) and not self.is_constant(s)),
                       key=self.order_key)
        m = {s: self.node_of(s, guarded) for s in atoms}
        # a coordinate atom lowers to the base scalar the kernel reads, as the tree does
        m.update({s: s.sym for s in e.free_symbols
                  if isinstance(s, BaseScalar) and type(s).__name__ == "UWCoordinate"})
        return e.xreplace(m) if m else e

    def node_of(self, atom, guarded):
        key = (id(atom), guarded)
        hit = self._node.get(key)
        if hit is not None:
            return hit[1]
        if key in self._busy:
            raise ValueError(f"cyclic expression: {atom} contains itself")
        self._busy.add(key)
        try:
            body = self.lower(atom.sym, guarded)
            if guarded:
                body = guard(body)
            app = self.make_node(body)
        finally:
            self._busy.discard(key)
        self._node[key] = (atom, app)
        return app

    def _display_name(self, body):
        canon = {a: sympy.Symbol(a.func.__name__) for a in body.atoms(_KernelNode)}
        for leaf in self.leaves(body):
            canon.setdefault(leaf, sympy.Symbol(self.display_key(leaf)))
        text = sympy.srepr(body.xreplace(canon))
        return "N" + hashlib.sha1(text.encode()).hexdigest()[:12]

    def make_node(self, body):
        """The node for ``body``: one per distinct body. A body that is a number, a leaf
        or a single node needs no temporary and is returned as itself."""
        body = sympy.sympify(body)
        if not self._splitting:
            body = self._split_shared(body)
        if body.is_Atom or isinstance(body, (AppliedUndef, BaseScalar)):
            return body
        cls = self._by_body.get(body)
        if cls is None:
            deps = tuple(sorted(self.leaves(body), key=self.order_key))
            cls = UndefinedFunction(self._display_name(body), bases=(_KernelNode,),
                                    real=True, _ctx=self.serial, _n=len(self._by_body),
                                    __dict__={"_graph": self})
            cls._body, cls._deps = body, deps
            self._by_body[body] = cls
        return cls(*cls._deps)

    def _split_shared(self, body):
        """Repeated unnamed sub-expressions of one body become anonymous nodes."""
        if body.is_Atom:
            return body
        repl, (reduced,) = sympy.cse([body], symbols=sympy.numbered_symbols("_cse", real=True),
                                     order="none")
        if not repl:
            return body
        self._splitting = True
        try:
            m = {}
            for sym, e in repl:
                m[sym] = self.make_node(e.xreplace(m))
            return reduced.xreplace(m)
        finally:
            self._splitting = False

    # ------------------------------------------------------------------ derivatives
    def slot_derivative(self, app, i):
        cls = app.func
        key = (cls, i)
        if key not in self._deriv:
            deps = cls._deps
            dummies = [sympy.Dummy(real=True) for _ in deps]
            b = cls._body.xreplace(dict(zip(deps, dummies)))
            d = sympy.diff(b, dummies[i]).xreplace(dict(zip(dummies, deps)))
            self._deriv[key] = self.make_node(d)
        d = self._deriv[key]
        if app.args == cls._deps:
            return d
        return d.xreplace(dict(zip(cls._deps, app.args)))


# ---------------------------------------------------------------------- emission
def emission_order(outputs, spell):
    """The temporaries a kernel needs, in the order to write them.

    ``spell(leaf)`` is the C the kernel reads for a leaf. Each node application gets a
    canonical key: a hash of its body with leaves spelled as C and children replaced by
    their own keys. Applications with equal keys compute the same C and share one
    temporary; temporaries follow the ones they use, ties broken by key.

    Returns ``(order, key)``: ``order`` is a list of ``(key, body)``; ``key`` maps each
    node application reached to its key.
    """
    key = {}

    def key_of(app):
        hit = key.get(app)
        if hit is not None:
            return hit
        body = body_of(app)
        sub = {c: sympy.Symbol("@" + key_of(c)) for c in body.atoms(_KernelNode)}
        for leaf in body.free_symbols | body.atoms(AppliedUndef):
            if not isinstance(leaf, _KernelNode) and leaf not in sub:
                sub[leaf] = sympy.Symbol(spell(leaf))
        k = hashlib.sha1(sympy.srepr(body.xreplace(sub)).encode()).hexdigest()[:16]
        key[app] = k
        return k

    order, placed = [], set()

    def visit(app):
        k = key_of(app)
        if k in placed:
            return
        placed.add(k)
        body = body_of(app)
        for child in sorted(body.atoms(_KernelNode), key=key_of):
            visit(child)
        order.append((k, body))

    for out in outputs:
        for child in sorted(sympy.sympify(out).atoms(_KernelNode), key=key_of):
            visit(child)
    return order, key
