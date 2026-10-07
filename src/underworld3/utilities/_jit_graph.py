r"""JIT kernels lowered onto the shared expression graph (#823, tier 2).

Design: ``docs/developer/design/jit-shared-graph-codegen.md``.

A constitutive law is a graph of named sub-expressions (``UWexpression`` atoms). The
expanded-tree route copies every atom into every place that refers to it before it
differentiates and prints. This module keeps the graph instead:

- each non-constant atom becomes a NODE, an applied undefined function of the leaves
  its value depends on (field values and gradients, coordinates, constant atoms), whose
  body is the atom's content with each child atom replaced by the child's node;
- SymPy's own chain rule differentiates through a node, because ``fdiff(i)`` returns the
  partial derivative of the body with respect to argument slot ``i``, itself a node;
- the emitter writes one C temporary per distinct computation, ordered and merged by a
  hash of the C it computes, so the generated source is a function of the mathematics
  and of the kernel's data layout alone.

Selected by the private development switch ``UW_JIT_GRAPH=1`` while both routes exist.
"""
import hashlib
import itertools
import os

import sympy
from sympy.core.function import AppliedUndef, UndefinedFunction
from sympy.tensor.array import NDimArray
from sympy.vector.scalar import BaseScalar

_EPS2 = sympy.Float(1.0e-36)
_serial = itertools.count(1)
_nodes_made = False


def enabled():
    """Whether the graph route is selected (``UW_JIT_GRAPH=1``)."""
    return os.environ.get("UW_JIT_GRAPH", "").lower() in ("1", "true", "yes")


def nodes_exist():
    """Whether any node has been made in this process; when not, no expression can
    hold one and the unwrappers skip the search."""
    return _nodes_made


def guard_half_integer_powers(e):
    r"""``e`` with :math:`10^{-36}` added to the base of every half-integer power
    whose base has free symbols: the sqrt guard of ``_jacobian_unwrap``, the same rule
    and the same rebuild. Node applications are not entered (a guarded node's body was
    guarded when it was made)."""
    memo = {}

    def guard(n):
        hit = memo.get(id(n))
        if hit is not None:
            return hit[1]
        out = n
        args = getattr(n, "args", None)
        if args and not isinstance(n, _KernelNode):
            new_args = tuple(guard(a) for a in args)
            if any(a is not b for a, b in zip(args, new_args)) and args != new_args:
                out = n.func(*new_args)
                # replace(simultaneous=True): a rebuild that collapses to one of the
                # changed arguments is not matched again
                if any(out == a and a != b for a, b in zip(args, new_args)):
                    memo[id(n)] = (n, out)
                    return out
            if (out.is_Pow and out.exp.is_Rational and out.exp.q == 2
                    and out.args[0].free_symbols):
                out = sympy.Pow(out.args[0] + _EPS2, out.exp)
        memo[id(n)] = (n, out)
        return out

    return guard(e)


class _KernelNode(AppliedUndef):
    """A named quantity of a kernel, applied to the leaves its value depends on."""

    def fdiff(self, argindex=1):
        return self._graph.slot_derivative(self, argindex - 1)

    def _ccode(self, printer):
        raise RuntimeError(
            f"JIT: node {self.func.__name__} reached the C printer; every node must be "
            f"emitted as a temporary (a defect in the graph lowering, #823)")


class _Temporary(sympy.Symbol):
    """A C temporary of one kernel, written ``uwt_<i>``."""

    __slots__ = ("_ccodestr",)

    def __new__(cls, index):
        obj = sympy.Symbol.__xnew__(cls, f"uwt_{index}", real=True)
        obj._ccodestr = f"uwt_{index}"
        return obj

    def _ccode(self, printer):
        return self._ccodestr


def body_of(app):
    """A node application's body, with the application's arguments substituted."""
    cls = app.func
    if app.args == cls._deps:
        return cls._body
    return cls._body.xreplace(dict(zip(cls._deps, app.args)))


def _holds_node(expr):
    from underworld3.utilities._jitextension import _holds_instance
    return _nodes_made and _holds_instance(expr, _KernelNode)


def expand_nodes(expr):
    """``expr`` (an expression, Matrix or Array) with every node application replaced
    by its body, recursively: the expanded tree of the expression. For code outside the
    JIT that evaluates a lowered block (a test's oracle, ``uw.function.evaluate``)."""
    if not _holds_node(expr):
        return expr
    memo = {}

    def expand(e):
        apps = e.atoms(_KernelNode)
        if not apps:
            return e
        rule = {}
        for app in apps:
            hit = memo.get(app)
            if hit is None:
                hit = memo[app] = expand(body_of(app))
            rule[app] = hit
        return e.xreplace(rule)

    if isinstance(expr, sympy.MatrixBase):
        return expr.applyfunc(expand)
    if isinstance(expr, NDimArray):
        return type(expr)([expand(e) for e in expr], expr.shape)
    return expand(expr)


class KernelGraph:
    """One lowering context: the nodes made for one compile, or for one Jacobian
    source. Each node class holds its body and this context, so a node is read
    back without the context being passed."""

    def __init__(self):
        self.serial = next(_serial)
        self._const = {}     # id(atom) -> (atom, bool)
        self._node = {}      # (id(atom), guarded) -> (atom, replacement)
        self._by_body = {}   # body -> node class
        self._deriv = {}     # (node class, slot) -> derivative in the class's deps
        self._busy = set()
        self._splitting = False

    # ------------------------------------------------------------------ leaves
    def is_constant(self, atom):
        hit = self._const.get(id(atom))
        if hit is None:
            from underworld3.function.expressions import UWexpression
            from underworld3.utilities._jitextension import _is_truly_constant
            hit = self._const[id(atom)] = (atom, _is_truly_constant(atom, UWexpression))
        return hit[1]

    def is_leaf(self, s):
        from underworld3.function.expressions import UWexpression

        if isinstance(s, _KernelNode):
            return False
        if isinstance(s, (AppliedUndef, BaseScalar)):
            return True
        if isinstance(s, UWexpression):
            return self.is_constant(s)
        return isinstance(s, sympy.Symbol)

    @staticmethod
    def display_key(s):
        """What a leaf is called, without creation counters. Not unique (two
        constants may share a name); used to name and order, never to identify."""
        from underworld3.function.expressions import UWexpression

        if isinstance(s, AppliedUndef):
            return f"F|{s.func.__name__}|{','.join(map(str, s.args))}"
        if isinstance(s, BaseScalar):
            return f"X|{s._id[1]}|{s._id[0]}"
        if isinstance(s, UWexpression):
            return f"C|{s.name}"
        return f"S|{getattr(s, 'name', s)}"

    @staticmethod
    def order_key(s):
        """Display key, ties broken by relative creation order (as the constants
        manifest breaks them)."""
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

    # ------------------------------------------------------------------ lowering
    def lower(self, e, guarded=False):
        """``e`` (an expression, Matrix or Array) with each UW atom lowered: a
        non-constant ``UWexpression`` to its node, a coordinate to its base scalar,
        a bare quantity to its non-dimensional value. Constant atoms stay as they
        are; they are the ``constants[]`` leaves. ``guarded`` selects the node
        variant with the sqrt guard in every body (a Newton source)."""
        from underworld3.coordinates import UWCoordinate
        from underworld3.function.expressions import UWexpression
        from underworld3.function.quantities import UWQuantity

        if isinstance(e, sympy.MatrixBase):
            return e.applyfunc(lambda x: self.lower(x, guarded))
        if isinstance(e, NDimArray):
            return type(e)([self.lower(x, guarded) for x in e], e.shape)
        if isinstance(e, (UWexpression, UWCoordinate)) or (
                isinstance(e, UWQuantity) and not isinstance(e, UWexpression)):
            return self._replacement(e, guarded)
        if not isinstance(e, sympy.Basic):
            return sympy.sympify(e)
        uw_types = (UWexpression, UWQuantity, UWCoordinate)
        atoms = sorted((s for s in e.free_symbols if isinstance(s, uw_types)),
                       key=self.order_key)
        rule = {}
        for s in atoms:
            r = self._replacement(s, guarded)
            if r is not s:
                rule[s] = r
        return e.xreplace(rule) if rule else e

    def _replacement(self, atom, guarded):
        from underworld3.coordinates import UWCoordinate
        from underworld3.function.expressions import UWexpression, _unwrap_atom

        if isinstance(atom, UWCoordinate):
            return atom.sym
        if not isinstance(atom, UWexpression):        # a bare quantity
            return sympy.sympify(_unwrap_atom(atom, "nondimensional"))
        if self.is_constant(atom):
            return atom
        return self.node_of(atom, guarded)

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
                body = guard_half_integer_powers(body)
            # a matrix-valued atom cannot be one scalar temporary: it is expanded
            # in place, as the tree route expands it
            out = self.make_node(body) if isinstance(body, sympy.Expr) else body
        finally:
            self._busy.discard(key)
        self._node[key] = (atom, out)
        return out

    def _display_name(self, body):
        canon = {a: sympy.Symbol(a.func.__name__) for a in body.atoms(_KernelNode)}
        for leaf in self.leaves(body):
            canon.setdefault(leaf, sympy.Symbol(self.display_key(leaf)))
        text = sympy.srepr(body.xreplace(canon))
        return "N" + hashlib.sha1(text.encode()).hexdigest()[:12]

    def make_node(self, body):
        """The node for ``body``, one per distinct body. A body that is a number, a
        single leaf, a single node, or reads no leaf is returned as itself, and so is a
        condition (a node is a value; a Piecewise refuses one as its condition)."""
        global _nodes_made

        body = sympy.sympify(body)
        if not isinstance(body, sympy.Expr):
            return body
        if not self._splitting:
            body = self._split_shared(body)
        if body.is_Atom or isinstance(body, (AppliedUndef, BaseScalar)):
            return body
        cls = self._by_body.get(body)
        if cls is None:
            deps = tuple(sorted(self.leaves(body), key=self.order_key))
            if not deps:
                return body
            cls = UndefinedFunction(self._display_name(body), bases=(_KernelNode,),
                                    real=True, _ctx=self.serial, _n=len(self._by_body),
                                    __dict__={"_graph": self})
            cls._body, cls._deps = body, deps
            self._by_body[body] = cls
            _nodes_made = True
        return cls(*cls._deps)

    def _split_shared(self, body):
        """Repeated unnamed sub-expressions of one body become anonymous nodes.
        Canonical order, so the split does not depend on the hash seed."""
        if body.is_Atom:
            return body
        repl, (reduced,) = sympy.cse([body], symbols=sympy.numbered_symbols("_cse", real=True))
        if not repl:
            return body
        self._splitting = True
        try:
            rule = {}
            for sym, e in repl:
                rule[sym] = self.make_node(e.xreplace(rule))
            return reduced.xreplace(rule)
        finally:
            self._splitting = False

    # ------------------------------------------------------------------ derivatives
    def slot_derivative(self, app, i):
        """The partial derivative of ``app`` with respect to its argument slot
        ``i``, as a node of the same leaves. Taken with every leaf replaced by an
        independent real dummy, so a field that is itself a function of the
        coordinates is not differentiated a second time through them."""
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


def lower_callbacks(fns, mesh):
    """Each callback lowered in one context and shaped as ``generate_c_source``
    prints it: a Matrix whose entries hold nodes, constant atoms and leaves."""
    from underworld3.function.expressions import UWDerivativeExpression

    graph = KernelGraph()
    out = []
    for fn in fns:
        if isinstance(fn, UWDerivativeExpression):
            fn = fn.doit()
        fn = graph.lower(fn)
        if isinstance(fn, sympy.vector.Vector):
            fn = fn.to_matrix(mesh.N)[0:mesh.dim, 0]
        elif isinstance(fn, sympy.vector.Dyadic):
            fn = fn.to_matrix(mesh.N)[0:mesh.dim, 0:mesh.dim]
        else:
            fn = sympy.Matrix([fn])
        out.append(fn)
    return out


def constant_leaves(lowered):
    """The constant atoms the lowered callbacks read. A node application's arguments
    are every leaf its body reads, child nodes included, so the free symbols of the
    outputs reach every constant of every temporary."""
    from underworld3.function.expressions import UWexpression

    found = set()
    for fn in lowered:
        found.update(s for s in fn.free_symbols if isinstance(s, UWexpression))
    return found


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
    spelt = {}

    def spelling(leaf):
        hit = spelt.get(leaf)
        if hit is None:
            hit = spelt[leaf] = sympy.Symbol(spell(leaf))
        return hit

    def key_of(app):
        hit = key.get(app)
        if hit is not None:
            return hit
        body = body_of(app)
        sub = {c: sympy.Symbol("@" + key_of(c)) for c in body.atoms(_KernelNode)}
        for leaf in body.free_symbols | body.atoms(AppliedUndef):
            if not isinstance(leaf, _KernelNode) and leaf not in sub:
                sub[leaf] = spelling(leaf)
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


def emit(fn, spell, constants_rule):
    """``fn`` (a lowered Matrix) as temporaries and outputs.

    Returns ``(temporaries, outputs)``: ``temporaries`` is a list of
    ``(_Temporary, body)`` in the order to write them, and ``outputs`` is ``fn`` with
    each node application replaced by its temporary. Constant atoms are replaced by
    ``constants_rule`` (their ``constants[]`` placeholders) in both.
    """
    order, key = emission_order(list(fn), spell)
    temp_of, temporaries = {}, []
    for i, (k, body) in enumerate(order):
        t = _Temporary(i)
        rule = {c: temp_of[key[c]] for c in body.atoms(_KernelNode)}
        rule.update(constants_rule)
        temporaries.append((t, body.xreplace(rule)))
        temp_of[k] = t
    rule = {app: temp_of[key[app]] for app in fn.atoms(_KernelNode)}
    rule.update(constants_rule)
    return temporaries, fn.xreplace(rule)
