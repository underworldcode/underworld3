"""Reverse-mode over a recorded run.

A transcript with snapshots is the forward tape: the ordered operators per
step, and the state each step started from. Every operator's linearisation
comes from the residual itself — the residual is SymPy, so the coupling of
one solve to the fields it reads is a symbolic derivative, and the adjoint of
one solve is a transpose against the Jacobian the SNES assembles
(``solver.adjoint_solve``). This module chains those backwards.

The chain rule, per solve, in reverse order of the record. A solve
:math:`R(u; f_1, f_2, \\dots, m) = 0` reads fields :math:`f_i` and parameters
:math:`m`. Given the accumulated dual :math:`\\bar u = \\partial J/\\partial u`
on its unknown,

.. math::

    K^T \\mu = -\\bar u, \\qquad
    \\bar f_i \\mathrel{+}= (\\partial R/\\partial f_i)^T \\mu, \\qquad
    \\bar m \\mathrel{+}= \\mu^T \\partial R/\\partial m.

A history slot read by the solve (``psi_star[0]``) is the tracked field at
the step's input, so its dual is the dual on that field at the previous
level — which is where the walk goes next.

Each solve is linearised at ITS OWN input state: the step's snapshot is
restored, the solves before it in the step are replayed, the histories it
reads are captured, it is replayed, and the captured inputs are put back in
the history slots the post-solve hook shifted. That is what
``docs/examples/adjoint`` did by hand for one solver; here the record says
which operators ran and the residuals say what they read.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, Optional

import numpy as np
import sympy

import underworld3 as uw
from underworld3.cython.generic_solvers import SNES_Scalar as _SNES_Scalar
from underworld3.cython.generic_solvers import SNES_Vector as _SNES_Vector
from underworld3.utilities._api_tools import Template


class _ScalarLoad(_SNES_Scalar):
    r"""A generic scalar solver used as an ASSEMBLER: its residual at zero is
    :math:`\int g_0\,\phi_j + \mathbf g_1\cdot\nabla\phi_j`, which is the
    dual of a field read through its value (``g0``) and its gradient (``g1``)
    in one assembly — no integration by parts, no boundary term to get wrong.
    """

    _solver_terms = (("_g0", "value part of the load"), ("_g1", "gradient part"))
    F0 = Template(r"g_0", lambda self: sympy.Matrix([[self._g0]]),
                  "value part of a dual load")
    F1 = Template(r"\mathbf{g}_1", lambda self: self._g1,
                  "gradient part of a dual load (1 x cdim)")


class _VectorLoad(_SNES_Vector):
    """The vector-field counterpart of :class:`_ScalarLoad`: ``g0`` a row of
    ``dim`` components, ``g1`` a ``dim x cdim`` matrix."""

    _solver_terms = (("_g0", "value part of the load"), ("_g1", "gradient part"))
    F0 = Template(r"\mathbf{g}_0", lambda self: self._g0, "value part of a dual load")
    F1 = Template(r"\mathbf{G}_1", lambda self: self._g1,
                  "gradient part of a dual load (dim x cdim)")


class _Scratch:
    """Fields and assemblers on a space, reused rather than re-created.

    A fresh MeshVariable per dual leaked sixteen registered variables per
    ``gradient()`` call and slowed the sixth call tenfold (found in review).
    Variables are handed out and taken back; an assembler is built once per
    space and re-pointed at each load.
    """

    def __init__(self):
        self.free = {}
        self.assemblers = {}

    @staticmethod
    def space(variable):
        return (variable.mesh, getattr(variable, "num_components", 1),
                str(variable.vtype), int(variable.degree),
                bool(getattr(variable, "continuous", True)))

    def take(self, like):
        key = self.space(like)
        pool = self.free.setdefault(key, [])
        if pool:
            var = pool.pop()
            var.array[...] = 0.0
            return var
        mesh, n, _, degree, continuous = key
        return uw.discretisation.MeshVariable(
            f"_adj_scratch_{_counter()}", mesh, num_components=n,
            vtype=like.vtype, degree=degree, continuous=continuous)

    def give(self, var):
        self.free.setdefault(self.space(var), []).append(var)

    def assembler(self, like):
        key = self.space(like)
        asm = self.assemblers.get(key)
        if asm is None:
            target = self.take(like)
            mesh, n = key[0], key[1]
            asm = (_ScalarLoad(mesh, u_Field=target) if n == 1
                   else _VectorLoad(mesh, u_Field=target))
            # Every solver family's setup reads constitutive_model._solver_is_setup
            # unguarded, so the assembler carries an inert model: its F0/F1
            # templates override the model's flux entirely.
            if n == 1:
                asm.constitutive_model = uw.constitutive_models.DiffusionModel
                asm.constitutive_model.Parameters.diffusivity = 1.0
            else:
                asm.constitutive_model = uw.constitutive_models.ViscousFlowModel
                asm.constitutive_model.Parameters.shear_viscosity_0 = 1.0
            asm.consistent_jacobian = False      # a residual evaluation only
            asm.petsc_options.delValue("ksp_monitor")
            self.assemblers[key] = asm
        return asm


_shared_scratch = _Scratch()


def dual_on(variable, value, grad=None, scratch=None, boundary=None):
    r"""The dual of a load on ``variable``'s space, held as a field.

    :math:`b_j = \int v\,\phi_j + \mathbf g\cdot\nabla\phi_j` for every basis
    function of ``variable``: ``value`` is the part read through the field's
    value, ``grad`` (optional; ``1 x cdim`` for a scalar field, ``dim x cdim``
    for a vector one) the part read through its gradient — a Crank–Nicolson
    step reads the old level's flux this way. Assembled as the residual of a
    generic solver at zero, so it is exactly the FEM load, with no linear
    solve and no integration by parts. The returned field comes from
    ``scratch`` (a :class:`_Scratch` pool; the module's shared one by
    default) — give it back with ``scratch.give(field)`` when done.

    With ``boundary`` (a mesh boundary label), the load is the facet
    integral :math:`b_j = \int_\Gamma v\,\phi_j` instead — a misfit on
    surface observations — assembled as a natural condition of the same
    generic solver with zero volume templates. A gradient part on a boundary
    is not assembled yet and raises.
    """
    scratch = _shared_scratch if scratch is None else scratch
    mesh = variable.mesh
    n = getattr(variable, "num_components", 1)
    dim, cdim = mesh.dim, mesh.cdim
    asm = scratch.assembler(variable)
    if boundary is not None:
        has_grad = grad is not None and not sympy.Matrix(grad).is_zero_matrix
        if has_grad and n == 1:
            raise NotImplementedError(
                "dual_on: a scalar load read through the gradient on a boundary "
                "is not assembled yet (the scalar assembler has no facet flux "
                "term); a vector one is")
        # zero volume templates; the natural condition IS the load
        asm._g0 = sympy.zeros(1, 1) if n == 1 else sympy.zeros(1, dim)
        asm._g1 = sympy.zeros(1, cdim) if n == 1 else sympy.zeros(dim, cdim)
        asm.natural_bcs.clear()
        asm.add_natural_bc(value, boundary)
        if has_grad:
            # the facet flux part int_Gamma g . grad(phi_j): the same slot a
            # Nitsche condition uses for its symmetry term
            bc = asm.natural_bcs[-1]
            asm.natural_bcs[-1] = bc._replace(fn_F=sympy.Matrix(grad).as_immutable())
        asm.is_setup = False                        # the facet kernels are registered on build
        asm._build(False, False, None)
    else:
        if grad is None:
            grad = sympy.zeros(1, cdim) if n == 1 else sympy.zeros(dim, cdim)
        if asm.natural_bcs:
            asm.natural_bcs.clear()
            asm.is_setup = False
        asm._g0 = value
        asm._g1 = sympy.Matrix(grad)
        asm._needs_function_rewire = True          # the templates re-evaluate
        asm._build(False, False, None)
    out_var = scratch.take(variable)
    gvec = asm.dm.getGlobalVec()
    gvec.set(0.0)
    mesh.update_lvec()
    asm.dm.setAuxiliaryVec(mesh.lvec, None)
    F = gvec.duplicate()
    asm.snes.computeFunction(gvec, F)
    lvec = asm.dm.getLocalVec()
    lvec.set(0.0)
    asm.dm.globalToLocal(F, lvec)
    out_var.vec.array[:] = lvec.array[:]
    asm.dm.restoreLocalVec(lvec)
    F.destroy()
    asm.dm.restoreGlobalVec(gvec)
    mesh._stale_lvec = True
    try:
        out_var._sync_lvec_to_gvec()
    except AttributeError:
        pass
    return out_var


def inner(variable, a, b):
    r"""``a . b`` over the OWNED degrees of freedom of ``variable``'s space,
    reduced across ranks.

    A dual is a covector on the basis, so ``dJ = sum_j dual_j * delta_j``
    is the right pairing — but ``.array`` on a rank holds ghost nodes as
    well, so a plain NumPy dot counts shared nodes twice (found in review:
    a different number on each rank, neither the finite difference). This
    routes both through the field's global vector, which holds each degree
    of freedom once.
    """
    mesh = variable.mesh
    dm = mesh.dm
    field = variable.field_id
    _is, subdm = dm.createSubDM(field)
    ga = subdm.getGlobalVec()
    gb = subdm.getGlobalVec()
    la = subdm.getLocalVec()
    lb = subdm.getLocalVec()
    la.array[:] = np.asarray(a).ravel()
    lb.array[:] = np.asarray(b).ravel()
    subdm.localToGlobal(la, ga)
    subdm.localToGlobal(lb, gb)
    value = float(ga.dot(gb))
    subdm.restoreLocalVec(la); subdm.restoreLocalVec(lb)
    subdm.restoreGlobalVec(ga); subdm.restoreGlobalVec(gb)
    return value


def _token_of(var):
    """How ``var`` prints inside an expression: everything before the
    coordinate arguments and, for a vector, before the component index."""
    text = str(_symbols_of(var)[0]).rsplit("(", 1)[0]
    if getattr(var, "num_components", 1) > 1:
        text = text.rsplit("_{", 1)[0]
    return text


def misfit_duals(misfit, variables, scratch=None, boundary=None):
    r"""``dJ/df`` as a dual field on each field the misfit reads.

    ``J = \int misfit`` over the mesh — or over the boundary ``boundary``
    when one is named, for a misfit on surface observations. A field enters
    through its value and, for a misfit on a stress or a strain rate,
    through its gradient. Both parts are differentiated symbolically and
    assembled as one load (:func:`dual_on`), so a misfit written in terms of
    :math:`\nabla u` needs no integration by parts by the caller. Returns
    ``{variable: dual}`` for the variables that appear; give each dual back
    to the scratch pool when done.
    """
    scratch = _shared_scratch if scratch is None else scratch
    peeled = _peel(misfit)
    text = str(peeled)
    atoms = set(peeled.atoms(sympy.Function))
    out = {}
    for var in variables:
        token = _token_of(var)
        if token not in text:
            continue
        symbols = _symbols_of(var)
        value = [sympy.diff(peeled, s) for s in symbols]
        pattern = re.compile(re.escape(token) + r"_\{ ?(\d*),(\d+)\}\(")
        g1 = None
        for atom in atoms:
            m = pattern.match(str(atom))
            if not m:
                continue
            if g1 is None:
                g1 = sympy.zeros(len(symbols), var.mesh.cdim)
            i = int(m.group(1)) if m.group(1) else 0
            g1[i, int(m.group(2))] = sympy.diff(peeled, atom)
        if all(v == 0 for v in value) and (g1 is None or g1.is_zero_matrix):
            continue
        out[var] = dual_on(var, _as_expression(value), g1, scratch, boundary=boundary)
    return out


def integral(mesh, expression, boundary=None):
    """``float(∫ expression)`` over the mesh, or over ``boundary`` if named."""
    if boundary is None:
        return float(uw.maths.Integral(mesh, expression).evaluate())
    return float(uw.maths.BdIntegral(mesh, expression, boundary).evaluate())


def _reads_of(solver, unknown, tokens):
    """What a solver's residual reads, other than its unknown.

    ``(variable, value symbols, {(component, direction): derivative atom})``
    per variable. A component prints as ``{v}_{ 0 }``; a derivative carries
    a comma — ``{v}_{ 0,1}`` for a vector, ``{T}_{,1}`` for a scalar — and
    is read through the gradient part of the load.
    """
    f0 = _peel(solver.F0.sym)
    f1 = _peel(solver.F1.sym)
    text = str(f0) + str(f1)
    atoms = set(f0.atoms(sympy.Function)) | set(f1.atoms(sympy.Function))
    found = []
    for token, var in tokens.items():
        if var is unknown or token not in text:
            continue
        derivatives = {}
        pattern = re.compile(re.escape(token) + r"_\{ ?(\d*),(\d+)\}\(")
        for atom in atoms:
            m = pattern.match(str(atom))
            if m:
                i = int(m.group(1)) if m.group(1) else 0
                derivatives[(i, int(m.group(2)))] = atom
        found.append((var, _symbols_of(var), derivatives))
    return found


def _tokens_of(variables):
    return {_token_of(var): var for var in variables if hasattr(var, "sym")}


def field_duals(solver, mu, variables, scratch=None):
    r"""``(\partial R/\partial f)^T \mu`` as a dual on each field ``f`` the
    solver's residual reads, among ``variables``.

    A field read through its value gives the value part of the load; one
    read through its gradient (a Crank–Nicolson step reads the old flux)
    gives the gradient part. Both come from :meth:`adjoint_integrand`, the
    symbolic derivative of the residual, and are assembled as one load.
    """
    scratch = _shared_scratch if scratch is None else scratch
    out = {}
    for var, symbols, derivatives in _reads_of(solver, solver.u, _tokens_of(variables)):
        value = [solver.adjoint_integrand(mu, s) for s in symbols]
        g1 = None
        if derivatives:
            cdim = var.mesh.cdim
            g1 = sympy.zeros(len(symbols), cdim)
            for (i, k), atom in derivatives.items():
                g1[i, k] = solver.adjoint_integrand(mu, atom)
        out[var] = dual_on(var, _as_expression(value), g1, scratch)
    return out


def gradient(solver, misfit, parameters=(), fields=(), scratch=None, boundary=None):
    r"""``dJ/dm`` and the duals on fields, by the adjoint of ONE solve.

    For :math:`J = \int` ``misfit`` over the mesh, evaluated in the state the
    solver ended in: the dual of :math:`J` on the unknown is assembled
    (:func:`misfit_duals`), the adjoint system :math:`K^T\mu = -\partial J/
    \partial u` is solved (:meth:`adjoint_solve`), and each parameter gets
    :math:`\partial J/\partial m + \mu^T\partial R/\partial m`
    (:meth:`sensitivity`). Each requested field gets :math:`\partial J/
    \partial f + (\partial R/\partial f)^T\mu` as a dual on its own space
    — the derivative through a field the residual reads, an initial
    condition or a coefficient field.

    ``boundary`` names a mesh boundary label over which the misfit is
    integrated instead of the volume: surface observations. A misfit with
    terms on several domains is a dict ``{None: volume integrand, "Top":
    surface integrand}``; then ``boundary`` is ignored.

    Returns ``{"J": float, "parameters": {expr: float}, "fields": {var: dual}}``;
    the duals are NumPy copies, safe to keep.
    """
    scratch = _shared_scratch if scratch is None else scratch
    parameters, fields = list(parameters), list(fields)
    u = solver.u
    mesh = u.mesh
    # One misfit, or several terms on different domains: {None: volume
    # integrand, "Top": surface integrand, ...}. J and every dual are sums.
    terms = dict(misfit) if isinstance(misfit, dict) else {boundary: misfit}
    J = 0.0
    duals = {}
    grad = {p: 0.0 for p in parameters}
    read = [u] + [f for f in fields if f is not u]
    for where, term in terms.items():
        J += integral(mesh, term, where)
        for var, dual in misfit_duals(term, read, scratch, boundary=where).items():
            if var in duals:
                duals[var].array[...] = np.asarray(duals[var].array) + np.asarray(dual.array)
                scratch.give(dual)
            else:
                duals[var] = dual
        for p in parameters:
            explicit = sympy.diff(_peel_except(term, p), p)
            if explicit != 0:
                grad[p] += integral(mesh, explicit, where)
    # The explicit part on a field the misfit reads directly — never on the
    # unknown, whose misfit dual is the adjoint's right-hand side.
    out_fields = {var: (np.array(duals[var].array, copy=True) if (var in duals and var is not u)
                        else np.zeros_like(np.asarray(var.array))) for var in fields}
    # A history slot the residual reads (psi_star[0]) holds the tracked field
    # at the solve's input, so its dual is the derivative with respect to
    # that field: read the slot, route the dual to the field.
    route = {}
    for history in (getattr(solver, "DuDt", None), getattr(solver, "DFDt", None)):
        if history is None or not getattr(history, "psi_star", None):
            continue
        text = str(history.psi_fn)
        for field in fields:
            # the unknown itself is a control through its INPUT level, which
            # is what the slot holds
            if _token_of(field) in text:
                route[history.psi_star[0]] = field
    read_vars = [f for f in fields if f is not u] + list(route)
    if u in duals:
        rhs = duals.pop(u)
        rhs.array[...] = -np.asarray(rhs.array)
        mu = scratch.take(u)
        if getattr(solver, "p", None) is not None and hasattr(solver, "_subdict"):
            lam = scratch.take(solver.p)
            _, reason = solver.adjoint_solve((rhs, None), target=(mu, lam))
            scratch.give(lam)
        else:
            _, reason = solver.adjoint_solve(rhs, target=mu)
        scratch.give(rhs)
        if reason <= 0:
            raise RuntimeError(f"gradient: the adjoint of {type(solver).__name__}({u.name}) "
                               f"did not converge ({reason})")
        for p in parameters:
            grad[p] += solver.sensitivity(mu, p)
        for var, dual in field_duals(solver, mu, read_vars, scratch).items():
            target = route.get(var, var)
            out_fields[target] = out_fields[target] + np.asarray(dual.array)
            scratch.give(dual)
        scratch.give(mu)
    for dual in duals.values():
        scratch.give(dual)
    return {"J": J, "parameters": grad, "fields": out_fields}


def minimise(objective, x0, bounds=None, method="lmvm", options=None, max_evaluations=100,
             gradient_tolerance=1e-8, callback=None):
    r"""Minimise ``objective(x) -> (J, dJ/dx)`` with PETSc TAO.

    The driver for an inversion: the misfit and its gradient come from the
    adjoint, the step along the gradient is the line search's, and the
    quasi-Newton update is TAO's limited-memory one (``"lmvm"``; ``"blmvm"``
    honours ``bounds``, a pair of arrays). ``x0`` is a NumPy array; a few
    scalar controls are replicated on every rank, each rank's TAO doing the
    same arithmetic on the same numbers, and the objective's own collective
    solves keep the ranks in step. Returns ``(x, info)`` with the iterations,
    the converged reason and the history of ``(J, x)`` per evaluation.
    """
    from petsc4py import PETSc
    x0 = np.asarray(x0, dtype=float).ravel()
    x = PETSc.Vec().createSeq(x0.size, comm=PETSc.COMM_SELF)
    x.setArray(x0)
    history = []

    def fg(tao, xv, g):
        values = np.array(xv.getArray(readonly=True), copy=True)
        J, grad = objective(values)
        g.setArray(np.asarray(grad, dtype=float).ravel())
        history.append((float(J), values))
        if callback is not None:
            callback(J, values)
        return float(J)

    tao = PETSc.TAO().create(comm=PETSc.COMM_SELF)
    tao.setType(method)
    tao.setObjectiveGradient(fg, None)
    if bounds is not None:
        lo, hi = bounds
        lower = x.duplicate(); lower.setArray(np.asarray(lo, dtype=float).ravel())
        upper = x.duplicate(); upper.setArray(np.asarray(hi, dtype=float).ravel())
        tao.setVariableBounds(lower, upper)
    tao.setMaximumFunctionEvaluations(int(max_evaluations))
    tao.setTolerances(gatol=gradient_tolerance)
    for key, value in (options or {}).items():
        PETSc.Options().setValue(key, value)
    tao.setFromOptions()
    tao.setSolution(x)
    tao.solve()
    info = {"iterations": int(tao.getIterationNumber()),
            "reason": int(tao.getConvergedReason()),
            "history": history}
    out = np.array(x.getArray(readonly=True), copy=True)
    tao.destroy()
    return out, info


_n = [0]


def _counter():
    _n[0] += 1
    return _n[0]


class TranscriptAdjoint:
    """The backward pass over a model's recorded run.

    Parameters
    ----------
    model : uw.Model
        With ``model.transcript`` holding one entry per step, each with the
        snapshot it started from (``model.record_every = 1``).
    final_state
        ``model.save_state()`` taken after the last step — the one level the
        transcript does not hold, since it records steps and an N-step run has
        N+1 levels.
    """

    def __init__(self, model, final_state):
        self.model = model
        self.final_state = final_state
        self.steps = list(model.transcript)
        self._scratch = _Scratch()
        missing = [s.index for s in self.steps if not s.restorable]
        if missing:
            raise RuntimeError(
                f"steps {missing} kept no snapshot; set model.record_every = 1 "
                f"before the run so every step keeps the state it started from")

    # ------------------------------------------------------------------
    def gradient(self, misfit, parameters: Iterable = (), fields: Iterable = ()):
        r"""``dJ/dm`` for each parameter, and the dual on each field at level 0.

        Parameters
        ----------
        misfit : sympy expression
            :math:`J` as an integrand over the mesh in the fields at the final
            level, e.g. ``(v.sym - v_target.sym).dot(v.sym - v_target.sym) / 2``.
        parameters
            Named expressions (``uw.expression``) the residuals depend on.
        fields
            MeshVariables whose INITIAL values are controls; the result holds
            :math:`\partial J/\partial f_0` as a dual field, so a control
            :math:`c` with :math:`f_0 = f_0(c)` finishes with a dot product
            against :math:`\partial f_0/\partial c`.

        Returns
        -------
        dict
            ``{"J": float, "parameters": {expr: float}, "fields": {var: dual}}``
        """
        parameters = list(parameters)
        fields = list(fields)
        model = self.model
        scratch = self._scratch

        # J and its dual on every field it touches, at the final level — and
        # the explicit dJ/dm, for a misfit that names a parameter directly.
        model.load_state(self.final_state)
        J = float(uw.maths.Integral(self._mesh(), misfit).evaluate())
        acc: Dict[str, object] = {}
        for var, dual in misfit_duals(misfit, self._tokens().values(), scratch).items():
            self._accumulate(acc, var, dual)

        grad = {p: 0.0 for p in parameters}
        for p in parameters:
            explicit = sympy.diff(_peel_except(misfit, p), p)
            if explicit != 0:
                grad[p] += float(uw.maths.Integral(self._mesh(), explicit).evaluate())

        for k in range(len(self.steps) - 1, -1, -1):
            step = self.steps[k]
            solves = [e for e in step.events if e["kind"] == "solve"]
            for j in range(len(solves) - 1, -1, -1):
                solver = model.part_object(solves[j]["part"])
                if solver is None:
                    raise RuntimeError(
                        f"step {step.index}: no live object for part "
                        f"{solves[j]['part']!r} — the backward pass needs the "
                        f"solvers of the run in this process")
                u = solver.u
                rhs = acc.get(u.name)
                # Decided on every rank together: the gate guards a collective
                # solve, and a misfit supported on one rank's cells deadlocked
                # here when each rank looked only at its own values.
                if not self._nonzero(rhs):
                    continue
                inputs = self._linearise_at(step, solves, j)
                mu = self._adjoint(solver, rhs)
                scratch.give(acc.pop(u.name))  # consumed: this level's output

                for p in parameters:
                    grad[p] += solver.sensitivity(mu, p)

                for var, symbols, derivatives in self._reads(solver, u):
                    value = [solver.adjoint_integrand(mu, s) for s in symbols]
                    g1 = None
                    if derivatives:
                        # g1[i, k] = the contraction with respect to d f_i / d x_k
                        cdim = self._mesh().cdim
                        g1 = sympy.zeros(len(symbols), cdim)
                        for (i, k), atom in derivatives.items():
                            g1[i, k] = solver.adjoint_integrand(mu, atom)
                    target = inputs.get(var.name, var)     # a history -> its field
                    self._accumulate(acc, target,
                                     dual_on(target, _as_expression(value), g1, scratch))
                scratch.give(mu)

        out_fields = {}
        for var in fields:
            held = acc.get(var.name)
            out_fields[var] = (np.zeros_like(np.asarray(var.array)) if held is None
                               else np.array(held.array, copy=True))
        for held in acc.values():
            scratch.give(held)
        return {"J": J, "parameters": grad, "fields": out_fields}

    # ------------------------------------------------------------------
    def _mesh(self):
        return next(iter(self.model._variables.values())).mesh

    def _tokens(self):
        """``{token: variable}`` — how each variable PRINTS inside a residual.

        A variable prints as its symbol, which need not be its name and can
        carry nested braces (a history slot is ``{\\psi^{*}_{...}}``), so the
        token is taken from the symbol's own text (:func:`_token_of`).
        Matching on the name found the user's fields and silently missed
        every history, which cut the chain at the first step."""
        return {_token_of(var): var for var in self.model._variables.values()
                if hasattr(var, "sym")}

    def _reads(self, solver, unknown):
        return _reads_of(solver, unknown, self._tokens())

    def _linearise_at(self, step, solves, j):
        """Restore the step's snapshot, replay solves 0..j, and put each
        history's input back where the post-solve hook shifted it. Returns
        ``{history-slot name: tracked field}`` for the histories solve j read."""
        model = self.model
        model.load_state(step.snapshot)
        for e in solves[:j]:
            self._replay(model.part_object(e["part"]), step)
        solver = model.part_object(solves[j]["part"])
        histories = [h for h in (getattr(solver, "DuDt", None), getattr(solver, "DFDt", None))
                     if h is not None and getattr(h, "psi_star", None)]
        captured = []
        for h in histories:
            tracked = self._tracked_field(h)
            if tracked is None:
                continue
            captured.append((h, tracked, np.array(tracked.array, copy=True)))
        self._replay(solver, step)
        inputs = {}
        for h, tracked, before in captured:
            h.psi_star[0].array[...] = before
            inputs[h.psi_star[0].name] = tracked
        return inputs

    def _replay(self, solver, step):
        if getattr(solver, "DuDt", None) is not None:
            solver.solve(timestep=step.dt, zero_init_guess=False)
        else:
            solver.solve(zero_init_guess=False)

    def _tracked_field(self, history):
        text = str(history.psi_fn)
        for token, var in self._tokens().items():
            if token in text:
                return var
        return None

    def _adjoint(self, solver, rhs):
        u = solver.u
        scratch = self._scratch
        mu = scratch.take(u)
        neg = scratch.take(u)
        neg.array[...] = -np.asarray(rhs.array)
        if getattr(solver, "p", None) is not None and hasattr(solver, "_subdict"):
            p_adj = scratch.take(solver.p)
            _, reason = solver.adjoint_solve((neg, None), target=(mu, p_adj))
            scratch.give(p_adj)
        else:
            _, reason = solver.adjoint_solve(neg, target=mu)
        scratch.give(neg)
        if reason <= 0:
            raise RuntimeError(f"adjoint of {type(solver).__name__}({u.name}) did not converge ({reason})")
        return mu

    def _accumulate(self, acc, var, dual):
        held = acc.get(var.name)
        if held is None:
            acc[var.name] = dual
            return
        held.array[...] = np.asarray(held.array) + np.asarray(dual.array)
        self._scratch.give(dual)

    @staticmethod
    def _nonzero(dual):
        """Whether the dual is nonzero ANYWHERE — reduced across ranks, and
        safe on a rank that holds no degrees of freedom of the space."""
        if dual is None:
            local = 0.0
        else:
            values = np.asarray(dual.array)
            local = float(np.abs(values).max()) if values.size else 0.0
        return uw.mpi.comm.allreduce(local, op=uw.MPI.MAX) > 0.0


def _symbols_of(var):
    n = getattr(var, "num_components", 1)
    return [var.sym[i] for i in range(n)] if n > 1 else [var.sym[0]]


def _as_expression(components):
    """A scalar for a scalar field, a row vector for a vector one — the shape
    a projection onto that field's space expects."""
    return components[0] if len(components) == 1 else sympy.Matrix([components])


def _peel_except(expression, wrt, depth=8):
    """Expand every named expression except ``wrt`` (see the solver's
    ``_peel_except``): ``_peel`` would substitute the parameter's value and
    the derivative of a number is zero."""
    for _ in range(depth):
        named = [e for e in uw.function.fn_extract_expressions(expression)
                 if e is not wrt and e != wrt
                 and not getattr(getattr(e, "sym", None), "is_Number", False)]
        if not named:
            break
        expression = expression.subs({e: e.sym for e in named})
    return expression


def _peel(expression, depth=8):
    for _ in range(depth):
        named = uw.function.fn_extract_expressions(expression)
        if not named:
            break
        expression = expression.subs({e: e.sym for e in named})
    return expression
