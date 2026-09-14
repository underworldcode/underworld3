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


def dual_on(variable, expression):
    r"""The dual of ``expression`` on ``variable``'s space, held as a field.

    :math:`b_j = \int e\,\phi_j` for every basis function of ``variable``,
    written into a fresh MeshVariable of the same discretisation (one
    coefficient per node) and returned. Assembled as a projection's residual
    at zero, so no linear solve is taken.
    """
    mesh = variable.mesh
    n = getattr(variable, "num_components", 1)
    scratch = uw.discretisation.MeshVariable(
        f"_dual_{variable.name}_{_counter()}", mesh, num_components=n,
        vtype=variable.vtype, degree=variable.degree,
        continuous=getattr(variable, "continuous", True))
    proj = (uw.systems.Projection(mesh, scratch) if n == 1
            else uw.systems.Vector_Projection(mesh, scratch))
    proj.smoothing = 0.0
    proj.petsc_options.delValue("ksp_monitor")
    proj.uw_function = expression
    proj._build(False, False, None)
    gvec = proj.dm.getGlobalVec()
    gvec.set(0.0)
    mesh.update_lvec()
    proj.dm.setAuxiliaryVec(mesh.lvec, None)
    F = gvec.duplicate()
    proj.snes.computeFunction(gvec, F)
    F.scale(-1.0)
    lvec = proj.dm.getLocalVec()
    lvec.set(0.0)
    proj.dm.globalToLocal(F, lvec)
    scratch.vec.array[:] = lvec.array[:]
    proj.dm.restoreLocalVec(lvec)
    F.destroy()
    proj.dm.restoreGlobalVec(gvec)
    mesh._stale_lvec = True
    try:
        scratch._sync_lvec_to_gvec()
    except AttributeError:
        pass
    return scratch


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

        # J and its dual on every field it touches, at the final level.
        model.load_state(self.final_state)
        J = float(uw.maths.Integral(self._mesh(), misfit).evaluate())
        acc: Dict[str, object] = {}
        peeled = _peel(misfit)
        for var, symbols in self._fields_in(misfit):
            dJ = [sympy.diff(peeled, s) for s in symbols]
            self._accumulate(acc, var, dual_on(var, _as_expression(dJ)))

        grad = {p: 0.0 for p in parameters}

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
                if rhs is None or not self._nonzero(rhs):
                    continue
                inputs = self._linearise_at(step, solves, j)
                mu = self._adjoint(solver, rhs)
                acc.pop(u.name, None)          # consumed: this level's output

                for p in parameters:
                    grad[p] += solver.sensitivity(mu, p)

                for var, symbols in self._reads(solver, u):
                    integrand = [solver.adjoint_integrand(mu, s) for s in symbols]
                    target = inputs.get(var.name, var)     # a history -> its field
                    self._accumulate(acc, target,
                                     dual_on(target, _as_expression(integrand)))

        out_fields = {}
        for var in fields:
            held = acc.get(var.name)
            out_fields[var] = (np.zeros_like(np.asarray(var.array)) if held is None
                               else np.array(held.array, copy=True))
        return {"J": J, "parameters": grad, "fields": out_fields}

    # ------------------------------------------------------------------
    def _mesh(self):
        return next(iter(self.model._variables.values())).mesh

    def _tokens(self):
        """``{token: variable}`` — how each variable PRINTS inside a residual.

        A variable prints as its symbol, which need not be its name and can
        carry nested braces (a history slot is ``{\\psi^{*}_{...}}``), so the
        token is taken from the symbol's own text: everything before the
        coordinate arguments and, for a vector, before the component index.
        Matching on the name found the user's fields and silently missed
        every history, which cut the chain at the first step."""
        out = {}
        for var in self.model._variables.values():
            if not hasattr(var, "sym"):
                continue
            text = str(_symbols_of(var)[0]).rsplit("(", 1)[0]     # drop (N.x, N.y)
            if getattr(var, "num_components", 1) > 1:
                text = text.rsplit("_{", 1)[0]                   # drop _{ i }
            out[text] = var
        return out

    def _fields_in(self, expression):
        text = str(_peel(expression))
        return [(v, _symbols_of(v)) for token, v in self._tokens().items()
                if token in text]

    def _reads(self, solver, unknown):
        """The (variable, component, symbol) triples a solver's residual reads,
        other than its unknown. Direct values only; a residual that reads the
        GRADIENT of another field needs the integration-by-parts term, which
        is not built, so it is refused by name rather than dropped."""
        text = str(_peel(solver.F0.sym)) + str(_peel(solver.F1.sym))
        found = []
        for token, var in self._tokens().items():
            if var is unknown:
                continue
            if token not in text:
                continue
            # a component prints as {v}_{ 0 }; a derivative carries a comma,
            # {v}_{ 0,1} for a vector and {T}_{,1} for a scalar
            if re.search(re.escape(token) + r"_\{[^}]*,", text):
                raise NotImplementedError(
                    f"{type(solver).__name__}: the residual reads a derivative "
                    f"of {var.name!r}; the dual on a field read through its "
                    f"gradient is not built yet")
            found.append((var, _symbols_of(var)))
        return found

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
        n = getattr(u, "num_components", 1)
        mu = uw.discretisation.MeshVariable(
            f"_mu_{u.name}_{_counter()}", u.mesh, num_components=n,
            vtype=u.vtype, degree=u.degree, continuous=getattr(u, "continuous", True))
        neg = uw.discretisation.MeshVariable(
            f"_rhs_{u.name}_{_counter()}", u.mesh, num_components=n,
            vtype=u.vtype, degree=u.degree, continuous=getattr(u, "continuous", True))
        neg.array[...] = -np.asarray(rhs.array)
        if getattr(solver, "p", None) is not None and hasattr(solver, "_subdict"):
            pv = solver.p
            p_adj = uw.discretisation.MeshVariable(
                f"_lam_{pv.name}_{_counter()}", pv.mesh, num_components=1,
                vtype=pv.vtype, degree=pv.degree, continuous=getattr(pv, "continuous", True))
            _, reason = solver.adjoint_solve((neg, None), target=(mu, p_adj))
        else:
            _, reason = solver.adjoint_solve(neg, target=mu)
        if reason <= 0:
            raise RuntimeError(f"adjoint of {type(solver).__name__}({u.name}) did not converge ({reason})")
        return mu

    @staticmethod
    def _accumulate(acc, var, dual):
        held = acc.get(var.name)
        if held is None:
            acc[var.name] = dual
            return
        held.array[...] = np.asarray(held.array) + np.asarray(dual.array)

    @staticmethod
    def _nonzero(dual):
        return np.abs(np.asarray(dual.array)).max() > 0.0


def _symbols_of(var):
    n = getattr(var, "num_components", 1)
    return [var.sym[i] for i in range(n)] if n > 1 else [var.sym[0]]


def _as_expression(components):
    """A scalar for a scalar field, a row vector for a vector one — the shape
    a projection onto that field's space expects."""
    return components[0] if len(components) == 1 else sympy.Matrix([components])


def _peel(expression, depth=8):
    for _ in range(depth):
        named = uw.function.fn_extract_expressions(expression)
        if not named:
            break
        expression = expression.subs({e: e.sym for e in named})
    return expression
