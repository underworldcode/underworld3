r"""Reference implementations of the analytic solutions, as originally published.

These are the machine-generated C kernels the suite's SymPy solutions were
transcribed from — Velic's, and PETSc's copies of them. They are kept, and
supported, for two reasons:

1. every transcription is validated against them (see the six gates in
   ``docs/developer/subsystems/analytic-solutions.md``), so they must stay
   available and unmodified;
2. when a benchmark result looks wrong, "is this the transcription or the
   model?" should be a one-line question:

   .. code-block:: python

       sol = uw.analytic.SolCx(mesh, ...)                  # SymPy
       ref = uw.analytic.SolCx(mesh, ..., reference=True)  # the C, verbatim

Reach for :mod:`underworld3.analytic` instead; nothing here is part of the
public API.
"""
