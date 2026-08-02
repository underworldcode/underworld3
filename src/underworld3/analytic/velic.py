r"""The Velic family of exact Stokes solutions.

These are the classical variable-viscosity benchmarks: a known body force and a
known viscosity structure on the unit box, with the exact velocity, pressure and
stress they produce. They are the standard way to show a Stokes solver is right
rather than merely converged.

Each is transcribed from its published Maple-generated kernel, which stays
vendored in :mod:`underworld3.analytic._reference` as the oracle the
transcription is validated against and as a supported escape hatch
(``reference=True``).

.. warning::
   **Under validation — not exported from** :mod:`underworld3.analytic` **yet.**
   ``uw.analytic.SolCx`` is still the reference kernel. Two gates are open, both
   recorded in ``docs/developer/subsystems/analytic-solutions.md``:

   * the transcription of the ``_solCx_B`` arrangement disagrees with the
     published kernel, while ``_solCx_A`` agrees to 1e-14 across every regime
     tested (contrast 1e-6 to 1e8, both directions, several ``x_c`` and ``n``).
     Since the dispatcher routes :math:`\eta_A < \eta_B` to ``_solCx_B``, and the
     ``_solCx_A`` transcription reproduces *that* output, the two arrangements
     should be equivalent and one of them is being read wrongly. Unexplained.
   * the isoviscous case :math:`\eta_A = \eta_B` returns zero, where the kernel
     does not — a degeneracy in the closed form at unit viscosity ratio that the
     transcription is not handling.

   Nothing here is used until both are closed.
"""

import functools
import os

import sympy

from ._base import AnalyticSolution, FreeSlipWalls
from ._transcribe import CSource, evaluate_block

_REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "_reference")

# Free symbols of the transcribed kernel, in the kernel's own naming.
_X, _Z = sympy.symbols("x z")
_XC, _KN, _KX = sympy.symbols("xc kn kx")
_ZA, _ZB, _ZR = sympy.symbols("ZA ZB ZR")

# What the kernel's shared tail leaves behind, and what each means.
_SOLCX_OUTPUTS = {
    "velocity_x": "u1",
    "velocity_z": "u2",
    "stress_xx": "u3",
    "stress_zx": "u4",
    "pressure": "u5",
    "stress_zz": "u6",
}


@functools.lru_cache(maxsize=None)
def _solcx_kernel(variant):
    r"""Transcribe one conditioning variant of the Velic SolCx kernel.

    The published kernel carries two arrangements of the same solution —
    ``_solCx_A`` for :math:`\eta_A > \eta_B` and ``_solCx_B`` for the reverse —
    because the integration constants lose precision differently depending on
    which side is stiff. Which one applies is decided by the viscosities, so it
    is resolved when a solution is constructed, not symbolically.

    Within a variant the only branch left is spatial (:math:`x < x_c`), and that
    one becomes a :class:`sympy.Piecewise`.

    Returns
    -------
    dict
        Field name -> Piecewise expression in the kernel's own symbols.
    """

    source = CSource(os.path.join(_REFERENCE_DIR, "solCx.c"))
    body = source.function(variant)
    left_block, right_block, tail = CSource.branches(
        body, "if (x<xc)", tail_ends_at="if( vel != NULL )"
    )

    inputs = {
        "x": _X,
        "z": _Z,
        "xc": _XC,
        "kn": _KN,
        "kx": _KX,
        "ZA": _ZA,
        "ZB": _ZB,
        "ZR": _ZR,
    }

    sides = {}
    for side, block in (("left", left_block), ("right", right_block)):
        # The tail turns the branch's integration constants into the fields, and
        # is shared, so it is replayed against each branch's environment.
        sides[side] = evaluate_block(tail, evaluate_block(block, inputs))

    # The kernel accumulates each field against its vertical mode before
    # returning; the tail stops short of that, so apply it here.
    vertical = {
        "u1": sympy.cos(_KN * _Z),
        "u2": sympy.sin(_KN * _Z),
        "u3": sympy.cos(_KN * _Z),
        "u4": sympy.sin(_KN * _Z),
        "u5": sympy.cos(_KN * _Z),
        "u6": sympy.cos(_KN * _Z),
    }

    return {
        field: sympy.Piecewise(
            (sides["left"][symbol] * vertical[symbol], _X < _XC),
            (sides["right"][symbol] * vertical[symbol], True),
        )
        for field, symbol in _SOLCX_OUTPUTS.items()
    }


class SolCx(FreeSlipWalls, AnalyticSolution):
    r"""Isoviscous-column Stokes flow with a viscosity step — the SolCx benchmark.

    Viscosity jumps from :math:`\eta_A` to :math:`\eta_B` at :math:`x = x_c` on
    the unit box, driven by the density forcing
    :math:`\mathbf f = (0, \cos(\pi x)\sin(n\pi z))`, with free slip on all four
    walls. The discontinuous viscosity and the resulting pressure jump make it
    the standard test of whether a Stokes solver handles a sharp material
    contrast, and the enclosed free-slip domain makes it a test of the pressure
    nullspace as well.

    Parameters
    ----------
    mesh : Mesh
        A 2D mesh on the unit box.
    eta_A, eta_B : float
        Viscosity left and right of the step. Both must be positive.
    x_c : float
        Position of the step, in :math:`[0, 1]`.
    n : int
        Vertical wavenumber of the forcing.
    reference : bool
        Evaluate through the vendored Maple kernel instead of the transcribed
        SymPy. Slower, not usable inside a solver, and intended for answering
        "is this the transcription or the model?" — see
        ``docs/developer/subsystems/analytic-solutions.md``.

    Examples
    --------
    >>> sol = uw.analytic.SolCx(mesh, eta_A=1.0, eta_B=1.0e6)
    >>> stokes.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    >>> stokes.bodyforce = sol.fn_bodyforce
    >>> sol.apply_boundary_conditions(stokes)
    >>> stokes.solve()
    >>> sol.error("velocity", stokes.u)

    Notes
    -----
    The body force is :math:`+\cos(\pi x)\sin(n\pi z)`. Underworld2's
    documentation quotes the opposite sign; the sign here is the one consistent
    with UW3's momentum convention and with the kernel's own pressure, which is
    what the validation checks against.
    """

    dim = 2
    reference = (
        "Velic; see also Duretz et al. (2011). Transcribed from the published "
        "kernel vendored at underworld3/analytic/_reference/solCx.c."
    )
    eqn_viscosity = r"\eta_A \;(x<x_c), \quad \eta_B \;(x \ge x_c)"
    eqn_bodyforce = r"(0,\; \cos(\pi x)\,\sin(n \pi z))"

    def __init__(self, mesh, eta_A=1.0, eta_B=1.0e6, x_c=0.5, n=1, reference=False):
        super().__init__(mesh)

        if not (float(eta_A) > 0.0 and float(eta_B) > 0.0):
            raise ValueError("eta_A and eta_B must be positive.")
        if not 0.0 <= float(x_c) <= 1.0:
            raise ValueError("x_c must lie in [0, 1].")
        if int(n) != n or int(n) < 1:
            raise ValueError("n (vertical wavenumber) must be a positive integer.")

        self.eta_A = float(eta_A)
        self.eta_B = float(eta_B)
        self.x_c = float(x_c)
        self.n = int(n)

        x, z = mesh.X

        # The kernel's own naming: kx is the horizontal wavenumber of the forcing
        # (fixed at pi), kn the vertical one, ZR the viscosity ratio.
        values = {
            _XC: self.x_c,
            _KN: self.n * sympy.pi,
            _KX: sympy.pi,
            _ZA: self.eta_A,
            _ZB: self.eta_B,
            _ZR: self.eta_B / self.eta_A,
            _X: x,
            _Z: z,
        }

        variant = "_solCx_A" if self.eta_A > self.eta_B else "_solCx_B"
        kernel = {
            field: expression.subs(values)
            for field, expression in _solcx_kernel(variant).items()
        }

        self.fn_velocity = sympy.Matrix(
            [[kernel["velocity_x"], kernel["velocity_z"]]]
        )
        self.fn_pressure = kernel["pressure"]
        self.fn_stress = sympy.Matrix(
            [
                [kernel["stress_xx"], kernel["stress_zx"]],
                [kernel["stress_zx"], kernel["stress_zz"]],
            ]
        )

        # The viscosity tie-break at x == x_c matches the kernel's own step, so a
        # point exactly on the interface is treated the same way by both.
        self.fn_viscosity = sympy.Piecewise((self.eta_A, x < self.x_c), (self.eta_B, True))

        # sigma = -p I + 2 eta edot, so the strain rate follows from the fields
        # the kernel returns. It also returns its own strain rate, derived
        # independently — the two are compared as one of the validation gates.
        self.fn_strainrate = (
            self.fn_stress + self.fn_pressure * sympy.eye(2)
        ) / (2 * self.fn_viscosity)

        self.fn_bodyforce = sympy.Matrix(
            [[0, sympy.cos(sympy.pi * x) * sympy.sin(self.n * sympy.pi * z)]]
        )

        if reference:
            self._use_reference_kernel()

    def _use_reference_kernel(self):
        """Replace the transcribed fields with the vendored kernel's own.

        Point evaluation only: these are opaque to the JIT, so a solution built
        this way cannot be handed to a solver.
        """

        from ._reference import _velic

        x, z = self.mesh.X
        parameters = (self.eta_A, self.eta_B, self.x_c, self.n)

        self.fn_velocity = sympy.Matrix(
            [
                [
                    _velic.AnalyticSolCx_velocity_x(*parameters, x, z),
                    _velic.AnalyticSolCx_velocity_y(*parameters, x, z),
                ]
            ]
        )
        self.fn_pressure = _velic.AnalyticSolCx_pressure(*parameters, x, z)
        self.fn_stress = sympy.Matrix(
            [
                [
                    _velic.AnalyticSolCx_stress_xx(*parameters, x, z),
                    _velic.AnalyticSolCx_stress_xy(*parameters, x, z),
                ],
                [
                    _velic.AnalyticSolCx_stress_xy(*parameters, x, z),
                    _velic.AnalyticSolCx_stress_yy(*parameters, x, z),
                ],
            ]
        )
