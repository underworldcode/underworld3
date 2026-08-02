r"""The Velic family of exact Stokes solutions.

These are the classical variable-viscosity benchmarks: a known body force and a
known viscosity structure on the unit box, with the exact velocity, pressure and
stress they produce. They are the standard way to show a Stokes solver is right
rather than merely converged.

Each is transcribed from its published Maple-generated kernel, which stays
vendored in :mod:`underworld3.analytic._reference` as the oracle the
transcription is validated against and as a supported escape hatch
(``reference=True``).

"""

import functools
import os

import sympy

from ._base import AnalyticSolution, FixedWalls, FreeSlipWalls
from ._transcribe import CSource, evaluate_block, evaluate_expression

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
def _solcx_kernel(variant="_solCx_A"):
    r"""Transcribe the Velic SolCx kernel into SymPy.

    The published source carries two arrangements, ``_solCx_A`` and
    ``_solCx_B``, and dispatches on :math:`\eta_A > \eta_B`. They are not two
    conditionings of one formula: evaluated exactly, ``_solCx_B`` is
    ``_solCx_A`` reflected, :math:`B(x, z) = A(1-x, z)`. It solves the mirrored
    problem so that the algebra derived for a stiff left column can be reused
    when the stiff column is on the right, and undoes the reflection on the way
    out.

    So only ``_solCx_A`` is transcribed, and no dispatch is needed. That is
    safe because the reason for the dispatch — conditioning — was measured
    rather than assumed: this arrangement reproduces the published kernel to
    1e-14 over viscosity ratios from 1e-6 to 1e8 in both directions. See
    ``docs/developer/subsystems/analytic-solutions.md``.

    The remaining branch is spatial (:math:`x < x_c`) and becomes a
    :class:`sympy.Piecewise`.

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
        if float(eta_A) == float(eta_B):
            # The closed form carries (ZR - 1) in several denominators, so equal
            # viscosities are a removable singularity. SymPy cancels it when the
            # expression is evaluated symbolically at a point, but not in the
            # compiled form, so this case would silently return nonsense. A
            # uniform-viscosity benchmark is a different solution anyway.
            raise ValueError(
                "SolCx is a viscosity-jump benchmark and is singular at "
                "eta_A == eta_B. Use a uniform-viscosity solution instead."
            )
        if not 0.0 <= float(x_c) <= 1.0:
            raise ValueError("x_c must lie in [0, 1].")
        if int(n) != n or int(n) < 1:
            raise ValueError("n (vertical wavenumber) must be a positive integer.")

        self.eta_A = float(eta_A)
        self.eta_B = float(eta_B)
        self.x_c = float(x_c)
        self.n = int(n)

        x, z = mesh.X

        # Exact parameters, not floats. The closed form carries (ZR - 1) in
        # several denominators, so at equal viscosities it has a removable
        # singularity: substituting exactly lets SymPy cancel it, while
        # substituting floats leaves a 0/0 that evaluates to nothing useful.
        # Rational() of a float is its exact binary value, so this costs nothing.
        eta_A_exact = sympy.Rational(self.eta_A)
        eta_B_exact = sympy.Rational(self.eta_B)

        # The kernel's own naming: kx is the horizontal wavenumber of the forcing
        # (fixed at pi), kn the vertical one, ZR the viscosity ratio.
        values = {
            _XC: sympy.Rational(self.x_c),
            _KN: self.n * sympy.pi,
            _KX: sympy.pi,
            _ZA: eta_A_exact,
            _ZB: eta_B_exact,
            _ZR: eta_B_exact / eta_A_exact,
            _X: x,
            _Z: z,
        }

        kernel = {
            field: expression.subs(values)
            for field, expression in _solcx_kernel().items()
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

    def velocity_error(self, velocity_var):
        """Global relative L2 velocity error. Equivalent to ``error("velocity", ...)``."""

        return self.error("velocity", velocity_var)

    def evaluate_stress(self, coords):
        """Exact total (Cauchy) stress at ``coords``, as ``(N, 3)`` columns
        :math:`(\\sigma_{xx}, \\sigma_{zz}, \\sigma_{xz})`."""

        import numpy as np

        components = [self.fn_stress[0, 0], self.fn_stress[1, 1], self.fn_stress[0, 1]]
        return np.column_stack(
            [np.asarray(self.evaluate(c, coords)).reshape(-1) for c in components]
        )

    def topography_top(self, coords):
        """Exact dynamic topography :math:`-\\mathbf n\\cdot\\sigma\\cdot\\mathbf n`
        on the top boundary, i.e. :math:`-\\sigma_{zz}`."""

        import numpy as np

        return -np.asarray(self.evaluate(self.fn_stress[1, 1], coords)).reshape(-1)

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


_ETA0, _N, _R = sympy.symbols("eta0 n r")


@functools.lru_cache(maxsize=None)
def _solnl_kernel():
    r"""Transcribe the Velic SolNL kernel into SymPy.

    Six short functions rather than one branching kernel, so each is read on its
    own. The tensor entries are written through a struct (``out.xx = ...``), and
    the viscosity is returned rather than assigned — hence the two ways of
    reading a body here.

    Returns
    -------
    dict
        Field name -> expression in the kernel's own symbols.
    """

    source = CSource(os.path.join(_REFERENCE_DIR, "AnalyticSolNL.c"))
    inputs = {"eta0": _ETA0, "n": _N, "r": _R, "x": _X, "z": _Z}

    def block(name, signature):
        return evaluate_block(source.function(name, returns=signature), inputs)

    velocity = block("SolNL_velocity", "vec2")
    bodyforce = block("SolNL_bodyforce", "vec2")
    stress = block("SolNL_stress", "tensor2")
    strainrate = block("SolNL_strainrate", "tensor2")
    pressure = block("SolNL_pressure", "double")

    viscosity_body = source.function("SolNL_viscosity", returns="double")
    viscosity = evaluate_expression(
        CSource.returned(viscosity_body), evaluate_block(viscosity_body, inputs)
    )

    return {
        "velocity_x": velocity["out.x"],
        "velocity_z": velocity["out.z"],
        "bodyforce_x": bodyforce["out.x"],
        "bodyforce_z": bodyforce["out.z"],
        "pressure": pressure["p"],
        "stress_xx": stress["out.xx"],
        "stress_zz": stress["out.zz"],
        "stress_xz": stress["out.xz"],
        "strainrate_xx": strainrate["out.xx"],
        "strainrate_zz": strainrate["out.zz"],
        "strainrate_xz": strainrate["out.xz"],
        "viscosity": viscosity,
    }


class SolNL(FixedWalls, AnalyticSolution):
    r"""Power-law viscous flow — the SolNL nonlinear benchmark.

    A manufactured solution for a shear-thinning fluid: the viscosity depends on
    the second invariant of the strain rate the solution itself produces,

    .. math::
        \eta = \eta_0 \left(\dot\varepsilon_{ij}\dot\varepsilon_{ij}\right)^{1/r - 1}

    so it tests a nonlinear solver rather than a linear one. The velocity is
    simple — :math:`\mathbf u = (-k\,e^{x}\cos kz,\; e^{x}\sin kz)`, divergence
    free by inspection — and the body force is whatever makes it exact.

    Parameters
    ----------
    mesh : Mesh
        A 2D mesh on the unit box.
    eta_0 : float
        Viscosity prefactor.
    n : int
        Vertical wavenumber.
    r : float
        Power-law exponent. ``r = 1`` is Newtonian; larger is more
        shear-thinning.
    reference : bool
        Evaluate through the vendored kernel instead of the transcription.

    Notes
    -----
    The velocity is not tangential to the walls, so this is posed with the exact
    velocity prescribed on the boundary rather than free slip.
    """

    dim = 2
    nonlinear = True
    reference = (
        "Velic. Transcribed from the published kernel vendored at "
        "underworld3/analytic/_reference/AnalyticSolNL.c."
    )
    eqn_velocity = r"(-k e^{x}\cos kz,\; e^{x}\sin kz), \quad k = n\pi"
    eqn_viscosity = r"\eta_0 (\dot\varepsilon_{ij}\dot\varepsilon_{ij})^{1/r - 1}"

    def __init__(self, mesh, eta_0=1.0, n=1, r=1.5, reference=False):
        super().__init__(mesh)

        if float(eta_0) <= 0.0:
            raise ValueError("eta_0 must be positive.")
        if int(n) != n or int(n) < 1:
            raise ValueError("n (vertical wavenumber) must be a positive integer.")
        if float(r) <= 0.0:
            raise ValueError("r (power-law exponent) must be positive.")

        self.eta_0 = float(eta_0)
        self.n = int(n)
        self.r = float(r)

        x, z = mesh.X
        values = {
            _ETA0: sympy.Rational(self.eta_0),
            _N: self.n,
            _R: sympy.Rational(self.r),
            _X: x,
            _Z: z,
        }
        kernel = {
            field: expression.subs(values)
            for field, expression in _solnl_kernel().items()
        }

        self.fn_velocity = sympy.Matrix(
            [[kernel["velocity_x"], kernel["velocity_z"]]]
        )
        self.fn_pressure = kernel["pressure"]
        self.fn_viscosity = kernel["viscosity"]
        self.fn_bodyforce = sympy.Matrix(
            [[kernel["bodyforce_x"], kernel["bodyforce_z"]]]
        )
        self.fn_stress = sympy.Matrix(
            [
                [kernel["stress_xx"], kernel["stress_xz"]],
                [kernel["stress_xz"], kernel["stress_zz"]],
            ]
        )
        self.fn_strainrate = sympy.Matrix(
            [
                [kernel["strainrate_xx"], kernel["strainrate_xz"]],
                [kernel["strainrate_xz"], kernel["strainrate_zz"]],
            ]
        )

        if reference:
            self._use_reference_kernel()

    def _use_reference_kernel(self):
        """Point-evaluation only: opaque to the JIT, so no solver can use it."""

        from ._reference import _velic

        x, z = self.mesh.X
        parameters = (self.eta_0, self.n, self.r)

        self.fn_velocity = sympy.Matrix(
            [
                [
                    _velic.AnalyticSolNL_velocity_x(*parameters, x, z),
                    _velic.AnalyticSolNL_velocity_y(*parameters, x, z),
                ]
            ]
        )
        self.fn_bodyforce = sympy.Matrix(
            [
                [
                    _velic.AnalyticSolNL_bodyforce_x(*parameters, x, z),
                    _velic.AnalyticSolNL_bodyforce_y(*parameters, x, z),
                ]
            ]
        )
        self.fn_viscosity = _velic.AnalyticSolNL_viscosity(*parameters, x, z)
