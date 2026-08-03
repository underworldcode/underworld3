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

        # The viscosity tie-break at x == x_c matches the kernel's own step, so a
        # point exactly on the interface is treated the same way by both.
        self.set_fields(
            velocity=(kernel["velocity_x"], kernel["velocity_z"]),
            pressure=kernel["pressure"],
            viscosity=sympy.Piecewise((self.eta_A, x < self.x_c), (self.eta_B, True)),
            bodyforce=(0, sympy.cos(sympy.pi * x) * sympy.sin(self.n * sympy.pi * z)),
            stress=(
                (kernel["stress_xx"], kernel["stress_zx"]),
                (kernel["stress_zx"], kernel["stress_zz"]),
            ),
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
    stress_is_deviatoric = True
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

        self.set_fields(
            velocity=(kernel["velocity_x"], kernel["velocity_z"]),
            pressure=kernel["pressure"],
            viscosity=kernel["viscosity"],
            bodyforce=(kernel["bodyforce_x"], kernel["bodyforce_z"]),
            # Both are published and both are consistent, so both are supplied:
            # the conformance check then compares them against each other.
            stress=(
                (kernel["stress_xx"], kernel["stress_xz"]),
                (kernel["stress_xz"], kernel["stress_zz"]),
            ),
            strainrate=(
                (kernel["strainrate_xx"], kernel["strainrate_xz"]),
                (kernel["strainrate_xz"], kernel["strainrate_zz"]),
            ),
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


_B, _M = sympy.symbols("B m")
_KN = sympy.Symbol("kn_solm")
_KM = sympy.Symbol("km")

# The kernel leaves the fields in u1..u6, each still to be multiplied by its
# vertical mode. Same convention as SolCx, and the source says so in a comment:
# "u1 = Vx, u2 = Vz, u3 = txx, u4 = tzx, u5 = pressure, u6 = tzz".
_SOLKX_OUTPUTS = {
    "velocity_x": ("u1", sympy.cos),
    "velocity_z": ("u2", sympy.sin),
    "stress_xx": ("u3", sympy.cos),
    "stress_zx": ("u4", sympy.sin),
    "pressure": ("u5", sympy.cos),
    "stress_zz": ("u6", sympy.cos),
}


@functools.lru_cache(maxsize=None)
def _solkx_kernel():
    r"""Transcribe the Velic SolKx kernel into SymPy.

    One straight-line block, no branches — the exponential viscosity has no
    interface to split on, so unlike SolCx there is no ``Piecewise`` here.

    Returns
    -------
    dict
        Field name -> expression in the kernel's own symbols.
    """

    source = CSource(os.path.join(_REFERENCE_DIR, "solKx.c"))
    body = source.function("SolKxSolution", returns="static PetscErrorCode")

    # Stop at the output section (comments are already stripped, so anchor on
    # code): it accumulates with `+=` and writes
    # through pointers, neither of which this reader interprets.
    body = body[: body.index("if (mu)")]

    # The kernel takes its coordinates from an array, so `pos` is bound to one.
    inputs = {"pos": (_X, _Z), "B": _B, "m": _M, "n": _N}
    scope = evaluate_block(body, inputs)

    km = _M * sympy.pi
    return {
        field: scope[symbol] * mode(km * _Z)
        for field, (symbol, mode) in _SOLKX_OUTPUTS.items()
    }


class SolKx(FreeSlipWalls, AnalyticSolution):
    r"""Stokes flow with an exponentially varying viscosity — the SolKx benchmark.

    Viscosity :math:`\eta = e^{2Bx}` on the unit box, driven by the density
    forcing :math:`\mathbf f = (0,\; \sin(m\pi z)\cos(n\pi x))`, free slip on all
    four walls.

    The companion to SolCx: same geometry and forcing, but the viscosity varies
    *smoothly* rather than jumping. A solver can do well on one and badly on the
    other — a jump tests how the discretisation handles a discontinuity, a
    gradient tests whether the operator stays well conditioned as the contrast
    builds across every element. Over the unit box the total contrast is
    :math:`e^{2B}`, so ``B = 5`` already spans four orders of magnitude.

    Parameters
    ----------
    mesh : Mesh
        A 2D mesh on the unit box.
    B : float
        Viscosity exponent.
    n : int
        Horizontal wavenumber of the forcing.
    m : int
        Vertical wavenumber.

    Examples
    --------
    >>> sol = uw.analytic.SolKx(mesh, B=2.302585, n=3, m=2.0)
    >>> stokes.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    >>> stokes.bodyforce = sol.fn_bodyforce
    >>> sol.apply_boundary_conditions(stokes)

    Notes
    -----
    Transcribed from PETSc's copy of the kernel rather than Underworld2's: it is
    self-contained, returns every field in one call, and is actively maintained
    upstream.

    Validated by the equations rather than against a compiled kernel. The forcing
    and boundary conditions are known, and a field set that satisfies Stokes with
    them is *the* solution by uniqueness — so the momentum and incompressibility
    residuals settle it without needing an oracle.
    """

    dim = 2
    reference = (
        "Velic; transcribed from PETSc src/snes/tutorials/ex69.c (SolKxSolution), "
        "vendored at underworld3/analytic/_reference/solKx.c (BSD-2-Clause)."
    )
    eqn_viscosity = r"e^{2Bx}"
    eqn_bodyforce = r"(0,\; \sin(m \pi z)\cos(n \pi x))"

    def __init__(self, mesh, B=2.302585092994046, n=3, m=2):
        super().__init__(mesh)

        if int(n) != n or int(n) < 1:
            raise ValueError("n (horizontal wavenumber) must be a positive integer.")
        if int(m) != m or int(m) < 1:
            # The kernel itself allows non-integral m, and PETSc says so. But the
            # vertical velocity carries sin(m pi z), which vanishes at z = 1 only
            # for integer m — so a fractional value silently stops satisfying free
            # slip on the top wall while still solving the equations, and the
            # benchmark quietly becomes a different problem.
            raise ValueError(
                "m (vertical wavenumber) must be a positive integer: the free-slip "
                "condition on the top wall requires sin(m*pi) = 0."
            )

        self.B = float(B)
        self.n = int(n)
        self.m = float(m)

        x, z = mesh.X
        values = {
            _B: sympy.Rational(self.B),
            _N: self.n,
            _M: sympy.Rational(self.m),
            _X: x,
            _Z: z,
        }
        kernel = {
            field: expression.subs(values)
            for field, expression in _solkx_kernel().items()
        }

        self.set_fields(
            velocity=(kernel["velocity_x"], kernel["velocity_z"]),
            pressure=kernel["pressure"],
            viscosity=sympy.exp(2 * sympy.Rational(self.B) * x),
            bodyforce=(
                0,
                sympy.sin(self.m * sympy.pi * z) * sympy.cos(self.n * sympy.pi * x),
            ),
            stress=(
                (kernel["stress_xx"], kernel["stress_zx"]),
                (kernel["stress_zx"], kernel["stress_zz"]),
            ),
        )


_Y = sympy.Symbol("y")
_BETA = sympy.Symbol("Beta")


@functools.lru_cache(maxsize=None)
def _soldb_kernel(dim):
    r"""Transcribe a Dohrmann–Bochev solution into SymPy.

    Six short methods rather than one kernel, each written as a C++ member of a
    header-only class, and each taking its coordinates from an array.

    Returns
    -------
    dict
        Field name -> expression in the kernel's own symbols.
    """

    names = {2: ("x", "z"), 3: ("x", "y", "z")}[dim]
    coordinates = {2: (_X, _Z), 3: (_X, _Y, _Z)}[dim]

    source = CSource(
        os.path.join(_REFERENCE_DIR, f"AnalyticSolDB{dim}d.hpp")
    )
    inputs = {"in": coordinates, "Beta": _BETA}

    def block(method):
        return evaluate_block(source.function(method), inputs)

    velocity = block("velocity")
    bodyforce = block("bodyforce")
    stress = block("stress")
    strainrate = block("strainrate")
    pressure = block("pressure")

    fields = {
        "pressure": pressure["p"],
        # SolDB2d writes its unit viscosity straight into the output array, so
        # there is no named variable to read; 3D has one.
        "viscosity": block("viscosity")["eta"] if dim == 3 else sympy.Integer(1),
    }
    for axis, name in enumerate(names):
        fields[f"velocity_{name}"] = velocity[f"v{name}"]
        fields[f"bodyforce_{name}"] = bodyforce[f"f{name}"]

    for i, a in enumerate(names):
        for b in names[i:]:
            fields[f"stress_{a}{b}"] = stress[f"t{a}{b}"]
            fields[f"strainrate_{a}{b}"] = strainrate[f"e{a}{b}"]

    return fields


class _SolDB(FixedWalls, AnalyticSolution):
    """Shared assembly for the Dohrmann–Bochev manufactured solutions."""

    stress_is_deviatoric = True

    def _assemble(self, mesh, values, names):
        kernel = {
            field: expression.subs(values)
            for field, expression in _soldb_kernel(self.dim).items()
        }

        def tensor(prefix):
            return sympy.Matrix(
                [
                    [
                        kernel[f"{prefix}_{min(a, b)}{max(a, b)}"]
                        if f"{prefix}_{min(a, b)}{max(a, b)}" in kernel
                        else kernel[f"{prefix}_{max(a, b)}{min(a, b)}"]
                        for b in names
                    ]
                    for a in names
                ]
            )

        self.set_fields(
            velocity=[kernel[f"velocity_{n}"] for n in names],
            pressure=kernel["pressure"],
            viscosity=kernel["viscosity"],
            bodyforce=[kernel[f"bodyforce_{n}"] for n in names],
            stress=tensor("stress"),
            strainrate=tensor("strainrate"),
        )


class SolDB2d(_SolDB):
    r"""Isoviscous polynomial manufactured solution — Dohrmann & Bochev, 2D.

    Unit viscosity, a polynomial velocity, and whatever body force makes it
    exact. There is no discontinuity and no large contrast, which is the point:
    it isolates the discretisation from the conditioning, so an error here is an
    error in the element or the solve rather than in how a hard coefficient was
    handled.

    Parameters
    ----------
    mesh : Mesh
        A 2D mesh on the unit box.

    Notes
    -----
    The velocity is not tangential to the walls, so this is posed with the exact
    velocity prescribed on the boundary.
    """

    dim = 2
    reference = (
        "Dohrmann & Bochev (2004), Int. J. Numer. Meth. Fluids 46, 183-201, "
        "doi:10.1002/fld.752. Transcribed from the kernel vendored at "
        "underworld3/analytic/_reference/AnalyticSolDB2d.hpp."
    )
    eqn_viscosity = r"1"

    def __init__(self, mesh):
        super().__init__(mesh)

        x, z = mesh.X
        self._assemble(mesh, {_X: x, _Z: z}, ("x", "z"))


class SolDB3d(_SolDB):
    r"""Variable-viscosity manufactured solution in 3D — Burstedde et al.

    Viscosity :math:`\eta = e^{1 - \beta\,[x(1-x) + y(1-y) + z(1-z))]}`, smooth
    and peaked in the interior, with a polynomial velocity and a body force
    chosen to make the pair exact.

    The only 3D solution in the suite, and the only one whose viscosity varies in
    every direction at once — a 2D benchmark cannot catch a term that is wrong
    only in the third dimension, and several parts of a Stokes discretisation
    (the pressure space, the null space, the tensor assembly) genuinely differ
    between 2D and 3D.

    Parameters
    ----------
    mesh : Mesh
        A 3D mesh on the unit cube.
    beta : float
        Viscosity exponent. Zero is isoviscous; larger makes the interior
        viscosity peak sharper.

    Notes
    -----
    The velocity is not tangential to the boundary, so the exact velocity is
    prescribed there.
    """

    dim = 3
    reference = (
        "Burstedde et al. (2013), Geophys. J. Int. 192(3), 889-906, "
        "doi:10.1093/gji/ggs070. Transcribed from the kernel vendored at "
        "underworld3/analytic/_reference/AnalyticSolDB3d.hpp."
    )
    eqn_viscosity = r"e^{1 - \beta [x(1-x) + y(1-y) + z(1-z)]}"

    def __init__(self, mesh, beta=4.0):
        super().__init__(mesh)

        self.beta = float(beta)

        x, y, z = mesh.X
        self._assemble(
            mesh,
            {_X: x, _Y: y, _Z: z, _BETA: sympy.Rational(self.beta)},
            ("x", "y", "z"),
        )


# SolKz transposes SolKx: the viscosity varies with depth, the modes run in x,
# and the kernel's u1 is the *vertical* velocity. Reading it with the SolCx
# convention would silently transpose the whole solution, so the mapping is
# spelled out from the kernel's own output section rather than assumed.
_SOLKZ_OUTPUTS = {
    "velocity_x": ("u2", sympy.sin),
    "velocity_z": ("u1", sympy.cos),
    "stress_xx": ("u6", sympy.cos),
    "stress_zz": ("u3", sympy.cos),
    "stress_zx": ("u4", sympy.sin),
    "pressure": ("u5", sympy.cos),
}


@functools.lru_cache(maxsize=None)
def _solkz_kernel():
    """Transcribe the Velic SolKz kernel into SymPy. One straight-line block."""

    source = CSource(os.path.join(_REFERENCE_DIR, "solKz.c"))
    body = source.function("_Velic_solKz")
    body = body[: body.index("rho =")]

    inputs = {"pos": (_X, _Z), "_sigma": sympy.Integer(1), "_km": _KM, "_n": _N, "_B": _B}
    scope = evaluate_block(body, inputs)

    kn = _N * sympy.pi
    return {
        field: scope[symbol] * mode(kn * _X)
        for field, (symbol, mode) in _SOLKZ_OUTPUTS.items()
    }


class SolKz(FreeSlipWalls, AnalyticSolution):
    r"""Stokes flow with a depth-dependent viscosity — the SolKz benchmark.

    Viscosity :math:`\eta = e^{2Bz}` on the unit box, free slip everywhere,
    forced by :math:`\mathbf f = (0,\; \sin(m\pi z)\cos(n\pi x))`.

    The vertical twin of :class:`SolKx`, and not a redundant one. A viscosity
    that varies with *depth* stratifies the flow along the direction the buoyancy
    acts, so the pressure and the vertical velocity are coupled through the
    varying coefficient in a way that a horizontal gradient never produces. It is
    also the closer analogue of a real mantle viscosity profile.

    Parameters
    ----------
    mesh : Mesh
        A 2D mesh on the unit box.
    B : float
        Viscosity exponent; the contrast across the box is :math:`e^{2B}`.
    n : int
        Horizontal wavenumber of the forcing.
    m : int
        Vertical wavenumber.

    Notes
    -----
    Validated by the equations rather than against a compiled kernel — the
    forcing and boundary conditions are known, so satisfying Stokes with them
    identifies the solution uniquely.
    """

    dim = 2
    stress_is_deviatoric = True
    reference = (
        "Velic. Transcribed from the published kernel vendored at "
        "underworld3/analytic/_reference/solKz.c."
    )
    eqn_viscosity = r"e^{2Bz}"
    eqn_bodyforce = r"(0,\; \sin(m \pi z)\cos(n \pi x))"

    def __init__(self, mesh, B=2.302585092994046, n=3, m=2):
        super().__init__(mesh)

        if int(n) != n or int(n) < 1:
            raise ValueError("n (horizontal wavenumber) must be a positive integer.")
        if int(m) != m or int(m) < 1:
            raise ValueError("m (vertical wavenumber) must be a positive integer.")

        self.B = float(B)
        self.n = int(n)
        self.m = int(m)

        x, z = mesh.X
        values = {
            _B: sympy.Rational(self.B),
            _N: self.n,
            _KM: self.m * sympy.pi,
            _X: x,
            _Z: z,
        }
        kernel = {
            field: expression.subs(values)
            for field, expression in _solkz_kernel().items()
        }

        self.fn_velocity = sympy.Matrix(
            [[kernel["velocity_x"], kernel["velocity_z"]]]
        )
        self.set_fields(
            velocity=(kernel["velocity_x"], kernel["velocity_z"]),
            pressure=kernel["pressure"],
            viscosity=sympy.exp(2 * sympy.Rational(self.B) * z),
            bodyforce=(
                0,
                sympy.sin(self.m * sympy.pi * z) * sympy.cos(self.n * sympy.pi * x),
            ),
            stress=(
                (kernel["stress_xx"], kernel["stress_zx"]),
                (kernel["stress_zx"], kernel["stress_zz"]),
            ),
        )


_SIGMA = sympy.Symbol("sigma")

# SolA and SolB share a kernel shape, and the source states it outright:
# "u1 = Vz, u2 = Vx, u3 = tzz, u4 = tzx, pp = pressure", with txx alongside.
# Modes run in x, as in SolKz. The stress is the TOTAL: u3 and txx both carry an
# explicit `- pp`.
_SOLAB_OUTPUTS = {
    "velocity_x": ("u2", sympy.sin),
    "velocity_z": ("u1", sympy.cos),
    "stress_xx": ("txx", sympy.cos),
    "stress_zz": ("u3", sympy.cos),
    "stress_zx": ("u4", sympy.sin),
    "pressure": ("pp", sympy.cos),
}


@functools.lru_cache(maxsize=None)
def _solab_kernel(name):
    """Transcribe SolA or SolB — one straight-line block, isoviscous."""

    source = CSource(os.path.join(_REFERENCE_DIR, f"{name}.c"))
    body = source.function(f"_Velic_{name}")
    body = body[: body.index("e_zz =")]

    inputs = {
        "pos": (_X, _Z),
        "sigma": _SIGMA,
        "Z": _ETA0,
        "n": _N,
        "km": _KM,
    }
    scope = evaluate_block(body, inputs)

    kn = _N * sympy.pi
    return {
        field: scope[symbol] * mode(kn * _X)
        for field, (symbol, mode) in _SOLAB_OUTPUTS.items()
    }


class _SolAB(FreeSlipWalls, AnalyticSolution):
    """Shared assembly for the two isoviscous Velic solutions."""

    dim = 2
    _kernel = None
    _vertical = None

    def __init__(self, mesh, sigma=1.0, eta=1.0, n=3, m=2):
        super().__init__(mesh)

        if int(n) != n or int(n) < 1:
            raise ValueError("n (horizontal wavenumber) must be a positive integer.")
        if float(eta) <= 0.0:
            raise ValueError("eta must be positive.")

        self.sigma = float(sigma)
        self.eta = float(eta)
        self.n = int(n)
        self.m = float(m)

        x, z = mesh.X
        values = {
            _SIGMA: sympy.Rational(self.sigma),
            _ETA0: sympy.Rational(self.eta),
            _N: self.n,
            _KM: sympy.Rational(self.m) * sympy.pi,
            _X: x,
            _Z: z,
        }
        kernel = {
            field: expression.subs(values)
            for field, expression in _solab_kernel(self._kernel).items()
        }

        self.set_fields(
            velocity=(kernel["velocity_x"], kernel["velocity_z"]),
            pressure=kernel["pressure"],
            viscosity=sympy.Rational(self.eta),
            bodyforce=(
                0,
                sympy.Rational(self.sigma)
                * self._vertical(sympy.Rational(self.m) * sympy.pi * z)
                * sympy.cos(self.n * sympy.pi * x),
            ),
            stress=(
                (kernel["stress_xx"], kernel["stress_zx"]),
                (kernel["stress_zx"], kernel["stress_zz"]),
            ),
        )


class SolA(_SolAB):
    r"""Isoviscous Stokes flow with a sinusoidal body force — the SolA benchmark.

    Constant viscosity on the unit box, free slip everywhere, forced by
    :math:`\mathbf f = (0,\; \sigma\sin(m\pi z)\cos(n\pi x))`.

    The simplest solution in the suite, and useful precisely for that: it removes
    the viscosity structure entirely, so a discrepancy here is in the
    discretisation or the solve rather than in how a hard coefficient is handled.
    Run it before concluding anything from SolCx or SolKx.

    Parameters
    ----------
    mesh : Mesh
        A 2D mesh on the unit box.
    sigma : float
        Forcing amplitude.
    eta : float
        The (constant) viscosity.
    n : int
        Horizontal wavenumber.
    m : float
        Vertical wavenumber; need not be an integer for the solution itself.
    """

    _kernel = "solA"
    _vertical = staticmethod(sympy.sin)
    reference = (
        "Velic. Transcribed from the published kernel vendored at "
        "underworld3/analytic/_reference/solA.c."
    )
    eqn_viscosity = r"\eta"
    eqn_bodyforce = r"(0,\; \sigma \sin(m \pi z)\cos(n \pi x))"


class SolB(_SolAB):
    r"""Isoviscous Stokes flow with a hyperbolic body force — the SolB benchmark.

    As :class:`SolA`, but the forcing grows with depth as
    :math:`\sinh(m\pi z)` rather than oscillating. The response is concentrated
    near one boundary instead of filling the box, which is a different test of
    the same machinery: it probes resolution where the solution is steep rather
    than accuracy where it is smooth.

    Parameters
    ----------
    mesh : Mesh
        A 2D mesh on the unit box.
    sigma : float
        Forcing amplitude.
    eta : float
        The (constant) viscosity.
    n : int
        Horizontal wavenumber.
    m : float
        Vertical decay parameter. Must differ from *n* — the kernel is singular
        when they coincide.
    """

    _kernel = "solB"
    _vertical = staticmethod(sympy.sinh)
    reference = (
        "Velic. Transcribed from the published kernel vendored at "
        "underworld3/analytic/_reference/solB.c."
    )
    eqn_viscosity = r"\eta"
    eqn_bodyforce = r"(0,\; \sigma \sinh(m \pi z)\cos(n \pi x))"

    def __init__(self, mesh, sigma=1.0, eta=1.0, n=3, m=2.0):
        if abs(float(n) - float(m)) < 1.0e-5:
            raise ValueError(
                "SolB is singular when the horizontal and vertical wavenumbers "
                "coincide; choose n != m."
            )
        super().__init__(mesh, sigma=sigma, eta=eta, n=n, m=m)


_KR = sympy.Symbol("kr")


@functools.lru_cache(maxsize=None)
def _solm_kernel():
    """Transcribe SolM. Short methods writing straight into the output array."""

    source = CSource(os.path.join(_REFERENCE_DIR, "AnalyticSolM.hpp"))

    # km, kn and kr are initialised in the class body rather than in any method,
    # so they are supplied rather than read.
    inputs = {"in": (_X, _Z), "eta0": _ETA0, "km": _KM, "kn": _KN, "kr": _KR}

    def block(method):
        return evaluate_block(source.function(method), inputs)

    velocity, bodyforce = block("velocity"), block("bodyforce")
    stress, strainrate = block("stress"), block("strainrate")

    return {
        "velocity_x": velocity["out[0]"],
        "velocity_z": velocity["out[1]"],
        "bodyforce_x": bodyforce["out[0]"],
        "bodyforce_z": bodyforce["out[1]"],
        "pressure": block("pressure")["p"],
        "viscosity": block("viscosity")["out[0]"],
        "stress_xx": stress["out[0]"],
        "stress_zz": stress["out[1]"],
        "stress_xz": stress["out[2]"],
        "strainrate_xx": strainrate["out[0]"],
        "strainrate_zz": strainrate["out[1]"],
        "strainrate_xz": strainrate["out[2]"],
    }


class SolM(FreeSlipWalls, AnalyticSolution):
    r"""Stokes flow with a laterally oscillating viscosity — the SolM benchmark.

    Viscosity :math:`\eta = 1 + \eta_0(1 + \cos(r\pi x))` on the unit box, free
    slip everywhere, with the body force chosen to make a simple sinusoidal
    velocity exact.

    The viscosity here *oscillates* rather than jumping (SolCx) or varying
    monotonically (SolKx, SolKz), and its wavelength is independent of the
    flow's. That makes it the one solution in the suite where the coefficient
    structure and the solution structure can be deliberately mismatched — set
    ``r`` incommensurate with ``n`` and every element sees a different viscosity
    profile, which is a sharper test of quadrature than a smooth gradient.

    Parameters
    ----------
    mesh : Mesh
        A 2D mesh on the unit box.
    eta_0 : float
        Amplitude of the viscosity oscillation.
    n : int
        Horizontal wavenumber of the flow.
    m : int
        Vertical wavenumber of the flow.
    r : float
        Wavenumber of the viscosity oscillation. Need not be an integer, and is
        most interesting when it is not commensurate with *n*.

    Notes
    -----
    The kernel names its wavenumbers the other way round — its ``m`` multiplies
    :math:`x` and its ``n`` multiplies :math:`z`. The parameters here follow the
    convention used across this suite, ``n`` horizontal and ``m`` vertical, and
    the mapping is done at construction.
    """

    dim = 2
    stress_is_deviatoric = True
    reference = (
        "Velic. Transcribed from the published kernel vendored at "
        "underworld3/analytic/_reference/AnalyticSolM.hpp."
    )
    eqn_velocity = r"(-\sin(n\pi x)\,m\pi\cos(m\pi z),\; \cos(n\pi x)\,n\pi\sin(m\pi z))"
    eqn_viscosity = r"1 + \eta_0\,(1 + \cos(r \pi x))"

    def __init__(self, mesh, eta_0=1.0, n=3, m=2, r=4.0):
        super().__init__(mesh)

        if int(n) != n or int(n) < 1:
            raise ValueError("n (horizontal wavenumber) must be a positive integer.")
        if int(m) != m or int(m) < 1:
            # Free slip on the horizontal walls needs sin(m pi z) to vanish at
            # z = 1, exactly as for SolKx.
            raise ValueError("m (vertical wavenumber) must be a positive integer.")

        self.eta_0 = float(eta_0)
        self.n = int(n)
        self.m = int(m)
        self.r = float(r)

        x, z = mesh.X
        values = {
            _ETA0: sympy.Rational(self.eta_0),
            _KM: self.n * sympy.pi,  # the kernel's km multiplies x
            _KN: self.m * sympy.pi,  # and its kn multiplies z
            _KR: sympy.Rational(self.r) * sympy.pi,
            _X: x,
            _Z: z,
        }
        kernel = {
            field: expression.subs(values)
            for field, expression in _solm_kernel().items()
        }

        self.set_fields(
            velocity=(kernel["velocity_x"], kernel["velocity_z"]),
            pressure=kernel["pressure"],
            viscosity=kernel["viscosity"],
            bodyforce=(kernel["bodyforce_x"], kernel["bodyforce_z"]),
            # The stress is DERIVED from the strain rate here rather than taken
            # from the kernel, because the kernel's is wrong. Its viscosity is
            # (1 + cos(kr x)) eta0 + 1, but its stress is 2 (eta - 1) edot: the
            # constant part is missing. Measured, not guessed — the difference
            # from 2 (eta - 1) edot is exactly zero, and using the published
            # stress leaves the momentum residual at 0.21 where deriving it gives
            # 1.7e-16. Everything else the kernel publishes is mutually
            # consistent; only this output is defective.
            strainrate=(
                (kernel["strainrate_xx"], kernel["strainrate_xz"]),
                (kernel["strainrate_xz"], kernel["strainrate_zz"]),
            ),
        )


_XC_C = sympy.Symbol("xc_solc")

# SolC accumulates over modes; the source labels each in the loop tail. Same
# transposed convention as SolA and SolKz — u1 is the vertical velocity.
_SOLC_OUTPUTS = {
    "velocity_x": ("u2", sympy.sin),
    "velocity_z": ("u1", sympy.cos),
    "stress_xx": ("u6", sympy.cos),
    "stress_zz": ("u3", sympy.cos),
    "stress_zx": ("u4", sympy.sin),
    "pressure": ("u5", sympy.cos),
}


@functools.lru_cache(maxsize=None)
def _solc_kernel(modes):
    r"""Transcribe the Velic SolC kernel, summing its Fourier series.

    Unlike the others this one is a truncated series: a step in density, resolved
    as ``modes`` cosine terms. The loop body is evaluated once per mode with the
    index bound to an integer, and the results summed in SymPy.

    The series is also why the body force here is the *truncated* step rather
    than an exact one. The kernel accumulates the density it actually used, and
    that is what the solution solves exactly; comparing against a sharp step
    instead would report the truncation error as though it were a defect.

    Returns
    -------
    dict
        Field name -> expression, including ``bodyforce_z`` for the resolved step.
    """

    source = CSource(os.path.join(_REFERENCE_DIR, "solC.c"))
    body = source.function("_Velic_solC")
    loop = CSource.loop_body(body, "for(n=1;n<nmodes;n++)")

    totals = {field: sympy.Integer(0) for field in _SOLC_OUTPUTS}
    density = sympy.Integer(0)

    for index in range(1, modes):
        scope = evaluate_block(
            loop,
            {
                "pos": (_X, _Z),
                "sigma": _SIGMA,
                "Z": _ETA0,
                "xc": _XC_C,
                "x": _X,
                "z": _Z,
                "n": index,
            },
        )
        k = index * sympy.pi
        for field, (symbol, mode) in _SOLC_OUTPUTS.items():
            totals[field] += scope[symbol] * mode(k * _X)
        density += scope["del_rho"] * sympy.cos(k * _X)

    # The n = 0 terms, applied after the loop exactly as the source does: the
    # pressure gains its constant mode first, and the normal stresses are then
    # completed from the finished pressure.
    density += _SIGMA * _XC_C
    totals["pressure"] += _SIGMA * _XC_C * (sympy.Rational(1, 2) - _Z)
    totals["stress_zz"] -= totals["pressure"]
    totals["stress_xx"] -= totals["pressure"]
    totals["bodyforce_z"] = density

    return totals


class SolC(FreeSlipWalls, AnalyticSolution):
    r"""Isoviscous flow driven by a dense column — the SolC benchmark.

    Constant viscosity on the unit box, free slip everywhere, driven by a density
    step: :math:`\sigma` for :math:`x < x_c` and zero beyond it.

    The forcing is discontinuous where SolCx's *viscosity* is, which makes the two
    a useful pair: SolCx asks whether a solver copes with a jump in the operator,
    SolC with a jump in the right-hand side. The response is smooth in both cases,
    so any trouble is in how the discontinuity is integrated.

    Parameters
    ----------
    mesh : Mesh
        A 2D mesh on the unit box.
    sigma : float
        Density contrast of the column.
    eta : float
        The (constant) viscosity.
    x_c : float
        Width of the dense column.
    modes : int
        Number of Fourier modes. The solution is exact for the step *as
        resolved*, so this sets how sharp the step is rather than how accurate
        the solution is — see Notes.

    Notes
    -----
    This is a truncated series, and :attr:`fn_bodyforce` is the resolved step
    rather than a sharp one. That is deliberate: the fields solve the problem
    with the density the kernel actually summed, so the pair is exact and the
    residual checks mean what they say. Raising *modes* sharpens the step and
    slows every evaluation, since the expression carries one term per mode.
    """

    dim = 2
    reference = (
        "Velic. Transcribed from the published kernel vendored at "
        "underworld3/analytic/_reference/solC.c."
    )
    eqn_viscosity = r"\eta"
    eqn_bodyforce = r"(0,\; \sigma \;\text{for}\; x < x_c,\; 0 \;\text{beyond})"

    def __init__(self, mesh, sigma=1.0, eta=1.0, x_c=0.5, modes=40):
        super().__init__(mesh)

        if float(eta) <= 0.0:
            raise ValueError("eta must be positive.")
        if not 0.0 < float(x_c) < 1.0:
            raise ValueError("x_c must lie strictly inside (0, 1).")
        if int(modes) != modes or int(modes) < 2:
            raise ValueError("modes must be an integer of at least 2.")

        self.sigma = float(sigma)
        self.eta = float(eta)
        self.x_c = float(x_c)
        self.modes = int(modes)

        x, z = mesh.X
        values = {
            _SIGMA: sympy.Rational(self.sigma),
            _ETA0: sympy.Rational(self.eta),
            _XC_C: sympy.Rational(self.x_c),
            _X: x,
            _Z: z,
        }
        kernel = {
            field: expression.subs(values)
            for field, expression in _solc_kernel(self.modes).items()
        }

        self.set_fields(
            velocity=(kernel["velocity_x"], kernel["velocity_z"]),
            pressure=kernel["pressure"],
            viscosity=sympy.Rational(self.eta),
            # The body force is MINUS the density. Every other kernel in this
            # family negates internally — they write `rho = -sigma*sin*cos` and
            # force with `+sigma*sin*cos` — but SolC accumulates the density
            # itself, so the negation happens here. Measured: as summed it leaves
            # the momentum residual at 1.8, negated at 1.6e-16.
            bodyforce=(0, -kernel["bodyforce_z"]),
            stress=(
                (kernel["stress_xx"], kernel["stress_zx"]),
                (kernel["stress_zx"], kernel["stress_zz"]),
            ),
        )


_ZC, _DX, _X0 = sympy.symbols("zc dx x0")

# SolDA uses the transposed convention of SolA and SolC: u1 is the vertical
# velocity and the modes run in x. Its pp and txx already carry their mode.
_SOLDA_MODED = {
    "velocity_x": ("u2", sympy.sin),
    "velocity_z": ("u1", sympy.cos),
    "stress_zz": ("u3", sympy.cos),
    "stress_zx": ("u4", sympy.sin),
}


@functools.lru_cache(maxsize=None)
def _solda_kernel(modes):
    r"""Transcribe the Velic SolDA kernel.

    The hardest of the family to read, and the only one combining every feature
    the others have separately: a truncated series (as SolC), a viscosity jump
    (as SolCx), and a rectangular forcing.

    Both of its ``if (z < zc)`` blocks branch on the same condition, so each mode
    is evaluated along one side and then the other and the two are combined into
    a single :class:`sympy.Piecewise` — the SolCx pattern, applied per mode.

    Returns
    -------
    dict
        Field name -> expression, including ``bodyforce_z`` for the resolved
        column.
    """

    source = CSource(os.path.join(_REFERENCE_DIR, "solDA.c"))
    body = source.function("_Velic_solDA")

    header = "for(n=1;n<nmodes;n++)"
    preamble = body[: body.index(header)]
    loop = CSource.loop_body(body, header)

    constants_a, constants_b, rest = CSource.branches(loop, "if (z < zc)")
    fields_a, fields_b, tail = CSource.branches(rest, "if (z < zc)")
    mode_preamble = loop[: loop.index("if (z < zc)")]

    inputs = {
        "pos": (_X, _Z),
        "_sigma": _SIGMA,
        "_eta_A": _ZA,
        "_eta_B": _ZB,
        "_z_c": _ZC,
        "_dx": _DX,
        "_x_0": _X0,
    }
    outer = evaluate_block(preamble, inputs)

    totals = {field: sympy.Integer(0) for field in _SOLDA_MODED}
    totals["pressure"] = sympy.Integer(0)
    totals["stress_xx"] = sympy.Integer(0)
    density = sympy.Integer(0)

    for index in range(1, modes):
        scope = evaluate_block(mode_preamble, {**outer, "n": index})

        sides = {}
        for side, (constants, fields) in (
            ("low", (constants_a, fields_a)),
            ("high", (constants_b, fields_b)),
        ):
            sides[side] = evaluate_block(
                tail, evaluate_block(fields, evaluate_block(constants, scope))
            )

        def across(symbol):
            return sympy.Piecewise(
                (sides["low"][symbol], _Z < _ZC), (sides["high"][symbol], True)
            )

        k = index * sympy.pi
        for field, (symbol, mode) in _SOLDA_MODED.items():
            totals[field] += across(symbol) * mode(k * _X)

        # pp and txx are formed in the tail and already carry their mode.
        totals["pressure"] += across("pp")
        totals["stress_xx"] += across("txx")
        density += scope["del_rho"] * sympy.cos(k * _X)

    # The n = 0 terms, exactly as the source appends them: a uniform column, and
    # a linear normal stress set to vanish at the viscosity interface.
    zero_mode = _SIGMA * _DX
    density += zero_mode
    linear = zero_mode * (_Z - _ZC)
    totals["stress_zz"] += linear
    totals["stress_xx"] += linear
    totals["pressure"] -= linear
    totals["bodyforce_z"] = density

    return totals


class SolDA(FreeSlipWalls, AnalyticSolution):
    r"""A dense column in a layered fluid — the SolDA benchmark.

    A rectangular density anomaly of width *dx* centred at *x_0*, in a fluid
    whose viscosity jumps from :math:`\eta_A` below :math:`z_c` to
    :math:`\eta_B` above. Free slip on the unit box.

    The most demanding solution in the family, and the only one that combines
    what the others test separately: a discontinuous forcing (as SolC), a
    discontinuous viscosity (as SolCx), and a truncated series. The
    discontinuities are perpendicular to each other, so a scheme that handles
    either alone still has to get their interaction right.

    Parameters
    ----------
    mesh : Mesh
        A 2D mesh on the unit box.
    sigma : float
        Density contrast of the column.
    eta_A, eta_B : float
        Viscosity below and above :math:`z_c`.
    z_c : float
        Height of the viscosity jump.
    dx : float
        Width of the dense column.
    x_0 : float
        Centre of the dense column.
    modes : int
        Number of Fourier modes. As with :class:`SolC` the fields are exact for
        the column *as resolved*, so this sets the sharpness of the column
        rather than the accuracy of the solution.

    Notes
    -----
    Each mode carries a ``Piecewise`` across the viscosity interface, so this is
    much the most expensive solution here to build: measured at 20 s for 8 modes
    and 47 s for 16, against 2.4 s for SolC at 40. The default is deliberately
    low for that reason — raise it when the sharpness of the column matters more
    than construction time, and note that the residuals are unchanged between 8
    and 16 modes, so they are measuring the transcription rather than the
    truncation.
    """

    dim = 2
    reference = (
        "Velic. Transcribed from the published kernel vendored at "
        "underworld3/analytic/_reference/solDA.c."
    )
    eqn_viscosity = r"\eta_A \;(z<z_c), \quad \eta_B \;(z \ge z_c)"
    eqn_bodyforce = r"(0,\; \sigma \;\text{in a column of width } dx \text{ at } x_0)"

    def __init__(
        self,
        mesh,
        sigma=1.0,
        eta_A=1.0,
        eta_B=10.0,
        z_c=0.75,
        dx=0.25,
        x_0=0.375,
        modes=8,
    ):
        super().__init__(mesh)

        if not (float(eta_A) > 0.0 and float(eta_B) > 0.0):
            raise ValueError("eta_A and eta_B must be positive.")
        if not 0.0 < float(z_c) < 1.0:
            raise ValueError("z_c must lie strictly inside (0, 1).")
        if int(modes) != modes or int(modes) < 2:
            raise ValueError("modes must be an integer of at least 2.")

        self.sigma = float(sigma)
        self.eta_A = float(eta_A)
        self.eta_B = float(eta_B)
        self.z_c = float(z_c)
        self.dx = float(dx)
        self.x_0 = float(x_0)
        self.modes = int(modes)

        x, z = mesh.X
        values = {
            _SIGMA: sympy.Rational(self.sigma),
            _ZA: sympy.Rational(self.eta_A),
            _ZB: sympy.Rational(self.eta_B),
            _ZC: sympy.Rational(self.z_c),
            _DX: sympy.Rational(self.dx),
            _X0: sympy.Rational(self.x_0),
            _X: x,
            _Z: z,
        }
        kernel = {
            field: expression.subs(values)
            for field, expression in _solda_kernel(self.modes).items()
        }

        self.set_fields(
            velocity=(kernel["velocity_x"], kernel["velocity_z"]),
            pressure=kernel["pressure"],
            viscosity=sympy.Piecewise(
                (sympy.Rational(self.eta_A), z < self.z_c),
                (sympy.Rational(self.eta_B), True),
            ),
            # Minus the density, as for SolC: this kernel accumulates rho itself.
            bodyforce=(0, -kernel["bodyforce_z"]),
            stress=(
                (kernel["stress_xx"], kernel["stress_zx"]),
                (kernel["stress_zx"], kernel["stress_zz"]),
            ),
        )
