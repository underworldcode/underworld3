r"""Curved-geometry Stokes solutions, wrapping the external `assess` library.

Kramer et al. (2021) give exact Stokes solutions in a cylindrical annulus and a
spherical shell, for smooth and delta-function density anomalies under free-slip
or zero-slip walls. They are the reference for benchmarking a solver on curved
boundaries, which nothing else in this suite covers — every other solution here
is posed on a box.

Unlike the rest of the suite these are **not SymPy**, and the difference is not
cosmetic. `assess` provides numeric callables, so a Kramer solution can be
compared against a solver but cannot be checked against the equations it claims
to solve: there is nothing to differentiate. It therefore declares
``symbolic = False``, exposes no ``fn_*``, and the conformance suite skips it on
that declaration rather than by name.

Treat it as an external oracle rather than a validated member of the family. It
is here so the curved-geometry benchmarks stop importing an undeclared
dependency, and so that when the dependency is missing the failure is a sentence
rather than a traceback.

Installation::

    pip install "underworld3[benchmarks]"      # or: pip install assess

References
----------
Kramer, S.C., Davies, D.R. & Wilson, C.R. (2021). Analytical solutions for
mantle flow in cylindrical and spherical shells. *Geoscientific Model
Development* 14, 1899-1919. doi:10.5194/gmd-14-1899-2021
"""

import numpy as np

from ._base import AnalyticSolution


_INSTALL_MESSAGE = (
    "The Kramer benchmark solutions need the 'assess' package, which "
    "Underworld3 does not depend on.\n"
    "    pip install \"underworld3[benchmarks]\"\n"
    "or  pip install assess\n"
    "See Kramer, Davies & Wilson (2021), doi:10.5194/gmd-14-1899-2021."
)


def _require_assess():
    """Import `assess`, or explain how to get it.

    Deferred to construction rather than to module import so that
    ``import underworld3`` works without it and
    :func:`underworld3.analytic.available` can still list the solution.
    """

    try:
        import assess
    except ImportError as missing:
        raise ImportError(_INSTALL_MESSAGE) from missing

    return assess


def assess_available():
    """Whether the optional `assess` dependency can be imported."""

    try:
        _require_assess()
    except ImportError:
        return False

    return True


class CylindricalStokes(AnalyticSolution):
    r"""Stokes flow in a cylindrical annulus, after Kramer et al. (2021).

    Four cases, chosen by *density* and *boundary*:

    ==========  ===========  ====================================================
    ``density`` ``boundary`` anomaly
    ==========  ===========  ====================================================
    ``"delta"`` ``"free"``   a delta-function density shell at radius ``r_int``
    ``"delta"`` ``"zero"``   the same, with zero-slip walls
    ``"smooth"`` ``"free"``  :math:`r^k` radial dependence, wavenumber ``n``
    ``"smooth"`` ``"zero"``  the same, with zero-slip walls
    ==========  ===========  ====================================================

    The delta cases are solved as two solutions, above and below the anomaly, and
    :meth:`evaluate` selects between them by radius — that discontinuity is the
    point of the benchmark, and flattening it would remove what is being tested.

    Parameters
    ----------
    mesh : Mesh
        A 2D annulus mesh.
    n : int
        Azimuthal wavenumber.
    k : int
        Radial wavenumber. Used by the smooth cases only.
    r_inner, r_outer : float
        Shell radii.
    r_int : float
        Radius of the delta-function anomaly. Delta cases only.
    density : {"delta", "smooth"}
    boundary : {"free", "zero"}

    Notes
    -----
    Built on the `assess` calls the curved-geometry benchmark scripts in
    ``docs/examples/`` already make — ``CylindricalStokesSolution*(n, sign,
    Rp=, Rm=, rp=, nu=, g=)`` with ``.velocity_cartesian`` and
    ``.pressure_cartesian``. The wrapper's own tests cover the missing-dependency
    path; the working path is only as good as those signatures.
    """

    dim = 2
    nonlinear = False
    symbolic = False
    solves = "stokes"
    requires = "assess"
    reference = (
        "Kramer, Davies & Wilson (2021), Geosci. Model Dev. 14, 1899-1919, "
        "doi:10.5194/gmd-14-1899-2021. Numeric, via the external 'assess' package."
    )
    boundaries = ("Upper", "Lower")

    def __init__(
        self,
        mesh,
        n=2,
        k=2,
        r_inner=0.5,
        r_outer=1.0,
        r_int=0.8,
        density="delta",
        boundary="free",
    ):
        super().__init__(mesh)

        if density not in ("delta", "smooth"):
            raise ValueError(f"density must be 'delta' or 'smooth'; got {density!r}")
        if boundary not in ("free", "zero"):
            raise ValueError(f"boundary must be 'free' or 'zero'; got {boundary!r}")

        assess = _require_assess()

        self.n = int(n)
        self.k = int(k)
        self.r_inner = float(r_inner)
        self.r_outer = float(r_outer)
        self.r_int = float(r_int)
        self.density = density
        self.boundary = boundary

        slip = "FreeSlip" if boundary == "free" else "ZeroSlip"
        family = getattr(
            assess, f"CylindricalStokesSolution{density.capitalize()}{slip}"
        )

        if density == "delta":
            shared = dict(
                Rp=self.r_outer, Rm=self.r_inner, rp=self.r_int, nu=1.0, g=-1.0
            )
            self._above = family(self.n, +1, **shared)
            self._below = family(self.n, -1, **shared)
        else:
            shared = dict(Rp=self.r_outer, Rm=self.r_inner, nu=1.0, g=1.0)
            self._above = family(self.n, self.k, **shared)
            self._below = self._above

    def _split(self, coords):
        """Indices of the points above and below the anomaly radius."""

        radius = np.linalg.norm(np.asarray(coords, dtype=float), axis=1)
        above = radius >= self.r_int
        return above, ~above

    def evaluate(self, field, coords):
        """Exact *field* at *coords*, choosing the branch by radius.

        Parameters
        ----------
        field : {"velocity", "pressure"}
        coords : array, shape (n, 2)
            Cartesian coordinates.

        Returns
        -------
        ndarray
            Shape ``(n, 2)`` for velocity, ``(n,)`` for pressure.
        """

        if field not in ("velocity", "pressure"):
            raise ValueError(
                f"{type(self).__name__} provides 'velocity' and 'pressure'; "
                f"got {field!r}"
            )

        coords = np.asarray(coords, dtype=float)
        above, below = self._split(coords)

        width = 2 if field == "velocity" else 1
        values = np.zeros((len(coords), width))

        for mask, solution in ((above, self._above), (below, self._below)):
            if not mask.any():
                continue
            call = getattr(solution, f"{field}_cartesian")
            values[mask] = np.array([call(point) for point in coords[mask]]).reshape(
                -1, width
            )

        return values if width == 2 else values[:, 0]

    def error(self, field, meshvar, norm="l2"):
        """Relative error of *meshvar* against the exact *field*.

        The mesh-variable coordinates are used directly, so this is the discrete
        :math:`\\ell_2` norm over nodes rather than an integral — `assess` gives
        no symbolic form to integrate.
        """

        if norm != "l2":
            raise ValueError(
                f"{type(self).__name__} has no symbolic form, so only the "
                f"discrete 'l2' norm is available; got {norm!r}"
            )

        from underworld3 import mpi

        coords = meshvar.coords
        exact = np.atleast_2d(self.evaluate(field, coords).T).T
        computed = meshvar.data.reshape(exact.shape)

        local = np.array(
            [((computed - exact) ** 2).sum(), (exact**2).sum()], dtype=float
        )
        difference, magnitude = mpi.comm.allreduce(local)

        return float(np.sqrt(difference / magnitude))

    def apply_boundary_conditions(self, solver):
        """Free-slip or zero-slip on both walls, as the case declares."""

        if self.boundary == "zero":
            for wall in self.boundaries:
                solver.add_dirichlet_bc((0.0, 0.0), wall)
            return

        for wall in self.boundaries:
            solver.add_rotated_freeslip_bc(0.0, wall)

        solver.petsc_use_pressure_nullspace = True
