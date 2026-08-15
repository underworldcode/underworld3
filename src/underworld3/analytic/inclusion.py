r"""A deformable elliptical inclusion in general shear.

Schmid, D. W. & Podladchikov, Y. Y. (2003), "Analytical solutions for deformable
elliptical inclusions in general shear", *Geophysical Journal International*
**155**(1), 269–288, doi:10.1046/j.1365-246X.2003.02042.x.

An ellipse of one viscosity sitting in a matrix of another, with a far-field pure
and/or simple shear. It is the natural test for anything that has to represent a
strong material contrast on a curved interface — a weak inclusion, a clast, a
fault-adjacent lens — because:

* the velocity and pressure are closed form both **inside and outside**, so a
  computed field can be compared point by point rather than only through a
  summary like "is the interior strain uniform";
* it is derived for incompressible viscous flow via Muskhelishvili complex
  potentials, so nothing has to be translated from an elastic result;
* there is no restriction on the viscosity ratio.

There is no body force. The flow is driven entirely by the far field, which makes
this a test of how a solver handles the contrast rather than of how it handles
forcing.

Provenance
----------
Transcribed from the authors' reference MATLAB, ``ell_dynamix.m`` and
``ell_rot_rate.m`` in ``github.com/dwschmid/muskhelishvili`` (BSD-3-Clause).

Those scripts give pressure, deviatoric stress and the rotation rate, but **not
the velocity**, which has to be reconstructed from the potentials. They are
recoverable from what is published, and the reading is checked rather than
assumed — see :func:`_potentials`.

"""

import functools

import sympy

from ._base import AnalyticSolution, FixedWalls


def _conjugate(expression):
    r"""Complex conjugate of an expression built from real symbols and ``I``.

    :func:`sympy.conjugate` will not distribute through ``sqrt`` of a symbolic
    argument — it has a branch cut, so SymPy leaves ``conjugate(sqrt(...))``
    unevaluated, and anything built on it can no longer be differentiated into
    printable code.

    Flipping the sign of ``I`` does the same job here and does it structurally:
    for a function of real variables assembled from real symbols and ``I``,
    :math:`\overline{f(z)} = f(\bar z)`, and the sign flip is exactly that.
    Verified against :func:`numpy.conj` on both sides of the branch cut.
    """

    return expression.subs(sympy.I, -sympy.I)


@functools.lru_cache(maxsize=None)
def _psi_shape():
    r"""The :math:`\zeta`-dependence of :math:`\psi`, integrated once.

    :math:`\mathrm d\psi/\mathrm d\zeta = \psi'(z)\,\mathrm dz/\mathrm d\zeta`
    splits into a far-field part, which integrates by inspection, and

    .. math::
        \int \frac{3\zeta^2 - 1}{\zeta^2 (\zeta^2-1)^2}\,\mathrm d\zeta

    which is a rational function with integer coefficients. Integrating it once
    with a bare symbol keeps SymPy in an exact domain; substituting the numeric
    constants first puts it in a floating complex polynomial ring, where the
    division algorithm cannot detect zero and the integration fails outright.
    """

    zeta = sympy.Symbol("_zeta")
    integrand = (3 * zeta**2 - 1) / (zeta**2 * (zeta**2 - 1) ** 2)

    return zeta, sympy.integrate(integrand, zeta)


def _shape_ratio(aspect_ratio):
    r"""The conformal radius :math:`r_c` for an ellipse of the given aspect ratio.

    The map :math:`z = \zeta + 1/\zeta` takes the circle :math:`|\zeta| = r_c` to
    an ellipse with semi-axes :math:`r_c \pm 1/r_c`, so the aspect ratio is
    :math:`(r_c^2+1)/(r_c^2-1)`. Inverting that is the relation the reference
    MATLAB writes as ``rc = sqrt((t-1)*(t+1))/(t-1)``.
    """

    t = sympy.sympify(aspect_ratio)
    return sympy.sqrt((t - 1) * (t + 1)) / (t - 1)


def _potentials(zeta, viscosity_ratio, aspect_ratio, alpha, pure_shear, simple_shear):
    r"""The Muskhelishvili potentials for the matrix, and the interior constants.

    The published scripts give the fields, not the potentials, so these are read
    back out of them:

    * the matrix pressure is :math:`p = -2\,\mathrm{Re}\,\varphi'(z)`, which fixes
      :math:`\varphi'(z) = A/(\zeta^2-1)`;
    * the matrix stress is :math:`\bar z\,\varphi''(z) + \psi'(z)`, and its first
      term must then reproduce :math:`\varphi''` derived from that same
      :math:`\varphi'`.

    The second point is a real check rather than a restatement: it is an
    independent expression in the source, and it agrees identically. Any error in
    reading :math:`\varphi` off the pressure would show up there.

    Returns
    -------
    dict
        ``phi``, ``phi_prime``, ``psi_prime``, ``psi`` as functions of
        :math:`\zeta`, plus the interior pressure and deviatoric stress.
    """

    mc = sympy.sympify(viscosity_ratio)
    rc = _shape_ratio(aspect_ratio)
    er = sympy.sympify(pure_shear)
    gr = sympy.sympify(simple_shear)

    # Far field, as the reference writes it.
    BC = (2 * er - sympy.I * gr) * sympy.exp(2 * sympy.I * sympy.sympify(alpha))
    ReBC, ImBC = sympy.re(BC), sympy.im(BC)

    B1 = rc**4 * mc + rc**4 - 1 + mc
    B2 = rc**4 * mc + rc**4 - mc + 1
    B3 = rc**4 * mc - mc - rc**4 + 1
    B4 = -(rc**4) * mc - mc - rc**4 + 1
    B5 = rc**8 * mc - mc - rc**8 + 1

    D = sympy.I * ImBC / B1 - ReBC / B2
    A = -(rc**2) * B3 * D

    # A purely imaginary constant in phi' is invisible to the published data:
    # the pressure is -2 Re[phi'] and the stress involves phi'', so neither sees
    # it. It is a far-field rigid rotation, and without it the reconstructed flow
    # has the right strain but no spin — a simple shear comes out as pure shear.
    #
    # Its value is fixed by the reference's own rotation rate: taken to a circle
    # the expression collapses to -gr/2 for every viscosity ratio, which is the
    # statement that a circular inclusion turns with the far field. Vorticity is
    # frame invariant, so alpha does not enter, and pure shear contributes none.
    far_field_spin = -gr / 2

    phi_prime = A / (zeta**2 - 1) + sympy.I * far_field_spin
    phi = -A / zeta + sympy.I * far_field_spin * (zeta + 1 / zeta)

    psi_prime = -BC - B5 * D * (3 * zeta**2 - 1) / (zeta**2 - 1) ** 3

    # psi needs integrating in zeta, since d/dz = (d/dzeta)/(dz/dzeta). The
    # far-field part integrates by inspection; the rest is the cached shape.
    symbol, shape = _psi_shape()
    psi = -BC * (zeta + 1 / zeta) - B5 * D * shape.subs(symbol, zeta)

    # Inside the inclusion both are uniform — this is the Eshelby property, and
    # it is what makes the interior a sharp test.
    interior_pressure = sympy.re(
        -sympy.I * mc * B4 / B1 * gr
        + 2 * rc**2 * (mc - 1) * (sympy.I * mc * ImBC / B1 - ReBC / B2)
    )
    interior_stress = -2 * mc * rc**4 * (sympy.I * ImBC / B1 + ReBC / B2)

    return {
        "phi": phi,
        "phi_prime": phi_prime,
        "psi_prime": psi_prime,
        "psi": psi,
        "interior_pressure": interior_pressure,
        "interior_stress": interior_stress,
        "rc": rc,
    }


class EllipticalInclusion(FixedWalls, AnalyticSolution):
    r"""A viscous elliptical inclusion in a matrix under general shear.

    Parameters
    ----------
    mesh : Mesh
        A 2D mesh. The inclusion is placed at *centre*, so the domain does not
        have to be the unit box.
    viscosity_ratio : float
        Inclusion viscosity over matrix viscosity. Any positive value; a weak
        inclusion is a ratio below one.
    aspect_ratio : float
        Long axis over short axis, strictly greater than one. Use something like
        1.001 for a near-circular inclusion — exactly one is a degenerate
        conformal map.
    alpha : float
        Angle of the far-field flow relative to the inclusion's long axis, which
        lies along *x*.
    pure_shear, simple_shear : float
        Far-field rates. The reference combines them as
        :math:`(2\dot\epsilon - i\dot\gamma)e^{2i\alpha}`.
    centre : tuple of float
        Position of the inclusion centre.
    semi_major : float
        Physical length of the long semi-axis.
    matrix_viscosity : float
        Viscosity of the matrix. Stresses scale with it; the velocity does not.

    Notes
    -----
    The inclusion **rotates**, so this is an instantaneous solution: it describes
    the flow at one moment, not a history. :attr:`rotation_rate` is the angular
    velocity, and it is an independent scalar to check a solve against — it comes
    from a separate published expression, not from these fields.

    The velocity is expressed through :func:`sympy.re` and :func:`sympy.im` of a
    complex potential. That evaluates and differentiates correctly, which is what
    the validation needs, but it has not been exercised through the JIT — so
    treat using this as a Dirichlet boundary value in a solver as unverified.
    """

    dim = 2
    reference = (
        "Schmid & Podladchikov (2003), Geophys. J. Int. 155(1), 269-288, "
        "doi:10.1046/j.1365-246X.2003.02042.x. Transcribed from the authors' "
        "reference MATLAB (github.com/dwschmid/muskhelishvili, BSD-3-Clause)."
    )
    eqn_viscosity = r"\mu_c \text{ inside},\quad \mu_m \text{ outside}"
    eqn_bodyforce = r"\mathbf 0 \quad(\text{driven by the far field})"

    def __init__(
        self,
        mesh,
        viscosity_ratio=1.0e3,
        aspect_ratio=2.0,
        alpha=0.0,
        pure_shear=0.0,
        simple_shear=1.0,
        centre=(0.0, 0.0),
        semi_major=1.0,
        matrix_viscosity=1.0,
    ):
        super().__init__(mesh)

        if float(aspect_ratio) <= 1.0:
            raise ValueError(
                "aspect_ratio must exceed 1; the conformal map is degenerate at "
                "a circle. Use 1.001 for a near-circular inclusion."
            )
        if float(viscosity_ratio) <= 0.0 or float(matrix_viscosity) <= 0.0:
            raise ValueError("viscosities must be positive.")

        self.viscosity_ratio = float(viscosity_ratio)
        self.aspect_ratio = float(aspect_ratio)
        self.alpha = float(alpha)
        self.pure_shear = float(pure_shear)
        self.simple_shear = float(simple_shear)
        self.matrix_viscosity = float(matrix_viscosity)

        rc = _shape_ratio(self.aspect_ratio)

        # Build on real-declared symbols and substitute the mesh coordinates at
        # the end. Mesh coordinates carry no reality assumption, so SymPy cannot
        # push re/im through a conjugate or a square root of them: the split is
        # left symbolic, and differentiating it produces a Derivative(re(...))
        # that no code printer can emit.
        u, v = sympy.symbols("_u _v", real=True)

        # The reference solution lives on the unit-focal ellipse, whose long
        # semi-axis is rc + 1/rc. Lengths scale, so map the physical point in and
        # scale the velocity back out; stress and pressure are scale invariant.
        self._scale = sympy.sympify(semi_major) / (rc + 1 / rc)
        z = (u + sympy.I * v) / self._scale

        # Invert z = zeta + 1/zeta on the branch outside the unit circle, which
        # is the one that maps the matrix to |zeta| > rc.
        #
        # Written as sqrt(z-2) sqrt(z+2) rather than sqrt(z^2-4): the two agree
        # in modulus but not in branch. The single square root cuts along a ray,
        # so for z left of the origin it selects the root inside the unit circle
        # — the wrong sheet — and the far field comes out asymmetric. The split
        # form cuts along the segment [-2, 2], which is the slit the map already
        # has, and gives |zeta| > 1 everywhere outside it.
        zeta = (z + sympy.sqrt(z - 2) * sympy.sqrt(z + 2)) / 2

        potentials = _potentials(
            zeta,
            self.viscosity_ratio,
            self.aspect_ratio,
            self.alpha,
            self.pure_shear,
            self.simple_shear,
        )

        # 2 mu (v_x + i v_y) = kappa phi - z conj(phi') - conj(psi), and kappa = 1
        # for an incompressible medium (the elastic 3 - 4 nu at nu = 1/2).
        outside = (
            potentials["phi"]
            - z * _conjugate(potentials["phi_prime"])
            - _conjugate(potentials["psi"])
        ) / 2

        # Real and imaginary parts without re()/im(), which cannot be pushed
        # through the branch cut. These are real-valued but complex-typed: SymPy
        # cannot prove the imaginary part vanishes, though it does to roundoff.
        outside_x = self._scale * (outside + _conjugate(outside)) / 2
        outside_y = self._scale * (outside - _conjugate(outside)) / (2 * sympy.I)

        # Inside, the velocity gradient is uniform — the Eshelby property, and
        # what makes this benchmark sharp. It is fixed by the interior deviatoric
        # stress and the rotation rate, both already known.
        #
        # The reference writes stress as (sigma_yy - sigma_xx)/2 + i sigma_xy, so
        # with a traceless deviator tau_yy = Re[T], tau_xx = -Re[T], tau_xy =
        # Im[T]. Dividing by 2 mu_c gives the strain rate, and mu_m cancels: the
        # interior strain rate depends on the viscosity *ratio* only.
        T = potentials["interior_stress"]
        rate = 2 * self.viscosity_ratio
        exx, eyy = -sympy.re(T) / rate, sympy.re(T) / rate
        exy = sympy.im(T) / rate
        spin = self.rotation_rate

        inside_x = exx * u + (exy - spin) * v
        inside_y = (exy + spin) * u + eyy * v

        x, y = mesh.X
        inside = sympy.Abs(zeta) < rc
        physical = {u: x - centre[0], v: y - centre[1]}

        self.fn_velocity = sympy.Matrix(
            [
                [
                    sympy.Piecewise((inside_x, inside), (outside_x, True)).subs(physical),
                    sympy.Piecewise((inside_y, inside), (outside_y, True)).subs(physical),
                ]
            ]
        )
        # p = -2 Re[phi'], written without re() so it can be differentiated.
        matrix_pressure = -(
            potentials["phi_prime"] + _conjugate(potentials["phi_prime"])
        )
        self.fn_pressure = sympy.Piecewise(
            (potentials["interior_pressure"], inside),
            (matrix_pressure, True),
        ).subs(physical)
        self.fn_viscosity = sympy.Piecewise(
            (self.matrix_viscosity * self.viscosity_ratio, inside),
            (self.matrix_viscosity, True),
        ).subs(physical)
        self.fn_bodyforce = sympy.Matrix([[0, 0]])

        # Strain rate and stress follow from the velocity and pressure above.
        # They are consistent with those rather than independent of them, so they
        # are not a check on the reconstruction — the checks are the Stokes
        # residual, velocity continuity across the interface, and the far field.
        strain = sympy.Matrix(
            [
                [
                    (
                        sympy.diff(self.fn_velocity[0, i], mesh.X[j])
                        + sympy.diff(self.fn_velocity[0, j], mesh.X[i])
                    )
                    / 2
                    for j in range(2)
                ]
                for i in range(2)
            ]
        )
        self.fn_strainrate = strain
        self.fn_stress = (
            2 * self.fn_viscosity * strain - self.fn_pressure * sympy.eye(2)
        )

        self._potentials = potentials
        self._centre = tuple(float(c) for c in centre)

    @property
    def semi_axes(self):
        """Long and short semi-axes of the inclusion, in physical units."""

        rc = float(_shape_ratio(self.aspect_ratio))
        scale = float(self._scale)
        return scale * (rc + 1 / rc), scale * (rc - 1 / rc)

    def sample_points(self, count=12):
        """Points in the matrix, clear of the inclusion and of the map's foci.

        The default unit-box sampler is wrong for this solution twice over: the
        inclusion is centred wherever the caller put it rather than filling the
        box, and the conformal map has branch points at the foci, where the
        fields are singular. A generic sampler lands on both.

        Points are laid on rings outside the inclusion, so they exercise the
        matrix solution at a range of distances and azimuths without straddling
        the interface — across which the stress is discontinuous, so a residual
        evaluated there is meaningless rather than merely inaccurate.
        """

        import numpy as np

        a, b = self.semi_axes
        rings = np.linspace(1.6, 4.0, max(count // 4, 2))
        angles = np.linspace(0.13, 2 * np.pi + 0.13, 4, endpoint=False)

        return np.array(
            [
                (
                    self._centre[0] + factor * a * np.cos(theta),
                    self._centre[1] + factor * b * np.sin(theta),
                )
                for factor in rings
                for theta in angles
            ]
        )

    @property
    def rotation_rate(self):
        r"""Angular velocity of the inclusion.

        A scalar oracle independent of the fields above: it comes from the
        reference's own closed form (``ell_rot_rate.m``), so comparing a computed
        interior vorticity against it is a genuine check rather than a
        restatement.
        """

        t = self.aspect_ratio
        mc = self.viscosity_ratio
        alpha = self.alpha

        return float(
            (
                -0.5
                * (t**2 - mc * t**2 + mc - 1)
                / (mc * t**2 + mc + 2 * t)
                * sympy.cos(2 * alpha)
                - 0.5
            )
            * self.simple_shear
            - 0.5
            * (2 * mc * t**2 - 2 * t**2 - 2 * mc + 2)
            / (mc * t**2 + mc + 2 * t)
            * sympy.sin(2 * alpha)
            * self.pure_shear
        )
