"""The constrained free-slip annulus of test_1063 converges to its exact solution.

The case (tests/parallel/test_1063_constrained_freeslip_parallel.py, ``iso``): annulus
:math:`r \\in [1/2, 1]`, viscosity 1, body force :math:`100 \\sin 3\\theta\\,\\hat r`,
zero slip at :math:`r = 1/2`, free slip (the ``Stokes_Constrained`` multiplier) at
:math:`r = 1`. It is a single Fourier mode, so it has an exact solution. With the stream
function :math:`\\psi = \\Psi(r) \\cos 3\\theta` (:math:`u_r = r^{-1}\\partial_\\theta\\psi`,
:math:`u_\\theta = -\\partial_r\\psi`), the pressure eliminated through the
:math:`\\theta` momentum equation, the :math:`r` momentum equation becomes a fourth-order
Euler equation whose homogeneous powers are :math:`r^5, r^3, r^{-1}, r^{-3}`. The
forcing is resonant with :math:`r^3`, so the particular solution is
:math:`\\tfrac{25}{4} r^3 \\log r`. The four constants come from :math:`\\Psi = \\Psi' = 0`
at :math:`r = 1/2`, and :math:`\\Psi = 0`, :math:`(u_\\theta / r)' = 0` at :math:`r = 1`.
:func:`_exact` carries this out with sympy and returns

* velocity :math:`\\|u\\|_{L^2} = 0.616928007786`,
* mean-stripped topography :math:`\\|\\sigma_{rr} - \\bar\\sigma_{rr}\\|_{L^2(r=1)} =
  37.7800672892` (amplitude :math:`|\\sigma_{rr}| = 21.3151` of :math:`\\sin 3\\theta`).

The topography is ``solver.topography()``, the full boundary traction
:math:`h + r(\\mathbf{u}\\cdot\\mathbf{n} - g)` (#617). Measured 2026-11-20 on a Linux
workstation (gmsh: 1650 and 6268 cells), relative errors:

==========  ==========  =============  ==========
cellSize    velocity    topography     bare h
==========  ==========  =============  ==========
0.12        4.1e-3      6.8e-3         2.1e-3
0.06        1.55e-3     7.4e-4         7.1e-4
0.03        4.5e-4      1.9e-4         2.2e-4
==========  ==========  =============  ==========

The test runs the two finer meshes and asserts each error against the exact value and
the observed order between them (velocity 1.8, topography 2.0, both asserted >= 1.6).
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2]


def _exact():
    """(velocity L2, mean-stripped topography L2) of the exact solution."""
    r = sympy.symbols("r", positive=True)
    n, F0 = 3, 100
    c = sympy.symbols("c0:4")

    def laplacian(g):   # scalar Laplacian of g(r) times sin or cos(n theta)
        return sympy.diff(g, r, 2) + sympy.diff(g, r) / r - n**2 * g / r**2

    def fields(Psi):
        U = -n * Psi / r                       # u_r = U sin(n theta)
        V = -sympy.diff(Psi, r)                # u_theta = V cos(n theta)
        # theta momentum gives the pressure P sin(n theta)
        P = r / n * (laplacian(V) - V / r**2 + 2 * n * U / r**2)
        residual = -sympy.diff(P, r) + laplacian(U) - U / r**2 + 2 * n * V / r**2 + F0
        return U, V, P, residual

    Psi = (c[0] * r**3 + c[1] * r**-3 + c[2] * r**5 + c[3] * r**-1
           + sympy.Rational(25, 4) * r**3 * sympy.log(r))
    inner, outer = sympy.Rational(1, 2), sympy.Integer(1)
    U, V, P, _ = fields(Psi)
    constants = sympy.solve([Psi.subs(r, inner), sympy.diff(Psi, r).subs(r, inner),
                             Psi.subs(r, outer),
                             (r * sympy.diff(V / r, r)).subs(r, outer)], c)
    U, V, P, residual = (e.subs(constants) for e in fields(Psi))
    assert sympy.simplify(residual) == 0
    velocity = sympy.sqrt(sympy.pi * sympy.integrate(sympy.expand(U**2 + V**2) * r,
                                                     (r, inner, outer)))
    sigma_rr = (-P + 2 * sympy.diff(U, r)).subs(r, outer)
    return float(velocity), float(abs(sigma_rr) * sympy.sqrt(sympy.pi))


def _solve(cell_size):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=cell_size, qdegree=4)
    v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    X = mesh.CoordinateSystem.X
    st = uw.systems.Stokes_Constrained(mesh, velocityField=v, pressureField=p)
    st.constitutive_model = uw.constitutive_models.ViscousFlowModel
    st.constitutive_model.Parameters.shear_viscosity_0 = 1
    st.tolerance = 1.0e-8
    st.add_essential_bc((0.0, 0.0), "Lower")
    st.add_constraint_bc(0.0, "Upper")
    st.bodyforce = 100 * sympy.sin(3 * sympy.atan2(X[1], X[0])) * mesh.CoordinateSystem.unit_e_0
    st.solve(zero_init_guess=True)
    velocity = float(np.sqrt(uw.maths.Integral(mesh, v.sym.dot(v.sym)).evaluate()))
    topography = st.topography("Upper", reference="mean")
    topography = float(np.sqrt(uw.maths.BdIntegral(
        mesh=mesh, fn=topography**2, boundary="Upper").evaluate()))
    return velocity, topography


@pytest.mark.tier_b
def test_the_exact_solution_is_the_recorded_one():
    """The derivation reproduces the values quoted in the module docstring."""
    velocity, topography = _exact()
    assert velocity == pytest.approx(0.616928007786226, rel=1e-13)
    assert topography == pytest.approx(37.7800672891751, rel=1e-13)


@pytest.mark.tier_b
def test_constrained_freeslip_annulus_converges_to_the_exact_solution():
    exact = np.array(_exact())
    errors = [np.abs(np.array(_solve(h)) - exact) / exact for h in (0.06, 0.03)]
    order = np.log2(errors[0] / errors[1])
    # Bounds: twice the measured errors at cellSize 0.03 (module docstring).
    assert errors[1][0] < 9.0e-4, f"velocity error {errors[1][0]:.2e} at cellSize 0.03"
    assert errors[1][1] < 4.0e-4, f"topography error {errors[1][1]:.2e} at cellSize 0.03"
    assert np.all(order >= 1.6), f"observed orders (velocity, topography): {order}"
