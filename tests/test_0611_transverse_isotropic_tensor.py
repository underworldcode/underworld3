r"""The transverse-isotropic viscosity tensor weakens shear on the weak plane, and
only that.

``TransverseIsotropicFlowModel`` builds
:math:`2\eta_{ijkl} = 2\eta_0 I_{ijkl} - 2(\eta_0 - \eta_1)\Lambda_{ijkl}`
(Moresi & Muhlhaus, 2006). Its docstring once printed the correction with a
``+`` sign, which would make the resolved shear on the plane
:math:`2(2\eta_0 - \eta_1)\dot\varepsilon_{nt}` -- stiffer than the bulk.
The baselines here are exact: a pure plane shear gives
:math:`\tau_{nt} = 2\eta_1 \dot\varepsilon_{nt}`, and a deviatoric stretch
normal to the plane sees only :math:`\eta_0`.
"""
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]

GAMMA = sympy.Symbol(r"\gamma", positive=True)

# A director that is not a coordinate axis, so every tensor entry takes part,
# and one in-plane unit tangent.
_DIRECTORS = {
    2: (sympy.Matrix([sympy.Rational(3, 5), sympy.Rational(4, 5)]),
        sympy.Matrix([-sympy.Rational(4, 5), sympy.Rational(3, 5)])),
    3: (sympy.Matrix([2, 3, 6]) / 7,
        sympy.Matrix([3, -2, 0]) / sympy.sqrt(13)),
}


def _stress(dim, strain_rate):
    """Contract the model's tensor with a strain rate, tau_ij = c_ijkl D_kl.

    Returns the stress and the model's own eta_0 and eta_1 expressions, which are
    the atoms baked into the tensor, so the comparison is exact and symbolic.
    """
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0,) * dim,
                                             maxCoords=(1.0,) * dim, cellSize=0.5)
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
    P = stokes.constitutive_model.Parameters
    P.shear_viscosity_0 = 1.0
    P.shear_viscosity_1 = 0.1
    P.director = _DIRECTORS[dim][0]

    c = stokes.constitutive_model.c
    tau = sympy.Matrix(dim, dim, lambda i, j: sum(
        c[i, j, k, l] * strain_rate[k, l] for k in range(dim) for l in range(dim)))
    return tau, P.shear_viscosity_0, P.shear_viscosity_1


@pytest.mark.parametrize("dim", [2, 3])
def test_shear_on_the_weak_plane_sees_eta_1(dim):
    n, t = _DIRECTORS[dim]
    strain_rate = GAMMA / 2 * (n * t.T + t * n.T)
    tau, _, eta_1 = _stress(dim, strain_rate)

    edot_nt = (n.T * strain_rate * t)[0]
    tau_nt = (n.T * tau * t)[0]
    assert sympy.simplify(edot_nt - GAMMA / 2) == 0
    assert sympy.simplify(tau_nt - 2 * eta_1 * edot_nt) == 0
    # The stress is pure plane shear too: no other component is excited.
    assert sympy.simplify(tau - 2 * eta_1 * strain_rate) == sympy.zeros(dim, dim)


@pytest.mark.parametrize("dim", [2, 3])
def test_stretch_normal_to_the_plane_sees_eta_0(dim):
    n, t = _DIRECTORS[dim]
    strain_rate = GAMMA / 2 * (n * n.T - t * t.T)
    tau, eta_0, _ = _stress(dim, strain_rate)
    assert sympy.simplify(tau - 2 * eta_0 * strain_rate) == sympy.zeros(dim, dim)
