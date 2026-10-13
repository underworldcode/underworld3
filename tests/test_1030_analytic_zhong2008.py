r"""Propagator-matrix responses from Zhong et al. (2008).

The values in parentheses in Tables 2 and 3 are the semi-analytic propagator
solutions, not the CitcomS finite-element results.  This test pins every
published propagator row so a sign, stress scaling, layer-order, or self-gravity
regression cannot hide behind agreement with one selected case.

Run: pixi run python -m pytest tests/test_1030_analytic_zhong2008.py -v
"""

import numpy as np
import pytest
import underworld3 as uw


pytestmark = pytest.mark.level_2


# depth fraction, degree, surface topography, CMB topography, surface geoid,
# CMB geoid, surface characteristic velocity, CMB characteristic velocity,
# surface velocity divergence, CMB velocity divergence.  These are the
# parenthesized propagator values in Zhong et al. (2008), Tables 2 and 3.
ISOVISCOUS_TABLE_2 = [
    (
        0.25,
        2,
        0.7716,
        0.5646,
        0.04058,
        0.04062,
        -9.949e-3,
        7.709e-3,
        5.969e-2,
        -8.410e-2,
    ),
    (
        0.25,
        5,
        0.7562,
        0.3894,
        0.02986,
        0.01556,
        -4.546e-3,
        2.134e-3,
        1.364e-1,
        -1.164e-1,
    ),
    (
        0.25,
        8,
        0.6458,
        0.1629,
        0.02018,
        0.004454,
        -2.128e-3,
        4.480e-4,
        1.532e-1,
        -5.865e-2,
    ),
    (
        0.25,
        15,
        0.3905,
        0.01094,
        0.008355,
        0.0001739,
        -5.074e-4,
        1.044e-5,
        1.218e-1,
        -4.557e-3,
    ),
    (
        0.50,
        2,
        0.4998,
        0.9313,
        0.04486,
        0.05461,
        -1.006e-2,
        1.186e-2,
        6.038e-2,
        -1.294e-1,
    ),
    (
        0.50,
        5,
        0.4238,
        0.7235,
        0.02427,
        0.02543,
        -3.593e-3,
        3.733e-3,
        1.078e-1,
        -2.036e-1,
    ),
    (
        0.50,
        8,
        0.2679,
        0.3794,
        0.01122,
        0.009474,
        -1.170e-3,
        9.890e-4,
        8.427e-2,
        -1.295e-1,
    ),
    (
        0.50,
        15,
        0.06905,
        0.05554,
        0.001804,
        0.0008399,
        -1.091e-4,
        5.095e-5,
        2.618e-2,
        -2.223e-2,
    ),
    (
        0.75,
        2,
        0.2323,
        1.090,
        0.02788,
        0.04267,
        -5.486e-3,
        1.046e-2,
        3.291e-2,
        -1.141e-1,
    ),
    (
        0.75,
        5,
        0.1664,
        1.008,
        0.01143,
        0.02741,
        -1.600e-3,
        4.201e-3,
        4.801e-2,
        -2.292e-1,
    ),
    (
        0.75,
        8,
        0.07634,
        0.7471,
        0.003644,
        0.01542,
        -3.707e-4,
        1.637e-3,
        2.669e-2,
        -2.143e-1,
    ),
    (
        0.75,
        15,
        0.007289,
        0.3006,
        0.0002061,
        0.004023,
        -1.238e-5,
        2.453e-4,
        2.973e-3,
        -1.071e-1,
    ),
]

LAYERED_TABLE_3 = [
    (
        0.25,
        2,
        0.8798,
        0.07981,
        0.05333,
        -0.006165,
        -1.231e-5,
        1.543e-3,
        7.384e-5,
        -1.683e-2,
    ),
    (
        0.25,
        5,
        0.7976,
        0.1254,
        0.03326,
        0.002543,
        -1.964e-6,
        7.093e-4,
        5.893e-5,
        -3.869e-2,
    ),
    (
        0.25,
        8,
        0.6913,
        0.07955,
        0.02284,
        0.001778,
        -6.509e-7,
        2.191e-4,
        4.687e-5,
        -2.869e-2,
    ),
    (
        0.25,
        15,
        0.4501,
        0.008396,
        0.01028,
        0.0001290,
        -1.103e-7,
        7.999e-6,
        2.647e-5,
        -3.490e-3,
    ),
    (
        0.50,
        2,
        0.6104,
        0.4354,
        0.05789,
        0.006754,
        -1.258e-5,
        5.556e-3,
        7.549e-5,
        -6.061e-2,
    ),
    (
        0.50,
        5,
        0.4567,
        0.5110,
        0.02696,
        0.01496,
        -1.577e-6,
        2.587e-3,
        4.730e-5,
        -1.411e-1,
    ),
    (
        0.50,
        8,
        0.2928,
        0.3323,
        0.01268,
        0.007963,
        -3.654e-7,
        8.598e-4,
        2.631e-5,
        -1.126e-1,
    ),
    (
        0.50,
        15,
        0.08166,
        0.05497,
        0.002211,
        0.0008298,
        -2.436e-8,
        5.040e-5,
        5.846e-6,
        -2.199e-2,
    ),
    (
        0.75,
        2,
        0.2927,
        0.8192,
        0.03500,
        0.01650,
        -6.879e-6,
        7.009e-3,
        4.127e-5,
        -7.646e-2,
    ),
    (
        0.75,
        5,
        0.1811,
        0.9128,
        0.01263,
        0.02272,
        -7.054e-7,
        3.688e-3,
        2.116e-5,
        -2.012e-1,
    ),
    (
        0.75,
        8,
        0.08421,
        0.7321,
        0.004105,
        0.01493,
        -1.164e-7,
        1.596e-3,
        8.384e-6,
        -2.089e-1,
    ),
    (
        0.75,
        15,
        0.008713,
        0.3006,
        0.0002520,
        0.004022,
        -2.789e-9,
        2.453e-4,
        6.693e-7,
        -1.070e-1,
    ),
]


def _computed_row(depth_fraction, harmonic_degree, *, layered):
    radius_inner = 0.55
    radius_outer = 1.0
    rint = radius_outer - depth_fraction * (radius_outer - radius_inner)
    options = {}
    if layered:
        # Four of the benchmark's 64 radial elements form the high-viscosity lid.
        lid_base = radius_outer - 4.0 * (radius_outer - radius_inner) / 64.0
        options = {
            "viscosity_interfaces": (lid_base,),
            "viscosities": (1.0, 1.0e4),
        }

    response = uw.analytic.Zhong2008(
        harmonic_degree=harmonic_degree,
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        internal_load_radius=rint,
        **options,
    ).response()
    return np.array(
        [
            response.self_gravity.surface_topography,
            response.self_gravity.cmb_topography,
            response.self_gravity.surface_geoid,
            response.self_gravity.cmb_geoid,
            response.surface_characteristic_velocity,
            response.cmb_characteristic_velocity,
            response.surface_velocity_divergence,
            response.cmb_velocity_divergence,
        ]
    )


@pytest.mark.parametrize("published", ISOVISCOUS_TABLE_2)
def test_isoviscous_responses_match_every_analytic_value_in_table_2(published):
    depth_fraction, harmonic_degree, *expected = published
    computed = _computed_row(depth_fraction, harmonic_degree, layered=False)

    np.testing.assert_allclose(computed, expected, rtol=5.0e-4, atol=1.0e-12)


@pytest.mark.parametrize("published", LAYERED_TABLE_3)
def test_layered_responses_match_every_analytic_value_in_table_3(published):
    depth_fraction, harmonic_degree, *expected = published
    computed = _computed_row(depth_fraction, harmonic_degree, layered=True)

    np.testing.assert_allclose(computed, expected, rtol=5.0e-4, atol=1.0e-12)


def test_no_self_gravity_outputs_match_the_mid_mantle_reference():
    response = uw.analytic.Zhong2008().response()

    assert np.isclose(response.surface_topography, 0.4191904157)
    assert np.isclose(response.cmb_topography, 0.7705825500)
    assert np.isclose(response.surface_geoid, 0.0257906289)
    assert np.isclose(response.cmb_geoid, 0.0320605845)


def test_boundary_states_satisfy_impermeable_free_slip():
    solution = uw.analytic.Zhong2008(
        harmonic_degree=8,
        internal_load_radius=0.6625,
        viscosity_interfaces=(0.8, 0.95),
        viscosities=(1.0, 30.0, 0.2),
    )
    cmb_state, surface_state = solution._boundary_states()

    np.testing.assert_allclose(cmb_state[[0, 3]], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(surface_state[[0, 3]], 0.0, atol=1.0e-10)


def test_response_is_linear_in_load_amplitude():
    unit = uw.analytic.Zhong2008(internal_load_coefficient=1.0).response()
    doubled = uw.analytic.Zhong2008(internal_load_coefficient=2.0).response()

    unit_values = np.array(
        [
            unit.surface_topography,
            unit.cmb_topography,
            unit.surface_geoid,
            unit.cmb_geoid,
            unit.surface_characteristic_velocity,
            unit.cmb_characteristic_velocity,
            unit.self_gravity.surface_topography,
            unit.self_gravity.cmb_topography,
            unit.self_gravity.surface_geoid,
            unit.self_gravity.cmb_geoid,
        ]
    )
    doubled_values = np.array(
        [
            doubled.surface_topography,
            doubled.cmb_topography,
            doubled.surface_geoid,
            doubled.cmb_geoid,
            doubled.surface_characteristic_velocity,
            doubled.cmb_characteristic_velocity,
            doubled.self_gravity.surface_topography,
            doubled.self_gravity.cmb_topography,
            doubled.self_gravity.surface_geoid,
            doubled.self_gravity.cmb_geoid,
        ]
    )

    np.testing.assert_allclose(doubled_values, 2.0 * unit_values, rtol=1.0e-12)


@pytest.mark.parametrize(
    "options, error, message",
    [
        ({"harmonic_degree": 2.0}, TypeError, "must be an integer"),
        ({"harmonic_degree": 0}, ValueError, "must be positive"),
        ({"internal_load_radius": 1.1}, ValueError, "ordered as"),
        (
            {"viscosity_interfaces": (0.8,), "viscosities": (1.0,)},
            ValueError,
            "one value more",
        ),
        (
            {"viscosity_interfaces": (0.9, 0.8), "viscosities": (1.0, 2.0, 3.0)},
            ValueError,
            "strictly increasing",
        ),
        ({"viscosities": (-1.0,)}, ValueError, "finite and positive"),
    ],
)
def test_invalid_parameters_are_rejected(options, error, message):
    with pytest.raises(error, match=message):
        uw.analytic.Zhong2008(**options)


def test_zhong_oracle_is_public_but_not_a_symbolic_mesh_solution():
    assert "Zhong2008" in uw.analytic.__all__
    assert "Zhong2008Response" in uw.analytic.__all__
    assert "Zhong2008" not in uw.analytic.available()
    assert "Zhong et al. (2008)" in uw.analytic.Zhong2008.reference
