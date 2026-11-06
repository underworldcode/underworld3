"""Independent geometric and affine-field oracles; no Stokes solve or h=1/8 mesh."""

from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp
import underworld3 as uw
from petsc4py import PETSc

from underworld3.diagnostics.spherical import (
    QuadratureOrders, _norm_result, _physical_inner_surface_rules,
    _physical_volume_batches, spherical_stokes_errors,
    spherical_stokes_field_differences,
)
from underworld3.diagnostics.spherical_reactions import (
    HarmonicTarget, _target_values, spherical_stokes_reactions,
)

pytestmark = pytest.mark.level_1


@pytest.mark.tier_b
def test_physical_support_and_octant_volume_moment():
    # Tetrahedron minus a spherical octant: the volume and x^2 moment are closed form.
    vertices = np.vstack((np.zeros(3), 2*np.eye(3)))[None]
    batches = list(_physical_volume_batches(vertices, np.array([0]), .4,
                                            QuadratureOrders(4, 28, 4), 16))
    volume = sum(weights.sum() for _, _, weights in batches)
    moment = sum(np.sum(weights*points[..., 0]**2) for _, points, weights in batches)
    np.testing.assert_allclose(volume, 8/6-np.pi*.4**3/6, rtol=3e-12)
    np.testing.assert_allclose(moment, 2**5/60-np.pi*.4**5/30, rtol=3e-12)
    void = np.array([[[1., 0, 0], [0, 1., 0], [0, 0, 1.], [-1., 0, 0]]])
    assert list(_physical_volume_batches(void, np.array([0]), 1.,
                                        QuadratureOrders(4, 12, 4), 16)) == []
    assert list(_physical_volume_batches(np.empty((0, 4, 3)), np.empty(0, dtype=int),
                                        .4, QuadratureOrders(4, 12, 4), 16)) == []


@pytest.mark.tier_b
@pytest.mark.parametrize('components', [1, 3])
def test_physical_inner_trace_has_cap_area_and_distributed_ownership(components):
    # Only rank zero owns the multiplier facet, and only the last rank owns the
    # cut cell. The multiplier cone covers one quarter of a spherical cap.
    comm = PETSc.COMM_WORLD.tompi4py()
    radius = .4
    mesh = SimpleNamespace(dm=SimpleNamespace(comm=PETSc.COMM_WORLD))
    facets = radius*np.eye(3)[None] if comm.rank == 0 else np.empty((0, 3, 3))
    lids = (np.arange(6*components, dtype=PETSc.IntType)[None] if comm.rank == 0
            else np.empty((0, 6*components), dtype=PETSc.IntType))
    wall = dict(facets=facets, lid=lids, radius=radius)
    height = .3
    vertices = np.array([[[2., 0, height], [-1., np.sqrt(3), height],
                          [-1., -np.sqrt(3), height], [0, 0, 2.]]])
    owned = np.array([0]) if comm.rank == comm.size-1 else np.empty(0, dtype=int)
    rules = _physical_inner_surface_rules(mesh, vertices, owned, wall, radius, 28)
    assert len(rules) == len(owned)
    for cell, points, weights, ids, values in rules:
        assert cell == 0
        np.testing.assert_allclose(weights.sum(), np.pi*radius*(radius-height)/2, rtol=3e-12)
        np.testing.assert_allclose(np.linalg.norm(points, axis=-1), radius, rtol=2e-14)
        np.testing.assert_array_equal(ids, np.broadcast_to(lids if comm.rank == 0
            else np.arange(6*components)[None], ids.shape))
        if components == 1:
            np.testing.assert_allclose(values.sum(axis=1), 1., rtol=2e-14)
        else:
            # Cartesian coefficients map by R/|y|, y in x+y+z=R.
            np.testing.assert_allclose(values.sum(axis=1), points.sum(axis=1)/radius,
                                       rtol=2e-14)


@pytest.mark.tier_b
def test_norm_zero_reference_and_signed_cancellation_policy():
    result = _norm_result(9., 4., 2., 13., 4.)
    assert result['relative_error_fraction'] == 1.5
    assert result['absolute_rms_error'] == np.sqrt(9/2)
    assert result['reference_rms'] == np.sqrt(2)
    assert result['relative_error_defined']
    for reference, prescribed in ((0., False), (1e-30, False), (4., True)):
        result = _norm_result(9., reference, 2., 13., 4., prescribed)
        assert result['relative_error_fraction'] is None
        assert not result['relative_error_defined']
        assert result['absolute_rms_error'] == np.sqrt(9/2)
        assert result['reference_rms'] == np.sqrt(reference/2)
        assert result['reference_is_prescribed_zero'] is prescribed
    rounded = _norm_result(-np.finfo(float).eps, 4., 2., 1., 4.)
    assert rounded['absolute_rms_error'] == 0.
    with pytest.raises(ArithmeticError, match='negative squared norm'):
        _norm_result(-1e-8, 4., 2., 1., 4.)
    with pytest.raises(ArithmeticError, match='positive measure'):
        _norm_result(1., 4., 0., 1., 4.)


@pytest.mark.tier_b
def test_constant_and_axisymmetric_harmonics_have_closed_form_values():
    points = np.array([[[1., 0, 0], [0, 0, 2.], [0, 0, -3.]]])
    np.testing.assert_array_equal(_target_values(points, HarmonicTarget(
        0, 0, 'cos', 'real_orthonormal')), np.full((1, 3), 1/np.sqrt(4*np.pi)))
    np.testing.assert_allclose(_target_values(points, HarmonicTarget(
        1, 0, 'cos', 'real_orthonormal')), [[0., np.sqrt(3/(4*np.pi)),
        -np.sqrt(3/(4*np.pi))]], atol=1e-15, rtol=2e-15)
    with pytest.raises(ValueError, match='Degree-zero'):
        _target_values(points, HarmonicTarget(0, 0, 'sin', 'real_orthonormal'))


@pytest.fixture(scope='module')
def shell():
    return uw.meshing.SphericalShell(radiusInner=.55, radiusOuter=1., cellSize=.5,
                                    degree=1, qdegree=3)


class _AffineReference:
    def evaluate(self, field: str, coords: np.ndarray) -> np.ndarray:
        if field == 'pressure':
            return coords[:, 0]
        if field == 'velocity':
            return coords*np.array([1., -1., 0.])
        if field == 'stress':
            return np.diag([3.4, -3.4, 0.])-coords[:, 0, None, None]*np.eye(3)
        raise ValueError(f'Unknown affine reference field: {field}')


@pytest.mark.tier_b
def test_affine_fe_stress_pressure_gauge_and_difference_oracles(shell):
    u = uw.discretisation.MeshVariable('NativeDiagnosticU', shell, 3, degree=2)
    p = uw.discretisation.MeshVariable('NativeDiagnosticP', shell, 1, degree=1)
    zero_u = uw.discretisation.MeshVariable('NativeDiagnosticZeroU', shell, 3, degree=2)
    zero_p = uw.discretisation.MeshVariable('NativeDiagnosticZeroP', shell, 1, degree=1)
    u.array[:, 0, :] = np.asarray(u.coords)*np.array([1., -1., 0.])
    p.array[:, 0, 0] = np.asarray(p.coords)[:, 0]+2.75
    zero_u.array[:] = 0.
    zero_p.array[:] = 0.
    for field in (u, p, zero_u, zero_p):
        field._sync_lvec_to_gvec()
    shell._stale_lvec = True
    before = [np.asarray(field.array).copy() for field in (u, p, zero_u, zero_p)]
    orders = QuadratureOrders(4, 16, 4)
    result = spherical_stokes_errors(shell, u, p, sp.S.Zero, _AffineReference(),
        .55, 1., orders, 32, 'Lower', 'Upper', viscosity=1.7,
        include_legacy_signed_gap=True)
    volume = 4*np.pi*(1-.55**3)/3
    moment = 4*np.pi*(1-.55**5)/15
    for domain in ('polyhedral_volume', 'physical_signed_gap_volume', 'physical_inner_cut_volume'):
        assert abs(result[domain]['pressure_alignment_offset']-2.75) < 2e-12
        for field in ('velocity', 'pressure', 'stress'):
            assert result[domain][field]['absolute_rms_error'] < 3e-12
    for domain in ('physical_signed_gap_volume', 'physical_inner_cut_volume'):
        np.testing.assert_allclose(result[domain]['measure'], volume, rtol=3e-10)
    for name, radius in (('Lower', .55), ('Upper', 1.)):
        wall = result['physical_sphere_walls'][name]
        np.testing.assert_allclose(wall['area'], 4*np.pi*radius**2, rtol=3e-10)
        for field in ('pressure', 'normal_velocity', 'tangential_velocity'):
            assert wall[field]['absolute_rms_error'] < 3e-12
    difference = spherical_stokes_field_differences(shell, u, p, zero_u, zero_p,
        .55, 1., orders, 32, 'Lower', 'Upper', viscosity=1.7)
    # u=(x,-y,0), p=x: integral |u|^2=2M, |p|^2=M, |sigma|^2=8mu^2 V+3M.
    np.testing.assert_allclose(difference['pressure_mean_difference'], 2.75, rtol=1e-12)
    np.testing.assert_allclose(difference['velocity_l2_difference'], np.sqrt(2*moment), rtol=3e-10)
    np.testing.assert_allclose(difference['pressure_l2_difference'], np.sqrt(moment), rtol=3e-10)
    np.testing.assert_allclose(difference['stress_l2_difference'], np.sqrt(8*1.7**2*volume+3*moment), rtol=3e-10)
    for field, saved in zip((u, p, zero_u, zero_p), before, strict=True):
        np.testing.assert_array_equal(field.array, saved)


class _ConstantStress:
    def __init__(self, tensor: np.ndarray):
        self.tensor = tensor

    def evaluate(self, field: str, coords: np.ndarray) -> np.ndarray:
        if field != 'stress':
            raise ValueError(f'Unknown stress reference field: {field}')
        return np.broadcast_to(self.tensor, (len(coords), 3, 3)).copy()


@pytest.mark.tier_b
@pytest.mark.parametrize('components', [1, 3])
def test_hydrostatic_and_affine_reaction_sign_gauge_and_mode_oracles(shell, components):
    u = uw.discretisation.MeshVariable(f'NativeReactionU{components}', shell, 3, degree=2)
    p = uw.discretisation.MeshVariable(f'NativeReactionP{components}', shell, 1, degree=1)
    normal = uw.discretisation.MeshVariable(f'NativeReactionN{components}', shell, 1, degree=2)
    tangent = uw.discretisation.MeshVariable(f'NativeReactionT{components}', shell, 3, degree=2)
    strain = np.zeros(3) if components == 1 else np.array([.2, -.1, -.1])
    mu, offset, raw_pressure = 1.3, 2.75, 3.2
    u.array[:, 0, :] = np.asarray(u.coords)*strain
    p.array[:, 0, 0] = raw_pressure
    tau = np.diag(2*mu*strain)
    coords = np.asarray(normal.coords)
    radii = np.linalg.norm(coords, axis=1)
    sign = np.where(abs(radii-.55) < abs(radii-1.), -1., 1.)
    n = sign[:, None]*coords/radii[:, None]
    rho = raw_pressure*n-n@tau.T
    node_normal = np.sum(rho*n, axis=1)
    normal.array[:, 0, 0] = node_normal
    tangent.array[:, 0, :] = rho-node_normal[:, None]*n
    for field in (u, p, normal, tangent):
        field._sync_lvec_to_gvec()
    shell._stale_lvec = True
    before = [np.asarray(field.array).copy() for field in (u, p, normal, tangent)]
    result = spherical_stokes_reactions(shell, u, p, sp.S.Zero, normal, tangent,
        components, _ConstantStress(tau-(raw_pressure-offset)*np.eye(3)),
        mu, .55, 1., offset, 16, 4, 32,
        HarmonicTarget(0, 0, 'cos', 'real_orthonormal'), 'Lower', 'Upper')
    expected_coefficient = (raw_pressure-offset)*np.sqrt(4*np.pi)
    # For trace-free tau, average (n.tau.n)^2 = 2 tr(tau^2)/15.
    expected_off_target = np.sqrt(2*np.sum(tau*tau)/15)
    for wall in result['walls'].values():
        for name in ('normal_reaction_error_rms', 'tangential_reaction_error_rms',
                     'normal_traction_residual_rms', 'full_traction_residual_rms'):
            assert wall[name] < 3e-12
        np.testing.assert_allclose(wall['pressure_target_coefficient'], expected_coefficient, rtol=3e-12)
        np.testing.assert_allclose(wall['normal_reaction_target_coefficient'], expected_coefficient, rtol=3e-10)
        assert wall['pressure_off_target_rms'] < 3e-12
        np.testing.assert_allclose(wall['normal_reaction_off_target_rms'], expected_off_target,
                                   rtol=3e-10, atol=3e-12)
        if components == 1:
            assert wall['free_natural_tangential_traction_residual_rms'] < 3e-12
    for field, saved in zip((u, p, normal, tangent), before, strict=True):
        np.testing.assert_array_equal(field.array, saved)
