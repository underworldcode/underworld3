"""Mathematical contracts for spatial circular trace reconstruction.

No Stokes solve is required. Independent kernel moments, circular integration,
frame transformations and conservation define the baselines.
"""

import numpy as np
import pytest
import underworld3 as uw

pytestmark = pytest.mark.level_1


@pytest.mark.tier_b
def test_stokes_exposes_explicit_spatial_trace_reconstruction():
    """The solver capability must be reachable without a private utility import."""
    assert callable(getattr(uw.systems.Stokes, "reconstruct_cylindrical_trace", None)), (
        "Stokes lacks reconstruct_cylindrical_trace(*, wall, source, target, degree, halfwidth_edges)."
    )


def _ring(count, graded, radius):
    steps = 1 + (0.7*np.sin(0.7*np.arange(count)) if graded else np.zeros(count))
    angles = 2*np.pi*np.r_[0, np.cumsum(steps[:-1])]/steps.sum()
    vertices = radius*np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    edges = np.stack((vertices, np.roll(vertices, -1, axis=0)), axis=1)
    nodes = np.unique(np.concatenate((edges, edges.mean(axis=1)[:, None]), axis=1).reshape(-1, 2), axis=0)
    return edges, nodes


def _reference_reconstruction(edges, values, radius, degree, halfwidth_edges):
    """Direct scalar quadrature with an independent angular/chord coordinate map."""
    from underworld3.utilities._cylindrical_reconstruction import _reconstruction_kernel

    vertices = np.unique(edges.reshape(-1, 2), axis=0)
    starts = np.sort(np.mod(np.arctan2(vertices[:, 1], vertices[:, 0]), 2*np.pi))
    ends = np.r_[starts[1:], starts[0]+2*np.pi]
    ordered_vertices = radius*np.stack((np.cos(starts), np.sin(starts)), axis=-1)
    ordered_edges = np.stack((ordered_vertices, np.roll(ordered_vertices, -1, axis=0)), axis=1)
    nodes = np.unique(np.concatenate((edges, edges.mean(axis=1)[:, None]), axis=1).reshape(-1, 2), axis=0)
    expected_nodes = np.concatenate((ordered_edges, ordered_edges.mean(axis=1)[:, None]), axis=1)
    index = np.argmin(np.linalg.norm(expected_nodes[:, :, None]-nodes[None, None], axis=-1), axis=-1)
    width = halfwidth_edges*2*np.pi/len(edges)
    gauss, weight = np.polynomial.legendre.leggauss(24)

    def shapes(theta):
        wrapped = (theta-starts[0]) % (2*np.pi)+starts[0]
        cell = np.searchsorted(starts, wrapped, side="right")-1
        half_arc = (ends[cell]-starts[cell])/2
        midpoint = (ends[cell]+starts[cell])/2
        chord = (np.tan(wrapped-midpoint)/np.tan(half_arc)+1)/2
        return cell, np.stack(((1-chord)*(1-2*chord), chord*(2*chord-1), 4*chord*(1-chord)), axis=-1)

    cuts = np.sort(np.r_[0, 2*np.pi, starts, (starts-width) % (2*np.pi), (starts+width) % (2*np.pi)])
    cuts = cuts[np.r_[True, np.diff(cuts)>1e-13]]
    mass = np.zeros((len(nodes), len(nodes)))
    load = np.zeros(len(nodes))
    conserved = np.zeros((3, len(nodes)))
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        for theta, outer_weight in zip((lo+hi)/2+(hi-lo)*gauss/2, radius*(hi-lo)*weight/2):
            cell, basis = shapes(np.asarray(theta))
            source_cuts = np.sort(np.r_[theta-width, theta+width,
                np.tile(starts, 3)+np.repeat([-2*np.pi, 0, 2*np.pi], len(starts))])
            source_cuts = source_cuts[(source_cuts>=theta-width) & (source_cuts<=theta+width)]
            integral = 0.0
            for first, last in zip(source_cuts[:-1], source_cuts[1:]):
                points = (first+last)/2+(last-first)*gauss/2
                source_cell, source_basis = shapes(points)
                source_value = np.sum(source_basis*values[index[source_cell]], axis=-1)
                integral += np.sum((last-first)*weight/2*source_value
                    *_reconstruction_kernel((points-theta)/width, degree)/width)
            rows = index[cell]
            mass[np.ix_(rows, rows)] += outer_weight*np.outer(basis, basis)
            load[rows] += outer_weight*basis*integral
            conserved[:, rows] += outer_weight*np.outer([1, np.cos(theta), np.sin(theta)], basis)
    result = np.linalg.solve(mass, load)
    correction = np.linalg.solve(mass, conserved.T)
    result += correction@np.linalg.solve(conserved@correction, conserved@(values-result))
    return result, conserved


@pytest.mark.tier_b
@pytest.mark.parametrize("degree", [4, 6, 8])
def test_kernel_reproduces_polynomial_moments(degree):
    from underworld3.utilities._cylindrical_reconstruction import _reconstruction_kernel

    points, weights = np.polynomial.legendre.leggauss(20)
    kernel = _reconstruction_kernel(points, degree)
    for power in range(degree+1):
        expected = 1 if power == 0 else 0
        np.testing.assert_allclose(np.sum(weights*kernel*points**power), expected, rtol=0, atol=2e-13)


@pytest.mark.tier_b
@pytest.mark.parametrize("graded", [False, True])
def test_constant_and_multimode_trace_against_direct_variational_integral(graded):
    from underworld3.utilities._cylindrical_reconstruction import _recover_circular_trace

    edges, nodes = _ring(24, graded, 1.7)
    _, constant, _ = _recover_circular_trace(edges, np.full(len(nodes), 1.234), 1.7, 6, 5, 12)
    np.testing.assert_allclose(constant, 1.234, rtol=0, atol=3e-12)
    theta = np.arctan2(nodes[:, 1], nodes[:, 0])
    values = 1+.2*np.cos(theta)-.3*np.sin(theta)+.4*np.cos(3*theta)-.1*np.sin(7*theta)
    original = values.copy()
    returned_nodes, actual, _ = _recover_circular_trace(edges, values, 1.7, 6, 5, 12)
    reference, conserved = _reference_reconstruction(edges, values, 1.7, 6, 5)
    np.testing.assert_array_equal(returned_nodes, nodes)
    np.testing.assert_allclose(actual, reference, rtol=2e-10, atol=3e-11)
    np.testing.assert_allclose(conserved@actual, conserved@values, rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(values, original)


@pytest.mark.tier_b
def test_edge_permutation_rotation_and_physical_scaling():
    from underworld3.utilities._cylindrical_reconstruction import _recover_circular_trace

    edges, nodes = _ring(40, True, 1.22)
    theta = np.arctan2(nodes[:, 1], nodes[:, 0])
    values = 1+.3*np.cos(3*theta)-.1*np.sin(5*theta)+.2*np.cos(8*theta)
    _, expected, _ = _recover_circular_trace(edges, values, 1.22, 6, 5, 12)
    reordered = edges[np.random.default_rng(223).permutation(len(edges)), ::-1]
    _, permuted, _ = _recover_circular_trace(reordered, values, 1.22, 6, 5, 12)
    np.testing.assert_allclose(permuted, expected, rtol=0, atol=2e-12)
    _, scaled, _ = _recover_circular_trace(edges*1.7, values, 1.22*1.7, 6, 5, 12)
    np.testing.assert_allclose(scaled, expected, rtol=0, atol=2e-12)
    angle = .324
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated_edges = edges@rotation.T
    rotated_nodes = np.unique(np.concatenate((rotated_edges, rotated_edges.mean(axis=1)[:, None]), axis=1).reshape(-1, 2), axis=0)
    index = np.argmin(np.linalg.norm(rotated_nodes[:, None]-(nodes@rotation.T)[None], axis=-1), axis=1)
    _, rotated, _ = _recover_circular_trace(rotated_edges, values[index], 1.22, 6, 5, 12)
    np.testing.assert_allclose(rotated, expected[index], rtol=0, atol=2e-12)


@pytest.mark.tier_b
def test_boundary_only_field_adapter_preserves_source_and_other_nodes():
    from types import SimpleNamespace
    from underworld3.utilities._cylindrical_reconstruction import _reconstruct_cylindrical_trace

    edges, nodes = _ring(24, False, 1.7)
    # Empty-boundary ranks participate in the same collective adapter path.
    selected = np.arange(len(nodes)) if uw.mpi.rank == 0 else np.empty(0, dtype=int)
    coordinates = np.vstack((nodes[selected], [[0, 0]]))
    mesh = SimpleNamespace(dim=2, cdim=2)
    wall = SimpleNamespace(mesh=mesh, edges=edges, nodes=nodes, radius=1.7, quadrature=12,
        local_boundary=np.r_[np.ones(len(selected), dtype=bool), False],
        local_index=np.r_[selected, 0])
    source = SimpleNamespace(mesh=mesh, degree=2, continuous=True, num_components=1,
        coords=coordinates, array=np.full((len(coordinates), 1, 1), 2.3))
    source.array[-1, 0, 0] = 777
    target = SimpleNamespace(mesh=mesh, degree=2, continuous=True, num_components=1,
        coords=coordinates, array=np.full((len(coordinates), 1, 1), -999.0))
    original = source.array.copy()
    _reconstruct_cylindrical_trace(SimpleNamespace(mesh=mesh), wall=wall,
        source=source, target=target, degree=6, halfwidth_edges=5)
    np.testing.assert_allclose(target.array[:-1, 0, 0], 2.3, rtol=0, atol=3e-12)
    assert target.array[-1, 0, 0] == -999
    np.testing.assert_array_equal(source.array, original)
    with pytest.raises(ValueError, match="distinct"):
        _reconstruct_cylindrical_trace(SimpleNamespace(mesh=mesh), wall=wall,
            source=source, target=source, degree=6, halfwidth_edges=5)


@pytest.mark.tier_b
@pytest.mark.parametrize("invalid", ["missing_edge", "wrong_radius", "wrong_values", "degree", "width"])
def test_incomplete_or_unsupported_circular_trace_fails_explicitly(invalid):
    from underworld3.utilities._cylindrical_reconstruction import _recover_circular_trace

    edges, nodes = _ring(24, False, 1.7)
    values, radius, degree, width = np.ones(len(nodes)), 1.7, 6, 5
    if invalid == "missing_edge":
        edges = edges[:-1]
    elif invalid == "wrong_radius":
        radius = 1.9
    elif invalid == "wrong_values":
        values = values[:-1]
    elif invalid == "degree":
        degree = 3
    else:
        width = 12
    with pytest.raises(ValueError):
        _recover_circular_trace(edges, values, radius, degree, width, 12)


@pytest.mark.tier_b
def test_degree_six_kernel_matches_exact_weighted_reproducing_polynomial():
    from underworld3.utilities._cylindrical_reconstruction import _reconstruction_kernel

    z = np.linspace(-1, 1, 31)
    expected = (1-z*z)**2*(24255-315315*z*z+945945*z**4-765765*z**6)/8192
    np.testing.assert_allclose(_reconstruction_kernel(z, 6), expected, rtol=0, atol=2e-14)
    np.testing.assert_array_equal(_reconstruction_kernel(np.array([-2, 2]), 6), [0, 0])


@pytest.mark.tier_b
def test_degree_six_kernel_has_expected_finite_bandwidth_response():
    """At E=32, n=8, halfwidth=5 the analytic response is about 0.8213."""
    from underworld3.utilities._cylindrical_reconstruction import _reconstruction_kernel

    z, weights = np.polynomial.legendre.leggauss(64)
    frequency = 5*np.pi/2
    actual = np.sum(weights*_reconstruction_kernel(z, 6)*np.cos(frequency*z))
    # Exact integral of the degree-ten polynomial times cos(a*z), sin(a)=1.
    expected = (3465/frequency**3-2848230/frequency**5
                +265540275/frequency**7-5037157125/frequency**9
                +10854718875/frequency**11)/16
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-13)


@pytest.mark.tier_b
def test_partitioned_nonconstant_trace_with_shared_boundary_ghosts():
    from types import SimpleNamespace
    from underworld3.utilities._cylindrical_reconstruction import (
        _reconstruct_cylindrical_trace, _recover_circular_trace,
    )

    edges, nodes = _ring(24, True, 1.7)
    theta = np.arctan2(nodes[:, 1], nodes[:, 0])
    values = 1+.2*np.cos(theta)-.3*np.sin(theta)+.4*np.cos(3*theta)-.1*np.sin(7*theta)
    _, reference, _ = _recover_circular_trace(edges, values, 1.7, 6, 5, 12)
    # Distributed owned entries plus the same five boundary ghosts on every rank.
    owned = np.arange(len(nodes))[uw.mpi.rank::uw.mpi.size]
    selected = np.unique(np.r_[owned, np.arange(5)])
    coordinates = np.vstack((nodes[selected], [[0, 0]]))
    mesh = SimpleNamespace(dim=2, cdim=2)
    wall = SimpleNamespace(mesh=mesh, edges=edges, nodes=nodes, radius=1.7, quadrature=12,
        local_boundary=np.r_[np.ones(len(selected), dtype=bool), False],
        local_index=np.r_[selected, 0])
    source = SimpleNamespace(mesh=mesh, degree=2, continuous=True, num_components=1,
        coords=coordinates, array=np.r_[values[selected], 777].reshape(-1, 1, 1))
    target = SimpleNamespace(mesh=mesh, degree=2, continuous=True, num_components=1,
        coords=coordinates, array=np.full((len(coordinates), 1, 1), -999.0))
    original = source.array.copy()
    checks = _reconstruct_cylindrical_trace(SimpleNamespace(mesh=mesh), wall=wall,
        source=source, target=target, degree=6, halfwidth_edges=5)
    np.testing.assert_allclose(target.array[:-1, 0, 0], reference[selected], rtol=0, atol=2e-12)
    assert target.array[-1, 0, 0] == -999
    np.testing.assert_array_equal(source.array, original)
    assert checks["fine_scale_cbf_moments_preserved"] is False
    assert checks["maximum_conserved_moment_error"] < 2e-12
