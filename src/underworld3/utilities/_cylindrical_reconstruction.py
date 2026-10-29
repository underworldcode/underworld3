"""Private spatial reconstruction of circular P2 boundary traces.

The compact local-L2 evaluation kernel reproduces polynomials with weight
``(1-z**2)**2`` and vanishes smoothly at its support endpoints. Convolve the
existing spatial trace, project onto the same circular P2 space, and preserve
its mean and Cartesian radial-resultant moments. No prescribed harmonic or
analytical solution enters the reconstruction.

Accuracy-enhancing convolution is discussed by Docampo-Sanchez, J., Jacobs,
G. B., Li, X., and Ryan, J. K. (2020), "Enhancing accuracy with a convolution
filter: What works and why!", Computers & Fluids 213, 104727,
https://doi.org/10.1016/j.compfluid.2020.104727. The weighted polynomial kernel,
circular projection and physical-moment constraints here are UW3 adaptations.
This is not that paper's spline/DG operator and inherits no SIAC convergence
theorem for Stokes reactions or nonuniform circular meshes.

Only boundary nodes are replicated. Sparse one-dimensional SciPy mass solves
are used, as in the existing circular reaction transfer; no volume field or
Stokes operator is gathered. Nondimensional inputs are required.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.special import eval_jacobi
from scipy.spatial import cKDTree
import underworld3 as uw


def _reconstruction_kernel(points, degree):
    r"""Evaluate the normalized weighted polynomial-reproducing kernel.

    The Jacobi polynomials :math:`P_j^{(2,2)}` are orthogonal with weight
    :math:`w=(1-z^2)^2`. Thus :math:`w(z)\sum_j P_j(0)P_j(z)/\|P_j\|_w^2`
    reproduces evaluation at zero through ``degree``. Its support is [-1, 1].
    """
    if degree not in (4, 6, 8):
        raise ValueError("Circular trace reconstruction supports degrees 4, 6 and 8.")
    points = np.asarray(points, dtype=float)
    inside = np.abs(points) <= 1
    result = np.zeros_like(points)
    z = points[inside]
    polynomial = np.zeros_like(z)
    for order in range(0, degree+1, 2):
        norm = 32*(order+1)*(order+2)/((2*order+5)*(order+3)*(order+4))
        polynomial += eval_jacobi(order, 2, 2, 0)*eval_jacobi(order, 2, 2, z)/norm
    result[inside] = (1-z*z)**2*polynomial
    return result


def _recover_circular_trace(edges, nodal_values, radius, degree, halfwidth_edges, quadrature):
    """Reconstruct a complete circular P2 trace from boundary arrays.

    Input values follow lexicographically unique endpoints and midpoints of
    ``edges``, matching cylindrical reaction recovery. The support halfwidth
    is ``halfwidth_edges`` times the mean physical boundary arc length. Split
    integration at original and shifted source vertices, also on graded meshes.
    This relaxes fine-scale CBF moments; the original trace is not overwritten.
    """
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 3 or edges.shape[1:] != (2, 2) or not np.all(np.isfinite(edges)):
        raise ValueError("Circular trace edges require finite (edges, 2, 2) coordinates.")
    if radius <= 0 or not np.isfinite(radius):
        raise ValueError("Circular trace radius must be positive and finite.")
    if degree not in (4, 6, 8):
        raise ValueError("Circular trace reconstruction supports degrees 4, 6 and 8.")
    if not np.isfinite(halfwidth_edges) or halfwidth_edges <= 0 or quadrature < 2:
        raise ValueError("Require positive finite halfwidth and quadrature >= 2.")
    vertices = np.unique(edges.reshape(-1, 2), axis=0)
    if not np.allclose(np.linalg.norm(vertices, axis=1), radius, rtol=1e-10, atol=0):
        raise ValueError("Boundary vertices must lie on the specified circle.")
    angles = np.mod(np.arctan2(vertices[:, 1], vertices[:, 0]), 2*np.pi)
    vertices = vertices[np.argsort(angles)]
    ordered = np.stack((vertices, np.roll(vertices, -1, axis=0)), axis=1)
    original_pairs = {tuple(sorted(map(tuple, pair))) for pair in edges}
    ordered_pairs = {tuple(sorted(map(tuple, pair))) for pair in ordered}
    if len(edges) != len(vertices) or original_pairs != ordered_pairs:
        raise ValueError("Input edges must form one complete circular wall.")
    edge_nodes = np.concatenate((ordered, ordered.mean(axis=1)[:, None]), axis=1)
    nodes, index = np.unique(edge_nodes.reshape(-1, 2), axis=0, return_inverse=True)
    index = index.reshape(-1, 3)
    values = np.asarray(nodal_values, dtype=float).reshape(-1)
    if len(values) != len(nodes) or not np.all(np.isfinite(values)):
        raise ValueError("Require one finite value per unique P2 boundary node.")
    starts = np.unwrap(np.mod(np.arctan2(ordered[:, 0, 1], ordered[:, 0, 0]), 2*np.pi))
    origin, period, count = starts[0], 2*np.pi, len(ordered)
    ends = np.r_[starts[1:], origin+period]
    if np.any(ends-starts >= np.pi):
        raise ValueError("Circular trace arcs must each span less than half a circle.")
    width = halfwidth_edges*period/count
    if not 0 < width < np.pi:
        raise ValueError("The recovery halfwidth must span less than half a circle.")
    gauss, weight = np.polynomial.legendre.leggauss(quadrature)

    def basis(theta):
        query = (theta-origin) % period+origin
        selected = np.searchsorted(starts, query, side="right")-1
        unit = np.stack((np.cos(query), np.sin(query)), axis=-1)
        first = ordered[selected, 0]
        direction = ordered[selected, 1]-first
        t = -(first[..., 0]*unit[..., 1]-first[..., 1]*unit[..., 0])/(
            direction[..., 0]*unit[..., 1]-direction[..., 1]*unit[..., 0])
        shape = np.stack(((1-t)*(1-2*t), t*(2*t-1), 4*t*(1-t)), axis=-1)
        return selected, shape

    def unwrapped_cell(theta):
        return (np.floor((theta-origin)/period).astype(int)*count
                + np.searchsorted(starts, (theta-origin) % period+origin, side="right")-1)

    def convolution(theta):
        first, last = unwrapped_cell(theta-width), unwrapped_cell(theta+width)
        result = np.zeros_like(theta)
        for offset in range(int(np.max(last-first))+1):
            cell = first+offset
            wrap, turn = cell % count, cell//count
            lo = np.maximum(starts[wrap]+turn*period, theta-width)
            hi = np.minimum(ends[wrap]+turn*period, theta+width)
            keep = (cell <= last) & (hi > lo)
            points = ((lo[keep, None]+hi[keep, None])/2
                      + (hi[keep, None]-lo[keep, None])*gauss/2)
            selected, shape = basis(points)
            samples = np.sum(shape*values[index[selected]], axis=-1)
            kernel = _reconstruction_kernel((points-theta[keep, None])/width, degree)
            result[keep] += np.sum(
                (hi[keep, None]-lo[keep, None])*weight/2*kernel*samples/width, axis=1)
        return result

    # Exact circular arclength defines the projection and physical moments.
    points = ((starts[:, None]+ends[:, None])/2
              + (ends[:, None]-starts[:, None])*gauss/2).ravel()
    weights = (radius*(ends[:, None]-starts[:, None])*weight/2).ravel()
    selected, shape = basis(points)
    local = np.einsum("q,qi,qj->qij", weights, shape, shape)
    rows = np.broadcast_to(index[selected, :, None], local.shape).ravel()
    columns = np.broadcast_to(index[selected, None, :], local.shape).ravel()
    mass = coo_matrix((local.ravel(), (rows, columns)), shape=(len(nodes), len(nodes))).tocsr()
    constraints = np.zeros((3, len(nodes)))
    for row, radial_component in enumerate((np.ones_like(points), np.cos(points), np.sin(points))):
        for j in range(3):
            np.add.at(constraints[row], index[selected, j], weights*shape[:, j]*radial_component)

    cuts = (np.r_[starts, starts-width, starts+width]-origin) % period+origin
    cuts = np.sort(np.r_[cuts, origin, origin+period])
    cuts = cuts[np.r_[True, np.diff(cuts) > 1e-13]]
    cuts[-1] = origin+period
    points = ((cuts[:-1, None]+cuts[1:, None])/2
              + (cuts[1:, None]-cuts[:-1, None])*gauss/2).ravel()
    weights = (radius*(cuts[1:, None]-cuts[:-1, None])*weight/2).ravel()
    selected, shape = basis(points)
    samples = convolution(points)
    load = np.zeros(len(nodes))
    for j in range(3):
        np.add.at(load, index[selected, j], weights*shape[:, j]*samples)
    recovered = spsolve(mass, load)
    correction_basis = spsolve(mass, constraints.T)
    recovered += correction_basis@np.linalg.solve(
        constraints@correction_basis, constraints@(values-recovered))
    change = recovered-values
    moment_change = constraints@change
    checks = dict(degree=degree, halfwidth_edges=halfwidth_edges,
        physical_arc_halfwidth=radius*width, quadrature=quadrature,
        maximum_conserved_moment_error=float(np.max(np.abs(moment_change))),
        maximum_nodal_moment_change=float(np.max(np.abs(mass@change))),
        circular_l2_change=float(np.sqrt(change@(mass@change))),
        circular_mean_change=float(moment_change[0]/(2*np.pi*radius)),
        cartesian_radial_resultant_change=moment_change[1:].tolist(),
        fine_scale_cbf_moments_preserved=False)
    return nodes, recovered, checks


def _reconstruct_cylindrical_trace(stokes, *, wall, source, target, degree, halfwidth_edges):
    """Write a separate reconstructed trace; replicate boundary nodes only."""
    from ._cylindrical_reactions import _validate_spatial_target

    if wall.mesh is not stokes.mesh:
        raise ValueError("The wall handle must belong to this Stokes mesh.")
    if source is target:
        raise ValueError("Source and reconstructed target must be distinct fields.")
    _validate_spatial_target(wall, source)
    _validate_spatial_target(wall, target)
    boundary = wall.local_boundary
    coordinates = np.concatenate(uw.mpi.comm.allgather(np.asarray(source.coords)[boundary]))
    values = np.concatenate(uw.mpi.comm.allgather(np.asarray(source.array)[boundary, 0, 0]))
    unique, first, inverse = np.unique(coordinates, axis=0, return_index=True, return_inverse=True)
    if not np.allclose(values, values[first][inverse], rtol=1e-12, atol=0):
        raise ValueError("Shared boundary nodes have inconsistent source values.")
    distance, index = cKDTree(unique).query(wall.nodes)
    if np.max(distance) > 1e-10:
        raise ValueError("The source field does not cover the circular wall nodes.")
    nodes, reconstructed, checks = _recover_circular_trace(
        wall.edges, values[first][index], wall.radius, degree, halfwidth_edges, 12)
    np.testing.assert_array_equal(nodes, wall.nodes)
    target.array[boundary, 0, 0] = reconstructed[wall.local_index[boundary]]
    return checks
