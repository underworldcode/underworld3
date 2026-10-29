"""Constant-pressure invariance of circular CBF transfer after calibration.

No Stokes solve is required: independently integrated constant-pressure loads
are the cached reaction, while the current pressure field is shifted by a gauge.
The physical mean-free circular stress must remain identically zero.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from petsc4py import PETSc
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from underworld3.utilities._cylindrical_reactions import _recover_circular_topography


pytestmark = pytest.mark.level_1


class _PressureDM:
    """Preserve the native distributed pressure-vector contract in this fixture."""

    def localToGlobal(self, source, target):
        source.copy(target)


def _constant_pressure_fixture(shift: float):
    """Build exact P2 line loads on a polygon and a separately shifted pressure."""
    angle = np.linspace(0, 2*np.pi, 33)[:-1]
    first = np.column_stack((np.cos(angle), np.sin(angle)))
    edges = np.stack((first, np.roll(first, -1, axis=0)), axis=1)
    vertices = np.concatenate((edges, .9*edges.mean(axis=1)[:, None]), axis=1)
    edge_nodes = np.concatenate((edges, edges.mean(axis=1)[:, None]), axis=1)
    nodes, index = np.unique(edge_nodes.reshape(-1, 2), axis=0, return_inverse=True)
    index = index.reshape(-1, 3)
    t, weights = np.polynomial.legendre.leggauss(12)
    t, weights = (t+1)/2, weights/2
    shape = np.array([(1-t)*(1-2*t), t*(2*t-1), 4*t*(1-t)]).T
    delta = edges[:, 1]-edges[:, 0]
    length = np.linalg.norm(delta, axis=1)
    normal = np.column_stack((delta[:, 1], -delta[:, 0]))/length[:, None]
    nodal_normal = edge_nodes/np.linalg.norm(edge_nodes, axis=-1)[..., None]
    local_mass = np.einsum("e,q,qi,qj->eij", length, weights, shape, shape)
    rows = np.broadcast_to(index[:, :, None], local_mass.shape).ravel()
    columns = np.broadcast_to(index[:, None, :], local_mass.shape).ravel()
    mass = coo_matrix((local_mass.ravel(), (rows, columns)), shape=(len(nodes), len(nodes))).tocsr()
    # sigma=-I gives the opposite solver-reaction convention +n_i.n_facet.
    loads = np.zeros(len(nodes))
    local_load = np.einsum("e,q,qi,eid,ed->ei", length, weights, shape, nodal_normal, normal)
    np.add.at(loads, index.ravel(), local_load.ravel())
    reaction = spsolve(mass, loads)
    uniform = mass@np.ones(len(nodes))
    reaction -= uniform@reaction/uniform.sum()
    cached = PETSc.Vec().createMPI((len(vertices)*3, None), comm=PETSc.COMM_WORLD)
    cached.set(1)
    current = cached.duplicate()
    current.set(1+shift)
    start, end = cached.getOwnershipRange()
    pressure_is = PETSc.IS().createStride(end-start, first=start, step=1, comm=PETSc.COMM_WORLD)
    mesh = object()
    wall = SimpleNamespace(mesh=mesh, name="Upper", nodes=nodes, vertices=vertices,
        edges=edges, radius=1., sign=1, quadrature=12, physics=(1., .5, 1., 2, 0.),
        cell_values=lambda: (np.zeros((len(vertices), 6, 2)), np.full((len(vertices), 3), 1+shift)))
    wall.rule = dict(points=edges[:, :1]+t[None, :, None]*delta[:, None])
    field = SimpleNamespace(mesh=mesh, degree=2, continuous=True, num_components=1,
                            coords=nodes, array=np.zeros((len(nodes), 1, 1)))
    solver = SimpleNamespace(p=SimpleNamespace(vec=current),
        _subdict={"pressure": (pressure_is, _PressureDM())},
        _rotated_freeslip_info={"U": cached},
        boundary_normal_traction=lambda boundary, mass: (nodes, reaction))
    return solver, wall, field, (cached, current, pressure_is)


@pytest.mark.tier_b
@pytest.mark.parametrize("shift", [0., -1., .37])
def test_circular_recovery_keeps_constant_pressure_gauge_out_of_shape(shift):
    solver, wall, field, owned = _constant_pressure_fixture(shift)
    cached_values, current_values = owned[0].array.copy(), owned[1].array.copy()
    try:
        for repeat in range(2):
            checks = _recover_circular_topography(solver, wall, field, buoyancy_scale=1.)
            np.testing.assert_allclose(field.array, 0, rtol=0, atol=4e-14,
                err_msg="A constant pressure gauge must not become a circular facet-scale stress.")
            np.testing.assert_allclose(checks["reaction_pressure_gauge_offset"], -shift, rtol=0, atol=2e-15)
        np.testing.assert_array_equal(owned[0].array, cached_values)
        np.testing.assert_array_equal(owned[1].array, current_values)
    finally:
        for item in owned:
            item.destroy()


@pytest.mark.tier_b
def test_nonconstant_pressure_change_is_not_accepted_as_a_gauge():
    solver, wall, field, owned = _constant_pressure_fixture(0.)
    try:
        # Change the first owned pressure entry on each rank, rather than a constant.
        solver.p.vec.array[0] += .01
        with pytest.raises(ValueError, match="Pressure changed nonuniformly"):
            _recover_circular_topography(solver, wall, field, buoyancy_scale=1.)
    finally:
        for item in owned:
            item.destroy()
