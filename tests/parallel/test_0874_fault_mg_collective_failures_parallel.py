"""Rank-local failures in the split-level transfer and the cut's inversion count
are collective (#874).

* ``fault_mg.cell_located_transfer_parallel``: a failure on one rank (no cell
  for a fine node, a singular nodal matrix in ``_node_weights``) raised there
  alone while the other ranks waited in the next allreduce. It is now raised on
  every rank, the same error, through ``fault_contact._collective_raise``.
* ``line_cut``: a surface mesh's winding convention was a rank-local majority,
  so a rank whose cells were mostly inverted flipped its own convention and the
  inversion count undercounted. On a collective path the vote is now global.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities import fault_mg, line_cut

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(300)]

TRACE = np.array([[0.3, 0.45], [0.7, 0.55]])


def _captured_transfer_call():
    """The arguments of the first parallel cell-located transfer a split-level
    hierarchy builds (W1's fixture: a fault cut into an adapt child)."""
    calls = []
    original = fault_mg.cell_located_transfer_parallel

    def spy(*args, **kwargs):
        if not calls:
            calls.append((args, kwargs))
        return original(*args, **kwargs)

    fault_mg.cell_located_transfer_parallel = spy
    try:
        base = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                                 cellSize=0.1, refinement=1, qdegree=2)
        pts = np.column_stack([np.linspace(*TRACE[:, 0], 9), np.linspace(*TRACE[:, 1], 9),
                               np.zeros(9)])
        guide = uw.meshing.Surface("guide_0874", base, pts)
        guide.discretize()
        child = base.adapt(guide.refinement_metric_function(
            h_near=0.05 / 4, h_far=0.05, width=0.08, profile="linear"), max_levels=2)
        mesh = child.add_fault([("Flt", TRACE)])
        x, y = mesh.X
        s = uw.systems.Stokes(mesh)
        s.constitutive_model = uw.constitutive_models.ViscousFlowModel
        s.constitutive_model.Parameters.shear_viscosity_0 = 1
        s.petsc_use_pressure_nullspace = True
        for wall in ("Top", "Bottom", "Left", "Right"):
            s.add_dirichlet_bc((y - 0.5, 0.0), wall)
        s.add_fault_bc(0, "Flt")
        s.solve()
    finally:
        fault_mg.cell_located_transfer_parallel = original
    assert uw.mpi.comm.allreduce(len(calls)) == uw.mpi.size, "no split-level transfer built"
    return calls[0]


def test_a_failure_on_one_rank_is_raised_on_every_rank():
    args, kwargs = _captured_transfer_call()
    original = fault_mg._node_weights

    def failing(*a, **k):
        if uw.mpi.rank == uw.mpi.size - 1:
            raise np.linalg.LinAlgError("Singular matrix (forced on one rank)")
        return original(*a, **k)

    fault_mg._node_weights = failing
    try:
        message = None
        try:
            fault_mg.cell_located_transfer_parallel(*args, **kwargs)
        except RuntimeError as err:
            message = str(err)
    finally:
        fault_mg._node_weights = original
    messages = uw.mpi.comm.allgather(message)
    assert all(m is not None and "Singular matrix" in m for m in messages), messages
    assert len(set(messages)) == 1


def _cap_triangle(centre_angle, inverted):
    """A small triangle on the unit sphere near the point at ``centre_angle``
    on the equator, wound outward (or inward if ``inverted``)."""
    th = [centre_angle, centre_angle + 0.05, centre_angle]
    ph = [0.0, 0.0, 0.05]
    P = np.array([[np.cos(p) * np.cos(t), np.cos(p) * np.sin(t), np.sin(p)]
                  for t, p in zip(th, ph)])
    return P[[0, 2, 1]] if inverted else P


def test_the_inversion_count_uses_a_global_winding_vote():
    # rank 0: enough outward triangles that outward is the global majority;
    # every other rank: two of its three inverted, so its own majority is inward
    inverted = [False] * (3 * uw.mpi.size) if uw.mpi.rank == 0 else [False, True, True]
    P = np.array([_cap_triangle(0.1 * k + uw.mpi.rank, inv) for k, inv in enumerate(inverted)])
    # the collective path's convention (before #874 there was only the
    # rank-local vote, which is what this falls back to)
    vote = getattr(line_cut, "_global_flip", None)
    areas = (line_cut._signed_twice_areas(P, vote(P)) if vote is not None
             else line_cut._signed_twice_areas(P))
    counted = uw.mpi.comm.allreduce(int((areas <= 0).sum()))
    truth = uw.mpi.comm.allreduce(int(sum(inverted)))
    assert counted == truth, (counted, truth)
