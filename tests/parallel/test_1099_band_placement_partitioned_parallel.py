"""The placed fault band on a partitioned add-on, in parallel.

The band route (``FaultNetwork.build`` in 3-D, ``place_fault_ribbon_2d`` in
gather mode) placed its band with ``place_thin_volume``, which gathered the
band's region onto a surgery rank; ``adopt_hierarchy`` then moved copies of
the base's coarse levels to follow it (``_colocate_level``), so every
transfer of the tail paired a level with one built for it. Both routes now
partition the add-on for the band first (``addon_partition.mark_band``, a
superset of the placement's own mask): the placement moves nothing, the tail
is the base's own levels with the partitioned host above them, and the pair
the partition separated is read from the recorded provenance.

Asserted: no cell moved in the placement (the record the placement keeps);
the tail's static levels are the base's own cached wraps, by identity, and
the level above them is the partitioned host; the split pairs are whole; the
solve runs on the geometric tail, the separated pair built by provenance.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(1800)]

H = 0.08
WIDTH = 0.04
P_A = np.array([[0.30, 0.50, 0.30], [0.70, 0.50, 0.30],
                [0.70, 0.50, 0.70], [0.30, 0.50, 0.70]])


def _network():
    fs = uw.meshing.FaultSurface("Main", P_A)
    fs.triangulate()
    net = uw.meshing.FaultNetwork([fs])
    net.prepare(h=H, ligament=1.0, verbose=False)
    net.build(width=WIDTH, realisation="split", h_far=0.24, margin_rings=0.5)
    return net


def test_band_placement_moves_nothing_and_keeps_the_base_levels():
    from underworld3.utilities.fault_split import _fault_labels_touch_seam
    net = _network()
    mesh = net.mesh
    comm = uw.mpi.comm
    assert net.info["n_moved"] == 0, net.info
    tail = mesh._custom_mg_coarse_meshes
    host = tail[-1]
    assert host._addon_partition is not None
    wraps = host.parent._coarse_level_meshes()
    assert len(tail) == len(wraps)
    assert all(t is w for t, w in zip(tail[:-1], wraps[:-1]))
    assert comm.allreduce(len(mesh._fault_point_pairs["Main"])) > 0
    labels = [(f"Main{side}", int(mesh.boundaries[f"Main{side}"].value))
              for side in ("Plus", "Minus")]
    assert not _fault_labels_touch_seam(mesh.dm, labels)


def test_band_solve_runs_on_the_tail_with_the_provenance_pair():
    net = _network()
    mesh = net.mesh
    x, y, z = mesh.X
    v = uw.discretisation.MeshVariable("vB", mesh, 3, degree=2)
    p = uw.discretisation.MeshVariable("pB", mesh, 1, degree=0, continuous=False)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = [0.0, 0.0, 0.0]
    for wall in ("Bottom", "Top", "Left", "Right", "Front", "Back"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0, 0.0), wall)
    net.apply(stokes)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1e-4
    info = net.solve(stokes)
    assert info.get("converged") and info.get("velocity_pc") == "custom-FMG", info
    stats = stokes._custom_mg["hierarchy"].level_transfer_stats
    n_static = len(mesh._custom_mg_coarse_meshes) - 1
    assert stats[n_static] is not None and "fetched_remote" in stats[n_static], stats
