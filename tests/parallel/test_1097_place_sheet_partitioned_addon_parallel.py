"""A 3-D sheet placed and split on a partitioned adapt child, in parallel.

The fault workflow is base -> ``adapt`` toward the sheet -> ``add_conforming_sheet``
-> ``split_fault``. Before the add-on partition the placement gathered the
sheet's region itself, with a mask that did not cover the cavity's growth rounds,
so on a partitioned adapt child the carve touched a shared point and the surgery
rank refused ("the gather mask under-reached") where the same placement passes
serially (measured 2026-10-17 on the box fault model at 8 ranks, and on this
fixture at 2 ranks). The split then moved the fault's star again.

Now ``add_conforming_sheet`` partitions the add-on for the sheet first — the
placement's own mask with two growth layers, on every add-on level — so the
placement gathers nothing, the split moves nothing, the placed mesh keeps the
domain volume and one orientation (as the serial test of the same fixture,
``tests/test_0867``, asserts), the facet and pair counts are the serial ones
(this module's ``__main__`` run at np = 1 on the same host), and the fault
solve runs on the geometric tail with no transfer miss.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from fault_fixtures import H_NEAR, base_3d, disc_sheet, graded_disc_metric
from serial_reference import emit, serial_reference
from underworld3.utilities.place_surface import _cell_volumes_signed6

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(1500)]


def _placed_and_split():
    from underworld3.utilities.fault_split import split_fault
    base = base_3d()
    child = base.adapt(graded_disc_metric(), max_levels=1)
    pts, tris = disc_sheet()
    placed = child.add_conforming_sheet(pts, tris, "Fault", size=H_NEAR)
    return base, placed, split_fault(placed, "Fault")


def _counts(placed, split):
    """The quantities compared with the serial run: sheet facets, pairs, cells."""
    comm = uw.mpi.comm
    return [float(placed._surface_info["n_surface_facets"]),
            float(comm.allreduce(len(split._fault_point_pairs["Fault"]))),
            float(comm.allreduce(split.dm.getHeightStratum(0)[1]
                                 - split.dm.getHeightStratum(0)[0]))]


def test_sheet_placement_gathers_nothing_and_the_split_moves_nothing():
    from underworld3.utilities.fault_split import _fault_labels_touch_seam
    comm = uw.mpi.comm
    base, placed, split = _placed_and_split()
    # the placement ran on a rank-interior region of the partitioned add-on
    assert placed._surface_info["n_gathered"] == 0
    assert placed.parent._addon_partition["units_moved"] > 0
    v6 = _cell_volumes_signed6(placed.dm)
    volume = comm.allreduce(float(np.abs(v6).sum()) / 6.0)
    assert abs(volume - 1.0) < 1e-9
    sign_ok = bool((np.sign(v6) == np.sign(v6[0])).all()) if len(v6) else True
    assert comm.allreduce(sign_ok, op=min)
    # the split found its star rank-interior and moved nothing
    assert split.parent is placed
    labels = [(f"Fault{side}", int(split.boundaries[f"Fault{side}"].value))
              for side in ("Plus", "Minus")]
    assert not _fault_labels_touch_seam(split.dm, labels)
    # the serial run of this fixture: the same sheet facets and pairs; the
    # cell count within the fill's partition noise (gmsh refills the cavity
    # from a slightly different ring)
    reference = serial_reference(__file__, "counts")
    facets, pairs, cells = _counts(placed, split)
    ref_facets, ref_pairs, ref_cells = reference["values"]
    assert facets == ref_facets and pairs == ref_pairs, (facets, pairs, reference)
    assert abs(cells - ref_cells) <= 0.02 * ref_cells, (cells, reference)
    # the tail rode through: this fixture has no intermediate add-on level, so
    # it is exactly the base's static wraps, the adapt child being the first
    # level above them (the same objects, never moved)
    tail = split._custom_mg_coarse_meshes
    wraps = base._coarse_level_meshes()
    assert tail is not None and len(tail) == len(wraps)
    assert all(t is w for t, w in zip(tail, wraps))


def test_fault_solve_on_the_placed_split_runs_on_the_geometric_tail():
    _base, _placed, split = _placed_and_split()
    v = uw.discretisation.MeshVariable("v", split, 3, degree=2)
    p = uw.discretisation.MeshVariable("p", split, 1, degree=1)
    stokes = uw.systems.Stokes(split, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = sympy.Matrix([0, 0, 0])
    stokes.tolerance = 1e-4
    stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Bottom")
    stokes.add_dirichlet_bc((1.0, 0.0, 0.0), "Top")
    stokes.add_fault_bc(0.0, "Fault")
    stokes.solve(zero_init_guess=True)
    info = stokes._rotated_freeslip_info
    assert info["velocity_pc"] == "custom-FMG" and info["converged"], info
    stats = stokes._custom_mg["hierarchy"].level_transfer_stats
    # the finest pair is the moved one: fetched by provenance
    assert stats[-1] is not None and stats[-1]["fetched_remote"] > 0, stats


def test_two_sheets_placed_in_turn_then_split_together():
    """The fault-network route places sheets one at a time and splits them
    together. The second placement partitions an add-on the first already
    partitioned and placed into: the first sheet stays rank-interior (its
    facets are glued), nothing is gathered, and the network splits with
    every pair whole."""
    from underworld3.utilities.fault_split import (_fault_labels_touch_seam,
                                                   split_faults)
    from fault_fixtures import CENTRE, TILTED
    comm = uw.mpi.comm
    c1, c2 = CENTRE - 0.17 * TILTED, CENTRE + 0.17 * TILTED
    base = base_3d()
    child = base.adapt(graded_disc_metric(discs=((c1, 0.15), (c2, 0.15))),
                       max_levels=1)
    p1, t1 = disc_sheet(centre=c1, radius=0.15)
    p2, t2 = disc_sheet(centre=c2, radius=0.15)
    one = child.add_conforming_sheet(p1, t1, "FltA", size=H_NEAR)
    two = one.add_conforming_sheet(p2, t2, "FltB", size=H_NEAR)
    assert one._surface_info["n_gathered"] == 0
    assert two._surface_info["n_gathered"] == 0
    split = split_faults(two, ["FltA", "FltB"])
    for name in ("FltA", "FltB"):
        assert comm.allreduce(len(split._fault_point_pairs[name])) > 0
        labels = [(f"{name}{side}", int(split.boundaries[f"{name}{side}"].value))
                  for side in ("Plus", "Minus")]
        assert not _fault_labels_touch_seam(split.dm, labels)
    v6 = _cell_volumes_signed6(split.dm)
    assert abs(comm.allreduce(float(np.abs(v6).sum()) / 6.0) - 1.0) < 1e-9


def _fingerprint(mesh):
    """Cell count and volume, from the cells themselves (the shared
    fingerprint integrates a field, and this mesh carries none)."""
    v6 = _cell_volumes_signed6(mesh.dm)
    comm = uw.mpi.comm
    return [float(comm.allreduce(len(v6))),
            comm.allreduce(float(np.abs(v6).sum()) / 6.0)]


if __name__ == "__main__":
    _base, _placed, _split = _placed_and_split()
    emit(_counts(_placed, _split), _fingerprint(_split))
