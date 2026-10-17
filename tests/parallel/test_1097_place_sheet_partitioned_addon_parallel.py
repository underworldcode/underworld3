"""A 3-D sheet placed and split on a partitioned adapt child, in parallel.

The fault workflow is base -> ``adapt`` toward the sheet -> ``add_conforming_sheet``
-> ``split_fault``. Before the add-on partition the placement gathered the
sheet's region itself, with a mask that did not cover the cavity's growth rounds,
so on a partitioned adapt child the carve touched a shared point and the surgery
rank refused ("the gather mask under-reached") where the same placement passes
serially (measured 2026-10-17 on the box fault model at 8 ranks). The split then
moved the fault's star again, level by level.

Now ``add_conforming_sheet`` partitions the add-on for the sheet first — the
placement's own mask with two growth layers, on every add-on level — so the
placement gathers nothing (``n_gathered`` is zero), the split moves nothing, and
the fault solve runs on the geometric tail. The placed mesh keeps the domain
volume and one orientation, as the serial test of the same fixture asserts
(``tests/test_0867_place_sheet_adapted_3d.py``).
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.utilities.place_surface import _cell_volumes_signed6

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(1500)]

CENTRE = np.array([0.5, 0.5, 0.5])
RADIUS = 0.2
H_NEAR = 0.05
TILTED = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])


def _disc_sheet(normal, n_rim=32):
    helper = np.eye(3)[np.argmin(np.abs(normal))]
    e1 = np.cross(normal, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    a = np.linspace(0.0, 2.0 * np.pi, n_rim, endpoint=False)
    rim = CENTRE + RADIUS * (np.outer(np.cos(a), e1) + np.outer(np.sin(a), e2))
    pts = np.vstack([CENTRE, rim])
    tris = np.array([(0, 1 + i, 1 + (i + 1) % n_rim) for i in range(n_rim)],
                    dtype=np.int64)
    return pts, tris


def _graded_metric(normal):
    def metric(X):
        rel = np.asarray(X)[:, :3] - CENTRE
        zn = rel @ normal
        r = np.linalg.norm(rel - zn[:, None] * normal, axis=1)
        d = np.sqrt(np.maximum(r - RADIUS, 0.0) ** 2 + zn ** 2)
        h = np.clip(H_NEAR + d * H_NEAR / 0.4, H_NEAR, 2.0 * H_NEAR)
        return 1.0 / h**2
    return metric


def _placed_and_split():
    from underworld3.utilities.fault_split import split_fault
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.2, regular=False, qdegree=2, refinement=1)
    child = base.adapt(_graded_metric(TILTED), max_levels=1)
    pts, tris = _disc_sheet(TILTED)
    placed = child.add_conforming_sheet(pts, tris, "Fault", size=H_NEAR)
    return placed, split_fault(placed, "Fault")


def test_sheet_placement_gathers_nothing_and_the_split_moves_nothing():
    from underworld3.utilities.fault_split import _fault_labels_touch_seam
    comm = uw.mpi.comm
    placed, split = _placed_and_split()
    # the placement ran on a rank-interior region of the partitioned add-on
    assert placed._surface_info["n_gathered"] == 0
    assert placed._surface_info["n_surface_facets"] > 0
    assert placed.parent._addon_partition is not None
    v6 = _cell_volumes_signed6(placed.dm)
    volume = comm.allreduce(float(np.abs(v6).sum()) / 6.0)
    assert abs(volume - 1.0) < 1e-9
    sign_ok = bool((np.sign(v6) == np.sign(v6[0])).all()) if len(v6) else True
    assert comm.allreduce(sign_ok, op=min)
    # the split found its star rank-interior and moved nothing: the split
    # mesh's parent is the placed mesh itself, every pair is whole
    assert split.parent is placed
    n_pairs = comm.allreduce(len(split._fault_point_pairs["Fault"]))
    assert n_pairs > 0
    labels = [(f"Fault{side}", int(split.boundaries[f"Fault{side}"].value))
              for side in ("Plus", "Minus")]
    assert not _fault_labels_touch_seam(split.dm, labels)
    # the tail rode through: static base levels plus the moved add-on level
    tail = split._custom_mg_coarse_meshes
    assert tail is not None and len(tail) >= 2, tail


def test_fault_solve_on_the_placed_split_runs_on_the_geometric_tail():
    _placed, split = _placed_and_split()
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
    assert all(st is None or st["misses"] == 0 for st in stats), stats
