"""A fault added by a second ``add_fault`` call keeps the faults already split
into the mesh (#1099).

Two faults in a sheared box, added by two calls. Before the fix the second
call's mesh listed the second fault's pairing only: a law on the first fault was
refused, and a solve with a law on the second alone left the first a
traction-free open crack (normal jump 0.017) with nothing said.
"""
import warnings

import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.fault_contact import (fault_pair_jumps, fault_slip,
                                                 warn_unpaired_split_faults)
from underworld3.utilities.fault_split import (_TRACK_LABEL,
                                               _carried_fault_records,
                                               _track_fault_records)

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(600)]

FLT = np.array([[0.30, 0.45], [0.70, 0.55]])
F2 = np.array([[0.30, 0.70], [0.70, 0.74]])
TRACES = {"Flt": FLT, "F2": F2}
# peak slip with a frictionless law on both faults, the two given in ONE call
# (development f8c60140, serial)
SLIP_ONE_CALL = {"Flt": 0.152175, "F2": 0.165226}
# coincident velocity node pairs on each trace
NODE_PAIRS = {"Flt": 47, "F2": 45}
# largest normal jump on Flt with a law on F2 alone: the crack the warning is about
OPENING_WITH_NO_LAW = 0.016910
# the same two faults on the adapt child of test_the_add_on_levels_keep_the_first_fault,
# given in one call: peak slip, coincident velocity node pairs
SLIP_ONE_CALL_ADAPT = {"Flt": 0.151818, "F2": 0.164490}
NODE_PAIRS_ADAPT = {"Flt": 63, "F2": 67}
# Flt alone, a frictionless law on it
SLIP_FLT_ALONE = 0.160387


def _box(regular=True):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 8, regular=regular,
        refinement=2, qdegree=2)


def _two_calls():
    first = _box().add_fault([("Flt", FLT)])
    return first, first.add_fault([("F2", F2)])


def _sheared(mesh, laws, solver=uw.systems.Stokes):
    """Simple shear on the four walls, a frictionless law on each named fault."""
    _x, y = mesh.X
    stokes = solver(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
    stokes.tolerance = 1.0e-6
    stokes.petsc_use_pressure_nullspace = True
    for wall in ("Top", "Bottom", "Left", "Right"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0), wall)
    for name in laws:
        stokes.add_fault_bc(0, name)
    return stokes


def _jump(stokes, trace):
    """(coincident node pairs on the trace, largest tangential jump, largest
    normal jump) of the velocity between them."""
    X = np.asarray(stokes.Unknowns.u.coords)
    V = np.asarray(stokes.Unknowns.u.array)[:, 0, :]
    a, b = trace
    s = np.clip(((X - a) @ (b - a)) / ((b - a) @ (b - a)), 0, 1)
    on = np.flatnonzero(np.linalg.norm(X - (a + s[:, None] * (b - a)), axis=1) < 1e-9)
    _u, inverse, count = np.unique(np.round(X[on], 10), axis=0,
                                   return_inverse=True, return_counts=True)
    t = (b - a) / np.linalg.norm(b - a)
    n = np.array([-t[1], t[0]])
    tangential = normal = 0.0
    for g in np.flatnonzero(count == 2):
        i, j = on[inverse == g]
        d = V[i] - V[j]
        tangential, normal = max(tangential, abs(d @ t)), max(normal, abs(d @ n))
    return int((count == 2).sum()), tangential, normal


def _vertex_xy(mesh, point):
    dm = mesh.dm
    section = dm.getCoordinateSection()
    xy = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, mesh.cdim)
    return xy[section.getOffset(int(point)) // mesh.cdim]


def _assert_pairing_kept(now_mesh, was_mesh, name, trace=None):
    """The pairing of ``name`` on ``now_mesh`` is the one ``was_mesh`` holds,
    point for point: as many pairs of each kind, each of two distinct points of
    one kind, the vertex pairs coincident (and on the 2-D ``trace``), and no
    mark left on the mesh."""
    was, now = was_mesh._fault_point_pairs[name], now_mesh._fault_point_pairs[name]
    assert len(now) == len(was) > 0
    assert not now_mesh.dm.hasLabel(_TRACK_LABEL)
    dm = now_mesh.dm
    vS, vE = dm.getDepthStratum(0)
    wS, wE = was_mesh.dm.getDepthStratum(0)
    n_vertex_pairs = 0
    for minus, plus in now.items():
        assert minus != plus
        assert dm.getPointDepth(minus) == dm.getPointDepth(plus)
        if vS <= minus < vE:
            n_vertex_pairs += 1
            xm, xp = _vertex_xy(now_mesh, minus), _vertex_xy(now_mesh, plus)
            assert np.linalg.norm(xm - xp) < 1e-12
            if trace is not None:
                a, b = trace
                s = np.clip(((xm - a) @ (b - a)) / ((b - a) @ (b - a)), 0, 1)
                assert np.linalg.norm(xm - (a + s * (b - a))) < 1e-9
    assert n_vertex_pairs == sum(wS <= q < wE for q in was) > 0


def test_the_second_call_keeps_every_record_of_the_first():
    first, second = _two_calls()
    assert sorted(second._fault_point_pairs) == ["F2", "Flt"]
    assert sorted(second._fault_traces) == ["F2", "Flt"]
    assert [spec[1] for spec in second._fault_specs] == ["Flt", "F2"]
    np.testing.assert_array_equal(second._fault_traces["Flt"], FLT)
    _assert_pairing_kept(second, first, "Flt", FLT)
    # the first mesh is as it was
    assert sorted(first._fault_point_pairs) == ["Flt"]
    assert not first.dm.hasLabel(_TRACK_LABEL)


def test_two_calls_solve_as_one_call_does():
    _first, second = _two_calls()
    stokes = _sheared(second, ["Flt", "F2"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # every fault has its law: silent
        stokes.solve()
    for name in ("Flt", "F2"):
        _s, slip, _l = fault_slip(stokes, name, stokes._rotated_freeslip_info)
        assert abs(np.abs(slip).max() - SLIP_ONE_CALL[name]) < 2e-5, name
        n_pairs, tangential, normal = _jump(stokes, TRACES[name])
        assert n_pairs == NODE_PAIRS[name]
        assert normal < 1e-10, name          # closed: the sides are tied
        assert abs(tangential - SLIP_ONE_CALL[name]) < 2e-5, name


def test_a_split_fault_with_no_law_is_named_at_solve():
    _first, second = _two_calls()
    stokes = _sheared(second, ["F2"])
    with pytest.warns(UserWarning, match="'Flt'") as caught:
        stokes.solve()
    message = "".join(str(w.message) for w in caught if "'Flt'" in str(w.message))
    assert "add_fault_bc" in message
    assert "'FltPlus', 'FltMinus'" in message
    assert "'F2'" not in message             # F2 has its law
    assert caught[0].filename == __file__    # reported at the caller's line
    # what the warning is about: the crack opens
    _n, _tangential, normal = _jump(stokes, FLT)
    assert abs(normal - OPENING_WITH_NO_LAW) < 1e-4
    # said once per solver
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stokes.solve()


def test_a_condition_on_each_side_is_not_warned_about():
    """The sides of a split fault may be driven by ordinary conditions on
    ``<name>Plus`` / ``<name>Minus`` instead of a law. One side alone leaves
    the other a free face, and that is said."""
    _first, second = _two_calls()
    stokes = _sheared(second, ["F2"])
    stokes.add_dirichlet_bc((0.0, 0.0), "FltPlus")
    with pytest.warns(UserWarning, match=r"'Flt' \(nothing on 'FltMinus'\)"):
        stokes.solve()
    stokes.add_dirichlet_bc((0.0, 0.0), "FltMinus")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stokes.solve()


@pytest.mark.parametrize("kind", ["natural", "multiplier"])
def test_a_condition_of_another_kind_on_each_side_counts(kind):
    """A traction stated on a side, and a multiplier constraint on it, are
    conditions as a Dirichlet one is. Read off the solver, with no solve."""
    first = _box().add_fault([("Flt", FLT)])
    if kind == "natural":
        stokes = _sheared(first, [])
        add = lambda side: stokes.add_natural_bc((0.0, 0.0), side)
    else:
        stokes = _sheared(first, [], solver=uw.systems.Stokes_Constrained)
        add = lambda side: stokes.add_constraint_bc(0.0, side)
    add("FltPlus")
    with pytest.warns(UserWarning, match=r"'Flt' \(nothing on 'FltMinus'\)"):
        warn_unpaired_split_faults(stokes)
    add("FltMinus")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_unpaired_split_faults(stokes)


def test_a_warning_raised_as_an_error_is_raised_again():
    """With warnings turned into errors the first solve stops. The next one
    must stop too: the fault is as free as it was."""
    first = _box().add_fault([("Flt", FLT)])
    stokes = _sheared(first, [])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _attempt in range(2):
            with pytest.raises(UserWarning, match="'Flt'"):
                warn_unpaired_split_faults(stokes)


def test_a_warning_raised_as_an_error_stops_every_rank(monkeypatch):
    """Only rank 0 shows the warning. Raised as an error it stops every rank:
    a rank that went on into the solve would wait there for rank 0."""
    first = _box().add_fault([("Flt", FLT)])
    stokes = _sheared(first, [])
    monkeypatch.setattr(uw.mpi, "rank", 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="'Flt'"):
            warn_unpaired_split_faults(stokes)
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("always")
        warn_unpaired_split_faults(stokes)
    assert not shown


def test_every_solver_is_told_under_the_default_filter():
    """Python shows a warning once per line by default. Two solvers made by
    one line are two models, and each is told (once)."""
    first = _box().add_fault([("Flt", FLT)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        for _k in range(2):
            stokes = _sheared(first, [])
            stokes.solve()
            stokes.solve()
    assert sum("'Flt'" in str(w.message) for w in caught) == 2


def test_the_thin_sheet_solve_names_a_fault_with_no_law():
    split = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1,
        qdegree=3).add_fault(("Flt", np.array([[0.3, 0.5], [0.7, 0.5]])))
    _x, y = split.X
    sheet = uw.systems.ThinSheet(split)
    sheet.constitutive_model = uw.constitutive_models.ViscousFlowModel
    sheet.constitutive_model.Parameters.shear_viscosity_0 = 1
    for wall in ("Left", "Right", "Top", "Bottom"):
        sheet.add_essential_bc([y, 0], wall)
    with pytest.warns(UserWarning, match=r"'Flt' \(nothing on"):
        sheet.solve()


def _adapt_child():
    """A refined box and an adapt child of it, finer around the two faults."""
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 6, regular=True,
        refinement=1, qdegree=2)
    metric = uw.discretisation.MeshVariable("M", base, 1, degree=1)
    near = np.abs(np.asarray(metric.coords)[:, 1] - 0.55) < 0.25
    metric.array[:, 0, 0] = np.where(near, 48.0, 12.0) ** 2
    return base.adapt(metric, max_levels=2, engine="nvb")


def _faults_on_the_tail(mesh):
    """The paired points each coarse level holds, by fault."""
    return [{name: len(pairs) for name, pairs in
             (getattr(level, "_fault_point_pairs", None) or {}).items()}
            for level in mesh._custom_mg_coarse_meshes]


# the two base levels are welded; the add-on level is split along both faults
TAIL = [{}, {}, {"Flt": 31, "F2": 33}]


def test_one_call_puts_every_fault_on_the_add_on_levels():
    """Two faults given in one call: the split level of the tail lists both,
    and the base levels beneath stay welded. Two calls leave the same."""
    both = _adapt_child().add_fault([("Flt", FLT), ("F2", F2)])
    assert _faults_on_the_tail(both) == TAIL
    second = _adapt_child().add_fault([("Flt", FLT)]).add_fault([("F2", F2)])
    assert _faults_on_the_tail(second) == TAIL


def test_the_add_on_levels_keep_the_first_fault():
    """Under an adapt child the add-on levels of the tail are split too; a level
    that carried the first fault carries both after the second call, with the
    first's pairing as it was, and the solve is the one-call solve."""
    host = _adapt_child()
    first = host.add_fault([("Flt", FLT)])
    held = [k for k, level in enumerate(first._custom_mg_coarse_meshes)
            if "Flt" in (getattr(level, "_fault_point_pairs", None) or {})]
    assert held
    second = first.add_fault([("F2", F2)])
    assert sorted(second._fault_point_pairs) == ["F2", "Flt"]
    _assert_pairing_kept(second, first, "Flt", FLT)
    for k in held:
        level = second._custom_mg_coarse_meshes[k]
        assert sorted(level._fault_point_pairs) == ["F2", "Flt"]
        _assert_pairing_kept(level, first._custom_mg_coarse_meshes[k], "Flt", FLT)

    stokes = _sheared(second, ["Flt", "F2"])
    stokes.solve()
    for name in ("Flt", "F2"):
        _s, slip, _l = fault_slip(stokes, name, stokes._rotated_freeslip_info)
        assert abs(np.abs(slip).max() - SLIP_ONE_CALL_ADAPT[name]) < 2e-5, name
        n_pairs, _tangential, normal = _jump(stokes, TRACES[name])
        assert n_pairs == NODE_PAIRS_ADAPT[name]
        assert normal < 1e-10, name


CROSSING = [[0.40, 0.30], [0.60, 0.70]]
T_JUNCTION = [[0.50, 0.50], [0.60, 0.80]]        # a tip on the first
SHARED_TIP = [[0.70, 0.55], [0.85, 0.70]]
# what the split says of a junction (its own refusal, older than the record),
# and what the record check says
JUNCTION = "already-split fault"
REFINED = r"[1-9]\d* recorded point\(s\) were refined or removed"
MOVED = r"[1-9]\d* node\(s\) moved"


@pytest.mark.parametrize("regular, trace, error, refusal", [
    (True, CROSSING, ValueError, JUNCTION),
    (True, T_JUNCTION, ValueError, JUNCTION),
    (True, SHARED_TIP, ValueError, JUNCTION),
    # the cut bisected the first's edges
    (False, CROSSING, RuntimeError, REFINED),
    # the tip pulled a node of the first
    (False, T_JUNCTION, RuntimeError, MOVED),
    (False, SHARED_TIP, ValueError, JUNCTION),
], ids=["crossing", "t_junction", "shared_tip", "crossing_irregular",
        "t_junction_irregular", "shared_tip_irregular"])
def test_a_second_fault_that_touches_the_first_is_refused(regular, trace, error,
                                                          refusal):
    """Refused, never added with the first fault's record lost or changed. Where
    the new trace lands on the first fault's own nodes the split refuses the
    junction, as it does within one call; where the cut changed the first fault
    on the way, the record check does."""
    first = _box(regular).add_fault([("Flt", FLT)])
    with pytest.raises(error, match=refusal):
        first.add_fault([("F2", np.array(trace))])


def test_the_record_is_carried_only_untouched():
    """The check itself: the marks read back as they were give the pairing
    back; a mark lost, a mark on more than one point (a bisected edge) and a
    marked node that moved are each refused."""
    first = _box().add_fault([("Flt", FLT)])
    dm, tracked = _track_fault_records(first)
    pairs, _normals = _carried_fault_records(dm, tracked, ["new"])
    assert pairs == first._fault_point_pairs
    assert not dm.hasLabel(_TRACK_LABEL)

    vertex = int(tracked["points"][tracked["is_vertex"]][0])
    edge = int(tracked["points"][~tracked["is_vertex"]][0])

    dm, tracked = _track_fault_records(first)
    label = dm.getLabel(_TRACK_LABEL)
    label.clearValue(edge, label.getValue(edge))
    with pytest.raises(RuntimeError, match="1 recorded point.s. were refined "
                                           "or removed and 0 node.s. moved"):
        _carried_fault_records(dm, tracked, ["new"])

    dm, tracked = _track_fault_records(first)
    label = dm.getLabel(_TRACK_LABEL)
    label.setValue(dm.getHeightStratum(0)[0], label.getValue(edge))
    with pytest.raises(RuntimeError, match="1 recorded point.s. were refined "
                                           "or removed and 0 node.s. moved"):
        _carried_fault_records(dm, tracked, ["new"])

    # a marked node 1e-6 from where it was (on a copy: a clone shares the
    # coordinates of the mesh it was cloned from). A paired node, then a tip:
    # the two tips are on the chain and in no pair, and are marked all the same
    paired = {int(q) for pair in first._fault_point_pairs["Flt"].items()
              for q in pair}
    tips = [int(q) for q in tracked["points"][tracked["is_vertex"]]
            if int(q) not in paired]
    assert len(tips) == 2
    for node in (vertex, tips[0]):
        dm, tracked = _track_fault_records(first)
        moved = dm.getCoordinatesLocal().copy()
        at = dm.getCoordinateSection().getOffset(node)
        moved.setValue(at, moved.getValue(at) + 1.0e-6)
        moved.assemble()
        dm.setCoordinatesLocal(moved)
        with pytest.raises(RuntimeError, match="0 recorded point.s. were refined "
                                               "or removed and 1 node.s. moved"):
            _carried_fault_records(dm, tracked, ["new"])
    assert not first.dm.hasLabel(_TRACK_LABEL)


def test_a_second_fault_a_cell_from_the_first_is_added():
    """Clear of the first by one cell (h = 1/32): added, the first intact."""
    first = _box().add_fault([("Flt", FLT)])
    second = first.add_fault([("F2", np.array([[0.50, 0.535], [0.60, 0.80]]))])
    assert sorted(second._fault_point_pairs) == ["F2", "Flt"]
    _assert_pairing_kept(second, first, "Flt", FLT)


def test_a_conforming_surface_keeps_the_split_fault():
    """A surface cut into a split mesh keeps the fault's record (it was
    dropped, with the fault left split), and a cut that would refine the
    fault's own edges is refused."""
    first = _box().add_fault([("Flt", FLT)])
    surface = uw.meshing.Surface("S", first, np.array([[-0.1, 0.80], [1.1, 0.84]]))
    child = first.add_conforming_surface(surface)
    assert sorted(child._fault_point_pairs) == ["Flt"]
    _assert_pairing_kept(child, first, "Flt", FLT)
    stokes = _sheared(child, ["Flt"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stokes.solve()
    _s, slip, _l = fault_slip(stokes, "Flt", stokes._rotated_freeslip_info)
    assert abs(np.abs(slip).max() - SLIP_FLT_ALONE) < 2e-5
    n_pairs, _tangential, normal = _jump(stokes, FLT)
    assert n_pairs == NODE_PAIRS["Flt"] and normal < 1e-10

    irregular = _box(regular=False).add_fault([("Flt", FLT)])
    across = uw.meshing.Surface("X", irregular,
                                np.array([[0.45, -0.1], [0.55, 1.1]]))
    with pytest.raises(RuntimeError, match=REFINED):
        irregular.add_conforming_surface(across)


# --------------------------------------------------------------------------- #
#  3-D: sheets
# --------------------------------------------------------------------------- #
def _disc(centre, normal, radius=0.15, h=0.06):
    centre, normal = np.asarray(centre, float), np.asarray(normal, float)
    normal = normal / np.linalg.norm(normal)
    e1 = np.cross(normal, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    n = int(np.ceil(2 * np.pi * radius / h))
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.vstack([centre, centre + radius * (np.outer(np.cos(a), e1)
                                                   + np.outer(np.sin(a), e2))])
    triangles = np.array([(0, 1 + i, 1 + (i + 1) % n) for i in range(n)])
    return points, triangles


NORMAL_A = np.array([0.3, 0.5, 0.81]) / np.linalg.norm([0.3, 0.5, 0.81])
CENTRE_A = np.array([0.5, 0.5, 0.30])
SHEET_A = ("FA",) + _disc(CENTRE_A, NORMAL_A)
SHEET_B = ("FB",) + _disc([0.5, 0.5, 0.70], [-0.2, 0.4, 0.9])
SHEET_THROUGH_A = ("FC",) + _disc([0.5, 0.5, 0.36], [0.5, -0.3, 0.8])
# what the sheet placement says when its cavity meets a wall; the two sides of
# a fault already split are walls to it
WALLS = (r"'FC' could not be placed: .*domain wall.* already carries the split "
         r"fault\(s\) \['FA'\], and the placement treats their two sides as "
         r"walls too: the wall it names may be one of them")


def _box_3d():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0,) * 3, maxCoords=(1.0,) * 3, cellSize=0.12,
        regular=False, qdegree=2, refinement=1)


@pytest.fixture(scope="module")
def first_3d():
    return _box_3d().add_fault([SHEET_A])


def _peak_slips_3d(mesh):
    """Peak slip on each sheet, a frictionless law on both. The solve is
    silent and neither fault opens."""
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    drive = (-(mesh.X[0] - 0.5), mesh.X[1] - 0.5, 0.0)
    for wall in ("Bottom", "Top", "Right", "Left", "Front", "Back"):
        stokes.add_dirichlet_bc(drive, wall)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-6
    for name in ("FA", "FB"):
        stokes.add_fault_bc(0, name)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        stokes.solve()
    slips = {}
    for name in ("FA", "FB"):
        _x, jump, normal = fault_pair_jumps(stokes, name,
                                            stokes._rotated_freeslip_info)
        assert np.abs(np.einsum("ij,ij->i", jump, normal)).max() < 1e-12, name
        slips[name] = np.linalg.norm(jump, axis=1).max()
    return slips


def test_two_calls_solve_as_one_call_does_3d(first_3d):
    """Before the fix the second call failed in the sheet clip, which read the
    two sides of the first fault as domain boundary. The slips are compared
    with those of the two sheets given in one call, solved here: the box's
    mesh, and the slip with it, differs between hosts (0.1452 and 0.1482 on
    FA)."""
    second = first_3d.add_fault([SHEET_B])
    assert sorted(second._fault_point_pairs) == ["FA", "FB"]
    _assert_pairing_kept(second, first_3d, "FA")
    # the mesh between the placement and the split holds the record as well
    assert sorted(second.parent._fault_point_pairs) == ["FA"]
    assert not second.parent.dm.hasLabel(_TRACK_LABEL)

    one_call = _peak_slips_3d(_box_3d().add_fault([SHEET_A, SHEET_B]))
    two_calls = _peak_slips_3d(second)
    for name in ("FA", "FB"):
        assert one_call[name] > 0.05, name
        assert abs(two_calls[name] - one_call[name]) < 2e-5, name


def test_a_sheet_through_the_first_is_refused_3d(first_3d):
    with pytest.raises(RuntimeError, match=WALLS):
        first_3d.add_fault([SHEET_THROUGH_A])


def test_a_sheet_needs_wall_clearance_from_the_first_3d(first_3d):
    """A sheet parallel to the first, one cell (0.06) off it, is refused with
    the first fault named; three cells off it is added. Measured: refused at
    0.06 to 0.12, added from 0.15."""
    near = ("FC",) + _disc(CENTRE_A + 0.06 * NORMAL_A, NORMAL_A)
    with pytest.raises(RuntimeError, match=WALLS):
        first_3d.add_fault([near])
    clear = ("FC",) + _disc(CENTRE_A + 0.18 * NORMAL_A, NORMAL_A)
    second = first_3d.add_fault([clear])
    assert sorted(second._fault_point_pairs) == ["FA", "FC"]
    _assert_pairing_kept(second, first_3d, "FA")


def test_a_conforming_sheet_keeps_the_split_fault_3d(first_3d):
    child = first_3d.add_conforming_sheet(SHEET_B[1], SHEET_B[2], "S")
    assert sorted(child._fault_point_pairs) == ["FA"]
    _assert_pairing_kept(child, first_3d, "FA")


def test_an_outcropping_sheet_on_a_split_mesh_3d(first_3d):
    """The outcrop frame looks every wall vertex up in the domain boundary, and
    the sides of the first fault are not in it: a sheet that reaches the top
    wall is placed, with the first fault kept. As a fault it is refused for the
    reason one call gives, that a fault may not daylight."""
    points, triangles = _disc([0.5, 0.5, 0.95], [1.0, 0.0, 0.2])
    child = first_3d.add_conforming_sheet(points, triangles, "S")
    assert sorted(child._fault_point_pairs) == ["FA"]
    _assert_pairing_kept(child, first_3d, "FA")
    with pytest.raises(ValueError, match="daylights"):
        first_3d.add_fault([("FC", points, triangles)])


def test_only_a_split_fault_vertex_may_be_missing_from_the_boundary():
    """The outcrop frame reads each wall vertex off the gathered boundary. One
    the boundary does not hold is an error, unless it was left out as a vertex
    of a split fault: then it is neither deletable nor near the outcrop."""
    from underworld3.utilities.place_surface import _outcrop_frame_3d
    tet = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    stray = np.array([[0.2, 0.2, 0.2]])
    nodes = np.vstack([tet, stray])
    chain = np.array([[0.2, 0.2, 0.0], [0.4, 0.3, 0.0]])    # on the face z = 0
    frame = dict(X=nodes, on_wall=np.ones(5, dtype=bool), dom_verts=tet,
                 dom_tris=faces, region=np.arange(4), skin_xyz=chain,
                 band_facets=np.array([[0, 1]]))
    with pytest.raises(RuntimeError, match="missing from the gathered boundary"):
        _outcrop_frame_3d(**frame)
    deletable, near, _regions = _outcrop_frame_3d(**frame, slit_verts=stray)
    assert not deletable[4] and not near[4]
    assert near[:3].all()


def test_only_a_wall_refusal_is_put_down_to_the_first_fault_3d(first_3d, monkeypatch):
    """A placement failure that is not about a wall is passed on as it is."""
    from underworld3.utilities import place_surface

    def fails(*_args, **_kwargs):
        raise RuntimeError("the sheet could not be meshed")

    monkeypatch.setattr(place_surface, "place_sheet", fails)
    with pytest.raises(RuntimeError) as refusal:
        first_3d.add_fault([SHEET_B])
    assert str(refusal.value) == "the sheet could not be meshed"
