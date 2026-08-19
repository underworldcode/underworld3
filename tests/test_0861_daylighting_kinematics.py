"""How a split-node fault reaches daylight: blind, under a damage zone.

The ruling (2026-08-18, superseding the split-through-the-wall acceptance
this file first carried): split-node faults do NOT split through the
surface. A fault hits the surface region and STOPS an element or two
below, blind, with a DAMAGED region above it carrying the deformation to
the surface. The measured basis is the listric hybrid study — a contact
tip inside weak material is free, a tip at the weak zone's edge is
pinned, and slip is forced to zero at a pinned tip — plus the
hybrid-seam result that an abutting composite is worse than either pure
representation. The junction is a process zone, and the damage region IS
that process zone at the surface.

Two tests. The kinematic acceptance turns the listric rule vertical,
against a FREE surface: the same blind frictionless fault solved three
ways — bare (pinned tip), damage abutting the tip (the seam defect,
reproduced deliberately as the negative control), and damage ENCLOSING
the tip (the ruled composition) — must order strictly on near-tip slip
AND on surface localization, with the ruled margins holding (measured:
near-tip/peak 0.45 / 0.65 / 0.80; localization 37% / 44% / 50% —
~/+Simulations/blind_fault_surface_expression/). The structural test
keeps the placement side honest: an outcropping sheet's trace chain is
labelled through to the wall — under the ruling the trace is the surface
LOCATOR for the damage region and the model's own compositions, not a
split path.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.place_surface import place_sheet

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.skipif(uw.mpi.size > 1,
                                 reason="serial acceptance form")]


def test_the_trace_chain_is_labelled_through_to_the_wall():
    """The placement contract: every trace edge on the wall AND bounding
    a labelled fault face — the chain is the fault's edge, not an orphan
    polyline — with the wall's own labels restored beside it. The trace
    locates the surface expression; the ruling puts the damage region
    there, not a split."""
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.12, regular=False, qdegree=2)
    strike = np.array([1.0, 0.0, 0.0])
    dip = np.array([0.0, 0.15, -1.0])
    dip /= np.linalg.norm(dip)
    top = np.array([0.5, 0.5, 1.1])              # 0.1 ABOVE the box
    s = np.linspace(-0.22, 0.22, 5)
    d = np.linspace(0.0, 0.5, 5)
    pts = np.array([top + a * strike + b * dip for b in d for a in s])
    tris, n = [], 5
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = i * n + j, i * n + j + 1
            c, e = (i + 1) * n + j, (i + 1) * n + j + 1
            tris += [(a, b, e), (a, e, c)]
    placed, info = place_sheet(base.dm, pts, np.array(tris, dtype=np.int64),
                               label="FltA", label_value=41)
    assert info["n_trace_edges"] > 0, "the sheet left no trace"

    trace = placed.getLabel("FltA_trace")
    assert trace.getStratumSize(41) > 0
    fault = placed.getLabel("FltA")
    n_edges = 0
    for p in trace.getStratumIS(41).getIndices():
        if placed.getPointDepth(int(p)) != 1:
            continue
        n_edges += 1
        faces = [int(f) for f in placed.getSupport(int(p))]
        assert any(len(placed.getSupport(f)) == 1 for f in faces), (
            "a trace edge is not on the domain wall")
        assert any(fault.getValue(f) == 41 for f in faces), (
            "a trace edge bounds no labelled fault face; the chain is "
            "detached from the sheet")
    assert n_edges == info["n_trace_edges"]

    fS, fE = placed.getHeightStratum(1)
    vS, vE = placed.getDepthStratum(0)
    X = np.asarray(placed.getCoordinatesLocal().array).reshape(-1, 3)
    top_label = placed.getLabel("Top")
    for f in range(fS, fE):
        if len(placed.getSupport(f)) != 1:
            continue
        verts = [int(q) - vS for q in placed.getTransitiveClosure(f)[0]
                 if vS <= int(q) < vE]
        if all(X[v][2] == 1.0 for v in verts):
            assert top_label.getValue(f) >= 0, (
                "an unlabelled boundary face beside the trace")


def test_a_blind_fault_slips_to_the_surface_through_its_damage_zone():
    """The kinematic acceptance, per the ruling.

    A vertical frictionless split fault ending two cells below a free
    surface, driven in antisymmetric dip-slip, solved bare / abutted /
    ruled. The tip-inside-weak rule must reproduce vertically: strict
    ordering on near-tip slip and on surface localization, and the ruled
    margins hold. The abutted case is the deliberate negative control —
    the hybrid-seam defect this composition exists to avoid.
    """
    import sympy
    from underworld3.utilities.fault_contact import (
        add_frictionless_fault_bc, fault_slip, solve_with_fault)
    from underworld3.utilities.fault_split import split_fault
    from underworld3.utilities.line_cut import (cut_along_lines,
                                                pull_vertex_onto)

    h = 1.0 / 24.0
    y_tip = 1.0 - 2.0 * h
    tip_a = np.array([0.5, 0.35])
    tip_b = np.array([0.5, y_tip])
    w = 0.08                                    # damage half-width
    overlap = 1.5 * h

    def split_box(name="Flt"):
        base = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
            cellSize=h, regular=False, qdegree=2)
        boundaries = base._boundaries_with(name)
        dm = pull_vertex_onto(base.dm, np.vstack([tip_a, tip_b]))
        dm, _info = cut_along_lines(dm, [np.vstack([tip_a, tip_b])],
                                    label=name,
                                    label_value=boundaries[name].value)
        cut = uw.discretisation.Mesh(
            dm, simplex=True,
            coordinate_system_type=base.CoordinateSystem.coordinate_type,
            qdegree=base.qdegree, boundaries=boundaries, verbose=False)
        cut.parent = base
        cut._relationship_kind = "refinement"
        cut._refine_dofs_coincide = False
        cut.regions = base.regions
        cut._parent_mesh_version = base._mesh_version
        return split_fault(cut, name)

    def run(case, damage_floor):
        mesh = split_box()
        x, y = mesh.X
        v = uw.discretisation.MeshVariable(f"v_{case}", mesh, 2, degree=2)
        p = uw.discretisation.MeshVariable(f"p_{case}", mesh, 1, degree=0,
                                           continuous=False)
        stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
        stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
        if damage_floor is None:
            stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
        else:
            damaged = sympy.And(sympy.Abs(x - 0.5) < w, y > damage_floor)
            stokes.constitutive_model.Parameters.shear_viscosity_0 = \
                sympy.Piecewise((1e-3, damaged), (1.0, True))
        stokes.tolerance = 1e-6
        for side in ("Bottom", "Left", "Right"):
            stokes.add_dirichlet_bc((0.0, x - 0.5), side)
        add_frictionless_fault_bc(stokes, "Flt")
        info = solve_with_fault(stokes)
        assert info["converged"], f"{case}: not converged"

        _s, V, leak = fault_slip(stokes, "Flt", info)
        assert np.abs(leak).max() < 1e-10, f"{case}: the contact opens"
        Vmag = np.abs(V)
        near_tip = Vmag[-3:].max() / Vmag.max()

        coords = np.asarray(v.coords)
        on_top = np.abs(coords[:, 1] - 1.0) < 1e-9
        xs = coords[on_top, 0]
        vy = np.asarray(v.data[on_top, 1])
        order = np.argsort(xs)
        xs, vy = xs[order], vy[order]
        inside = np.abs(xs - 0.5) <= 2.0 * w
        localized = (vy[inside][-1] - vy[inside][0]) / (vy[-1] - vy[0])
        return near_tip, localized

    pinned = run("pinned", None)
    abutted = run("abutted", y_tip)
    ruled = run("ruled", y_tip - overlap)

    # Strict ordering on both measures: enclosing the tip beats the
    # abutting seam beats bare pinning (measured 0.45/0.65/0.80 and
    # 0.37/0.44/0.50; thresholds hold generous margins).
    assert ruled[0] > abutted[0] > pinned[0], (
        f"near-tip slip does not order: {pinned[0]:.3f} / {abutted[0]:.3f}"
        f" / {ruled[0]:.3f}")
    assert ruled[1] > abutted[1] > pinned[1], (
        f"surface localization does not order: {pinned[1]:.3f} / "
        f"{abutted[1]:.3f} / {ruled[1]:.3f}")
    assert ruled[0] > 0.7, "the enclosed tip still pins"
    assert pinned[0] < 0.55, (
        "the bare tip does not pin; the control cannot validate the "
        "composition")
    assert ruled[1] > 1.2 * pinned[1], (
        "the damage zone adds no surface localization")


def test_a_setback_sheet_is_blind_located_and_splittable():
    """The deliberate workflow, end to end: the job is still to figure
    out the intersection so the fault can STOP before it hits it — it
    just never meshes or splits all the way there.

    A through-running sheet placed with a setback arrives BLIND: no
    trace on the wall, its shallowest point exactly a setback below it
    (the box's offset plane is exact), the would-be intersection
    reported as ``surface_trace`` ON the true boundary — the damage
    region's locator — and the placed patch splits as it stands: the
    rim is strictly interior, so the split's daylighting refusal never
    fires.
    """
    from underworld3.utilities.fault_split import split_along_label_3d

    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.12, regular=False, qdegree=2)
    strike = np.array([1.0, 0.0, 0.0])
    dip = np.array([0.0, 0.15, -1.0])
    dip /= np.linalg.norm(dip)
    top = np.array([0.5, 0.5, 1.1])              # 0.1 ABOVE the box
    s = np.linspace(-0.22, 0.22, 5)
    d = np.linspace(0.0, 0.55, 6)
    pts = np.array([top + a * strike + b * dip for b in d for a in s])
    tris, n = [], 5
    for i in range(len(d) - 1):
        for j in range(n - 1):
            a, b = i * n + j, i * n + j + 1
            c, e = (i + 1) * n + j, (i + 1) * n + j + 1
            tris += [(a, b, e), (a, e, c)]

    setback = 0.25
    placed, info = place_sheet(base.dm, pts, np.array(tris, dtype=np.int64),
                               label="FltA", label_value=41,
                               setback=setback)
    assert info["n_trace_edges"] == 0, "a setback sheet must not outcrop"
    assert info["n_surface_facets"] > 0

    # The would-be intersection is located ON the true boundary.
    trace = np.asarray(info["surface_trace"])
    assert trace is not None and len(trace) >= 2
    assert np.abs(trace[:, 2] - 1.0).max() < 1e-9, (
        "the surface trace is off the wall it locates")

    # The blind rim: the placed sheet's shallowest vertex sits exactly a
    # setback below the wall (the box's inward-offset plane is exact).
    X = np.asarray(placed.getCoordinatesLocal().array).reshape(-1, 3)
    vS, vE = placed.getDepthStratum(0)
    fault = placed.getLabel("FltA")
    fS, fE = placed.getHeightStratum(1)
    z_max = -np.inf
    for p in fault.getStratumIS(41).getIndices():
        if not (fS <= int(p) < fE):
            continue
        for q in placed.getTransitiveClosure(int(p))[0]:
            if vS <= int(q) < vE:
                z_max = max(z_max, float(X[int(q) - vS][2]))
    assert z_max == pytest.approx(1.0 - setback, abs=1e-12)

    # And it splits as it stands — the ruling's point.
    split, _point_map, _clone_map = split_along_label_3d(
        placed, "FltA", 41, "FltAPlus", 1, "FltAMinus", 2)
    plus = split.getLabel("FltAPlus")
    assert plus is not None
    n_plus = sum(1 for p in plus.getStratumIS(1).getIndices()
                 if split.getPointDepth(int(p)) == 2)
    assert n_plus == info["n_surface_facets"], (
        f"{n_plus} Plus faces for {info['n_surface_facets']} placed")


def test_a_resampled_sheet_matches_the_mesh_it_cuts():
    """The fault's resolution is a MESH choice, not the data's.

    The authored triangulation arrives at whatever spacing the source
    provided; embedding it verbatim into a mesh of a different size
    forces slivers around every mismatched triangle. ``size=``
    re-triangulates the clipped sheet in its own plane at the embedding
    resolution — rim corners exact, faces gmsh-quality, still blind and
    still splittable.
    """
    from underworld3.utilities.fault_split import split_along_label_3d

    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.12, regular=False, qdegree=2)
    strike = np.array([1.0, 0.0, 0.0])
    dip = np.array([0.0, 0.15, -1.0])
    dip /= np.linalg.norm(dip)
    top = np.array([0.5, 0.5, 1.1])
    # Deliberately COARSE authored data: 3 x 3 nodes over the same patch.
    s = np.linspace(-0.22, 0.22, 3)
    d = np.linspace(0.0, 0.55, 3)
    pts = np.array([top + a * strike + b * dip for b in d for a in s])
    tris = []
    for i in range(2):
        for j in range(2):
            a0 = i * 3 + j
            tris += [(a0, a0 + 1, a0 + 4), (a0, a0 + 4, a0 + 3)]

    setback, size = 0.25, 0.1
    placed, info = place_sheet(base.dm, pts, np.array(tris, dtype=np.int64),
                               label="FltA", label_value=41,
                               setback=setback, size=size)
    assert info["n_trace_edges"] == 0
    assert info["n_surface_facets"] > 3 * len(tris), (
        "the resample did not refine the coarse authored sheet")

    # Face quality at gmsh level, not clip-sliver level.
    X = np.asarray(placed.getCoordinatesLocal().array).reshape(-1, 3)
    fS, fE = placed.getHeightStratum(1)
    vS, vE = placed.getDepthStratum(0)
    qmin = 1.0
    z_max = -np.inf
    for p in placed.getLabel("FltA").getStratumIS(41).getIndices():
        if not (fS <= int(p) < fE):
            continue
        verts = [int(q) - vS
                 for q in placed.getTransitiveClosure(int(p))[0]
                 if vS <= int(q) < vE]
        A, B, C = (X[v] for v in verts)
        area = 0.5 * np.linalg.norm(np.cross(B - A, C - A))
        per = (np.linalg.norm(B - A) + np.linalg.norm(C - B)
               + np.linalg.norm(A - C))
        qmin = min(qmin, 4.0 * np.sqrt(3.0) * area / per ** 2)
        z_max = max(z_max, *(float(X[v][2]) for v in verts))
    assert qmin > 0.3, f"resampled fault face quality {qmin:.3f}"
    assert z_max == pytest.approx(1.0 - setback, abs=1e-12), (
        "the resample moved the blind rim off the offset plane")

    split, _pm, _cm = split_along_label_3d(
        placed, "FltA", 41, "FltAPlus", 1, "FltAMinus", 2)
    n_plus = sum(1 for p in split.getLabel("FltAPlus").getStratumIS(1)
                 .getIndices() if split.getPointDepth(int(p)) == 2)
    assert n_plus == info["n_surface_facets"]
