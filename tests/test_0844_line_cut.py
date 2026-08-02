"""Conforming surfaces added on top of an existing mesh
(:mod:`underworld3.utilities.line_cut`, :meth:`Mesh.add_conforming_surface`).

Every edge the surface crosses is split **at the crossing point**, so the surface
becomes a chain of element edges: no element straddles it, a material property can
be assigned per cell and be exactly right, and the surface can carry a boundary
condition because it is a labelled set of facets.

It drives the compiled ``uwnvb_bisect`` transform in **passes of pairwise-
independent edges**. The transform can also split two edges of one triangle at
once, which produces the whole cut in a single pass — correct in serial, and
wrong in parallel (it leaves the child point star-forest inconsistent and
``Mesh()`` aborts at np>=3). Independent single splits still build the cut: the
second pass joins its new vertex to the OPPOSITE vertex of the cell, which is the
first pass's new vertex.

What is asserted, and why each would have caught a defect found while building
this:

- **the geometric property** — every segment of the line between consecutive
  crossings is an edge of the mesh. This is the thing being built; the stress
  result is a consequence, so it is asserted first and separately.
- **no straddling cell** — no cell has vertices on both sides. This is the
  property a cell-wise viscosity needs in order to be *correct*, as opposed to
  merely smooth.
- **the cut is exactly on the line** — inserted vertices lie on it to machine
  precision, not merely close.
- **a vertex ON the line is used, not split beside** — gmsh puts boundary nodes
  at multiples of the cell size, so an interface at x=0.5 has vertices ~1e-12
  from it. An absolute "on the line" test on a knife edge missed them, split the
  edge alongside, and produced a cell of area 1e-24 with a zero angle. The
  along-edge snap fraction is what fixes it, and this test is what caught it.
- **no inverted cell**, at any snap fraction the cut accepts.
- **refusals are refusals** — a line ending inside the mesh is rejected rather
  than silently bisected without cutting, which would give a mesh that looks
  plausible and still leaks stress.
- **the base mesh is untouched** — the surface's position is a design variable, so
  re-cutting a moved surface against the same fixed base has to be possible;
- **a boundary condition applies on the surface** — a label is only useful if a
  solver actually constrains those DOFs, which is the point of the feature.

Partition-independence and the parallel BC solve are in
``tests/parallel/ptest_0844_line_cut_parallel.py``; the serial references it
asserts against are produced here.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.line_cut import (CUT_LABEL, cell_areas,
                                            cut_along_lines, min_angles)

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]

SLANTED = np.array([[-0.2, 0.317], [1.2, 0.683]])
VERTICAL = np.array([[0.5, -0.2], [0.5, 1.2]])

# Mirrored in tests/parallel/ptest_0844_line_cut_parallel.py.
SERIAL_VERTICES = 224
SERIAL_CELLS = 396
SERIAL_COORD_SHA = "c68821fc041cf94c"
SERIAL_BC_INTEGRAL = 0.3807400201042878


def _box(cell_size=1 / 16):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=cell_size, regular=False, qdegree=2)


def _coords(dm):
    return np.asarray(dm.getCoordinatesLocal().array).reshape(-1, 2)


def _signed_distance(pts, line):
    A, B = np.asarray(line[0], float), np.asarray(line[-1], float)
    d = B - A
    nrm = np.array([-d[1], d[0]]) / np.hypot(*d)
    return (np.atleast_2d(pts) - A) @ nrm


def _cell_vertex_indices(dm):
    vS, vE = dm.getDepthStratum(0)
    cS, cE = dm.getHeightStratum(0)
    return np.array([[int(p) - vS for p in dm.getTransitiveClosure(c)[0]
                      if vS <= p < vE] for c in range(cS, cE)])


@pytest.mark.parametrize("line", [SLANTED, VERTICAL])
def test_line_becomes_a_chain_of_mesh_edges(line):
    """Consecutive crossings are joined by a mesh edge, and it is labelled."""
    cut, info = cut_along_lines(_box().dm, [line])

    X = _coords(cut)
    s = _signed_distance(X, line).ravel()
    on = np.flatnonzero(np.abs(s) < 1e-11)
    assert len(on) == info["n_split"] + info["n_snapped"]

    edges = {frozenset(int(v) - cut.getDepthStratum(0)[0] for v in cut.getCone(e)): e
             for e in range(*cut.getDepthStratum(1))}
    labelled = set(cut.getLabel(CUT_LABEL).getStratumIS(1).getIndices())

    A, B = np.asarray(line[0], float), np.asarray(line[-1], float)
    d = B - A
    order = on[np.argsort(((X[on] - A) @ d) / (d @ d))]
    for u, v in zip(order[:-1], order[1:]):
        e = edges.get(frozenset((int(u), int(v))))
        assert e is not None, "a segment of the line is not a mesh edge"
        assert e in labelled, "a cut edge is not labelled"


@pytest.mark.parametrize("line", [SLANTED, VERTICAL])
def test_no_cell_straddles_the_line(line):
    """The property a cell-wise viscosity needs to be correct, not just smooth."""
    cut, _info = cut_along_lines(_box().dm, [line])
    s = _signed_distance(_coords(cut), line).ravel()[_cell_vertex_indices(cut)]
    straddling = ((s > 1e-11).any(axis=1) & (s < -1e-11).any(axis=1)).sum()
    assert straddling == 0


def test_cut_vertices_lie_exactly_on_the_line():
    cut, info = cut_along_lines(_box().dm, [SLANTED])
    s = np.abs(_signed_distance(_coords(cut), SLANTED).ravel())
    assert np.sort(s)[:info["n_split"] + info["n_snapped"]].max() < 1e-13


def test_vertices_already_on_the_line_are_used_not_split_beside():
    """gmsh puts nodes at multiples of the cell size, so x=0.5 hits vertices.

    Splitting the edge next to such a vertex gives a degenerate cell. Before the
    along-edge snap criterion this produced an area of 1e-24 and a 0.00 degree
    angle, which no positivity check catches because the area is still positive.
    """
    cut, info = cut_along_lines(_box().dm, [VERTICAL])
    assert info["n_snapped"] > 0, "the x=0.5 interface should meet mesh vertices"
    assert info["min_angle"] > 5.0
    assert info["min_area"] > 1e-8


@pytest.mark.parametrize("snap_frac", [0.0, 0.05, 0.1, 0.2])
def test_no_inverted_cells(snap_frac):
    cut, _info = cut_along_lines(_box().dm, [SLANTED], snap_frac=snap_frac)
    assert (cell_areas(cut) > 0.0).all()
    assert (min_angles(cut) > 0.0).all()


def test_snapping_raises_the_worst_angle():
    """The tolerance has to actually buy something, or it is just a knob."""
    _c0, no_snap = cut_along_lines(_box().dm, [SLANTED], snap_frac=0.0)
    _c1, snapped = cut_along_lines(_box().dm, [SLANTED], snap_frac=0.2)
    assert snapped["min_angle"] > no_snap["min_angle"]


def test_a_line_ending_inside_the_mesh_is_refused():
    """A tip bisects without cutting; refusing beats mis-meshing it silently."""
    with pytest.raises(ValueError, match="entered but not left"):
        cut_along_lines(_box().dm, [np.array([[-0.2, 0.4], [0.5, 0.5]])])


def test_the_base_mesh_is_not_modified():
    """The line is a design variable: the base must survive being cut against."""
    base = _box()
    before_cells = base.dm.getHeightStratum(0)[1] - base.dm.getHeightStratum(0)[0]
    before_coords = _coords(base.dm).copy()

    cut_along_lines(base.dm, [SLANTED])
    cut_along_lines(base.dm, [np.array([[-0.2, 0.5], [1.2, 0.5]])])

    after_cells = base.dm.getHeightStratum(0)[1] - base.dm.getHeightStratum(0)[0]
    assert after_cells == before_cells
    assert np.array_equal(_coords(base.dm), before_coords)


def test_surface_becomes_a_named_boundary():
    """The delivered feature: the surface can carry a boundary condition."""
    base = _box()
    cut = base.add_conforming_surface(SLANTED, name="Fault")

    assert cut.parent is base
    assert "Fault" in [b.name for b in cut.boundaries]
    value = cut.boundaries["Fault"].value
    assert cut.dm.getLabel("Fault").getStratumSize(value) > 0
    # UW_Boundaries is what the solver reads when resolving a boundary by name.
    assert cut.dm.getLabel("UW_Boundaries").getStratumSize(value) > 0


def test_a_dirichlet_condition_applies_on_the_surface():
    """A label is only useful if a solver actually constrains those DOFs."""
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 12,
        regular=False, qdegree=3, refinement=1)
    mesh = base.add_conforming_surface(VERTICAL, name="Fault")

    u = uw.discretisation.MeshVariable("u_bc", mesh, 1, degree=1)
    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 0.0
    for b in ("Left", "Right", "Top", "Bottom"):
        poisson.add_dirichlet_bc(0.0, b)
    poisson.add_dirichlet_bc(1.0, "Fault")
    poisson.solve()

    X, vals = np.asarray(u.coords), np.asarray(u.data[:, 0])
    on = np.abs(X[:, 0] - 0.5) < 1e-11
    assert on.sum() > 0
    assert np.allclose(vals[on], 1.0, atol=1e-10), "the surface BC was not applied"
    # And the solution is not simply the BC everywhere: the interior responds.
    interior = (~on) & (X[:, 0] > 0.1) & (X[:, 0] < 0.4)
    assert 0.0 < vals[interior].max() < 1.0


def test_second_surface_can_be_added_by_chaining():
    base = _box()
    one = base.add_conforming_surface(SLANTED, name="Fault")
    two = one.add_conforming_surface(np.array([[-0.2, 0.12], [1.2, 0.12]]),
                                     name="Moho")
    names = [b.name for b in two.boundaries]
    assert "Fault" in names and "Moho" in names
    assert two.dm.getLabel("Fault").getStratumSize(
        two.boundaries["Fault"].value) > 0, "the first surface was lost"


def test_a_duplicate_surface_name_is_refused():
    base = _box()
    one = base.add_conforming_surface(SLANTED, name="Fault")
    with pytest.raises(ValueError, match="already has a boundary"):
        one.add_conforming_surface(VERTICAL, name="Fault")


def test_serial_reference_for_parallel_confluence():
    """The numbers ``tests/parallel/ptest_0844_line_cut_parallel.py`` asserts.

    Kept here so a deliberate change to the contract shows up as a failure in the
    serial suite, rather than as a mysterious parallel-only failure.
    """
    import hashlib

    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 12,
        regular=False, qdegree=3)
    cut = base.add_conforming_surface(SLANTED, name="Fault")

    dm = cut.dm
    vS, vE = dm.getDepthStratum(0)
    cS, cE = dm.getHeightStratum(0)
    X = _coords(dm)[: vE - vS]
    Xs = X[np.lexsort((X[:, 1], X[:, 0]))]

    assert (vE - vS, cE - cS) == (SERIAL_VERTICES, SERIAL_CELLS)
    assert hashlib.sha256(np.round(Xs, 9).tobytes()).hexdigest()[:16] == SERIAL_COORD_SHA

    # The BC-solve reference the parallel file asserts against, at the same tight
    # tolerance, so the two files cannot drift apart silently.
    bc_base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 12,
        regular=False, qdegree=3, refinement=1)
    bc_mesh = bc_base.add_conforming_surface(VERTICAL, name="Fault")
    w = uw.discretisation.MeshVariable("u_ref", bc_mesh, 1, degree=1)
    poisson = uw.systems.Poisson(bc_mesh, u_Field=w)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 1.0
    for b in ("Left", "Right", "Top", "Bottom"):
        poisson.add_dirichlet_bc(0.0, b)
    poisson.add_dirichlet_bc(1.0, "Fault")
    poisson.petsc_options["ksp_rtol"] = 1.0e-14
    poisson.petsc_options["snes_rtol"] = 1.0e-14
    poisson.solve()
    assert abs(uw.maths.Integral(bc_mesh, w.sym[0]).evaluate()
               - SERIAL_BC_INTEGRAL) < 1e-12


def _pull_vertex_to(dm, target):
    """Move the nearest vertex onto `target` — how a tip or junction is placed."""
    out = dm.clone()
    vec = out.getCoordinatesLocal()
    arr = np.asarray(vec.array).reshape(-1, 2).copy()
    arr[int(np.argmin(np.linalg.norm(arr - target, axis=1)))] = target
    new = vec.duplicate()
    new.array[:] = arr.reshape(-1)
    out.setCoordinatesLocal(new)
    return out


@pytest.mark.parametrize("branches", [
    # Y: three arms from one junction. Two of them START there, which is the case
    # that failed.
    ([[-0.2, 0.20], [0.5, 0.5]], [[0.5, 0.5], [1.2, 0.30]],
     [[0.5, 0.5], [0.55, 1.2]]),
    # T: one fault abutting another.
    ([[-0.2, 0.34], [1.2, 0.66]], [[0.5, 0.5], [0.62, 1.2]]),
    # X: two faults crossing.
    ([[-0.2, 0.22], [1.2, 0.78]], [[0.30, -0.2], [0.70, 1.2]]),
])
def test_a_fault_network_cuts_at_a_shared_junction(branches):
    """Branching, abutting and crossing faults, joined at a shared vertex.

    A junction is the same problem as a tip: a distinguished point of the network
    that has to coincide with a mesh vertex, after which every branch arrives at
    the already-legal "one crossed edge, one on-surface corner" case.

    The Y case regressed on a real defect. `_resolve_snapping` initialised its
    on-surface set to all-False and only added vertices it decided to SNAP, so a
    vertex ALREADY on the surface — a junction — was invisible: the edges
    radiating from it have signed distance exactly zero and register no strict
    sign change, so nothing proposes them. The validation then read such a cell as
    "entered but not left" and refused a legal branch.
    """
    junction = np.array([0.5, 0.5])
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 20,
        regular=False, qdegree=3)
    dm = _pull_vertex_to(base.dm, junction)

    for k, br in enumerate(branches):
        dm, _info = cut_along_lines(dm, [np.asarray(br, dtype=float)],
                                    label=f"F{k}", label_value=20 + k)

    X = _coords(dm)
    vS = dm.getDepthStratum(0)[0]
    edges = {frozenset(int(v) - vS for v in dm.getCone(e)): e
             for e in range(*dm.getDepthStratum(1))}

    zone = set()
    cS, cE = dm.getHeightStratum(0)
    for k, br in enumerate(branches):
        a, b = np.asarray(br[0], float), np.asarray(br[-1], float)
        d = b - a
        n = np.array([-d[1], d[0]]) / np.hypot(*d)
        s = (X - a) @ n
        u = ((X - a) @ d) / (d @ d)
        on = np.flatnonzero((np.abs(s) < 1e-10) & (u > -1e-9) & (u < 1.0 + 1e-9))
        order = on[np.argsort(u[on])]
        labelled = set(dm.getLabel(f"F{k}").getStratumIS(20 + k).getIndices())
        for p, q in zip(order[:-1], order[1:]):
            e = edges.get(frozenset((int(p), int(q))))
            assert e is not None, f"branch {k}: a segment is not a mesh edge"
            assert e in labelled, f"branch {k}: a segment is not labelled"
        for e in labelled:
            zone.update(int(c) for c in dm.getSupport(e) if cS <= c < cE)

    # The fault zone of a network is the UNION of its branch zones — no geometry
    # to reconcile at the junction, which is why this route suits networks.
    assert 0 < len(zone) < cE - cS
    assert (cell_areas(dm) > 0.0).all()
