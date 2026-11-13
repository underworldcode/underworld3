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
import re

import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.line_cut import (CUT_LABEL, cell_areas,
                                            cut_along_lines, min_angles,
                                            pull_vertex_onto)

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]

SLANTED = np.array([[-0.2, 0.317], [1.2, 0.683]])
VERTICAL = np.array([[0.5, -0.2], [0.5, 1.2]])

# Mirrored in tests/parallel/ptest_0844_line_cut_parallel.py.
SERIAL_VERTICES = 224
SERIAL_CELLS = 396
SERIAL_COORD_SHA = "c68821fc041cf94c"
SERIAL_BC_INTEGRAL = 0.3807400201042878
SERIAL_SURFACE_FACETS = 26
SERIAL_ZONE_SHA = "94b098f3d3153eb5"


def _box(cell_size=1 / 16):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=cell_size, regular=False, qdegree=2)


def _surf(name, mesh, points):
    """A `Surface` for `add_conforming_surface`, which takes one rather than a
    (points, name) pair: it is what `fault_metric` and
    `refinement_metric_function` already take, so one object drives both the
    refinement and the cut."""
    return uw.meshing.Surface(name, mesh, np.asarray(points, dtype=float))


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
    assert len(on) == info["n_split"] + info["n_on_surface"]

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
    assert np.sort(s)[:info["n_split"] + info["n_on_surface"]].max() < 1e-13


def test_vertices_already_on_the_line_are_used_not_split_beside():
    """gmsh puts nodes at multiples of the cell size, so x=0.5 hits vertices.

    Splitting the edge next to such a vertex gives a degenerate cell. Before the
    along-edge snap criterion this produced an area of 1e-24 and a 0.00 degree
    angle, which no positivity check catches because the area is still positive.
    """
    cut, info = cut_along_lines(_box().dm, [VERTICAL])
    assert info["n_on_surface"] > 0, "the x=0.5 interface should meet mesh vertices"
    assert info["min_angle"] > 5.0
    assert info["min_area"] > 1e-8


# Worst interior angle of the cut, per snap fraction, on the 1/16 box. This is
# the table the module docstring uses to justify the 0.10 default, so it is
# pinned rather than described.
SERIAL_MIN_ANGLE = {0.0: 1.60, 0.05: 3.88, 0.1: 6.56, 0.2: 13.93}


@pytest.mark.parametrize("snap_frac", [0.0, 0.05, 0.1, 0.2])
def test_snapping_buys_the_documented_element_quality(snap_frac):
    """The snap tolerance has to deliver the angles the default rests on.

    Asserting positivity instead would assert nothing: ``cut_along_lines``
    already raises on ``(areas <= 0).any()`` computed from the SAME
    ``cell_areas``, so an inverted cell never reaches here, and ``min_angles``
    returns ``arccos`` of a clipped value, which cannot be negative. Both
    assertions were true by construction. The worst angle is the quantity that
    actually varies, and it is what the solver pays for.
    """
    cut, info = cut_along_lines(_box().dm, [SLANTED], snap_frac=snap_frac)
    assert (cell_areas(cut) > 0.0).all()

    expected = SERIAL_MIN_ANGLE[snap_frac]
    assert info["min_angle"] == pytest.approx(expected, abs=0.05), (
        f"snap_frac={snap_frac}: worst angle {info['min_angle']:.2f} deg, "
        f"documented {expected:.2f}")


def test_the_worst_angle_rises_monotonically_with_the_snap_tolerance():
    """The knob's whole justification: more snapping, better elements."""
    angles = [cut_along_lines(_box().dm, [SLANTED], snap_frac=f)[1]["min_angle"]
              for f in (0.0, 0.05, 0.1, 0.2)]
    assert angles == sorted(angles), f"not monotone: {angles}"
    assert angles[-1] > 5 * angles[0], (
        f"snapping bought only {angles[-1] / angles[0]:.1f}x in the worst angle")


@pytest.mark.parametrize("snap_frac,line", [
    (0.0, SLANTED), (0.05, SLANTED), (0.1, SLANTED), (0.2, SLANTED),
    (0.1, VERTICAL),
])
def test_the_cut_is_one_chain(snap_frac, line):
    """``n_cut_edges == n_split + n_on_surface - 1``.

    A single line crossing the mesh cuts ONE chain, so its facets number one
    fewer than the vertices along it, and every vertex along it is either one the
    routine inserted or one already on the line. An exact connectivity check that
    costs nothing: a chain that broke in two, or a labelled edge that is not part
    of it, breaks the identity immediately.
    """
    _cut, info = cut_along_lines(_box().dm, [line], snap_frac=snap_frac)
    assert info["n_cut_edges"] == info["n_split"] + info["n_on_surface"] - 1, info


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


def test_the_surface_exists_on_the_finest_level_only():
    """The stack-on invariant: nothing below the child is cut.

    The surface's position is a design variable in an outer optimisation, so the
    base and the multigrid hierarchy resting on it have to stay fixed while the
    surface moves. The child's coarse tail is therefore the base's OWN levels,
    the same objects, carrying neither the cut nor the label.

    Nor would cutting them buy anything: custom-P sets ``pc_mg_galerkin=both``,
    so every coarse operator is PᵀAP from the FINE operator and inherits the
    material contrast whatever the coarse mesh looks like.
    """
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 12,
        regular=False, qdegree=3, refinement=2)
    tail_before = base._coarse_level_meshes()
    counts_before = [m.dm.getHeightStratum(0)[1] - m.dm.getHeightStratum(0)[0]
                     for m in tail_before]

    child = base.add_conforming_surface(_surf("Fault", base, SLANTED))

    assert child.dm.hasLabel("Fault")
    for level in child._custom_mg_coarse_meshes:
        assert not level.dm.hasLabel("Fault"), (
            "a coarse level carries the surface; the base hierarchy must be "
            "reusable unchanged when the surface moves")
    counts_after = [m.dm.getHeightStratum(0)[1] - m.dm.getHeightStratum(0)[0]
                    for m in base._coarse_level_meshes()]
    assert counts_after == counts_before, "a coarse level gained cells"

    # The child's tail is made of the base's own level objects, and the cut
    # REPLACES the finest of them rather than sitting on top of it. A cut is not
    # a refinement — it re-represents the same grid with the surface conformed —
    # so the base finest and the child have the same resolution, and keeping
    # both would record a multigrid level that coarsens nothing. That is the
    # same test `adapt` applies to an engine pass, applied by the same routine.
    assert len(child._custom_mg_coarse_meshes) == len(tail_before) - 1, (
        "the cut was kept as a level of its own; it does not coarsen the mesh "
        "it was cut from, so it should have replaced it")
    finest = child._custom_mg_coarse_meshes[-1]
    assert np.array_equal(_coords(finest.dm), _coords(base.dm_hierarchy[-2]))
    assert _coords(finest.dm).shape[0] < _coords(child.dm).shape[0], (
        "the coarse tail's finest level has as many vertices as the child, so "
        "it is not an uncut base level")


def test_surface_becomes_a_named_boundary():
    """The delivered feature: the surface can carry a boundary condition."""
    base = _box()
    cut = base.add_conforming_surface(_surf("Fault", base, SLANTED))

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
    mesh = base.add_conforming_surface(_surf("Fault", base, VERTICAL))

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
    one = base.add_conforming_surface(_surf("Fault", base, SLANTED))
    two = one.add_conforming_surface(
        _surf("Moho", one, np.array([[-0.2, 0.12], [1.2, 0.12]])))
    names = [b.name for b in two.boundaries]
    assert "Fault" in names and "Moho" in names
    assert two.dm.getLabel("Fault").getStratumSize(
        two.boundaries["Fault"].value) > 0, "the first surface was lost"


def test_a_duplicate_surface_name_is_refused():
    base = _box()
    one = base.add_conforming_surface(_surf("Fault", base, SLANTED))
    with pytest.raises(ValueError, match="already has a boundary"):
        one.add_conforming_surface(_surf("Fault", one, VERTICAL))


def test_serial_reference_for_parallel_confluence():
    """The numbers ``tests/parallel/ptest_0844_line_cut_parallel.py`` asserts.

    Kept here so a deliberate change to the contract shows up as a failure in the
    serial suite, rather than as a mysterious parallel-only failure.
    """
    import hashlib

    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 12,
        regular=False, qdegree=3)
    cut = base.add_conforming_surface(_surf("Fault", base, SLANTED))

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
    bc_mesh = bc_base.add_conforming_surface(_surf("Fault", bc_base, VERTICAL))
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

    # The fault-zone reference the parallel file compares against.
    import hashlib
    zone = cut.cells_supporting("Fault")
    assert cut.dm.getLabel("Fault").getStratumSize(
        cut.boundaries["Fault"].value) == SERIAL_SURFACE_FACETS
    assert int(zone.sum()) == 2 * SERIAL_SURFACE_FACETS
    vS_, vE_ = dm.getDepthStratum(0)
    cS_, _cE_ = dm.getHeightStratum(0)
    cen = np.array([
        _coords(dm)[[int(p) - vS_ for p in dm.getTransitiveClosure(cS_ + int(c))[0]
                     if vS_ <= p < vE_]].mean(axis=0)
        for c in np.flatnonzero(zone)])
    cen = cen[np.lexsort((cen[:, 1], cen[:, 0]))]
    assert hashlib.sha256(
        np.round(cen, 9).tobytes()).hexdigest()[:16] == SERIAL_ZONE_SHA


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
    dm = pull_vertex_onto(base.dm, junction)

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


# ---------------------------------------------------------------------------
# The stress leak — the claim every docstring and commit message on this branch
# rests on, and the reason the feature exists at all.
# ---------------------------------------------------------------------------

ETA_WEAK, ETA_STRONG = 1.0, 1.0e4


def _barycentric_lattice(n=12, inset=1e-5):
    """Equally spaced barycentric points STRICTLY INSIDE a triangle.

    The inset is load-bearing, not hygiene. On a cut mesh the interface IS a cell
    edge, so a lattice including that edge samples points where the material is
    genuinely ambiguous: the signed distance is ~1e-16 and its sign is arbitrary.
    A seventh of the samples then take the wrong side and the metric reports a
    leak of 30 for a mesh whose true leak is zero. Pulling the lattice a hair
    inside asks the question that was meant — what does this cell CONTAIN.
    """
    ls = np.array([(i / n, j / n, (n - i - j) / n)
                   for i in range(n + 1) for j in range(n + 1 - i)])
    return (ls + inset) / (1.0 + 3.0 * inset)


def _cell_vertex_indices(dm):
    vS, vE = dm.getDepthStratum(0)
    cS, cE = dm.getHeightStratum(0)
    return np.array([[int(p) - vS for p in dm.getTransitiveClosure(c)[0]
                      if vS <= p < vE] for c in range(cS, cE)])


def _solve_pure_shear(mesh, eta_fn, tag):
    """Pure-shear Stokes with the given viscosity; return the P1 nodal strain rate."""
    v = uw.discretisation.MeshVariable(f"Vk{tag}", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable(f"Pk{tag}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_fn
    stokes.add_dirichlet_bc((0.5, None), "Left")
    stokes.add_dirichlet_bc((-0.5, None), "Right")
    stokes.add_dirichlet_bc((None, -0.5), "Bottom")
    stokes.add_dirichlet_bc((None, 0.5), "Top")
    stokes.solve()

    edot = uw.discretisation.MeshVariable(f"Ek{tag}", mesh, 1, degree=1)
    proj = uw.systems.Projection(mesh, edot)
    proj.uw_function = stokes.Unknowns.Einv2
    proj.solve()
    return np.asarray(edot.array[:, 0, 0]).ravel()


def _leak(mesh, eta_nodal, eta_cellwise, edot_nodal, line):
    """Stress a cell manufactures by misrepresenting the viscosity.

    The cell average of :math:`2\\eta\\dot\\varepsilon` computed from the DISCRETE
    viscosity, minus the same average using the TRUE step viscosity, with the
    strain rate held fixed:

        leak = 2 | <eta_h edot_h> - <eta_true edot_h> |

    Comparing the discrete field against ITSELF — the covariance of eta and edot
    over the cell's own vertices — cannot do this job: for a cell-wise viscosity
    that covariance is zero by construction on ANY mesh, cut or not, so it would
    report success without the mesh having to be right about anything. The true
    field is the only honest reference.
    """
    lattice = _barycentric_lattice()
    idx = _cell_vertex_indices(mesh.dm)
    X = _coords(mesh.dm)
    P = X[idx]                                            # (cells, 3, 2)

    area = 0.5 * np.abs((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                        - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1]))
    xs = np.einsum("sk,ckd->csd", lattice, P)
    edot_s = np.einsum("sk,ck->cs", lattice, edot_nodal[idx])
    eta_true = np.where(
        _signed_distance(xs.reshape(-1, 2), line).reshape(xs.shape[:2]) < 0.0,
        ETA_WEAK, ETA_STRONG)

    def integral(eta_s):
        err = 2.0 * np.abs((eta_s * edot_s).mean(axis=1)
                           - (eta_true * edot_s).mean(axis=1))
        return float((err * area).sum())

    s_nodes = _signed_distance(X, line).ravel()[idx]
    straddle = int(((s_nodes > 1e-11).any(axis=1)
                    & (s_nodes < -1e-11).any(axis=1)).sum())
    return {
        "straddle": straddle,
        "nodal": integral(np.einsum("sk,ck->cs", lattice, eta_nodal[idx])),
        "cellwise": integral(np.repeat(eta_cellwise[:, None], len(lattice), axis=1)),
    }


@pytest.mark.level_2
def test_a_cut_mesh_carries_a_step_viscosity_without_leaking_stress():
    """The headline claim, measured rather than argued.

    A cell straddling a viscosity jump evaluates stress from the interpolated
    viscosity times the interpolated strain rate, which differs from the honest
    cell average by ``-2 Cov(eta, edot)``. Refinement shrinks the straddling band
    but never empties it, so this is not a resolution problem — it is a
    representation problem, and cutting is the cure.

    Three things are asserted, and the middle one is the feature:

    * a cell-wise viscosity on the CUT mesh leaks essentially nothing, because
      every cell lies wholly on one side and can be given the true value;
    * the same viscosity on the UNCUT mesh leaks a great deal;
    * a continuous P1 viscosity leaks even on the cut mesh — the cut alone is NOT
      enough. The nodes ON the interface are shared by both sides and a
      continuous field has to take one value there. This is why the feature is
      "cut AND assign per cell", not "cut".

    Measured on a 1/16 box, viscosity 1 -> 1e4 across a slanted line:

    =======  =========  ================  ===================
    mesh     straddle   leak, P1 nodal    leak, cell-wise
    =======  =========  ================  ===================
    uncut    37         239.7             285.4
    cut      0          298.7             0.0 exactly
    =======  =========  ================  ===================

    Stubbing ``add_conforming_surface`` to return the mesh unchanged fails the
    first two assertions below, which is the check this suite has most needed.
    """
    base = _box(1 / 16)
    cut = base.add_conforming_surface(_surf("Fault", base, SLANTED))

    out = {}
    for name, mesh in (("uncut", base), ("cut", cut)):
        X = np.asarray(mesh.X.coords)
        s = _signed_distance(X, SLANTED).ravel()
        eta_nodal = np.where(s < 0.0, ETA_WEAK, ETA_STRONG)
        # An interface node belongs to both sides; a continuous field must pick
        # one. Which one does not matter — that it must be picked is the point.
        eta_nodal[np.abs(s) < 1e-11] = ETA_STRONG

        idx = _cell_vertex_indices(mesh.dm)
        centroids = _coords(mesh.dm)[idx].mean(axis=1)
        eta_cellwise = np.where(
            _signed_distance(centroids, SLANTED).ravel() < 0.0,
            ETA_WEAK, ETA_STRONG)

        eta_var = uw.discretisation.MeshVariable(f"etak_{name}", mesh, 1, degree=1)
        eta_var.array[:, 0, 0] = eta_nodal
        edot = _solve_pure_shear(mesh, eta_var.sym[0], name)
        out[name] = _leak(mesh, eta_nodal, eta_cellwise, edot, SLANTED)

    # The geometric precondition. Without this the rest is not interpretable.
    assert out["cut"]["straddle"] == 0, "the cut left straddling cells"
    assert out["uncut"]["straddle"] > 0, (
        "the uncut mesh does not straddle the line, so there is nothing to fix "
        "and this test is measuring nothing")

    # THE CLAIM: cut + cell-wise viscosity carries the true material exactly.
    assert out["cut"]["cellwise"] < 1e-9, (
        f"cell-wise viscosity on the cut mesh leaked "
        f"{out['cut']['cellwise']:.3e}; on a conforming mesh every cell lies "
        f"wholly on one side, so this must be zero to round-off.")

    # ... and it is the CUT doing the work, not the cell-wise assignment alone.
    assert out["uncut"]["cellwise"] > 1.0, (
        f"cell-wise viscosity on the UNCUT mesh leaked only "
        f"{out['uncut']['cellwise']:.3e}. If a P0 viscosity were enough on any "
        f"mesh, cutting would buy nothing and this feature would be pointless.")

    # ... and cutting ALONE is not enough, which is why the docs insist on both.
    assert out["cut"]["nodal"] > 1.0, (
        f"a continuous P1 viscosity on the cut mesh leaked only "
        f"{out['cut']['nodal']:.3e}, which contradicts the documented reason "
        f"cell-wise assignment is required.")


# ---------------------------------------------------------------------------
# The fault zone: the cells in the SUPPORT of the labelled facets.
# ---------------------------------------------------------------------------

def _zone_geometry(mesh, name, trace):
    """Zone mask, per-cell distance to the trace, and the LOCAL cell size."""
    from underworld3.utilities.edge_split import cell_diameters

    zone = mesh.cells_supporting(name)
    dm = mesh.dm
    idx = _cell_vertex_indices(dm)
    centroids = _coords(dm)[idx].mean(axis=1)
    distance = np.abs(_signed_distance(centroids, trace).ravel())
    return zone, distance, float(cell_diameters(dm)[zone].mean())


def test_the_fault_zone_is_the_support_of_its_facets():
    """One element each side, and nothing else.

    ``zone == 2 x facets`` exactly, because a cell carrying two labelled edges
    would have been cut in two — so no cell is counted twice and every facet
    contributes both its neighbours. That identity is what makes the definition
    usable: the zone terminates automatically where the chain of facets ends, it
    needs no end cap, and it says nothing about dimension.
    """
    base = _box(1 / 16)
    cut = base.add_conforming_surface(_surf("Fault", base, SLANTED))
    zone, distance, _h = _zone_geometry(cut, "Fault", SLANTED)

    facets = cut.dm.getLabel("Fault").getStratumSize(cut.boundaries["Fault"].value)
    assert facets > 0
    assert int(zone.sum()) == 2 * facets, (
        f"{int(zone.sum())} zone cells for {facets} facets; the zone is the "
        f"support of the facets, so it must be exactly twice as many")

    # A zone cell is one that owns a labelled edge, and nothing in the zone
    # straddles — a cell-wise viscosity on this set is therefore exactly right.
    cS, cE = cut.dm.getHeightStratum(0)
    labelled = set(cut.dm.getLabel("Fault").getStratumIS(
        cut.boundaries["Fault"].value).getIndices())
    for c in np.flatnonzero(zone):
        assert set(cut.dm.getCone(cS + int(c))) & labelled, (
            "a zone cell owns no labelled edge")

    s = _signed_distance(_coords(cut.dm), SLANTED).ravel()[_cell_vertex_indices(cut.dm)]
    assert int(((s > 1e-11).any(axis=1) & (s < -1e-11).any(axis=1)).sum()) == 0

    assert 0 < zone.sum() < (cE - cS), "the zone is the whole mesh"
    # Every zone cell is adjacent to the surface, so none sits far from it.
    assert distance[zone].max() < 2.0 * _h


def test_cells_supporting_is_in_p0_dof_order():
    """The documented assignment pattern has to be valid.

    ``cells_supporting`` returns plex cell order and the docstring assigns it
    straight into a ``degree=0`` MeshVariable. If those orders differed the
    viscosity would land on the wrong cells and every downstream result would be
    quietly wrong while looking plausible, so it is checked rather than assumed.
    """
    base = _box(1 / 16)
    cut = base.add_conforming_surface(_surf("Fault", base, SLANTED))

    eta = uw.discretisation.MeshVariable("eta_zone", cut, 1, degree=0)
    idx = _cell_vertex_indices(cut.dm)
    centroids = _coords(cut.dm)[idx].mean(axis=1)
    assert np.abs(np.asarray(eta.coords) - centroids).max() < 1e-12, (
        "degree-0 DOF order is not plex cell order; cells_supporting cannot be "
        "assigned straight across")

    zone = cut.cells_supporting("Fault")
    eta.array[:, 0, 0] = np.where(zone, 1.0e-3, 1.0)
    assert np.isclose(eta.array[:, 0, 0].min(), 1.0e-3)
    assert int((eta.array[:, 0, 0] < 1.0).sum()) == int(zone.sum())


# Mean distance from a zone cell's centroid to the trace, over the LOCAL cell
# size in the zone. The MAX will not do: it is one outlier cell and it was
# identical (0.01474) at two different adapt resolutions, reporting no scaling
# at all where the mean shows it cleanly.
ZONE_THICKNESS_OVER_H = 0.19


@pytest.mark.parametrize("cell_size", [1 / 8, 1 / 16, 1 / 32])
def test_the_fault_zone_is_one_element_wide_at_every_resolution(cell_size):
    """Zone thickness tracks `h`, which is what makes width a refinement knob.

    Measured 0.189 / 0.183 / 0.184 across a 4x refinement — constant, and the
    price of the facet-support definition: thickness is no longer a physical
    parameter. Under adapt-on-top that is the point rather than a defect, since
    the local `h` is whatever the metric asks for (see the test below).
    """
    base = _box(cell_size)
    cut = base.add_conforming_surface(_surf("Fault", base, SLANTED))
    zone, distance, h = _zone_geometry(cut, "Fault", SLANTED)

    facets = cut.dm.getLabel("Fault").getStratumSize(cut.boundaries["Fault"].value)
    assert int(zone.sum()) == 2 * facets

    ratio = distance[zone].mean() / h
    assert ratio == pytest.approx(ZONE_THICKNESS_OVER_H, abs=0.03), (
        f"cellSize={cell_size}: zone half-thickness is {ratio:.3f} h, not the "
        f"~{ZONE_THICKNESS_OVER_H} h it is at every other resolution")


@pytest.mark.level_2
def test_the_fault_zone_narrows_with_the_adapt_metric():
    """Fault width is a REFINEMENT parameter: the design this is all for.

    The surface lives at the finest adapted level, so the zone is one element
    wide *there* and the metric sets how wide that is — controlled locally and at
    bounded cost. Measured: the same fault at h_near 0.03 / 0.02 / 0.015 gives
    zone thicknesses in the same ratio, at 0.195 / 0.202 / 0.198 of the local `h`
    throughout, and reaches the resolution of a uniform 1/32 mesh with about half
    the cells.

    Also checks the composition itself works — ``adapt`` then
    ``add_conforming_surface`` — and that the child keeps its multigrid tail.
    """
    thickness = {}
    for h_near in (0.03, 0.015):
        base = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1,
            regular=False, refinement=1, qdegree=2)
        fault = _surf("Fault", base, SLANTED)
        fault.discretize()
        child = base.adapt(
            fault.refinement_metric_function(h_near=h_near, h_far=0.09,
                                             width=0.06),
            max_levels=3)
        cut = child.add_conforming_surface(_surf("Fault", child, SLANTED))

        zone, distance, h = _zone_geometry(cut, "Fault", SLANTED)
        facets = cut.dm.getLabel("Fault").getStratumSize(
            cut.boundaries["Fault"].value)
        assert int(zone.sum()) == 2 * facets

        # The cut child must keep the adapted hierarchy under it, or the solver
        # loses multigrid exactly where the fault is.
        assert len(cut._custom_mg_coarse_meshes) >= len(base.dm_hierarchy)

        ratio = distance[zone].mean() / h
        assert ratio == pytest.approx(ZONE_THICKNESS_OVER_H, abs=0.03), (
            f"h_near={h_near}: zone is {ratio:.3f} h wide, not ~"
            f"{ZONE_THICKNESS_OVER_H} h — one element each side is the claim")
        thickness[h_near] = distance[zone].mean()

    assert thickness[0.015] < 0.75 * thickness[0.03], (
        f"halving the requested h_near barely narrowed the zone "
        f"({thickness[0.03]:.5f} -> {thickness[0.015]:.5f}); fault width is "
        f"supposed to follow the metric")


# ---------------------------------------------------------------------------
# The two snapping guards. Both exist because of what SPLITTING costs: a split
# beside a vertex is what makes a sliver, so anything that replaces a split with
# a vertex move helps, and anything that turns a move back into a split hurts.
# ---------------------------------------------------------------------------

def test_the_quality_guard_turns_a_refusal_into_a_cut():
    """A tolerance large enough to flatten a cell must be survivable.

    Snapping pulls every corner of a cell thinner than the tolerance band onto
    the line, and the cell collapses: measured, all of them at snap_frac 0.4 had
    all three corners snapped, from opposite sides. Without a guard the whole cut
    is refused. With one, the offending moves are vetoed and those crossings are
    split instead, which is the path that already works.

    Note the guard must be on QUALITY, not on inversion. A flattened cell lands
    at ~1e-16 of either sign, so an inversion test passes about half of them —
    and then returns a mesh whose chain has silently broken.
    """
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 8,
        regular=False, qdegree=2)
    line = np.array([[-0.1, 0.503], [1.1, 0.541]])

    with pytest.raises(RuntimeError, match="inverted"):
        cut_along_lines(mesh.dm, [line], snap_frac=0.48, snap_quality=None)

    cut, info = cut_along_lines(mesh.dm, [line], snap_frac=0.48)
    assert info["n_cut_edges"] == info["n_split"] + info["n_on_surface"] - 1
    assert (cell_areas(cut) > 0.0).all()
    assert info["min_angle"] > 1.0, (
        f"guarded cut still has a {info['min_angle']:.3f} degree cell")


def test_snap_dist_reaches_vertices_snap_frac_cannot():
    """A vertex near the line whose edges are all crossed mid-way.

    ``snap_frac`` is measured ALONG an edge, so it cannot see a vertex that sits
    a fraction of an element from the line while every edge meeting it is crossed
    near its midpoint. That vertex becomes the apex of a cell with one edge on
    the cut, which is the cut's characteristic sliver — and no value of
    ``snap_frac`` removes it. ``snap_dist`` proposes such a vertex directly.

    The test is that at a FIXED along-edge tolerance it still finds vertices to
    move, and trades splits for them.
    """
    mesh = _box(1 / 24)
    base = cut_along_lines(mesh.dm, [SLANTED], snap_frac=0.30)[1]
    reached = cut_along_lines(mesh.dm, [SLANTED], snap_frac=0.30,
                              snap_dist=0.30)[1]
    assert reached["n_on_surface"] > base["n_on_surface"], (
        "snap_dist proposed no vertex the along-edge test had not already found")
    assert reached["n_split"] < base["n_split"], (
        f"splits did not fall: {base['n_split']} -> {reached['n_split']}")
    assert (reached["n_cut_edges"]
            == reached["n_split"] + reached["n_on_surface"] - 1)


def test_repair_fixes_the_cells_the_cut_leaves_thin():
    """``repair=True`` runs the two operations a cut does not have.

    A cut can only snap a vertex onto the surface or split an edge it crosses,
    so a crossing landing near a vertex either drags the vertex to it or carves
    a thin cell beside it. Flipping fixes the ones whose point set is fine and
    whose connectivity is not; deleting fixes the ones whose point set is the
    problem.

    The surface itself must come through untouched — that is the whole point of
    both passes refusing to act on a labelled edge — so the facet count is
    asserted, not just the quality.
    """
    # A GRADED mesh. On a uniform one the flip pass alone already clears the
    # thin cells and deletion is offered nothing worth taking (measured: 26 -> 25
    # cells under 15 degrees, 0 removals), so a uniform fixture would assert the
    # feature while exercising half of it.
    line = np.array([[-0.1, 0.37], [1.1, 0.63]])
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 6,
        regular=False, qdegree=2, refinement=2)
    surf = _surf("Grade", base, line)
    surf.discretize()
    mesh = base.adapt(surf.refinement_metric_function(
        h_near=1 / 48, h_far=1 / 6, width=1 / 12), max_levels=3)

    plain = mesh.add_conforming_surface(_surf("Rp", mesh, line), snap_frac=0.30)
    fixed = mesh.add_conforming_surface(_surf("Rr", mesh, line), snap_frac=0.30,
                                        repair=True)

    a0, a1 = min_angles(plain.dm), min_angles(fixed.dm)
    assert int((a1 < 15).sum()) < int((a0 < 15).sum()), (
        "repair did not reduce the count of thin cells")
    assert a1.min() >= a0.min() - 1e-12, "repair lowered the worst angle"

    info = fixed._surface_info
    assert info["n_repair_flips"] > 0 and info["n_repair_removals"] > 0, (
        "one of the two passes did nothing, so this is not testing both")
    assert info["min_angle"] == pytest.approx(float(a1.min())), (
        "_surface_info still reports the angle from before the repair")

    # The surface is a chain of the same facets, and deleting cells conserves area.
    n_plain = plain.dm.getLabel("Rp").getStratumSize(
        int(plain.boundaries["Rp"].value))
    n_fixed = fixed.dm.getLabel("Rr").getStratumSize(
        int(fixed.boundaries["Rr"].value))
    assert n_fixed == n_plain, "repair changed the surface itself"
    assert cell_areas(fixed.dm).sum() == pytest.approx(
        cell_areas(plain.dm).sum(), rel=1e-13)
    assert (cell_areas(fixed.dm) > 0).all()


def test_the_pulled_vertex_covers_every_cell_holding_the_target():
    """A target ON AN EDGE must take one of that edge's ends, not an apex.

    `pull_vertex_onto` used to move the nearest vertex in the mesh at large.
    That vertex can be the apex of one of the two cells sharing the edge the
    target lies on: it wins on distance, it belongs to one of the two cells,
    and the OTHER cell is left with no corner on the target. A line ending
    there is then "entered but not left" and `cut_along_lines` refuses a tip
    that does have a vertex on it — issue #542, measured inside a ribbon zone
    where the edge ends sit at exactly the zone half-width and an apex won by
    0.4%.

    The invariant asserted is the one the cut actually needs: EVERY cell whose
    closure contains the target ends up with a corner exactly on it. Edge
    midpoints are used because that is the configuration that fails.

    The mesh is FLATTENED in y first, and that is not cosmetic: on a
    well-shaped mesh an edge midpoint's nearest vertex is one of that edge's
    own ends, so the defect never appears and the test would pass without
    testing anything. Flat cells put an apex within reach of the opposite
    edge, which is the shape a ribbon zone has along its length. The negative
    control below asserts the mesh really does present the case.
    """
    from underworld3.utilities.line_cut import (_cell_vertices, _coords,
                                                _minimal_face_vertices,
                                                _set_coordinates)

    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 12,
        regular=False, qdegree=2)
    dm = base.dm.clone()
    squashed = _coords(dm).copy()
    squashed[:, 1] *= 0.12
    _set_coordinates(dm, np.arange(len(squashed)), squashed)

    X = _coords(dm)
    cells = _cell_vertices(dm)

    interior = [e for e in range(*dm.getDepthStratum(1))
                if len(dm.getSupport(e)) == 2]
    vS = dm.getDepthStratum(0)[0]
    midpoints = []
    for e in interior[::7]:
        a, b = (int(v) - vS for v in dm.getCone(e))
        midpoints.append(0.5 * (X[a, :2] + X[b, :2]))
    assert len(midpoints) > 5

    exposed = 0
    for t in midpoints:
        face = _minimal_face_vertices(X, cells, t)
        assert len(face) == 2, "an edge midpoint lies on an edge, two ends"
        nearest = int(np.linalg.norm(X[:, :2] - t, axis=1).argmin())
        if nearest not in face:
            exposed += 1

        out = pull_vertex_onto(dm, t)
        Xo = _coords(out)
        holders = [c for c in range(len(cells))
                   if len(_minimal_face_vertices(X, cells[c: c + 1], t))]
        assert holders, "the midpoint is inside the mesh"
        for c in holders:
            on = np.linalg.norm(Xo[cells[c], :2] - t, axis=1).min()
            assert on < 1e-12, (
                f"cell {c} holds the target and has no corner on it "
                f"(closest {on:.3e}); a line ending here would be refused")

    assert exposed > 0, (
        "no midpoint had its nearest vertex off the edge, so this mesh does "
        "not exercise the defect and the test proves nothing")


def _domain_boundary_vertices(dm):
    """Mask of the vertices on the domain boundary (serial: one-sided edges)."""
    vS, vE = dm.getDepthStratum(0)
    on = np.zeros(vE - vS, dtype=bool)
    for e in range(*dm.getDepthStratum(1)):
        if len(dm.getSupport(e)) == 1:
            for v in dm.getCone(e):
                on[int(v) - vS] = True
    return on


def _meshes_with_a_wall():
    box = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 12,
        regular=False, qdegree=2)
    annulus = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5,
                                 cellSize=0.1, qdegree=2)
    return {"box": box, "annulus": annulus}


@pytest.mark.parametrize("kind", ["box", "annulus"])
def test_a_target_inside_the_mesh_leaves_the_domain_boundary_alone(kind):
    """A target INSIDE a boundary cell takes the cell's interior corner.

    `pull_vertex_onto` took the nearest corner of the cell holding the target,
    whichever it was. For a target under a wall vertex that corner is the wall
    vertex, which was then moved off the wall: the domain lost the area
    between the wall and the target, and a fault tip placed this way became a
    boundary vertex, so the split refused a fault that does not touch the
    boundary.

    The target is 0.6 W + 0.2 A + 0.2 B in a cell with wall corners W, A and
    interior corner B: inside the cell, 0.2 of its height from the wall. The
    negative control counts the cells in which W is the nearest corner, the
    configuration that fails.

    Asserted: the interior corner B is on the target, every boundary vertex is
    bitwise where it was, and the mesh area is unchanged (for the box, 1).
    """
    from underworld3.utilities.line_cut import (_cell_vertices, _coords,
                                                _set_coordinates)

    dm = _meshes_with_a_wall()[kind].dm
    X = _coords(dm)
    cells = _cell_vertices(dm)
    wall = _domain_boundary_vertices(dm)
    area = cell_areas(dm).sum()
    if kind == "box":
        assert area == pytest.approx(1.0, rel=1e-13)

    exposed = 0
    for corners in cells[wall[cells].sum(axis=1) == 2]:
        W, A = corners[wall[corners]]
        B = corners[~wall[corners]][0]
        t = 0.6 * X[W] + 0.2 * X[A] + 0.2 * X[B]
        if np.linalg.norm(X[corners] - t, axis=1).argmin() == list(corners).index(W):
            exposed += 1

        Xo = _coords(pull_vertex_onto(dm, t))
        assert np.array_equal(Xo[wall], X[wall]), (
            f"a domain-boundary vertex moved for the interior target {t}")
        assert np.linalg.norm(Xo[B] - t) < 1e-12, (
            f"the interior corner of the cell holding {t} is not on it")
        moved = np.flatnonzero((Xo != X).any(axis=1))
        assert list(moved) == [B]
        out = dm.clone()
        _set_coordinates(out, np.arange(len(Xo)), Xo)
        assert cell_areas(out).sum() == pytest.approx(area, rel=1e-13)

    assert exposed > 10, (
        "the wall corner was the nearest corner in too few cells, so this "
        "mesh does not exercise the defect")


@pytest.mark.parametrize("kind", ["box", "annulus"])
def test_a_target_too_close_under_the_boundary_is_refused(kind):
    """An interior corner is not dragged up against the boundary unannounced.

    A target inside a boundary cell takes that cell's interior corner B. With
    the target at a fraction f of the cell's height above the boundary edge
    the cell is left with that much of its height, and at f = 0.02 its
    quality is 0.03 to 0.045 on these meshes: a sliver along the wall, where
    the fault tip is.

    The target is 0.98 (0.6 W + 0.4 A) + 0.02 B in each cell with boundary
    corners W, A and interior corner B. Asserted: the pull is refused, naming
    ``min_quality`` and the one interior corner; with the floor off B goes
    onto the target, the boundary is bitwise where it was, and the worst cell
    of the mesh is the one the refusal reported, to the three figures it
    prints. At f = 0.2 the default accepts the pull (the test above).
    """
    from underworld3.utilities.line_cut import (_cell_quality, _cell_vertices,
                                                _coords)

    dm = _meshes_with_a_wall()[kind].dm
    X = _coords(dm)
    cells = _cell_vertices(dm)
    wall = _domain_boundary_vertices(dm)
    assert _cell_quality(X, cells).min() > 0.5

    wall_cells = cells[wall[cells].sum(axis=1) == 2]
    assert len(wall_cells) > 40
    for corners in wall_cells:
        W, A = corners[wall[corners]]
        B = corners[~wall[corners]][0]
        t = 0.98 * (0.6 * X[W] + 0.4 * X[A]) + 0.02 * X[B]

        with pytest.raises(ValueError, match="min_quality=0.1") as refusal:
            pull_vertex_onto(dm, t)
        assert "None of the 1 interior corner(s)" in str(refusal.value)
        reported = float(re.search(r"\(worst ([^;]+);", str(refusal.value)).group(1))

        Xo = _coords(pull_vertex_onto(dm, t, min_quality=None))
        assert np.linalg.norm(Xo[B] - t) < 1e-12
        assert np.array_equal(Xo[wall], X[wall])
        worst = _cell_quality(Xo, cells).min()
        assert 0.0 < worst < 0.1
        assert reported == pytest.approx(worst, rel=6e-3)


def test_a_target_outside_the_mesh_takes_the_nearest_vertex_anywhere():
    """The boundary rule is for targets INSIDE the mesh.

    (0.52, 1.01) is above the top wall of the box. No cell holds it, so the
    nearest vertex anywhere takes it, and that is the wall vertex (0.5, 1):
    the box gains the triangle over the two wall edges at that vertex, 0.01
    high on a base of 2/12.
    """
    from underworld3.utilities.line_cut import _coords

    dm = _meshes_with_a_wall()["box"].dm
    X = _coords(dm)
    t = np.array([0.52, 1.01])
    out = pull_vertex_onto(dm, t)
    moved = np.flatnonzero((_coords(out) != X).any(axis=1))
    assert len(moved) == 1
    assert np.allclose(X[moved[0]], (0.5, 1.0), rtol=0.0, atol=1e-9)
    assert np.array_equal(_coords(out)[moved[0]], t)
    assert cell_areas(out).sum() == pytest.approx(1.0 + 0.01 / 12, rel=1e-9)


def test_a_target_on_the_domain_boundary_slides_a_boundary_vertex():
    """A target ON a wall edge takes one of that edge's ends, along the wall.

    The boundary rule must not stop a point of the wall being placed: here the
    target is 0.3 of the way along each wall edge of the box, the nearer end
    moves onto it, nothing else moves, and the box keeps its area of 1.

    The end that moves is never a corner of the box. A corner is on two walls
    and sliding it along one takes it off the other; that is not what this
    test is about, and `pull_vertex_onto` carries a TODO for it.
    """
    from underworld3.utilities.line_cut import (_coords, _edge_vertices,
                                                _set_coordinates)

    dm = _meshes_with_a_wall()["box"].dm
    X = _coords(dm)
    ends = _edge_vertices(dm)
    eS = dm.getDepthStratum(1)[0]
    wall_edges = [e - eS for e in range(*dm.getDepthStratum(1))
                  if len(dm.getSupport(e)) == 1]
    assert len(wall_edges) == 48

    def is_box_corner(v):
        return bool(np.isin(X[v], (0.0, 1.0)).all())

    for a, b in ends[wall_edges]:
        if is_box_corner(a):
            a, b = b, a
        t = 0.7 * X[a] + 0.3 * X[b]
        Xo = _coords(pull_vertex_onto(dm, t))
        assert np.linalg.norm(Xo[a] - t) < 1e-12
        assert list(np.flatnonzero((Xo != X).any(axis=1))) == [a]
        out = dm.clone()
        _set_coordinates(out, np.arange(len(Xo)), Xo)
        assert cell_areas(out).sum() == pytest.approx(1.0, rel=1e-13)


def _strip_with_every_vertex_on_the_boundary():
    """The rectangle [0, 2] x [0, 1] as four triangles, no interior vertex.

    Vertices (1, 0) and (1, 1) sit mid-wall, so they can slide along it; the
    edge from (1, 0) to (0, 1) is interior with both ends on the boundary.
    """
    from petsc4py import PETSc

    cells = np.array([[0, 1, 5], [1, 4, 5], [1, 2, 4], [2, 3, 4]], dtype=np.int32)
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                         [2.0, 1.0], [1.0, 1.0], [0.0, 1.0]])
    return PETSc.DMPlex().createFromCellList(2, cells, vertices,
                                             comm=PETSc.COMM_SELF)


@pytest.mark.parametrize("target, where", [
    ((0.4, 0.3), "inside a cell"),
    ((0.5, 0.5), "on an interior edge"),
])
def test_a_target_with_no_interior_corner_to_take_is_refused(target, where):
    """No corner of the face holding the target is interior: refuse, and say so.

    Every vertex of the strip is on the boundary. A target inside a cell, or
    on an interior edge whose ends are both on the boundary, could only be
    reached by moving a boundary vertex off the boundary, which is the defect.
    The refusal names the boundary rather than leaving a later step to fail
    on the dented wall.
    """
    dm = _strip_with_every_vertex_on_the_boundary()
    with pytest.raises(ValueError, match="domain boundary"):
        pull_vertex_onto(dm, np.array(target))


def test_a_boundary_target_is_placed_where_every_vertex_is_on_the_boundary():
    """The refusal is for interior targets only: a wall point is still placed.

    (0.7, 0) is on the bottom wall of the strip; the mid-wall vertex (1, 0)
    slides onto it and the strip keeps its area of 2.
    """
    from underworld3.utilities.line_cut import _coords

    dm = _strip_with_every_vertex_on_the_boundary()
    out = pull_vertex_onto(dm, np.array([0.7, 0.0]))
    moved = np.flatnonzero((_coords(out) != _coords(dm)).any(axis=1))
    assert [tuple(x) for x in _coords(dm)[moved]] == [(1.0, 0.0)]
    assert [tuple(x) for x in _coords(out)[moved]] == [(0.7, 0.0)]
    assert cell_areas(out).sum() == pytest.approx(2.0, rel=1e-13)


def _grid_of_eighteen_triangles(clockwise=False, wall_vertex_at=2.0):
    """The square [0, 3] x [0, 3] as eighteen triangles: each unit square is
    cut along its rising diagonal.

    The cell (0, 1), (1, 1), (1, 2) has one boundary corner and two interior
    ones. ``wall_vertex_at`` slides the boundary vertex (0, 2) along its wall:
    at 1.02 the cell (0, 1), (1, 2), (0, 1.02) is a needle of quality 0.0175.
    ``clockwise`` lists every cell the other way round.
    """
    from petsc4py import PETSc

    def row(i, j):
        return 4 * j + i

    vertices = np.array([[i, j] for j in range(4) for i in range(4)], dtype=float)
    vertices[row(0, 2)] = (0.0, wall_vertex_at)
    cells = []
    for j in range(3):
        for i in range(3):
            cells.append([row(i, j), row(i + 1, j), row(i + 1, j + 1)])
            cells.append([row(i, j), row(i + 1, j + 1), row(i, j + 1)])
    cells = np.array(cells, dtype=np.int32)
    if clockwise:
        cells = cells[:, ::-1].copy()
    return PETSc.DMPlex().createFromCellList(2, cells, vertices,
                                             comm=PETSc.COMM_SELF)


def _moves(dm, out):
    """What a pull moved: ``{where it was: where it is}``, and the quality of
    the worst cell it left (1 equilateral), whichever way the cells are wound."""
    from underworld3.utilities.line_cut import (_cell_quality, _cell_vertices,
                                                _coords)

    X, Y = _coords(dm), _coords(out)
    moved = np.flatnonzero((Y != X).any(axis=1))
    worst = np.abs(_cell_quality(Y, _cell_vertices(dm))).min()
    return {tuple(X[m]): tuple(Y[m]) for m in moved}, worst


@pytest.mark.parametrize("clockwise", [False, True])
def test_the_next_interior_corner_takes_a_target_the_nearest_would_flatten(clockwise):
    """A cell with two interior corners offers both before the pull is refused.

    (0.3, 1.28) is in the cell (0, 1), (1, 1), (1, 2), nearest to its boundary
    corner (0, 1) and 0.014 from the edge that runs from there to (1, 2). The
    nearer interior corner is (1, 1): put on the target it leaves that cell
    with a quality of 0.0218. The other, (1, 2), leaves every cell at 0.558 or
    better. A pull that stopped at the first corner would refuse a target
    0.3 of a cell from the boundary.

    Asserted: (1, 2) is on the target and nothing else moved; with the floor
    off the nearer corner is taken and the cell is at 0.0218. The same on a
    mesh whose cells are listed clockwise, where a quality reads negative
    before the pull and after it.
    """
    dm = _grid_of_eighteen_triangles(clockwise)
    target = (0.3, 1.28)

    moved, worst = _moves(dm, pull_vertex_onto(dm, np.array(target)))
    assert moved == {(1.0, 2.0): target}
    assert worst == pytest.approx(0.55847, abs=1e-5)

    moved, worst = _moves(dm, pull_vertex_onto(dm, np.array(target),
                                               min_quality=None))
    assert moved == {(1.0, 1.0): target}
    assert worst == pytest.approx(0.02181, abs=1e-5)


@pytest.mark.parametrize("clockwise", [False, True])
def test_a_target_no_interior_corner_can_take_is_refused_with_their_count(clockwise):
    """Refused only when every interior corner of the cell has been tried.

    (0.03, 1.02) is beside the boundary corner (0, 1) of the cell (0, 1),
    (1, 1), (1, 2). Either interior corner put there leaves a cell along the
    wall: 0.00888 with (1, 1), 0.0357 with (1, 2). The refusal names both
    corners tried and reports the nearest.
    """
    dm = _grid_of_eighteen_triangles(clockwise)
    with pytest.raises(ValueError, match="min_quality=0.1") as refusal:
        pull_vertex_onto(dm, np.array([0.03, 1.02]))
    message = str(refusal.value)
    assert "None of the 2 interior corner(s)" in message
    assert "the nearest, at (1.0, 1.0), would leave 2 cell(s)" in message
    assert "(worst 0.00888;" in message


def test_a_cell_already_below_the_floor_does_not_refuse_a_pull_that_improves_it():
    """The floor is the lower of ``min_quality`` and the cell's present quality.

    With the boundary vertex (0, 2) slid down to (0, 1.02) the cell (0, 1),
    (1, 2), (0, 1.02) is a needle of quality 0.0175 before anything is asked.
    The target (0.3, 1.28) takes (1, 2), as on the plain grid, and that brings
    the needle to 0.0637: below 0.1, and better than it was. A mesh that has
    been cut once carries such cells, and a floor of 0.1 on them would refuse
    every pull beside the first fault.
    """
    from underworld3.utilities.line_cut import _cell_quality, _cell_vertices, _coords

    dm = _grid_of_eighteen_triangles(wall_vertex_at=1.02)
    assert _cell_quality(_coords(dm), _cell_vertices(dm)).min() == pytest.approx(
        0.01749, abs=1e-5)

    target = (0.3, 1.28)
    moved, worst = _moves(dm, pull_vertex_onto(dm, np.array(target)))
    assert moved == {(1.0, 2.0): target}
    assert worst == pytest.approx(0.06368, abs=1e-5)


def test_a_later_target_is_tested_on_the_mesh_the_earlier_ones_left():
    """The quality test reads the mesh as the call has moved it so far.

    Alone, (0.3, 1.28) cannot take (1, 1): the cell (0, 1), (1, 1), (1, 2)
    would be left at 0.0218, so (1, 2) takes it (the test above). Here
    (0.8, 2.4) comes first and takes (1, 2), its nearest corner, which opens
    that cell out; (1, 1) on the second target then leaves it at 0.159, and
    takes it. Tested against the mesh as given, the second target would refuse
    (1, 1) and take (1, 2) from the first, which is left with no vertex.

    Asserted: each target carries its own vertex.
    """
    dm = _grid_of_eighteen_triangles()
    first, second = (0.8, 2.4), (0.3, 1.28)
    moved, worst = _moves(dm, pull_vertex_onto(dm, np.array([first, second])))
    assert moved == {(1.0, 2.0): first, (1.0, 1.0): second}
    assert worst == pytest.approx(0.15891, abs=1e-5)


@pytest.mark.parametrize("vertex", [(0.0, 1.0), (3.0, 3.0), (1.0, 1.0)])
def test_a_target_on_a_vertex_moves_nothing(vertex):
    """A target that is already a vertex is placed, wherever the vertex is.

    A vertex is the smallest face holding such a target. On a wall, or at a
    corner of the box, that face is on the domain boundary, so its one vertex
    is the candidate and the boundary rule has nothing to refuse.
    """
    dm = _grid_of_eighteen_triangles()
    moved, worst = _moves(dm, pull_vertex_onto(dm, np.array(vertex)))
    assert moved == {}
    assert worst == pytest.approx(0.86603, abs=1e-5)


@pytest.mark.parametrize("target, worst", [
    ((0.2, 2.18), 0.02047), ((0.4, 2.38), 0.02272), ((0.8, 2.78), 0.02076)])
def test_a_target_beside_the_interior_edge_across_a_domain_corner_is_refused(
        target, worst):
    """A cell can have two boundary corners and no boundary edge.

    The cell (0, 2), (1, 2), (1, 3) sits across the top left corner of the
    grid: (0, 2) is on the left wall, (1, 3) on the top wall, and the edge
    joining them is interior. Each target is 0.014 from that edge, 0.2 to 0.8
    of a cell from the nearer wall, and nearest to one of the two boundary
    corners. The one interior corner, (1, 2), put on it flattens the cell
    against that edge.

    Asserted: the pull is refused, naming the one corner and the quality it
    would leave; with the floor off (1, 2) is on the target and the worst
    cell is at that quality. (0.3, 2.1), 0.14 from the edge, is taken by
    (1, 2) and leaves every cell at 0.204 or better.
    """
    dm = _grid_of_eighteen_triangles()
    with pytest.raises(ValueError, match="min_quality=0.1") as refusal:
        pull_vertex_onto(dm, np.array(target))
    message = str(refusal.value)
    assert "None of the 1 interior corner(s)" in message
    assert "the nearest, at (1.0, 2.0), would leave 1 cell(s)" in message
    assert float(re.search(r"\(worst ([^;]+);", message).group(1)) == pytest.approx(
        worst, rel=6e-3)

    moved, left = _moves(dm, pull_vertex_onto(dm, np.array(target),
                                              min_quality=None))
    assert moved == {(1.0, 2.0): target}
    assert left == pytest.approx(worst, abs=1e-5)

    clear = (0.3, 2.1)
    moved, left = _moves(dm, pull_vertex_onto(dm, np.array(clear)))
    assert moved == {(1.0, 2.0): clear}
    assert left == pytest.approx(0.20377, abs=1e-5)


@pytest.mark.parametrize("first, second", [
    ((0.5, 0.2), (0.2, 0.5)), ((0.2, 0.5), (0.5, 0.2))])
def test_two_targets_that_choose_one_vertex_leave_the_earlier_without_one(
        first, second):
    """What a collision does today: the later target takes the vertex.

    (0.5, 0.2) is in the cell on the bottom wall at the corner (0, 0) and
    (0.2, 0.5) in the cell on the left wall there. Each cell has one interior
    corner, and it is the same one, (1, 1). Each target is chosen from the
    mesh as given, so both choose it, and the second moves it off the first.

    Asserted: one vertex moved, it is on the second target, and no vertex is
    on the first. Nothing is raised. This is the open defect of issue #559,
    held here so that a change to it is seen; the test changes with the fix.
    """
    from underworld3.utilities.line_cut import _coords

    dm = _grid_of_eighteen_triangles()
    out = pull_vertex_onto(dm, np.array([first, second]))
    moved, worst = _moves(dm, out)
    assert moved == {(1.0, 1.0): second}
    assert np.linalg.norm(_coords(out) - np.array(first), axis=1).min() == pytest.approx(
        0.3 * np.sqrt(2.0), rel=1e-12)
    assert worst == pytest.approx(0.43849, abs=1e-5)


def test_a_target_on_a_chord_of_a_curved_wall_takes_an_end_of_the_chord():
    """A point of a boundary edge is placed on a curved wall as on a flat one.

    The inner wall of the annulus is 32 equal chords of a circle of radius
    0.5. The target is 0.3 of the way along each chord. Its nearer end takes
    it and nothing else moves, as on the walls of the box.

    That end leaves the circle, since the next chord turns away from this
    one: every vertex of a curved wall is a corner in the sense of the TODO
    in `pull_vertex_onto`. Asserted with the placement: the moved vertex is
    at the radius of the chord point, and the mesh gains the triangle between
    the old neighbouring chord and the new one, 0.15 L^2 sin(2 pi / 32) for a
    chord of length L.
    """
    from underworld3.utilities.line_cut import (_coords, _edge_vertices,
                                                _set_coordinates)

    dm = _meshes_with_a_wall()["annulus"].dm
    X = _coords(dm)
    ends = _edge_vertices(dm)
    eS = dm.getDepthStratum(1)[0]
    wall_edges = [e - eS for e in range(*dm.getDepthStratum(1))
                  if len(dm.getSupport(e)) == 1]
    inner = [(a, b) for a, b in ends[wall_edges] if np.linalg.norm(X[a]) < 0.75]
    assert len(inner) == 32
    area = cell_areas(dm).sum()

    for a, b in inner:
        chord = np.linalg.norm(X[b] - X[a])
        assert chord == pytest.approx(np.sin(np.pi / 32), rel=1e-6)
        t = 0.7 * X[a] + 0.3 * X[b]
        Xo = _coords(pull_vertex_onto(dm, t))
        assert np.linalg.norm(Xo[a] - t) < 1e-12
        assert list(np.flatnonzero((Xo != X).any(axis=1))) == [a]
        assert np.linalg.norm(Xo[a]) == pytest.approx(
            np.hypot(0.5 * np.cos(np.pi / 32), 0.2 * chord), rel=1e-6)
        out = dm.clone()
        _set_coordinates(out, np.arange(len(Xo)), Xo)
        assert cell_areas(out).sum() - area == pytest.approx(
            0.15 * chord**2 * np.sin(2 * np.pi / 32), rel=1e-5)
