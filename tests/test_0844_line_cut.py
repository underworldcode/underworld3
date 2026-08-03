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
