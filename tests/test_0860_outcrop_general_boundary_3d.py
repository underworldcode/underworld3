"""The general 3-D outcrop: a zone meeting a boundary that is not a box wall.

Three configurations, each the failure mode of a different assumption the
box-framed cap made. A ROTATED box: no wall is axis-aligned, so the wall-code
frame cannot exist, but every wall is still one coplanar region — the
single-region collar. A band across a box EDGE: the collar spans two regions
and the crease overlay must conform the two sides' differing segmentations
(mesh vertices against assembly nodes). A SPHERICAL SHELL outer surface:
every facet is its own region, every vertex carries the faceting and is
protected — the case that kills any wall-partition design, so the case that
proves the trace design (the plan's ruling on #553).

Every test runs its NEGATIVE CONTROL first: the identical census on an
interior twin counts no trace, so the trace counted on the outcrop is
produced by the outcrop and not by a probe that fires on anything. The P2
Poisson oracle is the whole-chain check — topological gates alone have
missed real defects in this subsystem (#518, #520).
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.place_surface import place_thin_volume

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.skipif(uw.mpi.size > 1,
                                 reason="serial suite; the parallel form is "
                                        "ptest_0855")]


def _volume(dm):
    from underworld3.utilities.place_surface import _owned_cell_volume
    return _owned_cell_volume(dm)


def _rotation():
    a, b = np.deg2rad(20.0), np.deg2rad(15.0)
    Rz = np.array([[np.cos(a), -np.sin(a), 0.0],
                   [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(b), -np.sin(b)],
                   [0.0, np.sin(b), np.cos(b)]])
    return Rz @ Rx


def _rotate_dm(dm, R):
    vec = dm.getCoordinatesLocal()
    vec.array[:] = (vec.array.reshape(-1, 3) @ R.T).reshape(-1)
    dm.setCoordinatesLocal(vec)


def _boundary_census(new, zv, pred=None):
    """(band, trace, bare) over true-boundary faces ``pred`` selects.

    ``bare`` counts faces carrying NO label at all — a hole the relabel
    left in the wall. Band and collar nodes lie on the boundary FACETS,
    not on any smooth surface, so ``pred`` must be a plane test or None
    (all boundary faces).
    """
    fS, fE = new.getHeightStratum(1)
    vS, vE = new.getDepthStratum(0)
    Xn = np.asarray(new.getCoordinatesLocal().array
                    ).reshape(-1, 3)[: vE - vS]
    names = [new.getLabelName(i) for i in range(new.getNumLabels())]
    from underworld3.utilities import reconnect
    wall_names = [n for n in names
                  if n not in reconnect._TOPOLOGY_LABELS
                  and not n.startswith("Zone")]
    skin = new.getLabel("Zone_skin")
    trace = new.getLabel("Zone_trace")
    n_band = n_trace = n_bare = 0
    for f in range(fS, fE):
        if len(new.getSupport(f)) != 1:
            continue
        verts = [int(q) - vS for q in new.getTransitiveClosure(f)[0]
                 if vS <= int(q) < vE]
        if pred is not None and not all(pred(Xn[v]) for v in verts):
            continue
        if skin.getValue(f) == zv:
            n_band += 1
        if trace.getValue(f) == zv:
            n_trace += 1
        if all(new.getLabel(n).getValue(f) < 0 for n in wall_names):
            n_bare += 1
    return n_band, n_trace, n_bare


def _p2_oracle(new, base, walls):
    """P2 Poisson through the zone: exact for u = x^2 + y^2 + z^2."""
    import sympy

    bounds = base._boundaries_with("Zone")
    mesh = uw.discretisation.Mesh(
        new, simplex=True, qdegree=3, boundaries=bounds,
        coordinate_system_type=base.CoordinateSystem.coordinate_type)
    x, y, z = mesh.X
    exact = x**2 + y**2 + z**2
    t = uw.discretisation.MeshVariable("T_gen", mesh, 1, degree=2)
    poisson = uw.systems.Poisson(mesh, u_Field=t)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = -6.0
    for wall in walls:
        poisson.add_dirichlet_bc(sympy.Matrix([exact]), wall)
    poisson.tolerance = 1e-11
    poisson.solve()
    X = np.asarray(t.coords)
    err = np.abs(np.asarray(t.data[:, 0])
                 - (X[:, 0]**2 + X[:, 1]**2 + X[:, 2]**2))
    assert float(err.max()) < 1e-8, (
        f"the outcropped mesh assembles a wrong operator: "
        f"max |u - exact| = {float(err.max()):.3e}")


def test_an_outcropping_zone_on_a_rotated_box_embeds():
    """No wall axis-aligned: the general single-region collar."""
    R = _rotation()

    def box():
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
            cellSize=0.12, regular=False, qdegree=2)
        _rotate_dm(mesh.dm, R)
        return mesh

    # Negative control: the interior twin of the same patch, held clear
    # of every wall by more than the carve's reach.
    twin = np.array([[0.3, 0.45, 0.72], [0.7, 0.45, 0.72],
                     [0.72, 0.55, 0.45], [0.32, 0.55, 0.45]]) @ R.T
    base0 = box()
    new0, info0 = place_thin_volume(base0.dm, [twin], width=0.05,
                                    label="Zone", label_value=5)
    nb0, nt0, _bare0 = _boundary_census(new0, 5)
    assert info0["n_trace_facets"] == 0 and nb0 == 0 and nt0 == 0, (
        "the interior twin counts a band/trace; the census cannot "
        "validate the outcrop")

    patch = np.array([[0.3, 0.45, 1.1], [0.7, 0.45, 1.1],
                      [0.72, 0.55, 0.55], [0.32, 0.55, 0.55]]) @ R.T
    base = box()
    before = _volume(base.dm)
    new, info = place_thin_volume(base.dm, [patch], width=0.05,
                                  label="Zone", label_value=5)
    assert info["n_zone_cells"] > 0
    top_n = R @ np.array([0.0, 0.0, 1.0])
    n_band, n_trace, n_bare = _boundary_census(
        new, 5, pred=lambda p: abs(float(p @ top_n) - 1.0) < 1e-9)
    assert n_band > 0, "the zone left no band on the rotated top wall"
    assert n_trace == n_band == info["n_trace_facets"]
    assert n_bare == 0, "the relabel left rotated-top faces bare"
    assert _volume(new) == pytest.approx(before, rel=1e-12)

    _p2_oracle(new, base,
               ("Bottom", "Top", "Left", "Right", "Front", "Back"))


def test_a_zone_across_a_box_edge_leaves_a_trace_on_both_walls():
    """The crease overlay: the collar conforms across the Top/Front edge."""
    def box():
        return uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
            cellSize=0.09, regular=False, qdegree=2)

    Q = np.array([0.5, 0.0, 1.0])                # a point on the edge
    d1 = np.array([1.0, 0.2, 0.0])
    d1 /= np.linalg.norm(d1)
    d2 = np.array([0.0, -0.6, 0.75])
    d2 /= np.linalg.norm(d2)

    # Negative control: the same frame held inside the corner, clear of
    # both walls by more than the carve's reach.
    twin = np.array([Q - 0.22 * d1 - 0.7 * d2, Q + 0.22 * d1 - 0.7 * d2,
                     Q + 0.22 * d1 - 0.45 * d2,
                     Q - 0.22 * d1 - 0.45 * d2])
    new0, info0 = place_thin_volume(box().dm, [twin], width=0.05,
                                    label="Zone", label_value=5)
    nb0, nt0, _bare0 = _boundary_census(new0, 5)
    assert info0["n_trace_facets"] == 0 and nb0 == 0 and nt0 == 0, (
        "the interior twin counts a band/trace; the census cannot "
        "validate the outcrop")

    patch = np.array([Q - 0.22 * d1 - 0.5 * d2, Q + 0.22 * d1 - 0.5 * d2,
                      Q + 0.22 * d1 + 0.2 * d2, Q - 0.22 * d1 + 0.2 * d2])
    base = box()
    before = _volume(base.dm)
    new, info = place_thin_volume(base.dm, [patch], width=0.05,
                                  label="Zone", label_value=5)
    n_band, n_trace, n_bare = _boundary_census(new, 5)
    assert n_trace == n_band == info["n_trace_facets"] > 0
    assert n_bare == 0, "the relabel left wall faces bare"
    # The trace must straddle the crease: faces on BOTH walls, and each
    # wall's own label restored on its side (the per-removed-face rule).
    n_top, _t1, _b1 = _boundary_census(
        new, 5, pred=lambda p: abs(p[2] - 1.0) < 1e-9)
    n_front, _t2, _b2 = _boundary_census(
        new, 5, pred=lambda p: abs(p[1]) < 1e-9)
    assert n_top > 0 and n_front > 0, (
        "the band does not straddle the crease")
    top = new.getLabel("Top")
    front = new.getLabel("Front")
    bounds = base._boundaries_with("Zone")
    fS, fE = new.getHeightStratum(1)
    vS, vE = new.getDepthStratum(0)
    Xn = np.asarray(new.getCoordinatesLocal().array
                    ).reshape(-1, 3)[: vE - vS]
    for f in range(fS, fE):
        if len(new.getSupport(f)) != 1:
            continue
        verts = [int(q) - vS for q in new.getTransitiveClosure(f)[0]
                 if vS <= int(q) < vE]
        if all(abs(Xn[v][2] - 1.0) < 1e-9 for v in verts):
            assert top.getValue(f) == bounds["Top"].value
        elif all(abs(Xn[v][1]) < 1e-9 for v in verts):
            assert front.getValue(f) == bounds["Front"].value
    assert _volume(new) == pytest.approx(before, rel=1e-12)

    _p2_oracle(new, base,
               ("Bottom", "Top", "Left", "Right", "Front", "Back"))


def test_an_outcropping_zone_on_a_spherical_shell_leaves_a_trace():
    """Every facet its own region, every vertex protected faceting.

    The zone reaches the outer surface over a finite footprint; the trace
    is the footprint itself (never a partition of the sphere), the collar
    is re-triangulated facet by facet through the surviving vertices, and
    the domain's volume — a faceted S^2 x I, Euler number 2 — comes
    through exactly.
    """
    def shell():
        # The gap must hold an INTERIOR zone with the carve's envelope to
        # spare on both sides — a one-cell chain from a victim corner can
        # span h, so the margin each side is ~clearance*h + h.
        return uw.meshing.SphericalShell(radiusInner=0.25, radiusOuter=1.0,
                                         cellSize=0.13, qdegree=2)

    d1 = np.array([0.0, 1.0, 0.0])
    d2 = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    Q = np.array([0.75, 0.0, 0.75])              # outside the outer surface

    # Negative control: the same patch held mid-gap.
    twin = np.array([Q - 0.15 * d1 - 0.5 * d2, Q + 0.15 * d1 - 0.5 * d2,
                     Q + 0.15 * d1 - 0.42 * d2, Q - 0.15 * d1 - 0.42 * d2])
    new0, info0 = place_thin_volume(shell().dm, [twin], width=0.06,
                                    label="Zone", label_value=5)
    nb0, nt0, _bare0 = _boundary_census(new0, 5)
    assert info0["n_trace_facets"] == 0 and nb0 == 0 and nt0 == 0, (
        "the interior twin counts a band/trace; the census cannot "
        "validate the outcrop")

    patch = np.array([Q - 0.25 * d1 - 0.2 * d2, Q + 0.25 * d1 - 0.2 * d2,
                      Q + 0.25 * d1 + 0.35 * d2, Q - 0.25 * d1 + 0.35 * d2])
    base = shell()
    before = _volume(base.dm)
    new, info = place_thin_volume(base.dm, [patch], width=0.06,
                                  label="Zone", label_value=5)
    n_band, n_trace, n_bare = _boundary_census(new, 5)
    assert n_trace == n_band == info["n_trace_facets"] > 0
    assert n_bare == 0, "the relabel left outer-surface faces bare"
    assert _volume(new) == pytest.approx(before, rel=1e-12)
    assert info["min_volume"] > 0.0

    _p2_oracle(new, base, ("Lower", "Upper"))
