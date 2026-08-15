"""``mesh.boundary_normal()`` on a CURVED boundary must not depend on the partition.

The nodal normal is ``Σ_f |f| n̂_f`` over every facet of the boundary that meets the
node. A boundary facet is labelled on exactly one rank, so a node on a partition seam
sees only SOME of its facets locally, and normalising that partial sum gives it a
rotated normal (#564). The routine used to do exactly that, under a ``TODO(parallel)``
comment asserting it was harmless.

It was not. On ``Annulus(cellSize=0.12)``, worst nodal normal against the exact radial
one — the pre-fix numbers these tests are calibrated to fail at:

    boundary   np=1       np=2       np=3       np=4
    Upper      3.0e-10    5.8e-02    5.8e-02    5.8e-02
    Lower      5.5e-10    1.1e-01    1.1e-01    1.1e-01

5.8e-02 is 3.3 degrees, and it is not a rounding of the right answer: it is what you
get by taking ONE of a vertex's two facet normals instead of the average of both, so
its size is set by the facet's angular span and does not shrink with more ranks.
``mesh.boundary_normal()`` is the default constraint direction for
``add_constraint_bc`` and ``add_nitsche_bc``, and that error moved a constrained
free-slip answer by 3.4 % between np=1 and np=2 (#495, one member of #564).

Flat axis-aligned walls are immune — every facet of such a wall carries the same
normal, so a partial stencil normalises to the same answer — which is why the box
tests never saw this and why every test here is on a curved boundary or a corner.

Run with:
    mpirun -n 2 python -m pytest --with-mpi \\
      tests/parallel/test_1069_boundary_normal_parallel.py
    mpirun -n 4 python -m pytest --with-mpi \\
      tests/parallel/test_1069_boundary_normal_parallel.py
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.discretisation.discretisation_mesh import Mesh
from underworld3.utilities.facet_normals import facet_measure_and_normal

from serial_reference import compare, emit, mesh_fingerprint, serial_reference

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.timeout(600)]

_KEY_DECIMALS = 10


def _key(coord):
    return tuple(np.round(np.asarray(coord, dtype=float).ravel(), _KEY_DECIMALS))


def _assembled_normals(mesh, boundary):
    """``{coordinate key: normal}`` for every node the assembled field gave a normal,
    merged across ranks, plus the worst disagreement between two ranks holding the
    same node.

    The disagreement is expected to be exactly zero even on the UNFIXED code: writing
    the field through ``var.data`` runs a local→global (INSERT) → global→local round
    trip, which overwrites every ghost copy with the owner's value. Ranks agreeing is
    therefore NOT evidence that the normal is right — the owner can be confidently
    wrong and everyone will copy it. The load-bearing assertions below are the two
    oracles, not this number; it is checked because a non-zero value would mean the
    sync assumption has changed underneath us.
    """
    mesh.boundary_normal(boundary)
    var = mesh._boundary_normal_vars[boundary]
    coords = np.asarray(var.coords)
    data = np.asarray(var.data)
    live = np.linalg.norm(data, axis=1) > 0.5
    local = {_key(c): tuple(float(t) for t in v)
             for c, v in zip(coords[live], data[live])}

    merged, disagreement = {}, 0.0
    for rank_values in uw.mpi.comm.allgather(local):
        for key, value in rank_values.items():
            if key in merged:
                disagreement = max(disagreement, max(
                    abs(a - b) for a, b in zip(merged[key], value)))
            else:
                merged[key] = value
    return merged, disagreement


def _global_facet_sum_oracle(mesh, boundary):
    """``{coordinate key: normal}`` computed the way the routine SHOULD compute it,
    by an independent route: gather every boundary facet in the mesh onto every rank,
    de-duplicate by centroid, and do the whole ``Σ_f |f| n̂_f`` in numpy.

    This is an oracle rather than a re-run of the code under test: it never touches the
    variable's section, its local↔global scatter, or the DM at all after the gather, so
    it cannot reproduce a plumbing bug in the reduction. It is partition-independent by
    construction — the facet set it sums over is the global one on every rank.
    """
    dm = mesh.dm
    cdim = mesh.cdim
    vertex_start, vertex_end = dm.getDepthStratum(0)
    coord_section = dm.getCoordinateSection()
    coord_vector = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, cdim)

    local_facets = {}
    for facet in mesh._boundary_facets(boundary):
        measure, normal, exterior = facet_measure_and_normal(dm, facet)
        if not exterior:
            continue
        _, centroid, _ = dm.computeCellGeometryFVM(facet)
        vertices = [
            _key(coord_vector[coord_section.getOffset(point) // cdim])
            for point in dm.getTransitiveClosure(facet)[0]
            if vertex_start <= int(point) < vertex_end
        ]
        local_facets[_key(centroid)] = (measure, tuple(normal[:cdim]), vertices)

    all_facets = {}
    for rank_facets in uw.mpi.comm.allgather(local_facets):
        all_facets.update(rank_facets)

    accumulated = {}
    for measure, normal, vertices in all_facets.values():
        for vertex in vertices:
            accumulated[vertex] = accumulated.get(vertex, np.zeros(cdim)) \
                + measure * np.asarray(normal)
    return {k: tuple(v / np.linalg.norm(v)) for k, v in accumulated.items()}


def _worst_difference(left, right):
    """Largest ``|a - b|`` over the keys the two dictionaries share, and the number of
    keys only one of them has (which must be zero — a node set that depends on the
    partition is its own defect)."""
    shared = set(left) & set(right)
    assert shared, "the two normal fields share no nodes — the comparison is vacuous"
    worst = max(float(np.linalg.norm(np.asarray(left[k]) - np.asarray(right[k])))
                for k in shared)
    return worst, len(set(left) ^ set(right))


def _annulus(cell_size=0.12):
    return uw.meshing.Annulus(radiusInner=0.5, radiusOuter=1.0,
                              cellSize=cell_size, qdegree=3)


# --------------------------------------------------------------------------- #
#  Oracle 1 — the exact radial normal on an annulus
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("boundary,sign", [("Upper", 1.0), ("Lower", -1.0)])
def test_boundary_normal_annulus_matches_exact_radial(boundary, sign):
    """On a circular arc the measure-weighted average of a vertex's two chord normals
    is EXACTLY radial by symmetry, so the exact normal is an oracle with no golden and
    no serial run: serial lands at 3.0e-10 (Upper) / 5.5e-10 (Lower), and any partial
    stencil lands at ~half the facet's angular span, 5.8e-02 / 1.1e-01. The 1e-6 gate
    sits four orders below the defect and four above the discretisation floor.

    ``sign`` is the DOMAIN's outward direction: +r̂ on the outer arc, −r̂ on the inner
    one (the #560 convention — outward is away from the facet's own support cell, which
    on a concave boundary points at the centre of curvature).
    """
    mesh = _annulus()
    normals, disagreement = _assembled_normals(mesh, boundary)
    assert normals, f"no nodes carry a normal on {boundary}"
    worst = max(
        float(np.linalg.norm(
            np.asarray(n) - sign * np.asarray(k) / np.linalg.norm(k)))
        for k, n in normals.items())
    assert worst < 1.0e-6, (
        f"{boundary} nodal normal is {worst:.3e} from the exact radial one at "
        f"np={uw.mpi.size} over {len(normals)} nodes — a partial (rank-local) facet "
        f"stencil, #564")
    assert disagreement == 0.0, (
        f"{boundary} normals differ by {disagreement:.3e} between ranks holding the "
        f"same node")


# --------------------------------------------------------------------------- #
#  Oracle 2 — the global facet sum, in 2-D and 3-D
# --------------------------------------------------------------------------- #

def _shell(cell_size=0.35):
    return uw.meshing.SphericalShell(radiusInner=0.55, radiusOuter=1.0,
                                     cellSize=cell_size, qdegree=3)


@pytest.mark.parametrize("geometry", ["annulus", "shell"])
def test_boundary_normal_matches_global_facet_sum(geometry):
    """The assembled field equals the sum over the GLOBAL facet set, in 2-D and 3-D.

    This is the assertion that generalises: a 3-D shell's faceted normal is 8.8e-02
    (outer) / 2.3e-01 (inner) from the exact radial one at this resolution — that is
    honest discretisation error, not a defect, so the analytic oracle above cannot be
    used there. What must hold is that every rank count produces the sum over ALL the
    facets, which is what this compares against.
    """
    mesh = _annulus() if geometry == "annulus" else _shell()
    for boundary in ("Upper", "Lower"):
        assembled, _ = _assembled_normals(mesh, boundary)
        oracle = _global_facet_sum_oracle(mesh, boundary)
        worst, missing = _worst_difference(assembled, oracle)
        assert missing == 0, (
            f"{geometry} {boundary}: {missing} nodes are in one of the assembled / "
            f"oracle node sets and not the other at np={uw.mpi.size}")
        assert worst < 1.0e-12, (
            f"{geometry} {boundary}: assembled normal differs from the global facet "
            f"sum by {worst:.3e} at np={uw.mpi.size} over {len(assembled)} nodes "
            f"(#564: a rank-local stencil)")


def test_boundary_normal_oracle_fires_without_the_reduction():
    """NEGATIVE CONTROL. With the cross-rank completion disabled — which is exactly the
    pre-#564 code — both oracles above must FAIL. Without this, a fix that quietly
    stopped the assembly from producing anything, or an oracle that re-derived the same
    wrong answer, would pass the suite.
    """
    if uw.mpi.size == 1:
        pytest.skip("a rank-local stencil is the complete stencil in serial")

    original = Mesh._sum_local_dofs_across_ranks
    Mesh._sum_local_dofs_across_ranks = lambda self, subdm, values: values
    try:
        mesh = _annulus()
        normals, _ = _assembled_normals(mesh, "Upper")
        radial_error = max(
            float(np.linalg.norm(
                np.asarray(n) - np.asarray(k) / np.linalg.norm(k)))
            for k, n in normals.items())
        oracle_error, _ = _worst_difference(
            normals, _global_facet_sum_oracle(mesh, "Upper"))
    finally:
        Mesh._sum_local_dofs_across_ranks = original

    assert radial_error > 1.0e-3, (
        f"the radial oracle does not see a rank-local stencil at np={uw.mpi.size} "
        f"(error {radial_error:.3e}); it cannot be trusted to see a regression")
    assert oracle_error > 1.0e-3, (
        f"the global-facet-sum oracle does not see a rank-local stencil at "
        f"np={uw.mpi.size} (error {oracle_error:.3e})")


# --------------------------------------------------------------------------- #
#  The regression the fix is most likely to introduce
# --------------------------------------------------------------------------- #

def test_boundary_normal_corner_is_not_averaged_across_boundaries():
    """Each boundary is assembled into its OWN variable, so the vertex where Top meets
    Right keeps (0,1) for Top and (1,0) for Right rather than the 45-degree bisector.
    A cross-rank reduction done on a shared vector instead of per boundary would
    average across the discontinuity and this is what would catch it."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), qdegree=3)
    top, _ = _assembled_normals(mesh, "Top")
    right, _ = _assembled_normals(mesh, "Right")
    corner = _key((1.0, 1.0))
    assert corner in top and corner in right, (
        "the (1,1) corner should carry a normal on BOTH Top and Right")
    assert np.allclose(top[corner], (0.0, 1.0), rtol=0, atol=1e-12), (
        f"Top normal at the corner is {top[corner]}, not (0, 1)")
    assert np.allclose(right[corner], (1.0, 0.0), rtol=0, atol=1e-12), (
        f"Right normal at the corner is {right[corner]}, not (1, 0)")


# --------------------------------------------------------------------------- #
#  End to end: the two consumers of the default normal
# --------------------------------------------------------------------------- #

def _nitsche_annulus_diagnostics():
    """Nitsche free-slip on a curved boundary through the DEFAULT (assembled) normal —
    ``add_nitsche_bc`` shares ``mesh.boundary_normal`` with ``add_constraint_bc`` and
    was equally exposed, with no test covering it. Returns ((velocity L2, radial
    leakage on Upper), mesh fingerprint).

    With the cross-rank completion disabled this reads, at solver tolerance 1e-9:

        np=1  L2 1.413526513414e-02   leak 1.729537946e-04
        np=2     1.414677409720e-02        3.752196269e-04
        np=3     1.422846013917e-02        2.873963117e-04
        np=4     1.413111891962e-02        3.149424657e-04

    — 6.6e-03 in the velocity and more than DOUBLE the wall-normal leakage. With it,
    every rank count agrees to 3.6e-10 in the velocity and bit-identically in the
    leakage. Both are stable from tolerance 1e-9 to 1e-12, so neither is the linear
    solve.

    ``local_h=False`` is deliberate and it is not a workaround for this fix. The
    default ``local_h=True`` scales the Nitsche penalty by ``mesh.cell_size()``, which
    is built from ``Mesh._get_mesh_sizes`` — a kd-tree query against THIS RANK's cell
    centroids, and so partition-dependent in its own right (on this mesh the field's
    sum is 26.0822 at np=1, 26.1211 at np=2, 26.1386 at np=4, and its max moves at
    np=4). That is a SEPARATE defect from the boundary normal, it is not what #564 is
    about, and leaving it in would make this test measure the two together. See the
    TODO(BUG) on ``Mesh._assemble_cell_size``.
    """
    RI, RO = 0.5, 1.0
    mesh = uw.meshing.Annulus(radiusInner=RI, radiusOuter=RO, cellSize=0.12, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    theta = sympy.atan2(y, x)
    v = uw.discretisation.MeshVariable("Vn69", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("Pn69", mesh, 1, degree=1, continuous=True)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = sympy.Matrix([[
        x / r * sympy.cos(4 * theta) * (r - RI) * (RO - r) * 40.0,
        y / r * sympy.cos(4 * theta) * (r - RI) * (RO - r) * 40.0]])
    stokes.add_essential_bc((0.0, 0.0), "Lower")
    # default normal= is the assembled one — that is what is under test
    stokes.add_nitsche_bc(0.0, "Upper", local_h=False)
    stokes.tolerance = 1.0e-9
    stokes.petsc_options["snes_type"] = "ksponly"
    stokes.solve()

    L2 = float(np.sqrt(uw.maths.Integral(mesh, v.sym.dot(v.sym)).evaluate()))
    vr = v.sym[0] * x / r + v.sym[1] * y / r
    leak = float(np.sqrt(uw.maths.BdIntegral(
        mesh=mesh, fn=vr ** 2, boundary="Upper").evaluate()))
    return (L2, leak), mesh_fingerprint(mesh)


def test_nitsche_freeslip_annulus_partition_independent():
    """``add_nitsche_bc`` on a curved boundary with the default normal reproduces its
    OWN np=1 answer (computed in this environment, not a recorded constant)."""
    values, fingerprint = _nitsche_annulus_diagnostics()
    compare(values, serial_reference(__file__, "nitsche"),
            rtols=(1e-8, 1e-8), labels=("velocity L2", "Upper radial leakage"),
            fingerprint=fingerprint, what="Nitsche free-slip annulus")


if __name__ == "__main__":
    import sys

    _kind = sys.argv[1] if len(sys.argv) > 1 else "nitsche"
    if _kind == "nitsche":
        _values, _fingerprint = _nitsche_annulus_diagnostics()
        emit(_values, _fingerprint)
        uw.mpi.pprint(f"DIAG_NITSCHE L2={_values[0]:.12e} leak={_values[1]:.6e}")
    else:
        raise SystemExit(f"unknown kind {_kind!r}")
