r"""The thick twin of a sheet mesh: extrude a surface mesh through a thickness.

A thin sheet (``ThinSheet`` on a surface mesh) and a thick one (``Stokes`` on a
thin 3-D layer) are the same model when they share everything but the solver
class and the mesh. :func:`extrude_sheet` makes the second mesh from the first,
so the two carry the same lateral boundaries under the same names and the same
faults:

* every in-plane boundary of the sheet becomes the vertical wall it bounds, under
  the same name (a model's ``add_essential_bc(..., "Left")`` means the same
  thing on both);
* the two faces of the layer are added as ``Base`` and ``Surface`` (the drag and
  traction of ``add_basal_drag`` / ``add_basal_traction`` go on ``Base``);
* a split-node fault of the sheet becomes a vertical fault through the whole
  layer. Its slit is extruded with it, so the fault is split at the surface and
  at the base, as a fault through a plate is, and its tips become shared
  vertical edges — the tip rule of the sheet one dimension up. The fault's
  side labels and the minus-to-plus point pairing are rebuilt for the layer, so
  ``Stokes.add_fault_bc`` works on it as ``ThinSheet.add_fault_bc`` does on the
  sheet.

The layer is centred on the sheet: on a flat sheet it spans
:math:`z \in [-L/2, L/2]`, on a spherical manifold of radius :math:`R` the shell
:math:`r \in [R - L/2, R + L/2]` (each node moved along its own radius).

Each prism of the layer is cut into three tetrahedra by the global vertex order,
which makes the cut of every shared quadrilateral face the same from both
sides. The two copies of a fault vertex take the SAME place in that order (their
original's), so the two sides of a fault are triangulated identically and their
faces coincide pair by pair.
"""

from enum import Enum

import numpy as np
from petsc4py import PETSc

import underworld3 as uw


def _sheet_topology(mesh):
    """Vertex coordinates, triangles (as vertex indices) and the point-to-index
    map of a 2-D (or surface) simplex mesh."""
    dm = mesh.dm
    v_start, v_end = dm.getDepthStratum(0)
    c_start, c_end = dm.getHeightStratum(0)
    cdim = dm.getCoordinateDim()
    coords = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, cdim).copy()
    triangles = np.empty((c_end - c_start, 3), dtype=np.int64)
    for c in range(c_start, c_end):
        closure = dm.getTransitiveClosure(c)[0]
        verts = [int(p) - v_start for p in closure if v_start <= p < v_end]
        if len(verts) != 3:
            raise ValueError("extrude_sheet needs a triangle mesh")
        triangles[c - c_start] = verts
    return coords, triangles, v_start


def _edge_vertices(dm, edge, v_start):
    cone = dm.getCone(edge)
    return int(cone[0]) - v_start, int(cone[1]) - v_start


def extrude_sheet(mesh, thickness, layers=2, base="Base", surface="Surface",
                  qdegree=None, verbose=False, with_tail=True):
    r"""Extrude a sheet mesh through ``thickness`` into a layer of tetrahedra.

    Parameters
    ----------
    mesh : Mesh
        A 2-D triangle mesh, or a triangulated spherical manifold, possibly
        carrying split-node faults (``mesh.add_fault``).
    thickness : float
        Layer thickness :math:`L`, centred on the sheet.
    layers : int, default 2
        Elements through the thickness.
    base, surface : str
        Names of the two faces of the layer.
    qdegree : int, optional
        Quadrature degree of the new mesh; the sheet's by default.
    with_tail : bool, default True
        Give the layer a geometric-multigrid tail: the sheet with the layer
        count halved down to one, then the sheet's own coarse levels at one
        layer (coarsened through the thickness first, see the note in the code).

    Returns
    -------
    Mesh
        The layer, with the sheet's boundaries as walls, ``base`` and
        ``surface``, and the sheet's faults as vertical faults with their pairing.
    """
    from underworld3.coordinates import CoordinateSystemType

    if uw.mpi.size > 1:
        # TODO(DESIGN): build from the gathered sheet on one rank and distribute;
        # the fault pairing then has to migrate with the distribution, as
        # fault_split's redistribution does for a split.
        raise NotImplementedError("extrude_sheet runs in serial for now")
    if layers < 1:
        raise ValueError("layers must be at least 1")

    dm2 = mesh.dm
    coords, triangles, v_start = _sheet_topology(mesh)
    nv = len(coords)
    spherical = mesh.dim < mesh.cdim
    if spherical and (mesh.CoordinateSystem.coordinate_type
                      is not CoordinateSystemType.SPHERICAL):
        raise NotImplementedError("extrude_sheet: only flat sheets and spherical manifolds")

    # The order that cuts each prism: a fault's minus copy of a vertex takes its
    # plus copy's place, so both sides cut their prisms alike.
    key = np.arange(nv)
    pairs2 = getattr(mesh, "_fault_point_pairs", {})
    for pairs in pairs2.values():
        for q_minus, q_plus in pairs.items():
            if v_start <= q_minus < v_start + nv:
                key[q_minus - v_start] = q_plus - v_start

    # vertices, layer by layer
    offsets = -0.5 * thickness + thickness * np.arange(layers + 1) / layers
    if spherical:
        radius = np.linalg.norm(coords, axis=1)
        rhat = coords / radius[:, None]
        xyz = np.concatenate([rhat * (radius + s)[:, None] for s in offsets])
    else:
        xyz = np.concatenate([np.column_stack([coords, np.full(nv, s)]) for s in offsets])

    # tetrahedra, three per prism
    cells = []
    for k in range(layers):
        lo, hi = k * nv, (k + 1) * nv
        for tri in triangles:
            a, b, c = sorted(tri, key=lambda v: key[v])
            A, B, C = lo + a, lo + b, lo + c
            A1, B1, C1 = hi + a, hi + b, hi + c
            cells += [[A, B, C, A1], [B, C, A1, B1], [C, A1, B1, C1]]
    cells = np.array(cells, dtype=np.int32)
    # PETSc's orientation: the first three vertices see the fourth on the side
    # their right-hand normal points AWAY from (a negative triple product). With
    # the other sign every volume integral changes sign against the facet ones —
    # measured: the domain volume came out -0.1 and a basal drag pushed.
    p = xyz[cells]
    vol = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])
    flip = vol > 0
    cells[flip, 2], cells[flip, 3] = cells[flip, 3].copy(), cells[flip, 2].copy()

    dm = PETSc.DMPlex().createFromCellList(3, cells, xyz, interpolate=True,
                                           comm=PETSc.COMM_SELF)
    v3_start, _ = dm.getDepthStratum(0)

    def vertex(k, i):
        return v3_start + k * nv + i

    def face(verts):
        join = dm.getFullJoin(verts)
        if len(join) != 1:
            raise RuntimeError(f"extrude_sheet internal: no face on vertices {verts}")
        return int(join[0])

    def quad_faces(a, b):
        """The two triangles of each layer's quad over the sheet edge (a, b)."""
        if key[a] > key[b]:
            a, b = b, a
        out = []
        for k in range(layers):
            A, B, A1, B1 = vertex(k, a), vertex(k, b), vertex(k + 1, a), vertex(k + 1, b)
            out += [face([A, B, A1]), face([B, A1, B1])]
        return out

    def label_with_closure(name, value, points):
        dm.createLabel(name)
        for q in points:
            for r in dm.getTransitiveClosure(q)[0]:
                dm.setLabelValue(name, int(r), value)

    def label_faces(name, value, points):
        dm.createLabel(name)
        for q in points:
            dm.setLabelValue(name, int(q), value)

    # the sheet's boundaries -> walls (fault labels: faces only, as on the sheet)
    members = {}
    fault_side_names = set()
    for name in pairs2:
        fault_side_names |= {name, f"{name}Plus", f"{name}Minus"}
    e_start, e_end = dm2.getDepthStratum(1)
    for b in mesh.boundaries:
        if b.name == "All_Boundaries":
            continue
        lab = dm2.getLabel(b.name)
        if not lab or lab.getStratumSize(b.value) == 0:
            continue
        edges = [int(q) for q in lab.getStratumIS(b.value).getIndices() if e_start <= q < e_end]
        if not edges:
            # not a boundary of the sheet's area (a manifold labels the whole
            # surface, for one): nothing to stand up as a wall
            continue
        if b.name in (base, surface):
            raise ValueError(
                f"the sheet has a boundary called {b.name!r}, which the layer's "
                f"{'base' if b.name == base else 'surface'} would take; pass "
                "another name for it")
        members[b.name] = b.value
        faces3 = [f for e in edges for f in quad_faces(*_edge_vertices(dm2, e, v_start))]
        if b.name in fault_side_names:
            label_faces(b.name, b.value, faces3)
        else:
            label_with_closure(b.name, b.value, faces3)

    taken = set(members.values())
    for name in (base, surface):
        value = max([v for v in taken if v < 666] + [0]) + 1
        while value in taken:
            value += 1
        members[name] = value
        taken.add(value)
    for name, k in ((base, 0), (surface, layers)):
        label_with_closure(name, members[name],
                           [face([vertex(k, i) for i in tri]) for tri in triangles])

    # the fault pairing: every point on the minus side whose vertices include a
    # duplicated one, paired with the point on the same plus-copy vertices
    pairs3 = {}
    for name, pairs in pairs2.items():
        vmap = {}
        for q_minus, q_plus in pairs.items():
            if v_start <= q_minus < v_start + nv:
                for k in range(layers + 1):
                    vmap[vertex(k, q_minus - v_start)] = vertex(k, q_plus - v_start)
        value = mesh.boundaries[f"{name}Minus"].value
        lab = dm.getLabel(f"{name}Minus")
        minus_faces = lab.getStratumIS(value).getIndices()
        found = {}
        for f in minus_faces:
            for q in dm.getTransitiveClosure(int(f))[0]:
                q = int(q)
                if q in found:
                    continue
                if v3_start <= q < v3_start + nv * (layers + 1):
                    verts = [q]
                else:
                    verts = [int(r) for r in dm.getTransitiveClosure(q)[0]
                             if v3_start <= r < v3_start + nv * (layers + 1)]
                if not any(v in vmap for v in verts):
                    continue
                mapped = [vmap.get(v, v) for v in verts]
                found[q] = mapped[0] if len(mapped) == 1 else face(mapped)
        pairs3[name] = found

    # every exterior face, as the sheet's All_Boundaries; a split fault's faces
    # have one supporting cell each, so they are marked too and taken out again
    dm.markBoundaryFaces("All_Boundaries", 1001)
    all_label = dm.getLabel("All_Boundaries")
    for name in pairs2:
        lab = dm.getLabel(name)
        value = mesh.boundaries[name].value
        if lab and lab.getStratumSize(value):
            for q in lab.getStratumIS(value).getIndices():
                all_label.clearValue(int(q), 1001)

    boundaries = Enum("boundaries", members)
    thick = uw.discretisation.Mesh(
        dm, simplex=True,
        coordinate_system_type=mesh.CoordinateSystem.coordinate_type,
        qdegree=qdegree if qdegree is not None else mesh.qdegree,
        boundaries=boundaries, verbose=verbose)
    if pairs3:
        p_start, p_end = thick.dm.getChart()
        if (p_start, p_end) != dm.getChart():
            raise RuntimeError("extrude_sheet internal: the mesh renumbered its points")
        thick._fault_point_pairs = pairs3
    if getattr(mesh, "_fault_traces", None):
        thick._fault_traces = dict(mesh._fault_traces)
    thick._sheet_extrusion = {"sheet": mesh, "thickness": thickness, "layers": layers,
                              "base": base, "surface": surface}

    # The multigrid tail. A thin layer couples strongly through its thickness
    # (the vertical spacing is the smallest), and point smoothers only smooth
    # along the strong coupling, so the layer is coarsened through its thickness
    # FIRST — the fine sheet with the layer count halved down to one — and only
    # then laterally, through the sheet's own coarse levels at one layer. The
    # coarse end of the hierarchy is then essentially the thin sheet. Without a
    # tail the solve falls back to algebraic multigrid. The coarse levels do not
    # carry the faults (the fine level alone is cut).
    # TODO(BUG): with a split fault the layer's solve goes through the rotated
    # pair constraints, and custom geometric multigrid on that 3-D rotated
    # operator fails on its first application (DIVERGED_PC_FAILED, a NaN
    # residual) with any tail tried — barycentric or RBF transfers, with or
    # without the through-thickness levels — where algebraic multigrid
    # converges. The 2-D rotated pairs take an inherited tail without trouble.
    # Until that is found, a faulted layer is built without a tail.
    # Found 2026-10-12; see the thin-sheet plan file.
    if with_tail and not pairs2:
        if len(getattr(mesh, "dm_hierarchy", []) or []) > 1:
            sheet_levels = list(mesh._coarse_level_meshes())[:-1]
        else:
            sheet_levels = []
        vertical = []
        n = layers // 2
        while n >= 1:
            vertical.append(extrude_sheet(mesh, thickness, layers=n, base=base,
                                          surface=surface, qdegree=qdegree,
                                          with_tail=False))
            n //= 2
        lateral = [extrude_sheet(level, thickness, layers=1, base=base, surface=surface,
                                 qdegree=qdegree, with_tail=False)
                   for level in sheet_levels]
        tail = lateral + list(reversed(vertical))
        if tail:
            thick._custom_mg_coarse_meshes = tail
            thick._custom_mg_builder = getattr(mesh, "_custom_mg_builder", "barycentric")
            thick._custom_mg_fac_zone = None
    return thick
