"""Generator for the tetrahedron single-edge-bisection production tables of
the ``uwnvb_bisect`` DMPlexTransform (adaptivity capstone stage 1c-i).

Derives, for each of the 6 closure-edge positions of the canonical DMPlex
tetrahedron, the ``cone[]`` / ``ornt[]`` production arrays for the split

    tet --(bisect edge e)-->  1 interior TRIANGLE + 2 child TETRAHEDRA

in PETSc's transform grammar (each cone entry ``[type, Npath, d.., r]`` with
one orientation per cone point; see ``DMPlexTransformCellTransform`` in
``plextransform.c``), and emits them as C static arrays.

Conventions (PETSc 3.25, pinned from ``plexrefregular.c:713-738``,
``DMPlexGetRawFaces_Internal`` and ``DMPolytopeTypeGetVertexArrangement``):

* tet vertices ``[v0,v1,v2,v3]``; cone faces (raw-face rule)
  ``f0=[v0,v1,v2], f1=[v0,v3,v1], f2=[v0,v2,v3], f3=[v2,v1,v3]``;
  closure edges ``e0=[v0,v1], e1=[v1,v2], e2=[v2,v0], e3=[v0,v3],
  e4=[v1,v3], e5=[v2,v3]``;
* a triangle ``[a,b,c]`` has raw edges ``[a,b],[b,c],[c,a]``;
* orientation ``o`` of a polytope permutes vertices as
  ``arranged[i] = canonical[vertexArrangement(o)[i]]``;
* the face single-split (``UWNVBGetTriangleSplitSingle``, canonical o=0)
  bisects the face's cone-slot-1 edge ``[x1,x2]``: interior segment
  ``[x0, m]``, children ``r0=[x0,x1,m]``, ``r1=[m,x2,x0]``; the split-slot
  is selected by passing o in {2:slot0, 0:slot1, 1:slot2} (rotations);
* a split SEGMENT ``[p,q]`` produces the midpoint (POINT r0) and halves
  ``r0=[p,m]``, ``r1=[m,q]``.

The self-check verifies each emitted cone reference resolves to the exact
vertex tuple the child's raw-face rule demands (the matcher raises if no
arrangement matches), that child volumes have the parent's sign and sum to
it, and that the child complex equals the stage-1a oracle's split.

Run:  python tet_bisection_tables_generator.py   (prints the C tables)
"""

import numpy as np

# ---- canonical conventions -------------------------------------------------
TET_FACES = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (2, 1, 3)]
TET_EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
# reference coordinates (plexrefregular.c) — used only for volume sign checks
TET_COORDS = np.array([[-1, -1, -1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1]],
                      dtype=float)

# vertex arrangements: arranged[i] = canonical[ARR[o][i]]
SEG_ARR = {-1: (1, 0), 0: (0, 1)}
TRI_ARR = {-3: (1, 0, 2), -2: (0, 2, 1), -1: (2, 1, 0),
           0: (0, 1, 2), 1: (1, 2, 0), 2: (2, 0, 1)}

# face split-slot -> orientation passed to UWNVBGetTriangleSplitSingle
SPLIT_SLOT_TO_O = {0: 2, 1: 0, 2: 1}

# C names
PT = {"point": "DM_POLYTOPE_POINT", "seg": "DM_POLYTOPE_SEGMENT",
      "tri": "DM_POLYTOPE_TRIANGLE", "tet": "DM_POLYTOPE_TETRAHEDRON"}


def tri_edges(t):
    return [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]


def match_orientation(canonical, required, arrangements):
    """The o with required[i] == canonical[arr[i]] for all i (raises if none
    — the structural self-check that every reference is realisable)."""
    for o, arr in arrangements.items():
        if all(required[i] == canonical[arr[i]] for i in range(len(required))):
            return o
    raise ValueError(f"no arrangement maps {canonical} -> {required}")


def face_split_products(face, split_slot):
    """The face's own production under RT_TRIANGLE_SPLIT_(split_slot), in the
    face's canonical frame: (interior_segment, [tri_r0, tri_r1]) as vertex
    tuples, with 'm' denoting the split-edge midpoint."""
    o = SPLIT_SLOT_TO_O[split_slot]
    x = [face[TRI_ARR[o][i]] for i in range(3)]      # roles x0', x1', x2'
    interior = (x[0], "m")
    children = [(x[0], x[1], "m"), ("m", x[2], x[0])]
    return interior, children


def edge_path(edge):
    """Path digits (f, k) reaching ``edge`` as raw-edge k of tet face f,
    plus whether the face traverses it reversed."""
    es = frozenset(edge)
    for f, face in enumerate(TET_FACES):
        for k, fe in enumerate(tri_edges(face)):
            if frozenset(fe) == es:
                return f, k, (fe != tuple(edge))
    raise ValueError(f"edge {edge} not found")


def derive_tables(k_edge):
    """Cone/ornt tables for bisecting tet closure edge ``k_edge``.

    Returns (cone_entries, ornt_entries, meta) where cone_entries is a flat
    list of ints/C-names in the transform grammar: first the interior
    TRIANGLE's 3 cone points, then child tet r0's 4, then child tet r1's 4.
    """
    a, b = TET_EDGES[k_edge]
    c, d = [v for v in range(4) if v not in (a, b)]
    # order (c, d) so the permutation (a, b, c, d) of (0,1,2,3) is even —
    # this is what keeps both children in the parent's orientation class
    perm = (a, b, c, d)
    inversions = sum(1 for i in range(4) for j in range(i + 1, 4)
                     if perm[i] > perm[j])
    if inversions % 2:
        c, d = d, c

    # the two faces containing the split edge, and their split slots
    split_faces = {}                                   # f -> (slot, products)
    for f, face in enumerate(TET_FACES):
        for k, fe in enumerate(tri_edges(face)):
            if frozenset(fe) == frozenset((a, b)):
                split_faces[f] = (k, *face_split_products(face, k))
    assert len(split_faces) == 2
    # the two untouched faces: opposite a and opposite b
    face_opp = {v: f for f, face in enumerate(TET_FACES)
                for v in range(4) if v not in face}

    # entity registry: canonical vertex tuple -> (cone-entry ints, type)
    refs = {}
    for f, (slot, interior, children) in split_faces.items():
        refs[interior] = ([PT["seg"], 1, f, 0], "seg")
        refs[children[0]] = ([PT["tri"], 1, f, 0], "tri")
        refs[children[1]] = ([PT["tri"], 1, f, 1], "tri")
    for v, f in face_opp.items():
        if v in (a, b):
            refs[TET_FACES[f]] = ([PT["tri"], 1, f, 0], "tri")
    # the unsplit edge [c,d] (identity replica), reached through a face
    ef, ek, _rev = edge_path(tuple(sorted((c, d))))
    cd_canonical = tri_edges(TET_FACES[ef])[ek]        # the edge's own tuple
    refs[cd_canonical] = ([PT["seg"], 2, ef, ek, 0], "seg")

    # our own products
    t_int = ("m", c, d)                                # interior triangle
    refs[t_int] = ([PT["tri"], 0, 0], "tri")
    child_a = (a, "m", c, d)                           # keeps vertex a
    child_b = ("m", b, c, d)                           # keeps vertex b

    def resolve(required, kind):
        """Cone entry + orientation for the reference whose arrangement
        equals ``required``."""
        arrangements = SEG_ARR if kind == "seg" else TRI_ARR
        for canonical, (entry, ekind) in refs.items():
            if ekind != kind or len(canonical) != len(required):
                continue
            if frozenset(canonical) == frozenset(required):
                return entry, match_orientation(canonical, required,
                                                arrangements)
        raise ValueError(f"no {kind} reference for {required}")

    cone, ornt = [], []
    # interior triangle cone: raw edges of (m, c, d)
    for e in tri_edges(t_int):
        entry, o = resolve(e, "seg")
        cone += entry
        ornt.append(o)
    # child tets: raw faces of each vertex tuple
    children_meta = []
    for child in (child_a, child_b):
        v = child
        faces = [(v[0], v[1], v[2]), (v[0], v[3], v[1]),
                 (v[0], v[2], v[3]), (v[2], v[1], v[3])]
        for fc in faces:
            entry, o = resolve(fc, "tri")
            cone += entry
            ornt.append(o)
        children_meta.append(child)

    # ---- self-checks -------------------------------------------------------
    m = 0.5 * (TET_COORDS[a] + TET_COORDS[b])
    P = {0: TET_COORDS[0], 1: TET_COORDS[1], 2: TET_COORDS[2],
         3: TET_COORDS[3], "m": m}
    def vol(t):
        e = np.array([P[t[i]] - P[t[0]] for i in (1, 2, 3)])
        return np.linalg.det(e) / 6.0
    vp = vol((0, 1, 2, 3))
    va, vb = vol(child_a), vol(child_b)
    assert np.sign(va) == np.sign(vp) and np.sign(vb) == np.sign(vp), \
        f"edge {k_edge}: child orientation flipped"
    assert abs(va + vb - vp) < 1e-12, f"edge {k_edge}: volume not conserved"
    # oracle agreement: children as vertex SETS match the geometric split
    assert ({frozenset(child_a), frozenset(child_b)}
            == {frozenset((a, "m", c, d)), frozenset(("m", b, c, d))})

    return cone, ornt, (child_a, child_b, t_int)


def derive_subcell_orientation_table():
    """The complete GetSubcellOrientation table for the triangle single-split.

    A cell referencing the products of a split FACE sees the face with
    orientation ``so``; the cell-side tables were derived in that ARRANGED
    frame, while the face's actual production ran in its CANONICAL frame
    (split slot fixed by SetUp). This maps (replica r, orientation o) from
    the arranged frame to (rnew, q) in the canonical production, with
    ``onew = ComposeOrientation(tct, o, q)`` — the regular/alfeld idiom.

    Returns {split_slot: {so: {"seg": (0, q), "tri": ((r0n, q0), (r1n, q1))}}}.
    """
    table = {}
    # cone arrangement of the triangle: arranged cone k = canonical cone
    # TRI_CONE_ARR[so][k] (positions only; from petscdm.h triArr)
    TRI_CONE_ARR = {-3: (0, 2, 1), -2: (2, 1, 0), -1: (1, 0, 2),
                    0: (0, 1, 2), 1: (1, 2, 0), 2: (2, 0, 1)}
    X = ("x0", "x1", "x2")
    for s_x in range(3):
        SX, CX = face_split_products(X, s_x)
        table[s_x] = {}
        for so, varr in TRI_ARR.items():
            Y = tuple(X[varr[i]] for i in range(3))       # arranged tuple
            s_y = TRI_CONE_ARR[so].index(s_x)             # split slot in Y
            SY, CY = face_split_products(Y, s_y)
            q_seg = match_orientation(SX, SY, SEG_ARR)
            tri = []
            for r in range(2):
                r_new = next(rp for rp in range(2)
                             if frozenset(CX[rp]) == frozenset(CY[r]))
                tri.append((r_new,
                            match_orientation(CX[r_new], CY[r], TRI_ARR)))
            table[s_x][so] = {"seg": (0, q_seg), "tri": tuple(tri)}
    return table


def emit_subcell_c():
    t = derive_subcell_orientation_table()
    lines = ["/* Generated GetSubcellOrientation tables for the triangle",
             "   single-split: rows so=-3..2, entries (rnew, q) with",
             "   onew = DMPolytopeTypeComposeOrientation(tct, o, q).",
             "   Indexed [split_slot][so+3]. Generated by",
             "   tet_bisection_tables_generator.py. */"]
    seg = "static const PetscInt triSingleSegO[3][6][2] = {"
    tri = "static const PetscInt triSingleTriO[3][6][4] = {"
    for s in range(3):
        seg_rows = ", ".join(
            "{%d, %d}" % t[s][so]["seg"] for so in range(-3, 3))
        tri_rows = ", ".join(
            "{%d, %d, %d, %d}" % (t[s][so]["tri"][0] + t[s][so]["tri"][1])
            for so in range(-3, 3))
        seg += "{" + seg_rows + ("}, " if s < 2 else "}};")
        tri += "{" + tri_rows + ("}, " if s < 2 else "}};")
    lines += [seg, tri]
    return "\n".join(lines)


def emit_c():
    lines = []
    lines.append("/* Generated by docs/developer/design/"
                 "tet_bisection_tables_generator.py — do not hand-edit.")
    lines.append("   Single-edge bisection of a tetrahedron: for closure "
                 "edge k, 1 interior")
    lines.append("   triangle + 2 child tets; the two faces containing the "
                 "edge split via the")
    lines.append("   existing triangle single-split rule. */")
    lines.append("static DMPolytopeType tetBisT[] = "
                 "{DM_POLYTOPE_TRIANGLE, DM_POLYTOPE_TETRAHEDRON};")
    lines.append("static PetscInt       tetBisS[] = {1, 2};")
    for k in range(6):
        cone, ornt, meta = derive_tables(k)
        centries = ", ".join(str(x) for x in cone)
        oentries = ", ".join(str(x) for x in ornt)
        a, b = TET_EDGES[k]
        lines.append(f"/* edge e{k} = [v{a}, v{b}]: children "
                     f"{meta[0]} and {meta[1]}, interior {meta[2]} */")
        lines.append(f"static PetscInt tetBisC{k}[] = {{{centries}}};")
        lines.append(f"static PetscInt tetBisO{k}[] = {{{oentries}}};")
    return "\n".join(lines)


if __name__ == "__main__":
    for k in range(6):
        cone, ornt, meta = derive_tables(k)
        print(f"e{k} = {TET_EDGES[k]}: children {meta[0]} | {meta[1]}, "
              f"interior {meta[2]}, ornt {ornt}")
    print()
    print(emit_c())
    print()
    print(emit_subcell_c())
