"""3-D fault-network preparation: PLANAR patches, prepare-first.

The 3-D analogue of :func:`prepare_fault_network`: two planar fault
patches meet along a LINE SEGMENT (the clip of their plane-plane
intersection line to both polygons). The offset form generalises —
the JUNIOR patch is cut into two pieces by removing a ligament band
about the intersection line, measured IN ITS OWN PLANE, so the mesher
only ever embeds disjoint patches (surface-surface intersection never
reaches gmsh, exactly as shared vertices never reach ``add_fault`` in
2-D).

v1 scope, refused loudly outside it: PLANAR patches (the
``rim_polygon`` contract), CONVEX rims (clipping preserves convexity
and the triangulator is safe), genuine X crossings (a near-miss —
close but not crossing — is reported and refused rather than guessed
at).

Junction records carry the intersection SEGMENT: the damage glue in
3-D is a tube (distance to segment), not a disc.
"""

import numpy as np

from .faults import FaultSurface


# ---------------------------------------------------------------------------
# planar-polygon geometry
# ---------------------------------------------------------------------------

def _plane_basis(poly, tol=1e-8):
    """Orthonormal in-plane basis of a planar polygon.

    Returns ``(origin, n_hat, u_hat, v_hat)``; raises on non-planar or
    degenerate input.
    """
    P = np.asarray(poly, dtype=float)
    if P.ndim != 2 or P.shape[0] < 3 or P.shape[1] != 3:
        raise ValueError("polygon must be (N, 3) with N >= 3")
    origin = P.mean(axis=0)
    Q = P - origin
    # smallest principal direction = the normal
    _, s, vt = np.linalg.svd(Q, full_matrices=False)
    n_hat = vt[2]
    span = max(float(s[0]), 1e-30)
    if float(np.abs(Q @ n_hat).max()) > tol * span:
        raise ValueError("polygon is not planar within tolerance")
    u_hat = vt[0]
    v_hat = np.cross(n_hat, u_hat)
    return origin, n_hat, u_hat, v_hat


def _to_2d(pts, origin, u_hat, v_hat):
    rel = np.asarray(pts, dtype=float) - origin
    return np.column_stack([rel @ u_hat, rel @ v_hat])


def _to_3d(pts2, origin, u_hat, v_hat):
    pts2 = np.asarray(pts2, dtype=float)
    return (origin[None, :] + pts2[:, :1] * u_hat[None, :]
            + pts2[:, 1:2] * v_hat[None, :])


def _is_convex(poly2, tol=1e-12):
    P = np.asarray(poly2, dtype=float)
    n = len(P)
    cross = np.array([np.cross(P[(i + 1) % n] - P[i],
                               P[(i + 2) % n] - P[(i + 1) % n])
                      for i in range(n)])
    return bool((cross >= -tol).all() or (cross <= tol).all())


def _clip_halfplane(poly2, a, b, c):
    """Sutherland–Hodgman: keep the part of ``poly2`` with
    ``a x + b y + c >= 0``."""
    out = []
    P = list(np.asarray(poly2, dtype=float))
    n = len(P)
    for i in range(n):
        A, B = P[i], P[(i + 1) % n]
        da = a * A[0] + b * A[1] + c
        db = a * B[0] + b * B[1] + c
        if da >= 0:
            out.append(A)
            if db < 0:
                t = da / (da - db)
                out.append(A + t * (B - A))
        elif db >= 0:
            t = da / (da - db)
            out.append(A + t * (B - A))
    return np.array(out) if len(out) >= 3 else None


def _poly_area(poly2):
    P = np.asarray(poly2, dtype=float)
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1))
                           - np.dot(y, np.roll(x, -1))))


def _line_clip_interval(poly2, p0, d):
    """Clip the line ``p0 + t d`` (2-D) to a convex polygon; returns
    ``(t_lo, t_hi)`` or ``None``."""
    t_lo, t_hi = -np.inf, np.inf
    P = np.asarray(poly2, dtype=float)
    n = len(P)
    # polygon orientation for inward normals
    area2 = float(np.dot(P[:, 0], np.roll(P[:, 1], -1))
                  - np.dot(P[:, 1], np.roll(P[:, 0], -1)))
    sgn = 1.0 if area2 > 0 else -1.0
    for i in range(n):
        A, B = P[i], P[(i + 1) % n]
        e = B - A
        nrm = sgn * np.array([-e[1], e[0]])       # inward
        den = float(nrm @ d)
        num = float(nrm @ (A - p0))
        if abs(den) < 1e-30:
            # parallel edge: the line is inside this half-plane iff
            # nrm . (p0 - A) >= 0, i.e. num <= 0.  (num > 0 means p0
            # sits OUTSIDE the edge's half-plane.)  The sign here was
            # inverted; on LAPACK builds that return exact zeros for
            # axis-aligned plane bases (den == 0.0 exactly) that made
            # crossing_segment miss genuine crossings, while builds
            # with ~1e-17 jitter in den took the (correct) division
            # branch and passed by luck.
            if num > 0:
                return None                        # parallel, outside
            continue
        t = num / den
        if den > 0:
            t_lo = max(t_lo, t)
        else:
            t_hi = min(t_hi, t)
        if t_lo > t_hi:
            return None
    return (t_lo, t_hi) if t_lo < t_hi else None


def crossing_segment(polyA, polyB, tol=1e-9):
    """The 3-D segment where two planar polygons genuinely cross.

    Returns ``(P0, P1)`` or ``None`` (parallel planes, or the
    plane-plane line misses one of the polygons).
    """
    oA, nA, uA, vA = _plane_basis(polyA)
    oB, nB, uB, vB = _plane_basis(polyB)
    d = np.cross(nA, nB)
    nd = np.linalg.norm(d)
    if nd < tol:
        return None                                # parallel/coplanar
    d = d / nd
    # a point on both planes: solve [nA; nB; d] x = [nA.oA, nB.oB, 0]
    M = np.vstack([nA, nB, d])
    rhs = np.array([nA @ oA, nB @ oB, 0.0])
    x0 = np.linalg.solve(M, rhs)
    iA = _line_clip_interval(_to_2d(polyA, oA, uA, vA),
                             _to_2d(x0[None, :], oA, uA, vA)[0],
                             np.array([d @ uA, d @ vA]))
    iB = _line_clip_interval(_to_2d(polyB, oB, uB, vB),
                             _to_2d(x0[None, :], oB, uB, vB)[0],
                             np.array([d @ uB, d @ vB]))
    if iA is None or iB is None:
        return None
    lo, hi = max(iA[0], iB[0]), min(iA[1], iB[1])
    if hi - lo < tol:
        return None
    return x0 + lo * d, x0 + hi * d


def _min_rim_distance(polyA, polyB, samples=64):
    """Cheap minimum distance between two rims (sampled)."""
    def densify(P):
        P = np.asarray(P, dtype=float)
        out = []
        for i in range(len(P)):
            A, B = P[i], P[(i + 1) % len(P)]
            for t in np.linspace(0.0, 1.0, samples // len(P) + 2,
                                 endpoint=False):
                out.append(A + t * (B - A))
        return np.array(out)
    a, b = densify(polyA), densify(polyB)
    return float(np.sqrt(
        ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2).min()))


# ---------------------------------------------------------------------------
# the 3-D preparer
# ---------------------------------------------------------------------------

def prepare_fault_surfaces(surfaces, spacing, ligament=2.0,
                           hierarchy=None, verbose=True):
    """Convert crossing planar patches to the offset form.

    Parameters mirror :func:`prepare_fault_network`: ``surfaces`` is a
    sequence of triangulated :class:`FaultSurface` (or ``(name,
    rim_polygon)`` pairs), ``spacing`` the near-fault mesh size,
    ``ligament`` the band half-width in multiples of ``spacing``,
    ``hierarchy`` the seniority order (senior runs through; the junior
    patch is cut into two pieces by a ``2 * ligament * spacing`` band
    about the intersection line, in its own plane).

    Returns ``(prepared, report, junctions)`` where ``prepared`` is a
    list of ``(name, rim_polygon)`` ready for the multi-patch embed and
    ``junctions`` carry ``dict(kind, segment=(P0, P1), pull, faults)``
    for the tube glue.
    """
    lig = float(ligament) * float(spacing)
    entries = []
    for s in surfaces:
        if isinstance(s, FaultSurface):
            entries.append([s.name, np.asarray(s.rim_polygon(),
                                               dtype=float)])
        else:
            name, poly = s
            entries.append([str(name), np.asarray(poly,
                                                  dtype=float)[:, :3]])
    names = [n for n, _ in entries]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate patch names: {names}")
    ranks = ({n: k for k, n in enumerate(hierarchy)} if hierarchy
             else {n: k for k, n in enumerate(names)})

    for name, poly in entries:
        o, nh, uh, vh = _plane_basis(poly)
        if not _is_convex(_to_2d(poly, o, uh, vh)):
            raise NotImplementedError(
                f"patch {name!r} has a non-convex rim — the v1 planar "
                "preparer clips convex rims only.")

    report, junctions = [], []
    pieces = {n: [(n, p)] for n, p in entries}   # name -> live pieces

    order = sorted(names, key=lambda n: ranks.get(n, len(ranks)))
    for a_i in range(len(order)):
        for b_i in range(a_i + 1, len(order)):
            senior, junior = order[a_i], order[b_i]
            new_junior = []
            for jname, jpoly in pieces[junior]:
                cut_any = False
                for sname, spoly in pieces[senior]:
                    seg = crossing_segment(spoly, jpoly)
                    if seg is None:
                        dmin = _min_rim_distance(spoly, jpoly)
                        if dmin < lig:
                            raise NotImplementedError(
                                f"patches {sname!r} and {jname!r} come "
                                f"within {dmin:.4g} (< ligament "
                                f"{lig:.4g}) without crossing — the v1 "
                                "preparer handles genuine X crossings "
                                "only; separate them or extend them to "
                                "cross.")
                        continue
                    P0, P1 = seg
                    o, nh, uh, vh = _plane_basis(jpoly)
                    j2 = _to_2d(jpoly, o, uh, vh)
                    q0 = _to_2d(P0[None, :], o, uh, vh)[0]
                    q1 = _to_2d(P1[None, :], o, uh, vh)[0]
                    d2 = q1 - q0
                    d2 = d2 / max(np.linalg.norm(d2), 1e-30)
                    nrm2 = np.array([-d2[1], d2[0]])
                    c0 = float(nrm2 @ q0)
                    halves = []
                    for sign in (+1.0, -1.0):
                        clipped = _clip_halfplane(
                            j2, sign * nrm2[0], sign * nrm2[1],
                            -sign * c0 - lig)
                        if clipped is not None \
                                and _poly_area(clipped) > (2 * lig) ** 2:
                            halves.append(_to_3d(clipped, o, uh, vh))
                    junctions.append(dict(
                        kind="X crossing (surfaces)",
                        segment=(P0.copy(), P1.copy()), pull=lig,
                        faults=(sname, jname)))
                    report.append(
                        f"X crossing: {sname!r} (senior) crosses "
                        f"{jname!r} along a segment of length "
                        f"{np.linalg.norm(P1 - P0):.4g}; {jname!r} cut "
                        f"into {len(halves)} piece(s), band half-width "
                        f"{lig:.4g}.")
                    if len(halves) == 0:
                        report.append(
                            f"every piece of {jname!r} fell below the "
                            "minimum area — patch dropped.")
                    new_junior.extend(halves)
                    cut_any = True
                    break            # this piece is consumed; halves
                                     # re-checked against no one (v1:
                                     # one junction per junior piece)
                if not cut_any:
                    new_junior.append(jpoly)
            k = len(new_junior)
            if k == 1:
                pieces[junior] = [(junior, new_junior[0])]
            else:
                pieces[junior] = [(f"{junior}_{i + 1}", p)
                                  for i, p in enumerate(new_junior)]

    prepared = [pc for n in names for pc in pieces[n]]
    if verbose:
        for line in report:
            print(f"[prepare_fault_surfaces] {line}")
    return prepared, report, junctions
