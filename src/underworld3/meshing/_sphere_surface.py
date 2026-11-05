"""Triangulations of spherical gmsh surfaces that are the convex hull of their nodes (#946)."""
import numpy as np


def _flip_creased_sphere_edges(centre=(0.0, 0.0, 0.0)):
    r"""Make gmsh's triangulation of every spherical surface the convex hull of its nodes.

    gmsh meshes a sphere in a parameter plane, (longitude, latitude) for an OCC sphere.
    Near a pole it can join four neighbouring nodes along the long diagonal of their
    quadrilateral instead of the short one, and a structured surface splits every
    quadrilateral the same way whatever its shape. Either way the surface creases toward
    the centre. Where the domain lies outside the surface (the inner wall of a shell,
    either side of an internal surface), the pocket under the crease is meshed as one
    cell whose four vertices are on the sphere. Refinement snaps the midpoint of the
    creased edge outward onto the sphere, past the midpoint of the cell's other
    diagonal, and folds the cell's children (#946).

    An edge is creased when the far vertex of one of its triangles lies outside the
    plane of the other, on the side away from the centre. Flipping the edge to the other
    diagonal removes the crease. Repeating until no edge is creased leaves the convex hull
    of the nodes, which for points on a sphere is their spherical Delaunay triangulation;
    the flips terminate as Lawson's do. Four nodes on one circle tie and are left alone.
    Edges of curves embedded in a surface are never flipped.

    The surfaces flipped are those whose triangle nodes are all at one distance from
    ``centre``. Call it on gmsh's current model between ``generate(2)`` and
    ``generate(3)``: the volume mesh is then built on the flipped surfaces.

    Parameters
    ----------
    centre : tuple of float
        The centre of the spheres.
    """
    import gmsh

    centre = np.asarray(centre, dtype=float)
    node_tags, xyz, _ = gmsh.model.mesh.getNodes()
    coords = np.zeros((int(node_tags.max()) + 1, 3))
    coords[node_tags] = xyz.reshape(-1, 3)

    for _, surface in gmsh.model.getEntities(2):
        _, nodes = gmsh.model.mesh.getElementsByType(2, surface)
        if nodes.size == 0:
            continue
        radius = np.linalg.norm(coords[nodes] - centre, axis=1)
        if radius.max() - radius.min() > 1.0e-9 * radius.max():
            continue
        fixed = set()
        for dim, curve in gmsh.model.mesh.getEmbedded(2, surface):
            if dim == 1:
                _, lines = gmsh.model.mesh.getElementsByType(1, curve)
                fixed |= {frozenset(line) for line in lines.reshape(-1, 2)}
        triangles = [list(t) for t in nodes.reshape(-1, 3)]
        while _flip_creased_edges_once(triangles, coords, centre, fixed):
            pass
        gmsh.model.mesh.removeElements(2, surface)
        gmsh.model.mesh.addElementsByType(surface, 2, [], np.ravel(triangles))


def _flip_creased_edges_once(triangles, coords, centre, fixed):
    """One sweep over the creased edges of a consistently oriented triangulation, flipping
    each, in place; returns the number of flips. A triangle changed in this sweep waits
    for the next one, so every flip sees its two triangles as they are."""
    edges = {}
    for i, triangle in enumerate(triangles):
        for k in range(3):
            edges.setdefault(frozenset((triangle[k], triangle[k - 2])), []).append(i)

    changed = set()
    for edge, pair in edges.items():
        if len(pair) != 2 or edge in fixed or changed.intersection(pair):
            continue
        i, j = pair
        # a -> b is the edge in triangle i's orientation, so triangle j runs b -> a -> d
        k = next(k for k in range(3) if {triangles[i][k], triangles[i][k - 2]} == edge)
        a, b, c = triangles[i][k], triangles[i][k - 2], triangles[i][k - 1]
        d = (set(triangles[j]) - edge).pop()
        normal = np.cross(coords[b] - coords[a], coords[c] - coords[a])
        if normal @ (coords[a] + coords[b] + coords[c] - 3.0 * centre) < 0.0:
            normal = -normal
        rise = normal @ (coords[d] - coords[a])
        if rise <= 1.0e-12 * np.linalg.norm(normal) * np.linalg.norm(coords[d] - coords[a]):
            continue
        triangles[i], triangles[j] = [a, d, c], [d, b, c]
        changed.update(pair)
    return len(changed) // 2
