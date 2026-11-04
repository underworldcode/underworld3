"""Distributed evaluation of the original spherical-wall multiplier space."""

from itertools import product

import numpy as np


def _wall_cone_index(facets):
    """Index conservative spherical-cone boxes, retaining exact facet tests.

    A normalized positive combination of facet vertices stays in their
    spherical cap. Its chord distance from the cap centre therefore bounds
    each Cartesian coordinate. Bins select candidates; they never determine
    ownership or permit a nearest-facet fallback.
    """
    directions = facets/np.linalg.norm(facets, axis=2)[..., None]
    centres = directions.sum(axis=1)
    centres /= np.linalg.norm(centres, axis=1)[:, None]
    radii = np.linalg.norm(directions-centres[:, None], axis=2).max(axis=1)
    if np.any(np.einsum("fi,fvi->fv", centres, directions) <= 0):
        raise ValueError("Inner-wall facets must lie in an open spherical hemisphere.")
    lower, upper = centres-radii[:, None], centres+radii[:, None]
    pitch = 2*float(radii.max())
    bins = {}
    for facet, (a, b) in enumerate(zip(lower, upper, strict=True)):
        first, last = np.floor(a/pitch).astype(int), np.floor(b/pitch).astype(int)
        for key in product(*(range(first[d], last[d]+1) for d in range(3))):
            bins.setdefault(key, []).append(facet)
    return lower, upper, pitch, bins


def _inner_wall_cell_facets(mesh, wall, cell_vertices):
    """Return conservative original-wall facet candidates for each cut cell.

    Each returned pair contains facet coordinates ``(K,3,3)`` and their
    original multiplier IDs ``(K,6)`` or ``(K,18)``. Candidates are sorted by
    their global multiplier-ID facet key and retain their original vertex
    ordering. The caller clips angular integration domains to these cones;
    this routine only rejects disjoint conservative directional boxes.

    All ranks participate, including empty cell and facet ranks. Only six
    directional bounding-box numbers per rank are replicated; cell queries
    and candidate facets use bounded sparse all-to-all exchanges.
    """
    comm = mesh.dm.comm.tompi4py()
    facets = np.asarray(wall["facets"], dtype=float)
    lids = np.asarray(wall["lid"])
    cells = np.asarray(cell_vertices, dtype=float)
    radius = float(wall["radius"])
    error = None
    if (facets.ndim != 3 or facets.shape[1:] != (3, 3)
            or lids.ndim != 2 or len(lids) != len(facets)
            or lids.shape[1] not in (6, 18)
            or cells.ndim != 3 or cells.shape[1:] != (4, 3)):
        error = "Cell/facet routing requires tetrahedra, triangular facets and six or eighteen IDs."
    elif (not np.isfinite(cells).all() or not np.isfinite(facets).all()
            or not np.isfinite(radius) or radius <= 0
            or np.any(np.linalg.norm(cells, axis=2) == 0)):
        error = "Cell/facet routing requires finite nonzero coordinates and a positive radius."
    elif not np.allclose(np.linalg.norm(facets, axis=2), radius, rtol=1e-10, atol=0):
        error = "Original inner-wall facet vertices must lie on the prescribed sphere."
    errors = comm.allgather(error)
    if any(message is not None for message in errors):
        raise ValueError(f"Invalid distributed inner-wall cell geometry: {errors}")
    widths = comm.allgather(lids.shape[1])
    if min(widths) != max(widths):
        raise ValueError("Inner-wall multiplier component counts differ between ranks.")
    error = None
    boxes = []
    for vertices in (facets, cells):
        directions = vertices/np.linalg.norm(vertices, axis=2)[..., None]
        centres = directions.sum(axis=1)
        lengths = np.linalg.norm(centres, axis=1)
        if np.any(lengths == 0):
            error = "Cell and facet cones must lie in an open spherical hemisphere."
            boxes.append(None)
            continue
        centres /= lengths[:, None]
        if np.any(np.einsum("fi,fvi->fv", centres, directions) <= 0):
            error = "Cell and facet cones must lie in an open spherical hemisphere."
        radii = np.linalg.norm(directions-centres[:, None], axis=2).max(axis=1)
        boxes.append((centres-radii[:, None], centres+radii[:, None]))
    errors = comm.allgather(error)
    if any(message is not None for message in errors):
        raise ValueError(f"Invalid distributed inner-wall cell cones: {errors}")
    facet_lower, facet_upper = boxes[0]
    cell_lower, cell_upper = boxes[1]
    rank_box = np.array([[np.inf]*3, [-np.inf]*3])
    index = None
    if len(facets):
        if np.any(np.abs(np.linalg.det(np.swapaxes(facets/radius, 1, 2))) <= 1e-14):
            error = "Original inner-wall facets must have nondegenerate radial cones."
        else:
            index = _wall_cone_index(facets)
            rank_box = np.stack((facet_lower.min(axis=0), facet_upper.max(axis=0)))
    errors = comm.allgather(error)
    if any(message is not None for message in errors):
        raise ValueError(f"Invalid distributed original inner-wall cones: {errors}")
    rank_boxes = comm.allgather(rank_box)
    keys = [tuple(sorted(map(int, row))) for row in lids]
    received = [dict() for _ in cells]
    batch_size = 2048
    batches = max(comm.allgather((len(cells)+batch_size-1)//batch_size))
    tolerance = 1e-11
    for batch in range(batches):
        first = batch*batch_size
        lower = cell_lower[first:first+batch_size]
        upper = cell_upper[first:first+batch_size]
        queries = []
        for rank_lower, rank_upper in rank_boxes:
            selected = np.flatnonzero(np.all((lower <= rank_upper+tolerance)
                                              & (upper >= rank_lower-tolerance), axis=1))
            queries.append((first+selected, lower[selected], upper[selected]))
        proposals = []
        for indices, lower, upper in comm.alltoall(queries):
            records = []
            if index is not None:
                _, _, pitch, bins = index
                for cell, a, b in zip(indices, lower, upper, strict=True):
                    first_bin = np.floor((a-tolerance)/pitch).astype(int)
                    last_bin = np.floor((b+tolerance)/pitch).astype(int)
                    candidates = set()
                    count = int(np.prod(last_bin-first_bin+1, dtype=np.int64))
                    if count <= len(bins):
                        for key in product(*(range(first_bin[d], last_bin[d]+1) for d in range(3))):
                            candidates.update(bins.get(key, ()))
                    else:
                        for key, values in bins.items():
                            if all(first_bin[d] <= key[d] <= last_bin[d] for d in range(3)):
                                candidates.update(values)
                    candidates = np.asarray(sorted(candidates, key=lambda i: keys[i]), dtype=int)
                    overlap = np.all((facet_lower[candidates] <= b+tolerance)
                                     & (facet_upper[candidates] >= a-tolerance), axis=1)
                    candidates = candidates[overlap]
                    records.append((int(cell), facets[candidates], lids[candidates]))
            proposals.append(records)
        for rank, records in enumerate(comm.alltoall(proposals)):
            for cell, candidate_facets, candidate_ids in records:
                for triangle, ids in zip(candidate_facets, candidate_ids, strict=True):
                    key = tuple(sorted(map(int, ids)))
                    if key not in received[cell] or rank < received[cell][key][0]:
                        received[cell][key] = (rank, triangle, ids)
    result = []
    for records in received:
        ordered = [records[key] for key in sorted(records)]
        triangles = np.array([item[1] for item in ordered], dtype=float).reshape(-1, 3, 3)
        ids = np.array([item[2] for item in ordered], dtype=lids.dtype).reshape(-1, lids.shape[1])
        result.append((triangles, ids))
    return result
