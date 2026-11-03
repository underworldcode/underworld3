"""Distributed cell regions separated by a tagged fitted shell interface.

The labels follow mesh connectivity, not a smooth-surface radius test. This
matters for chordal interfaces: an outer tetrahedron can have every vertex
on the true sphere and its centroid strictly inside that sphere.
"""

from collections import defaultdict

import numpy as np
from mpi4py import MPI


def _label_fitted_shell_regions(
    dm, inner_boundary: tuple[str, int], outer_boundary: tuple[str, int],
    interface: tuple[str, int], inner_region: tuple[str, int],
    outer_region: tuple[str, int],
) -> dict[str, int]:
    """Flood cell components from the walls without crossing interface facets.

    Only partition-facet keys and, on overlapped meshes, duplicated cell keys
    are exchanged. Rendezvous owners communicate component colours, never a
    gathered mesh. Both owned and ghost cells receive the same final labels.
    Conflicting wall seeds, unseeded components and a nonseparating interface
    are collective errors rather than geometric fallbacks.
    """
    comm = dm.comm.tompi4py()
    cstart, cend = dm.getHeightStratum(0)
    fstart, fend = dm.getHeightStratum(1)
    vstart, vend = dm.getDepthStratum(0)
    cell_numbering = np.asarray(dm.getCellNumbering().getIndices(), dtype=np.int64)
    vertex_numbering = np.asarray(dm.getVertexNumbering().getIndices(), dtype=np.int64)
    owned = cell_numbering >= 0
    cell_ids = np.where(owned, cell_numbering, -cell_numbering - 1)
    vertex_ids = np.where(vertex_numbering >= 0, vertex_numbering, -vertex_numbering - 1)
    parent = np.arange(cend - cstart)

    def root(cell: int) -> int:
        while parent[cell] != cell:
            parent[cell] = parent[parent[cell]]
            cell = int(parent[cell])
        return cell

    facet_cells = {}
    barriers = set()
    messages = [[] for _ in range(comm.size)]
    missing = [name for name, _ in (inner_boundary, outer_boundary, interface)
               if not dm.hasLabel(name)]
    if comm.allreduce(bool(missing), op=MPI.LOR):
        raise ValueError("Missing a fitted-shell wall or interface label on at least one rank.")
    interface_label = dm.getLabel(interface[0])
    external = dm.getLabel("All_Boundaries") if dm.hasLabel("All_Boundaries") else None
    for face in range(fstart, fend):
        cells = np.asarray(dm.getSupport(face), dtype=int) - cstart
        cells = cells[(cells >= 0) & (cells < len(parent))]
        facet_cells[face] = cells
        if interface_label.getValue(face) == interface[1]:
            barriers.add(face)
        elif len(cells) == 2:
            parent[root(int(cells[1]))] = root(int(cells[0]))
    roots = np.array([root(cell) for cell in range(len(parent))], dtype=int)
    unique_roots, component = np.unique(roots, return_inverse=True)
    colours = np.zeros(len(unique_roots), dtype=np.int8)
    for boundary, colour in ((inner_boundary, 1), (outer_boundary, 2)):
        label = dm.getLabel(boundary[0])
        for face, cells in facet_cells.items():
            if label.getValue(face) == boundary[1]:
                for cell in cells:
                    colours[component[cell]] |= colour
    if comm.allreduce(bool(np.any(colours == 3)), op=MPI.LOR):
        raise ValueError("Fitted shell wall seeds conflict without crossing its interface.")

    # A key is routed to a deterministic owner; a shared face has records from
    # both partitions. Cell keys additionally connect duplicated overlap cells.
    records = []
    for face, cells in facet_cells.items():
        is_external = external is not None and external.getValue(face) >= 0
        if face not in barriers and (len(cells) != 1 or is_external):
            continue
        vertices = [int(v) for v in dm.getTransitiveClosure(face)[0] if vstart <= v < vend]
        key = (0, *sorted(int(vertex_ids[v - vstart]) for v in vertices))
        for cell in cells:
            records.append((key, int(component[cell]), int(cell_ids[cell]), face in barriers))
    has_ghosts = comm.allreduce(bool(np.any(~owned)), op=MPI.LOR)
    if has_ghosts:
        records.extend(((1, int(gid)), int(comp), int(gid), False)
                       for gid, comp in zip(cell_ids, component, strict=True))
    for index, (key, _, _, _) in enumerate(records):
        destination = sum((i + 1) * value for i, value in enumerate(key)) % comm.size
        messages[destination].append((comm.rank, index, key))
    groups = defaultdict(list)
    for batch in comm.alltoall(messages):
        for rank, index, key in batch:
            groups[key].append((rank, index))

    while True:
        updates = [[] for _ in range(comm.size)]
        for index, (_, comp, gid, barrier) in enumerate(records):
            key = records[index][0]
            destination = sum((i + 1) * value for i, value in enumerate(key)) % comm.size
            updates[destination].append((comm.rank, index, int(colours[comp]), gid, barrier))
        values = {}
        for batch in comm.alltoall(updates):
            for rank, index, colour, gid, barrier in batch:
                values[rank, index] = (colour, gid, barrier)
        replies = [[] for _ in range(comm.size)]
        invalid = False
        for peers in groups.values():
            data = [values[peer] for peer in peers]
            if data[0][2]:
                # Internal faces are barriers: check them only after convergence.
                continue
            colour = 0
            for value, _, _ in data:
                colour |= value
            invalid |= colour == 3
            for rank, index in peers:
                replies[rank].append((index, colour))
        if comm.allreduce(invalid, op=MPI.LOR):
            raise ValueError("Fitted shell wall seeds conflict across partition boundaries.")
        changed = False
        for batch in comm.alltoall(replies):
            for index, colour in batch:
                comp = records[index][1]
                merged = int(colours[comp]) | colour
                changed |= merged != colours[comp]
                colours[comp] = merged
        if not comm.allreduce(changed, op=MPI.LOR):
            break
    if comm.allreduce(bool(np.any(colours == 0)), op=MPI.LOR):
        raise ValueError("Fitted shell contains a component connected to neither wall.")
    invalid = False
    for peers in groups.values():
        data = [values[peer] for peer in peers]
        if data[0][2]:
            sides = {gid: colour for colour, gid, _ in data}
            invalid |= len(sides) != 2 or set(sides.values()) != {1, 2}
    if comm.allreduce(invalid, op=MPI.LOR):
        raise ValueError("A fitted internal facet does not separate Inner and Outer cells.")

    for name, value in (inner_region, outer_region):
        if dm.hasLabel(name):
            dm.removeLabel(name)
        dm.createLabel(name)
        label = dm.getLabel(name)
        colour = 1 if (name, value) == inner_region else 2
        for cell in np.flatnonzero(colours[component] == colour):
            label.setValue(int(cell) + cstart, value)
    counts = [comm.allreduce(int(np.count_nonzero(owned & (colours[component] == colour))), op=MPI.SUM)
              for colour in (1, 2)]
    if min(counts) == 0:
        raise ValueError("Both fitted shell regions require at least one owned cell.")
    return dict(inner_cells=counts[0], outer_cells=counts[1])
