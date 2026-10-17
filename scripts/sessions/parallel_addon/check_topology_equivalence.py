"""Every function converted to the compiled topology tables against its previous
(per-point Python) version, taken from the commit before the conversion: the
results must be identical. Serial and parallel.

    git show dd1d6eea:<file> > /tmp/parallel-addon/old/<name>_old.py   (see README)
    mpirun -np N python check_topology_equivalence.py
"""
import ast, importlib.util, sys, types
import numpy as np
import underworld3 as uw
from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD
OLD = "/tmp/parallel-addon/old/"


def load(name):
    spec = importlib.util.spec_from_file_location(name, OLD + name + ".py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def method(file, cls_name, meth):
    """An old Mesh method as a plain function of ``self``."""
    tree = ast.parse(open(OLD + file + ".py").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == meth:
            src = ast.get_source_segment(open(OLD + file + ".py").read(), node)
            import textwrap
            ns = {"numpy": np, "uw": uw, "PETSc": PETSc}
            exec(textwrap.dedent(src), ns)
            return ns[meth]
    raise KeyError(meth)


es_old, ps_old, ap_old, fs_old = (load("edge_split_old"), load("place_surface_old"),
                                  load("addon_partition_old"), load("fault_split_old"))
from underworld3.utilities import addon_partition as ap, edge_split as es, place_surface as ps, fault_split as fs

fails = []


def same(tag, a, b):
    ok = (np.array_equal(np.asarray(a), np.asarray(b)) if not isinstance(a, (set, dict, tuple))
          else a == b)
    ok = comm.allreduce(bool(ok), op=MPI.LAND)
    if comm.rank == 0:
        print(f"  {'ok  ' if ok else 'DIFF'} {tag}", flush=True)
    if not ok:
        fails.append(tag)


meshes = {
    "2d": uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.03, qdegree=2),
    "3d": uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=0.12, regular=False, qdegree=2),
}
cni_old = method("discretisation_mesh_old", "Mesh", "_cell_node_indices")
rows_old = method("discretisation_mesh_old", "Mesh", "_coord_rows_for_points")
kd_old = method("discretisation_mesh_old", "Mesh", "_build_kd_tree_index")
for tag, mesh in meshes.items():
    if comm.rank == 0:
        print(f"{tag} np {comm.size}", flush=True)
    dm = mesh.dm
    for deg in (1, 2):
        mesh._cell_node_array.clear()
        new = mesh._cell_node_indices(deg, True)
        saved = mesh._cell_node_array
        mesh._cell_node_array = {}
        from math import comb
        ns = cni_old.__globals__; ns["comb"] = comb
        old = cni_old(mesh, deg, True)
        mesh._cell_node_array = saved
        same(f"_cell_node_indices degree {deg}", new, old)
    # the point-location index
    proxy = types.SimpleNamespace(_nav_dm=mesh._nav_dm, dm=mesh.dm, _nav_coords=mesh._nav_coords,
                                  element=mesh.element, dim=mesh.dim, cdim=mesh.cdim, _index=None)
    proxy._coord_rows_for_points = lambda nav, pts: rows_old(proxy, nav, pts)
    kd_old(proxy)
    mesh._index = None
    mesh._build_kd_tree_index()
    same("kd index control points", mesh._indexCoords, proxy._indexCoords)
    same("kd index cell map", mesh._indexMap, proxy._indexMap)
    same("kd index centroids", mesh._nav_centroids, proxy._nav_centroids)
    same("kd index reach", mesh._local_cell_reach, proxy._local_cell_reach)
    from underworld3.utilities.fault_mg import _cell_vertex_table
    vxy, cv = _cell_vertex_table(mesh)
    vxy_o = np.asarray(mesh._get_coords_for_basis(1, True))
    cv_o = np.asarray(mesh._cell_node_indices(1, True))
    same("_cell_vertex_table: each cell's vertex coordinates", vxy[cv], vxy_o[cv_o])
    same("_edge_lengths", es._edge_lengths(dm), es_old._edge_lengths(dm))
    same("_cell_edges", np.array(es._cell_edges(dm)), np.array(es_old._cell_edges(dm)))
    same("_edge_strength", np.column_stack(es._edge_strength(dm)), np.column_stack(es_old._edge_strength(dm)))
    if tag == "3d":
        T = ps._tet_vertices(dm)
        same("_tet_vertices", T, ps_old._tet_vertices(dm))
        hv_n, hc_n = ps._vertex_h_3d(dm, T, dm.getDepthStratum(0)[1] - dm.getDepthStratum(0)[0])
        hv_o, hc_o = ps_old._vertex_h_3d(dm, T, dm.getDepthStratum(0)[1] - dm.getDepthStratum(0)[0])
        same("_vertex_h_3d", hv_n, hv_o)
        nv = dm.getDepthStratum(0)[1] - dm.getDepthStratum(0)[0]
        same("_true_wall_vertex_mask", ps._true_wall_vertex_mask(dm, nv), ps_old._true_wall_vertex_mask(dm, nv))
    # region growth: two regions near each other, one far
    pStart, pEnd = dm.getChart(); vS, vE = dm.getDepthStratum(0)
    X = dm.getCoordinatesLocal().array.reshape(-1, mesh.cdim)[: vE - vS]
    ids = np.zeros(pEnd - pStart, dtype=np.int32)
    ids[vS - pStart + np.flatnonzero(np.linalg.norm(X - 0.3, axis=1) < 0.08)] = 1
    ids[vS - pStart + np.flatnonzero(np.linalg.norm(X - 0.42, axis=1) < 0.06)] = 2
    ids[vS - pStart + np.flatnonzero(np.linalg.norm(X - 0.8, axis=1) < 0.08)] = 3
    for layers in (0, 1, 2):
        rn = ps._region_cells(dm, ids, layers=layers)
        ro = ps_old._region_cells(dm, ids, layers=layers)
        same(f"_region_cells layers {layers}: cells", rn[0], ro[0])
        same(f"_region_cells layers {layers}: merge", tuple(sorted(rn[1].items())), tuple(sorted(ro[1].items())))
    # marks
    poly = np.array([[0.3, 0.45], [0.7, 0.55]]) if tag == "2d" else None
    if poly is not None:
        same("mark_polylines", ap.mark_polylines([poly])(dm), ap_old.mark_polylines([poly])(dm))
    labels = [(b.name, int(b.value)) for b in mesh.boundaries if dm.hasLabel(b.name)][:3]
    same("mark_labels", ap.mark_labels(labels)(dm), ap_old.mark_labels(labels)(dm))
    same("_fault_labels_touch_seam", fs._fault_labels_touch_seam(dm, labels),
         fs_old._fault_labels_touch_seam(dm, labels))
    # unit adjacency, as sets of pairs (order is not part of the contract)
    cS, cE = dm.getHeightStratum(0)
    unit = np.arange(cE - cS) // 3
    fo = ap._face_other_rank(dm)
    i_n, s_n = ap._unit_adjacency(dm, unit, fo)
    i_o, s_o = ap_old._unit_adjacency(dm, unit, fo)
    same("_unit_adjacency inner", set(map(tuple, i_n.tolist())), set(map(tuple, i_o.tolist())))
    same("_unit_adjacency seam", set(map(tuple, s_n.tolist())), set(map(tuple, s_o.tolist())))
    # the redistribution's per-cell fields
    if comm.size > 1:
        assign = (np.arange(cE - cS) + comm.rank) % comm.size
        fields = [np.arange(cE - cS) + 1000 * comm.rank, np.full(cE - cS, comm.rank)]
        _w, _p, _d, f_n = fs.redistribute_by_assignment(dm, assign.astype(np.int32), fields=fields)
        _w, _p, _d, f_o = fs_old.redistribute_by_assignment(dm, assign.astype(np.int32), fields=fields)
        same("redistribute_by_assignment fields", np.column_stack(f_n), np.column_stack(f_o))
if comm.rank == 0:
    print("FAILURES:", fails if fails else "none", flush=True)
