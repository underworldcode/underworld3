"""The multigrid hierarchy of the layout fixture, seen: each level of the
tail on the z = 0.5 slice (through all three faults), a zoom on one band
with its fill, and at np=3 the partition on the same slice.

    python -u render_hierarchy.py                 # levels + zoom (serial)
    mpirun -np 3 python -u render_hierarchy.py    # partition (rank 0 renders)
"""
import numpy as np
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv
from mpi4py import MPI
pv.OFF_SCREEN = True

FIG = "/Users/lmoresi/+Simulations/fault_network_3d_parallel/figures"
H, W = 0.08, 0.04
def patch(x0, x1, y, z0=0.4, z1=0.6):
    return np.array([[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]])
A = patch(0.40, 0.70, 0.50); B = patch(2.20, 2.60, 0.50)
C = np.array([[2.20, 0.62, 0.42], [2.52, 0.30, 0.42], [2.52, 0.30, 0.58], [2.20, 0.62, 0.58]])
D = patch(5.20, 5.50, 0.50)
faults = []
for name, P in (("A", A), ("B", B), ("C", C), ("D", D)):
    f = uw.meshing.FaultSurface(name, P); f.triangulate(); faults.append(f)
net = uw.meshing.FaultNetwork(faults, hierarchy=["A", "B", "C", "D"])
net.prepare(h=H, ligament=1.5, verbose=False)
net.realisation, net.width = "split", W
net._build_3d_band(h_far=0.24, realisation="split", margin_rings=0.5, carve_clearance=0.3,
                   minCoords=(0.0, 0.0, 0.0), maxCoords=(6.0, 1.0, 1.0))
mesh = net.mesh
comm = uw.mpi.comm
levels = list(getattr(mesh, "_custom_mg_coarse_meshes", []) or []) + [mesh]

def slice_edges(m, z=0.5):
    grid = vis.mesh_to_pv_mesh(m)
    return grid, grid.slice(normal="z", origin=(0, 0, z))

if comm.size == 1:
    for k, lv in enumerate(levels):
        n_cells = int(lv.dm.getHeightStratum(0)[1])
        n_v = int(np.diff(lv.dm.getDepthStratum(0))[0])
        print(f"[hier] level {k}: {n_cells} cells, {n_v} vertices", flush=True)
    # each level on the slice, same camera
    for k, lv in enumerate(levels):
        grid, sl = slice_edges(lv)
        pl = pv.Plotter(off_screen=True, window_size=(2400, 500))
        pl.set_background("white")
        if k == len(levels) - 1:
            band = np.asarray(mesh.cells_labelled("Band", 71)).astype(float)
            grid.cell_data["band"] = band
            sl = grid.slice(normal="z", origin=(0, 0, 0.5))
            pl.add_mesh(sl, scalars="band", cmap="RdBu_r", clim=(0, 1), show_edges=True,
                        edge_color="black", line_width=0.6, lighting=False, show_scalar_bar=False)
        else:
            pl.add_mesh(sl, color="white", show_edges=True, edge_color="black", line_width=0.6, lighting=False)
        pl.view_xy(); pl.camera.parallel_projection = True
        pl.camera.focal_point = (3.0, 0.5, 0.5); pl.camera.parallel_scale = 0.55
        out = f"{FIG}/hierarchy_level{k}_z05.png"; pl.screenshot(out); pl.close(); print("[hier] wrote", out, flush=True)
    # zoom on fault B's band, fine level: band cells red, fill/base white
    grid, _ = slice_edges(mesh)
    grid.cell_data["band"] = np.asarray(mesh.cells_labelled("Band", 71)).astype(float)
    for tag, origin, normal, focal, scale, size in (
            ("z05", (0, 0, 0.5), "z", (2.4, 0.5, 0.5), 0.45, (1200, 1000)),
            ("x24", (2.4, 0, 0), "x", (2.4, 0.5, 0.5), 0.45, (1000, 1000))):
        sl = grid.slice(normal=normal, origin=origin)
        pl = pv.Plotter(off_screen=True, window_size=size); pl.set_background("white")
        pl.add_mesh(sl, scalars="band", cmap="RdBu_r", clim=(0, 1), show_edges=True,
                    edge_color="black", line_width=0.8, lighting=False, show_scalar_bar=False)
        # the coarse level's edges on top, in blue, to show the nesting
        _g0, sl0 = slice_edges(levels[0], z=0.5) if normal == "z" else (None, vis.mesh_to_pv_mesh(levels[0]).slice(normal="x", origin=origin))
        pl.add_mesh(sl0, color="white", opacity=0.0, show_edges=True, edge_color="blue", line_width=1.5, lighting=False)
        if normal == "z": pl.view_xy()
        else: pl.view_yz()
        pl.camera.parallel_projection = True; pl.camera.focal_point = focal; pl.camera.parallel_scale = scale
        out = f"{FIG}/hierarchy_zoomB_{tag}.png"; pl.screenshot(out); pl.close(); print("[hier] wrote", out, flush=True)
else:
    # partition: gather every rank's cells (P1 vertices) to rank 0 and colour by rank
    from underworld3.utilities.line_cut import _coords
    dm = mesh.dm; vS, vE = dm.getDepthStratum(0); X = _coords(dm)[: vE - vS]
    cn = np.asarray(mesh._cell_node_indices(1, True))
    T = X[cn]                                          # (n_cells, 4, 3)
    band = np.asarray(mesh.cells_labelled("Band", 71)).astype(float)
    allT = comm.gather(T, root=0); allB = comm.gather(band, root=0)
    ranks = comm.gather(np.full(len(T), comm.rank, dtype=float), root=0)
    if comm.rank == 0:
        pts = np.vstack([t.reshape(-1, 3) for t in allT])
        n = pts.shape[0] // 4
        cells = np.hstack([np.full((n, 1), 4), np.arange(4 * n).reshape(n, 4)]).ravel()
        grid = pv.UnstructuredGrid(cells, np.full(n, pv.CellType.TETRA), pts)
        grid.cell_data["rank"] = np.concatenate(ranks); grid.cell_data["band"] = np.concatenate(allB)
        sl = grid.slice(normal="z", origin=(0, 0, 0.5))
        pl = pv.Plotter(off_screen=True, window_size=(2400, 500)); pl.set_background("white")
        pl.add_mesh(sl, scalars="rank", cmap="RdBu_r", clim=(0, comm.size - 1), show_edges=True,
                    edge_color="black", line_width=0.4, lighting=False, show_scalar_bar=False)
        bsl = grid.threshold(0.5, scalars="band").slice(normal="z", origin=(0, 0, 0.5))
        if bsl.n_points:
            pl.add_mesh(bsl, color="black", show_edges=True, edge_color="black", line_width=1.0, lighting=False)
        pl.view_xy(); pl.camera.parallel_projection = True
        pl.camera.focal_point = (3.0, 0.5, 0.5); pl.camera.parallel_scale = 0.55
        out = f"{FIG}/partition_np{comm.size}_z05.png"; pl.screenshot(out); pl.close(); print("[hier] wrote", out, flush=True)
        print(f"[hier] np={comm.size} cells/rank {[len(t) for t in allT]} band/rank {[int(b.sum()) for b in allB]}", flush=True)
