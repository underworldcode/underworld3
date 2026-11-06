import os, sys, numpy as np
import pyvista as pv
pv.OFF_SCREEN = True
SP = os.environ.get("SP", ".")
pl = pv.Plotter(off_screen=True, window_size=(1600, 1600))
colors = {0: "lightgrey", 1: "gold", 2: "tomato"}
edge_colors = ["blue", "green"]
for r in range(2):
    d = np.load(os.path.join(SP, f"np2_rank{r}.npz"))
    X = np.column_stack([d["X"], np.zeros(len(d["X"]))])
    cells = d["cells"]
    faces = np.column_stack([np.full(len(cells), 3), cells]).ravel()
    grid = pv.PolyData(X, faces)
    grid.cell_data["cat"] = d["cat"] + 3 * r
    pl.add_mesh(grid, scalars="cat", show_edges=True, cmap=["#dddddd", "gold", "tomato", "#aaaaaa", "orange", "red"], clim=(0, 5), show_scalar_bar=False, line_width=0.5)
    sh = d["shared"]
    if sh.any():
        pl.add_points(X[sh], color="black", point_size=6)
    for a, b in d["edges"]:
        pl.add_lines(np.array([X[a], X[b]]), color=edge_colors[r], width=4)
pl.view_xy()
pl.camera.zoom(1.4)
pl.screenshot(os.path.join(SP, "np2_partition.png"))
print("saved")
