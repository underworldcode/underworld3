import os, numpy as np
import pyvista as pv
pv.OFF_SCREEN = True
SP = os.environ.get("SP", ".")
pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(2400, 1200))
for k, mode in enumerate(("gather", "ligament")):
    pl.subplot(0, k)
    for r in range(2):
        d = np.load(os.path.join(SP, f"cross_{mode}_rank{r}.npz"))
        X = np.column_stack([d["X"], np.zeros(len(d["X"]))])
        cells = d["cells"]
        faces = np.column_stack([np.full(len(cells), 3), cells]).ravel()
        grid = pv.PolyData(X, faces)
        grid.cell_data["cat"] = d["cat"] + 3 * r
        pl.add_mesh(grid, scalars="cat", show_edges=True, cmap=["#dddddd", "gold", "tomato", "#aaaaaa", "orange", "red"], clim=(0, 5), show_scalar_bar=False, line_width=0.5)
        if d["shared"].any():
            pl.add_points(X[d["shared"]], color="black", point_size=7)
        for a, b in d["edges"]:
            pl.add_lines(np.array([X[a], X[b]]), color=["blue", "green"][r], width=4)
    pl.add_text(f"seams='{mode}'", font_size=14)
    pl.view_xy(); pl.camera.zoom(1.5)
pl.screenshot(os.path.join(SP, "cross_modes.png"))
print("saved")
