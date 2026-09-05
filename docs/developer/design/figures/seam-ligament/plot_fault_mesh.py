import os, glob, numpy as np
import pyvista as pv
pv.OFF_SCREEN = True
SP = os.environ.get("SP", ".")
Y0, Y1, X0, X1 = 0.42, 0.78, 0.38, 0.62     # the crossing, zoomed

def panel(pl, tag, files, title):
    pl.subplot(0, tag)
    for r, f in enumerate(sorted(files)):
        d = np.load(f)
        X = np.column_stack([d["X"], np.zeros(len(d["X"]))]); cells = d["cells"]
        faces = np.column_stack([np.full(len(cells), 3), cells]).ravel()
        grid = pv.PolyData(X, faces)
        # 0 base/fill, 1 embedded band, 2 ligament; +3 on rank 1
        grid.cell_data["cat"] = d["cat"] + 3 * r
        pl.add_mesh(grid, scalars="cat", show_edges=True, edge_color="#404040", line_width=0.8,
                    cmap=["#e8e8e8", "#f2c744", "#e8613c", "#bdbdbd", "#d9a520", "#c8321a"],
                    clim=(0, 5), show_scalar_bar=False)
        weak = np.flatnonzero(d["eta"] < 0.5)
        if len(weak):
            wf = np.column_stack([np.full(len(weak), 3), cells[weak]]).ravel()
            pl.add_mesh(pv.PolyData(X, wf), style="wireframe", color="#7b1fa2", line_width=4)
        for a, b in d["plus"]:
            pl.add_lines(np.array([X[a], X[b]]), color="#1565c0", width=6)
        if d["shared"].any():
            pl.add_points(X[d["shared"]], color="black", point_size=10, render_points_as_spheres=True)
        if len(d["tips"]):
            pl.add_points(X[d["tips"]], color="#00c853", point_size=22, render_points_as_spheres=True)
    pl.add_text(title, font_size=12, position="upper_left")
    pl.view_xy()
    pl.camera.focal_point = (0.5 * (X0 + X1), 0.5 * (Y0 + Y1), 0.0)
    pl.camera.position = (0.5 * (X0 + X1), 0.5 * (Y0 + Y1), 1.0)
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = 0.5 * (Y1 - Y0)

pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(2200, 1500), border=True)
panel(pl, 0, glob.glob(f"{SP}/faultmesh_gather_np1_rank*.npz"),
      "serial (and the gather): one band, one cut, tips only at the fault's ends")
panel(pl, 1, glob.glob(f"{SP}/faultmesh_ligament_np2_rank*.npz"),
      "np=2 seam ligament: cut stops at a tip on each rank; ligament cells carry eta_1")
pl.subplot(0, 1)
legend = [["embedded band (rank 0 / rank 1)", "#f2c744"], ["ligament base cells", "#e8613c"],
          ["cells painted eta_1 (TI)", "#7b1fa2"], ["the cut (Plus facets)", "#1565c0"],
          ["sub-chain tip (pinned)", "#00c853"], ["shared (seam) vertices", "black"]]
pl.add_legend(legend, bcolor="white", size=(0.34, 0.2), loc="lower right", face="rectangle")
out = os.path.join(SP, "fault_mesh_serial_vs_ligament.png")
pl.screenshot(out); print(out)
