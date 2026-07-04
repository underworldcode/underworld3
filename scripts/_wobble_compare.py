"""Side-by-side a16e (single-knob equidistribution) vs a16c2
(legacy hand-tuned cc=2) at MATCHED step indices, one image:
rows = transient / developed / settled, cols = a16e | a16c2.
Same proven render as aniso_movie.py (P3 T, RdBu_r, white bg,
lighting off, deformed-mesh edges). minA/meanA annotated per
panel so the 'wobble' (transient over-reaction + grading pulse)
is directly comparable. Writes one PNG for Preview."""
import numpy as np
import underworld3 as uw
import underworld3.visualisation as vis
from underworld3.meshing.smoothing import _tri_cells, _signed_areas
import pyvista as pv

D = "/tmp/metric_mesh/sat"
STEPS = [20, 120, 300]                    # transient / dev / settled
TAGS = [("a16e", "equidist  R=2  (1 knob, parameter-free)"),
        ("a16c2", "legacy  cc=2  (2 hand-tuned knobs)")]
OUT = f"{D}/wobble_a16e_vs_a16c2.png"

pv.OFF_SCREEN = True
nrow, ncol = len(STEPS), len(TAGS)
pl = pv.Plotter(shape=(nrow, ncol), off_screen=True,
                window_size=(1000 * ncol, 1000 * nrow),
                border=True, border_color="#888888")
pl.set_background("white")
for r, step in enumerate(STEPS):
    for c, (tag, lab) in enumerate(TAGS):
        m = uw.discretisation.Mesh(
            f"{D}/sat_{tag}.mesh.{step:05}.h5")
        Tv = uw.discretisation.MeshVariable(
            "T", m, vtype=uw.VarType.SCALAR, degree=3,
            continuous=True)
        Tv.read_timestep(f"sat_{tag}", "T", step, outputPath=D)
        pv_T = vis.meshVariable_to_pv_mesh_object(Tv)
        pv_T.point_data["T"] = np.asarray(Tv.data[:, 0])
        edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
        Xc = np.asarray(m.X.coords)
        A = np.abs(_signed_areas(Xc, _tri_cells(m.dm)))
        q = A.min() / A.mean()
        phase = ("transient (Nu~overshoot)" if step <= 30
                 else "developed" if step <= 200 else "settled")
        pl.subplot(r, c)
        pl.add_text(f"{lab}\nstep {step}  [{phase}]   "
                    f"minA/meanA={q:.3f}",
                    font_size=13, color="black")
        pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
                    clim=(0.0, 1.0), show_edges=False,
                    lighting=False,
                    show_scalar_bar=(r == nrow - 1 and c == ncol - 1),
                    scalar_bar_args=dict(title="T", color="black"))
        pl.add_mesh(edges, color="#202020", line_width=0.7,
                    lighting=False)
        pl.view_xy()
        pl.camera.zoom(1.35)
        print(f"  {tag} step {step}: minA/meanA={q:.3f}",
              flush=True)
pl.screenshot(OUT)
pl.close()
print(f"saved {OUT}")
