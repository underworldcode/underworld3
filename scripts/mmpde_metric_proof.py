"""PROOF that the mmpde 'holes' are caused by metric_density_from_gradient's
over-aggressive R-scaling (density ratio = R^3), NOT the mover.

Three demonstrations, all on the SAME annulus + SAME analytic T (re-evaluated in
space each step so the metric is the only moving part):

  (A) metric density ratio vs R   -> shows ratio = R^3 (R=5 => 125, edge ratio 11x)
  (B) mover on a FIXED ANALYTIC metric -> converges dead-flat (mover is fine)
  (C) convection feedback loop, refinement=5 vs refinement=1.5
        -> R=5 area-ratio runs away (holes); R=1.5 stays flat (clean)
      + saves a mesh render of each final state to inspect.

Outputs -> ~/+Simulations/StagnantLid/mmpde_proof/
"""
import os, numpy as np, sympy
import underworld3 as uw, underworld3.visualisation as vis, pyvista as pv
pv.OFF_SCREEN = True
OUT = os.path.expanduser("~/+Simulations/StagnantLid/mmpde_proof")
os.makedirs(OUT, exist_ok=True)

def mk_mesh_T():
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1/24, qdegree=3)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3, continuous=True)
    X = mesh.CoordinateSystem.X; r = sympy.sqrt(X[0]**2+X[1]**2); th = sympy.atan2(X[1], X[0])
    Tf = sympy.log(r)/sympy.log(0.5) + 0.3*sympy.sin(3*th)*sympy.sin(np.pi*(r-0.5)/0.5)
    T.data[:, 0] = np.asarray(uw.function.evaluate(Tf, T.coords)).reshape(-1)
    return mesh, T, Tf

def area_stats(mesh):
    pvm = vis.mesh_to_pv_mesh(mesh); c = pvm.cells.reshape(-1, 4)[:, 1:]; p = pvm.points[:, :2]; v = p[c]
    a = 0.5*((v[:,1,0]-v[:,0,0])*(v[:,2,1]-v[:,0,1]) - (v[:,2,0]-v[:,0,0])*(v[:,1,1]-v[:,0,1]))
    folded = int((np.sign(a) != np.sign(np.median(a))).sum())
    return folded, np.abs(a).max()/max(np.abs(a).min(), 1e-30)

def render_mesh(mesh, title, fname):
    edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()
    pl = pv.Plotter(off_screen=True, window_size=(900, 900)); pl.set_background("white")
    pl.add_mesh(edges, color="black", line_width=0.6, lighting=False)
    pl.add_text(title, font_size=12, color="black")
    pl.view_xy(); pl.camera.zoom(1.3)
    out = os.path.join(OUT, fname); pl.screenshot(out); pl.close()
    print("  render ->", out)

MK = dict(step_frac=0.2, accel=None, momentum=0.0)

print("=== (A) metric density ratio vs R  (expect ratio = R^3) ===")
mesh, T, Tf = mk_mesh_T()
for R in [1.4, 2.0, 3.0, 5.0, 8.0]:
    rho = uw.meshing.metric_density_from_gradient(mesh, T, refinement=R, coarsening="auto", metric_choice="front-following")
    v = np.asarray(uw.function.evaluate(rho, T.coords))
    rr = v.max()/max(v.min(), 1e-30)
    print(f"  R={R:>4}: density ratio={rr:7.1f}  (R^3={R**3:7.1f})  edge-length ratio={np.sqrt(rr):.1f}")

print("=== (B) mover on a FIXED ANALYTIC metric (ring) — expect dead-flat ===")
mesh, T, Tf = mk_mesh_T()
X = mesh.CoordinateSystem.X; r = sympy.sqrt(X[0]**2+X[1]**2)
ring = 1.0 + 8.0*sympy.exp(-((r-0.9)/0.05)**2)
seq = []
for it in range(8):
    uw.meshing.smooth_mesh_interior(mesh, metric=ring, method="mmpde", method_kwargs=MK, slip_surfaces=True, skip_threshold=None)
    seq.append(area_stats(mesh)[1])
print("  area_ratio per iter: " + " ".join(f"{x:.0f}" for x in seq))
render_mesh(mesh, "(B) fixed analytic ring metric -> clean", "B_analytic_ring.png")

print("=== (C) convection feedback loop: refinement=5 vs refinement=1.5 ===")
for R in [5.0, 1.5]:
    mesh, T, Tf = mk_mesh_T()
    seq = []; foldlast = 0
    for it in range(8):
        rho = uw.meshing.metric_density_from_gradient(mesh, T, refinement=R, coarsening="auto", metric_choice="front-following")
        uw.meshing.smooth_mesh_interior(mesh, metric=rho, method="mmpde", method_kwargs=MK, slip_surfaces=True, skip_threshold=None)
        T.data[:, 0] = np.asarray(uw.function.evaluate(Tf, T.coords)).reshape(-1)  # keep metric fixed-in-space
        foldlast, x = area_stats(mesh); seq.append(x)
    verdict = "RUNAWAY/HOLES" if max(seq) > 50 else "clean/flat"
    print(f"  refinement={R}: area_ratio per iter: " + " ".join(f"{x:.0f}" for x in seq) + f"   [{verdict}]")
    render_mesh(mesh, f"(C) convection loop, refinement={R}  area_ratio={seq[-1]:.0f}",
                f"C_refine_{str(R).replace('.','p')}.png")
print("DONE ->", OUT)
