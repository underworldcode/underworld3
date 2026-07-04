"""Composition test under the non-tangling deal-breaker (no interior pins).

Question: does a graded BASE + a metric MATCHED to that grading let a FREE mmpde
solve MAINTAIN the fault refinement (vs a uniform base where the same metric must
CREATE it and hits the ~2.4x cap)?  And does the mesh stay VALID (no folding) —
the whole point of using mmpde.

Two arms, identical matched metric, free mmpde throughout:
  ARM 1  uniform base  + matched metric -> mmpde must CREATE  -> expect ~cap
  ARM 2  graded  base  + matched metric -> mmpde must MAINTAIN -> expect held

Reports fault/bulk NN-spacing ratio (lower=finer at fault) AND mesh validity
(folded-cell count + min triangle area), and RENDERS edges+fault for eyeball
comparison (documented pyvista pattern: white bg, lighting=False, mesh_to_pv_mesh
edges, RdBu not needed here — geometry only).
"""
import os, numpy as np, sympy, underworld3 as uw, underworld3.visualisation as vis, pyvista as pv
from underworld3.utilities.geometry_tools import signed_distance_pointcloud_polyline_2d
from scipy.spatial import cKDTree

pv.OFF_SCREEN = True
D = os.path.expanduser('~/+Simulations/StagnantLid+Fault'); os.makedirs(D, exist_ok=True)

R_o, R_i = 1.0, 0.5
BULK = 1/8
SMIN = 1/48                      # gmsh fine size at fault -> ratio ~6
RF   = BULK / SMIN              # metric anisotropy MATCHED to the gmsh grading
AMP  = 2.0
DIP  = np.deg2rad(30.0); DEPTH = 0.12; THETA = np.pi/2
SUBITERS = 4

def fault_polyline(theta0=THETA):
    P0 = np.array([np.cos(theta0), np.sin(theta0)])
    e  = np.array([np.cos(theta0), np.sin(theta0)])
    t  = np.array([-np.sin(theta0), np.cos(theta0)])
    dhat = np.cos(DIP)*t - np.sin(DIP)*e
    L = DEPTH/np.sin(DIP)
    return P0[None,:] + np.linspace(0,L,40)[:,None]*dhat[None,:], dhat
XY, DHAT = fault_polyline()

def apply_metric(mesh):
    """matched fault metric (same in both arms) + one batch of free mmpde."""
    dfac = uw.discretisation.MeshVariable("dist_fac", mesh, 1, degree=1, continuous=True)
    d = np.abs(signed_distance_pointcloud_polyline_2d(np.asarray(dfac.coords)[:,:2], XY))
    dfac.data[:,0] = d
    nrm = np.array([-DHAT[1], DHAT[0]]); nrm /= np.linalg.norm(nrm)
    nnT = sympy.Matrix(nrm.reshape(2,1)) * sympy.Matrix(nrm.reshape(2,1)).T
    refine_w, draw_w = 2.0*SMIN, 6.0*SMIN
    gw = sympy.exp(-(dfac.sym[0]/draw_w)**2)
    gs = sympy.exp(-(dfac.sym[0]/refine_w)**2)
    rho = 1.0 + AMP*gw
    tensor = rho*sympy.eye(2) + (RF**2 - 1.0)*gs*nnT
    for _ in range(SUBITERS):
        uw.meshing.smooth_mesh_interior(
            mesh, metric=tensor, method="mmpde", skip_threshold=None, slip_surfaces=True,
            method_kwargs=dict(step_frac=0.2, accel="cg", momentum=0.0), verbose=False)

def fault_ratio(mesh):
    C = np.asarray(mesh._coords)[:,:2]
    dist = np.abs(signed_distance_pointcloud_polyline_2d(C, XY))
    tr = cKDTree(C); dd,_ = tr.query(C, k=2); nn = dd[:,1]
    bulk = float(np.median(nn[dist > 0.3])); near = dist < 0.04
    return (float(np.median(nn[near]))/bulk if near.any() else np.nan), int(near.sum()), bulk

def validity(pvm):
    cells = pvm.cells.reshape(-1,4)[:,1:]; pts = pvm.points[:,:2]
    v = pts[cells]
    area = 0.5*((v[:,1,0]-v[:,0,0])*(v[:,2,1]-v[:,0,1]) - (v[:,2,0]-v[:,0,0])*(v[:,1,1]-v[:,0,1]))
    s = np.sign(np.median(area))                 # dominant orientation
    n_fold = int((np.sign(area) != s).sum())
    return n_fold, float(np.abs(area).min()), float(np.abs(area).max())

PANELS = []   # (title, mesh-snapshot edges, ratio-string)

def snap(mesh, title):
    pvm = vis.mesh_to_pv_mesh(mesh)
    r,n,b = fault_ratio(mesh); nf, amin, amax = validity(pvm)
    s = "%s | fault/bulk=%.2f (n=%d) | folded=%d min_area=%.1e" % (title, r, n, nf, amin)
    print(s, flush=True)
    PANELS.append((title, pvm.extract_all_edges(), "ratio %.2f  folded %d" % (r, nf)))
    return r, nf

# ---------------- ARM 1: uniform base ----------------
print("=== ARM 1: uniform base + matched metric (mmpde must CREATE) ===", flush=True)
m1 = uw.meshing.Annulus(radiusOuter=R_o, radiusInner=R_i, cellSize=BULK, qdegree=3)
snap(m1, "uniform base (as-built)")
apply_metric(m1)
r1, f1 = snap(m1, "uniform + metric (ARM 1)")

# ---------------- ARM 2: graded base ----------------
print("=== ARM 2: graded base (gmsh refine_lines) + matched metric (mmpde MAINTAINS) ===", flush=True)
m2 = uw.meshing.Annulus(radiusOuter=R_o, radiusInner=R_i, cellSize=BULK, qdegree=3,
                        refine_lines=[XY], refine_size_min=SMIN,
                        refine_dist_min=0.03, refine_dist_max=0.18)
snap(m2, "graded base (as-built)")
apply_metric(m2)
r2, f2 = snap(m2, "graded + metric (ARM 2)")

# ---------------- RENDER comparison ----------------
# zoom to the fault region
fx, fy = XY[len(XY)//2]
show = [PANELS[1], PANELS[2], PANELS[3]]   # uniform-after, graded-before, graded-after
pl = pv.Plotter(off_screen=True, shape=(1,3), window_size=(2100,750), border=False)
for i,(title, edges, lab) in enumerate(show):
    pl.subplot(0,i); pl.set_background('white')
    pl.add_mesh(edges, color='#333333', line_width=0.6, lighting=False)
    pl.add_mesh(pv.lines_from_points(np.column_stack([XY, np.zeros(len(XY))])),
                color='#00c000', line_width=4, lighting=False)
    pl.add_text("%s\n%s" % (title, lab), font_size=10, color='black')
    pl.view_xy(); pl.enable_parallel_projection()
    pl.camera.focal_point = (fx, fy, 0); pl.camera.position = (fx, fy, 1); pl.camera.zoom(4.5)
out = os.path.join(D, "composition_compare.png"); pl.screenshot(out); pl.close()
print("->", out, flush=True)
print("\nSUMMARY: ARM1 uniform->%.2f (folded %d) | ARM2 graded->%.2f (folded %d) | gmsh-target~%.2f"
      % (r1, f1, r2, f2, 1.0/RF), flush=True)
print("DONE", flush=True)
