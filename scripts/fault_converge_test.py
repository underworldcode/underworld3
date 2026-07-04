"""Convergence trace: does a FREE mmpde solve on a graded base, with a matched
metric, MAINTAIN the fault grading, RELAX it (Louis' prediction), or keep
SQUEEZING toward eventual quality loss?

Graded base built once (gmsh refine_lines). Metric re-evaluated at the CURRENT
node positions each iteration (Eulerian distance to the static fault). One mmpde
sweep per iteration; record fault/bulk ratio + folded-cell count + min triangle
area after each. Two metric strengths to separate metric-overshoot from drift:
  AMP=2  (as ARM 2 — isotropic draw on top of the matched anisotropy)
  AMP=0  (pure matched anisotropy — should sit at the gmsh grading if maintained)
Render mesh snapshots at iters 0/6/12/24 for the AMP=2 trace.
"""
import os, numpy as np, sympy, underworld3 as uw, underworld3.visualisation as vis, pyvista as pv
from underworld3.utilities.geometry_tools import signed_distance_pointcloud_polyline_2d
from scipy.spatial import cKDTree

pv.OFF_SCREEN = True
D = os.path.expanduser('~/+Simulations/StagnantLid+Fault'); os.makedirs(D, exist_ok=True)
R_o, R_i = 1.0, 0.5
BULK, SMIN = 1/8, 1/48
RF = BULK/SMIN
DIP, DEPTH, THETA = np.deg2rad(30.0), 0.12, np.pi/2
N_ITER, SNAP_AT = 24, [0, 6, 12, 24]

def fault_polyline(theta0=THETA):
    P0 = np.array([np.cos(theta0), np.sin(theta0)])
    e = np.array([np.cos(theta0), np.sin(theta0)]); t = np.array([-np.sin(theta0), np.cos(theta0)])
    dhat = np.cos(DIP)*t - np.sin(DIP)*e
    return P0[None,:] + np.linspace(0, DEPTH/np.sin(DIP), 40)[:,None]*dhat[None,:], dhat
XY, DHAT = fault_polyline()
NRM = np.array([-DHAT[1], DHAT[0]]); NRM /= np.linalg.norm(NRM)

def fault_ratio(mesh):
    C = np.asarray(mesh._coords)[:,:2]
    dist = np.abs(signed_distance_pointcloud_polyline_2d(C, XY))
    tr = cKDTree(C); dd,_ = tr.query(C, k=2); nn = dd[:,1]
    near = dist < 0.04; bulk = float(np.median(nn[dist > 0.3]))
    return (float(np.median(nn[near]))/bulk if near.any() else np.nan), int(near.sum())

def validity(mesh):
    pvm = vis.mesh_to_pv_mesh(mesh); cells = pvm.cells.reshape(-1,4)[:,1:]; p = pvm.points[:,:2]
    v = p[cells]
    a = 0.5*((v[:,1,0]-v[:,0,0])*(v[:,2,1]-v[:,0,1]) - (v[:,2,0]-v[:,0,0])*(v[:,1,1]-v[:,0,1]))
    s = np.sign(np.median(a))
    return int((np.sign(a) != s).sum()), float(np.abs(a).min()), pvm

def trace(AMP, want_snaps=False):
    mesh = uw.meshing.Annulus(radiusOuter=R_o, radiusInner=R_i, cellSize=BULK, qdegree=3,
                              refine_lines=[XY], refine_size_min=SMIN,
                              refine_dist_min=0.03, refine_dist_max=0.18)
    dfac = uw.discretisation.MeshVariable("dist_fac", mesh, 1, degree=1, continuous=True)
    nnT = sympy.Matrix(NRM.reshape(2,1)) * sympy.Matrix(NRM.reshape(2,1)).T
    refine_w, draw_w = 2.0*SMIN, 6.0*SMIN
    gw = sympy.exp(-(dfac.sym[0]/draw_w)**2); gs = sympy.exp(-(dfac.sym[0]/refine_w)**2)
    tensor = (1.0 + AMP*gw)*sympy.eye(2) + (RF**2 - 1.0)*gs*nnT

    def reeval():        # Eulerian: distance to the static fault at CURRENT positions
        d = np.abs(signed_distance_pointcloud_polyline_2d(np.asarray(dfac.coords)[:,:2], XY))
        dfac.data[:,0] = d
    reeval()
    traj, snaps = [], {}
    r,n = fault_ratio(mesh); nf,amin,pvm = validity(mesh)
    traj.append((0, r, nf, amin))
    if want_snaps and 0 in SNAP_AT: snaps[0] = (pvm.extract_all_edges(), r, nf)
    for it in range(1, N_ITER+1):
        reeval()
        uw.meshing.smooth_mesh_interior(mesh, metric=tensor, method="mmpde", skip_threshold=None,
            slip_surfaces=True, method_kwargs=dict(step_frac=0.2, accel="cg", momentum=0.0), verbose=False)
        r,n = fault_ratio(mesh); nf,amin,pvm = validity(mesh)
        traj.append((it, r, nf, amin))
        if want_snaps and it in SNAP_AT: snaps[it] = (pvm.extract_all_edges(), r, nf)
    return traj, snaps

print("AMP=2 (isotropic draw + matched anisotropy):", flush=True)
t2, snaps = trace(2.0, want_snaps=True)
for it,r,nf,amin in t2:
    if it in (0,1,2,4,6,9,12,16,20,24): print("  iter %2d  ratio %.3f  folded %d  min_area %.1e" % (it,r,nf,amin), flush=True)
print("AMP=0 (pure matched anisotropy):", flush=True)
t0,_ = trace(0.0)
for it,r,nf,amin in t0:
    if it in (0,1,2,4,6,9,12,16,20,24): print("  iter %2d  ratio %.3f  folded %d  min_area %.1e" % (it,r,nf,amin), flush=True)

# render AMP=2 snapshots
fx, fy = XY[len(XY)//2]
order = [k for k in SNAP_AT if k in snaps]
pl = pv.Plotter(off_screen=True, shape=(1,len(order)), window_size=(700*len(order),750), border=False)
for i,k in enumerate(order):
    edges, r, nf = snaps[k]
    pl.subplot(0,i); pl.set_background('white')
    pl.add_mesh(edges, color='#333333', line_width=0.6, lighting=False)
    pl.add_mesh(pv.lines_from_points(np.column_stack([XY, np.zeros(len(XY))])), color='#00c000', line_width=4, lighting=False)
    pl.add_text("iter %d\nratio %.2f  folded %d" % (k, r, nf), font_size=10, color='black')
    pl.view_xy(); pl.enable_parallel_projection()
    pl.camera.focal_point=(fx,fy,0); pl.camera.position=(fx,fy,1); pl.camera.zoom(4.5)
out = os.path.join(D, "converge_trace.png"); pl.screenshot(out); pl.close()
print("->", out, flush=True)
r_end2 = t2[-1][1]; r_end0 = t0[-1][1]
print("\nSUMMARY  AMP=2: 0.19-ish -> %.3f (24 iters)  |  AMP=0: -> %.3f  |  gmsh target ~%.2f"
      % (r_end2, r_end0, 1.0/RF), flush=True)
print("DONE", flush=True)
