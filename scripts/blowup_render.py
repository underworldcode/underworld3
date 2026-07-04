"""Frame-by-frame diagnostic of the convection-onset blow-up. Every step: T
(RdBu_r) and |v| (per-frame scaled, so the PATTERN shows even as magnitude grows)
side by side, with the adaptive mesh + fault trace, annotated with vrms / Tmin /
Tmax parsed from the log. Assembles a GIF (watch the instability kick in) + a
montage of the onset window.
"""
import os, glob, re, argparse, numpy as np
import underworld3 as uw, underworld3.visualisation as vis, pyvista as pv

pv.OFF_SCREEN = True
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="passive_blowup")
ap.add_argument("--sim-dir", default="~/+Simulations/StagnantLid+Fault")
ap.add_argument("--dip", type=float, default=30.0); ap.add_argument("--depth", type=float, default=0.15)
ap.add_argument("--theta", type=float, default=90.0)
ap.add_argument("--montage-from", type=int, default=0)   # montage window start step
args = ap.parse_args()
D = os.path.expanduser(os.path.join(args.sim_dir, args.tag))

# fault trace
d, th = np.deg2rad(args.dip), np.deg2rad(args.theta)
P0=np.array([np.cos(th),np.sin(th)]); e=np.array([np.cos(th),np.sin(th)]); t=np.array([-np.sin(th),np.cos(th)])
dh=np.cos(d)*t-np.sin(d)*e; XY=P0[None,:]+np.linspace(0,args.depth/np.sin(d),30)[:,None]*dh[None,:]

# parse per-step vrms / T range from the log
stats = {}
logf = os.path.join(os.path.dirname(D), args.tag + ".log")
if os.path.exists(logf):
    for ln in open(logf):
        m = re.match(r"\s+(\d+)\s+[\d.]+\s+([\d.eE+-]+)\s+[\d.]+s\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+\[([+-][\d.]+),([+-][\d.]+)\]", ln)
        if m: stats[int(m.group(1))] = dict(dt=float(m.group(2)), vrms=float(m.group(3)),
                                            Tmin=float(m.group(5)), Tmax=float(m.group(6)))

cands = sorted(glob.glob(os.path.join(D, "step*.mesh.00000.h5")),
               key=lambda c: int(re.search(r"step(\d+)\.mesh", c).group(1)))
labels = [re.search(r"(step\d+)\.mesh", os.path.basename(c)).group(1) for c in cands]
steps = [int(l[4:]) for l in labels]
print("frames:", len(labels), "steps", steps[0], "..", steps[-1], flush=True)

def panels(label, step):
    mesh = uw.discretisation.Mesh(os.path.join(D, f"{label}.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable("T_v2p1", mesh, 1, degree=3, continuous=True, varsymbol="T")
    V = uw.discretisation.MeshVariable("V_v2p1", mesh, mesh.dim, degree=2)
    T.read_timestep(label, "T_v2p1", 0, outputPath=D); V.read_timestep(label, "V_v2p1", 0, outputPath=D)
    pvT = vis.meshVariable_to_pv_mesh_object(T); pvT.point_data["T"]=np.asarray(T.data[:,0])
    pvV = vis.meshVariable_to_pv_mesh_object(V); vmag=np.linalg.norm(np.asarray(V.data),axis=1); pvV.point_data["vmag"]=vmag
    edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()
    return pvT, pvV, edges, float(vmag.max())

def frame(pl, label, step):
    pvT,pvV,edges,vmax = panels(label,step)
    st = stats.get(step, {})
    info = "vrms=%.2g Tmin=%.3f Tmax=%.3f" % (st.get('vrms',np.nan), st.get('Tmin',np.nan), st.get('Tmax',np.nan))
    for col,(pv_obj,scal,cmap,clim,name) in enumerate([
            (pvT,"T","RdBu_r",(0,1),"T"), (pvV,"vmag","inferno",(0,max(vmax,1e-6)),"|v| max=%.2g"%vmax)]):
        pl.subplot(0,col); pl.set_background('white')
        pl.add_mesh(pv_obj, scalars=scal, cmap=cmap, clim=clim, lighting=False, show_scalar_bar=False, opacity=0.7)
        pl.add_mesh(edges, color="#222222", line_width=0.4, lighting=False)
        pl.add_mesh(pv.lines_from_points(np.column_stack([XY,np.zeros(len(XY))])), color="#00ff00", line_width=4, lighting=False)
        pl.add_text("%s %s\n%s" % (label, name, info if col==0 else ""), font_size=10, color='black')
        pl.view_xy(); pl.enable_parallel_projection(); pl.camera.focal_point=(0,0,0); pl.camera.position=(0,0,1); pl.camera.zoom(0.92)

# onset window frames
win = [ (l,s) for l,s in zip(labels,steps) if s >= (args.montage_from or max(steps[0], steps[-1]-11)) ]
ncol = (len(win)+1)//2
# precompute panels once per frame (mesh load is the cost)
cache = {s: panels(l,s) for l,s in win}

def montage(field, fname, title):
    pl = pv.Plotter(off_screen=True, shape=(2,ncol), window_size=(440*ncol,920), border=False)
    for k,(lab,s) in enumerate(win):
        r,c = divmod(k, ncol); pvT,pvV,edges,vmax = cache[s]; st=stats.get(s,{})
        pl.subplot(r,c); pl.set_background('white')
        if field=="T":
            pl.add_mesh(pvT, scalars="T", cmap="RdBu_r", clim=(0,1), lighting=False, show_scalar_bar=False, opacity=0.7)
        else:
            pl.add_mesh(pvV, scalars="vmag", cmap="inferno", clim=(0,max(vmax,1e-9)), lighting=False, show_scalar_bar=False, opacity=0.85)
        pl.add_mesh(edges, color="#222222", line_width=0.3, lighting=False)
        pl.add_mesh(pv.lines_from_points(np.column_stack([XY,np.zeros(len(XY))])), color="#00ff66", line_width=3, lighting=False)
        lab2 = "%s v=%.2g" % (lab, st.get('vrms',np.nan)) + (" |v|max=%.2g"%vmax if field!="T" else " T<=%.3f"%st.get('Tmin',np.nan))
        pl.add_text(lab2, font_size=9, color='black')
        pl.view_xy(); pl.enable_parallel_projection(); pl.camera.focal_point=(0,0,0); pl.camera.position=(0,0,1); pl.camera.zoom(0.9)
    out = os.path.join(D, fname); pl.screenshot(out); pl.close(); print("->", out, flush=True)

montage("T", "blowup_onset_T.png", "T")
montage("V", "blowup_onset_V.png", "|v| (per-frame scaled)")
print("DONE", flush=True)
