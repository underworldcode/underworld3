"""Side-by-side T comparison: passive-FAULT vs NO-FAULT control at the same step,
same Ra/perturbation. The mode-5 perturbation seeds an UPWELLING at theta=90 (under
the fault). If it forms in no-fault but is suppressed in passive-fault, the fault
REFINEMENT (not mechanics — proven inert) is biasing the convection.
"""
import os, argparse, numpy as np, underworld3 as uw, underworld3.visualisation as vis, pyvista as pv
pv.OFF_SCREEN = True
ap = argparse.ArgumentParser()
ap.add_argument("--step", required=True)
ap.add_argument("--a", default="passive_blowup"); ap.add_argument("--b", default="passive_nofault")
ap.add_argument("--dip", type=float, default=30.0); ap.add_argument("--depth", type=float, default=0.15)
ap.add_argument("--theta", type=float, default=90.0)
args = ap.parse_args()
SIM = os.path.expanduser("~/+Simulations/StagnantLid+Fault")
d, th = np.deg2rad(args.dip), np.deg2rad(args.theta)
P0=np.array([np.cos(th),np.sin(th)]); e=np.array([np.cos(th),np.sin(th)]); t=np.array([-np.sin(th),np.cos(th)])
dh=np.cos(d)*t-np.sin(d)*e; XY=P0[None,:]+np.linspace(0,args.depth/np.sin(d),30)[:,None]*dh[None,:]

def load(tag):
    D=os.path.join(SIM,tag); f=os.path.join(D,f"{args.step}.mesh.00000.h5")
    if not os.path.exists(f): return None
    m=uw.discretisation.Mesh(f)
    T=uw.discretisation.MeshVariable("T_v2p1",m,1,degree=3,continuous=True,varsymbol="T")
    T.read_timestep(args.step,"T_v2p1",0,outputPath=D)
    pvT=vis.meshVariable_to_pv_mesh_object(T); pvT.point_data["T"]=np.asarray(T.data[:,0])
    return pvT, vis.mesh_to_pv_mesh(m).extract_all_edges(), m._coords.shape[0]

pl=pv.Plotter(off_screen=True,shape=(1,2),window_size=(2200,1150),border=False)
for i,(tag,title) in enumerate([(args.a,"PASSIVE FAULT"),(args.b,"NO FAULT (control)")]):
    res=load(tag)
    pl.subplot(0,i); pl.set_background('white')
    if res is None:
        pl.add_text("%s: no %s yet"%(title,args.step),font_size=12,color='red'); continue
    pvT,edges,n=res
    pl.add_mesh(pvT,scalars="T",cmap="RdBu_r",clim=(0,1),lighting=False,show_scalar_bar=False,opacity=0.6)
    pl.add_mesh(edges,color="#222222",line_width=0.4,lighting=False)
    pl.add_mesh(pv.lines_from_points(np.column_stack([XY,np.zeros(len(XY))])),color="#00c000",line_width=5,lighting=False)
    pl.add_text("%s %s (%d nodes)"%(title,args.step,n),font_size=12,color='black')
    pl.view_xy(); pl.enable_parallel_projection(); pl.camera.focal_point=(0,0,0); pl.camera.position=(0,0,1); pl.camera.zoom(0.95)
out=os.path.join(SIM,f"compare_{args.step}.png"); pl.screenshot(out); pl.close(); print("->",out,flush=True)
