"""Render the TI fault convection with the UW pyvista machinery (P3-aware).
Uses uw.visualisation.plot_scalar / plot_vector / plot_mesh — NOT matplotlib.
"""
import os, glob, re, argparse
import underworld3 as uw
import pyvista as pv

pv.OFF_SCREEN = True

ap = argparse.ArgumentParser()
ap.add_argument('--tag', type=str, default='fault_ti_Ra1e6_fmg')
ap.add_argument('--step', type=str, default='')
args = ap.parse_args()
DIR = os.path.expanduser(f'~/+Simulations/StagnantLid/{args.tag}')
cands = sorted(glob.glob(os.path.join(DIR, "step*.mesh.00000.h5")),
               key=lambda c: int(re.search(r"step(\d+)\.mesh", c).group(1)))
label = args.step or re.search(r"(step\d+)\.mesh", os.path.basename(cands[-1])).group(1)
print(f"rendering {label} of {args.tag} via uw.visualisation", flush=True)

mesh = uw.discretisation.Mesh(os.path.join(DIR, f"{label}.mesh.00000.h5"))
T = uw.discretisation.MeshVariable("T_v2p1", mesh, 1, degree=3, varsymbol="T")
V = uw.discretisation.MeshVariable("V_v2p1", mesh, mesh.dim, degree=2)
T.read_timestep(label, "T_v2p1", 0, outputPath=DIR)
V.read_timestep(label, "V_v2p1", 0, outputPath=DIR)

import sympy

# Temperature (P3) — UW pyvista, proper high-order sampling. Pass `.sym`
# (the 1x1 matrix) and scalar_name positionally, as the canonical examples do.
uw.visualisation.plot_scalar(
    mesh, T.sym, "T", cmap="RdBu_r", clim=(0, 1),
    save_png=True, dir_fname=os.path.join(DIR, f"pv_T_{label}.png"))

# Velocity — proper vector plot (arrows coloured by |v|), the UW way.
# clim passed explicitly: the tool's default clim="" trips np.any("") in this
# pyvista version (latent bug in uw.visualisation).
uw.visualisation.plot_vector(
    mesh, V, vector_name="V", vfreq=3, vmag=4e-3, cmap="magma", clim=(0.0, 80.0),
    save_png=True, dir_fname=os.path.join(DIR, f"pv_V_{label}.png"))

print("→", os.path.join(DIR, f"pv_T_{label}.png"))
print("→", os.path.join(DIR, f"pv_Vmag_{label}.png"))
print("→", os.path.join(DIR, f"pv_mesh_{label}.png"))
