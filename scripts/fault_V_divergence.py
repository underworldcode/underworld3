"""Discriminate physical-vs-numerical velocity 'blotches': render |v| and the
velocity divergence ∇·v side by side via the UW pyvista machinery. For true
incompressible Stokes ∇·v≈0 everywhere; if divergence SPIKES co-locate with the
dark |v| spots, the blotches are a discretisation/pressure artefact (continuous
P1 pressure under the strong FK viscosity contrast), not physical stagnation.
"""
import os, argparse
import numpy as np, sympy, underworld3 as uw, pyvista as pv
pv.OFF_SCREEN = True

ap = argparse.ArgumentParser()
ap.add_argument('--tag', type=str, default='fault_ti_Ra1e6_everystep')
ap.add_argument('--step', type=str, required=True)
args = ap.parse_args()
DIR = os.path.expanduser(f'~/+Simulations/StagnantLid/{args.tag}')
label = args.step

mesh = uw.discretisation.Mesh(os.path.join(DIR, f"{label}.mesh.00000.h5"))
V = uw.discretisation.MeshVariable("V_v2p1", mesh, mesh.dim, degree=2)
V.read_timestep(label, "V_v2p1", 0, outputPath=DIR)

X = mesh.CoordinateSystem.X
divV_expr = V.sym[0].diff(X[0]) + V.sym[1].diff(X[1])

# Project both to plottable scalars (P2 → samples the high-order field).
vmag = uw.discretisation.MeshVariable("vmag", mesh, 1, degree=2)
divv = uw.discretisation.MeshVariable("divv", mesh, 1, degree=2)
for var, expr in ((vmag, sympy.sqrt(V.sym.dot(V.sym))), (divv, divV_expr)):
    pr = uw.systems.Projection(mesh, var)
    pr.uw_function = expr
    pr.smoothing = 0.0
    pr.solve()

vm = np.sqrt((V.data**2).sum(1))
dv = divv.data[:, 0]
# Divergence scaled by local speed → a dimensionless incompressibility error.
print(f"[{label}] |v| max={vm.max():.2f} mean={vm.mean():.2f}", flush=True)
print(f"[{label}] div(v): max|·|={np.abs(dv).max():.2f}  "
      f"rms={np.sqrt((dv**2).mean()):.3f}  "
      f"rms/|v|_mean={np.sqrt((dv**2).mean())/vm.mean():.4f}", flush=True)

uw.visualisation.plot_scalar(
    mesh, vmag.sym, "Vmag", cmap="magma", clim=(0.0, float(vm.max())),
    save_png=True, dir_fname=os.path.join(DIR, f"diag_Vmag_{label}.png"))
dlim = float(np.abs(dv).max())
uw.visualisation.plot_scalar(
    mesh, divv.sym, "divV", cmap="RdBu_r", clim=(-dlim, dlim),
    save_png=True, dir_fname=os.path.join(DIR, f"diag_divV_{label}.png"))
print("→", os.path.join(DIR, f"diag_Vmag_{label}.png"))
print("→", os.path.join(DIR, f"diag_divV_{label}.png"))
