"""Check whether the free-slip outer boundary leaks (v·n != 0 throughflow) on a
saved snapshot. A clean free-slip BC has v·n ~ 0 on Upper; a leak (as seen in
the free-surface models) lets cold/hot material short-circuit through the top,
growing the cold BL and decaying convection.

Usage:
  python bc_leak_check.py --tag cmp2_uniform --step step0080 --sim-dir ~/+Simulations/StagnantLid
"""
import os, argparse
import numpy as np, underworld3 as uw

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--step", required=True)
ap.add_argument("--sim-dir", default="~/+Simulations/StagnantLid")
args = ap.parse_args()
D = os.path.expanduser(os.path.join(args.sim_dir, args.tag))

mesh = uw.discretisation.Mesh(os.path.join(D, f"{args.step}.mesh.00000.h5"))
V = uw.discretisation.MeshVariable("V_v2p1", mesh, vtype=uw.VarType.VECTOR,
                                   degree=2, continuous=True)
V.read_timestep(args.step, "V_v2p1", 0, outputPath=D)

C = np.asarray(V.coords)[:, :2]
r = np.sqrt((C ** 2).sum(axis=1))
Vd = np.asarray(V.data)[:, :2]
vmag = np.sqrt((Vd ** 2).sum(axis=1))
nhat = C / r[:, None]                                   # outward radial unit
vn = (Vd * nhat).sum(axis=1)                            # v·n (signed)
vt = Vd[:, 0] * (-nhat[:, 1]) + Vd[:, 1] * nhat[:, 0]   # tangential

r_outer = r.max()
upper = r > (r_outer - 1e-3)                            # outer-boundary nodes
inner = r < (r.min() + 1e-3)

print(f"=== {args.tag} {args.step} ===")
print(f"  |v| overall: max={vmag.max():.3e} mean={vmag.mean():.3e}")
for name, m in [("UPPER (outer free-slip)", upper), ("LOWER (no-slip)", inner)]:
    if not m.any():
        continue
    vnb = vn[m]; vtb = vt[m]; vmb = vmag[m]
    leak = np.abs(vnb).max() / max(vmb.max(), 1e-30)
    print(f"  {name}: n={m.sum()}")
    print(f"     |v·n| max={np.abs(vnb).max():.3e} rms={np.sqrt(np.mean(vnb**2)):.3e}")
    print(f"     |v·t| max={np.abs(vtb).max():.3e} rms={np.sqrt(np.mean(vtb**2)):.3e}")
    print(f"     leak ratio  |v·n|max/|v|max = {leak:.3%}   "
          f"(clean free-slip << 1%; >5% = leaking)")
    print(f"     net flux ∮v·n ~ {vnb.mean():.3e} (signed mean)")
