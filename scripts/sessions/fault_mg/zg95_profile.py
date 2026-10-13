"""Where a step's time goes: the meshing (placement, split, solver creation)
against the solve, and inside the solve the assembly, the hierarchy set-up
and the Krylov work. One add-on of the Zhong & Gurnis analogue, three solves:
the first (everything built), a repeat with nothing changed, and a repeat
after the temperature changed (operator refreshed, transfers kept).

python zg95_profile.py <outdir>
"""
import os
import sys
import time

import numpy as np
import sympy
from petsc4py import PETSc

out = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/+Simulations/zg95")
os.makedirs(out, exist_ok=True)
PETSc.Options()["log_view"] = f":{os.path.join(out, 'profile_log_view.txt')}"
PETSc.Log.begin()
import underworld3 as uw
from scipy.special import erf
from underworld3.utilities.custom_mg import set_custom_fmg
from underworld3.utilities.place_surface import place_fault_ribbon_2d

RI, RO = 0.55, 1.0
S, W = 0.005, 0.01
DIP, DEPTH, RA, DETA, ETA_W = np.deg2rad(45.0), 0.15, 1e6, 1e3, 1e-3
T = {}


def tic(k):
    T["_k"], T["_t"] = k, time.perf_counter()


def toc():
    T[T["_k"]] = T.get(T["_k"], 0.0) + time.perf_counter() - T["_t"]


tic("base mesh + levels")
base = uw.meshing.Annulus(radiusOuter=RO, radiusInner=RI, cellSize=0.16, refinement=3, qdegree=2)
tail = base._coarse_level_meshes()[:-1] + [base]
toc()
T_b = uw.discretisation.MeshVariable("T_b", base, 1, degree=2)
xb = np.asarray(T_b.coords)
z = (RO - np.hypot(xb[:, 0], xb[:, 1])) / (RO - RI)
th = np.arctan2(xb[:, 1], xb[:, 0])
T_b.array[:, 0, 0] = np.clip(0.6 * erf(z / 0.1) + 0.4 * (1 - erf((1 - z) / 0.1)) - 0.04 * np.cos(3 * th) * np.sin(np.pi * z), 0, 1)

t_hat, s_hat = np.array([1.0, 0.0]), np.array([0.0, 1.0])
d_hat = -np.cos(DIP) * t_hat + np.sin(DIP) * s_hat
n_hat = np.array([-d_hat[1], d_hat[0]])
top = t_hat * (RO + 0.02)
L = (DEPTH + 0.02) / np.cos(DIP)
n_s = int(round(L / S))
spine = np.array([top + d_hat * (L * i / n_s) for i in range(n_s + 1)])
trace = spine[np.hypot(spine[:, 0], spine[:, 1]) <= RO - 3 * S]

stage = PETSc.Log.Stage("placement+split")
stage.push()
tic("placement + split (ladder, cap, cut)")
mesh, rinfo = place_fault_ribbon_2d(base, [("Trench", trace)], W, spines=[("Trench", spine)], margin_rings=1,
                                    clearance=0.7, split=True, mesher="ladder")
toc()
stage.pop()
band = np.asarray(rinfo["band"], dtype=bool)

tic("variables + solver object")
T_f = uw.discretisation.MeshVariable("T_f", mesh, 1, degree=2)
v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True)
zone = uw.discretisation.MeshVariable("zone", mesh, 1, degree=0, continuous=False)
dm = mesh.dm
vS, vE = dm.getDepthStratum(0)
cS, cE = dm.getHeightStratum(0)
X = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, 2)
pairs = mesh._fault_point_pairs["Trench"]
pp = np.array([int(q) for q in pairs] + [int(q) for q in pairs.values()])
pv = pp[(pp >= vS) & (pp < vE)] - vS
r_tip = float(np.hypot(X[pv, 0], X[pv, 1]).max())
cent = np.array([X[[int(q) - vS for q in dm.getTransitiveClosure(c)[0] if vS <= q < vE]].mean(axis=0) for c in range(cS, cE)])
zone.array[:, 0, 0] = (band & (np.hypot(cent[:, 0], cent[:, 1]) > r_tip - 0.5 * W)).astype(float)
s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
eta0 = sympy.exp(sympy.log(DETA) * (1 - T_f.sym[0]))
s.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
s.constitutive_model.Parameters.shear_viscosity_0 = eta0
s.constitutive_model.Parameters.shear_viscosity_1 = eta0 * (1 - (1 - ETA_W) * zone.sym[0])
s.constitutive_model.Parameters.director = sympy.Matrix([[float(n_hat[0]), float(n_hat[1])]])
unit_r = mesh.CoordinateSystem.unit_e_0
s.bodyforce = RA * T_f.sym[0] * unit_r
s.add_rotated_freeslip_bc(0, "Upper", normal=unit_r)
s.add_rotated_freeslip_bc(0, "Lower", normal=unit_r)
s.add_fault_bc(0, boundary="Trench")
s.petsc_use_pressure_nullspace = True
s.tolerance = 1e-5
set_custom_fmg(s, tail, field_id=0)
toc()


def clamp(x):
    r = np.hypot(x[:, 0], x[:, 1])
    return x * (np.clip(r, RI + 1e-10, RO - 1e-10) / np.maximum(r, 1e-14))[:, None]


tic("T transfer")
T_f.array[:, 0, 0] = uw.function.evaluate(T_b.sym[0], clamp(np.asarray(T_f.coords))).ravel()
toc()
for name in ("solve 1 (first: hierarchy built)", "solve 2 (nothing changed)", "solve 3 (T changed)"):
    if name.startswith("solve 3"):
        T_f.array[:, 0, 0] = np.clip(T_f.array[:, 0, 0] * 0.98 + 0.01, 0, 1)
    stage = PETSc.Log.Stage(name.split(" (")[0].replace(" ", "_"))
    stage.push()
    tic(name)
    s.solve()
    toc()
    stage.pop()
    info = s._rotated_freeslip_info
    print(f"{name}: {T[name]:.1f} s, velocity its {info.get('vel_its_last')}, pressure its {info.get('pres_its_last')}", flush=True)
print("TIMES", {k: round(v_, 2) for k, v_ in T.items() if not k.startswith("_")}, flush=True)
print("DONE", flush=True)
