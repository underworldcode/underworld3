"""Memory per step of the convection loop's parts, to find a leak.

  python probe_memory_per_step.py --parts stokes,adv,diag --steps 60

Prints the largest resident size over ranks every 10 steps, and the growth per step
fitted over the second half.
"""
import argparse

import numpy as np
import sympy
import underworld3 as uw
from mpi4py import MPI

ap = argparse.ArgumentParser()
ap.add_argument("--parts", default="stokes,adv,diag")
ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--res", type=int, default=32)
ap.add_argument("--refinement", type=int, default=2)
args = ap.parse_args()
parts = set(args.parts.split(","))


def rss_mb():
    with open("/proc/self/statm") as f:
        pages = int(f.read().split()[1])
    return uw.mpi.comm.allreduce(pages * 4096 / 1e6, op=MPI.MAX)


mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5,
                          cellSize=2 ** args.refinement / args.res, refinement=args.refinement, qdegree=3)
x, y = mesh.X
r = sympy.sqrt(x ** 2 + y ** 2)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
V = uw.discretisation.MeshVariable("V", mesh, 2, degree=2)
P = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
T.array[:, 0, 0] = uw.function.evaluate(
    1 - (r - 0.5) / 0.5 + 0.01 * sympy.cos(sympy.atan2(y, x)), T.coords).reshape(-1)

stokes = uw.systems.Stokes(mesh, velocityField=V, pressureField=P)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = sympy.exp(np.log(1e3) * (1 - T.sym[0]))
stokes.add_essential_bc((0.0, 0.0), "Lower")
unit_r = mesh.CoordinateSystem.unit_e_0
stokes.add_natural_bc(1e6 * V.sym.dot(unit_r) * unit_r, "Upper")
stokes.bodyforce = 1e7 * T.sym[0] * unit_r
stokes.tolerance = 1e-5
stokes.petsc_options["fieldsplit_velocity_pc_type"] = "mg"
stokes.petsc_options["fieldsplit_velocity_pc_mg_type"] = "full"
stokes.petsc_options["fieldsplit_velocity_pc_mg_galerkin"] = "both"

adv = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=V.sym, theta=1.0, monotone_mode="clamp")
adv.constitutive_model = uw.constitutive_models.DiffusionModel
adv.constitutive_model.Parameters.diffusivity = 1.0
adv.add_dirichlet_bc(0.0, "Upper")
adv.add_dirichlet_bc(1.0, "Lower")
flux = uw.maths.BdIntegral(mesh=mesh, fn=T.sym[0], boundary="Upper")
volume = uw.maths.Integral(mesh, V.sym.dot(V.sym))
X = mesh.CoordinateSystem.X
normal = mesh.Gamma_N
surface_flux = uw.maths.BdIntegral(
    mesh=mesh, fn=-(T.sym[0].diff(X[0]) * normal[0] + T.sym[0].diff(X[1]) * normal[1]),
    boundary="Upper")

stokes.solve()
history = []
for step in range(args.steps):
    if "stokes" in parts:
        stokes.solve(zero_init_guess=False)
    if "adv" in parts:
        adv.solve(timestep=1.0e-4, zero_init_guess=False)
    if "advvary" in parts:
        adv.solve(timestep=1.0e-4 * (1.0 + 0.01 * step), zero_init_guess=False)
    if "diag" in parts:
        flux.evaluate()
        volume.evaluate()
    if "dt" in parts:
        adv.estimate_dt(direction_aware=True, percentile=50.0)
    if "eval" in parts:
        uw.function.evaluate(V.sym.dot(V.sym), mesh.X.coords)
    if "nu" in parts:
        surface_flux.evaluate()
    m = rss_mb()
    history.append(m)
    if step % 10 == 0:
        uw.pprint(f"step {step}: max rss {m:.0f} MB")
half = np.arange(len(history))[len(history) // 2:]
slope = np.polyfit(half, np.array(history)[half], 1)[0]
uw.pprint(f"RESULT parts={args.parts} np={uw.mpi.size}: growth {slope:.2f} MB per step per rank "
          f"(second half), final {history[-1]:.0f} MB")
