"""Thermal convection with the Eulerian SUPG solver against SLCN: box and annulus.

Identical Stokes problem, identical mesh, identical timestep rule; only the
transport scheme differs. Reports the Nusselt number (boundary heat-flux
integrals), Vrms, and the wall time of the Stokes and transport solves.

  python convection_benchmark.py -uw_geometry box -uw_scheme supg -uw_rayleigh 1e5
  mpirun -n 4 python convection_benchmark.py -uw_geometry annulus -uw_scheme slcn

Box: unit square, free-slip walls, T = 1 below and 0 above, Blankenbach
(1989) case 1b at Ra = 1e5 (Nu 10.534, Vrms 193.21). Annulus: r = 0.5 to 1,
no-slip walls as in the shipped example, buoyancy Ra T / r_i^3.
"""
import os, time
import numpy as np
import sympy
from mpi4py import MPI
import underworld3 as uw

params = uw.Params(
    uw_geometry="box", uw_scheme="supg", uw_rayleigh=1.0e5, uw_cell_size=1.0 / 32,
    uw_courant=1.0, uw_max_steps=2000, uw_t_end=0.0, uw_steady_tol=1.0e-4,
    uw_degree=2, uw_outdir="", uw_log_every=10, uw_checkpoint_every=25, uw_init_from="",
)
geometry, scheme = str(params.uw_geometry), str(params.uw_scheme)
Ra, cell, courant = float(params.uw_rayleigh), float(params.uw_cell_size), float(params.uw_courant)
comm = uw.mpi.comm
outdir = str(params.uw_outdir) or f"{geometry}_{scheme}_Ra{Ra:g}_h{cell:g}_C{courant:g}"
if uw.mpi.rank == 0:
    os.makedirs(os.path.join(outdir, "checkpoints"), exist_ok=True)
comm.barrier()
log = open(os.path.join(outdir, "steps.log"), "w") if uw.mpi.rank == 0 else None

if geometry == "box":
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=cell, regular=False, qdegree=3)
    x, y = mesh.X
    hot, cold = "Bottom", "Top"
    init_t = (1.0 - y) + 0.1 * sympy.cos(sympy.pi * x) * sympy.sin(sympy.pi * y)
    body = sympy.Matrix([0, Ra])
    Q_cond = 1.0                       # conductive flux through a unit width
else:
    r_i, r_o = 0.5, 1.0
    mesh = uw.meshing.Annulus(radiusInner=r_i, radiusOuter=r_o, cellSize=cell, degree=1, qdegree=3)
    x, y = mesh.X
    hot, cold = "Lower", "Upper"
    r = sympy.sqrt(x ** 2 + y ** 2)
    th = sympy.atan2(y + 1.0e-5, x + 1.0e-5)
    init_t = (r_o - r) / (r_o - r_i) + 0.05 * sympy.sin(4 * th) * sympy.sin(sympy.pi * (r - r_i) / (r_o - r_i))
    body = (mesh.X / (1.0e-10 + r)) * (Ra / r_i ** 3)
    Q_cond = 2 * np.pi / np.log(r_o / r_i)

v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=int(params.uw_degree))

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes.tolerance = 1.0e-4
if geometry == "box":
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Top")
    stokes.add_dirichlet_bc((0.0, sympy.oo), "Left")
    stokes.add_dirichlet_bc((0.0, sympy.oo), "Right")
else:
    stokes.add_dirichlet_bc((0.0, 0.0), "Upper")
    stokes.add_dirichlet_bc((0.0, 0.0), "Lower")
stokes.bodyforce = body * T.sym[0]

Solver = {"supg": uw.systems.AdvDiffusionSUPG, "slcn": uw.systems.AdvDiffusionSLCN}[scheme]
adv = Solver(mesh, u_Field=T, V_fn=v)
adv.constitutive_model = uw.constitutive_models.DiffusionModel
adv.constitutive_model.Parameters.diffusivity = 1.0
if scheme == "slcn":
    adv.theta = 0.5                    # SUPG: Crank-Nicolson by construction (order 1)
adv.add_dirichlet_bc(1.0, hot)
adv.add_dirichlet_bc(0.0, cold)

T.array[:, 0, 0] = uw.function.evaluate(init_t, T.coords).reshape(-1)
if str(params.uw_init_from):
    # Start from another run's last checkpoint (the steady state of a lower
    # Rayleigh number, say): read it on its own mesh and interpolate.
    import glob, re
    src = os.path.join(os.path.dirname(os.path.abspath(outdir)), str(params.uw_init_from), "checkpoints")
    last = max(int(re.search(r"\.T\.(\d+)\.h5$", f).group(1))
               for f in glob.glob(os.path.join(src, "checkpoint.mesh.T.*.h5")))
    mesh_src = uw.discretisation.Mesh(os.path.join(src, "checkpoint.mesh.00000.h5"))
    T_src = uw.discretisation.MeshVariable("T_src", mesh_src, 1, degree=int(params.uw_degree))
    T_src.read_timestep("checkpoint", "T", last, outputPath=src)
    T.array[:, 0, 0] = uw.function.evaluate(T_src.sym[0], T.coords).reshape(-1)
    if uw.mpi.rank == 0:
        print(f"initialised T from {params.uw_init_from} step {last}", flush=True)

vol = uw.maths.Integral(mesh, 1.0).evaluate()
v2 = uw.maths.Integral(mesh, v.sym.dot(v.sym))

# Nusselt number: the total heat transport q = v T - grad T, evaluated at the
# integration points and projected onto a nodal field, integrated over each
# wall along the wall-normal direction (y on the box, radial on the annulus);
# also its integral across the interior mid-line / mid-shell, which is the
# better-resolved number when the wall boundary layer is thin.
if geometry == "box":
    n_hat = sympy.Matrix([0, 1])
else:
    n_hat = (mesh.X / r).T
gradT = sympy.Matrix([T.sym[0].diff(x), T.sym[0].diff(y)])
q_n = ((v.sym.T * T.sym[0] - gradT).T * n_hat)[0]
qn_var = uw.discretisation.MeshVariable("q_n", mesh, 1, degree=int(params.uw_degree))
q_proj = uw.systems.Projection(mesh, qn_var)
q_proj.uw_function = q_n
q_proj.smoothing = 0.0
flux_cold = uw.maths.BdIntegral(mesh, qn_var.sym[0], cold)
flux_hot = uw.maths.BdIntegral(mesh, qn_var.sym[0], hot)
if geometry == "box":
    _xs = np.linspace(0.0, 1.0, 801); _mid_pts = np.c_[_xs, 0.5 * np.ones_like(_xs)]
else:
    _th = np.linspace(0, 2 * np.pi, 1441)[:-1]; _mid_pts = np.c_[0.75 * np.cos(_th), 0.75 * np.sin(_th)]

def diagnostics():
    vrms = float(np.sqrt(v2.evaluate() / vol))
    q_proj.solve()
    nu_cold = float(flux_cold.evaluate()) / Q_cond
    nu_hot = float(flux_hot.evaluate()) / Q_cond
    qm = uw.function.evaluate(qn_var.sym[0], _mid_pts).reshape(-1)
    nu_mid = (float(np.trapezoid(qm, _xs)) if geometry == "box" else float(qm.mean() * 2 * np.pi * 0.75)) / Q_cond
    return vrms, nu_cold, nu_hot, nu_mid

ckpt = os.path.join(outdir, "checkpoints", "checkpoint")
every = int(params.uw_checkpoint_every)

stokes.solve(zero_init_guess=True)
t_model, step = 0.0, 0
mesh.petsc_save_checkpoint(index=0, meshVars=[v, T], outputPath=ckpt)
wall_stokes = wall_adv = 0.0
nu_hist = []
t_end = float(params.uw_t_end)
while step < int(params.uw_max_steps):
    t0 = time.perf_counter()
    stokes.solve(zero_init_guess=False)
    t1 = time.perf_counter()
    dt = courant * stokes.estimate_dt()
    adv.solve(timestep=dt, zero_init_guess=False)
    t2 = time.perf_counter()
    wall_stokes += t1 - t0; wall_adv += t2 - t1
    t_model += dt; step += 1
    vrms, nu_cold, nu_hot, nu_mid = diagnostics()
    nu_hist.append(nu_cold)
    if uw.mpi.rank == 0 and (step % int(params.uw_log_every) == 0 or step == 1):
        line = (f"step {step:5d} t {t_model:.5f} dt {dt:.3e} vrms {vrms:.4f} "
                f"nu_cold {nu_cold:.4f} nu_hot {nu_hot:.4f} nu_mid {nu_mid:.4f} "
                f"stokes {t1 - t0:.3f}s adv {t2 - t1:.3f}s "
                f"ksp_its {adv.snes.getKSP().getIterationNumber()} snes_its {adv.snes.getIterationNumber()}")
        print(line, flush=True); log.write(line + "\n"); log.flush()
    if every > 0 and step % every == 0:
        mesh.petsc_save_checkpoint(index=step, meshVars=[v, T], outputPath=ckpt)
    if t_end > 0 and t_model >= t_end:
        break
    if len(nu_hist) > 50 and t_end <= 0:
        drift = abs(nu_hist[-1] - nu_hist[-51]) / abs(nu_hist[-1])
        if drift < float(params.uw_steady_tol):
            break

wall_stokes = comm.allreduce(wall_stokes, op=MPI.MAX); wall_adv = comm.allreduce(wall_adv, op=MPI.MAX)
vrms, nu_cold, nu_hot, nu_mid = diagnostics()
mesh.petsc_save_checkpoint(index=step, meshVars=[v, T], outputPath=ckpt)
if uw.mpi.rank == 0:
    cS, cE = mesh.dm.getHeightStratum(0)
    line = (f"RESULT geometry={geometry} scheme={scheme} Ra={Ra:g} h={cell:g} C={courant:g} np={comm.size} "
            f"steps={step} t={t_model:.5f} vrms={vrms:.4f} nu_cold={nu_cold:.4f} nu_hot={nu_hot:.4f} nu_mid={nu_mid:.4f} "
            f"stokes_per_step={wall_stokes / step:.3f}s adv_per_step={wall_adv / step:.3f}s")
    print(line, flush=True); log.write(line + "\n"); log.close()
