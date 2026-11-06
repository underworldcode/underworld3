"""State probe: MMPDE mover + multigrid + free slip on an annulus, current development.

One arm per process, one JSON record per solve on stdout, prefixed PROBE.

  python probe_state.py --bc rot-default --pc custom --moves 3
"""
import argparse
import json
import time

import numpy as np
import sympy
from mpi4py import MPI

import underworld3 as uw
from underworld3.meshing.smoothing import graph

ap = argparse.ArgumentParser()
ap.add_argument("--cell", type=float, default=1 / 8)
ap.add_argument("--refinement", type=int, default=2)
ap.add_argument("--R", type=float, default=5.0)
ap.add_argument("--moves", type=int, default=2, help="mover calls after the first solve")
ap.add_argument("--bc", default="rot-default",
                choices=["rot-default", "rot-radial", "nitsche", "penalty", "none"])
ap.add_argument("--pc", default="auto", choices=["auto", "custom", "gamg"])
ap.add_argument("--slip", type=int, default=1)
ap.add_argument("--transport", default="none", choices=["none", "supg", "slcn"])
ap.add_argument("--T-analytic", type=int, default=0,
                help="1: reset T from its formula after each move, so a v_rms change is the mesh's, not the remap's")
ap.add_argument("--rotation-nullspace", type=int, default=0)
ap.add_argument("--metric-eval", default="rbf", choices=["rbf", "frozen"])
ap.add_argument("--T-touch", type=int, default=0, help="1: rewrite T with its own values after each move")
ap.add_argument("--steps", type=int, default=0, help="time steps in the transport loop")
ap.add_argument("--dt-mult", type=float, default=1.0)
ap.add_argument("--no-move", type=int, default=0, help="transport loop only: never move the mesh")
args = ap.parse_args()


def say(**kw):
    if uw.mpi.rank == 0:
        print("PROBE " + json.dumps(kw, default=str), flush=True)


mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=args.cell,
                          refinement=args.refinement, qdegree=3)
X = mesh.X
r = sympy.sqrt(X[0] ** 2 + X[1] ** 2)
th = sympy.atan2(X[1], X[0])

T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
Tfield = (0.5 * (1 - sympy.tanh((r - 0.9) / 0.03))
          * (1 + 0.3 * sympy.cos(4 * th)))
T.array[:, 0, 0] = uw.function.evaluate(Tfield, T.coords).reshape(-1)


def quality():
    """Folded cells, area spread and wall radius error of the current mesh."""
    dm = mesh.dm
    cells = graph._tri_cells(dm)
    xy = dm.getCoordinatesLocal().array.reshape(-1, 2)
    area = graph._signed_areas(xy, cells)[graph._owned_cell_mask(dm)]
    sign = np.sign(uw.mpi.comm.allreduce(float(area.sum())))
    folded = uw.mpi.comm.allreduce(int(np.count_nonzero(area * sign <= 0)))
    a_min = uw.mpi.comm.allreduce(float(np.abs(area).min()), op=MPI.MIN)
    a_max = uw.mpi.comm.allreduce(float(np.abs(area).max()), op=MPI.MAX)
    radii = np.linalg.norm(xy, axis=1)
    wall = {}
    for lab, R0 in (("Upper", 1.0), ("Lower", 0.5)):
        mask = graph._pinned_mask(dm, (lab,))
        e = float(np.abs(radii[mask] - R0).max()) if mask.any() else 0.0
        wall[lab] = uw.mpi.comm.allreduce(e, op=MPI.MAX)
    return dict(folded=folded, area_ratio=a_max / a_min, wall_radius_err=wall)


def remap_error():
    """How far the remapped T is from its formula, at the wall nodes and inside."""
    exact = uw.function.evaluate(Tfield, T.coords).reshape(-1)
    err = np.abs(np.asarray(T.array).reshape(-1) - exact)
    rad = np.linalg.norm(np.asarray(T.coords), axis=1)
    wall = (np.abs(rad - 1.0) < 1e-9) | (np.abs(rad - 0.5) < 1e-9)
    e_wall = float(err[wall].max()) if wall.any() else 0.0
    e_in = float(err[~wall].max()) if (~wall).any() else 0.0
    return dict(T_err_wall=uw.mpi.comm.allreduce(e_wall, op=MPI.MAX),
                T_err_interior=uw.mpi.comm.allreduce(e_in, op=MPI.MAX))


def mover():
    t0 = time.time()
    rho = uw.meshing.metric_density_from_gradient(
        mesh, T, refinement=args.R, coarsening="auto", metric_choice="front-following")
    uw.meshing.node_redistribution(
        mesh, rho, method_kwargs=dict(step_frac=0.2, accel="cg", momentum=0.0, metric_eval=args.metric_eval),
        slip_surfaces=bool(args.slip), skip_threshold=None)
    a = np.asarray(T.array)
    say(event="move", seconds=time.time() - t0, geometry_version=mesh._geometry_version,
        levels=len(mesh.dm_hierarchy), vr_wall_after_remap=wall_normal_velocity(),
        T_range=[uw.mpi.comm.allreduce(float(a.min()), op=MPI.MIN),
                 uw.mpi.comm.allreduce(float(a.max()), op=MPI.MAX)],
        **remap_error(), **quality())


v = uw.discretisation.MeshVariable("V", mesh, 2, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes.bodyforce = 1.0e3 * T.sym[0] * sympy.Matrix([X[0] / r, X[1] / r])
stokes.tolerance = 1.0e-6
stokes.petsc_use_pressure_nullspace = True

rotated = args.bc.startswith("rot")
for wall in ("Lower", "Upper"):
    if args.bc == "rot-default":
        stokes.add_rotated_freeslip_bc(0, wall)
    elif args.bc == "rot-radial":
        stokes.add_rotated_freeslip_bc(0, wall, normal=mesh.CoordinateSystem.unit_e_0)
    elif args.bc == "nitsche":
        stokes.add_nitsche_bc(0.0, wall, gamma=100.0)
    elif args.bc == "penalty":
        stokes.add_natural_bc(1.0e6 * mesh.Gamma.dot(v.sym) * mesh.Gamma, wall)

if args.rotation_nullspace:
    stokes.petsc_use_nullspace = True

if args.pc == "gamg":
    stokes.preconditioner = "gamg"
elif args.pc == "custom":
    stokes.set_custom_fmg(mesh.multigrid_levels[:-1], field_id=0)


def vrms():
    vol = float(uw.maths.Integral(mesh, sympy.Integer(1)).evaluate())
    return float(np.sqrt(uw.maths.Integral(mesh, v.sym.dot(v.sym)).evaluate() / vol))


def rigid_rotation_rate():
    """Least-squares rigid rotation rate in the velocity: the free-slip null mode."""
    ang = float(uw.maths.Integral(mesh, X[0] * v.sym[1] - X[1] * v.sym[0]).evaluate())
    inertia = float(uw.maths.Integral(mesh, X[0] ** 2 + X[1] ** 2).evaluate())
    return ang / inertia


def rigid_rotation_rate_nodal():
    """The same rate in the nodal inner product, the one the post-solve gauge removal uses."""
    c = np.asarray(v.coords)
    w = np.asarray(v.array).reshape(-1, 2)
    # each node once over the ranks: weight a local copy by 1 / (copies of it)
    iset, subdm = mesh.dm.createSubDM(v.field_id)
    lvec = subdm.createLocalVec()
    gvec = subdm.createGlobalVec()
    lvec.set(1.0)
    gvec.zeroEntries()
    subdm.localToGlobal(lvec, gvec, addv=True)
    subdm.globalToLocal(gvec, lvec)
    copies = lvec.array.reshape(-1, 2)[:, 0].copy()
    for obj in (lvec, gvec, iset, subdm):
        obj.destroy()
    b = uw.mpi.comm.allreduce(float(((c[:, 0] * w[:, 1] - c[:, 1] * w[:, 0]) / copies).sum()))
    a = uw.mpi.comm.allreduce(float(((c * c).sum(axis=1) / copies).sum()))
    return b / a


def wall_normal_velocity():
    """Largest nodal |v.r_hat| on the wall nodes of the velocity space."""
    c = np.asarray(v.coords)
    rad = np.linalg.norm(c, axis=1)
    wall = (np.abs(rad - 1.0) < 1e-9) | (np.abs(rad - 0.5) < 1e-9)
    vr = (np.asarray(v.array).reshape(-1, 2) * c / rad[:, None]).sum(axis=1)
    e = float(np.abs(vr[wall]).max()) if wall.any() else 0.0
    return uw.mpi.comm.allreduce(e, op=MPI.MAX)


def solve(tag):
    t0 = time.time()
    stokes.solve(zero_init_guess=True)
    rec = dict(event="solve", tag=tag, seconds=time.time() - t0, vrms=vrms(),
               omega=rigid_rotation_rate(), omega_nodal=rigid_rotation_rate_nodal(), vr_wall=wall_normal_velocity())
    if rotated:
        info = stokes._rotated_freeslip_info
        rec.update(converged=info.get("converged"), velocity_pc=info.get("velocity_pc"),
                   coarse=info.get("coarse_solve"), vel_its=info.get("vel_its_last"),
                   pres_its=info.get("pres_its_last"),
                   null_modes=getattr(stokes, "_rotated_velocity_null_modes", None),
                   declared_modes_kept=info.get("declared_modes_kept"),
                   gauge_orthonormal=(info.get("rotation_gauge") or {}).get("orthonormal"))
    else:
        snes = stokes.snes
        ksp = snes.getKSP()
        rec.update(reason=snes.getConvergedReason(), snes_its=snes.getIterationNumber(),
                   outer_its=ksp.getIterationNumber(), pc=ksp.getPC().getType())
        try:
            sub = ksp.getPC().getFieldSplitSubKSP()
            rec.update(vel_its=sub[0].getIterationNumber(), vel_pc=sub[0].getPC().getType(),
                       pres_its=sub[1].getIterationNumber())
        except Exception as exc:
            rec.update(subksp=repr(exc))
    fb = getattr(stokes, "_pc_fallbacks", None)
    if fb:
        rec.update(fallbacks=fb)
    say(**rec)


say(event="start", args=vars(args), np=uw.mpi.size, levels=len(mesh.dm_hierarchy),
    cells=uw.mpi.comm.allreduce(mesh.dm.getHeightStratum(0)[1] - mesh.dm.getHeightStratum(0)[0]),
    **quality())
solve("uniform")

def T_bounds():
    a = np.asarray(T.array)
    lo = uw.mpi.comm.allreduce(float(a.min()), op=MPI.MIN)
    hi = uw.mpi.comm.allreduce(float(a.max()), op=MPI.MAX)
    mean = float(uw.maths.Integral(mesh, T.sym[0]).evaluate()) / float(
        uw.maths.Integral(mesh, sympy.Integer(1)).evaluate())
    return dict(T_min=lo, T_max=hi, T_mean=mean)


if args.transport == "none":
    for i in range(args.moves):
        mover()
        if args.T_analytic:
            T.array[:, 0, 0] = uw.function.evaluate(Tfield, T.coords).reshape(-1)
        if args.T_touch:
            T.array[:, 0, 0] = np.array(T.array[:, 0, 0])
        solve(f"moved{i + 1}")
else:
    if args.transport == "supg":
        adv = uw.systems.AdvDiffusion(mesh, T, v.sym, order=1, theta=1.0)
    else:
        adv = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=v.sym, theta=1.0,
                                          monotone_mode="clamp")
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = 1.0e-3
    adv.add_dirichlet_bc(0.0, "Upper")
    adv.add_dirichlet_bc(1.0, "Lower")
    for i in range(args.steps):
        if not args.no_move:
            mover()
        solve(f"step{i}")
        dt = args.dt_mult * float(adv.estimate_dt())
        t0 = time.time()
        adv.solve(timestep=dt, zero_init_guess=False)
        say(event="transport", step=i, dt=dt, seconds=time.time() - t0,
            snes_reason=adv.snes.getConvergedReason(), **T_bounds())
