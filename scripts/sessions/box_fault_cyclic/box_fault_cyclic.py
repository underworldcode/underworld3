"""A strike-slip fault under cyclic loading: the 3-D box toy model.

Wiki: Box-fault-cyclic-loading. A Maxwell (or viscous) block, x in [0, Lx],
y in [0, Ly], z in [0, H]. The x = 0 face is fixed; the x = Lx face carries a
harmonic traction with a normal component N sin(wt + phi) and a horizontal
shear component S sin(wt); the y faces are traction-free (or held at
u_x = u_z = 0 with --side_bc uxuz0); the base is free-slip; the top is free.

The fault is a vertical plane y = Ly/2 of length l along x, reaching from
depth w below the surface to a setback of a few cells under the top face
(the standing 3-D ruling: a fault reaches daylight blind, under a damage
region, never split through the surface). The sheet is placed with
add_conforming_sheet(setback=...) and split into coincident node pairs; the
damage band above it to the surface is a low-viscosity region located by
the placement's surface trace.

Non-dimensional: shear modulus 1, loading period 1, Maxwell time De periods
(Wallace number Wa = 1/De), tractions as fractions of the shear modulus.

Parallel-safe: every diagnostic is a collective reduction or an all-gathered
pair table; the surface displacement is a collective evaluation on a fixed
grid; files are written from rank 0 only.
"""
import argparse
import json
import os
import time

import numpy as np
import sympy

import underworld3 as uw
from underworld3.utilities.fault_split import split_fault
from underworld3.utilities import fault_contact

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--h", type=float, default=0.15, help="cell size away from the fault")
ap.add_argument("--h_fault", type=float, default=0.05, help="cell size on the fault (gmsh mesher; 0: uniform)")
ap.add_argument("--grading", type=float, default=0.0, help="distance over which the size grades back to --h (0: 8 h_fault)")
ap.add_argument("--mesher", default="gmsh", choices=["gmsh", "adapt", "place"])
ap.add_argument("--adapt_levels", type=int, default=2, help="adapt-on-top levels toward the fault")
ap.add_argument("--clearance", type=float, default=0.9, help="sheet placement clearance (0.9 on a bisection child)")
ap.add_argument("--allow_gamg", action="store_true", help="accept a GAMG velocity block (no multigrid tail); otherwise refused")
ap.add_argument("--Lx", type=float, default=3.0)
ap.add_argument("--Ly", type=float, default=2.0)
ap.add_argument("--H", type=float, default=1.0)
ap.add_argument("--fault_len", type=float, default=1.0, help="fault length along x")
ap.add_argument("--fault_depth", type=float, default=0.5, help="depth of the fault bottom below the surface")
ap.add_argument("--setback", type=float, default=2.0, help="setback of the fault top below the surface, in fault cells (adapt, gmsh) or cells (place)")
ap.add_argument("--damage", type=float, default=0.01,
                help="viscosity ratio in the damage band above the fault top (1: no damage band)")
ap.add_argument("--damage_width", type=float, default=0.0, help="half-width of the damage band (0: one cell)")
ap.add_argument("--De", type=float, default=1.0, help="Maxwell time in periods (0: viscous block with --eta)")
ap.add_argument("--eta", type=float, default=1.0)
ap.add_argument("--S", type=float, default=0.01, help="shear traction amplitude on the loaded face")
ap.add_argument("--N", type=float, default=0.01, help="normal traction amplitude on the loaded face")
ap.add_argument("--phi", type=float, default=80.0, help="phase of the normal traction ahead of the shear (deg)")
ap.add_argument("--fault_law", default="coulomb", choices=["frictionless", "viscous", "coulomb"])
ap.add_argument("--fault_eta", type=float, default=0.0)
ap.add_argument("--fault_mu", type=float, default=0.4)
ap.add_argument("--fault_V0", type=float, default=1e-3)
ap.add_argument("--fault_Tb", type=float, default=0.0, help="constant effective overburden on the fault (compression positive)")
ap.add_argument("--side_bc", default="free", choices=["free", "uxuz0"])
ap.add_argument("--periods", type=float, default=2.0)
ap.add_argument("--steps_per_period", type=int, default=32)
ap.add_argument("--tol", type=float, default=1e-6)
ap.add_argument("--picard", type=int, default=0)
ap.add_argument("--stress_transport", default="none", choices=["none", "eulerian", "semi_lagrangian", "integration_point"],
                help="flavour of the Maxwell stress history. Default none: the displacement per step is far below a cell, so transport is a cost not physics, and a traced-back query across the split fault reads the other side")
ap.add_argument("--surface_grid", type=int, default=40, help="points along x of the surface displacement grid")
ap.add_argument("--verbose", action="store_true")
args = ap.parse_args()

from mpi4py import MPI                          # noqa: E402  (collective bookkeeping only)
comm = MPI.COMM_WORLD
rank0 = comm.rank == 0
if rank0:
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)


def log(*a):
    if rank0:
        print(*a, flush=True)


# ----------------------------------------------------------------------------- mesh, fault, split
# Three routes to a blind vertical fault whose rim is strictly inside the box:
#  gmsh  — BoxInternalPatch: the rectangle is a conforming patch of a gmsh mesh graded
#          to the fault. No cavity, so a two-cell setback works. The mesh has no
#          multigrid tail of its own, so two plain boxes at 2h and 4h are REGISTERED
#          as the coarse tail (set_custom_fmg, non-nested, barycentric transfers) and
#          the rotated solve takes the custom-P FMG velocity block (measured serially:
#          3 velocity, 12 pressure iterations, converged). Default.
#  adapt — a box with one refinement level, adapt-on-top toward the fault column, the
#          sheet placed by carving a cavity, the patch split; the tail rides through
#          by itself. The cavity reaches ~2 cell diameters above the sheet top, so
#          the setback must be >= 5 fine cells (0.20 refused, 0.25 passed), and the
#          parallel placement currently under-reaches its gather on this geometry
#          (TODO(BUG) in place_surface.py; serial passes).
#  place — a uniform box, cavity placement with setback (needs three to four cells).
h = args.h
hf = args.h_fault if args.h_fault > 0 else h
yf = 0.5 * args.Ly
xa, xb = 0.5 * (args.Lx - args.fault_len), 0.5 * (args.Lx + args.fault_len)
zb = args.H - args.fault_depth
setback = args.setback * (h if args.mesher == "place" else hf)
ztop_fault = args.H - setback
trace = [[xa, yf, args.H], [xb, yf, args.H]]           # the would-be surface trace (a straight fault)


def fault_distance(pts):
    """Exact distance from points to the fault plane's column (axis-aligned rectangle in
    the plane y = yf from the fault bottom up THROUGH the free surface): the refinement
    must reach the surface above the fault, both for the damage band and so the
    placement's cavity (clearance x local h, plus the dropped cells) stays clear of
    the top face — a coarse cell above the fault top reached it (measured)."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    dx = np.maximum(np.maximum(xa - pts[:, 0], pts[:, 0] - xb), 0.0)
    dz = np.maximum(zb - pts[:, 2], 0.0)
    dy = np.abs(pts[:, 1] - yf)
    return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


def fault_metric(pts):
    """M = 1/h^2: h_fault within one fault cell of the plane, grading linearly to h."""
    d = fault_distance(pts)
    width = args.grading if args.grading > 0 else 6.0 * hf
    hh = np.where(d < hf, hf, np.minimum(hf + (h - hf) * (d - hf) / width, h))
    return 1.0 / hh ** 2


def sheet_triangulation(cell):
    """The fault rectangle as a triangulated sheet of size `cell`, run past the top so the
    placement clips it and stops it `setback` below the surface."""
    nx = max(2, int(round(args.fault_len / cell)) + 1)
    nz = max(2, int(round((args.H + 0.3 * cell - zb) / cell)) + 1)
    xs = np.linspace(xa, xb, nx)
    zs = np.linspace(zb, args.H + 0.3 * cell, nz)
    XX, ZZ = np.meshgrid(xs, zs, indexing="ij")
    pts = np.column_stack([XX.ravel(), np.full(XX.size, yf), ZZ.ravel()])
    tris = []
    for i in range(nx - 1):
        for j in range(nz - 1):
            a, b, c, d = i * nz + j, (i + 1) * nz + j, (i + 1) * nz + j + 1, i * nz + j + 1
            tris += [[a, b, c], [a, c, d]]
    return pts, np.array(tris, dtype=np.int64)


t0 = time.perf_counter()
if args.mesher == "gmsh":
    patch = np.array([[xa, yf, zb], [xb, yf, zb], [xb, yf, ztop_fault], [xa, yf, ztop_fault]])
    cut = uw.meshing.BoxInternalPatch(cellSize=h, minCoords=(0.0, 0.0, 0.0), maxCoords=(args.Lx, args.Ly, args.H),
                                      patch_points=patch, patch_name="Fault",
                                      patch_cellSize=hf if hf < h else None,
                                      grading_distance=args.grading if args.grading > 0 else None,
                                      qdegree=3)
elif args.mesher == "adapt":
    base = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0, 0.0), maxCoords=(args.Lx, args.Ly, args.H),
                                              cellSize=2.0 * h, regular=False, qdegree=3, refinement=1)
    child = base.adapt(fault_metric, max_levels=args.adapt_levels, verbose=args.verbose)
    n_tail = len(getattr(child, "_custom_mg_coarse_meshes", None) or [])
    log(f"adapt: {comm.allreduce(int(child.dm.getHeightStratum(0)[1] - child.dm.getHeightStratum(0)[0]), op=MPI.SUM)} cells, "
        f"multigrid tail of {n_tail} coarse levels")
    pts, tris = sheet_triangulation(hf)
    cut = child.add_conforming_sheet(pts, tris, "Fault", setback=setback, clearance=args.clearance,
                                     verbose=args.verbose)
else:
    base = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0, 0.0), maxCoords=(args.Lx, args.Ly, args.H),
                                              cellSize=h, regular=False, qdegree=3)
    pts, tris = sheet_triangulation(h)
    cut = base.add_conforming_sheet(pts, tris, "Fault", setback=setback, verbose=args.verbose)
if hasattr(cut, "_surface_info") and cut._surface_info.get("surface_trace"):
    trace = cut._surface_info["surface_trace"]
mesh = split_fault(cut, "Fault")
n_tail = len(getattr(mesh, "_custom_mg_coarse_meshes", None) or [])
ncell = comm.allreduce(int(mesh.dm.getHeightStratum(0)[1] - mesh.dm.getHeightStratum(0)[0]), op=MPI.SUM)
log(f"mesh ({args.mesher}) h {h} h_fault {hf}: {ncell} cells after fault + split ({time.perf_counter() - t0:.1f}s); "
    f"fault x [{xa:.2f}, {xb:.2f}], z [{zb:.2f}, {ztop_fault:.3f}], setback {setback:.3f}; "
    f"multigrid tail {n_tail} levels")
x, y, z = mesh.X

# ----------------------------------------------------------------------------- solver
v = uw.discretisation.MeshVariable("v", mesh, 3, degree=2)
p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.bodyforce = sympy.Matrix([0.0, 0.0, 0.0])
stokes.tolerance = args.tol

# the damage band: a low-viscosity region from the fault top to the surface,
# the width of a cell (or --damage_width) either side of the trace
wd = args.damage_width if args.damage_width > 0 else hf
ztop = ztop_fault - 0.5 * wd
g_y = sympy.exp(-((y - yf) / wd) ** 2)
g_z = 0.5 * (1 + sympy.tanh((z - ztop) / (0.5 * wd)))
g_x = 0.25 * (1 + sympy.tanh((x - xa) / (0.5 * wd))) * (1 + sympy.tanh((xb - x) / (0.5 * wd)))
damage = g_x * g_y * g_z                                   # 1 in the band, 0 away from it
eta_factor = 1.0 - (1.0 - args.damage) * damage if args.damage < 1.0 else 1.0

if args.De > 0:
    stokes.stress_transport = args.stress_transport          # before the model: it creates the history
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=1, integrator="etd", objective_rate="none")
    stokes.constitutive_model.Parameters.shear_modulus = 1.0
    stokes.constitutive_model.Parameters.shear_viscosity_0 = args.De * eta_factor
else:
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = args.eta * eta_factor

# the harmonic end load: two uniform P1 fields hold the instantaneous amplitudes so the
# traction changes each step without recompiling the residual
N_t = uw.discretisation.MeshVariable("N_t", mesh, 1, degree=1)
S_t = uw.discretisation.MeshVariable("S_t", mesh, 1, degree=1)
stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Left")                      # x = 0 fixed
# sign convention (measured): a positive value in add_natural_bc pushes INTO the domain, so
# the traction sigma.n = (N, S, 0) on the outward normal +x enters negated: N > 0 is tension
# (the loaded end moves +x), S > 0 is shear traction in +y on the x = Lx face.
stokes.add_natural_bc(sympy.Matrix([[-N_t.sym[0], -S_t.sym[0], 0.0]]), "Right")
stokes.add_dirichlet_bc((None, None, 0.0), "Bottom")                  # free-slip base
if args.side_bc == "uxuz0":
    stokes.add_dirichlet_bc((0.0, None, 0.0), "Front")
    stokes.add_dirichlet_bc((0.0, None, 0.0), "Back")

if args.mesher == "gmsh":
    from underworld3.utilities.custom_mg import set_custom_fmg
    coarse_tail = [uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0, 0.0), maxCoords=(args.Lx, args.Ly, args.H),
                                                      cellSize=c, regular=False, qdegree=3) for c in (4.0 * h, 2.0 * h)]
    set_custom_fmg(stokes, coarse_tail, field_id=0)
    log(f"coarse tail registered: {[comm.allreduce(int(m.dm.getHeightStratum(0)[1] - m.dm.getHeightStratum(0)[0]), op=MPI.SUM) for m in coarse_tail]} cells")

if args.fault_law == "coulomb":
    fault_contact.add_coulomb_fault_bc(stokes, args.fault_mu, "Fault", sigma_n="reaction",
                                       V0=args.fault_V0, background=args.fault_Tb)
else:
    stokes.add_fault_bc(args.fault_eta if args.fault_law == "viscous" else 0.0, "Fault")


load_now = {"N": 0.0, "S": 0.0}


def set_load(t):
    w = 2 * np.pi
    load_now["N"] = args.N * np.sin(w * t + np.radians(args.phi))
    load_now["S"] = args.S * np.sin(w * t)
    N_t.array[...] = load_now["N"]
    S_t.array[...] = load_now["S"]


# ----------------------------------------------------------------------------- diagnostics
# surface displacement on a fixed grid (collective evaluation), accumulated over the run
gx = np.linspace(0.02 * args.Lx, 0.98 * args.Lx, args.surface_grid)
gy = np.linspace(0.02 * args.Ly, 0.98 * args.Ly, max(4, int(round(args.surface_grid * args.Ly / args.Lx))))
GX, GY = np.meshgrid(gx, gy, indexing="ij")
surf_pts = np.column_stack([GX.ravel(), GY.ravel(), np.full(GX.size, args.H - 1e-6)])
disp = np.zeros((surf_pts.shape[0], 3))
disp_cycle = np.zeros_like(disp)

pair_tab = None          # coords, cumulative signed slip, per-cycle slip, keyed by the pair order


def fault_diagnostics(info, dt):
    """Signed strike slip (v+ - v-).x̂ at every pair, all-gathered; cumulative and per cycle."""
    global pair_tab
    coords, jumps, normals = fault_contact.fault_pair_jumps(stokes, "Fault", info, gather=True)
    if pair_tab is None:
        pair_tab = {"coords": coords, "cum": np.zeros(len(coords)), "cycle": np.zeros(len(coords)),
                    "act": np.zeros(len(coords))}
    leak = np.einsum("ij,ij->i", jumps, normals)
    rate = jumps[:, 0]                                     # strike direction is x
    pair_tab["cum"] += rate * dt
    pair_tab["cycle"] += rate * dt
    pair_tab["act"] += np.abs(rate) * dt
    _c, sig = fault_contact.fault_normal_traction(stokes, "Fault", info)
    sig = np.asarray(sig)
    sig_all = np.concatenate(comm.allgather(sig.ravel())) if comm.size > 1 else sig.ravel()
    return {"slip_rate_mean": float(rate.mean()) if len(rate) else 0.0,
            "slip_rate_absmax": float(np.abs(rate).max()) if len(rate) else 0.0,
            "leak_max": float(np.abs(leak).max()) if len(leak) else 0.0,
            "sigma_n_mean": float(sig_all.mean()) if len(sig_all) else 0.0,
            "sigma_n_min": float(sig_all.min()) if len(sig_all) else 0.0,
            "sigma_n_max": float(sig_all.max()) if len(sig_all) else 0.0,
            "n_pairs": int(len(rate))}


def save_cycle(step, t):
    if rank0:
        np.savez(os.path.join(args.out, f"cycle_{step:04d}.npz"), t=t,
                 pair_coords=pair_tab["coords"], slip_cum=pair_tab["cum"], slip_cycle=pair_tab["cycle"],
                 activity=pair_tab["act"], surf_pts=surf_pts, disp=disp, disp_cycle=disp_cycle,
                 trace=np.array(trace, dtype=float))
    pair_tab["cycle"][:] = 0.0
    disp_cycle[:] = 0.0


# ----------------------------------------------------------------------------- time loop
dt = 1.0 / args.steps_per_period
nsteps = int(round(args.periods * args.steps_per_period))
records = []
t = 0.0
log(f"De {args.De}, S {args.S}, N {args.N}, phi {args.phi}, law {args.fault_law} mu {args.fault_mu} Tb {args.fault_Tb}; "
    f"dt {dt}, {nsteps} steps")
for k in range(nsteps):
    tk = time.perf_counter()
    t_new = t + dt
    set_load(t_new)                                        # implicit: the load at the end of the step
    if args.De > 0:
        stokes.solve(timestep=dt, zero_init_guess=(k == 0), verbose=args.verbose, picard=args.picard)
    else:
        stokes.solve(zero_init_guess=(k == 0), verbose=args.verbose, picard=args.picard)
    t = t_new
    info = getattr(stokes, "_rotated_freeslip_info", None) or {}
    if k == 0:
        vpc = info.get("velocity_pc")
        log(f"velocity block: {vpc}; velocity its (last) {info.get('vel_its_last')}, pressure its (last) {info.get('pres_its_last')}")
        if vpc == "GAMG" and not args.allow_gamg:
            raise SystemExit("the velocity block fell back to GAMG: the mesh carries no multigrid tail. "
                             "Use --mesher adapt (or pass --allow_gamg to accept it).")
    rec = {"step": k, "t": t, "converged": bool(info.get("converged", True)),
           "nonlinear": info.get("nonlinear_iterations"), "t_step": round(time.perf_counter() - tk, 1)}
    rec.update(fault_diagnostics(info, dt))
    U = np.asarray(uw.function.global_evaluate(v.sym, surf_pts)).reshape(-1, 3)
    disp += U * dt
    disp_cycle += U * dt
    rec["surf_ux_absmax"] = float(np.abs(U[:, 0]).max())
    right = surf_pts[:, 0] > 0.9 * args.Lx
    rec["surf_ux_right"] = float(U[right, 0].mean())             # signed: +N on the loaded face must pull it out (+x)
    rec["load_N"], rec["load_S"] = float(load_now["N"]), float(load_now["S"])
    records.append(rec)
    log(f"step {k:4d} t {t:.4f} its {rec['nonlinear']} {rec['t_step']}s | slip rate mean {rec['slip_rate_mean']:+.2e} "
        f"max {rec['slip_rate_absmax']:.2e} leak {rec['leak_max']:.1e} | sigma_n [{rec['sigma_n_min']:+.3f}, {rec['sigma_n_max']:+.3f}]"
        f" | load N {rec['load_N']:+.4f} S {rec['load_S']:+.4f} | surface u_x(right) {rec['surf_ux_right']:+.2e}")
    if (k + 1) % args.steps_per_period == 0 or k == nsteps - 1:
        cyc = pair_tab["cycle"]
        log(f"  cycle {(k + 1) // args.steps_per_period}: net slip mean {cyc.mean():+.3e}, max {cyc.max():+.3e} min {cyc.min():+.3e}; "
            f"cumulative mean {pair_tab['cum'].mean():+.3e}; surface net |u| max {np.linalg.norm(disp_cycle, axis=1).max():.2e}")
        save_cycle(k, t)
    if rank0:
        with open(os.path.join(args.out, "records.json"), "w") as f:
            json.dump(records, f)
log("done")
