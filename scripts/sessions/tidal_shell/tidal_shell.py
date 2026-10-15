"""Global tidal response of a thin (visco)elastic shell on a sphere.

The shell floats on an ocean: no basal drag, and its radial motion is the tide's,
w = d/dt [h2 V / g], prescribed as the normal velocity of the sheet (the datum of
the tangency constraint). The sheet solves for the tangential velocity under the
plane-stress membrane closure; the strain rate carries the curvature term w/R P.

Non-dimensional: R = 1, orbital period 1, tidal radial amplitude A (the h2 V/g
scale), shear modulus 1, Maxwell time De (= eta / mu) in periods. Eccentricity
tide, synchronous rotation, degree 2 (Wahr et al. 2009 pattern):

    W(x, t) = A [ -3/2 P20 cos(nt) + 3/4 ( 3 (x^2 - y^2) cos(nt) + 8 x y sin(nt) ) ],
    P20 = (3 z^2 - 1) / 2,  n = 2 pi,

with the sub-planet point on +x and the spin axis +z. Closed form for the viscous
limit on a uniform membrane: a degree-2 radial pattern W Y2 gives the tangential
velocity (3/11) W grad_s Y2, which is checked every step (--check).

Parallel-safe: every diagnostic is a collective reduction; the maps are a
collective evaluation on a longitude-latitude grid drawn on rank 0.
"""
import argparse
import json
import os
import time

import numpy as np
import sympy

import underworld3 as uw
from underworld3.utilities.rotated_bc import SHEET_TANGENCY

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--h", type=float, default=0.15, help="cell size on the unit sphere")
ap.add_argument("--A", type=float, default=0.01, help="radial tidal amplitude (h2 V/g scale)")
ap.add_argument("--De", type=float, default=0.0, help="Maxwell time in periods; 0: viscous sheet")
ap.add_argument("--eta", type=float, default=1.0, help="viscosity of the viscous sheet (De = 0)")
ap.add_argument("--L", type=float, default=0.08, help="shell thickness / R (Enceladus ~ 20/252)")
ap.add_argument("--periods", type=float, default=2.0)
ap.add_argument("--steps_per_period", type=int, default=32)
ap.add_argument("--tol", type=float, default=1e-8)
ap.add_argument("--objective", default="none", help="objective rate of the Maxwell history: none (tides rotate negligibly per step) or jaumann")
ap.add_argument("--check", type=int, default=1, help="compare with the membrane closed form each step")
ap.add_argument("--fig_every", type=int, default=8)
ap.add_argument("--faults", type=int, default=0, help="number of parallel tiger-stripe faults near the south pole")
ap.add_argument("--fault_len", type=float, default=0.52, help="fault length (rad; Enceladus ~130 km on R 252 km)")
ap.add_argument("--fault_spacing", type=float, default=0.14, help="spacing between the stripes (rad; ~35 km)")
ap.add_argument("--fault_eta", type=float, default=0.0, help="viscous interface law coefficient (0: frictionless)")
ap.add_argument("--fault_law", default="frictionless", choices=["frictionless", "viscous", "coulomb"],
                help="coulomb: regularised Coulomb friction with the strength fed by the no-opening reaction")
ap.add_argument("--fault_mu", type=float, default=0.4, help="Coulomb friction coefficient")
ap.add_argument("--fault_V0", type=float, default=1e-3, help="Coulomb regularisation slip rate (well below the tidal slip rates)")
ap.add_argument("--fault_Tb", type=float, default=0.0, help="Coulomb: constant effective overburden as a STRESS (compression positive); enters the law as L * T_b")
ap.add_argument("--probe_stripes", type=int, default=0, help="unfaulted run: record the driving tractions on N virtual stripe traces (Berne et al. Fig 2a,b)")
ap.add_argument("--fault_lat", type=float, default=-90.0, help="latitude of the stripe set's centre (deg)")
ap.add_argument("--fault_lon", type=float, default=0.0, help="longitude of the stripe set's centre (deg; 0 = sub-planet meridian)")
ap.add_argument("--accumulate_displacement", type=int, default=0, help="integrate the tangential velocity on the global and polar grids every step; save the cumulative displacement at figure steps (far-field net deformation per orbit)")
ap.add_argument("--save_global_stress", type=int, default=0, help="save the 3x3 stress on the lat-lon grid every step (npz), for the site/orientation analysis")
ap.add_argument("--fault_azimuth", type=float, default=-90.0, help="stripe direction: angle from the sub-planet direction at the set's centre, counter-clockwise viewed from outside (deg); -90 = east-west")
ap.add_argument("--verbose", action="store_true")
args = ap.parse_args()

comm = uw.mpi.comm
rank0 = uw.mpi.rank == 0
os.makedirs(args.out, exist_ok=True)


def log(*a):
    if rank0:
        print(*a, flush=True)


# ----------------------------------------------------------------------------- mesh and sheet
sphere = uw.meshing.SphericalManifold(radius=1.0, cellSize=args.h, degree=1, qdegree=5)


def stripe_points(offset, half_len, n=9, lat0=-90.0):
    """A stripe as a polyline on the unit sphere: the exponential map at the set's
    centre (latitude lat0 on the x-z meridian), the stripe along the local e1
    direction at a distance `offset` along e2."""
    th0, ph0 = np.radians(lat0), np.radians(args.fault_lon)
    c = np.array([np.cos(th0) * np.cos(ph0), np.cos(th0) * np.sin(ph0), np.sin(th0)])   # centre
    east = np.array([-np.sin(ph0), np.cos(ph0), 0.0])
    sub = np.cross(c, east)                                     # toward the sub-planet point (+x) along the surface
    az = np.radians(args.fault_azimuth)
    e1 = np.cos(az) * sub + np.sin(az) * np.cross(c, sub)       # stripe direction
    e2 = np.cross(c, e1)                                        # across the stripes
    a = np.linspace(-half_len, half_len, n)
    pts = []
    for ai in a:
        rho = np.hypot(ai, offset)
        d = (ai * e1 + offset * e2) / max(rho, 1e-30)
        pts.append(np.cos(rho) * c + np.sin(rho) * d)
    return np.array(pts)


fault_names = []
if args.faults > 0:
    offsets = (np.arange(args.faults) - 0.5 * (args.faults - 1)) * args.fault_spacing
    faults = []
    for i, off in enumerate(offsets):
        name = f"Stripe{i}"
        faults.append((name, stripe_points(off, 0.5 * args.fault_len, lat0=args.fault_lat)))
        fault_names.append(name)
    sphere = sphere.add_fault(faults)
x, y, z = sphere.X
sheet = uw.systems.ThinSheet(sphere)
for name in fault_names:
    if args.fault_law == "coulomb":
        from underworld3.utilities.fault_contact import add_coulomb_fault_bc
        # the sheet's fault reaction is thickness-integrated (a force per unit length,
        # L times the stress), so the overburden given as a stress enters as L * T_b
        add_coulomb_fault_bc(sheet, args.fault_mu, name, sigma_n="reaction", V0=args.fault_V0,
                             background=args.fault_Tb * args.L)
    else:
        sheet.add_fault_bc(args.fault_eta if args.fault_law == "viscous" else 0.0, name)
cum_slip = {name: 0.0 for name in fault_names}          # signed, along the stripe direction
if args.De > 0:
    sheet.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        sheet.Unknowns, order=1, integrator="etd", objective_rate=args.objective)
    sheet.constitutive_model.Parameters.shear_modulus = 1.0
    sheet.constitutive_model.Parameters.shear_viscosity_0 = args.De      # Maxwell time De periods
else:
    sheet.constitutive_model = uw.constitutive_models.ViscousFlowModel
    sheet.constitutive_model.Parameters.shear_viscosity_0 = args.eta
sheet.thickness = args.L
sheet.tolerance = args.tol

n_orb = 2 * sympy.pi
P20 = (3 * z ** 2 - 1) / 2
pattern_c = -sympy.Rational(3, 2) * P20 + sympy.Rational(9, 4) * (x ** 2 - y ** 2)   # multiplies cos(nt)
pattern_s = 6 * x * y                                                                 # multiplies sin(nt)


def radial_velocity(t):
    """w = dW/dt at time t (a sympy expression of position)."""
    c, s = float(np.cos(2 * np.pi * t)), float(np.sin(2 * np.pi * t))
    return args.A * 2 * np.pi * (-pattern_c * s + pattern_s * c)


# closed form of the tangential velocity for the viscous membrane: (3/11) grad_s of the
# degree-2 radial pattern (both patterns are degree-2 harmonics on the unit sphere)
X = sympy.Matrix([x, y, z])
rhat = X / sympy.sqrt(x ** 2 + y ** 2 + z ** 2)
P = sympy.eye(3) - rhat * rhat.T


def grad_s(f):
    return P * sympy.Matrix([f.diff(v) for v in (x, y, z)])


v_c = sympy.Rational(3, 11) * grad_s(pattern_c)
v_s = sympy.Rational(3, 11) * grad_s(pattern_s)

# collective diagnostics: node arrays reduced over ranks
coords = np.asarray(sheet.u.coords)
rh = coords / np.linalg.norm(coords, axis=1)[:, None]
from mpi4py import MPI      # reductions only; the solver handles the rest

# the map grid (collective evaluation, drawn on rank 0)
lon = np.linspace(-np.pi, np.pi, 181)
lat = np.linspace(-np.pi / 2, np.pi / 2, 91)
LON, LAT = np.meshgrid(lon, lat)
grid = np.column_stack([(np.cos(LAT) * np.cos(LON)).ravel(), (np.cos(LAT) * np.sin(LON)).ravel(), np.sin(LAT).ravel()])
# local tangent basis on the grid (east, north) for the in-plane 2x2 stress
_e_east = np.column_stack([-np.sin(LON).ravel(), np.cos(LON).ravel(), np.zeros(LON.size)])
_e_north = np.column_stack([(-np.sin(LAT) * np.cos(LON)).ravel(), (-np.sin(LAT) * np.sin(LON)).ravel(), np.cos(LAT).ravel()])
# the faceted mesh lies inside the sphere (sagitta ~ h^2/8); evaluate just inside it
_grid_in = grid * (1.0 - 0.3 * args.h ** 2)


# The sheet stress holds velocity gradients, which the collective cross-rank
# evaluator cannot take (PETSc error 75 at four ranks on the faulted sphere; the
# same class as the add-on yield indicator): evaluate it rank-locally at the
# sheet's own P1 nodes into six scalar variables and evaluate THOSE collectively.
_S_names = ("xx", "yy", "zz", "xy", "xz", "yz")
_S_idx = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
_S_vars = [uw.discretisation.MeshVariable(f"S{n}", sphere, 1, degree=1) for n in _S_names]
_S_coords = np.asarray(_S_vars[0].coords)
_S_coords_in = _S_coords * (1.0 - 1e-9)


def stress_to_nodes():
    """Fill the six stress-component variables from the sheet stress (every rank
    evaluates its own nodes; the writes are collective)."""
    if _S_coords.shape[0]:
        S = np.asarray(uw.function.evaluate(sheet.stress, _S_coords_in)).reshape(-1, 3, 3)
    else:
        S = np.zeros((0, 3, 3))
    for var, (i, j) in zip(_S_vars, _S_idx):
        var.array[:, 0, 0] = S[:, i, j]


def stress_on(points):
    """The 3x3 sheet stress at arbitrary points, from the nodal components (collective)."""
    comps = [np.asarray(uw.function.global_evaluate(var.sym[0], points)).ravel() for var in _S_vars]
    S = np.zeros((len(points), 3, 3))
    for c, (i, j) in zip(comps, _S_idx):
        S[:, i, j] = c
        S[:, j, i] = c
    return S


# a coarser grid for the per-step stress diagnostics (collective evaluation)
_lon_d = np.linspace(-np.pi, np.pi, 73)[:-1]
_lat_d = np.linspace(-np.pi / 2, np.pi / 2, 37)[1:-1]
_LON_d, _LAT_d = np.meshgrid(_lon_d, _lat_d)
_grid_d = np.column_stack([(np.cos(_LAT_d) * np.cos(_LON_d)).ravel(), (np.cos(_LAT_d) * np.sin(_LON_d)).ravel(),
                           np.sin(_LAT_d).ravel()]) * (1.0 - 0.3 * args.h ** 2)
_w_d = np.cos(_LAT_d).ravel()                                  # area weights
_ee_d = np.column_stack([-np.sin(_LON_d).ravel(), np.cos(_LON_d).ravel(), np.zeros(_LON_d.size)])
_en_d = np.column_stack([(-np.sin(_LAT_d) * np.cos(_LON_d)).ravel(), (-np.sin(_LAT_d) * np.sin(_LON_d)).ravel(),
                         np.cos(_LAT_d).ravel()])
_pat_c = np.asarray(sympy.lambdify((x, y, z), pattern_c, "numpy")(*_grid_d.T)).ravel()
_pat_s = np.asarray(sympy.lambdify((x, y, z), pattern_s, "numpy")(*_grid_d.T)).ravel()


def stress_diagnostics(t):
    """Area-weighted rms of the in-plane stress invariant and the stress trace,
    and the trace's projection on the two forcing patterns (its phase)."""
    S = stress_on(_grid_d)
    s_ee = np.einsum("ni,nij,nj->n", _ee_d, S, _ee_d)
    s_nn = np.einsum("ni,nij,nj->n", _en_d, S, _en_d)
    s_en = 0.5 * (np.einsum("ni,nij,nj->n", _ee_d, S, _en_d) + np.einsum("ni,nij,nj->n", _en_d, S, _ee_d))
    tr = s_ee + s_nn
    inv2 = np.sqrt(0.5 * ((s_ee - 0.5 * tr) ** 2 + (s_nn - 0.5 * tr) ** 2) + s_en ** 2)
    W = _w_d.sum()
    cc, ss = (_w_d * _pat_c ** 2).sum(), (_w_d * _pat_s ** 2).sum()
    return {"stress_inv_rms": float(np.sqrt((_w_d * inv2 ** 2).sum() / W)),
            "stress_tr_rms": float(np.sqrt((_w_d * tr ** 2).sum() / W)),
            "tr_on_cos": float((_w_d * tr * _pat_c).sum() / cc),
            "tr_on_sin": float((_w_d * tr * _pat_s).sum() / ss)}


def diagnostics(t):
    u = np.asarray(sheet.u.array[:, 0, :])
    un = (u * rh).sum(axis=1)
    ut = u - un[:, None] * rh
    w_ex = np.asarray(uw.function.evaluate(radial_velocity(t), coords)).ravel() if coords.shape[0] else np.zeros(0)
    c, s = np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)
    v_ex = np.asarray(uw.function.evaluate(args.A * 2 * np.pi * (-v_c * s + v_s * c), coords)).reshape(-1, 3) if coords.shape[0] else np.zeros((0, 3))
    loc = np.array([float((ut ** 2).sum()), float(((ut - v_ex) ** 2).sum()), float((v_ex ** 2).sum()),
                    float(np.abs(un - w_ex).max()) if un.size else 0.0, float(coords.shape[0])])
    tot = np.zeros_like(loc)
    comm.Allreduce(loc, tot, op=MPI.SUM)
    nrm = comm.allreduce(loc[3], op=MPI.MAX)
    n = max(tot[4], 1.0)
    return {"vt_rms": float(np.sqrt(tot[0] / n)), "vex_rms": float(np.sqrt(tot[2] / n)),
            "rel_err_membrane": float(np.sqrt(tot[1] / max(tot[2], 1e-300))), "normal_datum_err": float(nrm)}


def figure(step, t):
    # collective evaluations on the grid (every rank; drawn on rank 0): a rank-local
    # evaluate() would only know the rank's own cells
    S = stress_on(_grid_in)
    U = np.asarray(uw.function.global_evaluate(sheet.u.sym, _grid_in)).reshape(-1, 3)
    if not rank0:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # the in-plane 2x2 stress in the (east, north) basis and its principal values
    s_ee = np.einsum("ni,nij,nj->n", _e_east, S, _e_east)
    s_nn = np.einsum("ni,nij,nj->n", _e_north, S, _e_north)
    s_en = 0.5 * (np.einsum("ni,nij,nj->n", _e_east, S, _e_north) + np.einsum("ni,nij,nj->n", _e_north, S, _e_east))
    mean, rad = 0.5 * (s_ee + s_nn), np.sqrt((0.5 * (s_ee - s_nn)) ** 2 + s_en ** 2)
    s_max = (mean + rad).reshape(LAT.shape)
    s_min = (mean - rad).reshape(LAT.shape)
    un = (U * grid).sum(axis=1)
    ut = np.linalg.norm(U - un[:, None] * grid, axis=1).reshape(LAT.shape)
    fig, axes = plt.subplots(1, 3, figsize=(19, 4.6), subplot_kw={"projection": "mollweide"})
    for ax, fld, ttl, cmap in ((axes[0], un.reshape(LAT.shape), "radial velocity w (prescribed)", "RdBu_r"),
                               (axes[1], ut, "tangential speed", "viridis"),
                               (axes[2], s_max, "largest in-plane principal stress", "RdBu_r")):
        lim = np.abs(fld).max() + 1e-30
        pc = ax.pcolormesh(LON, LAT, fld, cmap=cmap, shading="auto",
                           **({"vmin": -lim, "vmax": lim} if cmap == "RdBu_r" else {}))
        ax.set_title(ttl, fontsize=10, loc="left"); ax.grid(alpha=0.3)
        fig.colorbar(pc, ax=ax, shrink=0.6, orientation="horizontal", pad=0.05)
    fig.suptitle(f"tidal shell: step {step}, t = {t:.3f} periods, De = {args.De}, A = {args.A}", fontsize=10)
    fig.savefig(os.path.join(args.out, f"tide_{step:04d}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


# polar view: azimuthal (exponential-map) grid around the stripe set's centre
_pole_lat = args.fault_lat if args.faults > 0 else -90.0
_th0, _ph0 = np.radians(_pole_lat), np.radians(args.fault_lon)
_pc = np.array([np.cos(_th0) * np.cos(_ph0), np.cos(_th0) * np.sin(_ph0), np.sin(_th0)])
_sub = np.cross(_pc, np.array([-np.sin(_ph0), np.cos(_ph0), 0.0]))
_az = np.radians(args.fault_azimuth)
_pe1 = np.cos(_az) * _sub + np.sin(_az) * np.cross(_pc, _sub); _pe2 = np.cross(_pc, _pe1)
_prho = np.linspace(0.0, 0.9, 61)[1:]
_ppsi = np.linspace(-np.pi, np.pi, 145)
_PR, _PP = np.meshgrid(_prho, _ppsi)
_pa, _pb = (_PR * np.cos(_PP)).ravel(), (_PR * np.sin(_PP)).ravel()
_pd = (_pa[:, None] * _pe1[None] + _pb[:, None] * _pe2[None]) / np.maximum(_PR.ravel(), 1e-30)[:, None]
_pgrid = np.cos(_PR.ravel())[:, None] * _pc[None] + np.sin(_PR.ravel())[:, None] * _pd
_pgrid_in = _pgrid * (1.0 - 0.3 * args.h ** 2)
_pee = np.cross(_pgrid, _pd)                                   # azimuthal direction
_pee /= np.maximum(np.linalg.norm(_pee, axis=1), 1e-30)[:, None]
_pen = np.cross(_pee, _pgrid)                                  # radial-in-plane direction


def to_polar(points):
    """Points on the sphere -> (a, b) in the azimuthal projection about the centre."""
    pts = np.asarray(points) / np.linalg.norm(points, axis=1)[:, None]
    cosr = np.clip(pts @ _pc, -1, 1)
    rho = np.arccos(cosr)
    d = pts - cosr[:, None] * _pc[None]
    d /= np.maximum(np.linalg.norm(d, axis=1), 1e-30)[:, None]
    return rho * (d @ _pe1), rho * (d @ _pe2)


_fault_polylines = [to_polar(pts) for _n, pts in (faults if args.faults > 0 else [])]

# virtual stripe traces for the driving tractions of an unfaulted (locked) shell
_probe = []
if args.probe_stripes > 0 and args.faults == 0:
    offsets = (np.arange(args.probe_stripes) - 0.5 * (args.probe_stripes - 1)) * args.fault_spacing
    for i, off in enumerate(offsets):
        pts = stripe_points(off, 0.5 * args.fault_len, n=33, lat0=args.fault_lat)
        tt = np.gradient(pts, axis=0); tt -= (tt * pts).sum(1)[:, None] * pts
        tt /= np.maximum(np.linalg.norm(tt, axis=1), 1e-30)[:, None]
        nn = np.cross(pts, tt)
        _probe.append((f"Stripe{i}", pts * (1.0 - 0.3 * args.h ** 2), tt, nn))
    _fault_polylines = [to_polar(pts) for _n, pts, _t, _nn in _probe]


def driving_tractions():
    """Shear (along the stripe, signed) and normal (tension positive) traction on
    each virtual stripe, averaged along it: the locked shell's driving tractions."""
    out = {}
    for name, pts, tt, nn in _probe:
        S = stress_on(pts)
        tau = np.einsum("ni,nij,nj->n", tt, S, nn)
        sn = np.einsum("ni,nij,nj->n", nn, S, nn)
        out[name] = {"tau": float(tau.mean()), "sigma_n": float(sn.mean()), "tau_max": float(np.abs(tau).max())}
    return out


def polar_figure(step, t):
    S = stress_on(_pgrid_in)
    U = np.asarray(uw.function.global_evaluate(sheet.u.sym, _pgrid_in)).reshape(-1, 3)
    if not rank0:
        return
    # the polar fields, for post-processing against a reference run (Coulomb stress changes)
    np.savez(os.path.join(args.out, f"polar_{step:04d}.npz"), t=t, S=S, U=U, grid=_pgrid, rho=_PR, psi=_PP,
             centre=_pc, e1=_pe1, e2=_pe2, faults=np.array([np.column_stack(fp) for fp in _fault_polylines]) if _fault_polylines else np.zeros((0, 0, 2)))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s_ee = np.einsum("ni,nij,nj->n", _pee, S, _pee); s_nn = np.einsum("ni,nij,nj->n", _pen, S, _pen)
    s_en = 0.5 * (np.einsum("ni,nij,nj->n", _pee, S, _pen) + np.einsum("ni,nij,nj->n", _pen, S, _pee))
    mean, rad = 0.5 * (s_ee + s_nn), np.sqrt((0.5 * (s_ee - s_nn)) ** 2 + s_en ** 2)
    s_max = (mean + rad).reshape(_PR.shape); inv = rad.reshape(_PR.shape)
    un = (U * _pgrid).sum(axis=1)
    ut = np.linalg.norm(U - un[:, None] * _pgrid, axis=1).reshape(_PR.shape)
    A_, B_ = (_PR * np.cos(_PP)), (_PR * np.sin(_PP))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    for ax, fld, ttl, cmap, sym in ((axes[0], s_max, "largest in-plane principal stress", "RdBu_r", True),
                                    (axes[1], inv, "in-plane stress invariant (max shear)", "magma", False),
                                    (axes[2], ut, "tangential speed", "viridis", False)):
        lim = np.abs(fld).max() + 1e-30
        pc = ax.pcolormesh(A_, B_, fld, cmap=cmap, shading="auto", **({"vmin": -lim, "vmax": lim} if sym else {}))
        for pa, pb in _fault_polylines:
            ax.plot(pa, pb, color="w" if cmap != "RdBu_r" else "k", lw=2.0)
        for r in (0.25, 0.5, 0.75):
            ax.add_patch(plt.Circle((0, 0), r, fill=False, color="gray", lw=0.5, alpha=0.6))
        ax.set_aspect("equal"); ax.set_xlim(-0.9, 0.9); ax.set_ylim(-0.9, 0.9)
        ax.set_title(ttl, fontsize=10, loc="left"); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(pc, ax=ax, shrink=0.8)
    fig.suptitle(f"south polar view (centre lat {_pole_lat:.0f}, radius 0.9 rad): step {step}, t = {t:.3f} periods, De = {args.De}",
                 fontsize=10)
    fig.savefig(os.path.join(args.out, f"polar_{step:04d}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


# cumulative tangential displacement on the grids (rank 0 holds it; the evaluations are collective)
_disp_g = np.zeros((grid.shape[0], 3)); _disp_p = np.zeros((_pgrid.shape[0], 3))


def accumulate_displacement(dt_):
    global _disp_g, _disp_p
    Ug = np.asarray(uw.function.global_evaluate(sheet.u.sym, _grid_in)).reshape(-1, 3)
    Up = np.asarray(uw.function.global_evaluate(sheet.u.sym, _pgrid_in)).reshape(-1, 3)
    if rank0:
        Ug -= (Ug * grid).sum(1)[:, None] * grid; Up -= (Up * _pgrid).sum(1)[:, None] * _pgrid
        _disp_g += dt_ * Ug; _disp_p += dt_ * Up


def save_displacement(step, t):
    if rank0:
        np.savez(os.path.join(args.out, f"disp_{step:04d}.npz"), t=t, disp_global=_disp_g, grid=grid, lon=LON, lat=LAT,
                 disp_polar=_disp_p, pgrid=_pgrid, rho=_PR, psi=_PP, centre=_pc, e1=_pe1, e2=_pe2,
                 faults=np.array([np.column_stack(fp) for fp in _fault_polylines]) if _fault_polylines else np.zeros((0, 0, 2)))


# ----------------------------------------------------------------------------- time loop
dt = 1.0 / args.steps_per_period
nsteps = int(round(args.periods * args.steps_per_period))
records = []
t = 0.0
log(f"sphere h {args.h}, {comm.allreduce(int(sphere.dm.getHeightStratum(0)[1]), op=MPI.SUM)} cells; "
    f"De {args.De}, A {args.A}, dt {dt}, {nsteps} steps")
for k in range(nsteps):
    t0 = time.perf_counter()
    t_mid = t + dt                                   # implicit: the datum at the end of the step
    sheet.normal_velocity = radial_velocity(t_mid)
    if args.De > 0:
        sheet.solve(timestep=dt, zero_init_guess=(k == 0), verbose=args.verbose)
    else:
        sheet.solve(zero_init_guess=(k == 0), verbose=args.verbose)
    t = t_mid
    info = sheet._rotated_freeslip_info or {}
    rec = {"step": k, "t": t, "dt": dt, "converged": bool(info.get("converged")),
           "nonlinear": info.get("nonlinear_iterations"), "t_step": round(time.perf_counter() - t0, 1)}
    stress_to_nodes()
    if args.accumulate_displacement:
        accumulate_displacement(dt)
        if (k + 1) % args.steps_per_period == 0 or k == nsteps - 1:
            save_displacement(k, t)                      # at every cycle boundary
    if args.save_global_stress:
        Sg = stress_on(_grid_in)                                     # collective
        if rank0:
            np.savez(os.path.join(args.out, f"gstress_{k:04d}.npz"), t=t, S=Sg, lon=LON, lat=LAT)
    if args.check:
        rec.update(diagnostics(t))
        rec.update(stress_diagnostics(t))
    if _probe:
        rec["drive"] = driving_tractions()
    if fault_names:
        from underworld3.utilities.fault_contact import fault_pair_jumps
        slips = {}
        for name in fault_names:
            _c, jumps, normals = fault_pair_jumps(sheet, name, info, gather=True)   # every rank holds the fault
            v = np.linalg.norm(jumps, axis=1) if len(jumps) else np.zeros(1)
            # signed slip along the stripe direction (the stripes run along e1 at the
            # set's centre; its tangent projection at each pair)
            if len(jumps):
                pc_ = np.asarray(_c) / np.linalg.norm(_c, axis=1)[:, None]
                tdir = _pe1[None] - (pc_ @ _pe1)[:, None] * pc_
                tdir /= np.maximum(np.linalg.norm(tdir, axis=1), 1e-30)[:, None]
                signed = float(np.mean(np.einsum("ij,ij->i", jumps, tdir)))
            else:
                signed = 0.0
            cum_slip[name] += signed * dt
            # clamping: the normal traction on the stripe from the no-opening reaction
            # (negative in compression); rank-local per pair, reduced collectively
            from underworld3.utilities.fault_contact import fault_normal_traction
            _cs, sig = fault_normal_traction(sheet, name, info)
            sig = np.asarray(sig) / args.L                    # resultant -> stress (negative in compression)
            loc = np.array([float(np.sum(sig)), float(len(sig)), float(np.min(sig)) if len(sig) else np.inf,
                            float(np.max(sig)) if len(sig) else -np.inf])
            tot = comm.allreduce(loc[0], op=MPI.SUM), comm.allreduce(loc[1], op=MPI.SUM)
            sn_min, sn_max = comm.allreduce(loc[2], op=MPI.MIN), comm.allreduce(loc[3], op=MPI.MAX)
            slips[name] = {"max": float(v.max()), "rms": float(np.sqrt((v ** 2).mean())),
                           "signed_mean": signed, "cumulative": cum_slip[name],
                           "sigma_n_mean": float(tot[0] / max(tot[1], 1.0)), "sigma_n_min": float(sn_min), "sigma_n_max": float(sn_max),
                           "leak": float(np.abs(np.einsum("ij,ij->i", jumps, normals)).max()) if len(jumps) else 0.0}
        rec["slip"] = slips
    records.append(rec)
    log("STEP", json.dumps(rec))
    if k % args.fig_every == 0 or k == nsteps - 1:
        figure(k, t)
        polar_figure(k, t)
        if rank0:
            json.dump(records, open(os.path.join(args.out, "records.json"), "w"), indent=1)
log("DONE")
