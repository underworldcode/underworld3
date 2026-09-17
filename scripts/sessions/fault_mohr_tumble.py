"""Mohr diagrams sampled by randomly tumbling welded faults, 2-D and 3-D.

A homogeneous, traceless strain rate :math:`E` is imposed as Dirichlet
velocity :math:`v = E\\,(x - c)` on every wall of a closed box. For a
Newtonian fluid this is an exact Stokes solution with uniform deviatoric
stress :math:`\\sigma' = 2\\eta E`, so every point sees the same stress.

A WELDED split-node fault through the box centre is a passive stress probe:
the strong no-opening constraint's reaction is the normal traction
:math:`\\sigma_n`, and the stiff interface dashpot gives the shear traction
vector :math:`\\tau = \\eta_f [v]_t`. Each sample draws a random fault
orientation — a uniform angle in 2-D, a normal uniform on the sphere in
3-D — and builds its mesh from ONE static base, with the model machinery:

* ``base.adapt(metric)`` refines toward the fault from its exact distance;
* 2-D: ``child.add_fault`` cuts and splits the segment;
* 3-D: ``child.add_conforming_sheet`` places a disc, ``split_fault`` splits it.

Every sample is checkpointed on its own — the mesh and fields through
``write_timestep``, and the probe record (tractions, the analytic values,
per-node traction along the fault) beside it — and a sample already on disk
is skipped, so a sweep can be extended, sharded across processes, or
restarted, and ``fault_mohr_plot.py`` re-renders without re-solving.

Run one shard per process (samples are independent; the draw is shared)::

    for i in $(seq 0 23); do
      python scripts/sessions/fault_mohr_tumble.py -uw_dim 3 -uw_n_faults 480 \\
          -uw_shard $i -uw_n_shards 24 > shard_$i.log 2>&1 &
    done; wait
"""
import os
import time

import numpy as np
import sympy

import underworld3 as uw
from underworld3.utilities import fault_contact
from underworld3.utilities.fault_split import split_fault

params = uw.Params(
    uw_dim=2,
    uw_n_faults=64,
    uw_seed=20260915,           # one draw, identical in every shard
    uw_shard=0,
    uw_n_shards=1,
    # Stokes solve tolerance. The loading is an exact P2 state, so the probe's
    # error is the weld compliance, not the solve: 1e-3 reproduces the 1e-6
    # nodal tractions to 6e-6 at a fifth of the cost (measured, 3-D sample 0)
    uw_tolerance=1e-3,
    uw_penalty=0.0,             # augmented-Lagrangian grad-div on the velocity block
    uw_al_schur=0,              # 1: rescale the Schur preconditioner to 1/(eta(1+penalty))
    uw_friction=0.0,            # > 0: Coulomb friction coefficient mu, and the
                                # fault is allowed to FAIL (0 = welded probe)
    uw_cohesion=0.0,            # cohesion C: lifts the envelope to C + mu sigma,
                                # so fewer orientations fail and the forbidden
                                # zone shrinks. It also keeps strength into MILD
                                # tension (to sigma = -C/mu), which is where a
                                # cohesionless fault would otherwise be held
                                # shut by the no-opening constraint unphysically
    uw_slip_v0=1.0e-3,          # friction regularisation velocity: below it the
                                # fault sticks, above it slides at mu sigma_n
    uw_base_size=0.0,           # resolution overrides; 0 keeps the per-dimension
    uw_h_near=0.0,              # defaults below
    uw_levels=0,
    uw_ramp=0.0,                # how far the fine band reaches from the fault;
                                # 0 keeps the default below
    uw_trajectory=0,            # 1: orientations along a PATH, not scattered,
                                # so the Mohr point walks a trajectory
    uw_newton=0,                # 1: consistent Newton tangent; 2: "continuation"
                                # (staged Picard -> Newton). 0 keeps the solver
                                # default, which is the FROZEN tangent — note
                                # that picard>0 only adds a warm-up to that, it
                                # does not select Newton.
    uw_picard=2,                # Picard passes over the lagged interface state.
                                # A SLIDING fault is insensitive (the law
                                # saturates), a stuck one is not: its traction
                                # is read off the law at the achieved slip rate
    uw_output_dir=os.path.expanduser("~/+Simulations/fault_mohr_triaxial"),
)

DIM = params.uw_dim
ETA = 1.0
CENTRE = np.full(DIM, 0.5)
RADIUS = 0.2                    # half-length (2-D) or disc radius (3-D)
# the weld scale eta/a slips at about half the free rate; 200x that is
# effectively welded while the dashpot traction stays well conditioned
ETA_WELD = 200.0 * ETA / RADIUS
# tip and rim tractions carry the crack singularity; the central part of
# the fault is the gauge
CENTRAL_FRACTION = 0.6

# Mesh resolution and loading per dimension. The base is built once with one
# uniform refinement (the coarse tail adapt extends); adapt grades from
# h_far on the base finest down to h_near on the fault.
RESOLUTION = {2: dict(base_size=0.1, h_near=0.0125, levels=2),
              3: dict(base_size=0.2, h_near=0.05, levels=1)}[DIM]
# ... overridable, so a run can be made finer without editing the script. The
# base is built once per process and every orientation adapts onto it, so
# base_size sets the far field and h_near the fault band.
for _key, _override in (("base_size", params.uw_base_size),
                        ("h_near", params.uw_h_near),
                        ("levels", params.uw_levels)):
    if _override > 0:
        RESOLUTION[_key] = _override
# How far the fine band reaches from the fault. The dense region should hug
# the fault and nothing else: measured at h_near 0.0125 with three levels, a
# band of 0.4 (the old 4*h_far, most of a box of side 1) costs 90,015 cells
# while 0.1 costs 42,505 — 53% fewer for the SAME resolution on the fault.
RAMP = params.uw_ramp if params.uw_ramp > 0 else RESOLUTION["base_size"] / 2
# Principal strain rates, most shortening first; traceless. 3-D is
# deliberately triaxial (distinct intermediate) so the three circles differ.
PRINCIPAL_RATES = {2: np.array([-0.5, 0.5]),
                   3: np.array([-0.5, 0.1, 0.4])}[DIM]
WALLS = {2: ("Bottom", "Top", "Left", "Right"),
         3: ("Bottom", "Top", "Left", "Right", "Front", "Back")}[DIM]


def principal_axes():
    """Rotation whose columns are the principal directions: off the box axes
    so nothing about the result can come from the mesh alignment."""
    if DIM == 2:
        phi = np.radians(22.5)
        return np.array([[np.cos(phi), -np.sin(phi)],
                         [np.sin(phi), np.cos(phi)]])
    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    angle = np.radians(35.0)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def random_normals(n):
    rng = np.random.default_rng(params.uw_seed)
    if DIM == 2:
        theta = rng.uniform(0.0, np.pi, n)
        return np.column_stack([-np.sin(theta), np.cos(theta)])
    g = rng.normal(size=(n, 3))
    return g / np.linalg.norm(g, axis=1)[:, None]


def in_plane_basis(normal):
    """Unit vectors spanning the fault plane (one tangent in 2-D)."""
    if DIM == 2:
        return np.array([[normal[1], -normal[0]]])
    helper = np.eye(3)[np.argmin(np.abs(normal))]
    e1 = np.cross(normal, helper)
    e1 /= np.linalg.norm(e1)
    return np.array([e1, np.cross(normal, e1)])


def trajectory_normals(n_samples):
    """Orientations along a PATH rather than scattered over the sphere.

    A random tumble hops about the Mohr diagram; a trajectory walks it, which
    is what shows a fault approaching a failure envelope and crossing into
    yield. In 2-D the fault plane turns through a half-turn, so the point
    travels once around the circle. In 3-D the normal turns in the
    sigma_1-sigma_3 plane: the path around the LARGEST circle, the one that
    reaches a failure envelope first.
    """
    angles = np.linspace(0.0, np.pi, n_samples, endpoint=False)
    if DIM == 2:
        return np.column_stack([np.cos(angles), np.sin(angles)])
    directions = principal_axes()
    most, least = directions[:, 0], directions[:, -1]
    return np.array([np.cos(a) * most + np.sin(a) * least for a in angles])


def bulk_traction(mesh, velocity, normal):
    """The shear traction the MEDIUM applies, read off the bulk field a little
    away from the fault surface.

    Valid while the fault is STUCK: the field is then near-uniform and the
    reading is stable (welded control, measured: 0.0303 against an exact
    0.0304, and flat from half a cell out to four). It is NOT valid once the
    fault slides — the split-node layer's own gradient is decades above the
    far field and swamps it (measured across half a cell: +8.6, -0.96, +0.37,
    with the two sides disagreeing) — so the caller uses this only where the
    slip rate says the fault is stuck.

    TODO(FEATURE): ``fault_contact`` recovers the NORMAL traction from the
    constraint reaction but has no tangential counterpart, which is what a
    frictional fault's Mohr point actually needs. With one, both regimes would
    be the same measurement instead of two with a switch between them.
    """
    x = mesh.X
    grad = sympy.Matrix(DIM, DIM, lambda i, j: velocity.sym[i].diff(x[j]))
    edot = (grad + grad.T) / 2
    dev = 2 * ETA * (edot - edot.trace() / DIM * sympy.eye(DIM))
    basis = in_plane_basis(normal)
    reach = CENTRAL_FRACTION * RADIUS
    if DIM == 2:
        offsets = np.outer(np.linspace(-reach, reach, 21), basis[0])
    else:
        grid = np.linspace(-reach, reach, 7)
        a, b = np.meshgrid(grid, grid)
        keep = (a ** 2 + b ** 2) <= reach ** 2
        offsets = np.outer(a[keep], basis[0]) + np.outer(b[keep], basis[1])
    sides = []
    for side in (+1.0, -1.0):
        points = CENTRE + offsets + side * 1.5 * RESOLUTION["h_near"] * normal
        sigma = np.asarray(uw.function.evaluate(dev, points)).reshape(-1, DIM, DIM)
        traction = np.einsum("kij,j->ki", sigma, normal)
        sides.append(np.median(traction - np.outer(traction @ normal, normal),
                               axis=0))
    return np.mean(sides, axis=0)


def fault_distance(X, normal):
    """Exact distance to the centred segment (2-D) or disc (3-D)."""
    rel = np.asarray(X)[:, :DIM] - CENTRE
    zn = rel @ normal
    r = np.linalg.norm(rel - zn[:, None] * normal, axis=1)
    return np.sqrt(np.maximum(r - RADIUS, 0.0) ** 2 + zn ** 2)


def refinement_metric(normal):
    h_near, h_far = RESOLUTION["h_near"], RESOLUTION["base_size"] / 2
    ramp = RAMP

    def metric(X):
        h = np.clip(h_near + fault_distance(X, normal) * (h_far - h_near) / ramp,
                    h_near, h_far)
        return 1.0 / h**2

    return metric


def disc_sheet(normal):
    """A planar disc as a centre fan; ``size=`` re-triangulates it to the mesh."""
    e1, e2 = in_plane_basis(normal)
    n_rim = int(np.ceil(2.0 * np.pi * RADIUS / RESOLUTION["h_near"]))
    a = np.linspace(0.0, 2.0 * np.pi, n_rim, endpoint=False)
    rim = CENTRE + RADIUS * (np.outer(np.cos(a), e1) + np.outer(np.sin(a), e2))
    tris = [(0, 1 + i, 1 + (i + 1) % n_rim) for i in range(n_rim)]
    return np.vstack([CENTRE, rim]), np.array(tris, dtype=np.int64)


def faulted_mesh(base, normal):
    """The split mesh for one orientation, and the seconds each stage took."""
    timings = {}
    t0 = time.time()
    child = base.adapt(refinement_metric(normal), max_levels=RESOLUTION["levels"])
    timings["t_adapt"] = time.time() - t0
    t0 = time.time()
    if DIM == 2:
        tangent = in_plane_basis(normal)[0]
        split = child.add_fault(("Fault", np.array([CENTRE - RADIUS * tangent,
                                                    CENTRE + RADIUS * tangent])))
        timings["t_fault"] = time.time() - t0
        return split, timings
    pts, tris = disc_sheet(normal)
    placed = child.add_conforming_sheet(pts, tris, "Fault",
                                        size=RESOLUTION["h_near"])
    timings["t_place"] = time.time() - t0
    t0 = time.time()
    split = split_fault(placed, "Fault")
    timings["t_split"] = time.time() - t0
    return split, timings


def central_mask(positions, normal):
    """Pairs within CENTRAL_FRACTION of the radius of the fault centre."""
    rel = positions - CENTRE
    r = np.linalg.norm(rel - (rel @ normal)[:, None] * normal, axis=1)
    return r < CENTRAL_FRACTION * RADIUS


def fault_probe(mesh, strain_rate, normal, tag):
    """One solve: the fault's own normal and shear traction.

    Welded by default (a passive stress probe that barely slips). With
    ``-uw_friction`` the fault is allowed to FAIL, and the shear traction is
    whatever the friction law holds it to.
    """
    v = uw.discretisation.MeshVariable(f"V{tag}", mesh, DIM, degree=2)
    p = uw.discretisation.MeshVariable(f"P{tag}", mesh, 1, degree=0,
                                       continuous=False)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = ETA
    stokes.bodyforce = [0.0] * DIM

    x = mesh.X
    drive = tuple(sum(strain_rate[i, j] * (x[j] - CENTRE[j]) for j in range(DIM))
                  for i in range(DIM))
    for wall in WALLS:
        stokes.add_dirichlet_bc(drive, wall)
    stokes.petsc_use_pressure_nullspace = True
    if params.uw_newton:
        # The solver default is the FROZEN tangent, and picard>0 only warms that
        # up — it does not select Newton. The nonlinearity here lives in the
        # interface law, so the consistent tangent is the one that matches it.
        stokes.consistent_jacobian = ("continuation" if params.uw_newton == 2
                                      else True)
    stokes.penalty = params.uw_penalty
    if params.uw_al_schur:
        # The grad-div augmentation changes the Schur complement, so the mass
        # preconditioner has to be rescaled with it; left at the default 1/eta
        # it preconditions the UNaugmented operator (measured: pressure
        # iterations rise with the penalty instead of falling).
        stokes.saddle_preconditioner = 1.0 / (ETA * (1.0 + params.uw_penalty))
    stokes.tolerance = params.uw_tolerance
    if params.uw_friction > 0.0 and params.uw_cohesion > 0.0:
        # A COHESIVE Coulomb law, written from the documented pieces:
        # fault_contact ships mu*Max(sigma, 0) only, while its own note on
        # `normal_stress` says a law must clamp its OWN strength — "e.g.
        # Max(0, C + mu*normal_stress)" — which is exactly this.
        # TODO(FEATURE): there is no public cohesive Coulomb constructor and no
        # public way to register a SymbolicFaultLaw (`_register_law` is
        # private), so a cohesive fault cannot be set up through the API.
        strength = sympy.Max(0.0, params.uw_cohesion
                             + params.uw_friction * fault_contact.normal_stress)
        fault_contact.add_frictionless_fault_bc(stokes, "Fault")
        fault_contact._register_law(stokes, "Fault", fault_contact.SymbolicFaultLaw(
            strength * (2 / sympy.pi)
            * sympy.atan(fault_contact.slip_rate / params.uw_slip_v0)))
    elif params.uw_friction > 0.0:
        fault_contact.add_coulomb_fault_bc(
            stokes, params.uw_friction, boundary="Fault",
            sigma_n="reaction", V0=params.uw_slip_v0)
    else:
        stokes.add_fault_bc(ETA_WELD, boundary="Fault")
    result = fault_contact.solve_with_fault(stokes, picard=params.uw_picard)
    info = stokes._rotated_freeslip_info

    # sigma_n FIRST: a friction law's strength is set by it, so the shear
    # traction below cannot be read without it.
    positions, sigma_nodes = fault_contact.fault_normal_traction(stokes, "Fault", info)
    if uw.mpi.size > 1:
        # Unlike fault_pair_jumps this one has no gather of its own and is
        # documented as per-rank, so it is gathered here the same way the
        # library gathers the jumps — before any median is taken.
        parts = mesh.dm.comm.tompi4py().allgather(
            (np.asarray(positions, dtype=float), np.asarray(sigma_nodes, dtype=float)))
        stack = np.concatenate if DIM == 2 else (
            lambda arrays: np.vstack([a.reshape(-1, DIM) for a in arrays]))
        positions = stack([p[0] for p in parts])
        sigma_nodes = np.concatenate([p[1] for p in parts])
    if DIM == 2:
        # the 2-D helper returns the along-fault coordinate from one tip
        along = positions - 0.5 * (positions.min() + positions.max())
        central = np.abs(along) < CENTRAL_FRACTION * RADIUS
        positions = CENTRE + along[:, None] * in_plane_basis(normal)[0]
    else:
        central = central_mask(positions, normal)
    sigma_n = float(np.median(sigma_nodes[central]))

    # gather=True: the pairs are rank-LOCAL and split_fault leaves the fault
    # rank-interior (one owner), so without this every rank but one medians an
    # empty array and the answer is whichever rank happened to hold the fault.
    coords, jumps, normals = fault_contact.fault_pair_jumps(
        stokes, "Fault", info, gather=uw.mpi.size > 1)
    # The pairing's normal points Plus -> Minus and the jump is v+ - v-.
    # Aligning that normal with the drawn one makes ``slip`` the motion of
    # the side BEHIND n relative to the side n points into; the traction
    # sigma.n on the plane opposes it.
    side = np.sign(normals @ normal)[:, None]
    leak = np.einsum("ij,ij->i", jumps, normals)
    slip = side * (jumps - leak[:, None] * normals)
    speed = np.linalg.norm(slip, axis=1)
    inner = central_mask(coords, normal)
    slip_rate = float(np.median(speed[inner]))
    if params.uw_friction > 0.0:
        # Yield is decided by the LOADING against the strength — not by a
        # threshold on the slip rate. A slip-rate cut is arbitrary and it
        # misclassifies exactly the faults this figure is about, the ones
        # sitting on the envelope: measured, five of 120 ended up just outside
        # it, two of them in tension, where a cohesionless fault has no
        # strength at all and cannot be "stuck". The applied traction is the
        # imposed uniform state's, which is exact here; the strength is built
        # on the MEASURED sigma_n. The slip rate is then independent
        # corroboration rather than the classifier.
        traction = 2.0 * ETA * strain_rate @ normal
        applied = traction - (traction @ normal) * normal
        # positive in compression, as the law's own normal_stress is. Cohesion
        # holds strength into mild tension, down to sigma = -C/mu.
        strength = max(params.uw_cohesion + params.uw_friction * (-sigma_n), 0.0)
        sliding = bool(np.linalg.norm(applied) > strength)
        # Where strength has gone AND the normal stress is tensile, the fault
        # would open; the bilateral no-opening constraint holds it shut with a
        # tensile reaction instead, and fault_contact's own note says the
        # solution is then unphysical. Flagged rather than quietly plotted.
        glued = bool(strength <= 0.0 and sigma_n > 0.0)
        if sliding:
            # At yield the fault holds the friction limit, opposing the slip.
            #
            # Scaling this by the law's mobilised fraction (2/pi) arctan(V/V0)
            # at the measured slip rate was tried and REJECTED: the median
            # nodal slip rate is not the rate the law sees pointwise — it
            # over-reads a stuck fault by 82% and shifts by factors of 2-27
            # under refinement — so using it pulled every sliding point off
            # the envelope and regressed the 2-D figure. slip_nodes is
            # recorded below for anyone who wants to re-derive a traction
            # definition from a finished run.
            direction = np.median(slip[inner], axis=0)
            tau = -strength * direction / max(np.linalg.norm(direction), 1e-30)
        else:
            # stuck: it holds what the medium transmits, which bulk_traction
            # measures (welded control: 0.0303 against an exact 0.0304)
            tau = bulk_traction(mesh, v, normal)
        tau_nodes = np.broadcast_to(tau, slip.shape).copy()
    else:
        # welded: the interface dashpot's own law
        tau_nodes = -ETA_WELD * slip
        sliding = glued = False
        tau = np.median(tau_nodes[inner], axis=0)
    return dict(sigma_n=sigma_n, tau=tau, leak=float(np.abs(leak).max()),
                # how fast the fault is actually moving: the welded probe
                # creeps, a failed fault slides
                slip_rate=slip_rate, sliding=bool(sliding), glued=bool(glued),
                friction=float(params.uw_friction),
                cohesion=float(params.uw_cohesion),
                converged=bool(result.get("converged", True)),
                # which preconditioner the solve actually got: a silent fall back
                # from geometric multigrid to GAMG costs 2-10x and nothing else
                # shows it
                velocity_pc=str(info.get("velocity_pc")),
                vel_its=np.asarray(info.get("vel_its_last") or [], dtype=int),
                pres_its=np.asarray(info.get("pres_its_last") or [], dtype=int),
                # the per-node slip vectors, so a traction definition can be
                # re-derived from a finished run instead of re-solving it
                pair_coords=coords, slip_nodes=slip, tau_nodes=tau_nodes,
                sigma_coords=positions, sigma_nodes=sigma_nodes), (v, p)


axes = principal_axes()
strain_rate = axes @ np.diag(PRINCIPAL_RATES) @ axes.T
stress = 2.0 * ETA * strain_rate
normals = (trajectory_normals(params.uw_n_faults) if params.uw_trajectory
           else random_normals(params.uw_n_faults))
run_dir = os.path.join(params.uw_output_dir, f"tumble_{DIM}d")
os.makedirs(run_dir, exist_ok=True)

base = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0,) * DIM, maxCoords=(1.0,) * DIM,
    cellSize=RESOLUTION["base_size"], regular=False, qdegree=2, refinement=1)

mine = [k for k in range(params.uw_n_faults)
        if k % params.uw_n_shards == params.uw_shard]
for k in mine:
    sample_dir = os.path.join(run_dir, f"sample_{k:04d}")
    record_file = os.path.join(sample_dir, "probe.npz")
    if os.path.exists(record_file):
        continue
    os.makedirs(sample_dir, exist_ok=True)
    normal = normals[k]

    mesh, timings = faulted_mesh(base, normal)
    t0 = time.time()
    probe, fields = fault_probe(mesh, strain_rate, normal, k)
    timings["t_solve"] = time.time() - t0

    traction = stress @ normal
    sigma_exact = float(traction @ normal)
    tau_exact = traction - sigma_exact * normal
    if not probe["converged"]:
        # A record of a solve that did not converge is worse than no record:
        # it reads sigma_n = tau = 0 and plots as a point at the origin, which
        # no eye would flag (measured: two such records in the tight-band
        # sweep). Leave the sample unwritten, so a resumed run retries it.
        uw.pprint(f"[{DIM}d {k:4d}] DID NOT CONVERGE — no record written",
                  flush=True)
        continue

    mesh.write_timestep("fault", 0, outputPath=sample_dir, meshVars=list(fields))
    # getHeightStratum is this rank's share, so the count is reduced; and one
    # rank writes the record, or every rank writes the same file at once.
    n_cells = int(uw.mpi.comm.allreduce(mesh.dm.getHeightStratum(0)[1]))
    if uw.mpi.rank == 0:
        np.savez(record_file, sample=k, dim=DIM, normal=normal, stress=stress,
                 principal_axes=axes, principal_rates=PRINCIPAL_RATES,
                 centre=CENTRE, radius=RADIUS, eta_weld=ETA_WELD,
                 sigma_exact=sigma_exact, tau_exact=tau_exact,
                 # the resolution this record was made at: the renderer masks
                 # and scales in units of the fault-band cell size, so it has
                 # to read that from the run rather than assume a default
                 resolution_h_near=RESOLUTION["h_near"],
                 resolution_base=RESOLUTION["base_size"],
                 resolution_levels=RESOLUTION["levels"],
                 resolution_ramp=RAMP,
                 n_cells=n_cells, **timings, **probe)
    uw.mpi.comm.barrier()
    stages = "  ".join(f"{name[2:]} {seconds:.1f}s" for name, seconds in timings.items())
    uw.pprint(f"[{DIM}d {k:4d}] sigma_n {probe['sigma_n']:+.4f} "
              f"(exact {sigma_exact:+.4f})  |tau| {np.linalg.norm(probe['tau']):.4f} "
              f"(exact {np.linalg.norm(tau_exact):.4f})  leak {probe['leak']:.1e}  "
              f"cells {mesh.dm.getHeightStratum(0)[1]}  {probe['velocity_pc']}  {stages}",
              flush=True)
