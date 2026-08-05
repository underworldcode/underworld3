"""Shared harness for the split-node fault teaching examples.

One base mesh, re-faulted per case (the non-cumulative add_fault
pattern); P2 velocity / P0 discontinuous pressure per the fault
pressure-space ruling; slip and traction read through the DOF pairing.
Run inside the fault-split-node worktree env with its bin on PATH.
"""
import numpy as np

import underworld3 as uw
from underworld3.utilities import fault_contact

ETA = 1.0
CENTRE = np.array([0.5, 0.5])


def base_mesh(cell_size=0.04):
    return uw.meshing.UnstructuredSimplexBox(cellSize=cell_size)


def fault_segment(angle_deg, half_length=0.2, centre=CENTRE):
    t = np.array([np.cos(np.radians(angle_deg)),
                  np.sin(np.radians(angle_deg))])
    return np.array([centre - half_length * t, centre + half_length * t])


def split_with_fault(mesh, points, name="Fault"):
    return mesh.add_fault((name, points))


def stokes_on(child, drive, name="Fault"):
    """Stokes with the wall drive on all four walls; fault BC added by
    the caller. Variable names are derived from the mesh instance so
    repeated calls on fresh children never collide."""
    tag = f"f{stokes_on.counter}"
    stokes_on.counter += 1
    v = uw.discretisation.MeshVariable(f"V_{tag}", child, child.dim,
                                       degree=2)
    p = uw.discretisation.MeshVariable(f"P_{tag}", child, 1, degree=0,
                                       continuous=False)
    stokes = uw.systems.Stokes(child, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = ETA
    stokes.bodyforce = [0.0, 0.0]
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_dirichlet_bc(drive, wall)
    # all-Dirichlet walls leave the pressure level to the solver unless
    # the constant nullspace is declared — without it, two solves with
    # different fault laws land on DIFFERENT gauges and any differenced
    # quantity (Delta CFF above all) inherits a spurious constant
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1e-6
    return stokes


stokes_on.counter = 0


def simple_shear(child, rate=1.0):
    """v = (rate (y - 1/2), 0): sigma_xy = ETA * rate everywhere."""
    x, y = child.X
    return (rate * (y - 0.5), 0.0)


def shear_plus_stretch(child, a=0.5, gamma=1.0):
    """v = (a(x-c) + gamma(y-c), -a(y-c)): deviatoric stress
    [[2 eta a, eta gamma], [eta gamma, -2 eta a]], Mohr radius
    eta sqrt(4 a^2 + gamma^2)."""
    x, y = child.X
    return (a * (x - 0.5) + gamma * (y - 0.5), -a * (y - 0.5))


def slip_profile(stokes, name="Fault"):
    return fault_contact.fault_slip(stokes, name,
                                    stokes._rotated_freeslip_info)


def normal_traction(stokes, name="Fault"):
    return fault_contact.fault_normal_traction(
        stokes, name, stokes._rotated_freeslip_info)


def inner(s, fraction=0.6):
    """Mask selecting the central `fraction` of the along-fault range —
    tip fields carry the crack singularity, the middle is the gauge."""
    lo = s.min() + (1 - fraction) / 2 * (s.max() - s.min())
    hi = s.max() - (1 - fraction) / 2 * (s.max() - s.min())
    return (s >= lo) & (s <= hi)


def slip_vs_position(stokes, tangent, centre=CENTRE, name="Fault"):
    """(s, V) read through the pairing with EXACT positions: s is the
    signed along-fault coordinate of each pair about the fault centre,
    V the signed tangential jump. `fault_slip`'s arc-length origin sits
    at the first PAIR, not the tip, so profiles built from it are offset
    by half a node — this is the plotting-grade version."""
    coords, jumps, _normals = fault_contact.fault_pair_jumps(
        stokes, name, stokes._rotated_freeslip_info)
    t = np.asarray(tangent, dtype=float)
    t = t / np.linalg.norm(t)
    s = (coords - centre) @ t
    V = jumps @ t
    order = np.argsort(s)
    return s[order], V[order]


def pure_shear_drive(child, phi_deg, tau0=1.0):
    """Uniform pure shear with the COMPRESSION axis at ``phi_deg`` to x:
    sigma' = tau0 (e_perp e_perp - e_phi e_phi), i.e. sigma'_xx =
    -tau0 cos 2phi and sigma'_xy = -tau0 sin 2phi. (Check: compression
    at phi = 45 deg gives sigma_xy = -tau0; a plane of strike
    phi - 45 deg carries the full resolved shear.) Irrotational
    velocity field, imposed as Dirichlet on all walls — rotating phi
    rotates the whole regional stress field."""
    x, y = child.X
    two_phi = 2.0 * np.radians(phi_deg)
    exx = -tau0 * np.cos(two_phi) / (2.0 * ETA)
    exy = -tau0 * np.sin(two_phi) / (2.0 * ETA)
    return (exx * (x - 0.5) + exy * (y - 0.5),
            exy * (x - 0.5) - exx * (y - 0.5))


def boundary_simple_shear(child, trend_deg, rate=1.0):
    """RIGHT-LATERAL simple shear parallel to a plate-boundary trend:
    v = rate ((X - c) . n) t with t the boundary-parallel direction and
    n its CCW normal — material on the +n side moves along +t, which an
    observer on the fault sees moving to the RIGHT (dextral). Resolved
    shear tau0 = ETA * rate on boundary-parallel planes."""
    x, y = child.X
    t = np.array([np.cos(np.radians(trend_deg)),
                  np.sin(np.radians(trend_deg))])
    n = np.array([-t[1], t[0]])
    s = (x - 0.5) * n[0] + (y - 0.5) * n[1]
    return (rate * s * t[0], rate * s * t[1])


def ambient_sigma_n_simple(trend_deg, tangent, tau0=1.0):
    """Analytic ambient normal stress (tension-positive) on a plane of
    direction ``tangent`` under boundary_simple_shear(trend):
    sigma_nn = -tau0 sin 2(theta - trend)."""
    theta = np.degrees(np.arctan2(tangent[1], tangent[0]))
    return -tau0 * np.sin(2.0 * np.radians(theta - trend_deg))


def probe_nodes(stokes, name, tangent, eta_weld):
    """Per-node stress probes on a WELDED fault: along-fault coordinate,
    position, signed normal traction (tension-positive, as measured) and
    signed shear traction from the weld's own law tau = eta_f V.

    Works with SEVERAL law-carrying faults on one mesh: the interface
    assembler holds every fault's nodes, so this fault's are selected
    through its OWN pairing (the plus points), never by position."""
    from underworld3.utilities.rotated_bc import _point_coord

    assembler = fault_contact._InterfaceAssembler(stokes, include=(name,))
    sig_all = assembler.nodal_normal_traction(
        stokes, stokes._rotated_freeslip_info["reaction"])
    plus = set(stokes.mesh._fault_point_pairs[name].values())

    dm = stokes.dm
    dim = stokes.mesh.dim
    csec = dm.getCoordinateSection()
    cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dim)
    v0, v1 = dm.getDepthStratum(0)
    rows = [(np.asarray(_point_coord(dm, dim, cvec, csec, v0, v1, q)),
             sig_all[k]) for q, k in assembler._points.items()
            if q in plus]
    xy_sig = np.array([r[0] for r in rows])
    sig = np.array([r[1] for r in rows])

    t = np.asarray(tangent, dtype=float)
    t = t / np.linalg.norm(t)
    order_sig = np.argsort(xy_sig @ t)
    xy_sig, sig = xy_sig[order_sig], sig[order_sig]

    coords, jumps, _normals = fault_contact.fault_pair_jumps(
        stokes, name, stokes._rotated_freeslip_info)
    s = coords @ t
    order = np.argsort(s)
    V = (jumps @ t)[order]
    assert len(sig) == len(V), "pair-node sets disagree"
    assert np.allclose(coords[order], xy_sig), "node ordering disagrees"
    return (s[order], coords[order], sig, eta_weld * V)


def mohr_probe(theta, a_rate=0.5, gamma=1.0, eta_weld=None,
               half_length=0.2, cell_size=0.04):
    """One welded-fault stress probe: (sigma_n, tau_signed) at fault
    angle `theta` under the shear_plus_stretch drive. The weld's own
    law reads the shear traction (tau = eta_f V); the no-opening
    reaction reads the normal traction."""
    if eta_weld is None:
        eta_weld = 200.0 * ETA / half_length
    child = split_with_fault(base_mesh(cell_size),
                             fault_segment(theta, half_length))
    stokes = stokes_on(child, shear_plus_stretch(child, a_rate, gamma))
    stokes.add_fault_bc(eta_weld, boundary="Fault")
    fault_contact.solve_with_fault(stokes, picard=2)
    s, V, _leak = slip_profile(stokes)
    s_n, sig = normal_traction(stokes)
    tau = eta_weld * float(np.median(V[inner(s)]))
    sigma_n = float(np.median(sig[inner(s_n)]))
    return sigma_n, tau


def ambient_sigma_n(phi_deg, tangent, tau0=1.0):
    """The analytic ambient normal stress (tension-positive) on a plane
    of tangent direction ``tangent`` under pure_shear_drive(phi):
    sigma_nn = tau0 cos 2(phi - theta). Used to anchor a probe set's
    absolute pressure gauge, which a closed-box solve does not fix."""
    theta = np.degrees(np.arctan2(tangent[1], tangent[0]))
    return tau0 * np.cos(2.0 * np.radians(phi_deg - theta))


def far_field_anchor(points, dcff, segments, cut=0.3):
    """Gauge the DIFFERENCED stress to the physics: a slip event changes
    nothing far from the faults, so the far-field median of Delta CFF is
    the spurious pressure-gauge constant between the two solves. Returns
    (anchored dcff, the constant) — apply c / mu' to the probes' sigma."""
    p = np.asarray(points, dtype=float)[:, :2]
    far = np.ones(len(p), dtype=bool)
    for seg in segments:
        a, b = np.asarray(seg[0]), np.asarray(seg[-1])
        t = b - a
        L = np.linalg.norm(t)
        t = t / L
        s = np.clip((p - a) @ t, 0.0, L)
        far &= np.linalg.norm(p - (a + s[:, None] * t), axis=1) > cut
    c = float(np.median(np.asarray(dcff)[far]))
    return np.asarray(dcff) - c, c


def signed_log(x, linthresh=0.02):
    """Symmetric log transform for signed stress-change fields: linear
    inside |x| < linthresh, logarithmic beyond — the near-fault values
    saturate any linear colour scale and hide the far-field lobes."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log10(1.0 + np.abs(x) / linthresh)


def signed_log_annotations(values, linthresh=0.02):
    """Scalar-bar tick positions/labels in transformed units."""
    return {float(np.sign(v) * np.log10(1.0 + abs(v) / linthresh)):
            f"{v:+.2f}" if v else "0" for v in values}
