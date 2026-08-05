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
