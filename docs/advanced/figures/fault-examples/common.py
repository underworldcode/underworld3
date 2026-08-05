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
