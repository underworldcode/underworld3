"""Fixtures the parallel fault tests share.

The 2-D adapt-on-top fault (a refined unit box, an add-on of ``k`` levels toward
a straight trace, the trace cut and split) with its simple-shear solve, and the
3-D disc sheet on a bisection child. Charter section 5: one copy, imported by
every test that needs it.
"""
import numpy as np

import underworld3 as uw

TIP_A, TIP_B = np.array([0.30, 0.45]), np.array([0.70, 0.55])
TRACE = np.vstack([TIP_A, TIP_B])
H_BASE = 0.05


def base_2d():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1, refinement=1,
        qdegree=2)


def adapted_2d(k, base=None):
    """The add-on mesh of ``k`` levels toward the trace."""
    base = base_2d() if base is None else base
    pts = np.column_stack([np.linspace(TIP_A[0], TIP_B[0], 9),
                           np.linspace(TIP_A[1], TIP_B[1], 9), np.zeros(9)])
    guide = uw.meshing.Surface(f"guide_{k}", base, pts)
    guide.discretize()
    return base.adapt(guide.refinement_metric_function(
        h_near=H_BASE / 2**k, h_far=H_BASE, width=0.08, profile="linear"),
        max_levels=k)


def faulted_2d(k, split_levels=True):
    return adapted_2d(k).add_fault([("Flt", TRACE)], split_levels=split_levels)


def shear_fault_solve(mesh, law=0):
    """Simple shear on the walls, a fault condition ``law`` on ``Flt``; returns
    ``(stokes, info, peak slip)`` with the slip reduced over the ranks."""
    x, y = mesh.X
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
    stokes.tolerance = 1.0e-6
    stokes.petsc_use_pressure_nullspace = True
    for side in ("Top", "Bottom", "Left", "Right"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0), side)
    stokes.add_fault_bc(law, "Flt")
    stokes.solve()
    info = stokes._rotated_freeslip_info
    from underworld3.utilities.fault_contact import fault_slip
    _s, V, _leak = fault_slip(stokes, "Flt", info)
    local = float(np.abs(V).max()) if len(V) else 0.0
    return stokes, info, float(uw.mpi.comm.allreduce(local, op=max))


# --- the 3-D disc sheet on a bisection child (tests/test_0867 carries the serial
# copy; the two directories are separate import roots) ---------------------- #
CENTRE = np.array([0.5, 0.5, 0.5])
RADIUS = 0.2
H_NEAR = 0.05
TILTED = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])


def disc_sheet(normal=TILTED, n_rim=32, centre=CENTRE, radius=RADIUS):
    """A planar disc as a centre fan."""
    helper = np.eye(3)[np.argmin(np.abs(normal))]
    e1 = np.cross(normal, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    a = np.linspace(0.0, 2.0 * np.pi, n_rim, endpoint=False)
    rim = centre + radius * (np.outer(np.cos(a), e1) + np.outer(np.sin(a), e2))
    pts = np.vstack([centre, rim])
    tris = np.array([(0, 1 + i, 1 + (i + 1) % n_rim) for i in range(n_rim)],
                    dtype=np.int64)
    return pts, tris


def graded_disc_metric(normal=TILTED, discs=((CENTRE, RADIUS),)):
    """``M = 1/h^2`` from the exact distance to the nearest disc,
    ``h_near -> 2 h_near``."""
    def metric(X):
        d = np.full(len(X), np.inf)
        for centre, radius in discs:
            rel = np.asarray(X)[:, :3] - centre
            zn = rel @ normal
            r = np.linalg.norm(rel - zn[:, None] * normal, axis=1)
            d = np.minimum(d, np.sqrt(np.maximum(r - radius, 0.0) ** 2 + zn ** 2))
        h = np.clip(H_NEAR + d * H_NEAR / 0.4, H_NEAR, 2.0 * H_NEAR)
        return 1.0 / h**2
    return metric


def base_3d():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.2, regular=False, qdegree=2, refinement=1)
