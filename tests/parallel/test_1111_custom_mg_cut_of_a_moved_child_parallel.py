"""A fault cut into a moved adapt child solves under the #1101 guard (#1111).

The split level under the new mesh is not nested in it, and with the child
moved before the cut, coarse unknowns are left with dependent prolongation
columns: in the rotated transfer, three columns on one fine node's two
component rows. #1104's guard handled only columns on one shared row and
refused these ("share fewer fine images than there are of them"), so a solve
that had converged in 6 velocity iterations before the guard stopped. They are
now reduced exactly to as many columns as they span. W1's fixture (test_1176's
``_split_level_stokes``): the child moved by ``amp sin(pi x) sin(pi y)`` along
x or y (across the fault) before ``add_fault``. Every case must solve; the
0.002 case across the fault, which the guard never touched, stays at 6.

This file carries no ``mpi(min_size=2)`` mark: it runs serially and in the
parallel batches.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(300)]

TRACE = np.array([[0.3, 0.45], [0.7, 0.55]])

# (axis, amplitude): the amplitudes W1 measured; 0.005 and up were refused
CASES = [(1, 0.002), (0, 0.005), (1, 0.005), (0, 0.01), (1, 0.01), (0, 0.015), (1, 0.015)]


def _cut_of_a_moved_child(axis, amplitude):
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1, refinement=1, qdegree=2)
    pts = np.column_stack([np.linspace(*TRACE[:, 0], 9), np.linspace(*TRACE[:, 1], 9),
                           np.zeros(9)])
    guide = uw.meshing.Surface(f"guide_1111_{axis}_{amplitude}", base, pts)
    guide.discretize()
    child = base.adapt(guide.refinement_metric_function(
        h_near=0.05 / 4, h_far=0.05, width=0.08, profile="linear"), max_levels=2)
    c = np.asarray(child.X.coords).copy()
    c[:, axis] += amplitude * np.sin(np.pi * c[:, 0]) * np.sin(np.pi * c[:, 1])
    child.deform(c)
    mesh = child.add_fault([("Flt", TRACE)])

    x, y = mesh.X
    s = uw.systems.Stokes(mesh)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1
    s.tolerance = 1.0e-6
    s.petsc_use_pressure_nullspace = True
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_dirichlet_bc((y - 0.5, 0.0), wall)
    s.add_fault_bc(0, "Flt")
    return s


@pytest.mark.parametrize("axis,amplitude", CASES)
def test_a_cut_of_a_moved_child_solves(axis, amplitude):
    s = _cut_of_a_moved_child(axis, amplitude)
    s.solve()
    assert s.snes.getConvergedReason() > 0
    its = max(s._rotated_freeslip_info["vel_its_last"])
    if (axis, amplitude) == (1, 0.002):
        assert its == 6, its
